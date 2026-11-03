"""Tests for possession.dproxy (S22): the label-free D proxy and the adoption protocol."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from possession import dproxy
from possession.data import Match
from possession.structure import PITCH_L, PITCH_W, Structure

N = 1200
FPS = 25.0


# ------------------------------------------------------------------ synthetic match


def _make(speed_fn=None, missing_fn=None, N=N, h1=(100, 800), h2=(900, 1200), forced=(900, 950),
          keeper_hold=None):
    """Synthetic Match: 23 players a side walking at ``speed_fn(f)``; ball missing where asked.

    ``missing_fn(f) -> bool``. ``keeper_hold(f) -> xy or None`` overrides the ball entirely (used
    for the E38 gathered-hold test). Frames outside ``h1``/``h2`` are ungraded; ``forced`` frames
    are the published pre-kickoff block and are D by structure, not by the proxy.
    """
    P = np.full((N, 46, 2), np.nan, np.float32)
    V = np.full((N, 46), np.nan, np.float32)
    T = np.array([0] * 23 + [1] * 23, np.int8)
    half = np.zeros(N, np.int8)
    half[h1[0]:h1[1]] = 1
    half[h2[0]:h2[1]] = 2
    for f in range(N):
        s = float(speed_fn(f)) if speed_fn else 1.0
        for i in range(22):
            P[f, i] = (20 + 3 * i, 8 + 2 * i)
            V[f, i] = s
            P[f, 23 + i] = (85 - 3 * i, 8 + 2 * i)
            V[f, 23 + i] = s
        P[f, 22] = (3.0, PITCH_W / 2)      # home keeper
        V[f, 22] = 1.0
        P[f, 45] = (102.0, PITCH_W / 2)    # away keeper
        V[f, 45] = 1.0
    B = np.full((N, 2), np.nan, np.float32)
    for f in range(N):
        if keeper_hold is not None:
            xy = keeper_hold(f)
        else:
            xy = None if (missing_fn and missing_fn(f)) else (52.0, PITCH_W / 2)
        if xy is not None:
            B[f] = xy
    forced_mask = np.zeros(N, bool)
    forced_mask[forced[0]:forced[1]] = True
    graded = half > 0
    return Match(P, V, T, B, np.zeros(N, np.float32),
                 np.array([[22, 45], [22, 45]]), graded, half), forced_mask


def _st(forced_mask, N=N):
    return Structure(halves={1: (100, 800), 2: (900, 1200)}, kickoffs={1: 100, 2: 950},
                     attack_dir=np.array([[1, -1], [-1, 1]], np.int8),
                     forced_dead_mask=forced_mask)


def _gap(a, b):
    return lambda f: a <= f < b


# ------------------------------------------------------------------ proxy tests


class ProxyShape(unittest.TestCase):
    def test_deterministic(self):
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(300, 400))
        st = _st(fm)
        a = dproxy.proxy(m, st)
        b = dproxy.proxy(m, st)
        self.assertTrue(np.array_equal(a.mask, b.mask))
        self.assertEqual(len(a.runs), len(b.runs))
        self.assertEqual([r["start"] for r in a.runs], [r["start"] for r in b.runs])
        self.assertEqual(a.info["n_proxy_frames"], b.info["n_proxy_frames"])
        self.assertEqual(a.speed.tolist(), b.speed.tolist())

    def test_duration_floor(self):
        # 1.8 s of missing+still is below the 2 s floor; 2.2 s is above it.
        m, fm = _make(speed_fn=lambda f: 1.0,
                      missing_fn=lambda f: (300 <= f < 345) or (600 <= f < 655))
        dp = dproxy.proxy(m, _st(fm))
        self.assertFalse(dp.mask[300:345].any())
        self.assertTrue(dp.mask[600:655].all())
        self.assertEqual([(r["start"], r["end"]) for r in dp.runs], [(600, 655)])
        # the 1.8 s run is still listed as a missing run, just not qualifying
        short = [mr for mr in dp.missing_runs if mr["start"] == 300][0]
        self.assertFalse(short["qualifies"])
        self.assertEqual(short["proxy_frames"], 0)

    def test_stillness_gate(self):
        missing = _gap(300, 380)
        fast, fm = _make(speed_fn=lambda f: 2.4, missing_fn=missing)
        slow, _ = _make(speed_fn=lambda f: 1.0, missing_fn=missing)
        self.assertFalse(dproxy.proxy(fast, _st(fm)).mask[300:380].any())
        self.assertTrue(dproxy.proxy(slow, _st(fm)).mask[300:380].all())

    def test_speed_clip_guard(self):
        # one absurd spike in one slot must not drag the mean above the stillness cut
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(300, 380))
        m.V[320, 0] = 400.0
        dp = dproxy.proxy(m, _st(fm))
        self.assertTrue(dp.mask[300:380].all())
        self.assertLess(float(dp.speed[320]), 1.5)

    def test_half_boundary_splits_runs(self):
        # missing across the ungraded gap between halves must not become one run
        m, fm = _make(speed_fn=lambda f: 1.0,
                      missing_fn=lambda f: (750 <= f < 800) or (960 <= f < 1020))
        dp = dproxy.proxy(m, _st(fm))
        self.assertEqual(len(dp.runs), 2)
        self.assertTrue(all(r["end"] <= 800 or r["start"] >= 900 for r in dp.runs))

    def test_forced_block_never_proxy(self):
        # the 2H pre-kickoff block is D by published structure; missing+still there is not evidence
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(900, 960))
        dp = dproxy.proxy(m, _st(fm))
        self.assertFalse(dp.mask[900:960].any())
        self.assertEqual(dp.info["forced_frames_ignored"], 50)
        self.assertFalse(dp.mask[fm].any())

    def test_ungraded_never_proxy(self):
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(50, 99))
        dp = dproxy.proxy(m, _st(fm))
        self.assertFalse(dp.mask[:100].any())
        self.assertFalse(dp.mask[800:900].any())

    def test_run_metadata(self):
        m, fm = _make(speed_fn=lambda f: 0.8, missing_fn=_gap(300, 400))
        dp = dproxy.proxy(m, _st(fm))
        r = dp.runs[0]
        self.assertEqual((r["start"], r["end"], r["frames"]), (300, 400, 100))
        self.assertAlmostEqual(r["seconds"], 4.0)
        self.assertAlmostEqual(r["mean_speed"], (44 * 0.8 + 2 * 1.0) / 46, places=5)
        self.assertAlmostEqual(r["grade"], dproxy.GRADE_P[1])  # 3-6 s bin
        self.assertEqual(dp.missing_runs[r["missing_run"]]["start"], 300)
        self.assertAlmostEqual(dp.info["coverage_missing"], 1.0)

    def test_run_gate_run_level(self):
        # one gap with a fast first second then stillness: per-frame keeps only the still part,
        # whole-gap keeps the whole run
        m, fm = _make(speed_fn=lambda f: 3.0 if f < 325 else 1.0,
                      missing_fn=_gap(300, 400))
        st = _st(fm)
        frame = dproxy.proxy(m, st)
        whole = dproxy.proxy(m, st, dict(run_gate="run"))
        self.assertEqual(int(frame.mask.sum()), 75)
        self.assertEqual(int(whole.mask.sum()), 100)

    def test_duration_grade_monotone(self):
        self.assertLess(dproxy._grade(1.0), dproxy._grade(4.0))
        self.assertLess(dproxy._grade(4.0), dproxy._grade(8.0))
        self.assertLess(dproxy._grade(8.0), dproxy._grade(20.0))


class GoalkeeperExclusion(unittest.TestCase):
    """E38: a keeper gathering the ball is live, even though the ball is missing and players are."""

    def _hold(self, f):
        if f < 300:                       # ball resting at the keeper's feet (1.5 m away)
            return (4.5, PITCH_W / 2)
        if f < 375:                       # gathered: tracker loses it for 3 s
            return None
        return (22.0, PITCH_W / 2) if f < 377 else (52.0, PITCH_W / 2)

    def test_gathered_hold_is_excluded(self):
        m, fm = _make(speed_fn=lambda f: 1.0, keeper_hold=self._hold)
        st = _st(fm)
        on = dproxy.proxy(m, st, dict(exclude_gk=True))
        off = dproxy.proxy(m, st, dict(exclude_gk=False))
        self.assertFalse(on.mask[300:375].any())
        self.assertEqual(on.info["gk_source"], "gk_possession")
        self.assertGreater(on.info["gk_vetoed"], 0)
        self.assertTrue(off.mask[300:375].any())   # without E38 this is a false D

    def test_exclusion_can_be_disabled(self):
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(300, 400))
        st = _st(fm)
        self.assertEqual(dproxy.proxy(m, st, dict(exclude_gk=False)).info["gk_source"], "disabled")


class GkSourceContract(unittest.TestCase):
    """B21: name the real detector, and never fall back silently."""

    @staticmethod
    def _match():
        m, fm = _make(speed_fn=lambda f: 1.0, missing_fn=_gap(300, 400))
        return m, _st(fm)

    def test_real_detector_is_used_and_reported(self):
        from possession import gk_possession
        m, st = self._match()
        calls = []
        real = gk_possession.detect

        def spy(*args, **kwargs):
            calls.append(kwargs.get("params", {}))
            return real(*args, **kwargs)

        with mock.patch.object(gk_possession, "detect", side_effect=spy):
            dp = dproxy.proxy(m, st, dict(exclude_gk=True))
            pb = dproxy.present_ball(m, st, dict(exclude_gk=True))
        # Both proxies must report the imported module, not the inline copy / "unavailable".
        self.assertEqual(dp.info["gk_source"], "gk_possession")
        self.assertEqual(pb.info["gk_source"], "gk_possession")
        self.assertIsNone(dp.info["gk_note"])
        self.assertIsNone(pb.info["gk_note"])
        self.assertEqual(len(calls), 2)      # the real detect() ran once per proxy

    def test_missing_module_falls_back_loudly_and_records_it(self):
        m, st = self._match()
        err = io.StringIO()
        with mock.patch.object(dproxy, "_import_gk", side_effect=ImportError("no gk module")), \
                contextlib.redirect_stderr(err):
            dp = dproxy.proxy(m, st, dict(exclude_gk=True))
            pb = dproxy.present_ball(m, st, dict(exclude_gk=True))
        self.assertEqual(dp.info["gk_source"], "inline")
        self.assertEqual(pb.info["gk_source"], "unavailable")
        self.assertIn("no gk module", dp.info["gk_note"])
        self.assertIn("no gk module", pb.info["gk_note"])
        text = err.getvalue()
        self.assertIn("WARNING [dproxy]", text)
        self.assertIn("gk_source='inline'", text)
        self.assertIn("gk_source='unavailable'", text)

    def test_detector_bug_is_not_swallowed(self):
        from possession import gk_possession
        m, st = self._match()
        with mock.patch.object(gk_possession, "detect", side_effect=IndexError("array end")):
            with self.assertRaises(IndexError):
                dproxy.proxy(m, st, dict(exclude_gk=True))
            with self.assertRaises(IndexError):
                dproxy.present_ball(m, st, dict(exclude_gk=True))


# ------------------------------------------------------------------ adoption protocol tests


PROTO_N = 3000


def _dropout(f):
    """A 0.48 s live tracking dropout every 2.4 s: real missingness, too short to be proxy D."""
    return (f % 40) < 12


def _protocol_match():
    """N=3000 match: seven 5 s still gaps over a background of short (live) dropouts.

    The dropouts matter: they give the cue's *before* windows ball-missing frames (the proxy's
    domain) without ever qualifying as proxy D themselves (0.48 s << the 2 s floor), which is the
    real match's structure (41% missing, 99.9% of it in long runs, E36).
    """
    gaps = [(400 + 300 * k, 525 + 300 * k) for k in range(7)]

    def missing(f):
        return any(a <= f < b for a, b in gaps) or _dropout(f)

    def speed(f):
        return 2.5 if _dropout(f) else 1.0

    m, fm = _make(speed_fn=speed, missing_fn=missing, N=PROTO_N,
                  h1=(200, 1400), h2=(1600, 2900), forced=(1600, 1660))
    return m, gaps


PROTO_ST = _st(np.pad(np.r_[np.zeros(1600, bool), np.ones(60, bool),
                         np.zeros(PROTO_N - 1660, bool)], (0, max(0, PROTO_N - PROTO_N))))


def _proto_st():
    fm = np.zeros(PROTO_N, bool)
    fm[1600:1660] = True
    return Structure(halves={1: (200, 1400), 2: (1600, 2900)}, kickoffs={1: 200, 2: 1660},
                     attack_dir=np.array([[1, -1], [-1, 1]], np.int8), forced_dead_mask=fm)


class ProtocolRefusals(unittest.TestCase):
    def setUp(self):
        self.m, self.gaps = _protocol_match()
        self.st = _proto_st()
        self.dp = dproxy.proxy(self.m, self.st)

    def _signal(self, value=1.0):
        cue = np.zeros(PROTO_N, float)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = value
        return cue

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            dproxy.adoption_report(np.zeros(10, bool), self.m, self.dp)
        with self.assertRaises(ValueError):
            dproxy.adoption_report(np.zeros((PROTO_N, 1), bool), self.m, self.dp)

    def test_nan_share_raises(self):
        cue = self._signal(0.9)
        idx = np.flatnonzero(self.m.graded & ~self.st.forced_dead_mask)
        cue[idx[: int(0.06 * len(idx))]] = np.nan
        with self.assertRaises(ValueError):
            dproxy.adoption_report(cue, self.m, self.dp, active_thr=0.5)

    def test_nan_share_tolerated_below_limit(self):
        cue = self._signal(0.9)
        idx = np.flatnonzero(self.m.graded & ~self.st.forced_dead_mask)
        cue[idx[: int(0.02 * len(idx))]] = np.nan
        r = dproxy.adoption_report(cue, self.m, self.dp, active_thr=0.5)
        self.assertIn("cue_had_nans_scored_as_zero", r["warnings"])
        self.assertEqual(r["decision"], "adopt")

    def test_inf_raises(self):
        cue = self._signal(0.9)
        cue[410] = np.inf
        with self.assertRaises(ValueError):
            dproxy.adoption_report(cue, self.m, self.dp, active_thr=0.5)

    def test_continuous_without_threshold_raises(self):
        with self.assertRaises(ValueError):
            dproxy.adoption_report(np.zeros(PROTO_N, float), self.m, self.dp)

    def test_cue_outside_proxy_domain_raises(self):
        # fires only on frames with a present ball -> the proxy cannot score it
        cue = np.zeros(PROTO_N, bool)
        cue[200:210] = True
        with self.assertRaises(ValueError):
            dproxy.adoption_report(cue, self.m, self.dp)

    def test_needs_match(self):
        with self.assertRaises(ValueError):
            dproxy.adoption_report(np.zeros(PROTO_N, bool), None, self.dp)


class ProtocolDecisions(unittest.TestCase):
    def setUp(self):
        self.m, self.gaps = _protocol_match()
        self.st = _proto_st()
        self.dp = dproxy.proxy(self.m, self.st)

    def test_pure_signal_adopts(self):
        # a perfect onset cue: active in the second just before each gap opens
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = True
        r = dproxy.adoption_report(cue, self.m, self.dp)
        self.assertEqual(r["decision"], "adopt", dproxy.format_report(r))
        self.assertEqual(r["reason"], "lead_ci_excludes_zero")
        self.assertGreater(r["lead_gain"], 0.2)
        self.assertGreater(r["lead_ci"]["low"], 0)

    def test_pure_noise_parks(self):
        rng = np.random.default_rng(0)
        cue = np.zeros(PROTO_N, bool)
        cue[rng.choice(np.flatnonzero(self.m.graded), size=120, replace=False)] = True
        r = dproxy.adoption_report(cue, self.m, self.dp)
        self.assertEqual(r["decision"], "park", dproxy.format_report(r))
        self.assertIn(r["reason"], ("lead_ci_covers_zero", "cue_is_reactive"))

    def test_reactive_cue_parks(self):
        # fires mid-gap: the match is already dead before and after it, so it cannot lead
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a + 62:a + 75] = True
        r = dproxy.adoption_report(cue, self.m, self.dp)
        self.assertEqual(r["decision"], "park", dproxy.format_report(r))
        self.assertIn(r["reason"], ("cue_is_reactive", "effect_below_noise_floor",
                                    "lead_ci_covers_zero"))

    def test_report_is_deterministic(self):
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = True
        a = dproxy.adoption_report(cue, self.m, self.dp)
        b = dproxy.adoption_report(cue, self.m, self.dp)
        self.assertEqual(a["decision"], b["decision"])
        self.assertEqual(a["lead_gain"], b["lead_gain"])
        self.assertEqual(a["lead_ci"], b["lead_ci"])
        self.assertEqual(a["ablation_delta"], b["ablation_delta"])

    def test_threshold_is_recorded(self):
        cue = np.zeros(PROTO_N, float)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = 0.9
        with self.assertRaises(ValueError):
            dproxy.adoption_report(cue, self.m, self.dp)
        r = dproxy.adoption_report(cue, self.m, self.dp, active_thr=0.5)
        self.assertEqual(r["cue"]["active_thr"], 0.5)

    def test_format_report_lines(self):
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = True
        text = dproxy.format_report(dproxy.adoption_report(cue, self.m, self.dp))
        for token in ("inside", "lead_gain", "conditional lead", "ablation", "DECISION"):
            self.assertIn(token, text)


# ------------------------------------------------------------------ present-ball proxy tests (E42)


def _ball_path(xs):
    """ball_fn for ``_make`` (via keeper_hold): ``xs(f) -> x`` meters on the halfway y line."""
    return lambda f: (xs(f), PITCH_W / 2)


class PresentBallShape(unittest.TestCase):
    @staticmethod
    def _xs(f):
        # moving at 6 m/s until f=300, at rest on the spot 300-380 (3.2 s), moving again after
        if f < 300:
            return 95.0 - 0.25 * f
        if f < 380:
            return 20.0
        return 20.0 + 0.25 * (f - 380)

    def _pb(self, speed=1.0, **kw):
        m, fm = _make(speed_fn=lambda f: speed, keeper_hold=_ball_path(self._xs))
        return m, dproxy.present_ball(m, _st(fm), **kw)

    def test_static_ball_with_still_players_is_proxy_d(self):
        _, pb = self._pb()
        self.assertEqual(pb.info["kind"], "present")
        self.assertTrue(pb.mask[315:365].all())
        self.assertFalse(pb.mask[120:280].any())     # ball moving 6 m/s: live
        self.assertFalse(pb.mask[400:700].any())

    def test_busy_players_veto(self):
        _, pb = self._pb(speed=3.0)
        self.assertFalse(pb.mask.any())

    def test_hold_floor(self):
        # a 0.8 s pause is below the 1 s hold floor
        m, fm = _make(speed_fn=lambda f: 1.0,
                      keeper_hold=_ball_path(lambda f: 52.0 if 300 <= f < 320 else 20.0 + 0.4 * f))
        pb = dproxy.present_ball(m, _st(fm))
        self.assertFalse(pb.mask[300:320].any())

    def test_disjoint_from_missing_proxy(self):
        m, fm = _make(speed_fn=lambda f: 1.0, keeper_hold=lambda f: (
            None if 500 <= f < 600 else _ball_path(self._xs)(f)))
        st = _st(fm)
        dm, pb = dproxy.proxy(m, st), dproxy.present_ball(m, st)
        self.assertTrue(dm.mask[500:600].all())
        self.assertFalse((dm.mask & pb.mask).any())
        bo = dproxy.both(m, st)
        self.assertEqual(bo.info["kind"], "both")
        self.assertTrue(np.array_equal(bo.mask, dm.mask | pb.mask))

    def test_ungraded_and_forced_never_proxy(self):
        m, fm = _make(speed_fn=lambda f: 1.0, keeper_hold=_ball_path(lambda f: 52.0))
        pb = dproxy.present_ball(m, _st(fm))
        self.assertFalse(pb.mask[:100].any())
        self.assertFalse(pb.mask[800:900].any())
        self.assertFalse(pb.mask[fm].any())

    def test_deterministic(self):
        _, a = self._pb()
        _, b = self._pb()
        self.assertTrue(np.array_equal(a.mask, b.mask))

    def test_keeper_exclusion_is_wired(self):
        m, fm = _make(speed_fn=lambda f: 1.0, keeper_hold=_ball_path(lambda f: 4.5))
        st = _st(fm)
        on = dproxy.present_ball(m, st)
        off = dproxy.present_ball(m, st, dict(exclude_gk=False))
        self.assertEqual(on.info["gk_source"], "gk_possession")
        self.assertEqual(off.info["gk_source"], "disabled")
        self.assertGreaterEqual(int(off.mask.sum()), int(on.mask.sum()))

    def test_bad_params_raise(self):
        m, fm = _make(speed_fn=lambda f: 1.0)
        with self.assertRaises(ValueError):
            dproxy.present_ball(m, _st(fm), dict(near_player_m=0))


class PresentProtocol(unittest.TestCase):
    """adoption_report is proxy-kind aware: the domain follows the target."""

    def setUp(self):
        self.m, self.gaps = _protocol_match()
        self.st = _proto_st()

    def test_present_target_refuses_cue_outside_ball_present(self):
        pb = dproxy.present_ball(self.m, self.st)
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a + 10:a + 20] = True
        with self.assertRaises(ValueError):
            dproxy.adoption_report(cue, self.m, pb)

    def test_both_target_names_kind_and_decides(self):
        bo = dproxy.both(self.m, self.st)
        cue = np.zeros(PROTO_N, bool)
        for a, _ in self.gaps:
            cue[a - 50:a - 25] = True
        r = dproxy.adoption_report(cue, self.m, bo)
        self.assertEqual(r["decision"], "adopt", dproxy.format_report(r))

    def test_cli_requires_proxy(self):
        with self.assertRaises(SystemExit):
            dproxy.main([])
        with self.assertRaises(SystemExit):
            dproxy.main(["--proxy", "nonsense", "--cue", "x.npy"])


if __name__ == "__main__":
    unittest.main()
