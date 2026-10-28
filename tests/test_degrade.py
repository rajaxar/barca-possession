"""Degradation simulator: deterministic, ATD-shaped output, and no label leak through the ball (S06).

The external part is the leak-removal assertion of card S06 (metrics of scratch/S05/leaks.py): on
every degraded IDSSE/Metrica game the ball's frozen share and NaN share are the same in D and alive
frames, the IDSSE teleport at D end is gone, and single ball features stay far from separating D.
"""
import os
import unittest

import numpy as np

from possession import config, data, degrade, external

# The degrader is fitted to 25 fps sources (S06/S17). SkillCorner games (10 fps, E25's loader)
# enter list_games() but their degradation is not fitted, so the leak/kinematic assertions are
# scoped to the fitted sources.
GAMES = [g for g in external.list_games() if g.startswith(("idsse", "metrica"))]


def synthetic_match(n=3000):
    rng = np.random.default_rng(1)
    half = np.repeat(np.array([1, 2], np.int8), n // 2)
    T = np.repeat(np.array([0, 1], np.int8), 12)
    P = (rng.uniform([0, 0], [105, 68], (1, 24, 2)) + np.cumsum(rng.normal(0, 0.05, (n, 24, 2)), 0))
    B = (np.array([52.5, 34.0]) + np.cumsum(rng.normal(0, 0.2, (n, 2)), 0)).astype(np.float32)
    dead = np.zeros(n, bool)
    dead[1000:1400] = True
    B[1000:1400] = B[999]  # IDSSE-style frozen dead ball, then a teleport on the restart frame
    B[1400:] += 20.0
    gk = np.array([[0, 12], [0, 12]])
    m = data.Match(P=P.astype(np.float32), V=np.ones((n, 24), np.float32), T=T, B=B,
                   BS=np.full(n, np.nan, np.float32), gk=gk, graded=np.ones(n, bool), half=half)
    return m, dead


class Synthetic(unittest.TestCase):
    def test_deterministic_and_layout(self):
        m, dead = synthetic_match()
        a = degrade.degrade(m, seed=3, dead=dead)
        b = degrade.degrade(m, seed=3, dead=dead)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        k = degrade.PARAMS["slots_per_team"]
        self.assertEqual(a.P.shape, (len(m.B), 2 * k, 2))
        self.assertEqual(list(a.T), [0] * k + [1] * k)
        np.testing.assert_array_equal(a.half, m.half)
        self.assertFalse(np.array_equal(degrade.degrade(m, seed=4, dead=dead).B, a.B, equal_nan=True))

    def test_scrub_removes_freeze_and_teleport(self):
        m, dead = synthetic_match()
        B = degrade.scrub_ball(m.B, m.P, m.half, dead, np.random.default_rng(0))
        step = np.linalg.norm(np.diff(B, axis=0), axis=1)
        self.assertFalse(np.isnan(B).any())
        self.assertTrue((step[1000:1400] > 0).all())        # never frozen inside D
        self.assertLess(step[999:1400].max() * 25, 40.0)    # no teleport into or out of D
        np.testing.assert_array_equal(B[:1000], m.B[:1000])  # alive frames untouched

    def test_level_zero_keeps_ball_present(self):
        m, dead = synthetic_match()
        out = degrade.degrade(m, level=0.0, dead=dead)
        self.assertFalse(np.isnan(out.B).any())
        self.assertTrue(np.isnan(out.BS).all())

    def test_lowpass_never_crosses_gaps_or_halves(self):
        n = 400
        half = np.repeat(np.array([1, 2], np.int8), n // 2)
        X = (np.sin(np.arange(n) / 4.0) + 0.3 * np.arange(n))[:, None] * np.ones((1, 2))
        X[60:90] = np.nan    # gap inside half 1
        X[230:260] = np.nan  # gap inside half 2
        Y = degrade.lowpass(X, half, 8.0)
        self.assertTrue(np.isnan(Y[60:90]).all())
        self.assertTrue(np.isnan(Y[230:260]).all())
        # every present run equals an independent smoothing of that run alone:
        # nothing bleeds across a NaN gap or a half boundary (S17: smoothing must not mark D)
        for s, e in ((0, 60), (90, 200), (200, 230), (260, 400)):
            ref = degrade.lowpass(X[s:e], np.ones(e - s), 8.0)
            np.testing.assert_allclose(Y[s:e], ref, atol=1e-9, err_msg="run %d:%d" % (s, e))
        # and the output must actually be smoothed (not identity)
        self.assertFalse(np.allclose(Y[0:59], X[0:59]))


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class FittedTargets(unittest.TestCase):
    def test_constants_reproduce_from_our_atd(self):
        t = degrade.fit_targets(data.load_match())
        np.testing.assert_allclose(degrade.ATD_GAP_Q, t["gap_q"], atol=0.5)
        np.testing.assert_allclose(degrade.ATD_PRESENT_Q, t["present_q"], atol=0.5)
        np.testing.assert_allclose(degrade.ATD_SEG_Q, t["seg_q"], atol=0.5)
        np.testing.assert_allclose(degrade.PARAMS["bs_run_q"], t["bs_run_q"], atol=0.5)
        self.assertAlmostEqual(degrade.PARAMS["bs_rate"], t["bs_rate"], delta=1e-6)


@unittest.skipIf(len(GAMES) < 3, "external dev games not on disk")
class ExternalLeaks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        cls.kin = {}
        for game_id in GAMES:
            clean = external.load_game(game_id)
            game = degrade.degrade_game(clean, seed=0)
            cls.results[game_id] = degrade.leak_stats(game.match, game.ref, game.runs, n_null=0)
            # (degraded all-graded, clean alive) dv stats; the clean alive ball still separates
            alive = clean.match.graded & (np.asarray(clean.ref) != "D") & (clean.match.half > 0)
            cls.kin[game_id] = (_dv_stats(game.match), _dv_stats(clean.match, alive))

    def test_frozen_and_nan_share_equal_in_dead_and_alive(self):
        diffs = []
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                frozen = r["ball_frozen"]
                self.assertLess(abs(frozen["d_mean"] - frozen["alive_mean"]), 0.01)
                nan = r["ball_nan"]
                diffs.append(nan["d_mean"] - nan["alive_mean"])
                # one game: ~10 s blocks make the per-game spread ~0.05 (sd over seeds and games)
                self.assertLess(abs(diffs[-1]), 0.12)
        self.assertLess(abs(np.mean(diffs)), 0.03)  # no systematic P(NaN | D) shift

    def test_no_teleport_at_dead_end(self):
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                self.assertLess(r["teleport"]["d_end_fast"], 0.05)  # clean IDSSE: 0.87-0.93

    def test_single_ball_features_do_not_separate_dead(self):
        # AUC of D vs alive. Clean data: ball speed 0.00-0.6 (IDSSE ~0.003), nearest player up to
        # 0.81, ball NaN up to 0.98 (Metrica). Players alone (mean speed) reach 0.06-0.17.
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                for name in ("ball_nan", "ball_frozen", "bs_present"):
                    self.assertLess(abs(r[name]["auc"] - 0.5), 0.07, name)
                for name in ("ball_speed", "nearest_player_m"):
                    self.assertLess(abs(r[name]["auc"] - 0.5), 0.22, name)
                    self.assertLess(abs(r[name]["auc"] - 0.5),
                                    abs(r["player_speed_mean"]["auc"] - 0.5), name)

    def test_smoothing_reproduces_atd_kinematics(self):
        # S17: the vendor low-pass must make the degraded ball kinematics-blind like the ATD
        # (E18: separation ~0.95 ATD, ~1.65 IDSSE sigma=8, 3.5-4.7 clean alive IDSSE).
        degraded = np.array([self.kin[g][0] for g in self.kin])
        clean = np.array([self.kin[g][1] for g in self.kin])
        med, near, free, sep, jit = (np.nanmedian(degraded, 0) if degraded.shape[0] else
                                     [np.nan] * 5)
        self.assertGreater(med, 0.35)
        self.assertLess(med, 0.9)             # ATD 0.59 m/s
        self.assertGreater(near, 0.7)
        self.assertLess(near, 2.6)            # ATD 1.96 m/s
        self.assertLess(sep, 1.7)             # ATD 0.95; clean alive IDSSE 3.5-4.7
        self.assertLess(jit, 10.0)            # ATD 1-5 mm; clean IDSSE ~10-50 mm
        csep = np.nanmedian(clean[:, 3]) if len(clean) else np.nan
        self.assertGreater(csep, 2.5)         # the contrast: clean ball still separates


def _dv_stats(match, keep=None):
    """(dv median, q90 dv within 1.5 m of a player, q90 dv in free flight, separation, jitter mm).
    Mirrors E18 a7 / scratch/S17/kin.dv_stats with w=3; jitter from MAD of second differences."""
    keep = np.ones(len(match.B), bool) if keep is None else keep
    n = len(match.B)
    dv = np.full(n, np.nan)
    near = np.full(n, np.inf)
    B = np.where(keep[:, None], match.B, np.nan)
    d2s = []
    for h in (1, 2):
        idx = np.flatnonzero(match.half == h)
        if not len(idx):
            continue
        a0 = idx[0]
        c = B[idx].astype(float)
        step = np.r_[np.zeros((1, 2)), np.diff(c, axis=0)]
        c[np.linalg.norm(step, axis=1) > 40 / degrade.FPS] = np.nan  # drop implausible jumps
        for a, b in degrade._runs(~np.isnan(c[:, 0])):
            if b - a < max(9, 7):
                continue
            seg = c[a:b]
            vin = (seg[3:-3] - seg[:-6]) * degrade.FPS / 3
            vout = (seg[6:] - seg[3:-3]) * degrade.FPS / 3
            rows = a0 + a + np.arange(3, b - a - 3)
            dv[rows] = np.linalg.norm(vout - vin, axis=1)
            d = np.linalg.norm(match.P[rows] - match.B[rows][:, None, :], axis=-1)
            d = np.where(np.isnan(d), np.inf, d)
            near[rows] = d.min(1)
            if b - a < 50:
                continue
            sp = np.linalg.norm(seg[2:] - seg[:-2], axis=1) * degrade.FPS / 2
            k = sp < 3
            if k.sum() < 10:
                continue
            dd = seg[2:] - 2 * seg[1:-1] + seg[:-2]
            med = np.median(dd[k], axis=0)
            d2s.append(1.4826 * np.median(np.abs(dd[k] - med), axis=0).mean() / np.sqrt(6) * 1000)
    free = np.isfinite(dv) & (near > 5)
    nr = np.isfinite(dv) & (near <= 1.5)
    if not free.any() or not nr.any():
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    return (float(np.median(dv[free])), float(np.quantile(dv[nr], 0.9)),
            float(np.quantile(dv[free], 0.9)),
            float(np.quantile(dv[nr], 0.9) / np.quantile(dv[free], 0.9)),
            float(np.median(d2s)) if d2s else np.nan)


if __name__ == "__main__":
    unittest.main()
