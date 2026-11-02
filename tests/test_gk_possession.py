"""Tests for possession.gk_possession (E38): label-free keeper-in-possession detector."""
import unittest

import numpy as np

from possession import gk_possession as gk
from possession.data import Match
from possession.structure import PITCH_L, PITCH_W, Structure

N = 400
FPS = 25


def _make(ball_fn, gk_home_x=5.0, gk_away_x=100.0, v_gk=0.0, v_field=1.2,
          half_ranges=((50, 350, 1),)):
    """Synthetic Match: two keepers, 20 outfielders, ball from ``ball_fn(f) -> (x, y) or None``."""
    P = np.full((N, 46, 2), np.nan, np.float32)
    V = np.full((N, 46), np.nan, np.float32)
    T = np.array([0] * 23 + [1] * 23, np.int8)
    half = np.zeros(N, np.int8)
    for a, b, h in half_ranges:
        half[a:b] = h
    for f in range(N):
        P[f, 22] = (gk_home_x, PITCH_W / 2)
        V[f, 22] = v_gk
        P[f, 45] = (gk_away_x, PITCH_W / 2)
        V[f, 45] = v_gk
        for i in range(11):  # 10 outfield per team, spread out, walking
            P[f, i] = (20 + 4 * i, 5 + 3 * i)
            V[f, i] = v_field
            P[f, 23 + i] = (85 - 4 * i, 5 + 3 * i)
            V[f, 23 + i] = v_field
    B = np.full((N, 2), np.nan, np.float32)
    for f in range(N):
        xy = ball_fn(f)
        if xy is not None:
            B[f] = xy
    BS = np.zeros(N, np.float32)
    graded = half > 0
    return Match(P, V, T, B, BS, np.array([[22, 45], [22, 45]]), graded, half)


def _st():
    """Structure for the synthetic pitch: home defends low x in H1, away high x; no forced D."""
    return Structure(halves={1: (50, 350)}, kickoffs={1: 50},
                     attack_dir=np.array([[1, -1], [-1, 1]], np.int8),
                     forced_dead_mask=np.zeros(N, bool))


def _hold_ball(f, start=100, end=160, release_to=25.6):
    """Ball passes to the home keeper at 15 m/s, stationary hold, then a fast release."""
    if f < 60:
        return (30.0, PITCH_W / 2)
    if f < start:
        u = (f - 60) / (start - 60)
        return (30.0 + u * (5.6 - 30.0), PITCH_W / 2)
    if f <= end:
        return (5.6, PITCH_W / 2)
    if f < end + 30:
        u = (f - end) / 30.0
        return (5.6 + u * (release_to - 5.6), PITCH_W / 2)
    return (release_to, PITCH_W / 2)


class TestCarry(unittest.TestCase):
    def test_hold_is_detected_and_quiet_elsewhere(self):
        m = _make(_hold_ball)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live.shape, (N,))
        self.assertEqual(gp.live.dtype, np.float32)
        self.assertTrue(set(np.unique(gp.live)) <= {0.0, 1.0})
        self.assertEqual(gp.live[110], 1.0)
        self.assertEqual(gp.live[10], 0.0)
        self.assertEqual(gp.live[390], 0.0)
        self.assertGreaterEqual(len(gp.episodes), 1)
        ep = [e for e in gp.episodes if e["start"] <= 110 < e["end"]][0]
        self.assertEqual(ep["kind"], "carry")
        self.assertEqual(ep["team"], 0)

    def test_ball_far_from_keeper_never_fires(self):
        m = _make(lambda f: (54.0, PITCH_W / 2))
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live.sum(), 0.0)
        self.assertEqual(gp.episodes, [])

    def test_whistle_before_onset_suppresses(self):
        ev = np.zeros(1, dtype=[("start_frame", "i4")])
        ev["start_frame"] = 86
        m = _make(_hold_ball)
        gp = gk.detect(m, _st(), events=ev)
        self.assertEqual(gp.live.sum(), 0.0)

    def test_out_of_play_before_onset_suppresses(self):
        def ball(f):
            if f < 95:                       # ball rolls out of play 0.2 s before the hold
                return (PITCH_L + 1.0, PITCH_W / 2) if 90 <= f < 95 else (54.0, PITCH_W / 2)
            return _hold_ball(f)
        m = _make(ball)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live.sum(), 0.0)

    def test_implausible_gk_slot_is_ignored(self):
        # home slot parked in the opposition half with the ball at its feet: E30's failure mode
        m = _make(lambda f: (55.0, PITCH_W / 2) if f >= 100 else (54.0, PITCH_W / 2),
                  gk_home_x=54.0, gk_away_x=100.0)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live.sum(), 0.0)
        pos, spd, own, plaus = gk.keeper_tracks(m, _st())
        self.assertFalse(plaus[120, 0])
        self.assertGreater(gp.info["slot_rejected"], 0.0)

    def test_short_pass_through_is_not_a_hold(self):
        def ball(f):
            if 100 <= f <= 103:              # 0.16 s within 2 m: a pass rolling past
                return (5.0, PITCH_W / 2)
            return (54.0, PITCH_W / 2)
        m = _make(ball)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live.sum(), 0.0)

    def test_ungraded_frames_are_zero(self):
        m = _make(_hold_ball, half_ranges=((50, 350, 1),))
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.live[0], 0.0)
        self.assertEqual(gp.live[N - 1], 0.0)
        self.assertTrue(np.all(gp.d_delta <= 0.0))


class TestGathered(unittest.TestCase):
    def _ball(self, gap_end, ret_speed):
        def ball(f):
            if f < 100:
                return (5.6, PITCH_W / 2)
            if f < gap_end:
                return None
            x = 60.0 + ret_speed * (f - gap_end) / FPS
            return (x, PITCH_W / 2)
        return ball

    def test_short_gap_with_fast_return_is_live(self):
        m = _make(self._ball(100 + 75, 20.0))
        off = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(off.live[150], 0.0)     # off by default: the safe side
        gp = gk.detect(m, _st(), events=np.zeros(0), params=dict(include_gathered=True))
        self.assertEqual(gp.live[150], 1.0)
        self.assertTrue(any(e["kind"] == "gathered" for e in gp.episodes))

    def test_long_gap_is_not_live(self):
        m = _make(self._ball(100 + 200, 20.0))
        gp = gk.detect(m, _st(), events=np.zeros(0), params=dict(include_gathered=True))
        self.assertEqual(gp.live.sum(), 0.0)

    def test_slow_reappearance_is_not_live(self):
        m = _make(self._ball(100 + 75, 0.1))
        gp = gk.detect(m, _st(), events=np.zeros(0), params=dict(include_gathered=True))
        self.assertEqual(gp.live.sum(), 0.0)


class TestInterface(unittest.TestCase):
    def test_d_delta_is_bounded_and_aligned(self):
        m = _make(_hold_ball)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        self.assertEqual(gp.d_delta.shape, (N,))
        self.assertTrue(np.all(gp.d_delta <= 0.0))
        self.assertAlmostEqual(float(gp.d_delta.min()), -gk.LIVE_GAIN, places=6)
        self.assertTrue(np.array_equal(gp.d_delta < 0, gp.live > 0))

    def test_deterministic(self):
        m = _make(_hold_ball)
        a = gk.detect(m, _st(), events=np.zeros(0)).live
        b = gk.detect(m, _st(), events=np.zeros(0)).live
        self.assertTrue(np.array_equal(a, b))

    def test_info_keys(self):
        m = _make(_hold_ball)
        gp = gk.detect(m, _st(), events=np.zeros(0))
        for k in ("n_episodes", "n_seed_carry", "n_seed_gathered", "n_gathered_accepted",
                  "coverage", "slot_rejected", "neither_plausible"):
            self.assertIn(k, gp.info)

    def test_events_may_be_none_or_empty(self):
        m = _make(_hold_ball)
        empty = np.zeros(0, dtype=[("start_frame", "i4")])
        gp = gk.detect(m, _st(), events=empty)
        self.assertGreater(gp.live.sum(), 0.0)


def _st_full():
    """Structure covering the whole synthetic array (the tail tests run to frame N-1)."""
    return Structure(halves={1: (50, N)}, kickoffs={1: 50},
                     attack_dir=np.array([[1, -1], [-1, 1]], np.int8),
                     forced_dead_mask=np.zeros(N, bool))


class TestMissingRunAtArrayEnd(unittest.TestCase):
    """E07's one-frame overrun: a ball-missing run ending 0..40 frames before the array end.

    The gathered branch measured the return speed at ``present[j + 1 + 25]``; when the array
    ended 3-25 frames after the run, that index was exactly ``n``.  The present tail is the
    return, not the episode: it must not be labelled live.
    """

    GAP = 50  # 2 s, inside the 6 s miss cap

    def _case(self, tail):
        gap_end = N - 1 - tail
        gap_start = gap_end - self.GAP + 1

        def ball(f):
            if f < gap_start:
                return (5.6, PITCH_W / 2)          # at the home keeper, slow
            if f <= gap_end:
                return None                        # the tracker loses it
            # reappears away from the keeper at 20 m/s: a throw, not a carry
            return (60.0 + 20.0 * (f - gap_end - 1) / FPS, PITCH_W / 2)

        return _make(ball, half_ranges=((50, N, 1),)), gap_start, gap_end

    def test_no_indexerror_and_tail_not_live(self):
        for tail in range(41):
            for include in (False, True):
                with self.subTest(tail=tail, include_gathered=include):
                    m, gap_start, gap_end = self._case(tail)
                    gp = gk.detect(m, _st_full(), events=np.zeros(0),
                                   params=dict(include_gathered=include))
                    self.assertEqual(gp.live.shape, (N,))
                    self.assertTrue(np.all(gp.live[gap_end + 1:] == 0.0))
                    if include:
                        # the gap is still accepted: the clamp did not change the semantics
                        self.assertEqual(gp.info["n_gathered_accepted"], 1)
                        self.assertTrue(np.all(gp.live[gap_start:gap_end + 1] == 1.0))
                    else:
                        self.assertEqual(float(gp.live.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
