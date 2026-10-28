"""Synthetic tests for the touch detector."""
import unittest

import numpy as np

from possession.touches import detect_touches


def synthetic():
    n = 400
    half = np.ones(n, np.int8)
    t = np.arange(n)
    b = np.zeros((n, 2))
    b[:, 1] = 34.0
    # ball rolls at 2 m/s, is kicked at frame 200 to 15 m/s by a team-1 player, then flies
    b[:200, 0] = 20 + 0.08 * t[:200]
    b[200:, 0] = b[199, 0] + 0.6 * (t[200:] - 199)
    P = np.full((n, 4, 2), 0.0)
    P[:, :, 1] = [10, 60, 50, 5]
    P[:, 0, 0], P[:, 1, 0] = 5, 100
    P[:, 2, 0], P[:, 3, 0] = 90, 8
    P[:, 2] = np.c_[np.full(n, 36.0), np.full(n, 34.4)]  # slot 2 (team 1) beside the kick point
    P[:150, 2, 0] = 36.0 + (150 - t[:150]) * 0.0
    rng = np.random.default_rng(0)
    b = b + rng.normal(0, 0.01, b.shape)
    return P, np.array([0, 0, 1, 1], np.int8), b, half


class TouchTests(unittest.TestCase):
    def test_kick_found_with_team(self):
        P, T, B, half = synthetic()
        df = detect_touches(P, T, B, half, params=dict(mode="kinematic"))
        self.assertEqual(len(df), 1)
        self.assertLessEqual(abs(int(df.frame[0]) - 200), 3)
        self.assertEqual(int(df.team[0]), 1)
        self.assertEqual(df.kind[0], "kick")

    def test_no_player_no_touch(self):
        P, T, B, half = synthetic()
        P[:, :, 1] += 30
        df = detect_touches(P, T, B, half, params=dict(mode="kinematic"))
        self.assertEqual(len(df), 0)


if __name__ == "__main__":
    unittest.main()
