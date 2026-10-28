"""Synthetic tests for the generic tracking cleaner (E16)."""
import unittest
from types import SimpleNamespace

import numpy as np

from possession import clean

FPS = 25.0


def make(n=200, slots=6):
    """Two teams of three, GK = first slot of each team, one half, players drifting slowly."""
    half = np.ones(n, np.int8)
    t = np.arange(n, dtype=np.float32)[:, None, None] / FPS
    base = np.array([[10, 20], [30, 10], [50, 30], [70, 50], [80, 20], [90, 40]], np.float32)
    P = base[None] + 0.5 * t
    ball = np.stack([40 + 2 * t[:, 0, 0], 34 + 0 * t[:, 0, 0]], axis=1).astype(np.float32)
    return SimpleNamespace(
        P=P.astype(np.float32), V=np.ones((n, slots), np.float32), T=np.array([0, 0, 0, 1, 1, 1], np.int8),
        B=ball, BS=np.full(n, np.nan), gk=np.array([[0, 3], [0, 3]]), graded=np.ones(n, bool), half=half)


class Clean(unittest.TestCase):
    def test_clean_input_is_untouched(self):
        m = make()
        c, info = clean.clean_match(m)
        np.testing.assert_array_equal(c.P, m.P)
        np.testing.assert_array_equal(c.B, m.B)
        self.assertEqual(int((info["flags"] & 0x0F).sum()), 0)

    def test_ball_spike_dropped_relocation_kept_flagged(self):
        m = make()
        m.B[50] = [100.0, 60.0]
        m.B[120:] += [30.0, 0.0]  # tracker switches object and stays there
        c, info = clean.clean_match(m)
        self.assertTrue(np.isnan(c.B[50]).all())
        self.assertTrue(np.isfinite(c.B[120:]).all())
        self.assertTrue(info["flags"][50] & clean.FLAGS["ball_jump"])
        self.assertTrue(info["flags"][120] & clean.FLAGS["ball_relocation"])

    def test_player_jump_masks_both_sides(self):
        m = make()
        m.P[80, 1] += [20.0, 0.0]
        c, _ = clean.clean_match(m)
        self.assertTrue(np.isnan(c.P[80, 1]).all())
        self.assertTrue(np.isfinite(c.P[100, 1]).all())

    def test_duplicate_only_dropped_when_team_has_surplus(self):
        m = make(slots=6)
        # slot 2 shadows slot 1 for 60 frames (0.5 m apart) while the team has only 3 -> keep
        m.P[:, 2] = m.P[:, 1] + [0.5, 0.0]
        c, _ = clean.clean_match(m, params=dict(max_team=3))  # 3 tracked == not above 3
        self.assertTrue(np.isfinite(c.P[:, 1:3, 0]).all())
        # a fourth slot of team 0 appears: now count 4 > 3 and the shadow is a duplicate
        P = np.concatenate([m.P, np.full_like(m.P[:, :1], np.nan)], axis=1)
        P[:, 6] = P[:, 1] + [0.4, 0.0]
        T = np.array([0, 0, 0, 1, 1, 1, 0], np.int8)
        m2 = SimpleNamespace(**{**m.__dict__, "P": P, "V": np.ones(P.shape[:2], np.float32), "T": T})
        c2, info = clean.clean_match(m2, params=dict(max_team=3))
        self.assertTrue(np.isnan(c2.P[:, 6]).all())
        self.assertTrue(np.isfinite(c2.P[:, 1]).all())
        self.assertGreater(info["counts"]["duplicate_samples"], 0)

    def test_goalkeeper_never_dropped_as_duplicate_or_peripheral(self):
        m = make()
        P = np.concatenate([m.P, m.P[:, :1] + [0.3, 0.0]], axis=1)
        T = np.array([0, 0, 0, 1, 1, 1, 0], np.int8)
        m2 = SimpleNamespace(**{**m.__dict__, "P": P, "V": np.ones(P.shape[:2], np.float32), "T": T})
        c, _ = clean.clean_match(m2, params=dict(max_team=3))
        self.assertTrue(np.isfinite(c.P[:, 0]).all())  # slot 0 is the team-0 goalkeeper
        self.assertTrue(np.isnan(c.P[:, 6]).all())

    def test_peripheral_surplus_trimmed_central_surplus_flagged(self):
        m = make()
        P = np.concatenate([m.P, np.zeros_like(m.P[:, :1])], axis=1)
        P[:, 6] = [50.0, 0.5]  # standing on the touchline
        T = np.array([0, 0, 0, 1, 1, 1, 0], np.int8)
        m2 = SimpleNamespace(**{**m.__dict__, "P": P, "V": np.ones(P.shape[:2], np.float32), "T": T})
        c, info = clean.clean_match(m2, params=dict(max_team=3))
        self.assertTrue(np.isnan(c.P[:, 6]).all())
        P2 = P.copy()
        P2[:, 6] = [50.0, 34.0]  # central: cannot tell who is surplus
        m3 = SimpleNamespace(**{**m2.__dict__, "P": P2})
        c3, info3 = clean.clean_match(m3, params=dict(max_team=3))
        self.assertTrue(np.isfinite(c3.P[:, 6]).all())
        self.assertTrue((info3["flags"] & clean.FLAGS["team_extra"]).all())

    def test_input_not_mutated_and_deterministic(self):
        m = make()
        m.P[80, 1] += [20.0, 0.0]
        before = m.P.copy()
        a, ia = clean.clean_match(m)
        b, ib = clean.clean_match(m)
        np.testing.assert_array_equal(m.P, before)
        np.testing.assert_array_equal(a.P, b.P)
        np.testing.assert_array_equal(ia["flags"], ib["flags"])


if __name__ == "__main__":
    unittest.main()
