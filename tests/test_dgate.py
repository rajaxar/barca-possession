"""Synthetic tests for the missing-ball dead-ball gate (B04)."""
import unittest

import numpy as np

from possession import dgate


def scene(n=1000, gaps=((200, 400),)):
    half = np.ones(n, np.int8)
    B = np.zeros((n, 2))
    for a, b in gaps:
        B[a:b] = np.nan
    lab = np.array(["H"] * n)
    return lab, B, half


class Gate(unittest.TestCase):
    def test_long_dead_gap_becomes_d(self):
        lab, B, half = scene()
        pf = np.full(len(lab), 0.9)
        out, dead = dgate.gate(lab, B, half, pf)
        self.assertTrue((out[200:400] == "D").all())
        self.assertTrue(dead[200:400].all())
        self.assertTrue((out[:200] == "H").all() and (out[400:] == "H").all())

    def test_short_gap_or_low_pd_untouched(self):
        lab, B, half = scene(gaps=((200, 240),))
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.9))
        np.testing.assert_array_equal(out, lab)
        lab, B, half = scene()
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.3))
        np.testing.assert_array_equal(out, lab)

    def test_gate_uses_gap_mean_not_per_frame(self):
        lab, B, half = scene()
        pf = np.full(len(lab), 0.2)
        pf[200:300] = 0.95                                # first half of the gap high: mean 0.575 > .5
        out, _ = dgate.gate(lab, B, half, pf)
        self.assertTrue((out[200:400] == "D").all())

    def test_island_between_dead_gaps_absorbed(self):
        lab, B, half = scene(gaps=((200, 400), (450, 700)))
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.9))
        self.assertTrue((out[200:700] == "D").all())      # 50-frame island < ABSORB
        lab, B, half = scene(gaps=((200, 400), (600, 800)))
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.9))
        self.assertTrue((out[400:600] == "H").all())      # 200-frame island kept

    def test_ungraded_frames_not_gated(self):
        lab, B, half = scene()
        half[:300] = 0
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.9))
        np.testing.assert_array_equal(out[:300], lab[:300])

    def test_footprint_guard(self):
        lab, B, half = scene()
        out, _ = dgate.gate(lab, B, half, np.full(len(lab), 0.9))
        self.assertEqual(dgate.check_footprint(dgate.footprint(lab, out, half))["d_after"], 200)
        with self.assertRaises(RuntimeError):             # gate fired on nothing: broken model
            dgate.check_footprint(dgate.footprint(lab, lab, half))
        out[:] = "D"
        with self.assertRaises(RuntimeError):             # runaway gate
            dgate.check_footprint(dgate.footprint(lab, out, half))

    def test_features_ignore_ball_and_match_shapes(self):
        rng = np.random.default_rng(0)
        n = 600
        P = rng.normal(size=(n, 22, 2)) * 10
        V = np.abs(rng.normal(size=(n, 22))) * 3
        T = np.array([0] * 11 + [1] * 11)
        half = np.ones(n, np.int8)
        rows, X = dgate.sample_table(P, V, T, half)
        self.assertEqual(len(rows), len(X))
        self.assertFalse([c for c in X.columns if "ball" in c or c.startswith("pos_")])

    def test_training_is_deterministic_and_external_only(self):
        self.assertTrue(all(g.startswith(("idsse", "metrica")) for g in dgate.training_games()))


if __name__ == "__main__":
    unittest.main()
