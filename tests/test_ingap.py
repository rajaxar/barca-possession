"""Synthetic tests for the in-gap ball-free decoder (E53)."""
import unittest

import numpy as np

from possession import ingap


def scene(n=600, gap=(100, 400), ph_pattern=None):
    """One half; ball missing in `gap`; Baseline+-style labels persist H through the gap, A after."""
    half = np.ones(n, np.int8)
    B = np.zeros((n, 2))
    B[gap[0]:gap[1]] = np.nan
    lab = np.array(["H"] * gap[1] + ["A"] * (n - gap[1]))
    ph = np.full(n, 0.9) if ph_pattern is None else ph_pattern
    return lab, B, half, ph


class InGap(unittest.TestCase):
    def test_off_by_default_returns_input(self):
        lab, B, half, ph = scene()
        self.assertFalse(ingap.ENABLED)
        self.assertIs(ingap.apply(lab, B, half, ph), lab)

    def test_recovers_in_gap_handover(self):
        lab, B, half, _ = scene()
        ph = np.full(len(lab), 0.9)
        ph[250:] = 0.1                                # ball-free model sees A from frame 250 on
        out = ingap.gapdec(lab, B, half, ph)
        self.assertTrue((out[100:245] == "H").all())
        self.assertTrue((out[255:400] == "A").all())
        np.testing.assert_array_equal(out[:100], lab[:100])   # outside the gap untouched
        np.testing.assert_array_equal(out[400:], lab[400:])

    def test_d_and_x_never_touched_and_short_gaps_skipped(self):
        lab, B, half, _ = scene()
        lab[150:300] = "D"
        ph = np.full(len(lab), 0.1)
        out = ingap.gapdec(lab, B, half, ph)
        np.testing.assert_array_equal(out[150:300], lab[150:300])
        lab2, B2, half2, ph2 = scene(gap=(100, 140))   # 40 frames < MIN_GAP
        np.testing.assert_array_equal(ingap.gapdec(lab2, B2, half2, np.full(len(lab2), 0.1)), lab2)

    def test_no_ball_present_change_without_gap(self):
        lab, B, half, _ = scene(gap=(0, 0))
        B[:] = 0.0
        out = ingap.gapdec(lab, B, half, np.full(len(lab), 0.1))
        np.testing.assert_array_equal(out, lab)

    def test_deterministic(self):
        lab, B, half, _ = scene()
        ph = np.random.RandomState(0).uniform(0.05, 0.95, len(lab))
        np.testing.assert_array_equal(ingap.gapdec(lab, B, half, ph), ingap.gapdec(lab, B, half, ph))

    def test_footprint_guard(self):
        lab, B, half, _ = scene()
        ph = np.full(len(lab), 0.9)
        ph[250:] = 0.1
        out = ingap.gapdec(lab, B, half, ph)
        fp = ingap.footprint(lab, out, B, half)
        self.assertGreater(fp["changed"], 0)
        self.assertEqual(fp["changed_on_ball_present_share"], 0.0)
        with self.assertRaises(RuntimeError):
            ingap.check_footprint(fp, max_share=0.01)      # 150/600 changed
        dead = np.zeros(len(lab), bool)
        dead[260:270] = True
        with self.assertRaises(RuntimeError):
            ingap.check_footprint(ingap.footprint(lab, out, B, half, dead))
        self.assertIs(ingap.check_footprint(fp, max_share=0.5), fp)

    def test_apply_enabled_runs_guard(self):
        lab, B, half, _ = scene()
        ph = np.full(len(lab), 0.9)
        ph[250:] = 0.1
        with self.assertRaises(RuntimeError):         # raw-style input: 25% changed > 3%
            ingap.apply(lab, B, half, ph, enabled=True)


if __name__ == "__main__":
    unittest.main()
