"""Synthetic tests for the sub-gap gate (E66, from E69)."""
import unittest

import numpy as np

from possession import subgap

N = 1000


def scene(gap=(200, 400), gated_d=None):
    half = np.ones(N, np.int8)
    B = np.zeros((N, 2))
    B[gap[0]:gap[1]] = np.nan
    lab = np.array(["H"] * N)
    gated = lab.copy()
    if gated_d:
        gated[gated_d[0]:gated_d[1]] = "D"
    return lab, gated, B, half


class Overlay(unittest.TestCase):
    def test_hot_run_in_unfired_gap_becomes_d(self):
        lab, gated, B, half = scene()
        pf = np.full(N, 0.2)
        pf[250:320] = 0.9                                     # 70 frames >= T2
        out = subgap.overlay(lab, gated, B, half, pf)
        self.assertTrue((out[250:320] == "D").all())
        self.assertTrue((out[:250] == "H").all() and (out[320:] == "H").all())

    def test_short_hot_run_or_cold_gap_untouched(self):
        lab, gated, B, half = scene()
        pf = np.full(N, 0.2)
        pf[250:280] = 0.9                                     # 30 < T2
        np.testing.assert_array_equal(subgap.overlay(lab, gated, B, half, pf), lab)

    def test_fired_gap_and_short_gap_skipped(self):
        lab, gated, B, half = scene(gated_d=(300, 310))        # gate left a D inside the gap: gap counts as fired
        np.testing.assert_array_equal(subgap.overlay(lab, gated, B, half, np.full(N, 0.9)), lab)
        lab, gated, B, half = scene(gap=(200, 240))            # 40 < TF
        np.testing.assert_array_equal(subgap.overlay(lab, gated, B, half, np.full(N, 0.9)), lab)

    def test_only_ha_replaced_and_islands_absorbed(self):
        lab, gated, B, half = scene()
        lab[260:270] = "X"
        out = subgap.overlay(lab, gated, B, half, np.full(N, 0.9))
        self.assertTrue((out[260:270] == "X").all() and (out[200:260] == "D").all())
        lab, gated, B, half = scene(gap=(200, 400))
        pf = np.full(N, 0.1)
        pf[200:260] = 0.9
        pf[300:360] = 0.9                                      # 40-frame cold island between: < ABSORB
        out = subgap.overlay(lab, gated, B, half, pf)
        self.assertTrue((out[200:360] == "D").all())

    def test_disabled_returns_input(self):
        lab, gated, B, half = scene()
        m = type("M", (), dict(B=B, half=half))()
        out, fp = subgap.apply(lab, gated, m, np.full(N, 0.9), enabled=False)
        self.assertIs(out, lab)
        self.assertIsNone(fp)


class Guard(unittest.TestCase):
    def test_bounds(self):
        ok = dict(d_added_share_graded=0.01, changed_in_fired=0)
        self.assertIs(subgap.check_footprint(ok), ok)
        for bad in (dict(d_added_share_graded=0.0), dict(d_added_share_graded=0.2), dict(changed_in_fired=5)):
            with self.assertRaises(RuntimeError):
                subgap.check_footprint(dict(ok, **bad))

    def test_enabled_runs_guard(self):
        lab, gated, B, half = scene()
        m = type("M", (), dict(B=B, half=half))()
        with self.assertRaises(RuntimeError):                  # 200 of 1000 frames added > 15%
            subgap.apply(lab, gated, m, np.full(N, 0.9), enabled=True)


if __name__ == "__main__":
    unittest.main()
