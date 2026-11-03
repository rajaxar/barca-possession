"""The E105 ball-in-play diagnostic: explicit playing clock, both denominators, E84 acceptance.

Match-data tests skip when the frozen CSV or the external games are absent.
"""
import os
import unittest

import numpy as np

from possession import config, dev_v1, diagnostics, structure


class PlayingClock(unittest.TestCase):
    def test_forced_block_is_excluded_from_both_sides(self):
        """Labels inside the published pre-kickoff block never count as live or as clock."""
        n = structure.HALF_BLOCKS[-1][2] + 1
        labels = np.full(n, "X", dtype="<U1")
        labels[0:100] = "H"
        labels[100:200] = "D"
        forced = diagnostics.forced_dead_mask(n)
        a, b = int(np.flatnonzero(forced)[0]), int(np.flatnonzero(forced)[-1])
        self.assertEqual((a, b), (110808, 113741))
        labels[a:a + 10] = "H"  # structurally wrong, and the diagnostic must not reward it
        out = diagnostics.inplay(labels, forced)
        self.assertEqual(out["playing_clock_frames"], 200)
        self.assertEqual(out["live_frames"], 100)
        self.assertEqual(out["live_label_in_forced_block"], 10)
        self.assertAlmostEqual(out["live_share_clock"], 0.5)

    def test_frozen_csv_arithmetic(self):
        path = diagnostics.CANONICAL_CSV
        if not os.path.isfile(path):
            self.skipTest("frozen submission CSV not on disk")
        labels = diagnostics.read_submission(path)
        out = diagnostics.inplay(labels)
        self.assertEqual(out["frames"], config.N_VIDEO_FRAMES)
        self.assertEqual(out["graded_frames"], 158857)
        self.assertEqual(out["forced_dead_frames"], 2934)
        self.assertEqual(out["playing_clock_frames"], 155923)
        self.assertEqual(out["live_frames"], 86198)
        self.assertTrue(out["forced_dead_all_d"])
        self.assertAlmostEqual(out["playing_clock_min"], 103.9487, places=3)
        self.assertAlmostEqual(out["live_min"], 57.4653, places=3)
        self.assertAlmostEqual(out["live_share_clock"], 0.5528, places=3)
        self.assertAlmostEqual(out["live_share_graded"], 0.5426, places=3)
        self.assertAlmostEqual(out["d_share_graded"], 0.4574, places=3)
        self.assertAlmostEqual(out["d_share_clock"], 0.4472, places=3)


class E84Reference(unittest.TestCase):
    def test_external_reference_reproduces(self):
        """Acceptance test: 55.6 +/- 6.4 live min and 58.2 +/- 6.4% share on the 10 dev games."""
        if len([g for g in dev_v1.GAMES if g.startswith(("idsse:", "metrica:"))]) < 10:
            self.skipTest("external dev-v1 games not on disk")
        ref = diagnostics.reference_stats(diagnostics.external_inplay())
        self.assertEqual(ref["n"], 10)
        check = diagnostics.check_e84_reference(ref)
        self.assertTrue(all(c["ok"] for c in check.values()), check)
        self.assertAlmostEqual(ref["live_min_mean"], 55.56, delta=0.01)
        self.assertAlmostEqual(ref["live_min_sd"], 6.37, delta=0.01)
        self.assertAlmostEqual(ref["live_share_mean"], 0.582, delta=0.001)
        self.assertAlmostEqual(ref["live_share_sd"], 0.064, delta=0.001)

    def test_provider_split_is_the_denominator_lesson(self):
        """Against IDSSE alone the minute axis says 'above average' while the rate says 'average'."""
        if not os.path.isfile(diagnostics.CANONICAL_CSV):
            self.skipTest("frozen submission CSV not on disk")
        if len([g for g in dev_v1.GAMES if g.startswith(("idsse:", "metrica:"))]) < 10:
            self.skipTest("external dev-v1 games not on disk")
        sens = diagnostics.diagnose()["sensitivity"]
        self.assertEqual(sens["by_provider"]["idsse"]["n"], 7)
        self.assertEqual(sens["by_provider"]["metrica"]["n"], 3)
        self.assertGreater(sens["by_provider"]["idsse"]["z_minutes"], 1.0)
        self.assertLess(abs(sens["by_provider"]["idsse"]["z_rate"]), 0.2)
        self.assertLessEqual(sens["loo"]["z_rate_min"], sens["loo"]["z_rate_max"])


if __name__ == "__main__":
    unittest.main()
