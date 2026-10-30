"""dev-v1 harness: frozen spec, reference shift, statistics on synthetic counts, and one real-cell check against
evaluate.score. The real-cell test skips when the external games are absent."""
import inspect
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from possession import dev_v1, evaluate, external, make_submission


def _fake(name, tp, npd=100, ngt=100, seeds=dev_v1.SEEDS):
    """DevV1Result with the same per-game counts in every cell; `tp` (G,) scales the 5 IoU rows."""
    raw = {}
    for g in range(len(dev_v1.GAMES)):
        for s in seeds:
            for k, v in dev_v1.CELLS:
                for sh in dev_v1.SHIFT_FRAMES:
                    raw[(g, s, k, v, sh)] = (np.full(5, tp[g]), npd, ngt)
    return dev_v1.DevV1Result(name, raw)


class Frozen(unittest.TestCase):
    def test_spec(self):
        self.assertEqual(len(dev_v1.GAMES), 10)
        self.assertEqual(dev_v1.SEEDS, (0, 1, 2))
        self.assertEqual(len(dev_v1.GROUPS["primary"]), 6)
        self.assertEqual(len(dev_v1.GROUPS["stress"]), 4)
        self.assertEqual({v for _, v in dev_v1.GROUPS["primary"]}, {.05, .12, .23})
        self.assertEqual({v for _, v in dev_v1.GROUPS["stress"]}, {.30, .5})
        self.assertEqual(dev_v1.params(), dict(dropout_live_cap_s=6.0, d_gap_jitter=0.25))
        self.assertEqual(dev_v1._DRAWS.shape, (2000, 10))

    def test_not_in_submission_path(self):
        self.assertNotIn("dev_v1", inspect.getsource(make_submission))

    def test_provenance(self):
        line = dev_v1.provenance()
        self.assertIn("possession=", line)
        self.assertIn(dev_v1.DEGRADER_COMMIT, line)

    def test_game_list_exists(self):
        listed = external.list_games()
        if len(listed) < 3:
            self.skipTest("external dev games not on disk")
        self.assertTrue(set(dev_v1.GAMES) <= set(listed))


class Shift(unittest.TestCase):
    def test_shift_ref_stays_inside_runs(self):
        ref = np.array([0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 0, 0, 1, 1], np.int8)
        runs = np.array([[0, 8], [10, 14]])
        np.testing.assert_array_equal(dev_v1._shift_ref(ref, runs, 0), ref)
        later = dev_v1._shift_ref(ref, runs, 2)
        np.testing.assert_array_equal(later[:8], [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(later[8:10], ref[8:10])           # outside the runs untouched
        np.testing.assert_array_equal(later[10:], [0, 0, 0, 0])
        earlier = dev_v1._shift_ref(ref, runs, -2)
        np.testing.assert_array_equal(earlier[:8], [0, 1, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(earlier[10:], [1, 1, 1, 1])


class Stats(unittest.TestCase):
    def test_identical_results_compare_to_zero(self):
        a = _fake("a", np.linspace(30, 60, 10))
        c = dev_v1.compare_dev_v1(a, a)
        self.assertEqual(c["headline"]["delta"], 0.0)
        self.assertEqual(c["headline"]["neg"], 0)
        self.assertFalse(c["headline"]["excludes_0"])
        self.assertFalse(c["tier1"]["headline_ci_excludes_0"])

    def test_clear_win_passes_tier1_mechanics(self):
        base = np.linspace(30, 50, 10)
        a, b = _fake("a", base + 6), _fake("b", base)
        c = dev_v1.compare_dev_v1(a, b)
        self.assertGreater(c["headline"]["delta"], 0.03)
        self.assertTrue(c["headline"]["excludes_0"])
        self.assertEqual(c["headline"]["neg"], 0)
        self.assertTrue(all(c["tier1"].values()), c["tier1"])
        self.assertEqual(len(c["cells"]), 6 + 4 + 2)
        self.assertEqual(set(c["shift"]), set(dev_v1.SHIFT_FRAMES))
        rev = dev_v1.compare_dev_v1(b, a)
        self.assertAlmostEqual(rev["headline"]["delta"], -c["headline"]["delta"], places=4)
        self.assertEqual(rev["headline"]["neg"], 10)
        self.assertFalse(rev["tier1"]["headline_ci_excludes_0"])

    def test_draws_are_fixed_and_shared(self):
        a, b = _fake("a", np.linspace(30, 60, 10)), _fake("b", np.linspace(30, 60, 10)[::-1])
        self.assertEqual(dev_v1.compare_dev_v1(a, b)["headline"], dev_v1.compare_dev_v1(a, b)["headline"])

    def test_group_score_matches_pooled(self):
        r = _fake("a", np.full(10, 50.0), npd=100, ngt=100)
        self.assertAlmostEqual(r.group_score("primary"), 2 * 500 / (2 * 500 + 500 + 500), places=6)
        self.assertIn("headline", str(r))
        self.assertIn("tier-1", str(dev_v1.compare_dev_v1(r, r)))


@unittest.skipIf(len(external.list_games()) < 3, "external dev games not on disk")
class RealCell(unittest.TestCase):
    def test_cell_flags_match_evaluate_score(self):
        gid = "metrica:game1"
        clean = external.load_game(gid)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"MTAH_CACHE_DIR": d}):
            m = dev_v1.degraded_match(clean, 0, 1.0, 0.12)
            again = dev_v1.degraded_match(clean, 0, 1.0, 0.12)        # served from the cache
            self.assertTrue(np.array_equal(m.B, again.B, equal_nan=True))
            self.assertTrue(np.array_equal(m.P, again.P, equal_nan=True))
            cell = dev_v1.Cell(gid, clean, m, 0, 1.0, 0.12)
            pred = evaluate.to_codes(dev_v1.baseline_plus(cell))
        ref = evaluate.to_codes(clean.ref)
        f, npd = evaluate._spell_flags(pred, ref, clean.runs)
        want = evaluate.score(pred, ref, clean.runs)["spell_f1_mean"]
        self.assertAlmostEqual(float(evaluate._pooled_f1(f.sum(1), npd, f.shape[1])), want, places=9)


if __name__ == "__main__":
    unittest.main()
