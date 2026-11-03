"""dev-v1 harness: frozen spec, reference shift, statistics on synthetic counts, and one real-cell check against
evaluate.score. The real-cell test skips when the external games are absent.

S42 adds the D-interval diagnostic tests: fast synthetic checks of extraction/matching/aggregation and an
opt-in acceptance test (`MTAH_S42_ACCEPTANCE=1`) that reproduces E75's stage-ownership table and E07's
stage-B interval F1 at seed 0. The acceptance test runs 80 dev cells (~15 min on 2 forks), so it is not
part of the default suite; the numbers are recorded in research/S42_d_intervals.md.
"""
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


class DIntervalsSynthetic(unittest.TestCase):
    """Fast unit checks of S42's extraction, greedy matching, signed summaries and ownership trace."""

    def test_d_runs_splits_halves_and_clips(self):
        lab = np.array([0, 2, 2, 0, 2, 2, 2, 0, 2, 2], np.int8)
        half = np.array([1] * 5 + [2] * 5, np.int8)
        np.testing.assert_array_equal(dev_v1.d_runs(lab, half),
                                      np.array([[1, 3], [4, 5], [5, 7], [8, 10]]))
        np.testing.assert_array_equal(dev_v1.d_runs(lab, half, [(0, 4), (5, 10)]),
                                      np.array([[1, 3], [5, 7], [8, 10]]))
        np.testing.assert_array_equal(dev_v1.d_runs(np.array(["H", "D", "X", "D", "D"]),
                                                    np.ones(5, np.int8)),
                                      np.array([[1, 2], [3, 5]]))

    def test_greedy_iou_match_strongest_first(self):
        pred = np.array([[0, 10], [20, 30], [40, 50]])
        ref = np.array([[0, 9], [23, 30], [41, 49]])
        m = dev_v1._iou_match(pred, ref, .5)
        self.assertEqual([(i, j) for i, j, _ in m], [(0, 0), (2, 2), (1, 1)])
        self.assertAlmostEqual(m[0][2], 9 / 10, places=9)
        self.assertAlmostEqual(m[2][2], 7 / 10, places=9)
        self.assertEqual(dev_v1._iou_match(pred, ref, .95), [])

    def test_signed_distribution_shares(self):
        d = dev_v1._dist([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.assertEqual(d["n"], 5)
        self.assertAlmostEqual(d["median"], 0.0)
        self.assertAlmostEqual(d["late_1s"], .2)       # strictly > 1 s
        self.assertAlmostEqual(d["early_1s"], .2)      # strictly < -1 s
        self.assertAlmostEqual(d["within_1s"], .6)
        self.assertEqual(dev_v1._dist([np.nan]), {"n": 0})

    def test_ownership_traces_last_moved_end(self):
        # Three final blocks: one centred gate block, one growd extension, one floor-created block.
        half = np.ones(20, np.int8)
        gt = np.array([[2, 6], [10, 13], [16, 18]], np.int64)
        final = np.array(list("HH" + "D" * 6 + "HH" + "D" * 4 + "HH" + "D" * 2 + "HH"), dtype="<U1")
        baseline = final.copy()
        gate = final.copy()
        growd = final.copy()
        # block 1: baseline ends at 6, gate/growd/final at 8 -> gate owns the end
        baseline[6:8] = "H"
        # block 2: gate ends at 12, growd/final at 14 -> growd owns the end
        gate[12:14] = "H"
        growd[12:14] = "D"
        # block 3: no stage before the floor has D there -> floor creates it
        for arr in (baseline, gate, growd):
            arr[16:18] = "H"
        stages = {"baseline": baseline, "gate": gate, "growd": growd}
        rows = dev_v1._ownership_records(final, stages, gt, half, "floor")
        by_owner = {r[0]: r for r in rows}
        self.assertEqual(set(by_owner), {"gate", "growd", "floor"})
        self.assertAlmostEqual(by_owner["gate"][3], (8 - 6) / 25)     # +2 frames late
        self.assertAlmostEqual(by_owner["growd"][3], (14 - 13) / 25)  # +1 frame late
        self.assertAlmostEqual(by_owner["floor"][3], 0.0)

    def _raw(self, tp, n_pred=100, n_ref=100, name="x"):
        """Minimal raw records for 10 games x one seed x one cell."""
        out = {}
        for g in range(len(dev_v1.GAMES)):
            out[(g, 0, 0.0, 0.05)] = {
                "counts": {t: (tp, n_pred, n_ref, 0.5 * tp, tp) for t in (.3, .5, .7)},
                "edges": {t: [(0.2, 0.1, 0.6, 1, 2, 0, 1)] for t in (.3, .5, .7)},
                "overlap": [(0.0, 0.1, 1, 2, 0, 1)],
                "ownership": None, "n_pred": n_pred, "n_ref": n_ref}
        return out

    def test_public_score_and_compare_aggregate(self):
        raw = {"_s42": self._raw(50)}
        with mock.patch.object(dev_v1, "_run_d_intervals", return_value=raw):
            res = dev_v1.d_interval_score("ship", seeds=(0,), cells=[(0.0, 0.05)])
        self.assertAlmostEqual(res["f1"][.5], 2 * 500 / (1000 + 1000), places=9)
        self.assertEqual(res["edges"][.5]["n"], 10)
        self.assertIn("diagnostic, never a selector", res["note"])
        self.assertIn("D intervals", str(res))

        ra, rb = self._raw(60), self._raw(40)
        with mock.patch.object(dev_v1, "_run_d_intervals", return_value={"a": ra, "b": rb}):
            cmp = dev_v1.compare_d_intervals("master", "ship", seeds=(0,), cells=[(0.0, 0.05)])
        self.assertGreater(cmp["delta"][.5]["delta"], 0)
        self.assertEqual(cmp["delta"][.5]["neg"], 0)
        self.assertIn("D intervals", str(cmp))
        self.assertIn("diagnostic, never a selector", str(cmp))


_E75_OWNER_SEED0 = {"baseline": .229, "gate": .480, "growd": .065,
                    "ingap": .002, "subgap": .110, "floor": .114}
_E75_OWNER_PUBLISHED = {"baseline": .234, "gate": .482, "growd": .065,
                        "ingap": .002, "subgap": .104, "floor": .112}
_E07_SHIP = {(0.0, .05): .7952, (0.0, .12): .7609, (0.0, .23): .7288,
             (1.0, .05): .6459, (1.0, .12): .6099, (1.0, .23): .5966}
_E07_E66 = {(0.0, .05): .7928, (0.0, .12): .7762, (0.0, .23): .7713,
            (1.0, .05): .5959, (1.0, .12): .5824, (1.0, .23): .5943}


@unittest.skipUnless(os.environ.get("MTAH_S42_ACCEPTANCE") == "1",
                     "set MTAH_S42_ACCEPTANCE=1 to run the 80-cell S42 acceptance test (~15 min)")
@unittest.skipIf(len(external.list_games()) < 3, "external dev games not on disk")
class DIntervalsAcceptance(unittest.TestCase):
    """Reproduces E75's stage table and E07's stage-B interval F1 at seed 0 (S42 acceptance).

    E75's published table is pooled over seeds 0-2 and, as its n=20,013 shows, over primary **plus
    the v=0 reference** cells (its prose says "primary cells"; the stored rows are primary+v0). The
    seed-0 expected shares below are computed from E75's own stored records
    (scratch/E75/out/*.pkl, primary+v0, seed 0), so this is a seed-0 reproduction of the same trace.
    """

    @classmethod
    def setUpClass(cls):
        cls.cells = dev_v1.GROUPS["primary"] + dev_v1.GROUPS["v0_reference"]
        stages = {n: (lambda c, n=n: dev_v1.d_stages(c)[n]) for n in
                  ("baseline", "gate", "growd", "ingap", "subgap")}
        fns = {"master": lambda c: dev_v1.d_stages(c)["floor"],
               "ship": lambda c: dev_v1.d_stages(c)["ship"],
               "e66": lambda c: dev_v1.d_stages(c)["e66"]}
        cls.raw = dev_v1._run_d_intervals(fns, seeds=(0,), cells=cls.cells, stages=stages,
                                          final_name="floor", stages_for="master", n_jobs=2)
        cls.master = dev_v1._score_from_raw("master", cls.raw["master"], (0,), cls.cells,
                                            (.3, .5, .7))

    def test_e75_stage_ownership_table(self):
        own = self.master["ownership"]
        self.assertEqual(own["n"], 6690)              # seed 0, primary+v0, 3 seeds would be 20,013
        for name, want in _E75_OWNER_SEED0.items():
            self.assertAlmostEqual(own["owners"][name]["share"], want, delta=.002,
                                   msg="seed-0 share %s" % name)
        for name, want in _E75_OWNER_PUBLISHED.items():
            # The published table is the 3-seed run; seed 0 must land within 1 pp on every owner.
            self.assertAlmostEqual(own["owners"][name]["share"], want, delta=.01,
                                   msg="published share %s" % name)
        # E75's pooled row: median end -0.12 s, IQR [-1.44,+0.40], 13.4% late / 28.3% early.
        end = self.master["overlap_edges"]["end"]
        self.assertAlmostEqual(end["median"], -.08, delta=.02)
        self.assertAlmostEqual(end["q1"], -1.32, delta=.02)
        self.assertAlmostEqual(end["q3"], .44, delta=.02)
        self.assertAlmostEqual(end["late_1s"], .137, delta=.005)
        self.assertAlmostEqual(end["early_1s"], .276, delta=.005)

    def test_e07_stage_b_interval_f1(self):
        primary = dev_v1.GROUPS["primary"]
        for name, want_pooled, want_cells, want_iou in (
                ("ship", .6933, _E07_SHIP, .9043), ("e66", .6870, _E07_E66, .8873)):
            raw = {k: v for k, v in self.raw[name].items() if (k[2], k[3]) in primary}
            res = dev_v1._score_from_raw(name, raw, (0,), primary, (.3, .5, .7))
            self.assertAlmostEqual(res["f1"][.5], want_pooled, delta=.0005,
                                   msg="%s pooled" % name)
            self.assertAlmostEqual(res["mean_iou"][.5], want_iou, delta=.0005,
                                   msg="%s mean IoU" % name)
            for kv, want in want_cells.items():
                self.assertAlmostEqual(res["per_cell"][kv][.5]["f1"], want, delta=.0005,
                                       msg="%s cell %s" % (name, kv))


if __name__ == "__main__":
    unittest.main()
