"""E14 post-processing rules: label-only, deterministic, no match data.

Everything here runs on small hand-built label arrays, so the module's contract is pinned without the
external games (which are absent on a clean clone).
"""
import inspect
import unittest

import numpy as np

from possession import evaluate
from possession import postprocess as pp


def codes(s):
    return pp.to_codes(np.array(list(s)))


class Codes(unittest.TestCase):
    def test_strings_and_codes(self):
        np.testing.assert_array_equal(codes("HADX"), [0, 1, 2, 3])
        np.testing.assert_array_equal(pp.to_codes(np.array([0, 1, 2, 3], np.int8)), [0, 1, 2, 3])

    def test_rejects_unknown_label_and_out_of_range_code(self):
        with self.assertRaises(ValueError):
            pp.to_codes(["H", "Q"])
        with self.assertRaises(ValueError):
            pp.to_codes(np.array([0, 4]))

    def test_output_keeps_input_kind_and_does_not_mutate(self):
        a = np.array(list("HHADDH"))
        before = a.copy()
        out = pp.absorb_short(a, [(0, 6)], min_frames=3)
        self.assertEqual(out.dtype.kind, "U")
        np.testing.assert_array_equal(a, before)
        c = codes("HHADDH")
        beforec = c.copy()
        out = pp.absorb_short(c, [(0, 6)], min_frames=3)
        self.assertEqual(out.dtype.kind, "i")
        np.testing.assert_array_equal(c, beforec)


class Runs(unittest.TestCase):
    def test_runs_from_half(self):
        half = np.array([0, 1, 1, 1, 0, 0, 2, 2])
        self.assertEqual(pp.runs_from_half(half), [(1, 4), (6, 8)])

    def test_segments(self):
        lab = codes("HHADDH")
        self.assertEqual(pp.segments(lab, [(0, 6)]), [(0, 2, 0), (2, 3, 1), (3, 5, 2), (5, 6, 0)])

    def test_fill_matches_scorer(self):
        lab = codes("XHHDXXAX")
        runs = [(0, 8)]
        filled = pp._fill(lab, runs)
        np.testing.assert_array_equal(filled, evaluate.pm.fill_abstentions(lab, runs))

    def test_n_spells_matches_scorer_with_x(self):
        lab = codes("HHXXAADDXHHA")
        runs = [(0, 12)]
        d = evaluate.diagnostics(lab, runs=runs)
        self.assertEqual(pp.n_spells(lab, runs), d["n_spells"])

    def test_n_spells_counts_across_a_run_boundary(self):
        lab = codes("HHADDAA")
        self.assertEqual(pp.n_spells(lab, [(0, 7)]), 3)
        self.assertEqual(pp.n_spells(lab, [(0, 3), (3, 7)]), 3)   # the run split separates A from D


class Absorb(unittest.TestCase):
    def test_short_spell_goes_to_longest_live_neighbor(self):
        lab = codes("H" * 10 + "A" * 5 + "H" * 30)
        np.testing.assert_array_equal(pp.absorb_short(lab, [(0, 45)], 12, "longest_live"), [0] * 45)

    def test_short_spell_goes_to_the_other_neighbor_when_it_is_longer(self):
        lab = codes("H" * 5 + "A" * 30 + "D" * 10)
        np.testing.assert_array_equal(pp.absorb_short(lab, [(0, 45)], 12, "longest_live"),
                                      [1] * 35 + [2] * 10)

    def test_lone_spell_at_a_run_edge_becomes_d(self):
        lab = codes("H" * 5)
        np.testing.assert_array_equal(pp.absorb_short(lab, [(0, 5)], 12), [2] * 5)

    def test_d_is_never_absorbed_by_longest_live(self):
        lab = codes("H" * 30 + "D" * 5 + "A" * 30)
        np.testing.assert_array_equal(pp.absorb_short(lab, [(0, 65)], 12), [0] * 30 + [2] * 5 + [1] * 30)

    def test_policies(self):
        lab = codes("H" * 30 + "A" * 5 + "H" * 30)
        self.assertEqual(pp.absorb_short(lab, [(0, 65)], 12, "to_d")[30:35].tolist(), [2] * 5)
        self.assertEqual(pp.absorb_short(lab, [(0, 65)], 12, "prev")[30:35].tolist(), [0] * 5)
        self.assertEqual(pp.absorb_short(lab, [(0, 65)], 12, "next")[30:35].tolist(), [0] * 5)
        self.assertEqual(pp.absorb_short(lab, [(0, 65)], 12, "longest")[30:35].tolist(), [0] * 5)

    def test_shortest_first_and_deterministic(self):
        lab = codes("H" * 30 + "A" * 4 + "H" * 30 + "A" * 7 + "H" * 30)
        one = pp.absorb_short(lab, [(0, 101)], 12, "longest_live")
        two = pp.absorb_short(lab, [(0, 101)], 12, "longest_live")
        np.testing.assert_array_equal(one, two)
        self.assertEqual(pp.n_spells(one, [(0, 101)]), 1)

    def test_same_neighbors_guard_leaves_transitions_alone(self):
        flick = codes("H" * 30 + "A" * 5 + "H" * 30)
        self.assertEqual(pp.absorb_short(flick, [(0, 65)], 12, "to_d", True)[30:35].tolist(), [2] * 5)
        transition = codes("H" * 30 + "A" * 5 + "D" * 10 + "A" * 30)
        np.testing.assert_array_equal(pp.absorb_short(transition, [(0, 75)], 12, "to_d", True), transition)
        self.assertEqual(pp.absorb_short(transition, [(0, 75)], 12, "to_d", False)[30:35].tolist(), [2] * 5)

    def test_noop_below_two_frames(self):
        lab = codes("HADH")
        np.testing.assert_array_equal(pp.absorb_short(lab, [(0, 4)], 1), lab)
        with self.assertRaises(ValueError):
            pp.absorb_short(lab, [(0, 4)], 12, "sideways")


class Merge(unittest.TestCase):
    def test_same_team_short_d_is_filled(self):
        lab = codes("H" * 30 + "D" * 10 + "H" * 30)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 70)], 12), [0] * 70)

    def test_other_team_d_is_left_alone_by_default(self):
        lab = codes("H" * 30 + "D" * 10 + "A" * 30)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 70)], 12), lab)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 70)], 12, False), [0] * 40 + [1] * 30)

    def test_long_d_and_edge_d_are_untouched(self):
        lab = codes("H" * 30 + "D" * 20 + "H" * 30)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 80)], 12), lab)
        edge = codes("D" * 10 + "H" * 30 + "D" * 10)
        np.testing.assert_array_equal(pp.merge_across_d(edge, [(0, 50)], 12), edge)

    def test_exactly_max_d_is_filled(self):
        lab = codes("H" * 10 + "D" * 12 + "H" * 10)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 32)], 12), [0] * 32)

    def test_noop_at_zero(self):
        lab = codes("H" * 10 + "D" * 5 + "H" * 10)
        np.testing.assert_array_equal(pp.merge_across_d(lab, [(0, 25)], 0), lab)


class Flicker(unittest.TestCase):
    def test_fixed_point_is_idempotent(self):
        lab = codes("H" * 30 + "A" * 8 + "D" * 6 + "H" * 30 + "A" * 4 + "H" * 30)
        one = pp.flicker(lab, [(0, 108)], 12, 12, absorb="longest_live")
        two = pp.flicker(one, [(0, 108)], 12, 12, absorb="longest_live")
        np.testing.assert_array_equal(one, two)
        self.assertEqual(pp.n_spells(one, [(0, 108)]), 1)
        # and the shipping policy (to D) is a fixed point too
        d1 = pp.flicker(lab, [(0, 108)], 12, 0)          # shipping policy at a 0.48 s floor
        np.testing.assert_array_equal(d1, pp.flicker(d1, [(0, 108)], 12, 0))
        self.assertEqual(pp.n_spells(d1, [(0, 108)]), 3)  # H D H D H: two flickers became D
        self.assertEqual(pp.n_spells(lab, [(0, 108)]), 5)

    def test_both_orders_are_deterministic(self):
        lab = codes("H" * 30 + "A" * 8 + "D" * 6 + "A" * 30)
        a = pp.flicker(lab, [(0, 74)], 12, 12, order="merge_then_absorb")
        b = pp.flicker(lab, [(0, 74)], 12, 12, order="merge_then_absorb")
        np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            pp.flicker(lab, [(0, 74)], 12, 12, order="nope")


class ToX(unittest.TestCase):
    def test_x_on_short_d_only(self):
        lab = codes("H" * 30 + "D" * 10 + "H" * 30)
        out = pp.d_to_x(lab, [(0, 70)], 12)
        self.assertEqual(out[30:40].tolist(), [3] * 10)
        self.assertEqual(out[29], 0)
        self.assertEqual(out[40], 0)
        long_d = codes("H" * 30 + "D" * 30 + "H" * 30)
        np.testing.assert_array_equal(pp.d_to_x(long_d, [(0, 90)], 12), long_d)

    def test_scorer_sees_the_merge(self):
        lab = codes("H" * 30 + "D" * 10 + "A" * 30)
        as_x = pp.d_to_x(lab, [(0, 70)], 12)
        merged = pp.merge_across_d(lab, [(0, 70)], 12, same_team_only=False)
        np.testing.assert_array_equal(pp._fill(pp.to_codes(as_x), [(0, 70)]), merged)
        self.assertEqual(pp.n_spells(as_x, [(0, 70)]), pp.n_spells(merged, [(0, 70)]))


class XLowConf(unittest.TestCase):
    def test_x_delays_the_flip_inside_the_window(self):
        lab = codes("H" * 40 + "A" * 40)
        out = pp.x_low_conf(lab, [(0, 80)], np.ones(80, bool), half_width=5)
        filled = pp._fill(pp.to_codes(out), [(0, 80)])
        self.assertEqual(filled[44], 0)      # old team carried through the X
        self.assertEqual(filled[45], 1)
        self.assertEqual(pp.n_spells(out, [(0, 80)]), 2)

    def test_only_certain_frames_and_no_new_spells(self):
        lab = codes("H" * 40 + "A" * 40)
        unc = np.zeros(80, bool)
        unc[40:42] = True
        out = pp.x_low_conf(lab, [(0, 80)], unc, half_width=5)
        self.assertEqual((pp.to_codes(out) == 3).sum(), 2)
        self.assertEqual(pp.to_codes(out)[30], 0)
        self.assertEqual(pp.n_spells(out, [(0, 80)]), 2)

    def test_bad_uncertain_length(self):
        with self.assertRaises(ValueError):
            pp.x_low_conf(codes("HA"), [(0, 2)], np.ones(3, bool))


class Apply(unittest.TestCase):
    def test_flag_off_is_identity_and_on_is_flicker(self):
        lab = codes("H" * 30 + "A" * 8 + "D" * 6 + "H" * 30)
        np.testing.assert_array_equal(pp.apply(lab, [(0, 74)], enable=False), lab)
        np.testing.assert_array_equal(pp.apply(lab, [(0, 74)], 12, 12, enable=True),
                                      pp.flicker(lab, [(0, 74)], 12, 12))
        default = pp.apply(lab, [(0, 74)], 12, 12)
        expected = pp.flicker(lab, [(0, 74)], 12, 12) if pp.ENABLED else lab
        np.testing.assert_array_equal(default, expected)

    def test_enabled_is_a_flag_not_a_decision(self):
        self.assertIsInstance(pp.ENABLED, bool)


class Diagnose(unittest.TestCase):
    def test_shape(self):
        lab = codes("H" * 100 + "D" * 50 + "A" * 20)
        d = pp.diagnose(lab, [(0, 170)])
        self.assertEqual(d["n_spells"], 2)
        self.assertEqual(d["flickers_lt_1s"], 1)    # the 20-frame A spell is 0.8 s
        self.assertAlmostEqual(d["median_s"], (4.0 + 0.8) / 2)
        self.assertEqual(d["n_d_stretches"], 1)
        for k in ("n_spells_per_5min", "hist", "share_gt_20s", "d_median_s"):
            self.assertIn(k, d)

    def test_x_is_read_as_the_scorer_reads_it(self):
        self.assertEqual(pp.diagnose(codes("HHXXHH"), [(0, 6)])["n_spells"], 1)   # X bridges
        self.assertEqual(pp.diagnose(codes("HHDDHH"), [(0, 6)])["n_spells"], 2)   # D splits


class Purity(unittest.TestCase):
    def test_no_match_data_and_no_io(self):
        import ast
        src = inspect.getsource(pp)
        tree = ast.parse(src)
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertEqual(mods, {"numpy"})          # no match data, no fitted model, no I/O
        for bad in ("np.load", "open(", "requests"):
            self.assertNotIn(bad, src)
        self.assertEqual(pp.__dict__["np"].__name__, "numpy")

    def test_no_module_state_changes_between_calls(self):
        lab = codes("H" * 30 + "A" * 8 + "H" * 30)
        first = pp.apply(lab, [(0, 68)], 12, 12)
        second = pp.apply(lab, [(0, 68)], 12, 12)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(pp.DEFAULT["absorb"], "to_d")
        self.assertEqual(pp.DEFAULT["min_spell"], 38)   # 1.5 s, top of R08's label-safe band


if __name__ == "__main__":
    unittest.main()
