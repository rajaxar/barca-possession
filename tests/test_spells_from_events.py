"""Synthetic tests for the event-first possession rules (E03). No match data, no frame literals."""
import unittest

import numpy as np
import pandas as pd

from possession import spells_from_events as sfe


def scene(n=240):
    """Ball rests at (10, 10) with a team-0 player on it; team 1 is far away."""
    half = np.ones(n, np.int8)
    B = np.tile([10.0, 10.0], (n, 1))
    P = np.zeros((n, 2, 2))
    P[:, 0] = [10.0, 10.0]      # slot 0, team 0, on the ball
    P[:, 1] = [60.0, 60.0]      # slot 1, team 1, far
    T = np.array([0, 1], np.int8)
    return P, T, B, half


def events(rows):
    return pd.DataFrame(rows, columns=["frame", "team", "conf"])


class RuleTests(unittest.TestCase):
    def test_flip_follows_touches_and_keeps_d(self):
        P, T, B, half = scene()
        base = np.array(["H"] * 240, dtype="<U1")
        base[200:210] = "D"
        base[230:] = "X"
        ev = events([(50, 1, 1.0), (150, 0, 1.0)])
        out = sfe.predict(P, T, B, half, base, rule="flip", params=dict(merge_s=0.0), events=ev)
        self.assertEqual("".join(out[:50]), "H" * 50)
        self.assertEqual("".join(out[50:150]), "A" * 100)
        self.assertEqual("".join(out[150:200]), "H" * 50)
        self.assertEqual("".join(out[200:]), "".join(base[200:]))
        self.assertEqual("".join(base), "H" * 200 + "D" * 10 + "H" * 20 + "X" * 10)

    def test_two_touch_needs_a_second_touch_without_a_hold(self):
        P, T, B, half = scene()
        base = np.array(["H"] * 240, dtype="<U1")
        one = sfe.predict(P, T, B, half, base, rule="two_touch",
                          params=dict(merge_s=0.0), events=events([(50, 1, 1.0)]))
        self.assertTrue(np.all(one == "H"))
        two = sfe.predict(P, T, B, half, base, rule="two_touch",
                          params=dict(merge_s=0.0), events=events([(50, 1, 1.0), (60, 1, 1.0)]))
        self.assertEqual(two[50], "A")
        self.assertEqual(two[49], "H")

    def test_revert_cancels_a_deflection(self):
        P, T, B, half = scene()
        base = np.array(["H"] * 240, dtype="<U1")
        ev = events([(50, 1, 1.0), (58, 0, 1.0)])      # old team touches again within 0.5 s
        out = sfe.predict(P, T, B, half, base, rule="revert",
                          params=dict(merge_s=0.0, revert_s=0.5), events=ev)
        self.assertTrue(np.all(out == "H"))
        ev2 = events([(50, 1, 1.0), (80, 0, 1.0)])     # old team back only after the window
        out2 = sfe.predict(P, T, B, half, base, rule="revert",
                           params=dict(merge_s=0.0, revert_s=0.5), events=ev2)
        self.assertEqual(out2[50], "A")
        self.assertEqual(out2[80], "H")

    def test_decoder_is_bounded_between_the_teams(self):
        P, T, B, half = scene()
        base = np.array(["H"] * 240, dtype="<U1")
        ev = events([(60, 1, 1.0), (70, 1, 0.8), (150, 0, 1.0)])
        out = sfe.predict(P, T, B, half, base, rule="decoder",
                          params=dict(merge_s=0.0, beta=0.05), events=ev)
        self.assertTrue(set("".join(out)) <= {"H", "A"})
        self.assertIn("A", "".join(out))
        again = sfe.predict(P, T, B, half, base, rule="decoder",
                            params=dict(merge_s=0.0, beta=0.05), events=ev)
        self.assertEqual("".join(out), "".join(again))

    def test_predict_does_not_mutate_base(self):
        P, T, B, half = scene()
        base = np.array(["H"] * 240, dtype="<U1")
        before = base.copy()
        sfe.predict(P, T, B, half, base, rule="flip", params=dict(merge_s=0.0),
                    events=events([(50, 1, 1.0)]))
        self.assertEqual("".join(before), "".join(base))

    def test_merge_deflections_drops_a_quick_opposing_touch(self):
        df = events([(100, 0, 1.0), (105, 1, 1.0), (130, 1, 1.0)])
        kept = sfe.merge_deflections(df, merge_s=0.5)
        self.assertEqual(list(kept.frame), [100, 130])

    def test_init_base_keeps_the_span_start_owner(self):
        P, T, B, half = scene()
        base = np.array(["A"] * 240, dtype="<U1")
        ev = events([(50, 1, 1.0)])
        auto = sfe.predict(P, T, B, half, base, rule="flip", params=dict(merge_s=0.0), events=ev)
        self.assertEqual(auto[0], "H")                 # nearest player wins the start
        keep = sfe.predict(P, T, B, half, base, rule="flip",
                           params=dict(merge_s=0.0, init_mode="base"), events=ev)
        self.assertEqual(keep[0], "A")


class AnchorInsertTests(unittest.TestCase):
    def base(self):
        return np.array(["H"] * 240, dtype="<U1")

    def test_anchor_moves_a_hand_over_to_the_new_teams_touch(self):
        base = self.base()
        base[100:] = "A"                                # base hand-over at 100
        ev = events([(94, 1, 1.0), (150, 0, 1.0)])
        out = sfe.anchor(base, np.ones(240, np.int8), ev, params=dict(anchor_w=8))
        self.assertEqual(out[93], "H")
        self.assertEqual(out[94], "A")
        self.assertEqual(out[150], "A")     # anchor never adds a hand-over of its own

    def test_anchor_never_moves_to_the_old_team(self):
        base = self.base()
        base[100:] = "A"
        ev = events([(94, 0, 1.0)])                     # only a touch by the OLD team nearby
        out = sfe.anchor(base, np.ones(240, np.int8), ev)
        self.assertEqual("".join(out), "".join(base))

    def test_insert_adds_only_where_the_base_has_no_hand_over(self):
        base = self.base()
        base[100:] = "A"
        ev = events([(50, 1, 1.0), (105, 0, 1.0)])      # 50 is new, 105 duplicates the base flip
        out = sfe.insert(base, np.ones(240, np.int8), ev, params=dict(merge_s=0.0, insert_w=12))
        self.assertEqual(out[50], "A")
        self.assertEqual(out[49], "H")
        self.assertEqual("".join(out[100:]), "A" * 140)

    def test_insert_respects_the_minimum_spell_guard(self):
        base = self.base()
        base[60:] = "A"
        ev = events([(50, 1, 1.0)])
        without = sfe.insert(base, np.ones(240, np.int8), ev,
                             params=dict(merge_s=0.0, insert_w=4, insert_min_s=0.0))
        self.assertEqual(without[50], "A")
        with_guard = sfe.insert(base, np.ones(240, np.int8), ev,
                                params=dict(merge_s=0.0, insert_w=4, insert_min_s=1.0))
        self.assertEqual("".join(with_guard), "".join(base))

    def test_insert_ignores_low_confidence(self):
        base = self.base()
        ev = events([(50, 1, 0.2)])
        out = sfe.insert(base, np.ones(240, np.int8), ev,
                         params=dict(merge_s=0.0, min_conf=0.5))
        self.assertEqual("".join(out), "".join(base))


if __name__ == "__main__":
    unittest.main()
