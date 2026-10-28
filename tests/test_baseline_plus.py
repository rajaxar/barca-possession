import unittest

import numpy as np

from possession.baseline_plus import (_dead_mask, _labels_half, _merge_short,
                                      predict_arrays)


class BaselinePlus(unittest.TestCase):
    def test_hysteresis_persists_over_missing_and_backdates(self):
        ev = np.array([0, 0, 1, 1, -1, 1, 0, 0, 0], dtype=np.int8)
        got = _labels_half(ev, np.zeros(len(ev), bool), k=3)
        np.testing.assert_array_equal(got, [0, 0, 1, 1, 1, 1, 0, 0, 0])

    def test_restart_does_not_invent_a_turnover(self):
        ev = np.array([0, 0, -1, -1, -1], dtype=np.int8)
        dead = np.array([False, False, True, True, False])
        got = _labels_half(ev, dead, k=2)
        self.assertEqual(got[-1], 0)

    def test_dead_interval_ends_at_sustained_in_pitch_movement(self):
        n = 20
        b = np.tile(np.array([10.0, 10.0]), (n, 1))
        b[5:11, 0] = -1.0
        b[15:, 0] = 10.2 + 0.2 * np.arange(n - 15)
        params = dict(margin=0.5, min_out=6, v_restart=3.0, restart_len=5)
        got = _dead_mask(b, np.arange(n), np.zeros(n, bool), params, (105.3, 68.0))
        want = np.zeros(n, bool)
        want[5:15] = True
        np.testing.assert_array_equal(got, want)

    def test_no_observed_restart_keeps_dead_until_half_end(self):
        n = 12
        b = np.tile(np.array([10.0, 10.0]), (n, 1))
        b[3:9, 0] = -1.0
        params = dict(margin=0.5, min_out=6, v_restart=3.0, restart_len=5)
        got = _dead_mask(b, np.arange(n), np.zeros(n, bool), params, (105.3, 68.0))
        np.testing.assert_array_equal(got, np.arange(n) >= 3)

    def test_short_spells_merge_into_team_or_dead(self):
        np.testing.assert_array_equal(
            _merge_short(np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int8), 3),
            np.zeros(9, dtype=np.int8))
        np.testing.assert_array_equal(
            _merge_short(np.array([2, 2, 0, 0, 2, 2], dtype=np.int8), 3),
            np.full(6, 2, dtype=np.int8))

    def test_predict_masks_ungraded_and_published_forced_dead(self):
        n = 12
        half = np.zeros(n, dtype=np.int8)
        half[2:10] = 1
        forced = np.zeros(n, bool)
        forced[2:4] = True
        p = np.full((n, 2, 2), 50.0)
        p[:, 0] = 0.0
        b = np.tile(np.array([1.0, 1.0]), (n, 1))
        got = predict_arrays(p, np.array([0, 1], dtype=np.int8), b, half, forced, {"m": 0})
        np.testing.assert_array_equal(got, list("XXDDHHHHHHXX"))


if __name__ == "__main__":
    unittest.main()
