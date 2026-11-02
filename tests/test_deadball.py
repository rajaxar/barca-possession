"""Synthetic tests for the fused dead-ball detector (E07). No match data, no labels, no scores."""
import unittest
from types import SimpleNamespace

import numpy as np

from possession import deadball
from possession.structure import PITCH_L, PITCH_W

N = 3000
A, B_ = 1000, 1100


def match(n=N, half=None):
    """Minimal Match-shaped object: ball present at (50, 30), players at walking speed, 1 half."""
    h = np.ones(n, np.int8) if half is None else np.asarray(half, np.int8)
    B = np.tile([50.0, 30.0], (n, 1)).astype(np.float32)
    P = np.zeros((n, 4, 2), np.float32)
    P[:, :, 0] = np.array([20.0, 20.0, 80.0, 80.0])[None, :]
    P[:, :, 1] = np.array([20.0, 45.0, 20.0, 45.0])[None, :]
    V = np.full((n, 4), 0.5, np.float32)
    return SimpleNamespace(P=P, V=V, T=np.array([0, 0, 1, 1], np.int8), B=B,
                           BS=np.zeros(n, np.float32), gk=np.array([[0, 2]]),
                           graded=h > 0, half=h)


def labels_with(n=N, a=A, b=B_):
    lab = np.full(n, "H", dtype="<U1")
    lab[:] = "H"
    lab[a:b] = "D"
    return lab


class Intervals(unittest.TestCase):
    def test_splits_per_half_and_records_fields(self):
        half = np.ones(N, np.int8)
        half[1500:] = 2
        lab = labels_with()
        lab[1490:1510] = "D"                      # crosses the half boundary: must be two intervals
        out = deadball.interval_list(lab, half)
        spans = [(d["start"], d["end"], d["half"]) for d in out]
        self.assertIn((A, B_, 1), spans)
        self.assertIn((1490, 1500, 1), spans)
        self.assertIn((1500, 1510, 2), spans)
        d = out[0]
        self.assertEqual(d["onset"], A)
        self.assertEqual(d["end_est"], B_)
        self.assertAlmostEqual(d["seconds"], 4.0)

    def test_no_d_no_intervals(self):
        self.assertEqual(deadball.interval_list(np.full(N, "H"), np.ones(N, np.int8)), [])

    def test_structural_flag(self):
        st = SimpleNamespace(forced_dead_mask=np.zeros(N, bool))
        st.forced_dead_mask[A:B_] = True
        out = deadball.interval_list(labels_with(), np.ones(N, np.int8), st=st)
        self.assertTrue(out[0]["structural"])


class OutOfBounds(unittest.TestCase):
    def _gap(self, x):
        m = match()
        m.B[A:B_, 0] = np.nan
        m.B[A - 1, 0] = x
        return m

    def test_outside_pitch_qualifies_and_on_pitch_does_not(self):
        m = self._gap(PITCH_L + 2.0)
        g = deadball.oob_gaps(m.B, m.half)
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0]["outside"] and g[0]["qualifies"])
        self.assertEqual(g[0]["last_frame"], A - 1)
        m2 = self._gap(50.0)
        self.assertFalse(deadball.oob_gaps(m2.B, m2.half)[0]["qualifies"])

    def test_long_gap_is_not_an_out_of_play_gap(self):
        m = self._gap(PITCH_L + 2.0)
        m.B[A:B_ + 800, 0] = np.nan               # a missing run far longer than 30 s
        g = deadball.oob_gaps(m.B, m.half, max_s=30)
        self.assertEqual(len(g), 1)
        self.assertFalse(g[0]["qualifies"])

    def test_override_changes_only_h_a(self):
        m = self._gap(PITCH_L + 2.0)
        lab = np.full(N, "H", dtype="<U1")
        lab[150:200] = "X"
        out, gaps = deadball.oob_override(lab, m.B, m.half)
        self.assertEqual(out[A], "D")
        self.assertEqual(out[150], "X")            # X is never overwritten
        self.assertTrue(any(g["qualifies"] for g in gaps))


class Whistle(unittest.TestCase):
    def test_extends_core_d_over_still_frames_only(self):
        m = match()
        lab = np.full(N, "H", dtype="<U1")
        lab[1020:B_] = "D"
        speed = np.full(N, 0.5, np.float32)
        speed[1025:] = 2.5                          # fast after 5 frames of the interval
        events = np.array([(1005, 1010)], dtype=[("start_frame", "i8"), ("peak_frame", "i8")])
        out, applied = deadball.whistle_extend(lab, m.half, events, speed)
        self.assertEqual(len(applied), 1)
        self.assertEqual(out[1005], "D")            # still frames back to the whistle
        self.assertEqual(out[1025], "D")            # original interval is untouched
        self.assertEqual(applied[0]["new_start"], 1005)

    def test_stops_at_a_fast_frame(self):
        m = match()
        lab = np.full(N, "H", dtype="<U1")
        lab[1020:B_] = "D"
        speed = np.full(N, 0.5, np.float32)
        speed[1010:1015] = 2.5
        events = np.array([(1005, 1010)], dtype=[("start_frame", "i8"), ("peak_frame", "i8")])
        out, applied = deadball.whistle_extend(lab, m.half, events, speed)
        self.assertEqual(applied[0]["new_start"], 1015)   # fast frames are not a D onset

    def test_no_d_interval_means_no_audio_d(self):
        m = match()
        events = np.array([(1040, 1045)], dtype=[("start_frame", "i8"), ("peak_frame", "i8")])
        out, applied = deadball.whistle_extend(np.full(N, "H"), m.half, events, np.full(N, 0.5))
        self.assertEqual(applied, [])
        self.assertTrue((out == "H").all())

    def test_event_array_orders_are_parsed(self):
        ev = np.array([(10, 12), (30, 33)], dtype=[("start_frame", "i8"), ("peak_frame", "i8")])
        np.testing.assert_array_equal(deadball._event_onsets(ev), [10, 30])
        self.assertEqual(deadball._event_onsets(None).size, 0)


class StillnessEnd(unittest.TestCase):
    def test_finds_present_ball_burst_in_the_tail(self):
        m = match()
        lab = labels_with()
        speed = np.full(N, 0.5, np.float32)
        speed[1070:] = 2.5
        ends = deadball.stillness_end(lab, m.B, speed, m.half)
        self.assertEqual(ends[(A, B_)], 1070)

    def test_requires_a_present_ball(self):
        m = match()
        lab = labels_with()
        speed = np.full(N, 2.5, np.float32)
        m.B[1050:B_, 0] = np.nan
        self.assertEqual(deadball.stillness_end(lab, m.B, speed, m.half), {})

    def test_short_interval_not_trimmed(self):
        m = match()
        lab = np.full(N, "H", dtype="<U1")
        lab[1000:1020] = "D"                        # 0.8 s
        speed = np.full(N, 3.0, np.float32)
        self.assertEqual(deadball.stillness_end(lab, m.B, speed, m.half), {})


class KeeperOverride(unittest.TestCase):
    def test_episode_ends_the_interval_at_its_start(self):
        mask = np.zeros(N, bool)
        mask[A:B_] = True
        live = np.zeros(N, bool)
        live[1050] = True
        out, ov = deadball.keeper_override(mask, live)
        self.assertTrue(out[A:1050].all())
        self.assertFalse(out[1050:B_].any())
        self.assertEqual(ov[0]["cut"], 1050)
        self.assertEqual(ov[0]["start"], A)

    def test_no_overlap_no_change(self):
        mask = np.zeros(N, bool)
        mask[A:B_] = True
        out, ov = deadball.keeper_override(mask, np.zeros(N, bool))
        np.testing.assert_array_equal(out, mask)
        self.assertEqual(ov, [])


class Detect(unittest.TestCase):
    def test_flags_off_keep_labels_and_emit_intervals(self):
        m = match()
        lab = labels_with()
        r = deadball.detect(lab, m, oob=False, whistle=False, end_estimate=False, keeper=False)
        np.testing.assert_array_equal(r.labels, lab)
        self.assertEqual(len(r.intervals), 1)
        self.assertEqual(r.intervals[0]["onset"], A)
        self.assertEqual(r.intervals[0]["end_est"], B_)
        self.assertAlmostEqual(r.info["d_share"], 100 / N, places=6)

    def test_oob_flag_changes_label_and_onset_hint(self):
        m = match()
        m.B[A:B_, 0] = np.nan
        m.B[A - 1, 0] = PITCH_L + 3.0
        lab = np.full(N, "H", dtype="<U1")
        r = deadball.detect(lab, m, oob=True, whistle=False, end_estimate=False, keeper=False)
        self.assertEqual(r.labels[A], "D")
        self.assertEqual(r.intervals[0]["start"], A)
        self.assertEqual(r.intervals[0]["onset"], A - 1)   # last tracked ball
        self.assertIn("oob", r.intervals[0]["sources"])

    def test_deterministic(self):
        m = match()
        lab = labels_with()
        a = deadball.detect(lab, m, oob=False, whistle=False, end_estimate=True, keeper=False)
        b = deadball.detect(lab, m, oob=False, whistle=False, end_estimate=True, keeper=False)
        np.testing.assert_array_equal(a.labels, b.labels)
        self.assertEqual(a.intervals, b.intervals)
        self.assertEqual(a.info, b.info)

    def test_apply_returns_labels(self):
        m = match()
        lab = labels_with()
        out = deadball.apply(lab, m, oob=False, whistle=False, end_estimate=False, keeper=False)
        np.testing.assert_array_equal(out, lab)

    def test_end_estimate_can_move_without_label_change(self):
        m = match()
        lab = labels_with()
        m.V[:] = 0.5
        m.V[1070:] = 3.0
        r = deadball.detect(lab, m, oob=False, whistle=False, end_estimate=True, keeper=False)
        self.assertEqual(r.intervals[0]["end_est"], 1070)
        self.assertEqual(r.labels[1070], "D")            # estimate only: labels unchanged

    def test_empty_labels(self):
        m = match()
        r = deadball.detect(np.full(N, "X"), m, oob=False, end_estimate=False)
        self.assertEqual(r.intervals, [])
        self.assertEqual(r.info["d_share"], 0.0)


class Helpers(unittest.TestCase):
    def test_ball_speed_nan_on_missing(self):
        m = match()
        m.B[100:200, 0] = np.nan
        v = deadball.ball_speed(m.B)
        self.assertFalse(np.isfinite(v[100:200]).any())
        self.assertTrue(np.isfinite(v[500]))

    def test_match_structure_is_match_local(self):
        m = match()
        st = deadball.match_structure(m)
        self.assertEqual(st.halves[1], (0, N - 1))
        self.assertEqual(st.halves[2], (0, -1))
        self.assertFalse(st.forced_dead_mask.any())

    def test_set_piece_mask(self):
        tags = SimpleNamespace(code=np.array(["SET PIECES", "ATTACK"]),
                               start_frame=np.array([100, 500]), end_frame=np.array([200, 600]))
        m = deadball.set_piece_mask(tags, N)
        self.assertTrue(m[100:200].all())
        self.assertFalse(m[500:600].any())


if __name__ == "__main__":
    unittest.main()
