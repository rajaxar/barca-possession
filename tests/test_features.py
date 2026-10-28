"""Synthetic tests for the shared permutation-invariant feature library."""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from possession import features as feature_lib


def synthetic_match():
    n = 120
    half = np.repeat([1, 2], n // 2).astype(np.int8)
    p = np.zeros((n, 4, 2), dtype=np.float32)
    p[:, :, 1] = [10, 30, 40, 60]
    p[:60, :, 0] = [45, 5, 60, 100]
    p[60:, :, 0] = [45, 100, 60, 5]
    p += np.arange(n, dtype=np.float32)[:, None, None] * 0.001
    b = np.tile(np.array([50, 34], dtype=np.float32), (n, 1))
    b[10:15] = np.nan
    return SimpleNamespace(
        P=p,
        V=np.ones((n, 4), dtype=np.float32),
        T=np.array([0, 0, 1, 1], dtype=np.int8),
        B=b,
        BS=np.where(np.arange(n) % 2, np.nan, 1.0),
        gk=np.array([[1, 3], [1, 3]]),
        graded=np.ones(n, dtype=bool),
        half=half,
    )


class Features(unittest.TestCase):
    def test_shape_missing_ball_and_attack_orientation(self):
        match = synthetic_match()
        result = feature_lib.features(match)
        self.assertEqual(result.shape, (len(match.B), 64))
        self.assertEqual(tuple(result.columns), feature_lib.FEATURE_COLUMNS)
        ball_columns = [c for c in result if "ball_near" in c or "ball_count" in c]
        ball_columns += ["ball_nearest_team", "home_ball_time_to_ball_s",
                         "away_ball_time_to_ball_s", "ball_speed_mps",
                         "ball_accel_mps2", "ball_direction_change_deg", "ball_out_of_pitch"]
        self.assertTrue(result.loc[10:14, ball_columns].isna().all().all())
        self.assertTrue((result.loc[10:14, "ball_present"] == 0).all())
        self.assertTrue(np.isfinite(result.loc[10:14, "home_centroid_y_m"]).all())
        self.assertLess(result.loc[0, "home_gk_attack_x_m"], 0)
        self.assertLess(result.loc[0, "home_centroid_attack_x_m"], 0)
        self.assertLess(result.loc[60, "home_centroid_attack_x_m"], 0)
        self.assertGreater(result.loc[60, "home_centroid_attack_x_m"],
                           result.loc[0, "home_centroid_attack_x_m"])

    def test_player_permutation_does_not_change_features(self):
        match = synthetic_match()
        original = feature_lib.features(match)
        permutation = np.array([1, 0, 3, 2])
        shuffled = SimpleNamespace(**match.__dict__)
        shuffled.P = match.P[:, permutation]
        shuffled.V = match.V[:, permutation]
        shuffled.T = match.T[permutation]
        shuffled.gk = np.argsort(permutation)[match.gk]
        changed = feature_lib.features(shuffled)
        np.testing.assert_allclose(original.to_numpy(), changed.to_numpy(), atol=2e-5, equal_nan=True)

    def test_tags_distinguish_unavailable_from_zero_coverage(self):
        match = synthetic_match()
        unavailable = feature_lib.features(match)
        self.assertTrue(unavailable["set_piece_window"].isna().all())
        tags = pd.DataFrame([
            dict(code="BUILD UP", start_frame=0, end_frame=10, attacker_team=0),
            dict(code="SET PIECES", start_frame=5, end_frame=15, attacker_team=None),
        ])
        result = feature_lib.features(match, tags=tags)
        self.assertEqual(result.loc[0, "attacker_tag_coverage_home"], 1)
        self.assertEqual(result.loc[0, "attacker_tag_coverage_away"], 0)
        self.assertEqual(result.loc[4, "set_piece_window"], 0)
        self.assertEqual(result.loc[5, "set_piece_window"], 1)
        empty = feature_lib.features(match, tags=tags.iloc[:0])
        self.assertTrue((empty["set_piece_window"] == 0).all())

    def test_cache_round_trip(self):
        match = synthetic_match()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"MTAH_CACHE_DIR": directory}):
                first = feature_lib.features(match, cache_key="synthetic")
                path = feature_lib._cache_path(
                    "synthetic", feature_lib._input_digest(match, 25.0, (105.3, 68.0), None, None))
                self.assertTrue(os.path.isfile(path))
                second = feature_lib.features(match, cache_key="synthetic")
                pd.testing.assert_frame_equal(first, second)

    def test_cache_key_covers_fps_pitch_and_input(self):
        match = synthetic_match()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"MTAH_CACHE_DIR": directory}):
                feature_lib.features(match, cache_key="k")
                feature_lib.features(match, cache_key="k", fps=10.0)
                feature_lib.features(match, cache_key="k", pitch=(105.0, 68.0))
                moved = SimpleNamespace(**match.__dict__)
                moved.P = match.P + 1.0
                feature_lib.features(moved, cache_key="k")
                self.assertEqual(len(os.listdir(directory)), 4)

    def test_attack_features_are_nan_off_half(self):
        match = synthetic_match()
        match.half = match.half.copy()
        match.half[:10] = 0
        result = feature_lib.features(match)
        self.assertTrue(result.loc[:9, ["home_centroid_attack_x_m", "away_centroid_attack_x_m"]].isna().all().all())
        self.assertTrue(result.loc[10:, "home_centroid_attack_x_m"].notna().all())

    def test_tracking_jumps_are_masked_not_propagated(self):
        match = synthetic_match()
        match.B = match.B.copy()
        match.B[30] = [90.0, 5.0]  # one-frame teleport: implied speed >> 50 m/s
        match.P = match.P.copy()
        match.P[40, 0] += [30.0, 0.0]  # one-frame player jump: > 12 m/s
        result = feature_lib.features(match)
        self.assertTrue(np.nanmax(result["ball_speed_mps"]) <= feature_lib.BALL_VMAX_MPS)
        self.assertTrue(np.nanmax(result["home_ball_near1_rel_speed_mps"]) <= feature_lib.BALL_VMAX_MPS
                        + feature_lib.PLAYER_VMAX_MPS)

    def test_gk_slot_must_belong_to_team(self):
        match = synthetic_match()
        match.gk = np.array([[3, 3], [1, 3]])
        with self.assertRaises(ValueError):
            feature_lib.features(match)

    def test_camera_unprojects_image_centre(self):
        matrix = np.diag([1920.0, 1080.0, 1.0])
        columns = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
        table = pd.DataFrame({
            "frame": [0, 1], "valid": [1, 0],
            **{name: [matrix.ravel()[j], np.nan] for j, name in enumerate(columns)},
        })
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "homography.csv")
            table.to_csv(path, index=False)
            result = feature_lib.camera_look_at_x(path, n_frames=2)
        np.testing.assert_allclose(result, [105.3 / 2, np.nan], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
