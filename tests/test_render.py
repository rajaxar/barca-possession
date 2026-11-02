"""Clip renderer: annotation helpers, label loading, and one end-to-end MP4 render."""
import os
import shutil
import tempfile
import unittest

import numpy as np

from possession import config, data, render


def _fake_match(n=40, slots=4):
    """Minimal Match-shaped object for the label-free helpers (no file access)."""
    P = np.zeros((n, slots, 2), np.float32)
    P[..., 0] = 20.0 + np.arange(slots)[None, :]
    P[..., 1] = 30.0
    B = np.full((n, 2), np.nan, np.float32)
    B[10:25] = (40.0, 34.0)
    B[25:] = (50.0, 20.0)
    half = np.ones(n, np.int8)
    return data.Match(P=P, V=np.zeros((n, slots), np.float32),
                      T=np.zeros(slots, np.int8), B=B, BS=np.zeros(n, np.float32),
                      gk=np.zeros((2, 1), np.int64), graded=np.ones(n, bool), half=half)


class Labels(unittest.TestCase):
    def test_numeric_codes_and_confidence(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "labels.csv")
            with open(path, "w") as fh:
                fh.write("frame,label,conf\n0,2,0.9\n1,0,0.8\n3,1,0.7\n")
            labels, confidence = render.load_labels(path)
        self.assertEqual(labels.tolist(), ["D", "H", "", "A"])
        np.testing.assert_allclose(confidence[:4], [0.9, 0.8, np.nan, 0.7], equal_nan=True)

    def test_probability_columns(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "labels.csv")
            with open(path, "w") as fh:
                fh.write("frame,label,H,A,D,X\n0,D,0.1,0.2,0.6,0.1\n")
            labels, confidence = render.load_labels(path)
        self.assertEqual(labels.tolist(), ["D"])
        np.testing.assert_allclose(confidence[0], [0.1, 0.2, 0.6, 0.1])

    def test_codes_and_strings(self):
        np.testing.assert_array_equal(render.as_labels(np.array([0, 1, 2, 3])), list("HADX"))
        np.testing.assert_array_equal(render.as_labels(["h", "a"]), ["H", "A"])


class Bands(unittest.TestCase):
    def test_d_band_and_boundaries(self):
        labels = np.array(list("HHHDDAA"), dtype="<U1")
        bands = render.shade_d(labels, 0, 7, min_frames=1)
        self.assertEqual([(b["start"], b["end"]) for b in bands], [(3, 5)])
        edges = [(b["frame"], b["label"]) for b in render.spell_boundaries(labels, 0, 7)]
        self.assertEqual(edges, [(3, "D"), (5, "A")])
        self.assertEqual([e["kind"] for e in render.d_events(labels, 0, 7)], ["onset", "end"])

    def test_missing_ball_band(self):
        match = _fake_match()
        bands = render.shade_missing_ball(match, 0, 40, min_frames=5)
        self.assertEqual([(b["start"], b["end"]) for b in bands], [(0, 10)])

    def test_crop_is_landscape_and_padded(self):
        match = _fake_match()
        x0, x1, y0, y1 = render._crop_box(match, 0, 40, 8.0, (105.3, 68.0))
        self.assertGreaterEqual((x1 - x0) / (y1 - y0), 1.29)
        self.assertLessEqual((x1 - x0) / (y1 - y0), 1.86)
        self.assertLess(x0, 20.0)
        self.assertGreater(x1, 50.0)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class RenderClip(unittest.TestCase):
    def test_two_second_mp4(self):
        from possession import structure

        match = data.load_match()
        start = structure.KICKOFFS[1] + 500
        with tempfile.TemporaryDirectory() as td:
            res = render.render_clip(match, start, start + 50, out=td, name="test_clip",
                                     mp4=True, gif=False, fps=8, size=320, crop=10.0)
            self.assertTrue(os.path.isfile(res["mp4"]))
            self.assertGreater(os.path.getsize(res["mp4"]), 0)
            self.assertGreaterEqual(res["frames"], 15)


if __name__ == "__main__":
    unittest.main()
