"""Synthetic tests for the ball-free possession classifier (E04)."""
import unittest

import numpy as np
import pandas as pd

from possession import ballfree
from test_features import synthetic_match


def long_match(n=2000):
    """Two-half match whose home team is wide (high spread_y) in the first part of each half."""
    rng = np.random.RandomState(0)
    half = np.repeat([1, 2], n // 2).astype(np.int8)
    p = np.zeros((n, 6, 2), dtype=np.float32)
    # Physical motion (E16 masks implied speeds above 12 m/s): fixed slot spots plus small jitter.
    p[:, :, 0] = rng.uniform(20, 80, (1, 6)) + rng.normal(0, 0.02, (n, 6))
    p[:, :, 1] = np.array([[20, 34, 48, 22, 34, 46]]) + rng.normal(0, 0.02, (n, 6))
    wide = (np.arange(n) % 500) < 250
    # Ramp between the two shapes over 150 frames (<= 11 m/s) instead of jumping.
    ramp = np.convolve(wide.astype(float), np.ones(150) / 150, mode="same")[:, None]
    p[:, :3, 1] = (1 - ramp) * p[:, :3, 1] + ramp * np.array([[2, 34, 66]])
    p[:, 0, 0] = 3
    p[:, 3, 0] = 100
    return type(synthetic_match())(
        P=p, V=np.ones((n, 6), np.float32), T=np.array([0, 0, 0, 1, 1, 1], np.int8),
        B=np.full((n, 2), np.nan, np.float32), BS=np.full(n, np.nan), gk=np.array([[0, 3], [0, 3]]),
        graded=np.ones(n, bool), half=half), wide


class BallFree(unittest.TestCase):
    def test_no_ball_columns(self):
        match, _ = long_match()
        rows, X, Xs = ballfree.sample_table(match)
        self.assertFalse([c for c in X.columns if "ball" in c or "camera" in c or "tag" in c])
        self.assertEqual(list(X.columns), list(Xs.columns))

    def test_swap_negates_differences(self):
        match, _ = long_match()
        rows, X, Xs = ballfree.sample_table(match)
        for c in X.columns:
            if c.startswith("diff_"):
                np.testing.assert_allclose(X[c].to_numpy(), -Xs[c].to_numpy(), atol=1e-4, equal_nan=True)
            if c.startswith("gap_"):
                np.testing.assert_allclose(X[c].to_numpy(), Xs[c].to_numpy(), atol=1e-4, equal_nan=True)

    def test_learns_shape_signal_and_is_antisymmetric(self):
        match, wide = long_match()
        rows, X, Xs = ballfree.sample_table(match)
        y = wide[rows].astype(int)
        for kind in ("logreg", "hgb"):
            model = ballfree.fit([(X, Xs, y)], kind)
            p = ballfree.predict(model, X, Xs)
            self.assertGreater(((p > 0.5) == y).mean(), 0.9)
            np.testing.assert_allclose(p + ballfree.predict(model, Xs, X), 1.0, atol=1e-9)
            frames = ballfree.to_frames(rows, p, len(match.B), match.half)
            self.assertTrue(np.isfinite(frames).all())
            self.assertTrue(((frames >= 0) & (frames <= 1)).all())

    def test_robust_clip_handles_outliers_and_nan(self):
        x = np.r_[np.random.RandomState(0).normal(size=200), 1e9, np.nan][:, None]
        z = ballfree.RobustClip().fit(x).transform(x)
        self.assertTrue(np.isfinite(z).all())
        self.assertLess(z.max(), 10)

    def test_flip_sign_accuracy(self):
        ref = np.array(["H"] * 50 + ["A"] * 50)
        p = np.r_[np.full(50, 0.9), np.full(50, 0.1)]
        self.assertEqual(ballfree.flip_sign_accuracy(ref, p), (1, 1.0))
        self.assertEqual(ballfree.flip_sign_accuracy(ref, 1 - p), (1, 0.0))

    def test_degenerate_columns_are_dropped_and_inert(self):
        """An all-NaN column used to crash the HGB binner; a constant one must not change the fit."""
        match, wide = long_match()
        rows, X, Xs = ballfree.sample_table(match)
        y = wide[rows].astype(int)
        ref = {"logreg": ballfree.fit([(X, Xs, y)], "logreg"),
               "hgb": ballfree.fit([(X, Xs, y)], "hgb")}
        for name, value in (("e22_all_nan", np.nan), ("e22_constant", 1.0)):
            Xb, Xsb = X.copy(), Xs.copy()
            Xb[name] = value
            Xsb[name] = value
            j = list(Xb.columns).index(name)
            for kind in ("logreg", "hgb"):
                model = ballfree.fit([(Xb, Xsb, y)], kind)
                self.assertFalse(bool(model.keep_[j]), (name, kind))
                self.assertNotIn(name, list(model.columns_), (name, kind))
                p = ballfree.predict(model, Xb, Xsb)
                self.assertTrue(np.isfinite(p).all())
                np.testing.assert_allclose(p, ballfree.predict(ref[kind], X, Xs), atol=1e-9)

    def test_only_degenerate_columns_raise(self):
        n = 200
        X = pd.DataFrame({"a": np.full(n, np.nan), "b": np.ones(n)})
        with self.assertRaises(ValueError):
            ballfree.fit([(X, X, np.zeros(n, dtype=int))], "hgb")
        # a model pickled before the guard has no keep_ and still predicts on the full matrix
        m = ballfree.make_model("logreg").fit(np.c_[np.arange(n), np.arange(n) % 2], np.arange(n) % 2)
        self.assertIsNone(getattr(m, "keep_", None))
        z = ballfree._matrix(m, pd.DataFrame(np.zeros((3, 2))))
        self.assertEqual(z.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
