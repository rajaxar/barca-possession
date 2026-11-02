"""Synthetic tests for the E09 decoders (possession/H, A, D): spell count, forced published block,
25 Hz boundary refinement, determinism and full-match runtime. No match data is read."""
import time
import unittest

import numpy as np

from possession import hsmm, structure

DUR = hsmm.logdur_gamma([(0.743, 13.06), (0.743, 13.06), (1.015, 20.19)], hsmm.DMAX)
# H/A and D transitions from the external dev games (row-normalized, diagonal unused).
LT = np.log(np.array([[1e-12, 0.70, 0.30], [0.70, 1e-12, 0.30], [0.50, 0.50, 1e-12]]))


def synth(segs, sd=1.0, seed=0):
    """Per-frame H/A/D evidence with `segs` = [(state, n_frames), ...]."""
    rng = np.random.RandomState(seed)
    z = np.concatenate([np.full(d, s) for s, d in segs])
    E = rng.randn(len(z), hsmm.S) * sd
    E[np.arange(len(z)), z] += 2.0
    return z, E


def decode(E, half=None, **kw):
    half = np.ones(len(E), np.int8) if half is None else half
    return hsmm.decode_frames(E, half, DUR, LT, beta=1.0, kappa=0.0, **kw)


class Decoders(unittest.TestCase):
    def test_spell_count_and_boundaries(self):
        segs = [(0, 250), (1, 300), (2, 200), (0, 250), (1, 300)]
        z, E = synth(segs, sd=0.6)
        p = decode(E)
        runs = np.flatnonzero(np.diff(p)) + 1
        truth = np.cumsum([d for _, d in segs])[:-1]
        self.assertEqual(len(runs), len(truth))
        self.assertLessEqual(np.abs(runs - truth).max(), 10)       # <= 2 bins (0.4 s)

    def test_boundary_refinement_at_25hz(self):
        # The true change is at frame 1002, deliberately not a multiple of the 5-frame bin.
        n = 2005
        rng = np.random.RandomState(1)
        E = rng.randn(n, hsmm.S) * 0.3
        E[:1002, 0] += 1.5
        E[1002:, 1] += 1.5
        coarse = decode(E, refine_frames=0)
        fine = decode(E, refine_frames=12)
        b0 = np.flatnonzero(np.diff(coarse)) + 1
        b1 = np.flatnonzero(np.diff(fine)) + 1
        self.assertEqual(len(b1), 1)
        self.assertLessEqual(abs(int(b1[0]) - 1002), 1)             # refined to 25 Hz
        self.assertEqual(len(b0), 1)
        self.assertGreaterEqual(abs(int(b0[0]) - 1002), 2)          # bin decode is quantised

    def test_forced_published_2h_pre_kickoff_block(self):
        # Structure only (GUARDRAILS 1.0, D-007): the real half blocks and published kickoff frame.
        n = structure.HALF_BLOCKS[1][2] + 1                         # 190198 video frames
        half = np.zeros(n, np.int8)
        for h, a, z in structure.HALF_BLOCKS:
            half[a:z + 1] = h
        k2 = structure.KICKOFFS[2]
        dead = np.zeros(n, bool)
        dead[structure.HALF_BLOCKS[1][1]:k2] = True                 # 110808..113741, all D
        dead[:5] = True                                             # not graded: must stay X
        rng = np.random.RandomState(2)
        E = rng.randn(n, hsmm.S) * 0.05
        E[:, 0] += 1.0                                              # always prefers H unless forced
        p = decode(E, half, forced_dead=dead, dmax=hsmm.DMAX, cont_penalty=-5.0)
        a1, z1 = structure.HALF_BLOCKS[0][1], structure.HALF_BLOCKS[0][2]
        a2, z2 = structure.HALF_BLOCKS[1][1], structure.HALF_BLOCKS[1][2]
        self.assertEqual(int(dead.sum()) - 5, k2 - a2)              # 2934 published dead frames
        self.assertTrue((p[a2:k2] == 2).all())                      # forced D
        self.assertTrue((p[:a1] == 3).all())                        # ungraded -> X
        self.assertTrue((p[z1 + 1:a2] == 3).all())                  # half-time hole -> X
        self.assertTrue((p[k2:z2 + 1] == 0).all())                  # decoded, not forced
        self.assertEqual(len(p), n)

    def test_deterministic_and_x_outside(self):
        z, E = synth([(0, 100), (1, 100), (2, 60)], sd=0.5)
        half = np.ones(len(E), np.int8)
        half[:3] = half[-3:] = 0
        a = decode(E, half)
        b = decode(E, half)
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a[:3] == 3).all() and (a[-3:] == 3).all())
        self.assertTrue(np.isin(a[3:-3], (0, 1, 2)).all())

    def test_full_match_runtime(self):
        # 158,857 graded frames = the match's two graded halves; target is < 30 s (E09 done-criterion).
        _, E = synth([(0, 79467), (2, 4000), (1, 75390)], sd=0.5)
        self.assertEqual(len(E), 158857)
        t0 = time.perf_counter()
        p = decode(E, dmax=hsmm.DMAX, cont_penalty=-5.0)
        dt = time.perf_counter() - t0
        self.assertEqual(len(p), len(E))
        self.assertLess(dt, 30.0, "full-match HSMM decode took %.1f s" % dt)


if __name__ == "__main__":
    unittest.main()
