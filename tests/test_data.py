"""Shapes, alignment against the organizer's loaders, and Smart Tagging flips."""
import os
import sys
import time
import unittest

import numpy as np

from possession import config, data, structure


def _starter():
    sys.path.insert(0, config.repo_dir())
    sys.path.insert(0, os.path.join(config.repo_dir(), "starter"))
    import games  # noqa: F401
    return sys.modules["games"]


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class MatchData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = data.load_match()

    def test_shapes(self):
        m, n = self.m, config.N_VIDEO_FRAMES
        self.assertEqual(m.P.shape, (n, 46, 2))
        self.assertEqual(m.V.shape, (n, 46))
        self.assertEqual(m.B.shape, (n, 2))
        self.assertEqual(m.BS.shape, (n,))
        self.assertEqual(m.T.shape, (46,))
        self.assertEqual(int(m.graded.sum()), 158857)
        self.assertEqual(set(np.unique(m.half)), {0, 1, 2})

    def test_halves_match_organizer(self):
        games = _starter()
        self.assertEqual(tuple(games.half_video_blocks("BAR-ATH")), structure.HALF_BLOCKS)

    def test_gk_slots(self):
        self.assertTrue((self.m.T[self.m.gk[:, 0]] == 0).all())
        self.assertTrue((self.m.T[self.m.gk[:, 1]] == 1).all())
        self.assertEqual(self.m.gk.tolist(), [[23, 45], [23, 45]])

    def test_alignment_against_starter_loader(self):
        _starter()
        import load_atd  # organizer's loader; its path helpers and frame_id convention
        _, raw = load_atd.atd_paths()
        rng = np.random.default_rng(0)
        frames = sorted(int(f) for f in rng.choice(np.where(self.m.graded)[0], 3, replace=False))
        want = {load_atd.frame_id_of(f): f for f in frames}
        with open(raw) as fh:
            for line in fh:
                fid = int(line.split(":", 1)[0])
                if fid not in want:
                    continue
                f = want.pop(fid)
                players, ball = line.split(":")[1:3]
                first = np.array(players.split(";")[0].split(","), dtype=float)
                np.testing.assert_allclose(first[:2] * [data.PITCH_L, data.PITCH_W],
                                           self.m.P[f, 0], atol=1e-3, equal_nan=True)
                bx = float(ball.split(",")[0])
                np.testing.assert_allclose(bx * data.PITCH_L, self.m.B[f, 0], atol=1e-3, equal_nan=True)
                if not want:
                    break
        self.assertFalse(want)

    def test_ball_coverage_matches_audit(self):
        g = self.m.graded
        self.assertAlmostEqual(float((~np.isnan(self.m.B[g, 0])).mean()), 0.593, places=3)

    def test_warm_load_fast(self):
        t = time.time()
        data.load_match()
        self.assertLess(time.time() - t, 2.0)


@unittest.skipUnless(os.path.isfile(config.asset_path("smart_tagging")), "match data not present")
class Tags(unittest.TestCase):
    def test_attacker_flip(self):
        df = data.load_tags()
        self.assertEqual(len(df), 638)
        d = df[df.code == "DEFENSIVE TRANSITION"]
        self.assertTrue(all(a == 1 - t for a, t in zip(d.attacker_team, d.team)))
        b = df[df.code == "BUILD UP"]
        self.assertTrue(all(a == t for a, t in zip(b.attacker_team, b.team)))
        self.assertEqual(int(df[df.team.isna()].attacker_team.notna().sum()), 0)


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw", "broadcast")), "broadcast data not present")
class Broadcast(unittest.TestCase):
    def test_offsets_found(self):
        offs = data.broadcast_offsets()
        self.assertEqual(sorted(offs), [1, 2])
        for info in offs.values():
            self.assertGreater(info["matched"], 0.3 * info["windows"])


if __name__ == "__main__":
    unittest.main()
