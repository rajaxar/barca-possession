"""Half kickoffs, attack direction and the forced-D mask, with consistency checks against tags."""
import os
import sys
import unittest

import numpy as np

from possession import config, data, structure


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class Structure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = data.load_match()
        cls.s = structure.structure(cls.m)

    def test_halves_match_organiser(self):
        sys.path.insert(0, config.repo_dir())
        sys.path.insert(0, os.path.join(config.repo_dir(), "starter"))
        import games
        self.assertEqual(tuple(games.half_video_blocks("BAR-ATH")), structure.HALF_BLOCKS)

    def test_kickoffs_and_forced_dead(self):
        self.assertEqual(self.s.kickoffs, {1: 11185, 2: 113742})
        d = self.s.forced_dead_mask
        self.assertEqual(int(d.sum()), 113742 - 110808)
        self.assertTrue(d[110808] and d[113741] and not d[113742] and not d[11185])
        self.assertTrue(self.m.graded[d].all())

    def test_attack_direction(self):
        ad = self.s.attack_dir
        self.assertTrue((ad[:, 0] == -ad[:, 1]).all())  # teams attack opposite ways
        self.assertTrue((ad[0] == -ad[1]).all())  # and swap at half time
        self.assertTrue((ad == structure.centroid_direction(self.m.P, self.m.T, self.m.gk)).all())

    def test_kickoffs_agree_with_tags(self):
        """Check only: each published kickoff lies inside the Smart Tagging 'Kick Off (Start)' window."""
        tags = data.load_tags()
        ko = tags[tags.type == "Kick Off (Start)"].set_index("half")
        for h, f in structure.KICKOFFS.items():
            self.assertTrue(ko.loc[h, "start_frame"] <= f < ko.loc[h, "end_frame"])

    def test_kickoff_formation(self):
        """Check only: at the published kickoff players sit on their own side and the ball is near the centre spot."""
        for h, f in structure.KICKOFFS.items():
            r = structure.kickoff_formation(self.m, f)
            self.assertGreater(r["own_half_frac"], 0.85)
            self.assertLess(r["ball_to_centre_m"], 10)
            self.assertGreaterEqual(r["n_tracked"], 20)


if __name__ == "__main__":
    unittest.main()
