"""Synthetic tests for the grown present-ball D add-on (E66)."""
import unittest
from types import SimpleNamespace

import numpy as np

from possession import growd

N = 3000


def pre(n=N, spd=0.5, vball=0.0):
    """Ball at rest at (50, 30) all along, players slow, no keeper nearby."""
    B = np.tile([50.0, 30.0], (n, 1))
    return dict(B=B, spd=np.full(n, spd, np.float32), vball=np.full(n, vball, np.float32),
                kd=np.full(n, np.inf, np.float32), graded=np.ones(n, bool))


def anchored(p, a=1000, b=1100):
    """Make only [a, b) rest-like: ball moving and players fast elsewhere."""
    p["vball"][:] = 8.0
    p["spd"][:] = 2.0
    p["vball"][a:b] = 0.0
    p["spd"][a:b] = 0.5
    return p


class Anchor(unittest.TestCase):
    def test_rest_with_slow_players_is_anchor(self):
        p = anchored(pre())
        anc = growd.anchor(p)
        self.assertTrue(anc[1000:1100].all() and anc.sum() == 100)

    def test_short_fast_or_keeper_rest_is_not(self):
        p = anchored(pre(), 1000, 1020)                       # 0.8 s < 1 s hold
        self.assertFalse(growd.anchor(p).any())
        p = anchored(pre()); p["spd"][1000:1100] = 1.2        # team >= 1 m/s
        self.assertFalse(growd.anchor(p).any())
        p = anchored(pre()); p["kd"][1000:1100] = 1.0         # keeper within 2 m
        self.assertFalse(growd.anchor(p).any())


class Grow(unittest.TestCase):
    def setUp(self):
        self.p = pre()
        self.p["vball"][:] = 0.0                              # slow everywhere: growth is stopped only by the rules
        self.anc = np.zeros(N, bool)
        self.anc[1000:1100] = True

    def test_grows_both_sides_until_cap(self):
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=10)
        self.assertEqual(np.flatnonzero(out)[[0, -1]].tolist(), [1000 - 250, 1100 + 250 - 1])

    def test_ball_leaves_radius(self):
        self.p["B"][1300:, 0] = 50.0 + 30.0                   # 30 m away from the rest point
        self.assertEqual(growd.grow(self.p, self.anc, r=25.0, cap_s=30)[1100:].sum(), 200)
        r = growd.grow(self.p, self.anc, r=35.0, cap_s=30)
        self.assertGreater(r[1100:].sum(), 200)               # larger R keeps growing

    def test_team_sprint_stops(self):
        self.p["spd"][1400:] = 3.5
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=30)
        self.assertTrue(out[1399] and not out[1400])

    def test_both_move_stops_but_slow_ball_does_not(self):
        self.p["spd"][1300:] = 2.0                            # team >= 1.5 m/s
        self.p["vball"][1300:] = 8.0                          # and ball running
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=30)
        self.assertTrue(out[1299] and not out[1300])
        self.p["vball"][1300:] = 1.0                          # ball still: only the 1 s hysteresis applies
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=30)
        self.assertTrue(out[1299] and not out[1300])

    def test_ball_missing_is_ok(self):
        self.p["B"][1200:1500] = np.nan
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=30)
        self.assertTrue(out[1200:1500].all())

    def test_ungraded_stops(self):
        self.p["graded"][1300:] = False
        out = growd.grow(self.p, self.anc, r=25.0, cap_s=30)
        self.assertFalse(out[1300:].any())


def fake_match(n=N):
    P = np.full((n, 4, 2), [52.0, 34.0])
    V = np.full((n, 4), 0.5)
    B = np.tile([50.0, 30.0], (n, 1))
    return SimpleNamespace(P=P, V=V, T=np.array([0, 0, 1, 1]), B=B, gk=np.array([[0, 2], [0, 2]]),
                           half=np.ones(n, np.int8))


class Apply(unittest.TestCase):
    def test_disabled_returns_input(self):
        lab = np.array(["H"] * N)
        out, fp = growd.apply(lab, fake_match(), enabled=False)
        self.assertIs(out, lab)
        self.assertIsNone(fp)

    def test_enabled_runs_guard(self):
        lab = np.array(["H"] * N)
        with self.assertRaises(RuntimeError):                 # everything rests: D share added is 100% > bound
            growd.apply(lab, fake_match(), enabled=True)

    def test_flag_default_and_fallback_radius(self):
        self.assertEqual(growd.R_M, 25.0)
        self.assertEqual(growd.CAP_S, 30)


class Guard(unittest.TestCase):
    def fp(self, **kw):
        base = dict(d_added_share_graded=0.02, flickers_added=0, changed_in_dead=0)
        return dict(base, **kw)

    def test_absolute_runaway_cap(self):
        self.assertTrue(growd.check_footprint(self.fp()))
        for bad in (dict(d_added_share_graded=0.25), dict(d_added_share_graded=-0.01),
                    dict(flickers_added=growd.MAX_FLICKER_ADDED + 1), dict(changed_in_dead=3)):
            with self.assertRaises(RuntimeError):
                growd.check_footprint(self.fp(**bad))

    def test_relative_guard_on_own_mask(self):
        """The canary is growd's own footprint, so upstream D changes cannot trip it."""
        base = dict(d_added_share_graded=0.02, flickers_added=0, changed_in_dead=0,
                    mask_graded=1200, mask_share_graded=0.0755, eligible=450, changed_over_eligible=1.016)
        self.assertTrue(growd.check_footprint(base))
        for bad in (dict(mask_share_graded=growd.MAX_MASK_SHARE + 0.01),
                    dict(mask_share_graded=growd.MIN_MASK_SHARE / 2),
                    dict(changed_over_eligible=growd.MAX_CHANGED_PER_ELIGIBLE + 0.01)):
            with self.assertRaises(RuntimeError):
                growd.check_footprint(dict(base, **bad))

    def test_footprint_counts_added_d_flickers_and_dead(self):
        before = np.array(["H"] * 100 + ["A"] * 100)
        after = before.copy()
        after[40:50] = "D"                                   # splits H into 40 + 50 frames: no flicker (>= 1 s? 40 > 25)
        after[150:160] = "D"                                  # A: 50 + 40
        half = np.ones(200, np.int8)
        dead = np.zeros(200, bool)
        dead[45] = True
        fp = growd.footprint(before, after, half, dead)
        self.assertEqual(fp["changed"], 20)
        self.assertAlmostEqual(fp["d_added_share_graded"], 0.1)
        self.assertEqual(fp["flickers_added"], 0)
        self.assertEqual(fp["changed_in_dead"], 1)
        mask = np.zeros(200, bool)
        mask[40:50] = True
        fp = growd.footprint(before, after, half, dead, mask=mask)
        self.assertEqual(fp["mask_graded"], 10)
        self.assertEqual(fp["eligible"], 10)              # input is H there
        self.assertAlmostEqual(fp["changed_over_eligible"], 2.0)   # mask + the second D run
        after[10:20] = "D"; after[20:30] = "A"
        after[30:35] = "H"                                    # a 5-frame H spell inside: new flicker
        self.assertGreaterEqual(growd.footprint(before, after, half)["flickers_added"], 1)


if __name__ == "__main__":
    unittest.main()
