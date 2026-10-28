"""Degradation simulator: deterministic, ATD-shaped output, and no label leak through the ball (S06).

The external part is the leak-removal assertion of card S06 (metrics of scratch/S05/leaks.py): on
every degraded IDSSE/Metrica game the ball's frozen share and NaN share are the same in D and alive
frames, the IDSSE teleport at D end is gone, and single ball features stay far from separating D.
"""
import os
import unittest

import numpy as np

from possession import config, data, degrade, external

GAMES = external.list_games()


def synthetic_match(n=3000):
    rng = np.random.default_rng(1)
    half = np.repeat(np.array([1, 2], np.int8), n // 2)
    T = np.repeat(np.array([0, 1], np.int8), 12)
    P = (rng.uniform([0, 0], [105, 68], (1, 24, 2)) + np.cumsum(rng.normal(0, 0.05, (n, 24, 2)), 0))
    B = (np.array([52.5, 34.0]) + np.cumsum(rng.normal(0, 0.2, (n, 2)), 0)).astype(np.float32)
    dead = np.zeros(n, bool)
    dead[1000:1400] = True
    B[1000:1400] = B[999]  # IDSSE-style frozen dead ball, then a teleport on the restart frame
    B[1400:] += 20.0
    gk = np.array([[0, 12], [0, 12]])
    m = data.Match(P=P.astype(np.float32), V=np.ones((n, 24), np.float32), T=T, B=B,
                   BS=np.full(n, np.nan, np.float32), gk=gk, graded=np.ones(n, bool), half=half)
    return m, dead


class Synthetic(unittest.TestCase):
    def test_deterministic_and_layout(self):
        m, dead = synthetic_match()
        a = degrade.degrade(m, seed=3, dead=dead)
        b = degrade.degrade(m, seed=3, dead=dead)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        k = degrade.PARAMS["slots_per_team"]
        self.assertEqual(a.P.shape, (len(m.B), 2 * k, 2))
        self.assertEqual(list(a.T), [0] * k + [1] * k)
        np.testing.assert_array_equal(a.half, m.half)
        self.assertFalse(np.array_equal(degrade.degrade(m, seed=4, dead=dead).B, a.B, equal_nan=True))

    def test_scrub_removes_freeze_and_teleport(self):
        m, dead = synthetic_match()
        B = degrade.scrub_ball(m.B, m.P, m.half, dead, np.random.default_rng(0))
        step = np.linalg.norm(np.diff(B, axis=0), axis=1)
        self.assertFalse(np.isnan(B).any())
        self.assertTrue((step[1000:1400] > 0).all())        # never frozen inside D
        self.assertLess(step[999:1400].max() * 25, 40.0)    # no teleport into or out of D
        np.testing.assert_array_equal(B[:1000], m.B[:1000])  # alive frames untouched

    def test_level_zero_keeps_ball_present(self):
        m, dead = synthetic_match()
        out = degrade.degrade(m, level=0.0, dead=dead)
        self.assertFalse(np.isnan(out.B).any())
        self.assertTrue(np.isnan(out.BS).all())


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class FittedTargets(unittest.TestCase):
    def test_constants_reproduce_from_our_atd(self):
        t = degrade.fit_targets(data.load_match())
        np.testing.assert_allclose(degrade.ATD_GAP_Q, t["gap_q"], atol=0.5)
        np.testing.assert_allclose(degrade.ATD_PRESENT_Q, t["present_q"], atol=0.5)
        np.testing.assert_allclose(degrade.ATD_SEG_Q, t["seg_q"], atol=0.5)
        np.testing.assert_allclose(degrade.PARAMS["bs_run_q"], t["bs_run_q"], atol=0.5)
        self.assertAlmostEqual(degrade.PARAMS["bs_rate"], t["bs_rate"], delta=1e-6)


@unittest.skipIf(len(GAMES) < 3, "external dev games not on disk")
class ExternalLeaks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for game_id in GAMES:
            game = degrade.degrade_game(external.load_game(game_id), seed=0)
            cls.results[game_id] = degrade.leak_stats(game.match, game.ref, game.runs, n_null=0)

    def test_frozen_and_nan_share_equal_in_dead_and_alive(self):
        diffs = []
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                frozen = r["ball_frozen"]
                self.assertLess(abs(frozen["d_mean"] - frozen["alive_mean"]), 0.01)
                nan = r["ball_nan"]
                diffs.append(nan["d_mean"] - nan["alive_mean"])
                # one game: ~10 s blocks make the per-game spread ~0.05 (sd over seeds and games)
                self.assertLess(abs(diffs[-1]), 0.12)
        self.assertLess(abs(np.mean(diffs)), 0.03)  # no systematic P(NaN | D) shift

    def test_no_teleport_at_dead_end(self):
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                self.assertLess(r["teleport"]["d_end_fast"], 0.05)  # clean IDSSE: 0.87-0.93

    def test_single_ball_features_do_not_separate_dead(self):
        # AUC of D vs alive. Clean data: ball speed 0.00-0.6 (IDSSE ~0.003), nearest player up to
        # 0.81, ball NaN up to 0.98 (Metrica). Players alone (mean speed) reach 0.06-0.17.
        for game_id, r in self.results.items():
            with self.subTest(game=game_id):
                for name in ("ball_nan", "ball_frozen", "bs_present"):
                    self.assertLess(abs(r[name]["auc"] - 0.5), 0.07, name)
                for name in ("ball_speed", "nearest_player_m"):
                    self.assertLess(abs(r[name]["auc"] - 0.5), 0.22, name)
                    self.assertLess(abs(r[name]["auc"] - 0.5),
                                    abs(r["player_speed_mean"]["auc"] - 0.5), name)


if __name__ == "__main__":
    unittest.main()
