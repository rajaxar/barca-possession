"""External dev games load through one API, score with S04, and the documented IDSSE leak is present."""
import unittest

import numpy as np

from possession import evaluate, external

GAMES = external.list_games()


@unittest.skipIf(len(GAMES) < 3, "external dev games not on disk")
class External(unittest.TestCase):
    def test_load_and_oracle_score(self):
        for game_id in (GAMES[0], [g for g in GAMES if g.startswith("metrica:")][0]):
            game = external.load_game(game_id)
            self.assertEqual(game.match.P.shape[0], len(game.ref))
            self.assertEqual(game.match.B.shape[0], len(game.ref))
            self.assertLessEqual(set(np.unique(game.ref)), {"H", "A", "D"})
            self.assertEqual(evaluate.score(game.ref, game.ref, game.runs)["spell_f1_mean"], 1.0)

    def test_idsse_dead_ball_is_frozen(self):
        game = external.load_game(next(g for g in GAMES if g.startswith("idsse:")))
        frozen = np.r_[False, np.abs(np.diff(game.match.B, axis=0)).sum(axis=1) == 0]
        self.assertGreater(frozen[game.ref == "D"].mean(), 0.99)

    def test_speed_units_are_mps(self):
        for game_id in GAMES:
            match = external.load_game(game_id).match
            half = np.asarray(match.half)
            xy = external._speed(np.asarray(match.P), half)
            ok = np.isfinite(xy) & np.isfinite(match.V) & (xy > 0.5)
            self.assertAlmostEqual(np.median(match.V[ok] / xy[ok]), 1.0, delta=0.1, msg=game_id)
            ball = external._speed(np.asarray(match.B)[:, None, :], half)[:, 0]
            ok = np.isfinite(ball) & np.isfinite(match.BS) & (ball > 0.5)
            if ok.any():  # Metrica has no vendor ball speed
                self.assertAlmostEqual(np.median(match.BS[ok] / ball[ok]), 1.0, delta=0.1, msg=game_id)


if __name__ == "__main__":
    unittest.main()
