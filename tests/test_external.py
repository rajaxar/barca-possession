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


if __name__ == "__main__":
    unittest.main()
