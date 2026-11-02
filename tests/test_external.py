"""External dev games load through one API, score with S04, and the documented leaks are present.

SkillCorner's derived-D rules and 10 -> 25 Hz resampling are synthetic-tested; the on-disk games
are tested for load/score, gaps and label invariants. Match-data tests skip when the data is absent.
"""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from possession import evaluate, external

GAMES = external.list_games()
SKILL = [g for g in GAMES if g.startswith("skillcorner:")]


def _runs(mask):
    x = np.r_[0, np.asarray(mask, np.int8), 0]
    e = np.flatnonzero(np.diff(x) != 0)
    return e.reshape(-1, 2)


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


@unittest.skipIf(len(SKILL) < 5, "fewer than five SkillCorner games on disk")
class SkillCorner(unittest.TestCase):
    def test_five_games_load_score_and_have_a_d(self):
        """Done criterion: >=5 games load through the API and score through S04."""
        for game_id in SKILL[:5]:
            game = external.load_game(game_id)
            ref = np.asarray(game.ref)
            self.assertEqual(game.match.P.shape[0], len(ref))
            self.assertEqual(game.match.B.shape[0], len(ref))
            self.assertLessEqual(set(np.unique(ref)), {"H", "A", "D", "X"})
            self.assertTrue((ref == "D").any(), game_id)
            self.assertTrue((ref == "H").any() and (ref == "A").any(), game_id)
            self.assertEqual(evaluate.score(ref, ref, game.runs)["spell_f1_mean"], 1.0, game_id)
            # no scored frame is X, and every scored frame has a ball
            self.assertFalse((ref[game.match.graded] == "X").any(), game_id)
            self.assertFalse(np.isnan(game.match.B[game.match.graded]).any(), game_id)

    def test_derived_d_is_not_the_raw_null_group(self):
        """Null-group flight/line cases are relabeled; the raw vendor labels are kept for audit."""
        arrays = external._load_skillcorner(SKILL[0].split(":", 1)[1])
        ref, raw = np.asarray(arrays["ref"]), np.asarray(arrays["ref_raw"])
        graded = ref != "X"
        self.assertGreater((raw[graded] == "D").mean(), (ref[graded] == "D").mean())
        # a D onset moved to a line crossing cannot be earlier than the raw null onset
        self.assertLessEqual((ref == "D").sum(), (raw == "D").sum())

    def test_gaps_are_not_interpolated(self):
        game = external.load_game(SKILL[0])
        ref, P, B = np.asarray(game.ref), game.match.P, game.match.B
        gap = (ref == "X") & np.isnan(P).all(axis=(1, 2))
        self.assertTrue(gap.any())
        self.assertFalse(np.isnan(P[~gap]).all(axis=(1, 2)).any())
        self.assertTrue(np.isnan(B[gap]).all())
        # a scored frame never sits inside an all-NaN tracking block
        self.assertFalse(gap[game.match.graded].any())

    def test_ref_derivation_rules(self):
        n = 80
        players = np.zeros((n, 4, 2), np.float32)
        # two players per team; fast movement in the flight window, walk elsewhere
        players[:, 0, 0] = np.arange(n) * 0.5
        players[:, 1, 0] = np.arange(n) * 0.5 + 5
        players[:, 2, 0] = np.arange(n) * 0.1
        players[:, 3, 0] = np.arange(n) * 0.1 + 5
        B = np.zeros((n, 3), np.float32)
        B[:, 0] = 0.0
        B[:, 1] = 0.0
        B[:, 2] = 0.1
        group = np.ones(n, np.int8)
        group[:5] = 0          # pre-kickoff null -> X
        group[20:30] = 0       # live flight (ball moves fast, players move) -> previous H
        group[40:60] = 0       # dead ball: ball leaves the pitch at frame 45 -> D from 45
        B[20:30, 0] = np.arange(10) * 1.0  # 10 m/s, in pitch
        B[45:60, 0] = 60.0                 # clearly outside the 105 x 68 pitch
        half = np.ones(n, np.int8)
        tracked = np.ones(n, bool)
        ref = external._skillcorner_ref(players, B, group, half, tracked, (105.0, 68.0))
        self.assertTrue(np.all(ref[:5] == "X"))
        self.assertTrue(np.all(ref[5:20] == "H"))
        self.assertTrue(np.all(ref[20:30] == "H"))  # flight is not D
        self.assertTrue(np.all(ref[30:40] == "H"))
        self.assertTrue(np.all(ref[40:45] == "H"))  # pre-crossing frames keep the last owner
        self.assertTrue(np.all(ref[45:60] == "D"))  # D starts at the line crossing
        self.assertTrue(np.all(ref[60:] == "H"))

    def test_interp_never_crosses_a_gap(self):
        values = np.arange(5, dtype=np.float32).reshape(5, 1)
        ok = np.array([True, True, False, False, True])
        out, valid = external._interp_frames(values, ok, 11, 0.4)
        self.assertTrue(valid[:3].all())  # output frames bracketed by source 0-1 are valid
        self.assertFalse(valid[3:10].any())  # any bracket touching the gap is invalid
        self.assertTrue(valid[-1])
        self.assertTrue(np.isnan(out[3:10]).all())
        self.assertTrue(np.allclose(out[:3, 0], [0.0, 0.4, 0.8]))
        self.assertTrue(np.allclose(out[-1, 0], 4.0))

    def test_list_games_rejects_lfs_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "skillcorner", "1234")
            os.makedirs(folder)
            open(os.path.join(folder, "1234_match.json"), "w").close()
            stub = os.path.join(folder, "1234_tracking_extrapolated.jsonl")
            with open(stub, "wb") as fh:
                fh.write(b"version https://git-lfs.github.com/spec/v1\n")
            with mock.patch.dict(os.environ, {"MTAH_EXTERNAL_DIR": tmp}):
                self.assertNotIn("skillcorner:1234", external.list_games("skillcorner"))
                with open(stub, "wb") as fh:
                    fh.write(b"x" * 1_000_001)
                self.assertIn("skillcorner:1234", external.list_games("skillcorner"))


if __name__ == "__main__":
    unittest.main()
