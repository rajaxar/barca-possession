"""compare_dev: pooled score agrees with score_dev; paired/game bootstrap flag a ~0.04 perturbation
and (game scheme) not a null pair. Synthetic games, no external data."""
import unittest

import numpy as np

from possession import evaluate as ev


def _make_game(rng, n_spells=80):
    dur = np.maximum(25, rng.lognormal(np.log(90), 0.7, n_spells).astype(int))
    lab, prev = [], None
    for d in dur:
        prev = rng.choice([c for c in "HAD" if c != prev])
        lab += [prev] * d
    return np.array(lab)


def _degrade(ref, rng, p_split, jitter=8):
    """Boundary jitter plus a 25-frame flip in the middle of a fraction p_split of long spells."""
    ch = np.flatnonzero(ref[1:] != ref[:-1]) + 1
    b = np.r_[0, ch, len(ref)]
    out = ref.copy()
    for s, e in zip(b[:-1], b[1:]):
        if e - s > 80 and rng.rand() < p_split:
            m = (s + e) // 2
            out[m - 12:m + 13] = "A" if ref[s] != "A" else "H"
    pred = ref.copy()
    for c, k in zip(ch, np.round(rng.normal(0, jitter, len(ch))).astype(int)):
        k = int(np.clip(k, -20, 20))
        if k > 0:
            pred[c:c + k] = ref[c - 1]
        elif k < 0:
            pred[c + k:c] = ref[c]
    pred[out != ref] = out[out != ref]
    return pred


def _dataset(seed, pa, pb, n_games=8):
    rng = np.random.RandomState(seed)
    games = []
    for g in range(n_games):
        ref = _make_game(rng)
        games.append(("g%d" % g, (_degrade(ref, rng, pa), _degrade(ref, rng, pb)), ref))
    return games


A, B = (lambda m: m[0]), (lambda m: m[1])


class Compare(unittest.TestCase):
    def test_pooled_matches_score_dev_and_sign(self):
        ds = _dataset(1, 0.0, 0.1)
        r = ev.compare_dev(A, B, ds, n=50)
        self.assertAlmostEqual(r["a"], ev.score_dev(A, ds, 0)["pooled"]["spell_f1_mean"], places=12)
        self.assertAlmostEqual(r["b"], ev.score_dev(B, ds, 0)["pooled"]["spell_f1_mean"], places=12)
        self.assertGreater(r["delta"], 0)
        self.assertEqual(set(r["per_game_delta"]), {"g%d" % i for i in range(8)})

    def test_deterministic_and_antisymmetric(self):
        ds = _dataset(2, 0.0, 0.1)
        r1, r2 = ev.compare_dev(A, B, ds, n=100, seed=3), ev.compare_dev(A, B, ds, n=100, seed=3)
        self.assertEqual(r1, r2)
        rs = ev.compare_dev(B, A, ds, n=100, seed=3)
        self.assertAlmostEqual(rs["delta"], -r1["delta"], places=12)
        self.assertAlmostEqual(rs["game"]["low"], -r1["game"]["high"], places=12)

    def test_identical_predictors_zero(self):
        ds = _dataset(3, 0.05, 0.05)
        r = ev.compare_dev(A, A, ds, n=100)
        self.assertEqual(r["delta"], 0.0)
        self.assertEqual((r["possession"]["low"], r["game"]["high"]), (0.0, 0.0))
        self.assertFalse(r["game"]["significant"])

    def test_known_perturbation_flagged_null_not(self):
        reps = 25
        eff = {"possession": 0, "game": 0}
        null_game = 0
        deltas = []
        for s in range(reps):
            r = ev.compare_dev(A, B, _dataset(100 + s, 0.0, 0.035), n=200, seed=s)
            deltas.append(r["delta"])
            for k in eff:
                eff[k] += r[k]["significant"]
            null_game += ev.compare_dev(A, B, _dataset(500 + s, 0.05, 0.05), n=200, seed=s)["game"]["significant"]
        self.assertAlmostEqual(np.mean(deltas), 0.04, delta=0.01)
        self.assertGreaterEqual(eff["possession"] / reps, 0.8)
        self.assertGreaterEqual(eff["game"] / reps, 0.7)
        self.assertLessEqual(null_game / reps, 0.15)  # nominal alpha .05


if __name__ == "__main__":
    unittest.main()
