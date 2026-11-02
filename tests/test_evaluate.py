"""Scorer wrapper, diagnostics and diff are consistent. Never reads the public GT (the organizer-baseline
check lives in scripts/sanity_public.py --check-baselines, GUARDRAILS 2)."""
import os
import unittest

import numpy as np

from possession import config, data, evaluate

EVAL = os.path.join(config.repo_dir(), "evaluation")
METRICA = os.path.join(config.SOLUTION_DIR, "..", "external", "metrica", "labels", "game1_labels.csv")
pm = evaluate.pm


class Synthetic(unittest.TestCase):
    def test_runs_isolate_games_and_halves(self):
        ref = np.array(list("HHHHAAAADDDD" * 2))
        runs = [(0, 12), (12, 24)]  # touching runs must stay separate
        r = evaluate.score(ref, ref, runs)
        self.assertEqual(r["n_gt_spells"], 4)
        self.assertEqual(r["spell_f1_mean"], 1.0)

    def test_pooled_equals_sum_of_games(self):
        ref = np.array(list("HHHHHHAAAAAA"))
        res = evaluate.score_dev(lambda m: m, [("a", ref, ref), ("b", np.array(list("H" * 12)), ref)], 0)
        self.assertEqual(res["pooled"]["n_gt_spells"], 4)
        self.assertEqual(res["per_game"]["a"]["spell_f1_mean"], 1.0)
        self.assertLess(res["per_game"]["b"]["spell_f1_mean"], 1.0)

    def test_spell_diff_counts_match_scorer(self):
        rng = np.random.RandomState(0)
        ref = np.repeat(rng.randint(0, 3, 40), 50).astype(np.int8)
        pred = ref.copy()
        pred[rng.rand(len(ref)) < 0.05] = rng.randint(0, 3)
        d = evaluate.spell_diff(pred, ref, iou=0.5)
        r = evaluate.score(pred, ref)["spell_f1"][0.5]
        self.assertEqual(len(d["matched"]), r["n_matched"])
        self.assertEqual(len(d["missed_gt"]), r["n_missed"])
        self.assertEqual(len(d["false_pred"]), r["n_spurious"])

    def test_diagnostics_basic(self):
        pred = np.array(list("H" * 10 + "A" * 100 + "D" * 50 + "H" * 40))
        d = evaluate.diagnostics(pred)
        self.assertEqual(d["n_spells"], 3)
        self.assertEqual(d["flickers_lt_1s"], 1)  # 10 frames = 0.4 s
        self.assertEqual(d["n_d_stretches"], 1)


@unittest.skipUnless(os.path.isfile(METRICA), "Metrica labels not present")
class MetricaSmoke(unittest.TestCase):
    def test_trivial_predictor_on_metrica(self):
        import pandas as pd
        lab = pd.read_csv(METRICA)
        ref = lab.label.values
        per = lab.period.values
        runs = [(int(np.flatnonzero(per == p)[0]), int(np.flatnonzero(per == p)[-1]) + 1) for p in (1, 2)]
        res = evaluate.score_dev(lambda m: np.full(len(ref), "H"), [evaluate.Game("metrica1", None, ref, runs)], 50)
        p = res["pooled"]
        self.assertGreater(p["n_gt_spells"], 300)
        self.assertLess(p["spell_f1_mean"], 0.1)
        self.assertIsNotNone(p["spell_bootstrap_low"])


@unittest.skipUnless(os.path.isfile(config.asset_path("atd_raw")), "match data not present")
class DiagnosticsOnMatch(unittest.TestCase):
    def test_baseline_smooth50(self):
        m = data.load_match()
        pred = pm.label_codes(pm.read_label_csv(os.path.join(EVAL, "results", "baseline_smooth50.csv")))
        d = evaluate.diagnostics(pred, m)
        self.assertEqual(len(d["n_spells_per_half"]), 2)
        self.assertGreater(d["n_spells"], 100)


if __name__ == "__main__":
    unittest.main()
