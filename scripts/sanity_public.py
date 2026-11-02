"""The ONLY sanctioned entry point to the public-window ground truth (GUARDRAILS section 2).

Prints the organizer's numbers for a submission CSV so we can check the pipeline is wired
correctly. Record the number; never tune, select or pick a version by it.

    python -I scripts/sanity_public.py [submission.csv] [--bootstrap N]
    python -I scripts/sanity_public.py --check-baselines   # wrapper reproduces the organizer's reports
    python -I scripts/sanity_public.py --label-contingency # definitional check: does D coincide with ball absence?
"""
import argparse
import json
import os
import sys

import numpy as np

from possession import config, data, evaluate, structure

BANNER = """\
=============================================================================
 PUBLIC-GT SANITY CHECK. Record this number, ACT ON NOTHING.
 Do not tune a parameter, threshold or model choice by it (GUARDRAILS 2).
 5 minutes, ~17 possessions: it picks the wrong answer even for a baseline.
============================================================================="""


def check_baselines(gt_path):
    """Score the organizer's four published baselines (not our predictions) and compare with their reports."""
    eval_dir = os.path.join(config.repo_dir(), "evaluation")
    pm = evaluate.pm
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    bad = 0
    for name in ("raw", "smooth25", "smooth50", "smooth125"):
        rep = json.load(open(os.path.join(eval_dir, "reports", "baseline_%s.json" % name)))
        pred = pm.label_codes(pm.read_label_csv(os.path.join(eval_dir, "results", "baseline_%s.csv" % name)))
        r = evaluate.score(pred, gt)  # via the packing wrapper
        for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "kappa"):
            if round(abs(r[k] - rep[k]), 12) != 0:
                bad += 1
                print("MISMATCH {} {}: wrapper {} report {}".format(name, k, r[k], rep[k]))
    print("baselines match organizer reports" if not bad else "{} mismatches".format(bad))
    return 1 if bad else 0


def _bool_runs(mask):
    """Half-open [start, stop) spans where `mask` is True."""
    edges = np.diff(np.concatenate(([False], np.asarray(mask, bool), [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _cluster_interval(runs, labels, code, n=2000, seed=0):
    """Descriptive 95% interval for P(code | frames of these runs), resampling whole runs.

    Frames inside one missing/present run are not independent, so a frame-level CI would be
    far too narrow. This is a description of sampling error, not an estimate to be used anywhere.
    """
    if not runs:
        return float("nan"), float("nan")
    frames = np.array([b - a for a, b in runs])
    hits = np.array([int(np.sum(labels[a:b] == code)) for a, b in runs])
    rng = np.random.RandomState(seed)
    picks = rng.randint(0, len(runs), (n, len(runs)))
    p = hits[picks].sum(1) / frames[picks].sum(1)
    lo, hi = np.percentile(p, [2.5, 97.5])
    return float(lo), float(hi)


def label_contingency(gt_path, n_boot=2000, seed=0):
    """Contingency of the public-window GT label against ball presence (Q08).

    This is a DEFINITIONAL check of what our own `D` means ("does D coincide with the ball being
    untracked?"), allowed by GUARDRAILS 2 and D-008/D-013. It is not a threshold fit and nothing
    in solution/possession/ may depend on these numbers: they are printed once and recorded.
    """
    pm = evaluate.pm
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    runs = pm.valid_runs(gt)  # the public window is the only non-X span of this file
    match = data.load_match("tactical")
    present = np.isfinite(match.B).all(axis=1)

    print("=" * 77)
    print(" LABEL CONTINGENCY (definitional, report-only)")
    print(" Does GT `D` coincide with the ball being absent from the ATD?")
    print(" One shot. No threshold, rate or constant may be taken from this table (GUARDRAILS 2).")
    print("=" * 77)

    frames = np.concatenate([gt[a:b] for a, b in runs])
    pres = np.concatenate([present[a:b] for a, b in runs])
    n = len(frames)
    print("scope: {} graded frames in {} run(s): {}".format(
        n, len(runs), ", ".join("{}-{}".format(a, b - 1) for a, b in runs)))

    # -- 3x2 table ---------------------------------------------------------------------------
    rows = (("H", pm.H), ("A", pm.A), ("D", pm.D))
    col_tot = {"missing": int((~pres).sum()), "present": int(pres.sum())}
    print("\n3x2 table (frames)")
    print("      {:>9} {:>9} {:>9}   {:>9} {:>9}".format(
        "missing", "present", "total", "P(miss|.)", "P(.|miss)"))
    for name, c in rows:
        m = int(np.sum((frames == c) & ~pres))
        p = int(np.sum((frames == c) & pres))
        print("{:>3}   {:>9} {:>9} {:>9}   {:>9.3f} {:>9.3f}".format(
            name, m, p, m + p, m / (m + p) if m + p else float("nan"),
            m / col_tot["missing"] if col_tot["missing"] else float("nan")))
    print("{:>3}   {:>9} {:>9} {:>9}".format("all", col_tot["missing"], col_tot["present"], n))

    print("\nrow-normalized (share of the label's frames that are missing / present)")
    for name, c in rows:
        m = int(np.sum((frames == c) & ~pres))
        p = int(np.sum((frames == c) & pres))
        print("  {}: missing {:.3f}  present {:.3f}  (n={})".format(
            name, m / (m + p) if m + p else float("nan"),
            p / (m + p) if m + p else float("nan"), m + p))

    p_d_miss = np.mean(frames[~pres] == pm.D) if (~pres).any() else float("nan")
    p_d_pres = np.mean(frames[pres] == pm.D) if pres.any() else float("nan")
    d_share = float(np.mean(frames == pm.D))
    miss_share = float(np.mean(~pres))
    print("\nheadline")
    print("  P(D | missing) = {:.4f}".format(p_d_miss))
    print("  P(D | present) = {:.4f}".format(p_d_pres))
    print("  D share of graded window = {:.4f}  ({} frames)".format(
        d_share, int(np.sum(frames == pm.D))))
    print("  missing share of graded window = {:.4f}  ({} frames)".format(
        miss_share, int((~pres).sum())))
    print("  share of D frames that are missing = {:.4f}".format(
        np.sum((frames == pm.D) & ~pres) / max(1, np.sum(frames == pm.D))))

    # -- run-length split --------------------------------------------------------------------
    def frame_runs(span_runs, want_present):
        """Runs of the given presence value inside the graded runs (truncated at window edges)."""
        out = []
        for a, b in span_runs:
            for s, e in _bool_runs(present[a:b] if want_present else ~present[a:b]):
                out.append((a + s, a + e))
        return out

    miss_runs = frame_runs(runs, False)
    pres_runs = frame_runs(runs, True)
    bins = (("<=1s", 0, 25), ("1-4s", 26, 100), (">=4s", 101, 10 ** 9))
    for label, rr in (("missing", miss_runs), ("present", pres_runs)):
        print("\nP(D | {}), by {}-run length ({} runs; runs touching a window edge are truncated)".format(
            label, label, len(rr)))
        print("  {:>6} {:>7} {:>8} {:>9} {:>9}".format("bin", "runs", "frames", "P(D)", "95% CI"))
        for name, lo, hi in bins:
            sel = [(a, b) for a, b in rr if lo <= b - a <= hi]
            nf = sum(b - a for a, b in sel)
            if not nf:
                print("  {:>6} {:>7} {:>8} {:>9}".format(name, len(sel), nf, "-"))
                continue
            p = float(np.mean(np.concatenate([gt[a:b] for a, b in sel]) == pm.D))
            ci = _cluster_interval(sel, gt, pm.D, n_boot, seed)
            print("  {:>6} {:>7} {:>8} {:>9.3f}   [{:.3f},{:.3f}]".format(
                name, len(sel), nf, p, ci[0], ci[1]))
        allp = float(np.mean(np.concatenate([gt[a:b] for a, b in rr]) == pm.D)) if rr else float("nan")
        ci = _cluster_interval(rr, gt, pm.D, n_boot, seed)
        print("  {:>6} {:>7} {:>8} {:>9.3f}   [{:.3f},{:.3f}]".format(
            "all", len(rr), sum(b - a for a, b in rr), allp, ci[0], ci[1]))
        # episode view: frames over-weight long runs, so also count runs by their own label mix
        pure = sum(1 for a, b in rr if np.all(gt[a:b] == pm.D))
        maj = sum(1 for a, b in rr if np.mean(gt[a:b] == pm.D) > 0.5)
        print("  runs all-D {}  majority-D {}  of {}  ({:.3f} all-D, {:.3f} majority-D)".format(
            pure, maj, len(rr), pure / len(rr) if rr else float("nan"),
            maj / len(rr) if rr else float("nan")))

    # -- the forced-D 2H pre-kickoff block (organizer structure) -----------------------------
    half2_first = structure.HALF_BLOCKS[1][1]
    ko2 = structure.KICKOFFS[2]
    block = np.zeros(len(present), bool)
    block[half2_first:ko2] = True
    in_window = sum(int(np.sum(block[a:b])) for a, b in runs)

    # -- label-free context: is this window typical of the match? ---------------------------
    scope = (match.half > 0) & ~block
    wm_missing = scope & ~present
    wm_runs = _bool_runs(wm_missing)
    wm_frames = int(wm_missing.sum())
    print("\nlabel-free context (no GT used): is the public window typical of BAR-ATH?")
    print("  whole match, graded and excluding the forced-D block: {} frames, missing {:.4f} "
          "({} frames), {} missing runs".format(
              int(scope.sum()), wm_frames / max(1, int(scope.sum())), wm_frames, len(wm_runs)))
    print("  median missing-run duration {:.1f}s; share of missing frames in runs >=4s {:.3f}".format(
        float(np.median([b - a for a, b in wm_runs])) / 25.0 if wm_runs else float("nan"),
        sum(b - a for a, b in wm_runs if b - a > 100) / max(1, wm_frames)))
    print("  public window: {} frames, missing {:.4f} ({} frames), {} missing runs -> {:.1f}% of the".format(
        n, miss_share, int((~pres).sum()), len(miss_runs),
        100.0 * (~pres).sum() / max(1, wm_frames)))
    print("  match's missing frames and {:.1f}% of its missing runs.".format(
        100.0 * len(miss_runs) / max(1, len(wm_runs))))
    biggest = max((b - a for a, b in miss_runs), default=0)
    print("  largest window missing run {:.1f}s = {:.1f}% of the window's missing frames".format(
        biggest / 25.0, 100.0 * biggest / max(1, int((~pres).sum()))))

    print("\nper-run detail, missing runs (anonymous: duration and D share only)")
    for k, (a, b) in enumerate(sorted(miss_runs, key=lambda r: r[1] - r[0], reverse=True), 1):
        print("  run {:>2}  {:>6.1f}s  {:>5} frames  P(D)={:.3f}".format(
            k, (b - a) / 25.0, b - a, float(np.mean(gt[a:b] == pm.D))))
    print("per-run detail, present runs")
    for k, (a, b) in enumerate(sorted(pres_runs, key=lambda r: r[1] - r[0], reverse=True), 1):
        print("  run {:>2}  {:>6.1f}s  {:>5} frames  P(D)={:.3f}".format(
            k, (b - a) / 25.0, b - a, float(np.mean(gt[a:b] == pm.D))))

    print("\nforced-D 2H pre-kickoff block (organizer structure, {}-{}, {} frames)".format(
        half2_first, ko2 - 1, ko2 - half2_first))
    print("  frames of it inside the graded public window: {}".format(in_window))
    print("  (the public window is entirely in the 1st half, so the block is EXCLUDED from every")
    print("   number above; it is all D by the published 2H kickoff and would inflate P(D|missing).)")
    print("  context, not scored: ball-missing share inside the block = {:.3f}".format(
        float(np.mean(~present[half2_first:ko2]))))
    print("=" * 77)
    print(" report-only: nothing in solution/possession/ may depend on these numbers (GUARDRAILS 2)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("submission", nargs="?",
                    default=os.path.join(config.SOLUTION_DIR, "outputs", "submission.csv"))
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--check-baselines", action="store_true",
                    help="check the scorer wrapper against the organizer's baseline reports, then exit")
    ap.add_argument("--label-contingency", action="store_true",
                    help="definitional check only: GT label x ball presence over the graded public frames")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="run-resampling draws for the descriptive CIs in --label-contingency")
    args = ap.parse_args(argv)
    gt_path = os.path.join(config.repo_dir(), "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    if args.label_contingency:
        return label_contingency(gt_path, n_boot=args.n_boot)
    print(BANNER)
    pm = evaluate.pm
    if args.check_baselines:
        return check_baselines(gt_path)
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    table = pm.read_label_csv(args.submission)
    reason = pm.validate_submission(table, len(gt))
    if reason:
        print("INVALID submission: " + reason)
        return 1
    res = pm.evaluate(gt, pm.label_codes(table), bootstrap_n=args.bootstrap)
    print(os.path.basename(args.submission))
    print(evaluate.summary_line(res))
    print("graded {}  abstain-on-graded {}  gt/pred spells {}/{}".format(
        res["n_graded"], res["n_abstain_on_graded"], res["n_gt_spells"], res["n_pred_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
