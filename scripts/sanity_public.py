"""The ONLY sanctioned entry point to the public-window ground truth (GUARDRAILS section 2).

Prints the organiser's numbers for a submission CSV so we can check the pipeline is wired
correctly. Record the number; never tune, select or pick a version by it.

    python -I scripts/sanity_public.py [submission.csv] [--bootstrap N]
"""
import argparse
import os
import sys

from possession import config, evaluate

BANNER = """\
=============================================================================
 PUBLIC-GT SANITY CHECK. Record this number, ACT ON NOTHING.
 Do not tune a parameter, threshold or model choice by it (GUARDRAILS 2).
 5 minutes, ~17 possessions: it picks the wrong answer even for a baseline.
============================================================================="""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("submission", nargs="?",
                    default=os.path.join(config.SOLUTION_DIR, "outputs", "submission.csv"))
    ap.add_argument("--bootstrap", type=int, default=0)
    args = ap.parse_args(argv)
    print(BANNER)
    pm = evaluate.pm
    gt_path = os.path.join(config.repo_dir(), "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
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
