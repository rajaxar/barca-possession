"""Generate the deterministic E01 Baseline+ submission.

Usage: python -m possession.make_submission --out outputs/submission.csv
"""
import argparse
import os

from possession.baseline_plus import predict
from possession.data import load_match


def build_labels():
    """Load the published tracking inputs and run the unrandomised Baseline+ pipeline."""
    return predict(load_match())


def write_csv(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args(argv)
    write_csv(build_labels(), args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
