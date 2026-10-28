"""Entry point: python -m possession.make_submission --out outputs/submission.csv

Currently a stub that abstains everywhere (all X). Valid, scores nothing.
"""
import argparse
import os

from possession.config import N_VIDEO_FRAMES


def build_labels(n_frames=N_VIDEO_FRAMES):
    return ["X"] * n_frames


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
