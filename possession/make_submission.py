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


def audio_status_line():
    """`audio: ok|reason` for the run log. Never raises: audio is optional evidence (S26)."""
    try:
        from possession import audio
        return audio.status_line()
    except Exception:  # noqa: BLE001 - the submission build must not depend on audio at all
        return "audio: audio_error"


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
    print(audio_status_line())


if __name__ == "__main__":
    main()
