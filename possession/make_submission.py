"""Generate the deterministic submission from the D-028 pipeline.

One command: ``python -m possession.make_submission --out outputs/submission.csv``. All stage work
lives in :mod:`possession.pipeline` (structure -> Baseline+ -> D gate -> grown D -> in-gap decoder ->
sub-gap gate -> short-spell floor). The optional-evidence status lines (audio, ...) are printed after
the CSV is written and can never change the CSV or the exit code (S26).
"""
import argparse
import os

from possession import pipeline


def build_labels(match=None):
    """Backward-compatible entry point: the shipped pipeline's labels (see :func:`pipeline.build_labels`)."""
    return pipeline.build_labels(match=match)


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
