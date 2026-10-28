# Possession labelling — More than a Hack 2027

Frame-level football possession (`H`/`A`/`D`/`X`) for BAR-ATH, ranked on spellF1.

## Install

Python 3.9+. `pip install -r requirements.txt` (versions are pinned later).

## Data

The match data is gated and is never committed here. Unpack the bundle into the organiser repo's
`data/` folder, or point to it:

    export MTAH_DATA_DIR=/path/to/data          # default: ../more-than-a-hack-2027/data
    export MTAH_REPO_DIR=/path/to/organiser     # default: ../more-than-a-hack-2027 (validator)

The tactical video may be named `BAR-ATH_tactical.mp4` or `BAR-ATH_tactical-001.mp4`; both are
found (see `possession/config.py`).

## Run

    make submission        # writes outputs/submission.csv
    make validate          # ...and runs the organiser validator on it
    make test

or `python -m possession.make_submission --out outputs/submission.csv`.

The current output is an all-`X` placeholder.

## Layout

`possession/` package, `scripts/` helpers, `tests/`, `docs/`.
