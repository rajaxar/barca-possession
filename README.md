# Possession labelling — More than a Hack 2027

Frame-level football possession (`H`/`A`/`D`/`X`) for BAR-ATH, ranked on spellF1.

## What we tried

- [docs/WHAT_WE_TRIED.md](docs/WHAT_WE_TRIED.md): every idea we have tested or plan to test, why, and what happened.
- [docs/HOW_WE_WORK.md](docs/HOW_WE_WORK.md): how we choose methods without tuning on the match itself.
- [docs/GLOSSARY.md](docs/GLOSSARY.md): the few terms we use.
- [docs/REFERENCE.md](docs/REFERENCE.md): links from each item to our internal notes.

## Install

## Environment

Python 3.13 (the version the pins were built and tested on). Everything is pinned with `==` in
`requirements.txt`; `requirements-extra.txt` is an empty placeholder for gated extras.

    python3.13 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e . --no-deps      # installs the `possession` package

The Makefile uses `.venv/bin/python` by default (override with `make PYTHON=...`).
The `-e` install makes `python -I -m possession.make_submission` work from any directory, since
`-I` drops the working directory from `sys.path`. Match-data scripts should be run the same way (GUARDRAILS §4).

## Data

The match data is gated and is never committed here. Unpack the bundle into the organiser repo's
`data/` folder, or point to it:

    export MTAH_DATA_DIR=/path/to/data          # default: ../more-than-a-hack-2027/data
    export MTAH_REPO_DIR=/path/to/organiser     # default: ../more-than-a-hack-2027 (validator)

The tactical video may be named `BAR-ATH_tactical.mp4` or `BAR-ATH_tactical-001.mp4`; both are
found (see `possession/config.py`).

## Run

    make submission        # writes outputs/submission.csv with the deterministic E01 Baseline+
    make validate          # ...and runs the organiser validator on it
    make test

or `python -m possession.make_submission --out outputs/submission.csv`.

Baseline+ persists the last near-ball owner's team through gaps, confirms changes with hysteresis,
marks debounced out-of-bounds intervals dead until ball movement signals a restart, applies the
published pre-kickoff D mask, and merges sub-0.8 s spells. It writes X only outside the graded halves.

## Layout

`possession/` package, `scripts/` helpers, `tests/`, `docs/`.
