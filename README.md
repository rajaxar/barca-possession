# Possession labeling — More than a Hack 2027

Frame-level football possession (`H`/`A`/`D`/`X`) for BAR-ATH, ranked on spellF1.

## What we tried

- [docs/WHAT_WE_TRIED.md](docs/WHAT_WE_TRIED.md): every idea we have tested or plan to test, why, and what happened.
- [docs/HOW_WE_WORK.md](docs/HOW_WE_WORK.md): how we choose methods without tuning on the match itself.
- [docs/GLOSSARY.md](docs/GLOSSARY.md): the few terms we use.
- [docs/REFERENCE.md](docs/REFERENCE.md): links from each item to our internal notes.

## Install

## Environment

Python 3.13 (the version the pins were built and tested on). Everything is pinned with `==` and hash-locked
(`--hash=sha256:`, all cp313/abi3/pure-python wheels and sdists on PyPI, so pip enforces hashes automatically) in
`requirements.txt`; `requirements-extra.txt` is an empty placeholder for gated extras.

    python3.13 -m venv .venv
    .venv/bin/pip install --no-cache-dir -r requirements.txt   # hashes are enforced
    .venv/bin/pip install -e . --no-deps      # installs the `possession` package

The Makefile uses `.venv/bin/python` by default (override with `make PYTHON=...`).
The `-e` install makes `python -I -m possession.make_submission` work from any directory, since
`-I` drops the working directory from `sys.path`. Match-data scripts should be run the same way (GUARDRAILS §4).

## Data

The match data is gated and is never committed here. Unpack the bundle into the organizer repo's
`data/` folder, or point to it:

    export MTAH_DATA_DIR=/path/to/data          # default: ../more-than-a-hack-2027/data
    export MTAH_REPO_DIR=/path/to/organizer     # default: ../more-than-a-hack-2027 (validator)

The build also **retrains two models on 10 external practice matches** (7 IDSSE + 3 Metrica) on
every run, so those raw games must be on disk at `../external` — not committed, and not part of the
organizer bundle:

    export MTAH_EXTERNAL_DIR=/path/to/external        # default: ../external
    export MTAH_EXTERNAL_CACHE_DIR=/path/to/cache     # default: <external>/cache

Without them `make submission` fails at the model step rather than producing a degraded file. The
sources and licenses are listed in `docs/REFERENCE.md`.

The tactical video may be named `BAR-ATH_tactical.mp4` or `BAR-ATH_tactical-001.mp4`; both are
found (see `possession/config.py`).

## Run

    make submission        # writes outputs/submission.csv with the frozen D-028 stack
    make validate          # ...and runs the organizer validator on it
    make freeze-check      # asserts the D-029 freeze: provenance register, two builds, sha, validator, shape, rules guard, number manifest, manifest shape, public page audit
    make headline          # reproduces the frozen dev-v1 headline (master .4247 vs ship .4039) in ~71 min
    make test

or `python -m possession.make_submission --out outputs/submission.csv`.

`make freeze-check` is the one-command audit of the frozen artifact: it first checks every module-level constant and string-keyed dict field in the shipped path against `docs/constant_provenance.json` (`scripts/constant_provenance.py --check`, 112 constants / 99 dict fields, printed as one line; a missing register is a clean skip), then builds twice, compares the files byte for byte, checks the sha256 `ee7b5b6b…46d8`, runs the organizer validator, counts 370 spells / 138 D stretches / 40,462 X, runs `tests.test_rules_guard`, the walkthrough number manifest (`--check`), the manifest's shape (`notebooks/numbers.json` holds numbers and strings only, is tracked by git, and names only public figures or local-only paths) and, when the page exists, the 13-check publication audit of `notebooks/possession_walkthrough_public.html` (a missing page is a clean skip). It fails loudly on any mismatch and never repairs one. A HEAD ahead of the local tag `final-candidate` is only a warning, because a docs-only follow-up is allowed while the rebuilt CSV keeps the frozen sha.

Baseline+ persists the last near-ball owner's team through gaps, confirms changes with hysteresis,
marks debounced out-of-bounds intervals dead until ball movement signals a restart, applies the
published pre-kickoff D mask, and merges sub-0.8 s spells. It writes X only outside the graded halves.

## Layout

`possession/` package, `scripts/` helpers, `tests/`, `docs/`.
