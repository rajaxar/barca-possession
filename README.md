# Possession labeling — More than a Hack 2027

Team Data Kickers. Labels every frame of Barcelona v Athletic Club `H`, `A`, `D` or `X` from the provided
tracking data, with a deterministic pipeline. The method, data, possession rules and limitations are in
[docs/TECHNICAL_DOCUMENT.md](docs/TECHNICAL_DOCUMENT.md).

Other docs:

- [docs/WHAT_WE_TRIED.md](docs/WHAT_WE_TRIED.md): every idea we tested, why, and what happened.
- [docs/HOW_WE_WORK.md](docs/HOW_WE_WORK.md): how we chose methods without tuning on the match.
- [docs/GLOSSARY.md](docs/GLOSSARY.md): terms.
- [docs/REFERENCE.md](docs/REFERENCE.md): sources for each item in WHAT_WE_TRIED.md.

## Install

Python 3.13.5. Every dependency is pinned and hash-locked in `requirements.txt`, so pip enforces the
hashes.

    python3.13 -m venv .venv
    .venv/bin/pip install --no-cache-dir -r requirements.txt
    .venv/bin/pip install -e . --no-deps      # installs the `possession` package

The Makefile uses `.venv/bin/python` (override with `make PYTHON=...`).

## Data

The match data is not redistributable and is not in this repository. Unpack the organizers' bundle into
their repository's `data/` folder, or point to it:

    export MTAH_DATA_DIR=/path/to/data          # default: ../more-than-a-hack-2027/data
    export MTAH_REPO_DIR=/path/to/organizer     # default: ../more-than-a-hack-2027 (validator)

Every build retrains two models on 10 external practice matches (7 IDSSE, 3 Metrica). They are not in
this repository or the organizers' bundle; put them at `../external` or point to them:

    export MTAH_EXTERNAL_DIR=/path/to/external        # default: ../external
    export MTAH_EXTERNAL_CACHE_DIR=/path/to/cache     # default: <external>/cache

Without them `make submission` fails at the model step instead of writing a degraded file. Sources and
licenses are in the technical document, section 3.

The tactical video may be named `BAR-ATH_tactical.mp4` or `BAR-ATH_tactical-001.mp4`.

## Run

    make submission        # writes outputs/submission.csv (sha256 ee7b5b6b…46d8)
    make validate          # also runs the organizers' validator on it
    make freeze-check      # full audit of the shipped file, below
    make test
    make headline          # re-runs the practice-set comparison (.4247 vs .4039); about 71 min

`make freeze-check` fails on any mismatch and repairs nothing. It checks that Python and all 34 pinned
packages match (`--allow-env-drift` reports mismatches without failing, for a knowingly different
machine); that every module-level constant is registered in `docs/constant_provenance.json`; that two
builds are byte-identical with the expected sha256; that the organizers' validator passes; that the file
has 370 possessions, 138 dead-ball stretches and 40,462 `X` frames; and that the rules guard passes (no
frame literals tied to this match, no reads of the public labels, no network imports, no committed data).

## Layout

`possession/` the pipeline package, `scripts/` helpers, `tests/`, `docs/`.
