# The possession walkthrough notebook

`possession_walkthrough.ipynb` is the prose-led, traceable version of the work: six sections, a
centerpiece animation of the scoring rule, and an appendix that goes from any number to the command
that produced it. It has **no code cells**. Everything executable is in `../scripts/`.

## Build it

From `solution/`:

```bash
# 1. the numbers manifest (reads the cached dev-v1 reference table and the current submission)
.venv/bin/python scripts/walkthrough_numbers.py

# 2. the figures and the notebook, with real clips rendered locally
.venv/bin/python scripts/build_walkthrough.py --profile local

# 3. the publishable version: synthetic figures kept, real-tracking figures replaced
.venv/bin/python scripts/build_walkthrough.py --profile public

# figures only (useful when iterating on one panel)
.venv/bin/python scripts/build_walkthrough.py --figs-only --only fig_spellf1

# validate: every placeholder exists, every figure is on disk, no recomputed number has drifted
.venv/bin/python scripts/walkthrough_numbers.py --check
.venv/bin/python scripts/build_walkthrough.py --check --profile local
```

`build_walkthrough.py --no-figs` rebuilds the notebook from figures already on disk. Neither script
executes anything from inside the notebook, uses the network, or reads the public answers.

## What is local-only, and why

`notebooks/figs/` is gitignored. Any figure that renders real tracking — our match or an external
practice match — is rendered there and **never published** (decision D-002, guardrail §1.3). The
`public` profile replaces each of those figures with a neutral placeholder in
`notebooks/figs/public/`, keeping the caption. The synthetic schematics and the score animation are
drawn from scratch and appear in both profiles.

Which is which:

| figure | profile |
|---|---|
| `fig_three_labels`, `fig_nearest_player` | synthetic, both |
| `fig_spellf1`, `fig_spellf1_curve` | synthetic, both |
| `fig_missingness`, `fig_gate_instruments`, `fig_width_schematic` | aggregates only, both |
| `fig_match_gap` | our match, local only |
| `fig_practice_clean`, `fig_practice_degraded` | external tracking, local only |
| `fig_stage_*` (six) | external tracking, local only |
| `fig_shape_deform` | external tracking, local only |

## How the traceability works

`notebooks/numbers.json` holds every measured number the prose quotes. Each entry has a `value`, a
`display`, a `source`, an exact `command`, and a `recomputed` flag. The builder substitutes
`{key}` from that file; it refuses to build if a placeholder is missing or if a plain digit appears
in the prose. `walkthrough_numbers.py --check` rebuilds the recomputed values from the live code
(dev-v1 reference table, submission morphology, the synthetic score demonstration) and fails on
drift; digest-read values are re-parsed from `research/` on every build, so a changed digest changes
the notebook or fails loudly.

The appendix at the bottom of the notebook is generated from the same file: key, value, section,
source and command.

## Regenerating after a pipeline change

1. Rebuild the submission: `python -m possession.make_submission`.
2. Rerun `walkthrough_numbers.py` (the dev-v1 cache is reused; delete `.cache/dev_v1/` only if the
   harness changed).
3. Rerun `build_walkthrough.py --profile local`. The stage clips come from one Metrica cell at
   `v=0.12`, stillness knob 1, seed 0; if that cell's guards fail after a change, pick another
   primary cell in `_practice()`.

Nothing in this folder is part of the one-command submission rebuild. It is a reader for the work,
not a dependency of the CSV.
