# The possession walkthrough notebook

`possession_walkthrough.ipynb` is the prose-led, traceable version of the work: eight sections (problem,
score, practice set, pipeline, clues, negatives, results, limits), a centerpiece animation of the scoring
rule, and an appendix that goes from a headline number to its source and command. It is written for a reader who knows football and no machine learning: a start-here box, a concept section on the in-gap decoder, eight sections, a sorted glossary and the appendix. Figure text is raised to a minimum size by `_legible()` in the builder. It has **no code cells**. Everything executable is in `../scripts/`.

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

## Export the public page

The shareable artifact is the `public` profile rendered to one standalone HTML file: every image is
inlined as base64, there is no JavaScript, no web font and no external request, so it opens offline.

```bash
# numbers -> public figures -> single HTML (does not rewrite the notebook on disk)
.venv/bin/python scripts/build_walkthrough_html.py

# publication audit: exit 0 = publishable, 1 = stop (script with an exit code, not a visual check)
.venv/bin/python scripts/audit_walkthrough_public.py
```

The page is written to `notebooks/possession_walkthrough_public.html`. It is **not published** by
these commands: the file stays local until the human decides (D-002/D-026). The audit proves the
figure set, the absence of real-tracking images, and the absence of frame numbers, clock times,
match paths and work-card ids outside the appendix; it also rejects any `figs/public/*.png` that is
not pixel-identical to a freshly rendered neutral placeholder.

## Refresh the local reading copy

The human's personal reading copy is the `local` profile rendered to one standalone HTML file, with
every real-tracking clip inlined. It is **local-only**: do not commit it, publish it, upload it or
send it anywhere (D-002/D-026, GUARDRAILS §1.3). It is gitignored.

```bash
# one command: reuse or rebuild the frozen CSV, refresh numbers, figures, local notebook and page
make notebook-local
```

The target reuses `outputs/submission.csv` while `walkthrough_numbers.py --check` is green, so a
parallel card editing an unrelated file cannot force a three-minute rebuild. It rebuilds only if the
CSV is missing or the manifest reports drift, verifies the frozen sha, and exits non-zero on number
drift. The output is `notebooks/possession_walkthrough_local.html`; a footer records the CSV sha256
and the git commit the page was built from (never a wall-clock time, so rebuilds stay comparable).
The public export remains `python scripts/build_walkthrough_html.py` with no arguments.


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
| `fig_missingness`, `fig_gate_instruments`, `fig_width_schematic`, `fig_killed` | aggregates/synthetic, both |
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

The appendix at the bottom of the notebook is generated from the same file: a headline-key table (key,
value, section, recomputed) plus the figure table. Every key's source and exact command are in
`numbers.json`.

## Regenerating after a pipeline change

1. Rebuild the submission: `python -m possession.make_submission`.
2. Rerun `walkthrough_numbers.py` (the dev-v1 cache is reused; delete `.cache/dev_v1/` only if the
   harness changed).
3. Rerun `build_walkthrough.py --profile local`. The stage clips come from one Metrica cell at
   `v=0.12`, stillness knob 1, seed 0; if that cell's guards fail after a change, pick another
   primary cell in `_practice()`.

Nothing in this folder is part of the one-command submission rebuild. It is a reader for the work,
not a dependency of the CSV.
