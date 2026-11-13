# Reference

The experiment write-ups behind [WHAT_WE_TRIED.md](WHAT_WE_TRIED.md) are the team's working notes and are not in this repository. This page maps each of its sections to the code and tests that are. Every constant and its source is in [CONSTANTS.md](CONSTANTS.md); the possession rules are in [possession_rules.md](possession_rules.md). `possession/pipeline.py` lists the stages the build runs; the other modules are tooling or ideas we tested.

| Section of WHAT_WE_TRIED.md | Code in `possession/` | Tests in `tests/` |
|---|---|---|
| When is play stopped? | `dgate.py` (stoppage rule), `growd.py` (grown stoppages), `dproxy.py` (resting-ball and other label-free dead-ball clues, the adoption rule for match-only clues), `subgap.py` (sub-gap rule), `structure.py` (published half boundaries and kickoffs); `audio.py` (whistle), `gk_possession.py` (keeper holding), `deadball.py`, `restarts.py` (restart team) | `test_dgate`, `test_growd`, `test_subgap`, `test_structure`, `test_audio`, `test_audio_contract`, `test_dproxy`, `test_gk_possession`, `test_deadball` |
| Who has the ball when we can't see it? | `ballfree.py` (team-shape model), `ingap.py` (in-gap decoder); `hsmm.py` (duration-aware decoder) | `test_ballfree`, `test_ingap`, `test_hsmm` |
| Spotting touches of the ball | `touches.py`, `spells_from_events.py` | `test_touches`, `test_spells_from_events` |
| Making practice matches look like ours | `external.py` (loading), `degrade.py` (degrader), `clean.py` (tracking cleaning), `dev_v1.py` (frozen practice-set harness) | `test_external`, `test_degrade`, `test_clean`, `test_dev_v1`, `test_cache_concurrency` |
| Tuning for the scoring rule | `baseline_plus.py` (Baseline+), `postprocess.py` (short-spell floor); `supervised.py` | `test_baseline_plus`, `test_postprocess` |
| Better raw data | `broadcast.py` (broadcast ball track) | |
| Football-knowledge review | `diagnostics.py` (ball-in-play share against the playing clock) | `test_diagnostics` |
| Method and evaluation setup | `evaluate.py` (scoring), `dev_v1.py`, `scripts/headline.py` | `test_evaluate`, `test_compare`, `test_dev_v1` |
| Keeping ourselves honest | `scripts/freeze_check.py`, `scripts/constant_provenance.py` | `test_rules_guard`, `test_prose_guard`, `test_constant_provenance`, `test_submission_format` |

Retired claims are enforced by `tests/test_prose_guard.py`; extend the list there.
