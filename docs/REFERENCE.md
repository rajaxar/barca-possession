# Reference

Maps each checklist item in [WHAT_WE_TRIED.md](WHAT_WE_TRIED.md) to our internal ids and source files. Paths are relative to the team's working repository. `work/cards/` holds one file per piece of work, `research/` the longer write-ups. `work/FINDINGS.md` has one line per finished item. `work/BOARD.md` has the status of every card. Items marked "D-" are entries in `work/DECISIONS.md`.

## When is play stopped?

| Item | Ids | Source |
|---|---|---|
| Ball missing as stoppage sign, shipped rule | E25, I01, B04, Q22 | research/E25_mnar.md, research/I01_ideas.md, work/cards/B04-ship-interim-gate-and-gapdec.md, research/Q22_b04_repro.md |
| Stoppage rule gain on corrected practice data | E76, D-021, Q24 | research/E76_regrid.md, work/DECISIONS.md, research/Q24_e50_review.md |
| First look at the public sample | Q08 | research/Q08_public_mnar_check.md |
| Live gaps check, and its correction | E36, R13 | research/E36_gap_mechanisms.md, research/R13_gap_verdicts.md |
| Visible-ball share grid and its withdrawn reading | S21, Q13, E44, Q20 | research/E44_visibility_grid.md, research/Q13_s21_review.md, research/Q20_e44_review.md |
| Narrowing our match's visible-ball share | E64 | research/E64_present_d_share.md |
| Whistle detector, adoption rule | R06, E15, S22, Q09, S26 | research/E15_result.md, work/ADOPTION.md |
| Public audio model | R11 | research/R11_audio_model.md |
| Crowd loudness | I01 idea 18, E15 | research/E15_result.md |
| Goalkeeper holding | E38 | research/E38_gk_possession.md |
| Resting-ball proxy | E42 | research/E42_present_d_proxy.md |
| Resting-ball gate alone | E47 | work/cards/E47-present-ball-gate-cap.md |
| Grown stoppages | E61, Q21, E65, D-021 | research/E61_present_d_extension.md, research/Q21_e61_review.md, research/E65_grow_radius.md |
| Sub-gap rule | E69 | research/E69_short_hidden_d.md |
| Long-block splitting | E72, E74, D-020 | research/E72_split_d_blocks.md, research/E74_gap_composition.md |
| Restart team by type | E29, I03 | research/E29_restart_team.md, research/I03_football_gaps.md |
| Moving players, missing ball | E48 | research/E48_moving_stoppages.md |
| Look-live-but-dead catalog | E39 | research/E39_confounds.md |
| Ship the two add-ons (in progress) | E66 | work/cards/E66-ship-grown-present-d.md |
| Gap-edge shape (in progress) | E80 | work/cards/E80-gap-edge-shape.md |
| Gate false alarms (planned) | E62 | work/cards/E62-gate-false-alarm-live-certain.md |
| Block end lag (planned) | E75 | work/cards/E75-gated-block-edge-lag.md |
| Tracker resting-ball quality (planned) | E73 | work/cards/E73-tracker-rest-quality.md |
| Grow past 25 m (planned) | E71 | work/cards/E71-grow-radius-past-25m.md |
| Held ball dead vs live (planned) | E43 | work/cards/E43-held-ball-dead-vs-live.md |
| Anomaly, changepoints, restart whistle (planned) | E55, E56, E58 | research/I02_ideas.md |
| One fused detector, restart detector, after-goal chains, threshold under soft stillness (planned) | E07, E08, E37, E51 | work/cards/ |
| Dead-ball share target (planned) | D07 | work/cards/D07-d-prior-decision.md |

## Who has the ball when we can't see it?

| Item | Ids | Source |
|---|---|---|
| Team shape classifier | E04 | research/E04_result.md |
| Missed-possession anatomy, in-gap decoder | E46, Q17 | research/E46_missed_spells.md, research/Q17_e46_review.md |
| Shipping the in-gap decoder, replaced pass marks | E53, D08, D-018 | research/E53_ingap.md, research/D08_ingap_bars.md, work/DECISIONS.md |
| Nobody near the ball, ball visible | E60 | research/E60_gapdec_present.md |
| Smart Tagging | E12 | research/E12_tag_prior.md |
| Expected-score decoding | E26 | research/E26_mbr.md |
| Long ball, occlusion, contact tracing | I01 ideas 20, 21, 22 | work/IDEAS.md |
| In-gap switching rate, refine with expected-score decoding, small-eligible bound (planned) | E67, E68, E70 | work/cards/ |
| Instrument inside gaps (planned) | S37 | work/cards/S37-independent-ingap-instrument.md |
| Shape model on noisy matches (planned) | E22 | work/cards/ |

## Spotting touches

| Item | Ids | Source |
|---|---|---|
| Kick detection from acceleration | E02, R03 | work/cards/E02-detect-ball-touches-kicks-from-ball-kinematics-p.md, research/R03_possession_detection_lit.md |
| Ball smoothing | E18 | research/E18_ball_noise.md |
| Carry and flight evidence | E23 | work/cards/E23-ball-player-coupling-evidence.md |
| Audible kicks | I01 idea 19 | work/IDEAS.md |
| Ceiling on visible changes of owner | R15 | research/R15_flip_ceiling.md |
| Hand-over timing, delay lever, same-moment check | E79, Q25 | research/E79_handover_lag.md, research/Q25_flip_clock.md |
| Hand-over scatter (planned) | E83 | work/cards/E83-handover-scatter.md |
| Deconvolution, collective reversal (planned) | E35, E34 | work/cards/ |
| Decoder lag, timing-shift scoring (planned) | E32, E31 | work/cards/ |

## Making practice matches look like ours

| Item | Ids | Source |
|---|---|---|
| Dataset search, loading, licenses | R02, S05, S16, R10, D06 | research/R02_external_datasets.md, research/S05_external_manifest.md, research/R10_licenses.md, work/DECISIONS.md D-012 |
| Matching answer definitions | R01, R09, S11, Q10 | research/R01_metrica_semantics.md, research/R09_idsse_semantics.md, research/S11_metrica_labels.md, research/Q10_skillcorner_review.md |
| Degrader and smoothing | S06, S17 | research/S06_degradation_fit.md |
| Visible-ball setting | S21 | research/S06_degradation_fit.md |
| Resting ball placement, teleport bug | S32, Q18 | research/S06_degradation_fit.md, research/Q18_s32_review.md |
| Player stillness | S24 | research/S06_degradation_fit.md |
| Ball-clue inflation | E59, Q07 | research/E59_ball_cue_inflation.md, research/Q07_degrade_review.md |
| Resting-ball shake fix (flaw 1) | S34, Q23 | research/S06_degradation_fit.md, research/Q23_s34_review.md |
| Ball hidden in live play (flaw 2) | E74, E50, Q24, D-020 | research/E74_gap_composition.md, research/S06_degradation_fit.md, research/Q24_e50_review.md |
| Geometry check | E78 | research/E78_geometry_harshness.md |
| Frozen practice set, blind spots, difficulty | D01, D-022 | work/cards/D01-decide-the-official-dev-set-and-degradation-leve.md, work/DECISIONS.md |
| Timing noise | E19 | research/E19_timing_noise.md |
| Tracking cleaning, units | E16, B03, Q05 | research/E16_result.md, work/cards/B03-fix-idsse-speed-units.md |
| Process-safe caches | S18 | work/cards/S18-process-safe-caches.md |
| SkillCorner dead-rule check (in progress) | E41 | work/cards/E41-skillcorner-d-sensitivity.md |
| Common scoring harness, shift check | S40 | work/cards/S40-dev-v1-harness.md |
| Run finalists on SkillCorner (planned) | E81 | work/cards/E81-dev-v1b-runner.md |
| Ranking vs degrading level (planned) | E82 | work/cards/ |
| Sliver gaps (planned) | S39 | work/cards/S39-degrader-sliver-gaps.md |
| Smaller refinements (planned) | S33, S36, S29, S30, S13 | work/cards/ |
| Feature shift audit, leak tests, proxy per level, stillness bound, ledger re-issue (planned) | S15, Q19, Q12, Q15, Q16 | work/cards/ |
| World Cup 2022 loading (planned) | S19 | work/cards/S19-pff-raw-tracking-ingest.md |
| SkillCorner dead audit (planned) | E40 | work/cards/E40-skillcorner-d-audit.md |
| Other small checks (planned) | E24, E30 | work/cards/ |

## Tuning for the scoring rule

| Item | Ids | Source |
|---|---|---|
| First entry | E01, P01 | research/E01_result.md |
| Retune and its later correction | E20, Q14, E44, D01 | research/E20_result.md, research/E44_visibility_grid.md, work/DECISIONS.md D-022 |
| Possession length priors | R08 | research/R08_spell_priors.md |
| Decoder choice | R04 | research/R04_decoders.md |
| Label-free score | S10, B05 | work/cards/S10-label-free-model-selection-score-insurance-for-a.md, work/cards/B05-labelfree-vetoes-in-build.md |
| Reinforcement learning idea | I02 idea 12 | work/IDEAS.md |
| Planned | E45, E54, E09, E10, E14, E03, E17, E27, E28, E33, E57, E13, R12 | work/cards/ |
| Architecture race and freeze (planned) | D02, D04, B01, B02, D05, Z01 | work/cards/ |

## Better raw data

| Item | Ids | Source |
|---|---|---|
| Video ball detection | R05, D03, E11 | research/R05_video_ball_detection.md, work/DECISIONS.md D-015 |
| Particle filter | I02 idea 11 | work/IDEAS.md |
| Broadcast ball track | S03, R13 | research/R13_gap_verdicts.md |
| Planned | E49, E06, E52, E05, R14, E12 | work/cards/ |

## Football-knowledge review

| Item | Ids | Source |
|---|---|---|
| 40-situation audit and its killed ideas | I03 | research/I03_football_gaps.md, work/IDEAS.md |
| Ball-in-play sanity band | I03, E69, D07 | research/I03_football_gaps.md, research/E69_short_hidden_d.md |

## Method and evaluation setup, and keeping ourselves honest

| Item | Ids | Source |
|---|---|---|
| Whole-match resampling, two-tier policy, frozen selection rule | S14, D-009, D-010, D-022 | work/DECISIONS.md |
| Evaluation tools, structure | S04, S09 | work/FINDINGS.md |
| Rules audit, guard test | Q01, Q06 | research/Q01_audit.md |
| Reproduction of the first entry | Q02, Q14 | work/cards/Q02-independent-reproduction-check.md |
| Reproduction of the shipped entry | Q22, S12, B04 | research/Q22_b04_repro.md, work/cards/S12-lock-hashes-and-prune-dependencies.md |
| Final audit and reproduction (planned) | Q03, Q04 | work/cards/ |
| Code review | Q05 | research/Q05_review.md |
| Degrader reviews | Q07, Q13, Q18, Q23, Q24 | research/Q07_degrade_review.md, research/Q23_s34_review.md, research/Q24_e50_review.md |
| In-gap decoder review | Q17 | research/Q17_e46_review.md |
| Grown-stoppage review | Q21 | research/Q21_e61_review.md |
| Public sample score | D-008, D-013, B04 | work/DECISIONS.md |
| Adoption rule | D-014, S22 | work/ADOPTION.md |
| Team tooling (planned) | S20, S27, S28, S35, S38, V01, V02, V03, V04 | work/cards/ |

Retired claims are enforced by `tests/test_prose_guard.py`; extend the list there.
