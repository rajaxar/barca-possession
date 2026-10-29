# Reference

Maps each checklist item in [WHAT_WE_TRIED.md](WHAT_WE_TRIED.md) to our internal ids and source files. Paths are relative to the team's working repository. `work/cards/` holds one file per piece of work, `research/` the longer write-ups. `work/FINDINGS.md` has one line per finished item.

## When is play stopped?

| Item | Ids | Source |
|---|---|---|
| Ball missing as stoppage sign | E25, I01 | research/E25_mnar.md, research/I01_ideas.md |
| First look at the public sample | Q08 | research/Q08_public_mnar_check.md |
| Live gaps check, and its correction | E36, R13 | research/E36_gap_mechanisms.md, research/R13_gap_verdicts.md |
| Visible-ball share grid | S21, Q13, E44, Q20 | research/E44_visibility_grid.md, research/Q13_s21_review.md |
| Whistle detector, adoption rule | R06, E15, S22, Q09, S26 | research/E15_result.md, work/ADOPTION.md |
| Crowd loudness | I01 idea 18, E15 | research/E15_result.md |
| Goalkeeper holding | E38 | research/E38_gk_possession.md |
| Resting-ball proxy | E42 | research/E42_present_d_proxy.md |
| Resting-ball gate alone | E47 | work/cards/E47-present-ball-gate-cap.md |
| Moving players, missing ball | E48 | research/E48_moving_stoppages.md |
| Look-live-but-dead catalogue | E39 | research/E39_confounds.md |
| Grow resting moment (in progress) | E61 | work/cards/E61-present-ball-d-extension.md |
| Restart team by type (planned) | E29, I03 | research/I03_football_gaps.md |
| Gate false alarms (planned) | E62 | work/cards/E62-gate-false-alarm-live-certain.md |
| Anomaly, changepoints, restart whistle (planned) | E55, E56, E58 | research/I02_ideas.md |

## Who has the ball when we can't see it?

| Item | Ids | Source |
|---|---|---|
| Team shape classifier | E04 | research/E04_result.md |
| Missed-possession anatomy, in-gap decoder | E46, Q17 | research/E46_missed_spells.md, research/Q17_e46_review.md |
| Ball present, nobody near (in progress) | E60 | work/cards/E60-gapdec-low-confidence-present.md |
| Ship in-gap decoder (planned) | E53 | work/cards/E53-gapdec-integration.md |
| Long ball, occlusion, contact tracing | I01 ideas 20, 21, 22 | work/IDEAS.md |

## Spotting touches

| Item | Ids | Source |
|---|---|---|
| Kick detection from acceleration | E02, R03 | work/cards/E02-detect-ball-touches-kicks-from-ball-kinematics-p.md, research/R03_possession_detection_lit.md |
| Ball smoothing | E18 | research/E18_ball_noise.md |
| Carry and flight evidence | E23 | work/cards/E23-ball-player-coupling-evidence.md |
| Audible kicks | I01 idea 19 | work/IDEAS.md |
| Deconvolution, collective reversal, ceiling (planned) | E35, E34, R15 | work/cards/ |

## Making practice matches look like ours

| Item | Ids | Source |
|---|---|---|
| Dataset search, loading, licences | R02, S05, S16, R10, D06 | research/R02_external_datasets.md, research/S05_external_manifest.md, research/R10_licences.md, work/DECISIONS.md D-012 |
| Matching answer definitions | R01, R09, S11, Q10 | research/R01_metrica_semantics.md, research/R09_idsse_semantics.md, research/S11_metrica_labels.md, research/Q10_skillcorner_review.md |
| Degrader and smoothing | S06, S17 | research/S06_degradation_fit.md |
| Visible-ball setting | S21 | research/S06_degradation_fit.md |
| Resting ball placement | S32, Q18 | research/S06_degradation_fit.md, research/Q18_s32_review.md |
| Player stillness | S24 | research/S06_degradation_fit.md |
| Ball-clue inflation | E59 | research/E59_ball_cue_inflation.md |
| Make resting ball steady (in progress) | S34 | work/cards/S34-whiten-restart-ball-wobble.md |
| Range re-check (in progress) | Q20 | work/cards/Q20-verify-e44-visibility-grid.md |
| Timing noise | E19 | research/E19_timing_noise.md |
| Tracking cleaning, units | E16, B03, Q05 | research/E16_result.md, work/cards/B03-fix-idsse-speed-units.md |
| Sign-off, refinements (planned) | D01, E64, E50, S25, S33, S29, S30 | work/cards/ |

## Tuning for the scoring rule

| Item | Ids | Source |
|---|---|---|
| First entry | E01, P01 | research/E01_result.md |
| Retune and its later correction | E20, Q14, E44 | research/E20_result.md, research/E44_visibility_grid.md |
| Possession length priors | R08 | research/R08_spell_priors.md |
| Decoder choice | R04 | research/R04_decoders.md |
| Planned | E45, E54, E09, E10, E26, E27, E28, E33, E57, E13 | work/cards/ |
| Reinforcement learning idea | I02 idea 12 | work/IDEAS.md |

## Better raw data

| Item | Ids | Source |
|---|---|---|
| Video ball detection | R05, D03, E11 | research/R05_video_ball_detection.md, work/DECISIONS.md D-015 |
| Particle filter | I02 idea 11 | work/IDEAS.md |
| Broadcast ball track | S03, R13 | research/R13_gap_verdicts.md |
| Planned | E49, E06, E52, E05, E12 | work/cards/ |

## Football-knowledge review

| Item | Ids | Source |
|---|---|---|
| Clip watching | S23 | work/HUMAN_NOTES.md, work/FINDINGS.md |
| 40-situation audit and its killed ideas | I03 | research/I03_football_gaps.md, work/IDEAS.md |

## Method and evaluation setup, and keeping ourselves honest

| Item | Ids | Source |
|---|---|---|
| Whole-match resampling, two-tier policy | S14, D-009, D-010 | work/DECISIONS.md |
| Evaluation tools, structure | S04, S09 | work/FINDINGS.md |
| Rules audit, guard test | Q01, Q06 | research/Q01_audit.md |
| Reproduction | Q02, Q14 | work/cards/Q02-independent-reproduction-check.md |
| Code review | Q05 | research/Q05_review.md |
| Degrader review | Q07, Q13, Q18 | research/Q07_degrade_review.md |
| Public sample score | D-008, D-013 | work/DECISIONS.md |
| Adoption rule | D-014, S22 | work/ADOPTION.md |
