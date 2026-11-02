# Shared per-frame feature library

`possession.features.features(match)` is the common, label-free feature transform for BAR-ATH and
external games. It returns **64 `float32` columns per input frame**, in the stable order
`FEATURE_COLUMNS`. No player ID, reference label, public GT, possession output, or hand-specified
match event is an input. The only slot-specific operation is reading the goalkeeper *role* already
identified independently in each half by the loader.

## API

```python
from possession import config
from possession.data import load_match, load_tags
from possession.features import camera_look_at_x, features

m = load_match()
f = features(
    m,
    camera_x=camera_look_at_x(config.asset_path("homography"), len(m.B)),
    tags=load_tags(),
    cache_key="bar-ath-tactical-s07",
)
# f.shape == (199319, 64)
```

The same transform runs on an external game. Context that the source does not provide stays NaN:

```python
from possession.external import load_game

g = load_game("metrica:game3")
f3 = features(g.match, pitch=(105.0, 68.0), cache_key="metrica-game3-s07")
# f3.shape == (143761, 64)
```

Calling `features(match)` without optional context is valid and computes all tracking features;
`camera_look_at_x_m` and the three tag columns are then NaN. Supply the source's actual `fps` and
`pitch` when they differ from the defaults (25 Hz, 105.3 x 68 m). The function does not resample or
impute tracking. `tags=None` means *source unavailable*; an empty tag DataFrame means *observed zero
coverage*.

With `cache_key`, the DataFrame is written atomically to
`.cache/features-v2-<cache_key>-<digest>.parquet` (or `MTAH_CACHE_DIR`) and read back on the next
call. `<digest>` hashes the feature version, `fps`, `pitch`, every input array, `camera_x` and
`tags`, so a reused key with different parameters or data can never return stale features.

For an external game use `features.game_features(game)`: it passes that game's own pitch size
(`external.pitch(game)`, IDSSE/Metrica are 105 x 68) rather than the 105.3 x 68 default.
Parquet round-trip preserves rows, columns and values exactly.

## Common definitions

- **Coordinates:** origin at the top-left pitch corner, x along pitch length and y across its width.
- **Canonical attacking x:** `(x - pitch_length/2) * attack_sign`, so negative is toward the team's
  own goal and positive toward the opponent's. `attack_sign` is +1 when the team's mean goalkeeper x
  in that half is on the low-x side, otherwise -1. It resets between halves. Canonical columns are
  NaN outside halves 1/2.
- **Hygiene (v2, E16):** velocities beyond physical limits are tracking jumps and become NaN before
  any derived column: player xy-velocity > 12 m/s (`PLAYER_VMAX_MPS`), ball xy-velocity > 50 m/s
  (`BALL_VMAX_MPS`), vendor `V` > 12 m/s. `bbox_area_ratio_home_away` is NaN unless both areas are
  >= 10 m² (`BBOX_MIN_M2`); goalkeeper x/y are NaN when the keeper is beyond the halfway line
  (canonical x > 0, `GK_MAX_ATTACK_X_M`). `attack_sign` is NaN (not 0) outside halves 1/2, so every
  attack-canonical column, including the centroid, is NaN there. `match.gk` slots must belong to
  their team (ValueError otherwise). A goalkeeper substituted mid-half keeps the starter's slot for
  that half (the replacement is an outfield player). For fully cleaned inputs see
  `possession/clean.py` (duplicates, peripheral surplus, ball spikes), which is not wired into the
  feature call.
- **Velocity:** central xy finite differences, with a one-sided estimate when only one adjacent
  observation exists. Differences require finite endpoints in the same half; there is no gap
  interpolation. Relative speed is `||v_player - v_ball||`.
- **Closing speed:** `(v_player - v_ball) dot unit(ball - player)`, in m/s; positive means closing.
- **STRAIN-style time to ball:** per team, the minimum of `max(distance, 0.5 m) / closing_speed`
  over players with positive closing speed, capped at 60 s. It is NaN if nobody is closing or the
  ball is absent. This is a geometric estimate, not an intercept simulation.
- **Spread:** population standard deviation of tracked player x or y. Bounding-box area is
  `(max(x)-min(x)) * (max(y)-min(y))`; it is the cheap proxy for hull area and includes the GK.
- **Defensive line:** 10% nearest-rank order statistic of outfield canonical attacking x (GK excluded).
- **Pressing:** mean over tracked players of `v_player dot unit(opponent_centroid - player)`; positive
  is motion toward the opponent centroid. Coincident player/centroid observations are omitted.
- **Ball-relative columns:** nearest-player ranks are sorted by current distance separately within
  each team. Counts include all tracked players, including the GK. All are NaN when ball xy is absent;
  missing ball is never encoded as zero players nearby.
- **Mean team speed:** mean of the common-format `V` channel. Its delta is the per-second time
  derivative of that mean, not the mean of player accelerations.
- **Missing values:** NaN denotes unavailable/undefined, including unavailable context on external
  games. There are no infinity sentinels in the output.

## Column dictionary

All columns are `float32`. `home`/`away` are common-format team ids 0/1, not inferred possession.

| # | Column | Units / values | Definition |
|---:|---|---|---|
| 1 | `ball_present` | 0/1 | Both ball xy coordinates are finite. |
| 2 | `home_ball_near1_dist_m` | m | Home rank-1 player distance to the ball. |
| 3 | `home_ball_near1_rel_speed_mps` | m/s | Home rank-1 player/ball relative-speed magnitude. |
| 4 | `home_ball_near1_closing_mps` | m/s | Home rank-1 closing speed; positive approaches ball. |
| 5 | `home_ball_near2_dist_m` | m | Home rank-2 player distance to the ball. |
| 6 | `home_ball_near2_rel_speed_mps` | m/s | Home rank-2 player/ball relative-speed magnitude. |
| 7 | `home_ball_near2_closing_mps` | m/s | Home rank-2 closing speed. |
| 8 | `home_ball_near3_dist_m` | m | Home rank-3 player distance to the ball. |
| 9 | `home_ball_near3_rel_speed_mps` | m/s | Home rank-3 player/ball relative-speed magnitude. |
| 10 | `home_ball_near3_closing_mps` | m/s | Home rank-3 closing speed. |
| 11 | `away_ball_near1_dist_m` | m | Away rank-1 player distance to the ball. |
| 12 | `away_ball_near1_rel_speed_mps` | m/s | Away rank-1 player/ball relative-speed magnitude. |
| 13 | `away_ball_near1_closing_mps` | m/s | Away rank-1 closing speed. |
| 14 | `away_ball_near2_dist_m` | m | Away rank-2 player distance to the ball. |
| 15 | `away_ball_near2_rel_speed_mps` | m/s | Away rank-2 player/ball relative-speed magnitude. |
| 16 | `away_ball_near2_closing_mps` | m/s | Away rank-2 closing speed. |
| 17 | `away_ball_near3_dist_m` | m | Away rank-3 player distance to the ball. |
| 18 | `away_ball_near3_rel_speed_mps` | m/s | Away rank-3 player/ball relative-speed magnitude. |
| 19 | `away_ball_near3_closing_mps` | m/s | Away rank-3 closing speed. |
| 20 | `home_ball_count_1m` | players | Number of tracked home players within 1 m of the ball. |
| 21 | `home_ball_count_2m` | players | Number of tracked home players within 2 m of the ball. |
| 22 | `home_ball_count_4m` | players | Number of tracked home players within 4 m of the ball. |
| 23 | `home_ball_count_8m` | players | Number of tracked home players within 8 m of the ball. |
| 24 | `away_ball_count_1m` | players | Number of tracked away players within 1 m of the ball. |
| 25 | `away_ball_count_2m` | players | Number of tracked away players within 2 m of the ball. |
| 26 | `away_ball_count_4m` | players | Number of tracked away players within 4 m of the ball. |
| 27 | `away_ball_count_8m` | players | Number of tracked away players within 8 m of the ball. |
| 28 | `ball_nearest_team` | 0=home, 1=away | Team of the single nearest tracked player; NaN if no player/ball. |
| 29 | `home_ball_time_to_ball_s` | s, capped 60 | Home STRAIN-style minimum time to ball. |
| 30 | `away_ball_time_to_ball_s` | s, capped 60 | Away STRAIN-style minimum time to ball. |
| 31 | `ball_speed_mps` | m/s | Norm of position-derived ball velocity; not ATD `s`. |
| 32 | `ball_accel_mps2` | m/s² | Norm of the time derivative of estimated ball velocity. |
| 33 | `ball_direction_change_deg` | degrees, 0–180 | Angle between previous and current estimated ball velocity. |
| 34 | `home_tracked_count` | players | Number of home slots with finite xy. |
| 35 | `home_centroid_attack_x_m` | m | Home all-player centroid x in canonical attacking coordinates. |
| 36 | `home_centroid_y_m` | m | Home all-player centroid y. |
| 37 | `home_spread_x_m` | m | Home player-x population standard deviation. |
| 38 | `home_spread_y_m` | m | Home player-y population standard deviation. |
| 39 | `home_bbox_area_m2` | m² | Home all-player bounding-box area. |
| 40 | `home_defensive_line_attack_x_m` | m | Home 10% nearest-rank outfield defensive-line height. |
| 41 | `home_gk_attack_x_m` | m | Home goalkeeper x in canonical attacking coordinates. |
| 42 | `home_gk_y_m` | m | Home goalkeeper y. |
| 43 | `home_mean_speed_mps` | m/s | Mean of finite home `V` speed values. |
| 44 | `home_mean_speed_delta_mps2` | m/s² | Per-second derivative of home mean team speed. |
| 45 | `home_pressing_mps` | m/s | Home mean velocity component toward away centroid. |
| 46 | `away_tracked_count` | players | Number of away slots with finite xy. |
| 47 | `away_centroid_attack_x_m` | m | Away all-player centroid x in canonical attacking coordinates. |
| 48 | `away_centroid_y_m` | m | Away all-player centroid y. |
| 49 | `away_spread_x_m` | m | Away player-x population standard deviation. |
| 50 | `away_spread_y_m` | m | Away player-y population standard deviation. |
| 51 | `away_bbox_area_m2` | m² | Away all-player bounding-box area. |
| 52 | `away_defensive_line_attack_x_m` | m | Away 10% nearest-rank outfield defensive-line height. |
| 53 | `away_gk_attack_x_m` | m | Away goalkeeper x in canonical attacking coordinates. |
| 54 | `away_gk_y_m` | m | Away goalkeeper y. |
| 55 | `away_mean_speed_mps` | m/s | Mean of finite away `V` speed values. |
| 56 | `away_mean_speed_delta_mps2` | m/s² | Per-second derivative of away mean team speed. |
| 57 | `away_pressing_mps` | m/s | Away mean velocity component toward home centroid. |
| 58 | `bbox_area_ratio_home_away` | ratio | Home bounding-box area divided by away area; NaN unless both areas >= 10 m². |
| 59 | `camera_look_at_x_m` | m | Image center (960,540) unprojected through homography, x in meters. |
| 60 | `ball_out_of_pitch` | 0/1 | Ball is finite and outside `[0,L] x [0,W]`; NaN if ball absent. |
| 61 | `ball_s_present` | 0/1 | Raw common-format `BS` channel is finite; makes no semantic assumption. |
| 62 | `attacker_tag_coverage_home` | intervals | Active attacker-side Smart Tag intervals assigned to home. |
| 63 | `attacker_tag_coverage_away` | intervals | Active attacker-side Smart Tag intervals assigned to away. |
| 64 | `set_piece_window` | 0/1 | Inside any Smart Tagging `SET PIECES` interval (usually a broad window). |

Attacker-side coverage includes only `ATTACKING TRANSITION`, `BALL IN FINAL THIRD`,
`BALL IN THE BOX`, `BUILD UP`, `CREATING CHANCES`, `LONG BALL`, and `PROGRESSION`. Mirrored
`DEFENDING*` / `DEFENSIVE TRANSITION` intervals are excluded, not double-counted. Set pieces are
kept separate because their 35 s windows are context, not tight restart annotations.

## Verification and limits

- Synthetic tests cover exact schema, missing-ball NaNs, attack orientation, slot-permutation
  invariance, tag availability, cache equality, and homography unprojection.
- Full BAR-ATH: 199,319 x 64, **11.7 s cold including parquet write**, **0.13 s warm** on the team
  machine; no infinities. Camera/ball-x correlation on graded ball-present frames is **0.8284**,
  reproducing the independent audit's approximately 0.83 diagnostic.
- Full Metrica game 3 via S05: 143,761 x 64, **3.3 s cold including parquet write**, **0.11 s warm**;
  no infinities. Camera and Smart Tag columns are correctly all-NaN because that source lacks them.
- No public GT or scorer was used. These are raw features, not evidence that any column predicts
  possession. In particular, Metrica/IDSSE ball tracks contain known vendor dead-ball leaks; apply
  the S06 degradation/stripping protocol before fitting or selecting a downstream model. The feature
  library itself deliberately performs no label-aware cleaning.
