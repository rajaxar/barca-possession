"""Shared, label-free, permutation-invariant per-frame tracking features.

The input is the common ``possession.data.Match`` layout (or any object with the
same attributes). Coordinates are metres, team 0/1 is home/away, and rows are
video frames. No player-slot identity is used except the goalkeeper role already
identified by the loader. Ball features are missing, never zero-filled, when the
ball is missing.

Optional camera and Smart Tagging context is supplied explicitly so the exact
same tracking feature code runs on BAR-ATH and external games. Use ``cache_key``
to persist a parquet cache under ``config.cache_dir()``; a key must uniquely
identify the input arrays and context.
"""
import hashlib
import os
import re
import warnings

import numpy as np
import pandas as pd

from . import config

FEATURE_VERSION = "v2"  # v2 (E16): jump masking, NaN attack sign off-half, bbox/GK guards, hashed cache key
NEIGHBOURS = 3
RADII_M = (1, 2, 4, 8)
MAX_TIME_TO_BALL_S = 60.0
STRAIN_DISTANCE_FLOOR_M = 0.5
# Physical-limit hygiene (E16): velocities above these are tracking jumps, not motion, and become NaN.
PLAYER_VMAX_MPS = 12.0
BALL_VMAX_MPS = 50.0
BBOX_MIN_M2 = 10.0     # a team spread over less than this has no meaningful area ratio
GK_MAX_ATTACK_X_M = 0.0  # a goalkeeper beyond the halfway line is a tracking error

# Attacker-side interval codes from the Smart Tagging audit. Mirrored DEFENDING*
# codes are deliberately absent; load_tags() has already canonicalised team ids.
ATTACKER_TAG_CODES = frozenset({
    "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "BUILD UP",
    "CREATING CHANCES",
    "LONG BALL",
    "PROGRESSION",
})


def _column_names():
    names = ["ball_present"]
    for team in ("home", "away"):
        for rank in range(1, NEIGHBOURS + 1):
            for quantity in ("dist_m", "rel_speed_mps", "closing_mps"):
                names.append("{}_ball_near{}_{}".format(team, rank, quantity))
    for team in ("home", "away"):
        names.extend("{}_ball_count_{}m".format(team, radius) for radius in RADII_M)
    names.append("ball_nearest_team")
    names.extend("{}_ball_time_to_ball_s".format(team) for team in ("home", "away"))
    names.extend(("ball_speed_mps", "ball_accel_mps2", "ball_direction_change_deg"))
    for team in ("home", "away"):
        names.extend((
            "{}_tracked_count".format(team),
            "{}_centroid_attack_x_m".format(team),
            "{}_centroid_y_m".format(team),
            "{}_spread_x_m".format(team),
            "{}_spread_y_m".format(team),
            "{}_bbox_area_m2".format(team),
            "{}_defensive_line_attack_x_m".format(team),
            "{}_gk_attack_x_m".format(team),
            "{}_gk_y_m".format(team),
            "{}_mean_speed_mps".format(team),
            "{}_mean_speed_delta_mps2".format(team),
            "{}_pressing_mps".format(team),
        ))
    names.extend((
        "bbox_area_ratio_home_away",
        "camera_look_at_x_m",
        "ball_out_of_pitch",
        "ball_s_present",
        "attacker_tag_coverage_home",
        "attacker_tag_coverage_away",
        "set_piece_window",
    ))
    return names


FEATURE_COLUMNS = tuple(_column_names())


def _finite(x):
    """Finite mask, treating an xy/vector row as one observation."""
    return np.isfinite(x) if x.ndim == 1 else np.isfinite(x).all(axis=-1)


def _time_half_mask(same, target):
    return same.reshape((len(same),) + (1,) * (target.ndim - 1))


def _derivative(x, fps, half):
    """Central time derivative with one-sided fallback, reset at half boundaries.

    No interpolation is performed: a difference is used only when both endpoint
    observations are finite and belong to the same half. Works for (N,), (N,2),
    and (N,S,2) arrays and preserves float32.
    """
    x = np.asarray(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if len(x) < 2:
        return out
    valid = _finite(x)
    same_half = _time_half_mask(half[1:] == half[:-1], valid[1:])
    step_valid = valid[1:] & valid[:-1] & same_half
    difference = np.full(x.shape, np.nan, dtype=np.float32)
    difference[:-1][step_valid] = (fps * (x[1:] - x[:-1]))[step_valid]

    # Backward one-sided estimate everywhere it exists.
    out[1:][step_valid] = difference[:-1][step_valid]
    # Forward estimate only where no backward estimate exists.
    forward_only = step_valid & ~_finite(out[:-1])
    out[:-1][forward_only] = difference[:-1][forward_only]
    # Prefer the central estimate where both adjacent differences exist.
    central = step_valid[:-1] & step_valid[1:]
    out[1:-1][central] = (0.5 * (difference[:-2] + difference[1:-1]))[central]
    return out


def _mean(x, axis=1):
    finite = np.isfinite(x)
    count = finite.sum(axis=axis)
    total = np.nansum(x, axis=axis)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _minimum(x, axis=1):
    out = np.min(np.where(np.isfinite(x), x, np.inf), axis=axis)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def _maximum(x, axis=1):
    out = np.max(np.where(np.isfinite(x), x, -np.inf), axis=axis)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def _std(x, axis=1):
    mean = _mean(x, axis=axis)
    expanded = np.expand_dims(mean, axis=axis)
    delta = np.where(np.isfinite(x), x - expanded, np.nan)
    variance = _mean(delta * delta, axis=axis)
    return np.sqrt(np.maximum(variance, 0)).astype(np.float32)


def _nearest_rank(x, fraction):
    """Per-row nearest-rank quantile over axis 1; NaN for an empty row."""
    finite = np.isfinite(x)
    count = finite.sum(axis=1)
    ordered = np.sort(np.where(finite, x, np.inf), axis=1)
    rank = np.maximum(np.ceil(fraction * count).astype(np.int64) - 1, 0)
    out = ordered[np.arange(len(x)), rank]
    out = np.where(count > 0, out, np.nan)
    return out.astype(np.float32)


def _attack_directions(match, pitch_length):
    """Per-frame attack sign from each half's mean goalkeeper x position."""
    n = len(match.B)
    signs = np.full((n, 2), np.nan, dtype=np.float32)  # NaN off-half: no attack direction exists
    for h in (1, 2):
        rows = np.flatnonzero(match.half == h)
        if not len(rows):
            continue
        for team in (0, 1):
            slot = int(match.gk[h - 1, team])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                gk_x = np.nanmean(match.P[rows, slot, 0])
            if not np.isfinite(gk_x):
                raise ValueError("cannot infer attack direction: goalkeeper absent in half {}".format(h))
            signs[rows, team] = 1.0 if gk_x < pitch_length / 2 else -1.0
    return signs


def _validate_match(match, fps, pitch):
    p, v, b = np.asarray(match.P), np.asarray(match.V), np.asarray(match.B)
    n, slots = p.shape[:2]
    if p.shape != (n, slots, 2):
        raise ValueError("P must have shape (frames, slots, 2)")
    if v.shape != (n, slots) or b.shape != (n, 2):
        raise ValueError("V/B shapes do not agree with P")
    if np.asarray(match.BS).shape != (n,) or np.asarray(match.half).shape != (n,):
        raise ValueError("BS/half must have one value per frame")
    if np.asarray(match.T).shape != (slots,) or set(np.unique(match.T)) - {0, 1}:
        raise ValueError("T must assign every slot to team 0 or 1")
    if np.asarray(match.gk).shape != (2, 2):
        raise ValueError("gk must have shape (2 halves, 2 teams)")
    if fps <= 0 or len(pitch) != 2 or min(pitch) <= 0:
        raise ValueError("fps and pitch dimensions must be positive")
    for team in (0, 1):
        if not np.any(np.asarray(match.T) == team):
            raise ValueError("team {} has no tracking slots".format(team))
    gk = np.asarray(match.gk)
    for h in (0, 1):
        for team in (0, 1):
            if not 0 <= gk[h, team] < slots or np.asarray(match.T)[gk[h, team]] != team:
                raise ValueError("gk slot {} of half {} is not a team-{} slot".format(gk[h, team], h + 1, team))
    return n


def camera_look_at_x(path, n_frames=None, image_size=(1920, 1080), pitch_length=105.3):
    """Unproject image centre to pitch x (m) from an organiser homography CSV."""
    table = pd.read_csv(path)
    columns = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    required = {"frame", "valid", *columns}
    if not required.issubset(table.columns):
        raise ValueError("homography is missing columns: {}".format(sorted(required - set(table.columns))))
    if n_frames is not None and len(table) != n_frames:
        raise ValueError("homography has {} rows, expected {}".format(len(table), n_frames))
    if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
        raise ValueError("homography frame column must be contiguous and zero-based")
    matrices = table[columns].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    valid = table["valid"].to_numpy(dtype=bool) & np.isfinite(matrices).all(axis=(1, 2))
    centre = np.array([image_size[0] / 2, image_size[1] / 2, 1.0])
    result = np.full((len(table), 2), np.nan, dtype=np.float64)
    if valid.any():
        rhs = np.broadcast_to(centre, (valid.sum(), 3))[..., None]
        solved = np.linalg.solve(matrices[valid], rhs)[..., 0]
        good = np.isfinite(solved).all(axis=1) & (np.abs(solved[:, 2]) > 1e-12)
        selected = result[valid]
        selected[good] = solved[good, :2] / solved[good, 2, None]
        result[valid] = selected
    return (result[:, 0] * pitch_length).astype(np.float32)


def tag_features(tags, n_frames):
    """Attacker-side interval counts and set-piece-window flag per frame."""
    coverage = np.zeros((n_frames, 2), dtype=np.float32)
    set_piece = np.zeros(n_frames, dtype=np.float32)
    required = {"code", "start_frame", "end_frame", "attacker_team"}
    if not required.issubset(tags.columns):
        raise ValueError("tags are missing columns: {}".format(sorted(required - set(tags.columns))))
    for row in tags.itertuples(index=False):
        start = max(0, int(row.start_frame))
        end = min(n_frames, int(row.end_frame))  # Smart Tagging end is exclusive.
        if end <= start:
            continue
        if row.code == "SET PIECES":
            set_piece[start:end] = 1.0
        if row.code in ATTACKER_TAG_CODES and row.attacker_team in (0, 1):
            coverage[start:end, int(row.attacker_team)] += 1.0
    return {
        "attacker_tag_coverage_home": coverage[:, 0],
        "attacker_tag_coverage_away": coverage[:, 1],
        "set_piece_window": set_piece,
    }


def _input_digest(match, fps, pitch, camera_x, tags):
    """Short hash of every input that changes the output, so a reused cache_key cannot go stale."""
    h = hashlib.blake2b(digest_size=6)
    h.update(repr((FEATURE_VERSION, float(fps), tuple(map(float, pitch)))).encode())
    for name in ("P", "V", "T", "B", "BS", "gk", "half"):
        a = np.ascontiguousarray(getattr(match, name))
        h.update("{}{}{}".format(name, a.shape, a.dtype).encode())
        h.update(a.tobytes())
    if camera_x is not None:
        h.update(np.ascontiguousarray(camera_x, dtype=np.float32).tobytes())
    if tags is not None:
        h.update(pd.util.hash_pandas_object(tags.astype(str), index=True).to_numpy().tobytes())
        h.update(",".join(tags.columns).encode())
    return h.hexdigest()


def _cache_path(cache_key, digest):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", cache_key).strip("-")
    if not safe or safe != cache_key:
        raise ValueError("cache_key may contain only letters, digits, '.', '_' and '-'")
    return os.path.join(config.cache_dir(), "features-{}-{}-{}.parquet".format(FEATURE_VERSION, safe, digest))


def features(match, *, fps=25.0, pitch=(105.3, 68.0), camera_x=None, tags=None,
             cache_key=None):
    """Return a ``frames x 64`` DataFrame of deterministic per-frame features.

    Parameters
    ----------
    match : possession.data.Match or compatible object
        Tracking arrays in metres. ``V`` is player speed in m/s; velocities used
        for relational features are estimated from xy so they also work for
        external formats without a speed channel.
    fps : float
        Tracking sampling rate. Resample external games to their native/common
        timeline before calling; no hidden resampling is done here.
    pitch : (length, width)
        Pitch dimensions in metres, used for canonical x and out-of-pitch flags.
    camera_x : optional (frames,) array
        Camera look-at x in metres. Use :func:`camera_look_at_x` for BAR-ATH.
        NaN throughout when this channel is unavailable.
    tags : optional DataFrame
        ``load_tags()``-format Smart Tagging intervals. ``None`` means source
        unavailable (NaN), whereas an empty DataFrame means known zero coverage.
    cache_key : optional str
        Stable input/context identifier for a parquet cache in ``.cache/``.
    """
    n = _validate_match(match, fps, pitch)
    path = None
    if cache_key is not None:
        path = _cache_path(cache_key, _input_digest(match, fps, pitch, camera_x, tags))
    if path is not None and os.path.isfile(path):
        cached = pd.read_parquet(path)
        if len(cached) != n or tuple(cached.columns) != FEATURE_COLUMNS:
            raise ValueError("feature cache does not match this input/version: {}".format(path))
        return cached

    p = np.asarray(match.P, dtype=np.float32)
    v = np.asarray(match.V, dtype=np.float32)
    b = np.asarray(match.B, dtype=np.float32)
    half = np.asarray(match.half)
    team_ids = np.asarray(match.T)
    pitch_length, pitch_width = map(float, pitch)
    ball_present = _finite(b)
    v = np.where(v > PLAYER_VMAX_MPS, np.nan, v).astype(np.float32)
    player_v = _derivative(p, fps, half)
    player_v[np.linalg.norm(player_v, axis=-1) > PLAYER_VMAX_MPS] = np.nan
    ball_v = _derivative(b, fps, half)
    ball_v[np.linalg.norm(ball_v, axis=-1) > BALL_VMAX_MPS] = np.nan
    attack_sign = _attack_directions(match, pitch_length)
    values = {name: np.full(n, np.nan, dtype=np.float32) for name in FEATURE_COLUMNS}
    values["ball_present"] = ball_present.astype(np.float32)
    values["ball_s_present"] = np.isfinite(match.BS).astype(np.float32)

    # Ball dynamics are computed from position, not the unexplained ATD s channel.
    values["ball_speed_mps"] = np.linalg.norm(ball_v, axis=1).astype(np.float32)
    ball_a = _derivative(ball_v, fps, half)
    values["ball_accel_mps2"] = np.linalg.norm(ball_a, axis=1).astype(np.float32)
    previous_v = np.full_like(ball_v, np.nan)
    previous_v[1:] = ball_v[:-1]
    previous_v[1:][half[1:] != half[:-1]] = np.nan
    dot = np.sum(previous_v * ball_v, axis=1)
    cross = previous_v[:, 0] * ball_v[:, 1] - previous_v[:, 1] * ball_v[:, 0]
    values["ball_direction_change_deg"] = np.degrees(np.arctan2(np.abs(cross), dot)).astype(np.float32)
    values["ball_out_of_pitch"] = (ball_present & (
        (b[:, 0] < 0) | (b[:, 0] > pitch_length) | (b[:, 1] < 0) | (b[:, 1] > pitch_width)
    )).astype(np.float32)
    values["ball_out_of_pitch"][~ball_present] = np.nan

    nearest_by_team = []
    for team in (0, 1):
        prefix = "home" if team == 0 else "away"
        indices = np.flatnonzero(team_ids == team)
        positions = p[:, indices, :]
        velocities = player_v[:, indices, :]
        distance = np.linalg.norm(positions - b[:, None, :], axis=2)
        order = np.argsort(np.where(np.isfinite(distance), distance, np.inf), axis=1)[:, :NEIGHBOURS]
        relative_v = velocities - ball_v[:, None, :]
        relative_speed = np.linalg.norm(relative_v, axis=2)
        to_ball = b[:, None, :] - positions
        closing = np.sum(relative_v * to_ball, axis=2) / np.maximum(distance, 1e-6)
        for j in range(NEIGHBOURS):
            if j >= order.shape[1]:  # Defensive support for tiny synthetic/external teams.
                continue
            selected = order[:, j]
            rows = np.arange(n)
            for suffix, source in (("dist_m", distance), ("rel_speed_mps", relative_speed),
                                   ("closing_mps", closing)):
                values["{}_ball_near{}_{}".format(prefix, j + 1, suffix)] = source[rows, selected]
        for radius in RADII_M:
            count = (distance <= radius).sum(axis=1).astype(np.float32)
            count[~ball_present] = np.nan
            values["{}_ball_count_{}m".format(prefix, radius)] = count
        nearest_by_team.append(_minimum(distance))
        with np.errstate(invalid="ignore", divide="ignore"):
            strain_time = np.where(
                closing > 0,
                np.maximum(distance, STRAIN_DISTANCE_FLOOR_M) / np.maximum(closing, 1e-6),
                np.nan,
            )
        values["{}_ball_time_to_ball_s".format(prefix)] = np.minimum(
            _minimum(strain_time), MAX_TIME_TO_BALL_S
        )

    home_near, away_near = nearest_by_team
    nearest = np.where(home_near <= away_near, 0.0, 1.0).astype(np.float32)
    nearest[~np.isfinite(home_near) & ~np.isfinite(away_near)] = np.nan
    # If only one team is tracked, that team is nearest by definition.
    nearest[np.isfinite(home_near) & ~np.isfinite(away_near)] = 0.0
    nearest[np.isfinite(away_near) & ~np.isfinite(home_near)] = 1.0
    values["ball_nearest_team"] = nearest

    # Ball-free team geometry and motion.
    centroids = []
    for team in (0, 1):
        indices = np.flatnonzero(team_ids == team)
        positions = p[:, indices, :]
        centroids.append(np.stack((_mean(positions[:, :, 0]), _mean(positions[:, :, 1])), axis=1))
    for team in (0, 1):
        prefix = "home" if team == 0 else "away"
        indices = np.flatnonzero(team_ids == team)
        positions = p[:, indices, :]
        velocities = player_v[:, indices, :]
        valid = _finite(positions)
        sign = attack_sign[:, team]
        centroid = centroids[team]
        values["{}_tracked_count".format(prefix)] = valid.sum(axis=1).astype(np.float32)
        values["{}_centroid_attack_x_m".format(prefix)] = (
            (centroid[:, 0] - pitch_length / 2) * sign
        ).astype(np.float32)
        values["{}_centroid_y_m".format(prefix)] = centroid[:, 1]
        values["{}_spread_x_m".format(prefix)] = _std(positions[:, :, 0])
        values["{}_spread_y_m".format(prefix)] = _std(positions[:, :, 1])
        width = _maximum(positions[:, :, 0]) - _minimum(positions[:, :, 0])
        height = _maximum(positions[:, :, 1]) - _minimum(positions[:, :, 1])
        values["{}_bbox_area_m2".format(prefix)] = (width * height).astype(np.float32)

        # Canonical attacking x: own goal < 0, opponent goal > 0.
        canonical_x = (positions[:, :, 0] - pitch_length / 2) * sign[:, None]
        defensive = np.full(n, np.nan, dtype=np.float32)
        gk_x = np.full(n, np.nan, dtype=np.float32)
        gk_y = np.full(n, np.nan, dtype=np.float32)
        for h in (1, 2):
            rows = np.flatnonzero(half == h)
            if not len(rows):
                continue
            gk_slot = int(match.gk[h - 1, team])
            local_gk = int(np.flatnonzero(indices == gk_slot)[0])
            outfield = np.flatnonzero(indices != gk_slot)
            defensive[rows] = _nearest_rank(canonical_x[rows][:, outfield], 0.10)
            gk_x[rows] = canonical_x[rows, local_gk]
            gk_x[rows[gk_x[rows] > GK_MAX_ATTACK_X_M]] = np.nan
            gk_y[rows] = positions[rows, local_gk, 1]
            gk_y[rows[~np.isfinite(gk_x[rows])]] = np.nan
        values["{}_defensive_line_attack_x_m".format(prefix)] = defensive
        values["{}_gk_attack_x_m".format(prefix)] = gk_x
        values["{}_gk_y_m".format(prefix)] = gk_y

        mean_speed = _mean(v[:, indices])
        values["{}_mean_speed_mps".format(prefix)] = mean_speed
        values["{}_mean_speed_delta_mps2".format(prefix)] = _derivative(mean_speed, fps, half)
        to_opponent = centroids[1 - team][:, None, :] - positions
        opponent_distance = np.linalg.norm(to_opponent, axis=2)
        towards = np.sum(velocities * to_opponent, axis=2) / np.maximum(opponent_distance, 1e-6)
        towards[opponent_distance <= 1e-6] = np.nan
        values["{}_pressing_mps".format(prefix)] = _mean(towards)

    np.divide(
        values["home_bbox_area_m2"], values["away_bbox_area_m2"],
        out=values["bbox_area_ratio_home_away"],
        where=(values["away_bbox_area_m2"] >= BBOX_MIN_M2) & (values["home_bbox_area_m2"] >= BBOX_MIN_M2),
    )

    if camera_x is None:
        camera_x = np.full(n, np.nan, dtype=np.float32)
    camera_x = np.asarray(camera_x, dtype=np.float32)
    if camera_x.shape != (n,):
        raise ValueError("camera_x must have one value per frame")
    values["camera_look_at_x_m"] = camera_x
    if tags is not None:
        values.update(tag_features(tags, n))

    result = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    if path is not None:
        temporary = path + ".tmp.parquet"
        result.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    return result


def game_features(game, **kwargs):
    """``features()`` for an external ``evaluate.Game`` with its own pitch size (not the 105.3 x 68 default)."""
    from . import external
    kwargs.setdefault("pitch", external.pitch(game))
    return features(game.match, **kwargs)
