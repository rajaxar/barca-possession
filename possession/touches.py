"""Ball touch / kick detector (E02): kinematic discontinuity AND player proximity.

Per contiguous half the ball track is cleaned (physically implausible jumps are dropped, gaps of at
most ``fill`` frames are interpolated, then the track is split into present stretches). A touch
candidate is a local maximum of the velocity change across a frame (``|v_out - v_in|`` over ``w``
frames each side, on a lightly smoothed track) that exceeds a threshold *calibrated on the game
itself* and has a player within ``r`` meters. The threshold is ``median + c * sigma_MAD`` of the same statistic
on free-flight frames (no player within ``far`` meters), floored at ``floor`` m/s, so no number
transfers between datasets (R03). A tail quantile was tried and dropped: on the ATD the free-flight
tail holds real touches by untracked players. No labels, public GT or event data are read.

``detect_touches`` returns a DataFrame with one row per touch: ``frame`` (row index into the arrays),
``slot`` (player column in ``P``), ``team`` (0/1), ``conf`` (0-1, from distance and the gap between
the nearest player of each team), ``kind`` (kick / receive / deflection / touch), plus ``dist``,
``dv``, ``speed_in`` and ``speed_out`` for diagnostics. ``ball_stretches`` reports ball-present
stretches separately (a touch inside a long gap cannot be seen at all).
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

FPS = 25
COLUMNS = ["frame", "slot", "team", "conf", "kind", "dist", "dv", "speed_in", "speed_out"]
DEFAULT_PARAMS = dict(w=3, r=1.5, far=5.0, c=6.0, floor=1.5, vmax=40.0, fill=3, nms=5, dv_kind=3.0,
                      smooth=5, min_run=9,
                      sep_min=1.5, prom=0.5, psep=8)


def _clean_ball(b, fps, vmax, fill):
    """Drop implausible jumps, fill short gaps. Returns xy with NaN for missing/rejected frames."""
    b = b.astype(np.float64).copy()
    last = None
    dropped = 0
    for i in range(len(b)):
        if np.isnan(b[i, 0]):
            continue
        if last is not None:
            gap = i - last
            if np.linalg.norm(b[i] - b[last]) * fps / gap > vmax and dropped < 5:
                b[i] = np.nan
                dropped += 1
                continue
        dropped = 0
        last = i
    ok = ~np.isnan(b[:, 0])
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return b
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, c = idx[g], idx[g + 1]
        if c - a - 1 <= fill:
            t = np.arange(a + 1, c)
            for k in (0, 1):
                b[t, k] = np.interp(t, [a, c], [b[a, k], b[c, k]])
    return b


def _runs(mask):
    x = np.asarray(mask, np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, x, 0]) != 0)
    return [(int(a), int(b)) for a, b in e.reshape(-1, 2)]


def ball_stretches(B, half, fps=FPS, min_frames=1):
    """Ball-present stretches per half: DataFrame(half, start, end, seconds)."""
    rows = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        for a, b in _runs(~np.isnan(B[idx, 0])):
            if b - a >= min_frames:
                rows.append((int(h), int(idx[0] + a), int(idx[0] + b), (b - a) / fps))
    return pd.DataFrame(rows, columns=["half", "start", "end", "seconds"])


def _nearest(P, T, b):
    """Distance to, and slot of, the nearest player of each team for every row. -> (d[n,2], j[n,2])."""
    n = len(b)
    d = np.linalg.norm(P - b[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    out_d = np.full((n, 2), np.inf)
    out_j = np.full((n, 2), -1, np.int64)
    for team in (0, 1):
        slots = np.flatnonzero(T == team)
        sub = d[:, slots]
        k = sub.argmin(axis=1)
        out_d[:, team] = sub[np.arange(n), k]
        out_j[:, team] = slots[k]
    return out_d, out_j


def proximity_touches(P, T, B, half, fps=FPS, params=None):
    """Fallback: local minima of the nearest-player distance (<= ``r``, prominence >= ``prom``).

    Used when the ball track carries no kinematic touch signature (see ``separation`` in
    :func:`detect_touches`). Same columns as :func:`detect_touches`; ``kind`` is "proximity".
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    out = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        c = _clean_ball(B[idx], fps, p["vmax"], p["fill"])
        for a, b in _runs(~np.isnan(c[:, 0])):
            if b - a < p["min_run"]:
                continue
            d, j = _nearest(P[idx[0] + a:idx[0] + b], T, c[a:b])
            nd = savgol_filter(d.min(axis=1), 5, 2)
            peaks, _ = find_peaks(-nd, height=-p["r"], prominence=p["prom"], distance=p["psep"])
            for k in peaks:
                team = int(np.argmin(d[k]))
                conf = max(0.0, 1 - d[k, team] / p["r"]) * np.tanh(max(d[k, 1 - team] - d[k, team], 0.0))
                out.append((int(idx[0] + a + k), int(j[k, team]), team, float(conf), "proximity",
                            float(d[k, team]), np.nan, np.nan, np.nan))
    return pd.DataFrame(out, columns=COLUMNS)


def detect_touches(P, T, B, half, fps=FPS, params=None, return_info=False):
    """Detect touches. ``P`` (n, slots, 2), ``T`` (slots,) team 0/1, ``B`` (n, 2), ``half`` (n,).

    ``separation`` = q90 of the velocity change near players / q90 on free-flight frames. Kinematics
    only help when it is well above 1 (IDSSE ~5); below ``sep_min`` the proximity fallback is used.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    w = p["w"]
    n = len(B)
    dv = np.full(n, np.nan)
    sp_in = np.full(n, np.nan)
    sp_out = np.full(n, np.nan)
    dnear = np.full((n, 2), np.inf)
    jnear = np.full((n, 2), -1, np.int64)
    clean = np.full((n, 2), np.nan)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        a0 = idx[0]
        c = _clean_ball(B[idx], fps, p["vmax"], p["fill"])
        clean[idx] = c
        for a, b in _runs(~np.isnan(c[:, 0])):
            if b - a < max(p["min_run"], 2 * w + 1):
                continue
            seg = c[a:b]
            win = min(p["smooth"], b - a - (1 - (b - a) % 2))
            if win >= 3:
                seg = savgol_filter(seg, win, min(2, win - 1), axis=0)
            vin = (seg[w:-w] - seg[:-2 * w]) * fps / w
            vout = (seg[2 * w:] - seg[w:-w]) * fps / w
            rows = a0 + a + np.arange(w, b - a - w)
            dv[rows] = np.linalg.norm(vout - vin, axis=1)
            sp_in[rows] = np.linalg.norm(vin, axis=1)
            sp_out[rows] = np.linalg.norm(vout, axis=1)
            dnear[rows], jnear[rows] = _nearest(P[rows], T, c[a + w:b - w])

    nearest = dnear.min(axis=1)
    free = np.isfinite(dv) & (nearest > p["far"])
    if free.sum() < 100:
        raise ValueError("too few free-flight frames to calibrate the touch threshold")
    med = float(np.median(dv[free]))
    sigma = 1.4826 * float(np.median(np.abs(dv[free] - med)))
    theta = max(med + p["c"] * sigma, p["floor"])

    near_f = np.isfinite(dv) & (nearest <= p["r"])
    separation = float(np.quantile(dv[near_f], 0.9) / max(np.quantile(dv[free], 0.9), 1e-6)) \
        if near_f.sum() >= 100 else 0.0
    if p.get("mode", "auto") == "proximity" or (p.get("mode", "auto") == "auto"
                                                and separation < p["sep_min"]):
        df = proximity_touches(P, T, B, half, fps, params)
        return (df, dict(theta=theta, separation=separation, mode="proximity", n_free=int(free.sum()),
                         params=p)) if return_info else df

    # Candidates: local maxima of dv above theta with a player close by (any frame within +-w).
    pad = np.where(np.isfinite(dv), dv, -1.0)
    near_win = nearest.copy()
    for s in range(1, w + 1):
        near_win[s:] = np.minimum(near_win[s:], nearest[:-s])
        near_win[:-s] = np.minimum(near_win[:-s], nearest[s:])
    cand = np.flatnonzero((pad >= theta) & (near_win <= p["r"]))
    cand = cand[np.argsort(-pad[cand], kind="stable")]  # strongest first, non-max suppression
    taken = np.zeros(n, bool)
    keep = []
    for i in cand:
        lo, hi = max(0, i - p["nms"]), min(n, i + p["nms"] + 1)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        keep.append(i)
    keep = np.sort(np.array(keep, np.int64))

    out = []
    for i in keep:
        lo, hi = max(0, i - w), min(n, i + w + 1)
        sel = np.arange(lo, hi)[half[lo:hi] == half[i]]
        best = sel[np.argmin(nearest[sel])]
        d = dnear[best]
        team = int(np.argmin(d))
        d_best, d_other = d[team], d[1 - team]
        conf = max(0.0, 1 - d_best / p["r"]) * np.tanh(max(d_other - d_best, 0.0))
        diff = sp_out[i] - sp_in[i]
        if diff > p["dv_kind"]:
            kind = "kick"
        elif diff < -p["dv_kind"]:
            kind = "receive"
        elif dv[i] > 2 * theta:
            kind = "deflection"
        else:
            kind = "touch"
        out.append((int(i), int(jnear[best, team]), team, float(conf), kind, float(d_best),
                    float(dv[i]), float(sp_in[i]), float(sp_out[i])))
    df = pd.DataFrame(out, columns=COLUMNS)
    if return_info:
        return df, dict(theta=theta, med=med, sigma=sigma, separation=separation, mode="kinematic", n_free=int(free.sum()), params=p)
    return df
