"""Missing-ball dead-ball gate (E25 whole-gap rule, S21 ``dgated``, verified by Q20), shipped as B04.

Baseline+ persists the last owner through every ball-missing gap, but in the broadcast tracking a
long ball-missing run is mostly a stoppage. A player-only classifier P(D) (no ball columns, 5 Hz) says
how dead-ball-like the players' movement is. Whole-gap rule: a ball-missing run of at least ``TF``
frames whose mean P(D) exceeds ``Q`` becomes D in full; present H/A islands shorter than ``ABSORB``
frames between two D stretches are then absorbed into D.

P(D) model (E25 'hgb' on the 'phys' columns): a HistGradientBoostingClassifier trained at build time,
deterministically (fixed seed), on the external games only (:data:`TRAIN_PREFIXES`: 7 IDSSE + 3
Metrica; SkillCorner is excluded because its D labels are derived, and E25 never saw it). Their
players are low-passed with sigma 8 (E18) to look like the vendor-smoothed BAR-ATH players. No match
data and no public GT are read: the match contributes only its tracked players and its ball presence.
No model artefact is committed; ``build_model`` retrains from the external data.

:func:`footprint` / :func:`check_footprint` are the label-free build guard (no public GT).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

from . import external

ENABLED = True                  # master switch: when False make_submission ships plain Baseline+ (ingap off too)
SEED = 0
FPS, STRIDE = 25, 5
HZ = FPS // STRIDE
TF = 50                         # min ball-missing run, frames (2 s)
Q = 0.5                         # gap-mean P(D) threshold
ABSORB = 100                    # H/A islands shorter than this between D stretches -> D (4 s)
LOWPASS_SIGMA = 8.0             # external players, frames (E18)
TRAIN_PREFIXES = ("idsse", "metrica")
MAX_D_SHARE_ADDED = 0.5         # guard: the gate may add D to at most 50% of graded frames (measured 0.36)
MIN_D_SHARE_ADDED = 0.10        # ... and at least 10%: a dead gate is as suspect as a runaway one


def _runs(mask):
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


# ---------------------------------------------------------------- player-only features (E25 feat.py)

def _nanstat(f, x, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(x, axis=axis)


def _centroid_speed(P, half):
    """All-player centroid speed (m/s) over +-0.5 s within each half, at full frame rate."""
    allx = _nanstat(np.nanmean, P[..., 0])
    ally = _nanstat(np.nanmean, P[..., 1])
    cs = np.full(len(P), np.nan)
    k = FPS // 2
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx) < 2 * k + 1:
            continue
        a = idx[k:-k]
        cs[a] = np.hypot(allx[a + k] - allx[a - k], ally[a + k] - ally[a - k]) * FPS / (2 * k)
    return cs


def _frame_table(P, V, T):
    ok = np.isfinite(P[..., 0])
    v = np.where(ok, np.clip(np.nan_to_num(V, nan=0.0), 0, 12), np.nan)
    d = {}
    d["spd_mean"] = _nanstat(np.nanmean, v)
    d["spd_med"] = _nanstat(np.nanmedian, v)
    d["spd_q90"] = _nanstat(lambda a, axis: np.nanquantile(a, 0.9, axis=axis), v)
    d["still"] = _nanstat(np.nanmean, np.where(ok, v < 1.0, np.nan))
    d["run"] = _nanstat(np.nanmean, np.where(ok, v > 4.0, np.nan))
    sx, sy, cx = [], [], []
    for t in (0, 1):
        sel = T == t
        sx.append(_nanstat(np.nanstd, P[:, sel, 0]))
        sy.append(_nanstat(np.nanstd, P[:, sel, 1]))
        cx.append(_nanstat(np.nanmean, P[:, sel, 0]))
    d["spread_x"] = (sx[0] + sx[1]) / 2
    d["spread_y"] = (sy[0] + sy[1]) / 2
    d["spread_y_absdiff"] = np.abs(sy[0] - sy[1])
    d["team_dx"] = np.abs(cx[0] - cx[1])
    return pd.DataFrame(d)


def _window(df, half):
    parts = []
    for h in np.unique(half[half > 0]):
        sub = df[half == h]
        cols = {}
        for s in (3, 8):
            r = sub.rolling(s * HZ + 1, center=True, min_periods=1).mean()
            cols.update({"{}_w{}".format(c, s): r[c].to_numpy() for c in sub.columns})
        t = 3 * HZ
        x = sub["spd_mean"]
        fut = x[::-1].rolling(t, min_periods=1).mean()[::-1].shift(-1)
        past = x.rolling(t, min_periods=1).mean()
        cols["spd_trend"] = (fut - past).to_numpy()
        m1 = x.rolling(HZ + 1, center=True, min_periods=1).mean()
        cols["spd_min8"] = m1.rolling(8 * HZ + 1, center=True, min_periods=1).min().to_numpy()
        parts.append(pd.DataFrame(cols, index=sub.index))
    return pd.concat(parts).reindex(df.index)


def sample_table(P, V, T, half):
    """5 Hz player-only feature rows over graded frames: (frame indices, DataFrame)."""
    rows = np.flatnonzero(half > 0)[::STRIDE]
    ft = _frame_table(P[rows], V[rows], T)
    ft["centroid_spd"] = _centroid_speed(P, half)[rows]
    return rows, _window(ft, half[rows]).astype(np.float32)


# ---------------------------------------------------------------- training and inference

def _lowpass_players(P, V, half, sigma):
    """Gaussian low-pass of every player track (positions and speeds) within each half and finite run."""
    def lp(X):
        out = np.array(X, np.float64)
        finite = np.isfinite(out.reshape(len(out), -1)).all(axis=1)
        for h in np.unique(half[half > 0]):
            idx = np.flatnonzero(half == h)
            a, b = idx[0], idx[-1] + 1
            for s, e in _runs(finite[a:b]) + a:
                out[s:e] = gaussian_filter1d(out[s:e], sigma, axis=0, mode="nearest")
        return out
    P2, V2 = P.copy(), V.copy()
    for s in range(P.shape[1]):
        P2[:, s] = lp(P[:, s])
        V2[:, s] = lp(np.where(np.isnan(P[:, s, 0]), np.nan, V[:, s]))
    return P2, V2


def _classifier():
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=150,
                                          min_samples_leaf=300, l2_regularization=5.0, random_state=SEED)


def training_games():
    return [g for g in external.list_games() if g.startswith(TRAIN_PREFIXES)]


def build_model(game_ids=None):
    """Train P(D) on external games only; returns ``(model, columns)``. Deterministic."""
    Xs, ys = [], []
    for gid in (game_ids or training_games()):
        g = external.load_game(gid)
        mm = g.match
        P, V = _lowpass_players(mm.P, mm.V, mm.half, LOWPASS_SIGMA)
        rows, X = sample_table(P, V, mm.T, mm.half)
        ref = np.asarray(g.ref)[rows]
        keep = np.isin(ref, ["H", "A", "D"])
        Xs.append(X[keep])
        ys.append((ref[keep] == "D").astype(int))
    X = pd.concat(Xs)
    cols = list(X.columns)
    return _classifier().fit(X.to_numpy(), np.concatenate(ys)), cols


def frame_pd(model, cols, P, V, T, half):
    """Per-frame P(D) (NaN outside the halves): 5 Hz predictions interpolated inside each half."""
    rows, X = sample_table(P, V, T, half)
    p5 = model.predict_proba(X[cols].to_numpy())[:, 1]
    pf = np.full(len(half), np.nan)
    for h in (1, 2):
        sel = half[rows] == h
        if sel.sum() < 2:
            continue
        fr = np.flatnonzero(half == h)
        pf[fr] = np.interp(fr, rows[sel], p5[sel])
    return pf


# ---------------------------------------------------------------- the gate

def absorb_islands(lab, max_len):
    """Merge H/A runs shorter than ``max_len`` frames that have D on both sides into D."""
    out = np.asarray(lab).copy()
    for a, b in _runs(np.isin(out, ("H", "A"))):
        if b - a < max_len and a > 0 and b < len(out) and out[a - 1] == "D" and out[b] == "D":
            out[a:b] = "D"
    return out


def gate(labels, B, half, pf, tf=TF, q=Q, absorb=ABSORB):
    """Returns ``(gated labels, dead)``: ``dead`` marks the gap frames the gate fired on."""
    out = np.asarray(labels).copy()
    dead = np.zeros(len(out), bool)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        lo, hi = idx[0], idx[-1] + 1
        for a, b in _runs(~np.isfinite(B[lo:hi, 0])):
            a, b = a + lo, b + lo
            if b - a >= tf and np.nanmean(pf[a:b]) > q:
                out[a:b] = "D"
                dead[a:b] = True
    return absorb_islands(out, absorb), dead


def footprint(before, after, half):
    """Label-free change footprint of the gate over graded frames (half > 0)."""
    before, after = np.asarray(before), np.asarray(after)
    graded = np.asarray(half) > 0
    n = int(graded.sum())
    changed = int(((before != after) & graded).sum())
    d0, d1 = int(((before == "D") & graded).sum()), int(((after == "D") & graded).sum())
    return dict(graded=n, changed=changed, changed_share_graded=changed / max(1, n),
                d_before=d0, d_after=d1, d_added_share_graded=(d1 - d0) / max(1, n))


def check_footprint(fp):
    """Raise if the gate added a runaway or a negligible share of D (bounds from Q20 / E44, not public GT)."""
    s = fp["d_added_share_graded"]
    if not MIN_D_SHARE_ADDED <= s <= MAX_D_SHARE_ADDED:
        raise RuntimeError("dgate added D to %.3f of graded frames, outside [%.2f, %.2f]; "
                           "is the P(D) model broken?" % (s, MIN_D_SHARE_ADDED, MAX_D_SHARE_ADDED))
    return fp


def apply(labels, match, model=None):
    """Train (unless ``model`` given), score the match's players, gate, guard; returns ``(labels, dead, fp)``."""
    mdl, cols = model or build_model()
    pf = frame_pd(mdl, cols, match.P, match.V, match.T, match.half)
    out, dead = gate(labels, match.B, match.half, pf)
    return out, dead, check_footprint(footprint(labels, out, match.half))
