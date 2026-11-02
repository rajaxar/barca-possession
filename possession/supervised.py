"""Supervised frame model for H/A/D (E10): two gradient-boosted heads trained on degraded external games.

**Status: not shipped.**  ``make_submission`` does not import this module.  The card's verdict
(``research/E10_result.md``) decides whether it becomes a lever (D-010 tier 2) or is dropped; a future
adoption card would wire it in behind a flag, not here.

Model
-----
Two ``HistGradientBoostingClassifier`` heads on 5 Hz windowed features (1/3/8 s centered means plus a
+-3 s trend), the same windows :mod:`possession.ballfree` uses:

* **D head** — ``P(D | H/A/D row)`` from symmetric, orientation-free features (player shape and motion
  from :func:`possession.dgate.sample_table`, plus the ball blocks the variant allows).
* **H/A head** — ``P(H | live row)``, trained on both home/away orientations and averaged at prediction
  time, which makes ``P(H) + P(A) = 1`` exact (as in :mod:`possession.ballfree`).

Three feature variants (pre-registered, E10 card):

``bf``    ball-free: player team shape + D physics only.  No ball column of any kind, no missingness.
``prox``  ``bf`` + ball-player proximity windows (nearest distance, counts within 2/4/8 m, time to
          ball) + ball presence / missing-run length.  No ball kinematics.
``full``  ``prox`` + ball speed / acceleration / direction change / out-of-pitch windows.  Ball-kinematic
          cues are inflated on the degraded practice data (D-022 standing discount): treat as an upper
          bound.

Every column is label-free.  Training reads external games only: no part of this module reads our
match, and no public ground truth is read anywhere.  For the practice set the external games are
degraded with :func:`possession.degrade.degrade_game` at the frozen dev-v1 parameters, because the
point of the model is to match *our* tracker's noise; this is why a build costs minutes to hours and
why the module is not on the submission path.

Determinism: fixed game order, fixed seeds, fixed row subsample (1 in ``SUB`` rows of every cell).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import ballfree, dgate
from . import features as feature_lib

FPS = 25
STRIDE = 5
HZ = FPS // STRIDE
WIN = (25, 75, 200)             # 1 s, 3 s, 8 s centered, frames
TREND = 75                      # +-3 s trend, frames
SEED = 0
SUB = 6                         # pre-registration: fixed random 1/6 of every cell's rows
# dev-v1 grid (D-022): (knob d_stationarity, v d_ball_visible); all rows of a game are used for training.
# Order matches stage A's per-game file order (seed, knob, then v with v=0 last) so the fixed 1/SUB row
# subsample is the same one the pre-registered LOGO folds used.
CELLS = tuple((k, v) for k in (0.0, 1.0) for v in (0.05, 0.12, 0.23, 0.30, 0.50, 0.0))
SEEDS = (0, 1, 2)

# Ball-player *kinematics* live in the "prox" block for convenience but are excluded from the prox
# variant, which is geometry + presence only.  They return in "full" with the absolute ball kinematics.
_KIN_REL = ("closing", "_rel")
VARIANTS = {
    "bf": dict(ha=("team",), d=("dphys",)),
    "prox": dict(ha=("team", "prox", "miss"), d=("dphys", "dprox", "miss")),
    "full": dict(ha=("team", "prox", "miss", "kin"), d=("dphys", "dprox", "miss", "kin")),
}
DEFAULT_VARIANT = "bf"           # D-022: prefer ball-free evidence when candidates tie


# ---------------------------------------------------------------- feature tables (label-free)

def _wmean(x, lo, hi, rows, a, b):
    """NaN-aware mean of ``x`` over ``[rows+a, rows+b)`` clipped to each row's half ``[lo, hi)``."""
    fin = np.isfinite(x)
    s_cum = np.r_[0.0, np.cumsum(np.where(fin, x, 0.0))]
    c_cum = np.r_[0, np.cumsum(fin)]
    s = np.clip(rows + a, lo, hi)
    e = np.clip(rows + b, lo, hi)
    n = c_cum[e] - c_cum[s]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (s_cum[e] - s_cum[s]) / np.maximum(n, 1), np.nan)


def _win_cols(name, x, lo, hi, rows, wins=WIN, trend=True, extra=()):
    out = {}
    for w, tag in zip(wins, ("w1", "w3", "w8")):
        out["%s_%s" % (name, tag)] = _wmean(x, lo, hi, rows, -(w // 2), w // 2 + 1)
    for w in extra:
        out["%s_w%d" % (name, w // 25)] = _wmean(x, lo, hi, rows, -(w // 2), w // 2 + 1)
    if trend:
        out["%s_trend" % name] = (_wmean(x, lo, hi, rows, 1, TREND + 1)
                                  - _wmean(x, lo, hi, rows, -TREND, 0))
    return out


def _missing_runs(present, half):
    """log1p seconds: containing missing-run length, time since last present, time to next present."""
    n = len(present)
    runlen, since, until = np.zeros(n), np.zeros(n), np.zeros(n)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        a0, b0 = idx[0], idx[-1] + 1
        miss = ~present[a0:b0]
        d = np.diff(np.r_[0, miss.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            length = e - s
            k = np.arange(length)
            runlen[a0 + s:a0 + e] = length
            since[a0 + s:a0 + e] = k + 1
            until[a0 + s:a0 + e] = length - k
    return np.log1p(runlen / FPS), np.log1p(since / FPS), np.log1p(until / FPS)


def feature_tables(match, pitch=ballfree.PITCH, cache_key=None):
    """5 Hz windowed design matrices for one match.

    Returns ``dict(rows, half, cols_ha, X_ha, grp_ha, cols_d, X_d, grp_d)``; ``X_ha`` is in the
    home-is-own orientation (use :func:`swap` for the away orientation).  Pure function of the match.
    """
    half = np.asarray(match.half)
    key2 = None if cache_key is None else cache_key + "-bf"
    f = feature_lib.features(match, pitch=pitch, cache_key=cache_key)
    rows = np.flatnonzero(half > 0)[::STRIDE]
    hlo = np.zeros(len(half), int)
    hhi = np.zeros(len(half), int)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if len(idx):
            hlo[idx] = idx[0]
            hhi[idx] = idx[-1] + 1
    lo, hi = hlo[rows], hhi[rows]
    g = lambda c: f[c].to_numpy(np.float64)

    # TEAM block (ballfree) and D-phys block (dgate), exactly as the shipped models build them.
    _, X, _ = ballfree.sample_table(match, pitch=pitch, cache_key=key2)
    team = X.copy()
    _, dphys = dgate.sample_table(match.P, match.V, match.T, half)

    # Ball-player proximity block (own/opp/diff where a side matters).
    prox = {}
    for side, t in (("own", "home"), ("opp", "away")):
        for c, cc in (("near1_dist", "ball_near1_dist_m"), ("near2_dist", "ball_near2_dist_m"),
                      ("near1_closing", "ball_near1_closing_mps"), ("near1_rel", "ball_near1_rel_speed_mps"),
                      ("cnt2", "ball_count_2m"), ("cnt4", "ball_count_4m"), ("cnt8", "ball_count_8m"),
                      ("ttb", "ball_time_to_ball_s")):
            prox["%s_b_%s" % (side, c)] = g("%s_%s" % (t, cc))
    for c in ("near1_dist", "cnt4"):
        prox["diff_b_%s" % c] = prox["own_b_%s" % c] - prox["opp_b_%s" % c]
    nearest = g("ball_nearest_team")
    prox["nearest_own"] = np.where(np.isfinite(nearest), (nearest == 0).astype(float), np.nan)
    prox_cols = {}
    for k, x in prox.items():
        prox_cols.update(_win_cols(k, x, lo, hi, rows))

    # Ball presence / missingness block (symmetric).
    present = np.isfinite(np.asarray(match.B)[:, 0])
    rl, sn, un = _missing_runs(present, half)
    miss = _win_cols("present", present.astype(float), lo, hi, rows, trend=False, extra=(750,))
    miss["miss_runlen"] = rl[rows]
    miss["miss_since"] = sn[rows]
    miss["miss_until"] = un[rows]

    # Absolute ball kinematics (symmetric) — the D-022-discounted block.
    kin = {}
    for c in ("ball_speed_mps", "ball_accel_mps2", "ball_direction_change_deg", "ball_out_of_pitch"):
        kin.update(_win_cols("kin_" + c, g(c), lo, hi, rows))

    # Symmetric ball-player geometry for the D head (both teams pooled: min distance, total counts).
    sym = {}
    sym.update(_win_cols("dball_near", np.fmin(g("home_ball_near1_dist_m"), g("away_ball_near1_dist_m")),
                         lo, hi, rows))
    for r in (2, 4, 8):
        sym.update(_win_cols("dball_cnt%d" % r,
                             g("home_ball_count_%dm" % r) + g("away_ball_count_%dm" % r), lo, hi, rows))

    def cat(*blocks):
        names, grp, arrs = [], {}, []
        for gname, b in blocks:
            if isinstance(b, pd.DataFrame):
                cols, arr = list(b.columns), b.to_numpy(np.float32)
            else:
                cols, arr = list(b.keys()), np.stack([np.asarray(v) for v in b.values()], 1).astype(np.float32)
            grp[gname] = np.arange(len(names), len(names) + len(cols))
            names += cols
            arrs.append(arr)
        return names, np.concatenate(arrs, 1), grp

    cols_ha, X_ha, grp_ha = cat(("team", team), ("prox", prox_cols), ("miss", miss), ("kin", kin))
    cols_d, X_d, grp_d = cat(("dphys", dphys), ("dprox", sym), ("miss", miss), ("kin", kin))
    return dict(rows=rows, half=half[rows], cols_ha=cols_ha, X_ha=X_ha, grp_ha=grp_ha,
                cols_d=cols_d, X_d=X_d, grp_d=grp_d)


def swap_plan(cols):
    """``(perm, sign, shift)`` with ``swap(X) = X[:, perm] * sign + shift`` (home <-> away)."""
    idx = {c: i for i, c in enumerate(cols)}
    perm = np.arange(len(cols))
    sign = np.ones(len(cols), np.float32)
    shift = np.zeros(len(cols), np.float32)
    for i, c in enumerate(cols):
        if c.startswith("own_"):
            perm[i] = idx["opp_" + c[4:]]
        elif c.startswith("opp_"):
            perm[i] = idx["own_" + c[4:]]
        elif c.startswith("diff_"):
            sign[i] = -1
        elif c.startswith("nearest_own"):
            sign[i] = -1
            shift[i] = 1
    return perm, sign, shift


def swap(X, plan):
    perm, sign, shift = plan
    return X[:, perm] * sign + shift


def columns(meta, head, variant):
    """Column indices used by ``variant`` for ``head`` ('ha' or 'd')."""
    v = VARIANTS[variant]
    g = meta["grp_" + head]
    cols = np.sort(np.concatenate([g[k] for k in v[head]]))
    if variant == "prox":                       # geometry/presence only (pre-registration)
        names = meta["cols_" + head]
        cols = np.array([i for i in cols if not any(k in names[i] for k in _KIN_REL)], dtype=int)
    return cols


# ---------------------------------------------------------------- model

def _hgb(seed):
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=150,
                                          min_samples_leaf=300, l2_regularization=5.0, random_state=seed)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _concat(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in ("X_ha", "X_d", "y")}


class Model:
    """Fitted pair of heads plus their Platt maps.  ``variant`` is recorded for provenance."""

    def __init__(self, meta, variant, seed=SEED):
        self.variant, self.seed = variant, seed
        v = VARIANTS[variant]
        self.cd = columns(meta, "d", variant)
        self.cha = columns(meta, "ha", variant)
        self.plan = swap_plan([meta["cols_ha"][i] for i in self.cha])
        self.d_ab, self.ha_a = (1.0, 0.0), 1.0
        self.md = self.mh = None

    # -- rows
    def _d_xy(self, S):
        k = S["y"] >= 0
        return S["X_d"][k][:, self.cd], (S["y"][k] == 2).astype(int)

    def _ha_xy(self, S):
        k = (S["y"] == 0) | (S["y"] == 1)
        return S["X_ha"][k][:, self.cha], (S["y"][k] == 0).astype(int)

    def _fit(self, S):
        X, y = self._d_xy(S)
        md = _hgb(self.seed).fit(X, y)
        X, y = self._ha_xy(S)
        Xs = swap(X, self.plan)
        mh = _hgb(self.seed).fit(np.concatenate([X, Xs]), np.r_[y, 1 - y])
        return md, mh

    def _raw(self, X_d, X_ha):
        pd_ = self.md.predict_proba(X_d[:, self.cd])[:, 1]
        X = X_ha[:, self.cha]
        ph = 0.5 * (self.mh.predict_proba(X)[:, 1] + 1 - self.mh.predict_proba(swap(X, self.plan))[:, 1])
        return pd_, ph

    def fit(self, tables, calibration=None):
        """Fit on ``tables``; if ``calibration`` is given, fit Platt maps on it (never on its own fit
        data), then refit both heads on ``tables + calibration`` (pre-registration)."""
        if calibration:
            self.md, self.mh = self._fit(_concat(tables))
            C = _concat(calibration)
            pd_, ph = self._raw(C["X_d"], C["X_ha"])
            k = C["y"] >= 0
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1e4, max_iter=200).fit(
                _logit(pd_[k])[:, None], (C["y"][k] == 2).astype(int))
            self.d_ab = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
            kk = (C["y"] == 0) | (C["y"] == 1)
            lr2 = LogisticRegression(C=1e4, max_iter=200, fit_intercept=False).fit(
                _logit(ph[kk])[:, None], (C["y"][kk] == 0).astype(int))
            self.ha_a = float(lr2.coef_[0, 0])
            tables = list(tables) + list(calibration)
        self.md, self.mh = self._fit(_concat(tables))
        return self

    def predict(self, X_d, X_ha, calibrated=True):
        """Per-row ``(P(D), P(H))``.  Uncalibrated output is the raw head probability."""
        pd_, ph = self._raw(X_d, X_ha)
        if calibrated:
            a, b = self.d_ab
            pd_ = 1 / (1 + np.exp(-(a * _logit(pd_) + b)))
            ph = 1 / (1 + np.exp(-(self.ha_a * _logit(ph))))
        return pd_, ph


def fit(tables, variant=DEFAULT_VARIANT, calibration=None, seed=SEED, meta=None):
    """Fit a :class:`Model` on ``tables`` (each a dict with X_ha, X_d, y and the column meta)."""
    return Model(meta or tables[0], variant, seed).fit(tables, calibration)


# ---------------------------------------------------------------- inference on a match

def frame_probs(model, match, pitch=None, chunk=1 << 16):
    """Per-frame ``(P(D), P(H))`` at 25 Hz (float32, NaN outside the halves).  Deterministic."""
    t = feature_tables(match, pitch=pitch or ballfree.PITCH)
    pd5, ph5 = model.predict(t["X_d"], t["X_ha"])
    n = len(match.half)
    pd = np.full(n, np.nan, dtype=np.float32)
    ph = np.full(n, np.nan, dtype=np.float32)
    for h in (1, 2):
        sel = t["half"] == h
        if sel.sum() < 2:
            continue
        fr = np.flatnonzero(match.half == h)
        pd[fr] = np.interp(fr, t["rows"][sel], pd5[sel])
        ph[fr] = np.interp(fr, t["rows"][sel], ph5[sel])
    return pd, ph


# ---------------------------------------------------------------- training data from externals

def subsample(table, index, sub=SUB):
    """Fixed random 1/``sub`` of a cell's rows (seeded by the cell index), as in the pre-registration."""
    rng = np.random.RandomState(1000 * int(index[0]) + int(index[1]))
    keep = np.flatnonzero(rng.rand(len(table["y"])) < 1.0 / sub)
    return {k: table[k][keep] for k in ("X_ha", "X_d", "y")}


def degraded_tables(gid, cells=CELLS, seeds=SEEDS, params=None, cache=True, game_index=0):
    """Training tables for one external game from its degraded copies (dev-v1 params, label-free).

    ``cells`` are ``(knob, v)``; the reference labels come from the *clean* game, as in dev-v1.
    ``game_index`` is the game's index in the caller's training list; it seeds the fixed 1/SUB row
    subsample together with the cell index (the pre-registered fold recipe).  This is the expensive
    step: one call degrades and featurises ``len(seeds) * len(cells)`` copies.
    """
    from . import degrade, evaluate, external
    clean = external.load_game(gid)
    ref = evaluate.to_codes(clean.ref)
    if params is None:
        params = degrade.DEV_V1_PARAMS
    out = []
    for si, seed in enumerate(seeds):
        for ci, (knob, v) in enumerate(cells):
            m = degrade.degrade_game(clean, seed=seed, d_ball_visible=v, d_stationarity=knob,
                                     params=params).match
            key = None if not cache else "e10-%s-s%d-k%g-v%g" % (gid.replace(":", "-"), seed, knob, v)
            t = feature_tables(m, pitch=external.pitch(gid), cache_key=key)
            t["y"] = ref[t["rows"]].astype(np.int8)
            s = subsample(t, (game_index, si * len(cells) + ci))
            s.update({k: t[k] for k in ("cols_ha", "cols_d", "grp_ha", "grp_d")})
            out.append(s)
    return out


def build(gids, variant=DEFAULT_VARIANT, calibration_gids=(), seed=SEED, **kw):
    """Train on every game in ``gids``; optional game-grouped Platt calibration on ``calibration_gids``.

    Never pass our match.  ``gids`` are dev-v1 game ids (``external.list_games()`` naming).
    """
    tables = [t for gi, gid in enumerate(gids) for t in degraded_tables(gid, game_index=gi, **kw)]
    calib = [t for gi, gid in enumerate(calibration_gids, start=len(gids))
             for t in degraded_tables(gid, game_index=gi, **kw)]
    return fit(tables, variant=variant, calibration=calib or None, seed=seed)
