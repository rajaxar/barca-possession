"""Ball-free possession classifier (E04): P(home has the ball | team shape and motion over a window).

Only ball-independent columns of :mod:`possession.features` are used (team centroid, spread, bounding
box, defensive line, goalkeeper, speed, pressing), so the model is unaffected by ball gaps and by the
vendor ball-channel leaks (FINDINGS S05). Features are sampled at 5 Hz, windowed per half (1 s, 3 s
and 8 s centered means plus a +-3 s trend), and assembled team-symmetrically: the "own" team is the
one whose probability is asked for, so swapping home/away negates differences and flips the label.
Training uses both orientations and prediction averages them, which makes ``P(H) + P(A) = 1`` exact.

Outliers (bbox ratio up to 18000, stray GK x, FINDINGS S07) are handled inside the model: log
transform plus quantile clipping for the logistic model, tree splits for HistGradientBoosting.

``run_logo(games)`` is the single entry point for the external leave-one-game-out experiment; pass
degraded games (S06) instead of ``external.load_games()`` and nothing else changes. ``predict_frames``
returns per-frame P(H) at 25 Hz for any Match. No public GT is read anywhere in this module.
"""
import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as feature_lib

SEED = 0
FPS = 25
STRIDE = 5                      # 5 Hz sampling
HZ = FPS // STRIDE
WINDOWS_S = (1, 3, 8)           # centered mean windows, seconds
TREND_S = 3                     # trend = mean(next 3 s) - mean(previous 3 s)
PITCH = (105.0, 68.0)           # externals are meter-scaled to ~105 x 68

# ball-free per-team columns of features.features(); {t} is home/away
TEAM_COLS = ("tracked_count", "centroid_attack_x_m", "centroid_y_m", "spread_x_m", "spread_y_m",
             "bbox_area_m2", "defensive_line_attack_x_m", "gk_attack_x_m", "mean_speed_mps",
             "pressing_mps")
FORBIDDEN_PREFIXES = ("ball_", "home_ball", "away_ball", "camera", "attacker_tag", "set_piece")


def base_table(f):
    """Ball-free per-team arrays {own: {...}, opp: {...}} from a features frame (home = own)."""
    out = {}
    for side, team in (("h", "home"), ("a", "away")):
        d = {}
        for c in TEAM_COLS:
            name = "{}_{}".format(team, c)
            assert not name.startswith(FORBIDDEN_PREFIXES)
            d[c] = f[name].to_numpy(np.float64)
        d["bbox_area_m2"] = np.log1p(np.clip(d["bbox_area_m2"], 0, 6000))
        d["gk_attack_x_m"] = np.clip(d["gk_attack_x_m"], -60, 0)
        out[side] = d
    return out


def assemble(base, swap):
    """Team-symmetric per-frame columns with own = home (or away when ``swap``)."""
    own, opp = (base["a"], base["h"]) if swap else (base["h"], base["a"])
    cols = {}
    for c in TEAM_COLS:
        cols["own_" + c] = own[c]
        cols["opp_" + c] = opp[c]
        cols["diff_" + c] = own[c] - opp[c]
    # shared-frame gaps: both centroids/lines are in own-attack coordinates, so the sum is the gap
    cols["gap_centroid_x"] = own["centroid_attack_x_m"] + opp["centroid_attack_x_m"]
    cols["gap_line_x"] = own["defensive_line_attack_x_m"] + opp["defensive_line_attack_x_m"]
    return cols


def _window(df, half):
    """Centered rolling means and +-TREND_S trend per half, on a 5 Hz frame."""
    parts = []
    for h in np.unique(half[half > 0]):
        sub = df[half == h]
        cols = {}
        for s in WINDOWS_S:
            r = sub.rolling(s * HZ + 1, center=True, min_periods=1).mean()
            cols.update({"{}_w{}".format(c, s): r[c].to_numpy() for c in sub.columns})
        t = TREND_S * HZ
        fut = sub[::-1].rolling(t, min_periods=1).mean()[::-1].shift(-1)
        past = sub.rolling(t, min_periods=1).mean()
        cols.update({"{}_trend".format(c): (fut[c] - past[c]).to_numpy() for c in sub.columns})
        parts.append(pd.DataFrame(cols, index=sub.index))
    return pd.concat(parts).reindex(df.index)


def sample_table(match, *, fps=FPS, pitch=PITCH, cache_key=None):
    """5 Hz windowed ball-free design matrices (original and home/away-swapped) for one match.

    Returns (rows, X, Xswap) where ``rows`` are the frame indices of the 5 Hz samples
    (frames inside a half only) and X/Xswap are DataFrames with identical columns.
    """
    f = feature_lib.features(match, fps=fps, pitch=pitch, cache_key=cache_key)
    rows = np.flatnonzero(match.half > 0)[::STRIDE]
    base = base_table(f.iloc[rows])
    half = match.half[rows]
    X = _window(pd.DataFrame(assemble(base, False)), half)
    Xs = _window(pd.DataFrame(assemble(base, True)), half)
    return rows, X.astype(np.float32), Xs.astype(np.float32)


class RobustClip(BaseEstimator, TransformerMixin):
    """Quantile-clip each column (fit on train), then median-impute."""

    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, np.float64)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        self.med_ = np.nanmedian(np.clip(X, self.lo_, self.hi_), axis=0)
        self.med_ = np.nan_to_num(self.med_)
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, np.float64), self.lo_, self.hi_)
        return np.where(np.isfinite(X), X, self.med_)


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(RobustClip(), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=500, random_state=SEED))
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=150, min_samples_leaf=300,
            l2_regularization=5.0, random_state=SEED)
    raise ValueError(kind)


def fit(tables, kind="hgb"):
    """tables: iterable of (X, Xswap, y) with y = 1 when home owns (H), 0 when away (A).

    Degenerate design columns (all-NaN, or a single distinct value) are dropped before the
    estimator sees them and recorded in ``model.keep_`` / ``model.columns_``; :func:`predict`
    applies the same selection. This is inert unless a training table actually contains such a
    column -- on the ten external games every one of the 128 columns takes at least two values --
    but without it an all-NaN column crashes the HGB binner ("window shape cannot be larger than
    input array shape"), which is exactly what a new dataset with a missing (e.g. goalkeeper)
    channel would look like.
    """
    X = pd.concat([t for tb in tables for t in (tb[0], tb[1])])
    y = np.concatenate([np.r_[tb[2], 1 - tb[2]] for tb in tables])
    keep = X.nunique(dropna=False).to_numpy() > 1
    if not keep.any():
        raise ValueError("every design column is constant or all-NaN; nothing to fit")
    model = clone(make_model(kind)).fit(X.to_numpy()[:, keep], y)
    model.keep_ = keep
    model.columns_ = np.asarray(X.columns)[keep]
    return model


def _matrix(model, X):
    """Numeric matrix with the training-time column selection of ``model`` applied."""
    a = X.to_numpy()
    keep = getattr(model, "keep_", None)
    if keep is None:  # a model pickled before the degenerate-column guard
        return a
    if a.shape[1] != len(keep):
        raise ValueError("design matrix has {} columns, model was fitted on {}".format(a.shape[1], len(keep)))
    return a[:, keep]


def predict(model, X, Xswap):
    """Antisymmetrised P(H) for the sampled rows."""
    Xa, Xb = _matrix(model, X), _matrix(model, Xswap)
    return 0.5 * (model.predict_proba(Xa)[:, 1] + 1 - model.predict_proba(Xb)[:, 1])


def to_frames(rows, p, n_frames, half):
    """Linear interpolation of 5 Hz P(H) to every frame of each half; NaN outside halves."""
    out = np.full(n_frames, np.nan)
    for h in (1, 2):
        sel = half[rows] == h
        if sel.sum() < 2:
            continue
        fr = np.flatnonzero(half == h)
        out[fr] = np.interp(fr, rows[sel], p[sel])
    return out


def predict_frames(model, match, **kw):
    """Per-frame P(H) at 25 Hz for a Match (NaN outside halves)."""
    rows, X, Xs = sample_table(match, **kw)
    return to_frames(rows, predict(model, X, Xs), len(match.B), match.half)


def game_table(game, cache_prefix="e04"):
    """(rows, X, Xswap, y, alive) for a Game; y is 1/0 for H/A rows and -1 elsewhere."""
    key = None if cache_prefix is None else "{}-{}".format(cache_prefix, game.name.replace(":", "-"))
    rows, X, Xs = sample_table(game.match, cache_key=key)
    ref = np.asarray(game.ref)[rows]
    y = np.where(ref == "H", 1, np.where(ref == "A", 0, -1))
    return rows, X, Xs, y


def _metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return dict(n=int(len(y)), acc=accuracy_score(y, p > 0.5), logloss=log_loss(y, p),
                auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"))


def flip_sign_accuracy(ref, p, margin_s=5.0, span_s=5.0):
    """At each true H<->A change between consecutive alive spells, does mean P(H) over the ``span_s``
    after the change sit on the new team's side of the mean over the ``span_s`` before it?
    ``ref`` and ``p`` are per-sample (5 Hz). Returns (n_changes, fraction correct)."""
    k = int(span_s * HZ)
    live = np.flatnonzero(np.isin(ref, ("H", "A")))
    ok = n = 0
    prev = None
    for i in live:
        if prev is not None and ref[i] != ref[prev] and i - prev <= HZ * margin_s:
            a, b = p[max(i - k, 0):i], p[i:i + k]
            if np.isfinite(a).any() and np.isfinite(b).any():
                shift = np.nanmean(b) - np.nanmean(a)
                n += 1
                ok += (shift > 0) == (ref[i] == "H")
        prev = i
    return n, (ok / n if n else float("nan"))


def run_logo(games, kinds=("logreg", "hgb"), cache_prefix="e04"):
    """Leave-one-game-out over ``games`` (iterable of Game; pass degraded games here).

    Returns (summary DataFrame per game and model, predictions dict game -> kind -> P(H) per 5 Hz row,
    importance DataFrame from held-out permutation of the HGB model).
    """
    tabs = {}
    for g in games:
        rows, X, Xs, y = game_table(g, cache_prefix)
        keep = y >= 0
        tabs[g.name] = dict(rows=rows, X=X, Xs=Xs, y=y, keep=keep, ref=np.asarray(g.ref)[rows],
                            half=g.match.half[rows])
    names = sorted(tabs)
    recs, preds = [], {}
    for held in names:
        train = [(t["X"][t["keep"]], t["Xs"][t["keep"]], t["y"][t["keep"]])
                 for n, t in tabs.items() if n != held]
        t = tabs[held]
        preds[held] = {}
        for kind in kinds:
            model = fit(train, kind)
            p = predict(model, t["X"], t["Xs"])
            preds[held][kind] = p
            m = _metrics(t["y"][t["keep"]], p[t["keep"]])
            m["flips"], m["flip_acc"] = flip_sign_accuracy(t["ref"], p)
            recs.append(dict(game=held, model=kind, **m))
    return pd.DataFrame(recs), preds, tabs


def _stem(col):
    col = re.sub(r"_(w\d+|trend)$", "", col)
    return re.sub(r"^(own|opp|diff)_", "", col)


def permutation_importance(tabs, held, n_repeat=3):
    """Held-out permutation importance (logloss increase) of the HGB model for one game, grouped by
    base feature (own/opp/diff and every window of it are permuted together)."""
    rng = np.random.RandomState(SEED)
    train = [(t["X"][t["keep"]], t["Xs"][t["keep"]], t["y"][t["keep"]])
             for n, t in tabs.items() if n != held]
    model = fit(train, "hgb")
    t = tabs[held]
    X, Xs, y = t["X"][t["keep"]], t["Xs"][t["keep"]], t["y"][t["keep"]]
    base = log_loss(y, np.clip(predict(model, X, Xs), 1e-4, 1 - 1e-4))
    out = {}
    for stem in sorted({_stem(c) for c in X.columns}):
        sel = [c for c in X.columns if _stem(c) == stem]
        inc = []
        for _ in range(n_repeat):
            perm = rng.permutation(len(X))
            Xp, Xsp = X.copy(), Xs.copy()
            Xp[sel] = X[sel].to_numpy()[perm]
            Xsp[sel] = Xs[sel].to_numpy()[perm]
            inc.append(log_loss(y, np.clip(predict(model, Xp, Xsp), 1e-4, 1 - 1e-4)) - base)
        out[stem] = float(np.mean(inc))
    return pd.Series(out).sort_values(ascending=False)
