"""dev-v1 harness (D-022): one frozen way to score and compare pipelines on the degraded practice set.

Frozen here (never edit to suit a method): the 10 games, degrader params, grid, seeds, bootstrap draws.
  games   7 IDSSE + 3 Metrica (``GAMES``); anything trained is leave-one-game-out over these 10.
  degrader ``degrade.degrade_game(clean, seed, d_ball_visible=v, d_stationarity=knob, params=DEV_V1_PARAMS)``, level 1,
           code = master 0adf48b (``DEGRADER_COMMIT``).
  grid    primary v {.05,.12,.23} x knob {0,1} (6 cells; headline = their mean), stress v {.30,.5} x knob {0,1},
          reference v=0 x knob {0,1} (never a headline); seeds 0-2. ``include_clean`` adds the undegraded game.
  stats   pooled spellF1, seed-mean inside each bootstrap draw, game-cluster CI (2000 draws, ONE fixed set of draws shared
          by every comparison), per-game deltas for the games-negative count, and the same cells with the reference
          shifted by +-0.5 s (E19: a win must hold at 0 and +-0.5 s).

A pipeline is a function ``pred_fn(cell) -> labels`` (H/A/D/X strings or 0..3 codes, one per frame). ``cell`` has
``game`` (id), ``clean`` (evaluate.Game, undegraded), ``match`` (degraded Match), ``pitch``, ``seed``, ``knob``, ``v``,
``memo`` (dict shared by every pipeline run on this cell) and ``models`` (shipped-stack leave-one-game-out models).
Built-in reference pipelines: ``smooth50``, ``baseline_plus``, ``gate``, ``ship`` (the pre-E66
gate+in-gap row, .4039, kept for history) and ``master`` (the D-028 shipped stack + floor, which
calls the real :func:`possession.pipeline.build_labels`).

    from possession import dev_v1
    res = dev_v1.score_dev_v1(my_fn)                    # headline, per-cell, stress, v=0 and shift rows
    print(res)
    cmp = dev_v1.compare_dev_v1(my_fn, "ship")          # either side: callable, builtin name or a saved result
    print(cmp)

The D-interval view (:func:`d_interval_score`, :func:`compare_d_intervals`, S42) is a
**diagnostic, never a selector** (D-022): spellF1 is the ranked objective. It answers "how many
stoppages, how long, how well matched, and which edge moved", and E07 showed it can move opposite
to spellF1 (the E66 add-ons lower interval F1 while raising spellF1). Never adopt a variant on it.

Not imported by make_submission. Developer tooling only: reads external games, never the match.
"""
import multiprocessing
import os
import pickle
import subprocess
import sys

import numpy as np

from . import config, evaluate, external

DEGRADER_COMMIT = "0adf48b"
GAMES = ("idsse:DFL-MAT-J03WMX", "idsse:DFL-MAT-J03WN1", "idsse:DFL-MAT-J03WOH", "idsse:DFL-MAT-J03WOY",
         "idsse:DFL-MAT-J03WPY", "idsse:DFL-MAT-J03WQQ", "idsse:DFL-MAT-J03WR9",
         "metrica:game1", "metrica:game2", "metrica:game3")
SEEDS = (0, 1, 2)
KNOBS = (0.0, 1.0)
PRIMARY_V = (0.05, 0.12, 0.23)
STRESS_V = (0.30, 0.5)
REFERENCE_V = (0.0,)
SHIFT_FRAMES = (-12, 0, 12)          # reference shift, 0.5 s at 25 Hz (positive = reference edges move later)
N_DRAWS = 2000
DRAW_SEED = 0
FPS = 25

GROUPS = {
    "primary": [(k, v) for k in KNOBS for v in PRIMARY_V],
    "knob0": [(0.0, v) for v in PRIMARY_V],
    "knob1": [(1.0, v) for v in PRIMARY_V],
    "stress": [(k, v) for k in KNOBS for v in STRESS_V],
    "v0_reference": [(k, 0.0) for k in KNOBS],
}
CELLS = tuple(kv for g in ("primary", "stress", "v0_reference") for kv in GROUPS[g])
_DRAWS = np.random.RandomState(DRAW_SEED).randint(0, len(GAMES), (N_DRAWS, len(GAMES)))
_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def params():
    from . import degrade
    return degrade.DEV_V1_PARAMS


def provenance():
    """One line for every log: where `possession` is imported from and its git hash (GUARDRAILS §4)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        h = subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        h = "unknown"
    return "dev_v1: possession=%s git=%s degrader=%s params=%s" % (here, h, DEGRADER_COMMIT, dict(params()))


# -- cache ---------------------------------------------------------------------------------------

def _dir():
    p = os.path.join(config.cache_dir(), "dev_v1")
    os.makedirs(p, exist_ok=True)
    return p


def _tag(game):
    return game.replace(":", "_")


def _atomic(path, write):
    tmp = config.tmp_path(path)
    write(tmp)
    os.replace(tmp, path)


def _stamp():
    return "%s-%s" % (DEGRADER_COMMIT, "-".join("%s%g" % kv for kv in sorted(params().items())))


def degraded_match(clean, seed, knob, v):
    """Degraded Match of `clean` (an evaluate.Game) at (seed, knob, v), cached. Players, half and gk do not depend on v
    (checked), so they are stored once per (game, seed, knob) and only B per v."""
    from . import data, degrade
    base = os.path.join(_dir(), "pl_%s_s%d_k%g_%s.npz" % (_tag(clean.name), seed, knob, _stamp()))
    bp_ = os.path.join(_dir(), "b_%s_s%d_k%g_v%g_%s.npy" % (_tag(clean.name), seed, knob, v, _stamp()))
    if os.path.isfile(base) and os.path.isfile(bp_):
        z = np.load(base)
        return data.Match(P=z["P"], V=z["V"], T=z["T"], B=np.load(bp_), BS=z["BS"], gk=z["gk"],
                          graded=z["graded"], half=z["half"])
    m = degrade.degrade_game(clean, seed=seed, d_ball_visible=v, d_stationarity=knob, params=params()).match
    if not os.path.isfile(base):
        _atomic(base, lambda t: np.savez(t, P=m.P, V=m.V, T=m.T, BS=m.BS, gk=m.gk, graded=m.graded, half=m.half))
    _atomic(bp_, lambda t: np.save(t, m.B))
    return m


# -- shipped-stack models, leave-one-game-out over the 10 -----------------------------------------

def _train_tables(gid):
    """(dgate (X, y), ballfree (X, Xs, y)) training rows of one clean game, exactly as the shipped build uses."""
    import joblib
    from . import ballfree, dgate
    p = os.path.join(_dir(), "tab_%s.pkl" % _tag(gid))
    if os.path.isfile(p):
        return joblib.load(p)
    g = external.load_game(gid)
    mm = g.match
    P, V = dgate._lowpass_players(mm.P, mm.V, mm.half, dgate.LOWPASS_SIGMA)
    rows, X = dgate.sample_table(P, V, mm.T, mm.half)
    ref = np.asarray(g.ref)[rows]
    keep = np.isin(ref, ["H", "A", "D"])
    _, Xb, Xs, y = ballfree.game_table(g)
    k = y >= 0
    out = ((X[keep], (ref[keep] == "D").astype(int)), (Xb[k], Xs[k], y[k]))
    _atomic(p, lambda t: joblib.dump(out, t))
    return out


_MODELS = {}


def shipped_models(held):
    """(dgate model, dgate columns, ballfree model) trained on the other 9 games, as the shipped build trains on all 10."""
    import joblib
    import pandas as pd
    from . import ballfree, dgate
    if held in _MODELS:
        return _MODELS[held]
    p = os.path.join(_dir(), "logo_%s.joblib" % _tag(held))
    if not os.path.isfile(p):
        tabs = {g: _train_tables(g) for g in GAMES if g != held}
        X = pd.concat([t[0][0] for t in tabs.values()])
        m = dgate._classifier().fit(X.to_numpy(), np.concatenate([t[0][1] for t in tabs.values()]))
        bf = ballfree.fit([t[1] for t in tabs.values()], "hgb")
        _atomic(p, lambda t: joblib.dump((m, list(X.columns), bf), t))
    _MODELS[held] = joblib.load(p)
    return _MODELS[held]


# -- cell and built-in pipelines -----------------------------------------------------------------

class Cell:
    """One (game, seed, knob, v) to predict. knob and v are None for the undegraded game."""

    def __init__(self, game, clean, match, seed, knob, v):
        self.game, self.clean, self.match, self.seed, self.knob, self.v = game, clean, match, seed, knob, v
        self.pitch = external.pitch(game)
        self.memo = {}

    @property
    def models(self):
        return shipped_models(self.game)


def _memo(cell, key, build):
    if key not in cell.memo:
        cell.memo[key] = build()
    return cell.memo[key]


def smooth50(cell):
    """Starter baseline: nearest player to the ball, majority-smoothed over 50 frames."""
    sys.path.insert(0, os.path.join(config.repo_dir(), "starter"))
    import baseline as starter
    m = cell.match
    d = np.linalg.norm(m.P - m.B[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(1)
    codes = np.where(np.isnan(m.B[:, 0]) | ~np.isfinite(d.min(1)), starter.NOBODY, m.T[j]).astype(np.int8)
    out = np.full(len(codes), "D", dtype="<U1")
    for h in (1, 2):
        idx = np.flatnonzero(m.half == h)
        if len(idx):
            a, b = idx[0], idx[-1] + 1
            out[a:b] = starter.smooth(codes[a:b], 50)
    return out


def baseline_plus(cell):
    from . import baseline_plus as bp
    m = cell.match
    return _memo(cell, "base", lambda: bp.predict_arrays(m.P, m.T, m.B, m.half, np.zeros(len(m.B), bool), pitch=cell.pitch))


def _gate(cell):
    from . import dgate
    def build():
        m = cell.match
        pdm, cols, _ = cell.models
        pf = dgate.frame_pd(pdm, cols, m.P, m.V, m.T, m.half)
        return dgate.gate(baseline_plus(cell), m.B, m.half, pf)[0]
    return _memo(cell, "gate", build)


def gate(cell):
    return _gate(cell)


def ship(cell):
    """Shipped stack **before E66**: Baseline+ -> missing-ball D gate -> in-gap decoder.

    Kept as the D-021 historical reference (.4039 on dev-v1); every future comparison against the
    real shipped build must use :func:`master` instead (B01).
    """
    from . import ballfree, ingap
    m = cell.match
    ph = ballfree.predict_frames(cell.models[2], m, pitch=cell.pitch)
    return ingap.gapdec(_gate(cell), m.B, m.half, ph)


def master(cell):
    """The shipped D-028 stack: Baseline+ -> D gate -> grown D -> in-gap -> sub-gap -> short-spell floor.

    Calls the real :func:`possession.pipeline.build_labels` with this cell's leave-one-game-out models,
    so a dev number here is the same stage order the CSV is built with (B01). The BAR-ATH-calibrated
    build canaries are disabled (``guard=False``) while the label path is identical: their bounds do not
    hold on degraded practice games (e.g. v=0 legitimately adds no gated D).
    """
    from . import pipeline
    return pipeline.build_labels(match=cell.match, models=cell.models, log=None, guard=False,
                                 forced=np.zeros(len(cell.match.B), bool), pitch=cell.pitch)


STAGES = {"smooth50": smooth50, "baseline_plus": baseline_plus, "gate": gate, "ship": ship,
          "master": master}


# -- running -------------------------------------------------------------------------------------

def _shift_ref(ref, runs, s):
    if s == 0:
        return ref
    out = ref.copy()
    for a, b in runs:
        seg = ref[a:b]
        if s > 0:
            out[a + s:b] = seg[:b - a - s]
            out[a:a + s] = seg[0]
        else:
            out[a:b + s] = seg[-s:]
            out[b + s:b] = seg[-1]
    return out


_FNS = {}
_CLEAN = False


def _run_game(gi):
    gid = GAMES[gi]
    clean = external.load_game(gid)
    ref = evaluate.to_codes(clean.ref)
    refs = {s: _shift_ref(ref, clean.runs, s) for s in SHIFT_FRAMES}
    out = {name: {} for name in _FNS}

    def one(seed, knob, v, match):
        cell = Cell(gid, clean, match, seed, knob, v)
        for name, fn in _FNS.items():
            pred = evaluate.to_codes(fn(cell))
            if len(pred) != len(ref):
                raise ValueError("%s: pred length %d != %d at %s" % (name, len(pred), len(ref), (gid, seed, knob, v)))
            for s in SHIFT_FRAMES:
                f, npd = evaluate._spell_flags(pred, refs[s], clean.runs)
                out[name][(gi, seed, knob, v, s)] = (f.sum(1), npd, f.shape[1])

    for seed in SEEDS:
        for knob, v in CELLS:
            one(seed, knob, v, degraded_match(clean, seed, knob, v))
    if _CLEAN:
        one(0, None, None, clean.match)
    return out


def run_dev_v1(fns, n_jobs=4, include_clean=False, expect_path=None):
    """{name: DevV1Result} for {name: pred_fn}; all functions share each cell (and its memo), one pass over the grid.
    expect_path: assert `possession` is imported from under this directory (pinned snapshots, GUARDRAILS §4)."""
    global _FNS, _CLEAN
    import possession
    if expect_path is not None:
        assert os.path.abspath(possession.__file__).startswith(os.path.abspath(expect_path)), possession.__file__
    print(provenance(), flush=True)
    _FNS = {n: (STAGES[n] if isinstance(f, str) else f) for n, f in fns.items()}
    _CLEAN = include_clean
    if n_jobs > 1:
        with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
            parts = pool.map(_run_game, range(len(GAMES)), chunksize=1)
    else:
        parts = [_run_game(i) for i in range(len(GAMES))]
    return {n: DevV1Result(n, {k: v for p in parts for k, v in p[n].items()}) for n in _FNS}


def score_dev_v1(pred_fn, name="pred_fn", **kw):
    """DevV1Result of one pipeline (callable or builtin name). print() it for the table."""
    return run_dev_v1({name: pred_fn}, **kw)[name]


def reference_results(refresh=False, **kw):
    """DevV1Result of the four built-in pipelines, cached on disk (delete or refresh=True after any change to them).

    A cache *hit* is logged, with the file and the code provenance, because the pickle carries scores
    computed by whatever stage code wrote it: a rerun after a stage change silently reports the old
    reference unless the cache is refreshed (Q34)."""
    p = os.path.join(_dir(), "reference_%s.pkl" % _stamp())
    if os.path.isfile(p) and not refresh:
        print("dev_v1: reference cache HIT %s (scores from the code that wrote it; "
              "refresh=True after a stage change)" % p, flush=True)
        print(provenance(), flush=True)
        return pickle.load(open(p, "rb"))
    res = run_dev_v1(dict(STAGES), **kw)
    _atomic(p, lambda t: pickle.dump(res, open(t, "wb")))
    return res


# -- results and statistics ----------------------------------------------------------------------

def _ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


class DevV1Result:
    """Per-cell spell counts of one pipeline. Picklable; compare two of them without re-running anything."""

    def __init__(self, name, raw):
        self.name, self.raw = name, raw

    def _sk(self, knob, v, shift=0):
        """Per seed: (tp (G, n_iou), n_pred (G,), n_gt (G,))."""
        out = []
        for s in SEEDS:
            rows = [self.raw[(g, s, knob, v, shift)] for g in range(len(GAMES))]
            out.append((np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])))
        return out

    def cell_score(self, knob, v, shift=0):
        """Pooled spellF1, seed-mean."""
        return float(np.mean([evaluate._pooled_f1(tp.sum(0), n.sum(), g.sum()) for tp, n, g in self._sk(knob, v, shift)]))

    def _draws(self, knob, v, shift=0):
        return np.mean([evaluate._pooled_f1(tp[_DRAWS].sum(1), n[_DRAWS].sum(1), g[_DRAWS].sum(1))
                        for tp, n, g in self._sk(knob, v, shift)], axis=0)

    def _per_game(self, knob, v, shift=0):
        return np.mean([[evaluate._pooled_f1(tp[i], n[i], g[i]) for i in range(len(GAMES))]
                        for tp, n, g in self._sk(knob, v, shift)], axis=0)

    def group_score(self, group, shift=0):
        return float(np.mean([self.cell_score(k, v, shift) for k, v in GROUPS[group]]))

    def summary(self):
        """{"headline", "groups": {name: score}, "cells": {(knob, v): score}, "shift": {frames: headline}, "clean"}."""
        d = {"headline": self.group_score("primary"),
             "groups": {g: self.group_score(g) for g in GROUPS},
             "cells": {kv: self.cell_score(*kv) for kv in CELLS},
             "shift": {s: self.group_score("primary", s) for s in SHIFT_FRAMES}}
        if (0, 0, None, None, 0) in self.raw:
            rows = [self.raw[(g, 0, None, None, 0)] for g in range(len(GAMES))]
            d["clean"] = float(evaluate._pooled_f1(np.sum([r[0] for r in rows], 0), sum(r[1] for r in rows),
                                                   sum(r[2] for r in rows)))
        return d

    def __str__(self):
        s = self.summary()
        lines = ["dev-v1 %s: headline %.4f (mean of 6 primary cells)" % (self.name, s["headline"])]
        lines.append("  groups  " + "  ".join("%s %.4f" % kv for kv in s["groups"].items()))
        lines.append("  shift   " + "  ".join("%+.1fs %.4f" % (k / FPS, x) for k, x in s["shift"].items()))
        for (k, v), x in s["cells"].items():
            lines.append("  knob %g v %-4g %.4f" % (k, v, x))
        if "clean" in s:
            lines.append("  clean (undegraded) %.4f" % s["clean"])
        return "\n".join(lines)


def _delta(a, b, cells, shift=0):
    """Mean over cells of the seed-mean pooled delta; CI from the shared game draws; games with a negative delta."""
    d0 = float(np.mean([a.cell_score(k, v, shift) - b.cell_score(k, v, shift) for k, v in cells]))
    dd = np.mean([a._draws(k, v, shift) - b._draws(k, v, shift) for k, v in cells], axis=0)
    pg = np.mean([a._per_game(k, v, shift) - b._per_game(k, v, shift) for k, v in cells], axis=0)
    ci = _ci(dd)
    return {"delta": round(d0, 4), "ci": ci, "neg": int((pg < 0).sum()), "excludes_0": bool(ci[0] > 0 or ci[1] < 0),
            "per_game": [round(float(x), 4) for x in pg]}


class DevV1Compare(dict):
    def __str__(self):
        f = lambda r: "%+.4f %s neg%d" % (r["delta"], r["ci"], r["neg"])
        lines = ["dev-v1 %s - %s  (game-cluster CI, %d shared draws, seed-mean inside each draw)" % (
            self["a"], self["b"], N_DRAWS)]
        lines.append("  headline (6 primary)  %s" % f(self["headline"]))
        for g in ("knob0", "knob1", "stress", "v0_reference"):
            lines.append("  %-20s  %s" % (g, f(self["groups"][g])))
        for s, r in self["shift"].items():
            lines.append("  headline ref shift %+.1fs  %s" % (s / FPS, f(r)))
        lines += ["  cell knob %g v %-4g  %s" % (k, v, f(r)) for (k, v), r in self["cells"].items()]
        t = self["tier1"]
        lines.append("  tier-1 mechanics: " + "  ".join("%s=%s" % kv for kv in t.items()))
        lines.append("  (dev-v1b H/A check, standing discounts and the on-our-match size are the caller's: D-022)")
        return "\n".join(lines)


def _as_result(x, refs):
    if isinstance(x, DevV1Result):
        return x
    if isinstance(x, str):
        return refs()[x]
    return None


def compare_dev_v1(a, b, n_jobs=4, include_clean=False, expect_path=None):
    """A - B on dev-v1. Each side: callable, builtin name (``STAGES``, from the cached reference) or a DevV1Result.
    Returns a DevV1Compare (dict; print() it): headline, groups, v=0 reference, +-0.5 s shift rows, every cell, and
    the mechanical D-022 tier-1 checks (the dev-v1b and standing-discount parts are not decided here)."""
    kw = dict(n_jobs=n_jobs, include_clean=include_clean, expect_path=expect_path)
    cache = {}
    refs = lambda: cache.setdefault("r", reference_results(**kw))
    sides = {"a": a, "b": b}
    res = {k: _as_result(x, refs) for k, x in sides.items()}
    todo = {k: sides[k] for k in res if res[k] is None}
    if todo:
        res.update(run_dev_v1(todo, **kw))
    ra, rb = res["a"], res["b"]
    out = DevV1Compare(a=ra.name if not isinstance(a, str) else a, b=rb.name if not isinstance(b, str) else b)
    out["headline"] = _delta(ra, rb, GROUPS["primary"])
    out["groups"] = {g: _delta(ra, rb, GROUPS[g]) for g in ("knob0", "knob1", "stress", "v0_reference")}
    out["shift"] = {s: _delta(ra, rb, GROUPS["primary"], s) for s in SHIFT_FRAMES}
    out["cells"] = {kv: _delta(ra, rb, [kv]) for kv in CELLS}
    out["tier1"] = {
        "headline_ci_excludes_0": out["headline"]["excludes_0"] and out["headline"]["delta"] > 0,
        "primary_cells_all_ge_0": all(out["cells"][kv]["delta"] >= 0 for kv in GROUPS["primary"]),
        "stress_cells_all_ge_0": all(out["cells"][kv]["delta"] >= 0 for kv in GROUPS["stress"]),
        "wins_at_each_shift": all(r["excludes_0"] and r["delta"] > 0 for r in out["shift"].values()),
        "v0_ci_low_ge_-.001": out["groups"]["v0_reference"]["ci"][0] >= -0.001,
    }
    return out


# -- D-interval diagnostic (S42) -----------------------------------------------------------------
#
# DIAGNOSTIC, NEVER A SELECTOR (D-022). spellF1 stays the ranked objective. This view exists to
# say *what* the D blocks look like -- count, duration, match quality, and the signed position of
# each edge -- because E07/E75 showed that interval numbers can move opposite to spellF1 (E66's
# add-ons split long blocks and lower interval F1 while raising spellF1) and that the onset edge is
# the expensive half. Never choose a variant on these numbers; use compare_dev_v1 for adoption.

D_INTERVAL_NOTE = ("D-interval F1 is a stoppage-level diagnostic, never a selector (D-022). "
                   "spellF1 is the ranked objective; use compare_dev_v1 for adoption.")


def _runs_bool(mask):
    """Half-open [start, stop) runs of a boolean mask."""
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _is_d(labels):
    """D mask from label strings (H/A/D/X) or organiser codes (2 = D)."""
    lab = np.asarray(labels)
    return (lab == 2) if lab.dtype.kind in "iu" else (lab == "D")


def d_runs(labels, half, runs=None):
    """Maximal runs of D (organiser code 2) inside the graded halves, clipped to ``runs``.

    A run never crosses a half or a scored run boundary. ``runs`` is the scored set (dev-v1 passes
    ``clean.runs``); ``None`` means every graded frame of each half. This is the interval set the
    scorer uses, and the same extraction E07's ``variants.d_runs`` implemented.
    """
    lab = np.asarray(labels)
    is_d = _is_d(lab)
    half = np.asarray(half)
    valid = None if runs is None else [(int(a), int(b)) for a, b in runs]
    out = []
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not rows.size:
            continue
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        for a, b in _runs_bool(is_d[lo:hi]):
            a, b = int(a) + lo, int(b) + lo
            if valid is None:
                out.append((a, b))
                continue
            for va, vb in valid:
                u, v = max(a, va), min(b, vb)
                if v > u:
                    out.append((u, v))
    return np.asarray(out, np.int64).reshape(-1, 2)


def _iou_match(pred, ref, iou=0.5):
    """Greedy IoU matching, strongest pair first (E07's rule); returns [(pred_i, ref_j, iou)]."""
    pred = np.asarray(pred, np.int64).reshape(-1, 2)
    ref = np.asarray(ref, np.int64).reshape(-1, 2)
    cand = []
    for i, (a, b) in enumerate(pred):
        for j, (c, d) in enumerate(ref):
            u, v = max(int(a), int(c)), min(int(b), int(d))
            if v > u:
                score = (v - u) / float(max(int(b), int(d)) - min(int(a), int(c)))
                if score >= iou:
                    cand.append((score, i, j))
    cand.sort(reverse=True)
    up, ur, matches = set(), set(), []
    for score, i, j in cand:
        if i in up or j in ur:
            continue
        up.add(i)
        ur.add(j)
        matches.append((i, j, score))
    return matches


def _run_ends(labels):
    """For every frame inside a D run, the end of that run (-1 outside D); E75's ownership trace."""
    ends = np.full(len(labels), -1, np.int32)
    for a, b in _runs_bool(_is_d(labels)):
        ends[a:b] = b
    return ends


def _dist(values):
    """Robust summary of signed errors in seconds (positive = later than GT)."""
    x = np.asarray([v for v in values if np.isfinite(v)], float)
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)),
            "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
            "mean": float(x.mean()),
            "late_1s": float((x > 1).mean()), "early_1s": float((x < -1).mean()),
            "within_1s": float((np.abs(x) <= 1).mean())}


def _overlap_records(pred, gt, half):
    """E75 block matching: every predicted block with >=1 overlapping GT D run, matched to the
    first overlapping GT start and the last overlapping GT end. Returns signed onset/end errors."""
    out = []
    for a, b in pred:
        if a >= len(half) or half[a] == 0:
            continue
        ov = gt[(gt[:, 1] > a) & (gt[:, 0] < b)]
        if not len(ov):
            continue
        ga, gb = int(ov[0, 0]), int(ov[-1, 1])
        out.append((float(a - ga) / FPS, float(b - gb) / FPS, int(a), int(b), ga, gb))
    return out


def _ownership_records(labels, stage_labels, gt, half, final_name):
    """E75 end-ownership: for each final D block, trace the last stage whose D-run end changed.

    ``stage_labels`` is an ordered {name: labels} of the stages *before* the final one; the final
    labels are ``labels`` and its name is ``final_name``. Returns (owner, creator, onset_err,
    end_err, ...) for blocks with at least one overlapping GT D run.
    """
    names = list(stage_labels) + [final_name]
    ends = [_run_ends(np.asarray(stage_labels[n])) for n in stage_labels] + [_run_ends(np.asarray(labels))]
    out = []
    for a, b in _runs_bool(_is_d(labels)):
        if a >= len(half) or half[a] == 0:
            continue
        ov = gt[(gt[:, 1] > a) & (gt[:, 0] < b)]
        if not len(ov):
            continue
        es = [int(e[b - 1]) for e in ends]
        creator = next((n for n, e in zip(names, es) if e >= 0), final_name)
        owner = creator
        for j in range(1, len(es)):
            if es[j] != es[j - 1]:
                owner = names[j]
        ga, gb = int(ov[0, 0]), int(ov[-1, 1])
        out.append((owner, creator, float(a - ga) / FPS, float(b - gb) / FPS,
                    int(a), int(b), ga, gb))
    return out


def d_stages(cell):
    """The ordered D-028 stage labels of one cell, for S42's stage-ownership diagnostic.

    Keys: ``baseline``, ``gate``, ``growd``, ``ingap``, ``subgap``, ``floor`` (the shipped stack),
    plus ``ship`` (gate+in-gap, the pre-E66 row) and ``e66`` (gate+growd+in-gap+sub-gap, no floor).
    The calls are the same functions :func:`possession.pipeline.build_labels` uses, in the same
    order, with ``guard=False``; ``floor`` is label-identical to :func:`master`. Diagnostic only:
    nothing here selects anything (D-022), and the result is memoised on the cell.
    """
    def build():
        from . import ballfree, baseline_plus as bp_mod, dgate, growd, ingap, postprocess, subgap
        m = cell.match
        dmodel, cols, bf = cell.models
        pf = dgate.frame_pd(dmodel, cols, m.P, m.V, m.T, m.half)
        base = _memo(cell, "base", lambda: bp_mod.predict_arrays(
            m.P, m.T, m.B, m.half, np.zeros(len(m.B), bool), pitch=cell.pitch))
        gated, dead = dgate.gate(base, m.B, m.half, pf)
        grown, _ = growd.apply(gated, m, dead=dead, check=False)
        ph = ballfree.predict_frames(bf, m, pitch=cell.pitch)
        ship = ingap.gapdec(gated, m.B, m.half, ph)
        ing = ingap.gapdec(grown, m.B, m.half, ph)
        sub, _ = subgap.apply(ing, gated, m, pf, fired=dead, check=False)
        floor = postprocess.apply(sub, postprocess.runs_from_half(m.half), enable=postprocess.ENABLED)
        return {"baseline": base, "gate": gated, "growd": grown, "ingap": ing, "subgap": sub,
                "floor": floor, "ship": ship, "e66": sub}
    return _memo(cell, "d_stages", build)


# -- D-interval runner ---------------------------------------------------------------------------

_D_FNS = {}
_D_STAGES = None
_D_STAGES_FOR = None
_D_FINAL = "pred"
_D_SEEDS = (0,)
_D_CELLS = ()
_D_IOU = (.3, .5, .7)


def _d_cell(gi, clean, gt, seed, knob, v):
    """All interval records of one cell: per-threshold counts, greedy-IoU edges, overlap edges,
    and (when the caller supplied stages) the E75 ownership breakdown."""
    m = degraded_match(clean, seed, knob, v)
    cell = Cell(GAMES[gi], clean, m, seed, knob, v)
    stage_lab = {}
    if _D_STAGES and _D_STAGES_FOR in _D_FNS:
        for name, fn in _D_STAGES.items():
            stage_lab[name] = evaluate.to_codes(fn(cell))
    out = {}
    for name, fn in _D_FNS.items():
        labels = evaluate.to_codes(fn(cell))
        pred = d_runs(labels, m.half, clean.runs)
        counts, edges = {}, {}
        for t in _D_IOU:
            pairs = _iou_match(pred, gt, t)
            counts[t] = (len(pairs), len(pred), len(gt), float(np.sum([p[2] for p in pairs])),
                         len(pairs))
            edges[t] = [(float(pred[i, 0] - gt[j, 0]) / FPS, float(pred[i, 1] - gt[j, 1]) / FPS,
                         float(iou), int(pred[i, 0]), int(pred[i, 1]), int(gt[j, 0]), int(gt[j, 1]))
                        for i, j, iou in pairs]
        own = None
        if stage_lab and name == _D_STAGES_FOR:
            own = _ownership_records(labels, stage_lab, gt, m.half, _D_FINAL)
        out[name] = {"counts": counts, "edges": edges, "n_pred": int(len(pred)), "n_ref": int(len(gt)),
                     "overlap": _overlap_records(pred, gt, m.half), "ownership": own}
    return out


def _d_run_game(gi):
    gid = GAMES[gi]
    clean = external.load_game(gid)
    ref = evaluate.to_codes(clean.ref)
    gt = d_runs(ref, clean.match.half, clean.runs)
    out = {name: {} for name in _D_FNS}
    for seed in _D_SEEDS:
        for knob, v in _D_CELLS:
            for name, rec in _d_cell(gi, clean, gt, seed, knob, v).items():
                out[name][(gi, seed, knob, v)] = rec
    return out


def _run_d_intervals(fns, seeds=(0,), cells=None, stages=None, final_name="pred", stages_for=None,
                     iou=(.3, .5, .7), n_jobs=1):
    """Shared runner: interval records for one or more pipelines over the frozen grid.

    ``fns`` is {name: pred_fn}; the cell (and its ``memo``) is shared, so two pipelines that reuse
    products only pay for them once. ``stages`` is an ordered {name: pred_fn} of the stages before
    the final one, applied to ``stages_for`` (a key of ``fns``), for the E75 ownership breakdown.
    Internal; the public entry points are :func:`d_interval_score` / :func:`compare_d_intervals`.
    """
    global _D_FNS, _D_STAGES, _D_STAGES_FOR, _D_FINAL, _D_SEEDS, _D_CELLS, _D_IOU
    if cells is None:
        cells = GROUPS["primary"]
    _D_FNS = {n: (STAGES[f] if isinstance(f, str) else f) for n, f in fns.items()}
    _D_STAGES = None if stages is None else {n: (STAGES[f] if isinstance(f, str) else f)
                                             for n, f in stages.items()}
    _D_STAGES_FOR = stages_for if _D_STAGES is not None else None
    _D_FINAL = final_name
    _D_SEEDS = tuple(seeds)
    _D_CELLS = tuple(cells)
    _D_IOU = tuple(iou)
    if n_jobs > 1:
        with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
            parts = pool.map(_d_run_game, range(len(GAMES)), chunksize=1)
    else:
        parts = [_d_run_game(i) for i in range(len(GAMES))]
    return {n: {k: v for p in parts for k, v in p[n].items()} for n in _D_FNS}


def _score_from_raw(name, raw, seeds, cells, thresholds):
    """Aggregate the raw per-cell interval records into a DIntervalScore."""
    tps = {t: 0 for t in thresholds}
    npr = {t: 0 for t in thresholds}
    nrf = {t: 0 for t in thresholds}
    iou_sum = {t: [] for t in thresholds}
    per_cell, per_game = {}, {}
    edges = {t: [] for t in thresholds}
    overlap, own = [], []
    for (gi, seed, knob, v), rec in raw.items():
        cell_key = (knob, v)
        pc = per_cell.setdefault(cell_key, {t: [0, 0, 0, 0.0, 0] for t in thresholds})
        pg = per_game.setdefault(gi, {t: [0, 0, 0] for t in thresholds})
        for t in thresholds:
            tp, np_, nr, si, nm = rec["counts"][t]
            tps[t] += tp
            npr[t] += np_
            nrf[t] += nr
            iou_sum[t].append((si / nm) if nm else np.nan)
            pc[t][0] += tp
            pc[t][1] += np_
            pc[t][2] += nr
            pc[t][3] += si
            pc[t][4] += nm
            pg[t][0] += tp
            pg[t][1] += np_
            pg[t][2] += nr
            for (onset, end, iou, op, ep, og, eg) in rec["edges"][t]:
                edges[t].append({"game": GAMES[gi], "seed": seed, "knob": knob, "v": v,
                                 "onset_pred": op, "end_pred": ep, "onset_gt": og, "end_gt": eg,
                                 "onset_err": onset, "end_err": end, "iou": iou})
        for (onset, end, op, ep, og, eg) in rec["overlap"]:
            overlap.append({"game": GAMES[gi], "seed": seed, "knob": knob, "v": v,
                            "onset_pred": op, "end_pred": ep, "onset_gt": og, "end_gt": eg,
                            "onset_err": onset, "end_err": end})
        for row in (rec["ownership"] or ()):
            owner, creator, onset, end, a, b, ga, gb = row
            own.append({"owner": owner, "creator": creator, "game": GAMES[gi], "seed": seed,
                        "knob": knob, "v": v, "onset_pred": a, "end_pred": b, "onset_gt": ga,
                        "end_gt": gb, "onset_err": onset, "end_err": end})

    def f1(tp, np_, nr):
        return 2.0 * tp / max(1, np_ + nr)

    res = {
        "name": name, "seeds": tuple(seeds), "cells": tuple(cells), "thresholds": tuple(thresholds),
        "tp": dict(tps), "n_pred": dict(npr), "n_ref": dict(nrf),
        "f1": {t: f1(tps[t], npr[t], nrf[t]) for t in thresholds},
        "mean_iou": {t: (float(np.nanmean(iou_sum[t])) if np.any(np.isfinite(iou_sum[t])) else float("nan"))
                     for t in thresholds},
        "per_cell": {}, "per_game": {},
        "edges": {t: {"n": len(edges[t]),
                      "onset": _dist([e["onset_err"] for e in edges[t]]),
                      "end": _dist([e["end_err"] for e in edges[t]]),
                      "records": edges[t]} for t in thresholds},
        "overlap_edges": {"n": len(overlap),
                          "onset": _dist([e["onset_err"] for e in overlap]),
                          "end": _dist([e["end_err"] for e in overlap]),
                          "records": overlap},
        "ownership": None,
        "note": D_INTERVAL_NOTE,
    }
    for (knob, v), pc in per_cell.items():
        res["per_cell"][(knob, v)] = {
            t: {"tp": pc[t][0], "n_pred": pc[t][1], "n_ref": pc[t][2],
                "f1": f1(pc[t][0], pc[t][1], pc[t][2]),
                "mean_iou": (pc[t][3] / pc[t][4]) if pc[t][4] else float("nan")}
            for t in thresholds}
    for gi, pg in per_game.items():
        res["per_game"][gi] = {t: f1(pg[t][0], pg[t][1], pg[t][2]) for t in thresholds}
    if own:
        table = {}
        for name_ in {r["owner"] for r in own} | {r["creator"] for r in own}:
            rows = [r for r in own if r["owner"] == name_]
            if not rows:
                continue
            table[name_] = {"n": len(rows), "share": len(rows) / len(own),
                            "end": _dist([r["end_err"] for r in rows]),
                            "onset": _dist([r["onset_err"] for r in rows])}
        res["ownership"] = {"n": len(own), "owners": table,
                            "creator": {n: sum(1 for r in own if r["creator"] == n)
                                        for n in {r["creator"] for r in own}},
                            "records": own}
    return DIntervalScore(res)


class DIntervalScore(dict):
    """Result of :func:`d_interval_score` (dict; print() it for the table).

    Keys: ``f1``/``tp``/``n_pred``/``n_ref``/``mean_iou`` per IoU threshold (pooled over all scored
    cell-games), ``per_cell`` and ``per_game``, ``edges`` (signed per-interval errors from greedy
    IoU matching at every threshold), ``overlap_edges`` (E75's any-overlap block matching) and
    ``ownership`` (only when stage callables were supplied). Diagnostic, never a selector (D-022).
    """

    def __str__(self):
        s = ["dev-v1 D intervals %s: %d seeds, %d cells" % (
            self["name"], len(self["seeds"]), len(self["cells"]))]
        for t in self["thresholds"]:
            e = self["edges"][t]
            s.append("  IoU %.1f  F1 %.4f  tp %d / pred %d / ref %d  meanIoU %.4f  "
                     "end med %+.2fs  late %.1f%%  early %.1f%%" % (
                         t, self["f1"][t], self["tp"][t], self["n_pred"][t], self["n_ref"][t],
                         self["mean_iou"][t], e["end"].get("median", float("nan")),
                         100 * e["end"].get("late_1s", float("nan")),
                         100 * e["end"].get("early_1s", float("nan"))))
        ov = self["overlap_edges"]
        if ov["n"]:
            s.append("  overlap-matched (E75) n %d  onset med %+.2fs  end med %+.2fs  "
                     "end late %.1f%% / early %.1f%%" % (
                         ov["n"], ov["onset"]["median"], ov["end"]["median"],
                         100 * ov["end"]["late_1s"], 100 * ov["end"]["early_1s"]))
        if self["ownership"]:
            s.append("  stage ownership (n %d):" % self["ownership"]["n"])
            for name, r in self["ownership"]["owners"].items():
                s.append("    %-9s %5.1f%%  end med %+.2fs  late %.1f%%  early %.1f%%" % (
                    name, 100 * r["share"], r["end"]["median"],
                    100 * r["end"]["late_1s"], 100 * r["end"]["early_1s"]))
        s.append("  " + self["note"])
        return "\n".join(s)


def d_interval_score(pred_fn, iou=(.3, .5, .7), *, seeds=SEEDS, cells=None, stages=None,
                     final_name="pred", n_jobs=1):
    """D-interval F1 / IoU and signed edge errors for one pipeline.

    **Diagnostic, never a selector (D-022).** spellF1 is the ranked objective; this is the
    stoppage-level view (count, duration, IoU, onset/end error).

    ``pred_fn`` is a callable (or a name in :data:`STAGES`). The frozen 10 games and the requested
    cells/seeds are used; ``cells`` defaults to the 6 primary cells and ``seeds`` to the frozen
    ``SEEDS`` (0, 1, 2). Pass ``seeds=(0,)`` for E07's stage-B convention (the acceptance check).

    GT intervals are maximal runs of D inside each game's scored runs (``clean.runs``). Matching is
    greedy by IoU, strongest pair first (E07). ``f1`` is pooled (2*tp/(pred+ref)) and also
    reported per cell; ``mean_iou`` is the mean per-cell-game mean IoU.

    ``edges[t]`` holds one record per matched interval: ``onset_pred - onset_gt`` and
    ``end_pred - end_gt`` in seconds (positive = later than GT), the IoU, and the share of errors
    later/earlier than 1 s. ``overlap_edges`` is E75's any-overlap block matching (each predicted
    block to the first overlapping GT onset and the last overlapping GT end).

    ``stages`` (optional) is an ordered {name: pred_fn} of the stages **before** ``pred_fn``
    (e.g. baseline, gate, growd, ingap, subgap). It adds ``ownership``: for each predicted D block,
    the last stage whose D-run end differs from the previous stage's end -- the E75 table. Use
    :func:`d_stages` for the frozen stack's stage callables. ``final_name`` is the name the scored
    pipeline gets in that table (E75 used ``"floor"``).
    """
    res = _run_d_intervals({"_s42": pred_fn}, seeds=seeds, cells=cells, stages=stages,
                           final_name=final_name, stages_for="_s42", iou=iou, n_jobs=n_jobs)
    name = pred_fn if isinstance(pred_fn, str) else getattr(pred_fn, "__name__", "pred_fn")
    return _score_from_raw(name, res["_s42"], tuple(seeds), tuple(cells or GROUPS["primary"]),
                           tuple(iou))


def _d_boot_delta(raw_a, raw_b, seeds, cells, t, draws):
    """Shared-draw game-cluster bootstrap of pooled interval-F1(a) - F1(b)."""
    def series(raw):
        per_seed = []
        for s in seeds:
            tp = np.array([[raw[(g, s, k, v)]["counts"][t][0] for g in range(len(GAMES))]
                           for k, v in cells])
            np_ = np.array([[raw[(g, s, k, v)]["counts"][t][1] for g in range(len(GAMES))]
                            for k, v in cells])
            nr = np.array([[raw[(g, s, k, v)]["counts"][t][2] for g in range(len(GAMES))]
                           for k, v in cells])
            tp_d = tp[:, draws].sum(axis=2).sum(axis=0)
            np_d = np_[:, draws].sum(axis=2).sum(axis=0)
            nr_d = nr[:, draws].sum(axis=2).sum(axis=0)
            per_seed.append(2.0 * tp_d / np.maximum(1, np_d + nr_d))
        return np.mean(per_seed, axis=0)
    return series(raw_a) - series(raw_b)


class DIntervalCompare(dict):
    """Result of :func:`compare_d_intervals` (dict; print() it)."""

    def __str__(self):
        s = ["dev-v1 D intervals %s - %s (%d shared draws, %d seeds, %d cells)" % (
            self["a"], self["b"], N_DRAWS, len(self["seeds"]), len(self["cells"]))]
        for t in self["thresholds"]:
            d = self["delta"][t]
            e = self["edges"][t]
            s.append("  IoU %.1f  F1 %s %.4f vs %s %.4f  delta %+.4f %s neg%d  "
                     "end med %+.2fs vs %+.2fs" % (
                         t, self["a"], d["f1_a"], self["b"], d["f1_b"], d["delta"],
                         d["ci"], d["neg"], e["end_a"].get("median", float("nan")),
                         e["end_b"].get("median", float("nan"))))
        s.append("  " + D_INTERVAL_NOTE)
        return "\n".join(s)


def compare_d_intervals(a, b, iou=(.3, .5, .7), *, seeds=SEEDS, cells=None, n_jobs=1):
    """A - B on the D-interval diagnostic, with the same 10 games, cells and shared draws as spellF1.

    **Diagnostic, never a selector (D-022).** Each side is a callable or a name in :data:`STAGES`.
    Returns per-threshold ``{f1_a, f1_b, delta, ci, neg, per_cell, per_game}`` plus the signed edge
    summaries of both sides. The CI is a game-cluster bootstrap over the frozen shared draws
    (``_DRAWS``), seed-mean inside each draw; ``neg`` is the number of games with a negative delta.
    """
    if cells is None:
        cells = GROUPS["primary"]
    seeds, cells = tuple(seeds), tuple(cells)
    ra = a if isinstance(a, str) else getattr(a, "__name__", "a")
    rb = b if isinstance(b, str) else getattr(b, "__name__", "b")
    raw = _run_d_intervals({"a": a, "b": b}, seeds=seeds, cells=cells, iou=iou, n_jobs=n_jobs)
    A = _score_from_raw(ra, raw["a"], seeds, cells, tuple(iou))
    B = _score_from_raw(rb, raw["b"], seeds, cells, tuple(iou))
    out = DIntervalCompare(a=ra, b=rb, seeds=seeds, cells=cells, thresholds=tuple(iou))
    out["f1_a"] = dict(A["f1"])
    out["f1_b"] = dict(B["f1"])
    out["delta"], out["ci"], out["neg"] = {}, {}, {}
    for t in iou:
        draws = _d_boot_delta(raw["a"], raw["b"], seeds, cells, t, _DRAWS)
        out["delta"][t] = {"f1_a": A["f1"][t], "f1_b": B["f1"][t],
                           "delta": A["f1"][t] - B["f1"][t],
                           "ci": _ci(draws), "neg": int(sum(1 for g in range(len(GAMES))
                                                           if A["per_game"][g][t] < B["per_game"][g][t]))}
        out["ci"][t] = out["delta"][t]["ci"]
        out["neg"][t] = out["delta"][t]["neg"]
    out["per_cell"] = {t: {kv: {"a": A["per_cell"][kv][t]["f1"],
                                "b": B["per_cell"][kv][t]["f1"],
                                "delta": A["per_cell"][kv][t]["f1"] - B["per_cell"][kv][t]["f1"]}
                           for kv in cells} for t in iou}
    out["edges"] = {t: {"n_a": A["edges"][t]["n"], "n_b": B["edges"][t]["n"],
                        "onset_a": A["edges"][t]["onset"], "onset_b": B["edges"][t]["onset"],
                        "end_a": A["edges"][t]["end"], "end_b": B["edges"][t]["end"]} for t in iou}
    return out
