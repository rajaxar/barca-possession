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
Built-in reference pipelines: ``smooth50``, ``baseline_plus``, ``gate``, ``ship`` (``STAGES``).

    from possession import dev_v1
    res = dev_v1.score_dev_v1(my_fn)                    # headline, per-cell, stress, v=0 and shift rows
    print(res)
    cmp = dev_v1.compare_dev_v1(my_fn, "ship")          # either side: callable, builtin name or a saved result
    print(cmp)

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
    """Shipped stack: Baseline+ -> missing-ball D gate -> in-gap decoder."""
    from . import ballfree, ingap
    m = cell.match
    ph = ballfree.predict_frames(cell.models[2], m, pitch=cell.pitch)
    return ingap.gapdec(_gate(cell), m.B, m.half, ph)


STAGES = {"smooth50": smooth50, "baseline_plus": baseline_plus, "gate": gate, "ship": ship}


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
    """DevV1Result of the four built-in pipelines, cached on disk (delete or refresh=True after any change to them)."""
    p = os.path.join(_dir(), "reference_%s.pkl" % _stamp())
    if os.path.isfile(p) and not refresh:
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
