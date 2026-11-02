"""Scoring and label-free diagnostics for any (prediction, reference) pair.

The metric is the organizer's `evaluation/possession_metrics.py`, imported (config.repo_dir()), never
re-implemented. Everything here only adapts inputs to it: arbitrary games, arbitrary graded runs,
pooling over games, and CIs.

Labels are accepted as strings (H/A/D/X) or as the organizer's codes (0..3). A *run* is a half-open
[start, stop) frame span that is scored; the metric never matches or counts across runs. External
games pass their own runs (e.g. the two periods); if omitted they are the non-X spans of the reference.
Predictions on frames outside the runs are ignored.
"""
import os
import sys
from collections import namedtuple

import numpy as np

from . import config

FPS = 25


def _pm():
    path = os.path.join(config.repo_dir(), "evaluation")
    if path not in sys.path:
        sys.path.insert(0, path)
    import possession_metrics
    return possession_metrics


pm = _pm()

Game = namedtuple("Game", "name match ref runs")
Game.__new__.__defaults__ = (None,)

DURATION_BINS_S = (0, 1, 2, 5, 10, 20, 40, np.inf)


def to_codes(labels):
    """Organizer codes (int8) from strings or from codes already."""
    a = np.asarray(labels)
    if a.dtype.kind in "iu":
        if a.size and (a.min() < 0 or a.max() > 3):
            raise ValueError("codes outside 0..3")
        return a.astype(np.int8)
    return pm.label_codes(a)


def _runs(ref, runs):
    return pm.valid_runs(ref) if runs is None else [(int(a), int(b)) for a, b in runs]


def _pack(items):
    """Concatenate (pred, ref, runs) items into one pair, run pieces separated by one X/X frame,
    so valid_runs() recovers exactly the intended runs. Returns (pred, ref, index) where
    index[i] is the source frame of packed frame i (-1 for gaps) and item id per frame."""
    preds, refs, idx, who = [], [], [], []
    gap = np.array([pm.X], dtype=np.int8)
    for k, (pred, ref, runs) in enumerate(items):
        for a, b in _runs(ref, runs):
            if b <= a:
                continue
            preds += [pred[a:b], gap]
            refs += [ref[a:b], gap]
            idx += [np.arange(a, b), np.array([-1])]
            who += [np.full(b - a, k), np.array([-1])]
    if not refs:
        raise ValueError("no scored frames")
    return (np.concatenate(preds), np.concatenate(refs), np.concatenate(idx), np.concatenate(who))


def score(pred, ref, runs=None, bootstrap_n=0, taus=pm.TAUS):
    """Organizer `evaluate()` dict for one pair. Keys include spell_f1_mean (ranking), macro_f1,
    spell_bootstrap_low/high (possession bootstrap, organizer's seed) when bootstrap_n > 0."""
    pred, ref = to_codes(pred), to_codes(ref)
    if len(pred) != len(ref):
        raise ValueError("length mismatch: pred {} vs ref {}".format(len(pred), len(ref)))
    p, r, _, _ = _pack([(pred, ref, runs)])
    return pm.evaluate(r, p, taus=taus, bootstrap_n=bootstrap_n)


def score_dev(pred_fn, games, bootstrap_n=1000):
    """Run pred_fn(match) -> labels on each game and score against its reference.

    games: iterable of Game or (match, ref) / (name, match, ref[, runs]) tuples.
    Returns {"per_game": {name: result}, "pooled": result, "n_games": int}. `pooled` matches
    possessions over all games' runs together (the sum of the games, not a mean of means)."""
    items, names = [], []
    for i, g in enumerate(games):
        if not isinstance(g, Game):
            g = Game(*g) if len(g) >= 3 else Game("game{}".format(i), g[0], g[1])
        names.append(g.name)
        pred, ref = to_codes(pred_fn(g.match)), to_codes(g.ref)
        if len(pred) != len(ref):
            raise ValueError("{}: pred length {} != ref {}".format(g.name, len(pred), len(ref)))
        items.append((pred, ref, g.runs))
    per = {}
    for name, (pred, ref, runs) in zip(names, items):
        per[name] = score(pred, ref, runs, bootstrap_n)
    p, r, _, _ = _pack(items)
    return {"per_game": per, "pooled": pm.evaluate(r, p, bootstrap_n=bootstrap_n),
            "n_games": len(items)}


def summary_line(res):
    ci = ""
    if res.get("spell_bootstrap_low") is not None:
        ci = " [{:.3f}, {:.3f}]".format(res["spell_bootstrap_low"], res["spell_bootstrap_high"])
    return "spellF1 {:.4f}{}  macroF1 {:.4f}  spells pred/gt {}/{}".format(
        res["spell_f1_mean"], ci, res["macro_f1"], res["n_pred_spells"], res["n_gt_spells"])


def spell_diff(pred, ref, runs=None, iou=0.5):
    """Per-spell outcome at one IoU threshold, in the original frame numbers.

    Returns dict of lists of (start, stop, code): matched_gt, missed_gt, false_pred, plus matched
    as (gt_spell, pred_spell, iou). Pred is X-filled exactly as the scorer does. Counts agree with
    the scorer's n_matched / n_missed / n_spurious."""
    pred, ref = to_codes(pred), to_codes(ref)
    p, r, idx, _ = _pack([(pred, ref, runs)])
    pruns = pm.valid_runs(r)
    filled = pm.fill_abstentions(p, pruns)
    gt = pm.possession_spells(r, pruns)
    pr = pm.possession_spells(filled, pruns)
    used_g, used_p, matched = set(), set(), []
    for v, i, j in pm.spell_overlaps(gt, pr):
        if v < iou:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, v))

    def back(s):  # packed -> source frames (spells never straddle a gap)
        return (int(idx[s[0]]), int(idx[s[1] - 1]) + 1, s[2])

    return {"matched": [(back(gt[i]), back(pr[j]), v) for i, j, v in matched],
            "missed_gt": [back(s) for i, s in enumerate(gt) if i not in used_g],
            "false_pred": [back(s) for j, s in enumerate(pr) if j not in used_p]}


def diagnostics(pred, match=None, runs=None):
    """Label-free description of a prediction (no reference needed).

    Scope = `runs`, else the per-half graded spans of `match.half` (>0), else the whole array.
    Spells are maximal H/A runs inside a half, X forward-filled first (as the scorer does)."""
    pred = to_codes(pred)
    if runs is None:
        if match is not None:
            half = np.asarray(match.half)
            runs = [(int(np.flatnonzero(half == h)[0]), int(np.flatnonzero(half == h)[-1]) + 1)
                    for h in np.unique(half[half > 0])]
        else:
            runs = [(0, len(pred))]
    runs = [(int(a), int(b)) for a, b in runs]
    filled = pm.fill_abstentions(pred, runs)
    n = sum(b - a for a, b in runs)
    inrun = np.concatenate([pred[a:b] for a, b in runs])
    spells = pm.possession_spells(filled, runs)
    dur = np.array([(b - a) / FPS for a, b, _ in spells])
    d_stretches = [s for s in pm.segments(filled, runs) if s[2] == pm.D]
    out = {"n_frames": n, "minutes": n / FPS / 60.0,
           "frame_share": {k: float(np.mean(inrun == c)) for k, c in
                           (("H", pm.H), ("A", pm.A), ("D", pm.D), ("X", pm.X))},
           "n_spells": len(spells),
           "n_spells_per_half": [sum(1 for a, _, _ in spells if r0 <= a < r1) for r0, r1 in runs],
           "n_d_stretches": len(d_stretches),
           "flickers_lt_1s": int((dur < 1.0).sum()),
           "spells_per_5min": len(spells) / (n / FPS / 300.0) if n else 0.0}
    alive = out["frame_share"]["H"] + out["frame_share"]["A"]
    out["team_share_of_alive"] = {"H": out["frame_share"]["H"] / alive if alive else 0.0,
                                  "A": out["frame_share"]["A"] / alive if alive else 0.0}
    out["spell_duration_quantiles_s"] = (
        {q: float(np.percentile(dur, q)) for q in (5, 25, 50, 75, 95)} if len(dur) else {})
    hist, _ = np.histogram(dur, bins=DURATION_BINS_S)
    out["spell_duration_hist"] = {"{}-{}s".format(a, "inf" if b == np.inf else b): int(h)
                                  for a, b, h in zip(DURATION_BINS_S[:-1], DURATION_BINS_S[1:], hist)}
    return out


def format_diagnostics(d):
    q = d["spell_duration_quantiles_s"]
    lines = ["{:.1f} min scored; H/A/D/X share {}".format(
                 d["minutes"], "/".join("{:.2f}".format(d["frame_share"][k]) for k in "HADX")),
             "spells {} (per half {}), {:.1f} per 5 min; D stretches {}; flickers <1s {}".format(
                 d["n_spells"], d["n_spells_per_half"], d["spells_per_5min"],
                 d["n_d_stretches"], d["flickers_lt_1s"]),
             "team share of alive H {:.2f} A {:.2f}".format(
                 d["team_share_of_alive"]["H"], d["team_share_of_alive"]["A"]),
             "duration quantiles s " + " ".join("p{}={:.1f}".format(k, v) for k, v in q.items()),
             "duration hist " + " ".join("{}:{}".format(k, v) for k, v in d["spell_duration_hist"].items())]
    return "\n".join(lines)


# -- paired comparison of two predictors on dev games (S14) ----------------------------------------

def _spell_flags(pred, ref, runs):
    """(flags[len(SPELL_IOUS), n_gt] bool, n_pred) for one game: matched-or-not per GT possession at
    each IoU, exactly as the scorer's bootstrap builds them (X forward-filled first)."""
    p, r, _, _ = _pack([(pred, ref, runs)])
    pruns = pm.valid_runs(r)
    gt = pm.possession_spells(r, pruns)
    pr = pm.possession_spells(pm.fill_abstentions(p, pruns), pruns)
    ov = pm.spell_overlaps(gt, pr)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gt), t) for t in pm.SPELL_IOUS])
    return flags, len(pr)


def _pooled_f1(tp, n_pred, n_gt):
    """spell_f1_mean from summed counts. tp: (..., n_iou); n_pred, n_gt broadcast against tp[..., 0].
    As the organizer's bootstrap, fp is clamped at 0 when a resample implies tp > n_pred."""
    n_pred, n_gt = np.asarray(n_pred)[..., None], np.asarray(n_gt)[..., None]
    fp = np.maximum(0, n_pred - tp)
    den = 2 * tp + fp + (n_gt - tp)
    return np.where(tp > 0, 2.0 * tp / np.maximum(den, 1), 0.0).mean(axis=-1)


def _ci(delta, alpha):
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p: how often the resampled delta lies on the other side of zero
    p = min(1.0, 2 * min(np.mean(delta <= 0), np.mean(delta >= 0)))
    return {"low": float(lo), "high": float(hi), "p": float(p),
            "significant": bool(lo > 0 or hi < 0)}


def compare_dev(pred_fn_a, pred_fn_b, games, n=1000, seed=0, alpha=0.05):
    """Paired bootstrap of delta = spellF1(A) - spellF1(B) over dev games (pooled over games, as
    score_dev's "pooled"). `games` as in score_dev. Two resampling schemes, same draws for A and B:
      possession: within every game, resample its GT possessions (stratified by game), same picks
                  for both methods -> uncertainty from which possessions happened to be in the games;
      game:       resample whole games (cluster bootstrap), observed per-game counts -> uncertainty
                  from which games we have (the honest one when choosing methods for a new game).
    The game scheme is meaningless below ~5 games; with fewer its CI is returned but `reliable` False.
    Returns {"delta", "a", "b", "possession": {low, high, p, significant}, "game": {...},
    "per_game_delta": {name: delta}, "n_games", "n_draws", "alpha"}. significant = CI excludes 0."""
    rng = np.random.RandomState(seed)
    names, fa, fb, npa, npb, ngt, per = [], [], [], [], [], [], {}
    for i, g in enumerate(games):
        if not isinstance(g, Game):
            g = Game(*g) if len(g) >= 3 else Game("game{}".format(i), g[0], g[1])
        ref = to_codes(g.ref)
        fl = []
        for fn in (pred_fn_a, pred_fn_b):
            pred = to_codes(fn(g.match))
            if len(pred) != len(ref):
                raise ValueError("{}: pred length {} != ref {}".format(g.name, len(pred), len(ref)))
            fl.append(_spell_flags(pred, ref, g.runs))
        if fl[0][0].shape[1] == 0:
            continue  # no GT possessions: contributes nothing
        names.append(g.name)
        fa.append(fl[0][0]); npa.append(fl[0][1])
        fb.append(fl[1][0]); npb.append(fl[1][1])
        ngt.append(fl[0][0].shape[1])
    if not names:
        raise ValueError("no games with GT possessions")
    G = len(names)
    npa, npb, ngt = np.array(npa), np.array(npb), np.array(ngt)
    tpa = np.array([f.sum(axis=1) for f in fa])  # (G, n_iou)
    tpb = np.array([f.sum(axis=1) for f in fb])

    def pooled(ta, tb, sel=slice(None)):
        a = _pooled_f1(ta, npa[sel].sum(-1), ngt[sel].sum(-1))
        b = _pooled_f1(tb, npb[sel].sum(-1), ngt[sel].sum(-1))
        return a, b

    a0, b0 = pooled(tpa.sum(0), tpb.sum(0))
    # possession bootstrap: stratified by game, shared picks
    ta, tb = np.zeros((n, tpa.shape[1])), np.zeros((n, tpa.shape[1]))
    for g in range(G):
        picks = rng.randint(0, ngt[g], (n, ngt[g]))
        ta += fa[g][:, picks].sum(axis=2).T
        tb += fb[g][:, picks].sum(axis=2).T
    a, b = _pooled_f1(ta, npa.sum(), ngt.sum()), _pooled_f1(tb, npb.sum(), ngt.sum())
    poss = _ci(a - b, alpha)
    # game bootstrap
    gs = rng.randint(0, G, (n, G))
    a, b = (_pooled_f1(tpa[gs].sum(1), npa[gs].sum(1), ngt[gs].sum(1)),
            _pooled_f1(tpb[gs].sum(1), npb[gs].sum(1), ngt[gs].sum(1)))
    game = _ci(a - b, alpha)
    game["reliable"] = G >= 5
    per_game = {nm: float(_pooled_f1(tpa[g], npa[g], ngt[g]) - _pooled_f1(tpb[g], npb[g], ngt[g]))
                for g, nm in enumerate(names)}
    return {"delta": float(a0 - b0), "a": float(a0), "b": float(b0), "possession": poss, "game": game,
            "per_game_delta": per_game, "n_games": G, "n_draws": n, "alpha": alpha}


def format_compare(res):
    f = lambda k: "[{:+.3f}, {:+.3f}] p={:.3f}{}".format(
        res[k]["low"], res[k]["high"], res[k]["p"], " *" if res[k]["significant"] else "")
    return ("delta spellF1 {:+.4f} (A {:.4f}, B {:.4f}) over {} games\n  possession-paired {}\n"
            "  game-cluster      {}{}").format(res["delta"], res["a"], res["b"], res["n_games"],
                                              f("possession"), f("game"),
                                              "" if res["game"]["reliable"] else "  (few games: unreliable)")
