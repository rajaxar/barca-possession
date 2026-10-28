"""Scoring and label-free diagnostics for any (prediction, reference) pair.

The metric is the organiser's `evaluation/possession_metrics.py`, imported (config.repo_dir()), never
re-implemented. Everything here only adapts inputs to it: arbitrary games, arbitrary graded runs,
pooling over games, and CIs.

Labels are accepted as strings (H/A/D/X) or as the organiser's codes (0..3). A *run* is a half-open
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
    """Organiser codes (int8) from strings or from codes already."""
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
    """Organiser `evaluate()` dict for one pair. Keys include spell_f1_mean (ranking), macro_f1,
    spell_bootstrap_low/high (possession bootstrap, organiser's seed) when bootstrap_n > 0."""
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
