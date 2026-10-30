"""Sub-gap dead-ball gate (E69, shipped as E66): short hidden stoppages the whole-gap gate misses.

:func:`possession.dgate.gate` fires on a ball-missing run only through the gap-mean P(D). Inside
ball-missing runs of at least ``TF`` frames that the gate left untouched (no D from the gate or Baseline+),
the maximal runs where P(D) > ``Q`` for at least ``T2`` frames become D; H/A islands shorter than ``ABSORB``
frames between D stretches are then absorbed. It is an overlay on the shipped stack (gate + ingap) after
ingap, never a re-decode (E69: re-decoding after the carve is negative in 24/24 variants). P(D) is the
player-only model of :mod:`possession.dgate` (no ball columns).

:func:`footprint` / :func:`check_footprint` are the label-free build guard: no added frame inside a gap the
gate fired on, and the added D share within [``MIN_ADDED``, ``MAX_ADDED``] of the graded frames.
:func:`apply` returns its input untouched unless enabled (module flag ``ENABLED``).
"""
import numpy as np

from . import dgate

ENABLED = True                   # E66: on after the BAR-ATH guards pass (scratch/E66/barath.json); False = shipped gate+ingap
Q = 0.5                          # P(D) threshold inside an unfired gap
TF = 50                          # min ball-missing run, frames (2 s; as dgate)
T2 = 50                          # min P(D) > Q run inside the gap, frames
ABSORB = dgate.ABSORB
MIN_ADDED = 0.005                # guard: added D share of graded, E69 measured 0.0066 on BAR-ATH
MAX_ADDED = 0.15


def overlay(labels, gated, B, half, pf, q=Q, tf=TF, t2=T2, absorb=ABSORB):
    """``labels``: shipped stack (gate + ingap); ``gated``: the gate's output, which says which gaps were left unfired."""
    out = np.asarray(labels).copy()
    gated = np.asarray(gated)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        lo, hi = idx[0], idx[-1] + 1
        for a, b in dgate._runs(~np.isfinite(B[lo:hi, 0])):
            a, b = a + lo, b + lo
            if b - a < tf or (gated[a:b] == "D").any():
                continue
            for x, y in dgate._runs(pf[a:b] > q):
                if y - x >= t2:
                    seg = out[a + x:a + y]
                    seg[np.isin(seg, ("H", "A"))] = "D"
    return dgate.absorb_islands(out, absorb)


def footprint(before, after, half, fired=None):
    """Label-free footprint over graded frames: added D share and runs, changes inside gaps the gate fired on."""
    before, after = np.asarray(before), np.asarray(after)
    graded = np.asarray(half) > 0
    n = max(1, int(graded.sum()))
    chg = (before != after) & graded
    return dict(graded=int(graded.sum()), changed=int(chg.sum()), d_added_share_graded=float(chg.sum()) / n,
                added_runs=len(dgate._runs(chg)),
                changed_in_fired=int((chg & fired).sum()) if fired is not None else None)


def check_footprint(fp):
    if fp["changed_in_fired"]:
        raise RuntimeError("subgap changed %d frames inside gaps the gate fired on" % fp["changed_in_fired"])
    if not MIN_ADDED <= fp["d_added_share_graded"] <= MAX_ADDED:
        raise RuntimeError("subgap added D to %.4f of graded frames, outside [%.3f, %.2f]"
                           % (fp["d_added_share_graded"], MIN_ADDED, MAX_ADDED))
    return fp


def apply(labels, gated, match, pf, fired=None, enabled=None):
    """Flag-gated entry point. Returns ``(labels, footprint or None)``."""
    if not (ENABLED if enabled is None else enabled):
        return labels, None
    out = overlay(labels, gated, match.B, match.half, pf)
    return out, check_footprint(footprint(labels, out, match.half, fired))
