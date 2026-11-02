"""Baseline+ (E01): persistent last owner, hysteresis, out-of-bounds D and spell merging.

The input evidence is deliberately simple: when the ball is present, the nearest tracked player's
team is evidence only if that player is within ``r`` meters. Otherwise the previous owner persists
through flights, loose balls and tracking gaps. A new owner needs ``k`` supporting evidence frames;
missing frames neither advance nor cancel that candidacy. The accepted switch is back-dated to the
candidacy's first frame, so confirmation does not add output lag.

A debounced excursion beyond the pitch by ``margin`` starts D. It ends at the first sustained run of
in-pitch ball movement (``v_restart`` and ``restart_len``). Published pre-kickoff frames are supplied
by :mod:`possession.structure`; all other ungraded frames remain X. Finally, H/A spells shorter than
``m`` frames are absorbed into a neighboring H/A spell, or into D when both neighbors are D.

No public labels are read or fitted. The defaults were re-selected on the external practice set (E01
design; k raised 8 -> 16 by E20 at tier 1, with Q14's caveat that it is regime-dependent): 2.5 m rejects
distant receivers during flight; 16 evidence frames are 0.64 s at 25 fps; and the 20-frame (0.8 s) merge
floor is below R08's 1.0 s hard ceiling. The 0.5 m boundary margin, 6-frame out debounce, and the 3 m/s
sustained-for-10-frames restart test reject tracking jitter without imposing a minimum stoppage duration
or an arbitrary maximum dead time. On the shipped path this module's merge is effectively a no-op: the
in-gap decoder re-applies the same 20-frame merge (B14, E83).
"""
import numpy as np

from .structure import PITCH_L, PITCH_W, structure

FPS = 25
DEFAULT_PARAMS = dict(r=2.5, k=16, m=20, margin=0.5, min_out=6, v_restart=3.0, restart_len=10)


def _evidence(P, T, B, rows, r):
    """Return nearest-team codes (0/1) within r meters for ``rows``, else -1."""
    ev = np.full(len(rows), -1, np.int8)
    ok = ~np.isnan(B[rows, 0])
    idx = rows[ok]
    d = np.linalg.norm(P[idx] - B[idx][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    near = d[np.arange(len(idx)), j] <= r
    ev[ok] = np.where(near, T[j], -1).astype(np.int8)
    return ev


def _dead_mask(B, rows, forced, p, pitch):
    """Detect out-of-bounds D intervals over one contiguous half, including published forced D."""
    length, width = pitch
    b = B[rows]
    ok = ~np.isnan(b[:, 0])
    margin = p["margin"]
    out = ok & ((b[:, 0] < -margin) | (b[:, 0] > length + margin) |
                (b[:, 1] < -margin) | (b[:, 1] > width + margin))
    inside = ok & (b[:, 0] >= 0) & (b[:, 0] <= length) & (b[:, 1] >= 0) & (b[:, 1] <= width)

    speed = np.full(len(rows), np.nan)
    speed[1:] = np.linalg.norm(np.diff(b, axis=0), axis=1) * FPS
    fast = inside & (speed >= p["v_restart"])
    fast[1:] &= ok[:-1]

    n = len(rows)
    dead = forced[rows].copy()
    i = 0
    while i < n:
        if not out[i] or dead[i]:
            i += 1
            continue
        j = i
        while j < n and out[j] and not dead[j]:
            j += 1
        if j - i < p["min_out"]:
            i = j
            continue

        # Find the first frame of restart_len consecutive fast, genuinely in-pitch frames. If no
        # restart is observable, retain D through the half rather than inventing a live-ball time cap.
        end, run = j, 0
        while end < n:
            run = run + 1 if fast[end] else 0
            if run >= p["restart_len"]:
                end = end - run + 1
                break
            end += 1
        dead[i:end] = True
        i = max(end, j)
    return dead


def _labels_half(ev, dead, k):
    """Apply persistent-owner hysteresis over one half; return owner codes, with D represented later."""
    n = len(ev)
    out = np.full(n, -1, np.int8)
    cur = -1
    cand, cstart, ccount = -1, 0, 0
    prev_dead = False
    for i in range(n):
        if dead[i]:
            prev_dead = True
            cand, ccount = -1, 0
            continue
        if prev_dead:
            # A restart does not imply a turnover: carry the last owner until actual ball evidence.
            cand, ccount = -1, 0
            prev_dead = False
        e = ev[i]
        if e < 0:
            out[i] = cur
            continue
        if cur < 0:
            cur = e
            out[:i] = np.where(out[:i] < 0, cur, out[:i])
            out[i] = cur
        elif e == cur:
            cand, ccount = -1, 0
            out[i] = cur
        else:
            if e != cand:
                cand, ccount, cstart = e, 0, i
            ccount += 1
            out[i] = cur
            if ccount >= k:
                cur = e
                out[cstart:i + 1] = cur
                cand, ccount = -1, 0
    return out


def _merge_short(lab, m):
    """Absorb H/A spells (codes 0/1; D=2) shorter than m frames, shortest first."""
    lab = lab.copy()
    while True:
        edges = np.flatnonzero(np.diff(lab)) + 1
        starts = np.r_[0, edges]
        stops = np.r_[edges, len(lab)]
        vals = lab[starts]
        lens = stops - starts
        candidates = [(lens[s], s) for s in range(len(starts)) if vals[s] < 2 and lens[s] < m]
        if not candidates:
            return lab
        _, s = min(candidates)
        neighbors = [t for t in (s - 1, s + 1) if 0 <= t < len(starts) and vals[t] < 2]
        if neighbors:
            t = max(neighbors, key=lambda t: lens[t])
            lab[starts[s]:stops[s]] = vals[t]
        else:
            lab[starts[s]:stops[s]] = 2


def predict_arrays(P, T, B, half, forced, params=None, pitch=(PITCH_L, PITCH_W)):
    """Array implementation, separated from match loading for external games and synthetic tests."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    codes = np.full(len(B), 3, np.int8)  # X outside published graded halves
    for h in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == h)
        if len(rows) == 0:
            continue
        dead = _dead_mask(B, rows, forced, p, pitch)
        ev = _evidence(P, T, B, rows, p["r"])
        lab = _labels_half(ev, dead, p["k"])
        lab[dead] = 2
        lab[lab < 0] = 2  # no owner evidence anywhere in the half
        codes[rows] = _merge_short(lab, p["m"])
    return np.array(["H", "A", "D", "X"])[codes]


def predict(match, params=None, forced=None, pitch=None):
    """Return an (N,) H/A/D/X label array for a :class:`possession.data.Match`.

    ``forced`` is the published pre-kickoff D mask (``structure.forced_dead_mask``); computed from
    the match when omitted, so the shipped pipeline can run ``structure`` as its own first stage.
    ``pitch`` overrides the BAR-ATH (105.3 x 68) default for external practice games.
    """
    if forced is None:
        forced = structure(match).forced_dead_mask
    if pitch is None:
        return predict_arrays(match.P, match.T, match.B, match.half, forced, params)
    return predict_arrays(match.P, match.T, match.B, match.half, forced, params, pitch=pitch)
