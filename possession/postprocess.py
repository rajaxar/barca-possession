"""Metric-aware post-processing of a label sequence (E14).

spellF1 counts possessions: a spurious flicker costs a false positive *and* breaks the IoU of the
host spell, while a missing short spell costs only recall.  This module holds the deterministic,
label-only rules that trade those two errors:

  :func:`absorb_short`    delete H/A spells shorter than a floor, into D or a neighbor
  :func:`merge_across_d`  fill short D runs that sit between the *same* team with that team
  :func:`flicker`         iterate the two rules to a fixed point (flicker suppression)
  :func:`d_to_x`          replace short D runs with X (X is forward-filled with the last team)
  :func:`x_low_conf`      write X near uncertain hand-overs (acts as lag; measured, not recommended)
  :func:`diagnose`        label-free spell / D-length summary used by the E14 curves

E14 verdict (research/E14_result.md): only the short-spell floor pays.  On dev-v1 the shipped
labels carry 474 H/A spells under 1.5 s per 10 games and the scorer's own matcher gives a TP to
**2** of them, so the floor is pruning spells that cannot match; absorbing each to D (``absorb=
"to_d"``, 38 frames) is +.0099 spellF1 [.0063,.0138], 0/10 games negative, positive at +-0.5 s.
Absorbing to a neighbor instead, merging short D runs, writing X on short D and X at low-confidence
hand-overs were all flat or negative (details and K in the result).

Everything is applied inside the supplied graded runs (the two halves), never across the half-time
hole and never outside a run; X written in a run is left for the scorer to forward-fill.  Rules are
deterministic: ties are broken by the earlier segment, and ``flicker`` stops after a fixed number of
passes.  No reference labels, no match data and no fitted constants are read here.

Labels may be H/A/D/X strings or the organizer's 0..3 codes; the output keeps the input kind.
Selection of a rule and its parameters happens on external dev-v1 only (D-022); this module only
implements the candidates.
"""
import numpy as np

FPS = 25
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
NAME = np.array(["H", "A", "D", "X"])
LIVE = (0, 1)

# The shipping default is the E14 winner, not a fitted constant: a 1.5 s (38-frame) floor absorbed
# to D.  1.5 s is the top of R08's label-safe band (a perfect predictor loses .118 by deleting all
# true sub-1.5 s spells); the dev curve is flat from 1.35 s to 2.5 s (+.0090..+.0112) and R08 says do
# not go higher.  D merging is off (max_d=0): every merge variant was flat (see the result).
DEFAULT = dict(min_spell=38, absorb="to_d", max_d=0, merge_same_team_only=True)

# Switch (same convention as growd/subgap): tier 1 in E14 (D-010), so the default is on.  Nothing
# calls this module yet; see research/E14_result.md for the exact line the pipeline owner adds.
ENABLED = True


def to_codes(labels):
    """(N,) int8 organizer codes from strings or codes. Unknown labels are an error, never silently X."""
    a = np.asarray(labels)
    if a.dtype.kind in "iu":
        if a.size and (a.min() < 0 or a.max() > 3):
            raise ValueError("codes outside 0..3")
        return a.astype(np.int8)
    bad = ~np.isin(a, list(CODE))
    if bad.any():
        raise ValueError("unknown labels: %r" % sorted(set(a[bad].tolist()))[:5])
    out = np.full(a.shape, 3, np.int8)
    for k, c in CODE.items():
        out[a == k] = c
    return out


def _as_out(codes, like):
    return NAME[codes] if np.asarray(like).dtype.kind == "U" else codes.astype(np.int8)


def runs_from_half(half):
    """Graded runs [(start, stop), ...] from a (N,) half array (0 outside, 1/2 inside)."""
    half = np.asarray(half)
    runs = []
    for h in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == h)
        runs.append((int(rows[0]), int(rows[-1]) + 1))
    return runs


def _segments(lab, a, b):
    """Maximal constant-label segments of lab[a:b] as [(start, stop, code), ...] in absolute frames."""
    if b <= a:
        return []
    edges = np.flatnonzero(np.diff(lab[a:b])) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, b - a]
    return [(int(a + s), int(a + e), int(lab[a + s])) for s, e in zip(starts, stops)]


def _fill(lab, runs):
    """Copy of the labels with X forward-filled inside each run (leading X back-filled), as the scorer does.
    This module never writes X except in :func:`d_to_x` / :func:`x_low_conf`, where the scorer's reading
    is what the metric sees, so diagnostics must use the same reading."""
    out = lab.copy()
    for a, b in runs:
        seg = out[a:b]
        x = seg == 3
        if not x.any() or x.all():
            continue
        idx = np.where(~x, np.arange(b - a), -1)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.flatnonzero(~x)[0])
        idx[idx < 0] = first
        out[a:b] = seg[idx]
    return out


def segments(labels, runs):
    """All maximal constant-label segments inside ``runs`` (absolute frame spans), raw labels."""
    lab = to_codes(labels)
    out = []
    for a, b in runs:
        out += _segments(lab, int(a), int(b))
    return out


def n_spells(labels, runs):
    """Number of H/A spells (maximal live runs) inside the runs, as the scorer counts them (X filled first)."""
    lab = _fill(to_codes(labels), runs)
    n = 0
    for a, b in runs:
        if b <= a:
            continue
        live = np.isin(lab[a:b], LIVE)
        n += int(live[0]) + int((live[1:] & (lab[a + 1:b] != lab[a:b - 1])).sum())
    return n


# -- rules ---------------------------------------------------------------------------------------

def absorb_short(labels, runs, min_frames=DEFAULT["min_spell"], absorb=DEFAULT["absorb"],
                 same_neighbors=False):
    """Absorb every H/A spell shorter than ``min_frames``, shortest first (as Baseline+ does).

    ``absorb`` decides the target color:
      ``longest_live``  longest adjacent H/A neighbor, else D (Baseline+'s rule);
      ``longest``       longest adjacent segment of any color (D competes on length);
      ``to_d``          always D;
      ``prev``/``next`` previous/next adjacent segment (D if none, or at a run edge).
    A spell at a run edge has only one neighbor.  If no neighbor exists the spell becomes D.
    ``same_neighbors`` restricts the rule to spells flanked by the same team (the pure flicker case);
    a short spell at a transition is then left alone.
    """
    if absorb not in ("longest_live", "longest", "to_d", "prev", "next"):
        raise ValueError("unknown absorb policy %r" % (absorb,))
    lab = to_codes(labels).copy()
    if min_frames <= 1:
        return _as_out(lab, labels)
    runs = [(int(a), int(b)) for a, b in runs]
    while True:
        cand = []
        for a, b in runs:
            segs = _segments(lab, a, b)
            for s, (x, y, c) in enumerate(segs):
                if c in LIVE and y - x < min_frames:
                    if same_neighbors:
                        prev = segs[s - 1][2] if s > 0 else None
                        nxt = segs[s + 1][2] if s + 1 < len(segs) else None
                        if prev not in LIVE or prev != nxt:
                            continue
                    cand.append((y - x, x, s, segs))
        if not cand:
            return _as_out(lab, labels)
        _, x, s, segs = min(cand, key=lambda t: (t[0], t[1]))
        nbrs = [(i, segs[i]) for i in (s - 1, s + 1) if 0 <= i < len(segs)]
        if absorb == "to_d":
            target = 2
        elif not nbrs:
            target = 2
        elif absorb == "prev":
            target = segs[s - 1][2] if s - 1 >= 0 else 2
        elif absorb == "next":
            target = segs[s + 1][2] if s + 1 < len(segs) else 2
        elif absorb == "longest_live":
            live = [(i, g) for i, g in nbrs if g[2] in LIVE]
            target = max(live, key=lambda t: (t[1][1] - t[1][0], -t[0]))[1][2] if live else 2
        else:  # longest, any color
            target = max(nbrs, key=lambda t: (t[1][1] - t[1][0], -t[0]))[1][2]
        lab[x:segs[s][1]] = target


def merge_across_d(labels, runs, max_d=DEFAULT["max_d"], same_team_only=DEFAULT["merge_same_team_only"]):
    """Fill D runs of at most ``max_d`` frames that sit between two live spells.

    ``same_team_only`` (default): only when both neighbors are the same team (the D is then a
    spurious split of one possession).  Otherwise every interior short D run is filled with the
    previous team, which is exactly what writing X there does after the scorer's forward fill.
    D runs at a run edge or next to D are untouched.
    """
    lab = to_codes(labels).copy()
    if max_d <= 0:
        return _as_out(lab, labels)
    for a, b in runs:
        segs = _segments(lab, a, b)
        for s in range(1, len(segs) - 1):
            x, y, c = segs[s]
            if c != 2 or y - x > max_d:
                continue
            prev, nxt = segs[s - 1][2], segs[s + 1][2]
            if prev not in LIVE or nxt not in LIVE:
                continue
            if same_team_only and prev != nxt:
                continue
            lab[x:y] = prev
    return _as_out(lab, labels)


def flicker(labels, runs, min_frames=DEFAULT["min_spell"], max_d=DEFAULT["max_d"],
            absorb=DEFAULT["absorb"], same_team_only=DEFAULT["merge_same_team_only"],
            order="merge_then_absorb", max_passes=8, same_neighbors=False):
    """Flicker suppression: alternate :func:`merge_across_d` and :func:`absorb_short` to a fixed point.

    ``order`` is ``merge_then_absorb`` (default) or ``absorb_then_merge``; the rule stops when a
    pass changes nothing, after at most ``max_passes`` (deterministic either way).
    """
    if order not in ("merge_then_absorb", "absorb_then_merge"):
        raise ValueError("unknown order %r" % (order,))
    out = labels
    for _ in range(max_passes):
        before = to_codes(out)
        if order == "merge_then_absorb":
            out = merge_across_d(out, runs, max_d, same_team_only)
            out = absorb_short(out, runs, min_frames, absorb, same_neighbors)
        else:
            out = absorb_short(out, runs, min_frames, absorb, same_neighbors)
            out = merge_across_d(out, runs, max_d, same_team_only)
        if np.array_equal(to_codes(out), before):
            break
    return out


def d_to_x(labels, runs, max_d=DEFAULT["max_d"]):
    """Replace D runs of at most ``max_d`` frames with X inside the runs.

    The scorer forward-fills X with the previous label, so this is :func:`merge_across_d` applied
    to *every* short D run, plus a macroF1 cost (X on a graded frame is a FN).  Present to show why
    the shipping default writes D, not X.
    """
    lab = to_codes(labels).copy()
    if max_d <= 0:
        return _as_out(lab, labels)
    for a, b in runs:
        for x, y, c in _segments(lab, a, b):
            if c == 2 and y - x <= max_d:
                lab[x:y] = 3
    return _as_out(lab, labels)


def x_low_conf(labels, runs, uncertain, half_width=12):
    """Write X on uncertain frames within +-``half_width`` frames of an H/A hand-over.

    ``uncertain``: (N,) bool, e.g. ``abs(P(home) - 0.5) < margin`` from the ball-free decoder.  X
    is written *inside* a spell and is forward-filled with the old team by the scorer, so at a real
    hand-over it delays the flip (pure lag) and never creates a new spell.  Kept as an explicit,
    testable policy because R04 3.4 says never to emit X mid-spell inside graded ranges.
    """
    lab = to_codes(labels).copy()
    unc = np.asarray(uncertain, bool)
    if len(unc) != len(lab):
        raise ValueError("uncertain must be one per frame")
    for a, b in runs:
        live = np.isin(lab[a:b], LIVE)
        chg = np.flatnonzero(live[1:] & (lab[a + 1:b] != lab[a:b - 1])) + a + 1
        for t in chg:
            lo, hi = max(int(a), int(t) - half_width), min(int(b), int(t) + half_width)
            m = unc[lo:hi]
            lab[lo:hi][m] = 3
    return _as_out(lab, labels)


def apply(labels, runs, min_spell=DEFAULT["min_spell"], max_d=DEFAULT["max_d"],
          absorb=DEFAULT["absorb"], same_team_only=DEFAULT["merge_same_team_only"], enable=ENABLED,
          same_neighbors=False):
    """The E14 rule: short-spell floor (+ optional D merge) with the shipping defaults, behind ``ENABLED``."""
    if not enable:
        return labels
    return flicker(labels, runs, min_frames=min_spell, max_d=max_d, absorb=absorb,
                   same_team_only=same_team_only, same_neighbors=same_neighbors)


# -- diagnostics ---------------------------------------------------------------------------------

def diagnose(labels, runs, bins=(0, 1, 2, 5, 10, 20, 40, np.inf)):
    """Label-free spell / D-length summary inside the runs (used by the E14 tables). X is filled as the
    scorer fills it, so a rule that writes X is described by what the metric sees."""
    lab = _fill(to_codes(labels), runs)
    spells, durs = [], []
    for a, b in runs:
        for x, y, c in _segments(lab, a, b):
            if c in LIVE:
                spells.append((x, y, c))
                durs.append((y - x) / FPS)
    durs = np.array(durs, float)
    hist, _ = np.histogram(durs, bins=bins)
    d_stretches = [y - x for a, b in runs for x, y, c in _segments(lab, a, b) if c == 2]
    graded = sum(int(b) - int(a) for a, b in runs)
    return {
        "n_spells": len(spells),
        "n_spells_per_5min": len(spells) / (graded / FPS / 300.0) if graded else 0.0,
        "flickers_lt_1s": int((durs < 1.0).sum()),
        "flickers_lt_2s": int((durs < 2.0).sum()),
        "median_s": float(np.median(durs)) if len(durs) else 0.0,
        "share_gt_20s": float((durs > 20).mean()) if len(durs) else 0.0,
        "hist": {"%g-%s" % (lo, "inf" if np.isinf(hi) else "%g" % hi): int(v)
                 for lo, hi, v in zip(bins[:-1], bins[1:], hist)},
        "n_d_stretches": len(d_stretches),
        "d_median_s": float(np.median(d_stretches)) / FPS if d_stretches else 0.0,
    }
