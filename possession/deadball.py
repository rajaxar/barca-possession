"""Fused dead-ball detector: intervals with onset and end estimates (E07).

The **primary component is not implemented here and is never redesigned**: it is E25's whole-gap
rule in :mod:`possession.dgate` -- a ball-missing run of at least 2 s whose mean player-only
``P(D)`` exceeds 0.5 becomes ``D`` in full, with present H/A islands shorter than 4 s absorbed.
The shipped stack (``dgate`` -> ``growd`` -> ``ingap`` -> ``subgap``) consumes that rule and returns
a label array. This module turns that array into a **D interval list** (onset, end estimate,
source, confidence) and applies only the sharpeners that survived their pre-registered test
(``scratch/E07/prereg.md``, ``research/E07_result.md``).

Nothing here fits a threshold to a score, reads the public GT, or re-tunes the shipped components.
Every constant is a football/physics constant or is inherited from the card that measured it.

What ships in v1
----------------
Nothing changes the label array by default: every sharpener below is behind a flag that is **off**.
The pre-registered tests (`scratch/E07/prereg.md`, K = 5) are reported in `research/E07_result.md`;
the two sharpeners that pass on dev evidence still fail their own label-free bar, so the conservative
default is the one the card asks for ("default OFF unless a cue passes tier 1").

* **S1 ball out of bounds (outside the pitch)** -- ``OOB_ENABLED = False``. A missing run 2-30 s whose
  last tracked ball is **outside the pitch rectangle by** ``OOB_MARGIN_M`` is D; the onset hint is the
  last sighting. Dev: +.0006 spellF1 [+.0003,+.0009] (all 6 primary + 4 stress cells >= 0, 1/10 games
  negative, holds at +-0.5 s) and +.0038 D-interval F1 -- but the pre-registered label-free bar fails:
  the rescued frames score mean P(D) .41 (bar > .5) because they are throw-ins where the players walk.
  Footprint on our match is 280 graded frames (0.18%) in 4 of 113 intervals; the size is 60x below the
  noise band. Re-test: E52 (broadcast out-of-pitch ball) or a labeled game with verified out-of-play
  geometry, then this flag can be flipped with one line.
* **S4 collective-stillness end estimate** -- ``END_ESTIMATE_ENABLED = False``. The end of a D interval
  only ever moves to the first sustained team-speed burst with the ball present in its last 15 s (the
  hand-off to E08; it does **not** assign the restarting team). It failed its bar: D-interval F1
  -.0048 (IoU .5, seed 0, 6 primary cells; worse in all 6), mean matched IoU +.0009, and the frames it
  trims still score P(D) .58 -- the burst it finds is usually inside the stoppage. Kept as a function.
* **S2 whistle onset** -- tier 2, behind ``WHISTLE_ENABLED`` (default off). Match-only cue: it extends
  a *core-D* interval back to a whistle over still frames only, so audio can never declare D by itself.
  Lead contrast +.1001 [+.0093,+.1966] and introduced frames P(D) .958 pass, but only 1 of 9 candidate
  intervals extends (0 have a still path to the whistle) -> the pre-registered ">= 5 intervals" fails.
  Re-tested by E88.
* **S5 keeper live override** -- :func:`keeper_override` is the interval operation E38 asked for (an
  episode **ends** a D interval). ``KEEPER_ENABLED`` is off: no episode overlaps a D interval on our
  match and the dev delta is -.0001 [.0004,+.0001]. Exported so a consumer can switch it on.
* **S3 Smart Tagging set-piece prior** -- tested and **dropped** (D-rate contrast -.137 [-.266,-.012];
  the windows are *less* dead than average, reproducing R13). :func:`set_piece_mask` is a diagnostic.

Determinism: pure NumPy, no RNG, no I/O; ``detect`` is a function of its arguments.

Interval operations
-------------------
A D interval is a half-open ``[start, end)`` frame span **inside one half**. ``onset`` and
``end_est`` are estimates in the same frame convention; they may lie outside ``[start, end)`` only
for an onset hint that predates the label start (OOB last sighting / whistle). The label array is
changed only by S1, S2 and (if enabled) S5; S4 changes ``end_est`` alone.
"""
from collections import namedtuple

import numpy as np

from . import dproxy
from .structure import PITCH_L, PITCH_W, Structure

FPS = 25.0

# ---------------------------------------------------------------- inherited / physical constants
OOB_MARGIN_M = 0.5        # "outside the rectangle", not "on the line": R13's error mode is a carried ball
OOB_MAX_GAP_S = 30.0      # an out-of-play gap is short; longer runs are injury/break mechanisms (E36)
OOB_SEARCH_S = 2.0        # search for the last tracked ball before a gap
OOB_SPEED_MS = 8.0        # reported near-line check only (E36/R13's physical out-of-play rule)
WHISTLE_LEAD_S = 4.0      # E15/R06: a whistle can precede the tracking state by ~2 s
WHISTLE_LAG_S = 0.5       # E15's onset resolution; a whistle after the interval starts is not an onset
WHISTLE_STILL_V = 1.0     # E39: below 1.0 m/s team speed is the safe still band (1.5 is a loss)
STILL_END_V = 2.0         # restart burst: a jog/sprint, well above the 1.5 m/s walk boundary
STILL_END_HOLD_S = 0.5    # a burst, not a single tracking spike
STILL_END_TAIL_S = 15.0   # only look at the tail of a stoppage
STILL_END_MIN_D_S = 2.0   # never trim a stoppage shorter than the E25 duration floor

OOB_ENABLED = False       # S1: dev-positive but its pre-registered label-free bar failed (.41 < .5)
END_ESTIMATE_ENABLED = False  # S4: failed its bar (D-interval F1 -.0048, worse in all 6 cells)
WHISTLE_ENABLED = False   # S2: match-only, tier 2 under D-014; 1/9 intervals extend (bar was >= 5)
KEEPER_ENABLED = False    # S5: no-op here (0/14 episodes touch a D interval; dev -.0001)

DEFAULT_FLAGS = dict(oob=OOB_ENABLED, whistle=WHISTLE_ENABLED, end_estimate=END_ESTIMATE_ENABLED,
                     keeper=KEEPER_ENABLED)

DeadBall = namedtuple("DeadBall", "labels mask intervals info")
# labels:  (N,) the input labels with S1/S2/S5 applied
# mask:    (N,) bool, final D frames (graded halves only)
# intervals: list of dict(start, end, onset, end_est, seconds, half, sources, mean_pd, structural)
# info:    dict(flags, params, n_intervals, d_share, oob_gaps, oob_applied, whistle_applied,
#               keeper_overrides, n_onset_moved, n_end_moved)


# ================================================================ helpers


def _runs(mask):
    """Maximal True runs as an (K, 2) array, end exclusive."""
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _graded_half_runs(half):
    half = np.asarray(half)
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if rows.size:
            yield h, int(rows[0]), int(rows[-1]) + 1


def match_structure(match):
    """A :class:`structure.Structure` for any Match (no BAR-ATH frame constants).

    ``gk_possession`` needs ``halves``, ``attack_dir`` and ``forced_dead_mask``; for an external
    game those come from its own ``half`` array (BAR-ATH's published blocks must not be reused).
    """
    half = np.asarray(match.half)
    halves = {}
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        halves[h] = (int(rows[0]), int(rows[-1])) if rows.size else (0, -1)
    attack = np.zeros((2, 2), np.int8)
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not rows.size:
            continue
        for t in (0, 1):
            s = int(match.gk[h - 1, t])
            x = np.nanmean(np.asarray(match.P)[rows, s, 0])
            attack[h - 1, t] = 1 if x < PITCH_L / 2 else -1
    return Structure(halves=halves, kickoffs={}, attack_dir=attack,
                     forced_dead_mask=np.zeros(len(half), bool))


def ball_speed(B):
    """(N,) robust ball speed in m/s (0.5 s baseline, NaN when the ball is missing)."""
    return np.asarray(dproxy._robust_ball_speed(np.asarray(B, float)), np.float32)


def interval_list(labels, half, st=None, pf=None):
    """D intervals per half from a label array; no cue is applied here.

    ``st`` (optional) marks intervals wholly inside the published forced-dead block as
    ``structural``; ``pf`` (optional) records the mean player-only P(D) of each interval.
    """
    labels = np.asarray(labels)
    half = np.asarray(half)
    forced = (np.asarray(st.forced_dead_mask, bool) if st is not None
              else np.zeros(len(labels), bool))
    out = []
    for h, lo, hi in _graded_half_runs(half):
        for a, b in _runs(labels[lo:hi] == "D"):
            a, b = a + lo, b + lo
            mean_pd = None
            if pf is not None:
                q = np.asarray(pf)[a:b]
                mean_pd = float(np.nanmean(q)) if np.isfinite(q).any() else None
            out.append(dict(start=int(a), end=int(b), onset=int(a), end_est=int(b),
                            seconds=float((b - a) / FPS), half=int(h),
                            sources=[], mean_pd=mean_pd,
                            structural=bool(forced[a:b].all())))
    return out


def _pre_label(labels, a, half):
    """Last H/A label before ``a`` in the same half (fallback 'D' if there is none)."""
    h = half[a]
    for k in range(a - 1, -1, -1):
        if half[k] != h:
            break
        if labels[k] in ("H", "A"):
            return labels[k]
    return "D"


# ================================================================ S1: ball out of bounds


def oob_gaps(B, half, pitch=(PITCH_L, PITCH_W), min_s=2.0, max_s=OOB_MAX_GAP_S,
             margin=OOB_MARGIN_M, search_s=OOB_SEARCH_S, vball=None):
    """Every ball-missing run with its last tracked ball and an out-of-pitch verdict.

    Returns a list of dicts (one per missing run in a graded half):
    ``start, end, seconds, last_frame, x, y, outside, line_m, ball_speed, qualifies``.
    ``qualifies`` is the S1 label rule: outside the rectangle by ``margin`` and 2-30 s long.
    """
    B = np.asarray(B, float)
    half = np.asarray(half)
    v = ball_speed(B) if vball is None else np.asarray(vball, float)
    L, W = pitch
    out = []
    for h, lo, hi in _graded_half_runs(half):
        for a, b in _runs(~np.isfinite(B[lo:hi, 0])):
            a, b = a + lo, b + lo
            k, lim = a - 1, max(lo, a - int(round(search_s * FPS)))
            while k >= lim and not np.isfinite(B[k, 0]):
                k -= 1
            if k < lim or not np.isfinite(B[k, 0]):
                continue
            x, y = float(B[k, 0]), float(B[k, 1])
            outside = bool(x < -margin or x > L + margin or y < -margin or y > W + margin)
            line = float(min(abs(x), abs(x - L), abs(y), abs(y - W)))
            speed = float(v[k]) if np.isfinite(v[k]) else float("nan")
            dur = (b - a) / FPS
            out.append(dict(start=int(a), end=int(b), seconds=float(dur), half=int(h),
                            last_frame=int(k), x=x, y=y, outside=outside, line_m=line,
                            ball_speed=speed,
                            qualifies=bool(outside and min_s - 1e-9 <= dur <= max_s)))
    return out


def oob_override(labels, B, half, pitch=(PITCH_L, PITCH_W), gaps=None, **kw):
    """S1: make every qualifying out-of-play gap D in full; returns ``(labels, gaps)``.

    Only H/A frames are overwritten (X/ungraded and structural frames are left alone). The
    interval onset hint is the gap's last tracked ball (see :func:`detect`).
    """
    out = np.asarray(labels).copy()
    gaps = oob_gaps(B, half, pitch=pitch, **kw) if gaps is None else gaps
    for g in gaps:
        if not g["qualifies"]:
            continue
        seg = out[g["start"]:g["end"]]
        seg[np.isin(seg, ("H", "A"))] = "D"
    return out, gaps


# ================================================================ S2: whistle onset


def _event_onsets(events):
    """Frame onsets from E15's event array / DataFrame / sequence (``start_frame`` preferred)."""
    if events is None:
        return np.zeros(0, np.int64)
    if isinstance(events, np.ndarray) and events.dtype.names:
        key = "start_frame" if "start_frame" in events.dtype.names else "peak_frame"
        return np.asarray(events[key], np.int64)
    try:  # pandas DataFrame
        key = "start_frame" if "start_frame" in events.columns else "peak_frame"
        return np.asarray(events[key], np.int64)
    except AttributeError:
        return np.asarray(events, np.int64).reshape(-1)


def whistle_extend(labels, half, events, speed, lead_s=WHISTLE_LEAD_S, lag_s=WHISTLE_LAG_S,
                   still_v=WHISTLE_STILL_V):
    """S2: extend a core-D interval back to a whistle over still frames only.

    For each D interval starting at ``s`` with an event onset in ``[s - lead, s + lag]``, the
    interval is extended to that onset over the contiguous run of frames ending at ``s`` whose
    mean team speed is below ``still_v`` and whose label is H/A. Returns ``(labels, applied)``;
    ``applied`` records ``start`` (old), ``new_start`` and ``whistle`` (frame). Audio never
    creates a D interval: it only moves an existing one.
    """
    out = np.asarray(labels).copy()
    half = np.asarray(half)
    onsets = _event_onsets(events)
    applied = []
    if onsets.size == 0:
        return out, applied
    speed = np.asarray(speed, float)
    for h, lo, hi in _graded_half_runs(half):
        for a, b in _runs(out[lo:hi] == "D"):
            s = a + lo
            cand = onsets[(onsets >= s - int(round(lead_s * FPS)))
                          & (onsets <= s + int(round(lag_s * FPS)))]
            if cand.size == 0:
                continue
            w = int(cand.max())
            a_new = max(int(w), s - int(round(lead_s * FPS)), lo)
            ok = (np.isin(out[a_new:s], ("H", "A"))
                  & np.isfinite(speed[a_new:s]) & (speed[a_new:s] < still_v))
            j = s - 1
            while j >= a_new and ok[j - a_new]:
                j -= 1
            a_new = j + 1
            if a_new >= s:
                continue
            out[a_new:s] = "D"
            applied.append(dict(start=int(s), new_start=int(a_new), whistle=int(w)))
    return out, applied


# ================================================================ S4: stillness end estimate


def stillness_end(labels, B, speed, half, tail_s=STILL_END_TAIL_S, v=STILL_END_V,
                  hold_s=STILL_END_HOLD_S, min_d_s=STILL_END_MIN_D_S, require_present=True):
    """S4: the restart burst inside each D interval as an **end estimate** (no label change).

    Returns ``{(start, end): end_estimate}`` for intervals where a burst is found. The burst is
    the first frame in ``[max(start+2 s, end-15 s), end)`` with the ball present and the next
    ``hold_s`` of mean team speed all >= ``v``. This is the hand-off to E08, which owns the exact
    restart touch and the restarting team.
    """
    B = np.asarray(B, float)
    speed = np.asarray(speed, float)
    half = np.asarray(half)
    present = np.isfinite(B[:, 0])
    fast = np.isfinite(speed) & (speed >= v)
    hold = max(1, int(round(hold_s * FPS)))
    out = {}
    for h, lo, hi in _graded_half_runs(half):
        for a, b in _runs(np.asarray(labels)[lo:hi] == "D"):
            a, b = a + lo, b + lo
            if (b - a) / FPS < min_d_s:
                continue
            start = max(a + int(round(2 * FPS)), b - int(round(tail_s * FPS)))
            for t in range(start, b - hold + 1):
                if require_present and not present[t]:
                    continue
                if fast[t:t + hold].all():
                    out[(int(a), int(b))] = int(t)
                    break
    return out


# ================================================================ S5: keeper live override


def _column(table, name):
    """A column from a pandas DataFrame, a dict-like or an attribute-style object."""
    try:
        return np.asarray(table[name])
    except (KeyError, TypeError, IndexError):
        return np.asarray(getattr(table, name))


def set_piece_mask(tags, n):
    """(N,) bool from Smart Tagging rows whose code is ``SET PIECES`` (diagnostic only)."""
    m = np.zeros(int(n), bool)
    codes = [str(c).upper() for c in _column(tags, "code")]
    sel = np.array(["SET PIECE" in c for c in codes], bool)
    for a, b in zip(_column(tags, "start_frame")[sel], _column(tags, "end_frame")[sel]):
        m[max(0, int(a)):min(int(n), int(b))] = True
    return m


def keeper_override(mask, live):
    """S5: a keeper-in-possession episode **ends** overlapping D intervals at its start.

    ``mask`` is the D mask; ``live`` is E38's ``gp.live`` (or any bool/float live mask). Returns
    ``(mask, overrides)`` with one dict per interval that was cut. This is deliberately an
    interval operation, not a middle carve (E38: "the episode must be able to end a D interval").
    """
    out = np.asarray(mask, bool).copy()
    live = np.asarray(live) > 0
    overrides = []
    for a, b in _runs(out):
        idx = np.flatnonzero(live[a:b])
        if idx.size == 0:
            continue
        cut = a + int(idx[0])
        out[cut:b] = False
        overrides.append(dict(start=int(a), end=int(b), cut=int(cut)))
    return out, overrides


# ================================================================ fusion


def detect(labels, match, st=None, pf=None, pitch=None, events=None, gk=None, gk_live=None,
           oob=None, whistle=None, end_estimate=None, keeper=None):
    """Fuse the shipped D labels into intervals with onset/end estimates.

    ``labels``: (N,) H/A/D/X from the shipped stack (already D-gated; this module never runs the
    primary rule itself). ``pf``: optional per-frame P(D) for interval confidence. ``events``:
    E15 whistle events (tier-2 S2, only if ``whistle`` is true). ``gk`` / ``gk_live``: E38 output
    or mask (S5, only if ``keeper`` is true). Flags default to the module constants; pass
    ``False`` to ablate. ``st`` supplies the forced-dead mask (default: computed from ``half``).
    Returns a :class:`DeadBall`.
    """
    labels = np.asarray(labels).copy()
    half = np.asarray(match.half)
    B = np.asarray(match.B)
    if st is None:
        st = match_structure(match)
    if pitch is None:
        pitch = (PITCH_L, PITCH_W)
    flags = dict(DEFAULT_FLAGS)
    flags.update({k: v for k, v in dict(oob=oob, whistle=whistle, end_estimate=end_estimate,
                                        keeper=keeper).items() if v is not None})

    gaps, applied_w, overrides = [], [], []
    speed = dproxy.mean_player_speed(match)

    if flags["oob"]:
        labels, gaps = oob_override(labels, B, half, pitch=pitch, gaps=gaps or None)
    if flags["whistle"] and events is not None:
        labels, applied_w = whistle_extend(labels, half, events, speed)
    if flags["keeper"]:
        live = gk_live
        if live is None and gk is not None:
            live = np.asarray(gk.live) > 0
        if live is not None:
            d = np.asarray(labels) == "D"
            d2, overrides = keeper_override(d, live)
            cut = d & ~d2
            for o in overrides:                      # the keeper's team owns the live frames
                team = _episode_team(gk, o["cut"])
                fill = ("H" if team == 0 else "A") if team in (0, 1) else _pre_label(labels, o["start"], half)
                labels[o["cut"]:o["end"]] = fill
            assert not (cut & (np.asarray(labels) == "D")).any()

    intervals = interval_list(labels, half, st=st, pf=pf)
    # onset hints: OOB last sighting, then a whistle extension (later start wins neither: keep min)
    oob_by_start = {g["start"]: g for g in gaps if g["qualifies"]}
    whis_by_start = {w["start"]: w for w in applied_w}
    n_on = n_end = 0
    for d in intervals:
        hint = d["start"]
        g = oob_by_start.get(d["start"])
        if g is not None:
            hint = min(hint, int(g["last_frame"]))
            d["sources"].append("oob")
        w = whis_by_start.get(d["start"])
        if w is not None:
            hint = min(hint, int(w["new_start"]))
            d["sources"].append("whistle")
        if hint != d["onset"]:
            d["onset"] = int(hint)
            n_on += 1
        if flags["end_estimate"]:
            e = stillness_end(labels, B, speed, half).get((d["start"], d["end"]))
            if e is not None:
                d["end_est"] = int(e)
                d["sources"].append("stillness_end")
                n_end += 1
    mask = np.asarray(labels) == "D"
    graded = np.asarray(half) > 0
    n = max(1, int(graded.sum()))
    info = dict(flags=flags, pitch=pitch, n_intervals=len(intervals),
                d_share=float((mask & graded).sum()) / n,
                oob_gaps=gaps, oob_applied=[g for g in gaps if g["qualifies"]],
                whistle_applied=applied_w, keeper_overrides=overrides,
                n_onset_moved=n_on, n_end_moved=n_end)
    return DeadBall(labels, mask, intervals, info)


def _episode_team(gk, frame):
    """Team of the E38 episode covering ``frame`` (None when unavailable)."""
    if gk is None:
        return None
    for ep in getattr(gk, "episodes", []) or []:
        if int(ep["start"]) <= frame < int(ep["end"]):
            return ep.get("team")
    return None


def apply(labels, match, **kw):
    """Deterministic pipeline entry point: returns the fused label array only."""
    return detect(labels, match, **kw).labels
