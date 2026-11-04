"""Restart detection inside a dead-ball block: when play resumes and who restarts (E08 candidate).

**Status: research candidate, not imported by the build.** Nothing in ``make_submission`` or
``pipeline.build_labels`` imports this module, so it cannot change the frozen CSV. It exists to give
E37/E58 (and any later integration card) a measured restart interface instead of a hand-made one.

What is measured
----------------
* **Timing.** Inside each predicted D block, the restart is the last frame at which the ball
  *reappears* after at least ``GAP_S`` seconds of being untracked. This is the one tracking-only cue
  that beat the block's own end on external dev (10 games, seeds 0 and 1, 6 primary cells): it fires
  on 79-81% of blocks, and on the paired ``|block-end error| <= 3 s`` sample it cuts MAE from .732 to
  .686 s (seed 0) and .724 to .681 s (seed 1), raising the share within 1 s from .744 to .789 / .782.
  The bias moves from +.146 to -.013 s (seed 0); the gain is mostly in the tail where a grown/capped
  block runs past the ball gap's end. On blocks where the ball never disappeared there is no cue.
* **Team.** The restarting team is the team of the nearest player to the ball at the detected frame,
  provided that player is within ``TAKER_M`` and at least ``TAKER_MARGIN_M`` clear of the opponent's
  nearest player. When that test fails the module returns ``team = -1`` and the caller keeps the
  stack's team. On the same sample this raises team accuracy from .728 to .776 (seed 0) and .741 to
  .786 (seed 1) - +4.5 to +4.8 pp, game-cluster CI [+.036,+.065] / [+.030,+.059] - while touching
  only 12-13% of events. With the *true* restart frame the same cue is .831, so timing is the binding
  constraint, not the cue. It does **not** pay as a label override: replacing the first post-D spell's
  team lowers spellF1 by .007-.008 (10/10 and 9/10 games negative). Use the number as a measurement /
  prior, not as a reason to rewrite labels.

The E29 grammar prior (out-of-bounds stops flip the team, in-pitch stops usually keep it) is exposed
as ``choose_team(..., mode="prior")`` but is **not** the default: on external dev its accuracy is
.57-.58 against .73-.74 for the stack and .78 for geometry, because a predicted D block's pre-D owner is
not the last toucher (E29's finding still holds). The prior direction is real on the *true* pre-D
owner (OOB flip .792, in-pitch .672), but the predicted owner is too contaminated to use.

Kickoff-like restarts
---------------------
``kickoff_formation`` is the S09 signature (outfield players on their own half, ball near the center
spot), implemented with the match's *own* half array and attack direction so it can also run on
external dev. ``is_kickoff_like`` flags the kickoff-like subset (not only after-goal: a center-spot
free kick with both teams set in their own halves counts too). On external dev it fires on 35-37 of
~885 GT D ends (4%; 3-4 per game, median D duration 64 s), and on that subset the stack's team is
already right 96-100% and its D end is at the truth (median -0.08 s, 92-94% within 1 s). So
formation is an *existence/team* cue for E37, not a timing cue; a kickoff-specific center-displacement
detector got the median to -0.08 s on 13/18 flagged kickoffs in the exploratory pass but needs its own
card. The Smart Tagging cross-check on our match stays with E37, which owns the tag windows.

``labels`` is the shipped string array (``H``/``A``/``D``/``X``), one entry per video frame.
Nothing here reads a label from our match or the public window. Every threshold is either a
physical/Laws constant, inherited from a cited card, or selected on the external dev set (recorded
in ``research/E08_result.md``). Determinism: pure NumPy, no RNG, no I/O.
"""
from collections import namedtuple

import numpy as np

from . import deadball
from .structure import PITCH_L, PITCH_W

FPS = 25.0

# -- timing -------------------------------------------------------------------------------------
GAP_S = 0.4             # E08 dev: a reappearance after >= 0.4 s missing is the restart; shorter
                        # gaps are tracking flicker (dev sweep 0.2-1.0 s, flat around 0.4)
SEARCH_S = 8.0          # only the tail of a D block can contain the restart that ends it
PAD_S = 0.4             # allow a restart just after the stack's block end

# -- who restarts -------------------------------------------------------------------------------
TAKER_M = 4.0           # dev: 4 m covers the 0.4 s max timing offset at a 10 m/s release; the
                        # 2.5 m "at a player's feet" radius loses 2.3 pp of coverage
TAKER_MARGIN_M = 0.5    # a taker must be this much closer than the opponent's nearest player
TEAM_SPAN_F = 2         # nearest player is measured over +-2 frames (dev: pm2 was the best span)
UNKNOWN = -1

# -- restart geometry (Laws / pitch) ------------------------------------------------------------
OOB_M = 1.0             # E29/I01: last ball within 1 m of a line = out-of-bounds stop
LINE_M = 2.0            # a restart inside 2 m of a line is a throw-in/corner
GOAL_AREA_M = 5.5       # Laws: goal-area depth
CENTER_M = 9.15         # Laws: center-circle radius

# -- kickoff formation (S09) --------------------------------------------------------------------
OWN_HALF_FRAC = 0.80    # S09: 90-100% own-side at the Kick Off (After Goal) tags
KICKOFF_WINDOW_F = 25   # 1 s of "set" before the restart
KICKOFF_CENTER_M = 9.15 # a kickoff ball is at/near the center spot (Laws)

Restart = namedtuple("Restart", "start end frame team kind confidence evidence")
# start/end: the predicted D block [start, end) the restart belongs to (video frames)
# frame:    the estimated restart frame (may be < end when the ball reappears inside the block)
# team:     0/1, or UNKNOWN when no taker passes the confidence test
# kind:     kickoff | goal_kick | throw_in | corner | free_kick | unknown
# confidence: high | medium | low | block_end
# evidence: dict (see block_restart)


# ================================================================ helpers


def _runs(mask):
    """Maximal True runs as a (K, 2) array, end exclusive."""
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _graded_blocks(labels, half):
    """(start, end, half) for every D run inside a graded half."""
    labels = np.asarray(labels)
    half = np.asarray(half)
    out = []
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not rows.size:
            continue
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        for a, b in _runs(labels[lo:hi] == "D"):
            out.append((a + lo, b + lo, h))
    return out


def _ball_present(match):
    B = np.asarray(match.B, float)
    return np.isfinite(B[:, 0])


def reappearance_frame(match, start, end, gap_s=GAP_S, search_s=SEARCH_S, pad_s=PAD_S):
    """Last frame in the block's tail where the ball reappears after >= gap_s missing.

    Returns None when the ball was present throughout (no gap to end), in which case the block end
    is the only available estimate.
    """
    pres = _ball_present(match)
    n = len(pres)
    lo = max(0, int(start), int(end - search_s * FPS))
    hi = min(n, int(end + pad_s * FPS))
    gap = max(1, int(gap_s * FPS))
    best = None
    for t in range(lo, hi):
        if not pres[t]:
            continue
        if pres[max(0, t - gap):t].any():
            continue
        best = t
    return best


def nearest_player(match, frame, span=TEAM_SPAN_F):
    """Nearest tracked player to the ball around ``frame``.

    Returns dict(team, dist, margin, frame): ``team`` is UNKNOWN when the ball is missing; ``margin``
    is the opponent's nearest distance minus this one (NaN when the opponent has no tracked player).
    The frame with the smallest distance in ``frame +- span`` is used, so a one-frame ball teleport
    cannot decide the team.
    """
    P = np.asarray(match.P, float)
    B = np.asarray(match.B, float)
    T = np.asarray(match.T)
    n = len(B)
    best = None
    for k in range(max(0, int(frame) - span), min(n, int(frame) + span + 1)):
        if not np.isfinite(B[k, 0]):
            continue
        d = np.linalg.norm(P[k] - B[k], axis=1)
        d[~np.isfinite(d)] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]):
            continue
        if best is None or d[j] < best[0]:
            best = (float(d[j]), k, j)
    if best is None:
        return dict(team=UNKNOWN, dist=np.nan, margin=np.nan, frame=int(frame))
    d0, k, j = best
    team = int(T[j])
    opp = 1 - team
    dd = np.linalg.norm(P[k] - B[k], axis=1)
    dd[~np.isfinite(dd)] = np.inf
    same = dd[T == opp]
    margin = float(np.min(same) - d0) if same.size and np.isfinite(np.min(same)) else np.nan
    return dict(team=team, dist=d0, margin=margin, frame=int(k))


def _signed_line_distance(p, pitch=(PITCH_L, PITCH_W)):
    """Signed distance to the nearest pitch boundary (negative = outside)."""
    L, W = pitch
    return float(min(p[0], L - p[0], p[1], W - p[1]))


def _ball_at(match, frame, back_s=2.0):
    """Ball position at ``frame`` or the closest tracked ball within ``back_s`` before it."""
    B = np.asarray(match.B, float)
    frame = int(np.clip(frame, 0, len(B) - 1))
    if np.isfinite(B[frame, 0]):
        return B[frame].copy()
    lo = max(0, int(frame - back_s * FPS))
    fin = np.flatnonzero(np.isfinite(B[lo:frame + 1, 0]))
    return B[lo + fin[-1]].copy() if len(fin) else None


def restart_kind(ball, pitch=(PITCH_L, PITCH_W)):
    """Restart type from the ball's position (Laws/geometry; no labels)."""
    if ball is None:
        return "unknown"
    L, W = pitch
    x, y = float(ball[0]), float(ball[1])
    to_center = float(np.hypot(x - L / 2, y - W / 2))
    if to_center <= CENTER_M:
        return "kickoff"
    to_touch = min(y, W - y)
    to_goal = min(x, L - x)
    if (x <= GOAL_AREA_M or x >= L - GOAL_AREA_M) and to_touch <= 2 * GOAL_AREA_M:
        return "goal_kick"
    if to_touch <= LINE_M:
        return "throw_in"
    if to_goal <= LINE_M:
        return "corner"
    return "free_kick"


def pre_owner(labels, frame, half):
    """Last H/A label before ``frame`` in the same half, or UNKNOWN."""
    labels = np.asarray(labels)
    half = np.asarray(half)
    h = half[frame]
    for k in range(int(frame) - 1, -1, -1):
        if half[k] != h:
            break
        if labels[k] in ("H", "A"):
            return 0 if labels[k] == "H" else 1
    return UNKNOWN


def grammar_prior(pre_team, oob_onset):
    """E29's restart grammar: an out-of-bounds stop flips the pre-D owner; an in-pitch stop keeps it.

    Measured on external dev at the *true* pre-D owner (flip rate .792 OOB / .672 in-pitch) but
    **not** the default here: the predicted pre-D owner is too contaminated (E29).
    """
    if pre_team == UNKNOWN:
        return UNKNOWN
    return (1 - pre_team) if oob_onset else pre_team


# ================================================================ kickoff formation


def kickoff_formation(match, frame, st=None, window=KICKOFF_WINDOW_F, direction="back", ball_pad=10):
    """S09's kickoff signature over a window ending at ``frame`` (or starting there).

    Same geometry as :func:`possession.structure.kickoff_formation`, but the halves and attack
    direction come from the match itself (``deadball.match_structure``), so it is valid on external
    games too. Returns dict(own_half_frac, ball_to_center_m, n_tracked).
    """
    if st is None:
        st = deadball.match_structure(match)
    n = len(match.half)
    f = int(np.clip(frame, 0, n - 1))
    if direction == "back":
        sl = slice(max(0, f - window), f)
        bsl = slice(max(0, f - window), min(n, f + ball_pad + 1))
    else:
        sl = slice(f, min(n, f + window))
        bsl = sl
    P = np.asarray(match.P, float)[sl]
    B = np.asarray(match.B, float)[bsl]
    h = int(match.half[f])
    if h == 0 or P.size == 0:
        return dict(own_half_frac=np.nan, ball_to_center_m=np.nan, n_tracked=0.0)
    own = tot = 0
    for t in (0, 1):
        idx = [i for i in np.where(match.T == t)[0] if i != match.gk[h - 1, t]]
        x = P[:, idx, 0]
        side = st.attack_dir[h - 1, t]
        own += np.nansum((x - PITCH_L / 2) * side <= 1.0)  # 1 m slack for the center line
        tot += np.sum(~np.isnan(x))
    d = np.linalg.norm(B - np.array([PITCH_L / 2, PITCH_W / 2]), axis=1)
    return dict(own_half_frac=float(own / tot) if tot else 0.0,
                ball_to_center_m=float(np.nanmedian(d)) if np.any(np.isfinite(d)) else np.nan,
                n_tracked=float(np.median(np.sum(np.isfinite(P[..., 0]), axis=1))))


def is_kickoff_like(match, restart, st=None):
    """True when the 1 s before the restart looks like a kickoff (S09 formation + center ball)."""
    f = kickoff_formation(match, restart.frame, st=st)
    return bool(np.isfinite(f["own_half_frac"]) and f["own_half_frac"] >= OWN_HALF_FRAC
                and np.isfinite(f["ball_to_center_m"]) and f["ball_to_center_m"] <= KICKOFF_CENTER_M)


# ================================================================ the detector


def block_restart(match, labels, start, end, st=None, half=None, params=None):
    """Estimate the restart that ends the predicted D block ``[start, end)``.

    ``st`` is optional (only the kickoff-formation cross-check uses it). ``half`` defaults to
    ``match.half``. ``params`` may override any of the module constants (diagnostics only).
    """
    p = dict(gap_s=GAP_S, search_s=SEARCH_S, pad_s=PAD_S, taker_m=TAKER_M,
             taker_margin=TAKER_MARGIN_M, team_span=TEAM_SPAN_F, oob_m=OOB_M)
    p.update(params or {})
    labels = np.asarray(labels)
    n = len(labels)
    start, end = int(start), int(end)
    half = np.asarray(match.half if half is None else half)
    t = reappearance_frame(match, start, end, p["gap_s"], p["search_s"], p["pad_s"])
    found = t is not None
    frame = end if t is None else int(t)
    near = nearest_player(match, frame, p["team_span"])
    team = near["team"] if (found and near["dist"] <= p["taker_m"]
                            and np.isfinite(near["margin"]) and near["margin"] >= p["taker_margin"]) else UNKNOWN
    ball = _ball_at(match, frame)
    kind = restart_kind(ball)
    pre = pre_owner(labels, start, half)
    on = _ball_at(match, start, back_s=2.0)
    oob = bool(on is not None and _signed_line_distance(on) <= p["oob_m"])
    prior = grammar_prior(pre, oob)
    form = kickoff_formation(match, frame, st=st)
    kick = bool(np.isfinite(form["own_half_frac"]) and form["own_half_frac"] >= OWN_HALF_FRAC
                and np.isfinite(form["ball_to_center_m"]) and form["ball_to_center_m"] <= KICKOFF_CENTER_M)
    if not found:
        conf = "block_end"
    elif team == UNKNOWN:
        conf = "low"
    elif near["dist"] <= 2.0 and near["margin"] >= 1.0:
        conf = "high"
    else:
        conf = "medium"
    ev = dict(found=bool(found), gap_s=p["gap_s"], nearest_m=near["dist"], margin_m=near["margin"],
              nearest_frame=near["frame"], ball_xy=None if ball is None else [float(ball[0]), float(ball[1])],
              to_line_m=np.nan if ball is None else _signed_line_distance(ball),
              pre_team=pre, oob_onset=oob, prior_team=prior,
              own_half_frac=form["own_half_frac"], ball_to_center_m=form["ball_to_center_m"],
              kickoff_like=kick, structural=bool(st is not None and getattr(st, "forced_dead_mask", None)
                                                 is not None and len(st.forced_dead_mask) > start
                                                 and st.forced_dead_mask[start:end].all()))
    return Restart(start=start, end=end, frame=frame, team=int(team), kind=kind,
                   confidence=conf, evidence=ev)


def find_restarts(match, labels, st=None, params=None):
    """One :class:`Restart` per predicted D block in the graded halves."""
    half = np.asarray(match.half)
    if st is None:
        st = deadball.match_structure(match)
    return [block_restart(match, labels, a, b, st=st, half=half, params=params)
            for a, b, _ in _graded_blocks(labels, half)]


def choose_team(restart, base_team, mode="geometry"):
    """Team for the first post-D spell. ``base_team`` is the stack's own answer (0/1/UNKNOWN).

    mode:
      * ``geometry`` (default) - the taker's team when it passed the confidence test, else the stack;
      * ``prior``              - E29's grammar prior (kept for comparison; measured worse);
      * ``geometry_prior``     - geometry, else the prior, else the stack;
      * ``base``               - never override.
    """
    if mode == "base":
        return int(base_team)
    if mode == "prior":
        q = restart.evidence.get("prior_team", UNKNOWN)
        return int(q) if q != UNKNOWN else int(base_team)
    if mode == "geometry_prior":
        if restart.team != UNKNOWN:
            return int(restart.team)
        q = restart.evidence.get("prior_team", UNKNOWN)
        return int(q) if q != UNKNOWN else int(base_team)
    if mode != "geometry":
        raise ValueError("unknown team mode %r" % (mode,))
    return int(restart.team) if restart.team != UNKNOWN else int(base_team)


def override_first_spell(labels, restarts, mode="geometry", trim=False):
    """Copy of ``labels`` with the first post-D spell's team replaced (research convenience only).

    ``trim=True`` also moves the D end back to the detected restart frame, extending that spell
    earlier. Neither option is wired into the build; use this only from a measurement script.
    """
    out = np.array(labels, copy=True)
    code = {"H": 0, "A": 1, "D": 2, "X": 3}
    for r in restarts:
        if not (0 <= r.end < len(out)) or out[r.end] not in ("H", "A"):
            continue
        s = r.end
        if trim and r.frame < r.end:
            out[r.frame:r.end] = out[r.end]
            s = r.frame
        team = choose_team(r, code.get(out[s], UNKNOWN), mode)
        if team == UNKNOWN:
            continue
        e = s
        while e < len(out) and out[e] == out[s]:
            e += 1
        out[s:e] = "H" if team == 0 else "A"
    return out
