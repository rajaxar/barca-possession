"""Grown present-ball dead-ball D (E47 anchor + E61/E65 growth, shipped as E66), run on top of the D-gated labels.

E47 anchor: a tracked ball that rests (0.5 s-baseline speed <= 1 m/s for >= 1 s) while the players are
slow (team mean speed < ``G``) and no plausible keeper is within ``VETO_M`` is a dead ball. Such a
resting moment is only the visible part of a stoppage, so each anchor run is grown backwards and forwards
until the stoppage ends: the ball leaves ``R_M`` metres of its rest point, the team mean speed reaches
``S_JUMP`` (sprint / restart), the team stays >= ``S_HI`` for ``H_F`` frames, or the ball runs faster than
``S_BALL`` while the team is >= ``S_HI`` (both move). Each side grows at most ``CAP_S`` seconds. Frames with the
ball missing or a NaN speed are 'ok'. Grown frames become D and H/A islands shorter than ``ABSORB``
frames between D stretches are absorbed (as :mod:`possession.dgate`).

Ball speed only (no acceleration, jerk or texture; E59). Only safe after the missing-ball gate (E65: on raw
Baseline+ 55% of its changes fall in dead gaps); :func:`footprint` / :func:`check_footprint` are the
label-free build guard (no public GT). ``R_M = 12`` with the same cap is the pre-registered fallback.
Default off/on per ``ENABLED``; :func:`apply` returns its input untouched unless enabled.
"""
import numpy as np

from . import dgate, dproxy, structure

ENABLED = True                   # E66: on after the BAR-ATH label-free precision check (scratch/E66/barath.json); False = shipped gate+ingap
R_M = 25.0                       # E65 ship radius (m); 12.0 = pre-registered fallback
CAP_S = 30                       # per-side growth cap, seconds (E65)
FPS = 25
G = 1.0                          # anchor team-speed cap (E47 g1_v2)
VETO_M = 2.0                     # anchor keeper-approach veto (E47)
PLAUS_M = 45.0                   # keeper stands within 45 m of own goal (gk_possession.PLAUS_M)
S_HI = 1.5                       # team mean speed above which a frame is 'fast' (E39/E42)
S_JUMP = 3.0                     # immediate stop
S_BALL = 3.0                     # both-move stop: ball faster than this while the team is >= S_HI
H_F = 25                         # hysteresis frames
HOLD_F = int(round(dproxy.MIN_HOLD_S * FPS))
ABSORB = dgate.ABSORB
MAX_D_SHARE_ADDED = 0.029        # guard: E65 measured +2.90 pp at R25 (gated analogue)
MIN_D_SHARE_ADDED = 0.005        # ... and a dead rule is as suspect as a runaway one
MAX_FLICKER_ADDED = 5            # spells < 1 s added (E65: +3 at R25)


def _runs(mask):
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _keep_long(mask, k=HOLD_F):
    out = np.zeros(len(mask), bool)
    for a, b in _runs(mask):
        if b - a >= k:
            out[a:b] = True
    return out


def _keeper_dist(match, pitch):
    """(N,) ball distance to the nearer *plausible* keeper (own-goal distance < PLAUS_M); inf if none."""
    L = pitch[0]
    out = np.full(len(match.B), np.inf, np.float32)
    half = np.asarray(match.half)
    for h in (1, 2):
        sel = np.flatnonzero(half == h)
        if sel.size == 0:
            continue
        for t in (0, 1):
            kp = match.P[sel, int(match.gk[h - 1, t])]
            gx = 0.0 if np.nanmean(kp[:, 0]) < L / 2 else L
            plaus = np.hypot(kp[:, 0] - gx, kp[:, 1] - pitch[1] / 2) < PLAUS_M
            d = np.linalg.norm(kp - match.B[sel], axis=1)
            out[sel] = np.minimum(out[sel], np.where(plaus & np.isfinite(d), d, np.inf))
    return out


def prep(match):
    """Per-frame ball and team-speed inputs of the rule."""
    B = np.asarray(match.B, float)
    return dict(B=B, spd=dproxy.mean_player_speed(match),
                vball=dproxy._roll_median(dproxy._robust_ball_speed(B), dproxy.BALL_MED_F),
                kd=_keeper_dist(match, (structure.PITCH_L, structure.PITCH_W)),
                graded=np.asarray(match.half) > 0)


def anchor(pre):
    """E47 g1_v2 present-ball resting-ball anchor mask."""
    rest = _keep_long(pre["graded"] & np.isfinite(pre["B"][:, 0]) & np.isfinite(pre["vball"])
                      & (pre["vball"] <= dproxy.BALL_STILL_V))
    return _keep_long(rest & np.isfinite(pre["spd"]) & (pre["spd"] < G) & ~(pre["kd"] < VETO_M))


def grow(pre, anc, r=R_M, cap_s=CAP_S):
    """Anchor runs grown until the stoppage ends (see module docstring)."""
    B, spd, graded, vb = pre["B"], pre["spd"], pre["graded"], pre["vball"]
    n, cap = len(anc), int(round(cap_s * FPS))
    present = np.isfinite(B[:, 0])
    out = anc.copy()
    for a, b in _runs(anc):
        c = np.nanmedian(B[a:b], axis=0)
        for direction in (-1, +1):
            t0 = a - 1 if direction < 0 else b
            t = t0
            last_ok = a if direction < 0 else b - 1
            streak = 0
            while 0 <= t < n and graded[t] and abs(t - t0) < cap:
                if present[t] and np.hypot(*(B[t] - c)) > r:
                    break
                sp = spd[t]
                if np.isfinite(sp):
                    if sp >= S_JUMP:
                        break
                    if sp >= S_HI:
                        if np.isfinite(vb[t]) and vb[t] > S_BALL:
                            break
                        streak += 1
                        if streak >= H_F:
                            break
                        t += direction
                        continue
                streak = 0
                last_ok = t
                t += direction
            if direction < 0:
                out[last_ok:a] = True
            else:
                out[b:last_ok + 1] = True
    return out


def grown_mask(match, r=R_M, cap_s=CAP_S):
    pre = prep(match)
    return grow(pre, anchor(pre), r, cap_s)


def footprint(before, after, half, dead=None):
    """Label-free change footprint over graded frames: D share added, spells < 1 s added, changes inside ``dead``."""
    before, after = np.asarray(before), np.asarray(after)
    graded = np.asarray(half) > 0
    n = max(1, int(graded.sum()))
    chg = (before != after) & graded
    return dict(graded=int(graded.sum()), changed=int(chg.sum()), changed_share_graded=float(chg.sum()) / n,
                d_added_share_graded=float(((after == "D") & graded).sum() - ((before == "D") & graded).sum()) / n,
                flickers_added=_flickers(after, half) - _flickers(before, half),
                changed_in_dead=int((chg & dead).sum()) if dead is not None else None)


def _flickers(labels, half):
    """H/A spells (maximal same-team runs) shorter than 1 s; X is not forward-filled here."""
    n = 0
    lab = np.asarray(labels)
    for h in (1, 2):
        rows = np.flatnonzero(np.asarray(half) == h)
        if not len(rows):
            continue
        seg = lab[rows[0]:rows[-1] + 1]
        cut = np.r_[0, np.flatnonzero(seg[1:] != seg[:-1]) + 1, len(seg)]
        n += int(sum(1 for a, b in zip(cut[:-1], cut[1:]) if seg[a] in ("H", "A") and b - a < FPS))
    return n


def check_footprint(fp):
    """Raise on a runaway or dead rule, on added flickers, or on any change inside an E25 dead gap."""
    s = fp["d_added_share_graded"]
    if not MIN_D_SHARE_ADDED <= s <= MAX_D_SHARE_ADDED:
        raise RuntimeError("growd added D to %.4f of graded frames, outside [%.3f, %.3f]"
                           % (s, MIN_D_SHARE_ADDED, MAX_D_SHARE_ADDED))
    if fp["flickers_added"] > MAX_FLICKER_ADDED:
        raise RuntimeError("growd added %d spells < 1 s (> %d)" % (fp["flickers_added"], MAX_FLICKER_ADDED))
    if fp["changed_in_dead"]:
        raise RuntimeError("growd changed %d frames inside E25 dead gaps" % fp["changed_in_dead"])
    return fp


def apply(labels, match, dead=None, enabled=None, r=None):
    """Flag-gated entry point: ``labels`` must be D-gated. Returns ``(labels, footprint or None)``."""
    if not (ENABLED if enabled is None else enabled):
        return labels, None
    mask = grown_mask(match, R_M if r is None else r)
    out = np.asarray(labels).copy()
    out[mask] = "D"
    out = dgate.absorb_islands(out, ABSORB)
    return out, check_footprint(footprint(labels, out, match.half, dead))
