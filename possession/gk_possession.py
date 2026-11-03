"""Goalkeeper in possession: a label-free "this is live play" signal (E38).

The dead-ball cues we rely on are *slow players* and *a missing ball*. A goalkeeper holding,
bouncing or carrying the ball satisfies both the "slow" half and, when the tracker loses it, the
"missing" half — yet it is live possession and must not become ``D``. This module detects that
state from tracking alone and emits evidence that cancels a stillness-driven D score.

Two mechanisms, both conservative by construction:

``carry``  the ball is tracked within ``r`` meters of a *plausible* keeper, the keeper is walking,
           and the ball is slow (a seed) or moving with him (extension). Only accepted when
           (a) a whistle did not just sound, (b) the ball was not out of play in the last 3 s,
           (c) the ball was continuously tracked for 3 s before (so this is not a restart after a
           stoppage), and (d) the ball leaves at ``release_v`` m/s within 2 s — a throw or punt.
           Length is capped at 7 s: the Laws allow a keeper six seconds with the ball in his hands.
``gathered``  the ball disappears while within ``miss_r`` of a plausible keeper after continuous
           tracked play, for at most 6 s, and reappears moving fast (a throw/punt). A long gap is
           a stoppage (E36: P(D) rises with gap length), so it is deliberately not accepted.

No labels, no public GT, no scores are read. Every constant is a football or physics constant
(ball radius, walking speed, the six-second law, a punt's speed), not a fitted threshold.

The GK slot is wrong on ~5.7% of ATD frames (E16 -> E30), so the slot is *not* trusted blindly:
a per-frame plausibility band (``PLAUS_M`` from that team's own goal, just under halfway) rejects
an impostor slot, and when neither keeper slot is plausible the frame produces nothing. The
``info`` dict reports how often that happens, so a consumer can see the coverage cost.

Interface for E07::

    from possession import gk_possession as gk
    gp = gk.detect(match)          # GkPossession(live, d_delta, episodes, info)
    score_D += gp.d_delta          # (N,) float32, <= 0, cancels stillness evidence only

``live`` is 1.0 exactly on the detected frames and 0 elsewhere. ``d_delta`` is
``-LIVE_GAIN`` on carried-ball frames and ``-LIVE_GAIN * GATHERED_WEIGHT`` on the missing-ball
branch; both are 0 on ungraded and published forced-dead frames.
"""
from collections import namedtuple

import numpy as np

from .structure import PITCH_L, PITCH_W, structure

FPS = 25.0

# A keeper stands at his own goal; 45 m is just under halfway, so a slot reused by an outfielder in
# the opposition half fails, while a keeper genuinely caught upfield still passes.
PLAUS_M = 45.0
# Walking/jogging boundary used by E15/E25 for "still"; a keeper carrying the ball walks.
V_GK = 1.5
# Extension (a carry) uses a looser cap: a keeper jogging with the ball still has it, and the seed
# already required a slow ball. 2.5 m/s is below a sprint.
V_GK_EXT = 2.5
# A ball being carried is slow but ticks up between touches; 3 m/s is a brisk walk with the ball.
V_BALL = 3.0
# Half-length of the smoothing medians (frames): keeper speed, ball speed.
GK_MED = 9
BALL_MED = 5
MIN_S = 0.6          # 15 frames: rejects a pass rolling past the keeper
MAX_S = 6.0          # Laws of the Game: a keeper may hold the ball six seconds
WHISTLE_WIN = (-2.0, 0.5)
OOB_LOOK_S = 3.0     # a goal kick follows the ball going out; a live catch does not
PRE_PRESENT_S = 3.0  # continuous tracked ball before onset: not a restart
MIN_MOVE_M = 1.0     # the ball came to the keeper; a placed ball has not moved
MAX_STEP_MS = 25.0   # reject a placement/teleport disguised as an arrival
RELEASE_V = 4.0      # a throw/punt leaves the keeper fast
RELEASE_S = 2.0
MISS_R = 2.0
MISS_MAX_S = 6.0
MERGE_GAP_S = 0.3
# Log-odds cap for E07. About -1.2 cancels a stillness cue (E25's still-present P(D) 0.63 is
# +0.52 log-odds; a slow-present frame is around +1.1) without overriding a missing-ball or
# whistle cue in a real stoppage.
LIVE_GAIN = 1.2
# The gathered branch is weaker: a short missing gap with a fast return can also be a quick restart
# after a stoppage, so its evidence is half. A missed hold costs nothing; a false live in a real
# stoppage costs a broken spell, so the two branches are not weighted alike.
GATHERED_WEIGHT = 0.5

PARAMS = dict(plaus_m=PLAUS_M, v_gk=V_GK, v_gk_ext=V_GK_EXT, v_ball=V_BALL, r=2.0,
              min_s=MIN_S, max_s=MAX_S,
              whistle_win=WHISTLE_WIN, oob_look_s=OOB_LOOK_S, pre_present_s=PRE_PRESENT_S,
              min_move_m=MIN_MOVE_M, max_step_ms=MAX_STEP_MS, release_v=RELEASE_V,
              release_s=RELEASE_S, miss_r=MISS_R, miss_max_s=MISS_MAX_S, merge_gap_s=MERGE_GAP_S,
              extend=True, require_release=True, use_ball_motion=True,
              include_gathered=False)

GkPossession = namedtuple("GkPossession", "live d_delta episodes info")
# live: (N,) float32, 1.0 = keeper in possession
# d_delta: (N,) float32, additive log-odds for E07 (<= 0)
# episodes: list of dicts(start, end, seconds, team, kind, release_ms, min_dist_m)
# info: dict(n_episodes, n_seed_carry, n_seed_gathered, n_gathered_accepted,
#            coverage, slot_rejected, neither_plausible)

# ------------------------------------------------------------------ helpers


def _runs(mask):
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _roll_nanmedian(x, w):
    k = w // 2
    pad = np.pad(np.asarray(x, float), k, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, w)
    with np.errstate(all="ignore"):
        return np.nanmedian(win, axis=1)


def _whistle_mask(n, events, window):
    m = np.zeros(n, bool)
    if events is None or len(events) == 0 or window is None:
        return m
    lo, hi = int(round(window[0] * FPS)), int(round(window[1] * FPS))
    for ev in events:
        a = max(0, int(ev["start_frame"]) - 1 + lo)
        b = min(n, int(ev["start_frame"]) - 1 + hi)
        if b > a:
            m[a:b] = True
    return m


def keeper_tracks(match, st, plaus_m=PLAUS_M):
    """Per-frame keeper xy for both teams, NaN where untracked or implausible.

    ``own_x`` (N,2) is the x of the goal each team defends, from the attack direction.
    Rejecting the slot per frame is the E30 guard; it never invents a keeper.
    """
    P, V, half, gk = match.P, match.V, match.half, match.gk
    n = len(match.B)
    own_x = np.full((n, 2), np.nan, np.float32)
    pos = np.full((n, 2, 2), np.nan, np.float32)
    spd = np.full((n, 2), np.nan, np.float32)
    plaus = np.zeros((n, 2), bool)
    for h in (1, 2):
        sel = half == h
        if not sel.any():
            continue
        for t in (0, 1):
            s = int(gk[h - 1, t])
            own = 0.0 if st.attack_dir[h - 1, t] > 0 else PITCH_L
            own_x[sel, t] = own
            xy = P[sel, s]
            ok = np.isfinite(xy[:, 0]) & (np.abs(xy[:, 0] - own) <= plaus_m)
            plaus[sel, t] = ok
            pos[sel, t] = np.where(ok[:, None], xy, np.nan)
            spd[sel, t] = np.where(ok, V[sel, s], np.nan)
    return pos, spd, own_x, plaus


def _ball_speed(B, half):
    d = np.full(len(B), np.nan)
    d[1:] = np.linalg.norm(np.diff(B, axis=0), axis=1) * FPS
    d[half <= 0] = np.nan
    return _roll_nanmedian(d, BALL_MED)


def _oob(B, margin=0.5):
    ok = ~np.isnan(B[:, 0])
    out = np.zeros(len(B), bool)
    out[ok] = ((B[ok, 0] < -margin) | (B[ok, 0] > PITCH_L + margin) |
               (B[ok, 1] < -margin) | (B[ok, 1] > PITCH_W + margin))
    return out


def _arrival(B, s, look):
    """Ball motion in the ``look`` frames before ``s``: displacement, max step, full presence."""
    a = max(0, s - look)
    seg = B[a:s + 1]
    if len(seg) < 2 or not np.isfinite(seg).all():
        return np.nan, np.nan
    step = np.linalg.norm(np.diff(seg, axis=0), axis=1) * FPS
    return float(np.linalg.norm(seg[-1] - seg[0])), float(step.max()) if step.size else 0.0


def _release(B, e, n, look):
    a, z = e, min(e + look, n)
    if z - a < 2:
        return np.nan
    seg = B[a:z]
    if not np.isfinite(seg).all():
        return np.nan
    return float(np.max(np.linalg.norm(np.diff(seg, axis=0), axis=1) * FPS))


def _extend(B, d, gkv, pos, kt, oob, wh, s, e, p, n):
    """Grow a slow-ball seed over the surrounding carry (same keeper, ball near, keeper walking).

    The seed needs a slow ball and a walking keeper; inside a carry both tick up between touches,
    so extension uses the looser ``v_gk_ext`` (a keeper jogging with the ball still has it).
    Extension never crosses a missing ball, an out-of-play ball, a whistle, or a keeper-slot jump
    (the slot can be reused mid-half, E30), and never exceeds ``max_s``.
    """
    lim = int(p["max_s"] * FPS)
    a, z = s, e
    while True:
        moved = False
        if a > 0 and (z - a + 1) < lim:
            f = a - 1
            if (np.isfinite(d[f, kt]) and d[f, kt] <= p["r"] and gkv[f] <= p["v_gk_ext"]
                    and not oob[f] and not wh[f]
                    and np.linalg.norm(pos[f, kt] - pos[f + 1, kt]) <= 2.5):
                a, moved = f, True
        if z < n and (z + 1 - a) < lim:
            f = z
            if (np.isfinite(d[f, kt]) and d[f, kt] <= p["r"] and gkv[f] <= p["v_gk_ext"]
                    and not oob[f] and not wh[f]
                    and np.linalg.norm(pos[f, kt] - pos[f - 1, kt]) <= 2.5):
                z, moved = f + 1, True
        if not moved:
            break
    return a, z


# ------------------------------------------------------------------ detector


def detect(match, st=None, events=None, params=None):
    """Detect goalkeeper possession. ``events`` is E15's whistle event array; None tries the audio.

    For a non-ATD game pass ``st`` built from that game's own halves: :func:`possession.structure.
    structure` carries BAR-ATH's published half blocks and kickoffs, which are not valid elsewhere.

    Never raises on missing audio: the whistle gate simply contributes nothing (``ok=False`` in
    :mod:`possession.audio` is handled by an empty array). Because an empty event array means both
    "audio was absent" and "no whistle was detected", ``info["events_source"]`` always records which
    one it was -- ``"audio"``, the audio track's own failure token (``"no_ffmpeg"``, ``"no_video"``,
    ``"audio_error"``, ...), ``"import_error"``, or ``"given"`` for a caller-supplied array -- and
    ``info["n_events"]`` records how many events arrived (Q34). A reported episode count is only
    comparable across runs when this source is the same.
    """
    p = dict(PARAMS, **(params or {}))
    if st is None:
        st = structure(match)
    if events is None:
        try:
            from . import audio
            track = audio.whistle_events()
            events = track.events
            events_source = "audio" if track.ok else str(track.reason or "audio_error")
        except ImportError:
            events = np.zeros(0)
            events_source = "import_error"
        except Exception:
            events = np.zeros(0)
            events_source = "audio_error"
    else:
        events_source = "given"
    n = len(match.B)
    half, graded, B = match.half, match.graded, match.B
    graded = graded & ~st.forced_dead_mask
    present = ~np.isnan(B[:, 0])
    pos, spd, own_x, plaus = keeper_tracks(match, st, p["plaus_m"])
    d = np.linalg.norm(B[:, None, :] - pos, axis=2)  # (N,2), NaN when missing/rejected
    dmin = np.full(n, np.nan)
    ok = np.isfinite(d).any(1)
    dmin[ok] = np.nanmin(d[ok], axis=1)

    okv = np.isfinite(spd)
    gkv = np.full(n, np.nan)
    anyv = okv.any(1)
    gkv[anyv] = np.min(np.where(okv[anyv], spd[anyv], np.inf), axis=1)
    gkv = _roll_nanmedian(gkv, GK_MED)
    gkv[gkv > 50] = np.nan
    vball = _ball_speed(B, half)
    oob = _oob(B)
    wh = _whistle_mask(n, events, p["whistle_win"])

    core = (graded & present & (dmin <= p["r"]) & (gkv <= p["v_gk"])
            & (vball <= p["v_ball"]) & ~oob)
    pre = int(p["pre_present_s"] * FPS)
    look = int(1.5 * FPS)
    rel_look = int(p["release_s"] * FPS)
    carry = np.zeros(n, bool)
    seeds = []
    for s0, e0 in _runs(core):
        dd = d[(s0 + e0) // 2]
        kt = int(np.nanargmin(np.where(np.isfinite(dd), dd, np.inf)))
        s, e = (_extend(B, d, gkv, pos, kt, oob, wh, s0, e0, p, n)
                if p["extend"] else (s0, e0))
        sec = (e - s) / FPS
        disp, step = _arrival(B, s, look)
        rel = _release(B, e, n, rel_look)
        ok_len = p["min_s"] <= sec <= p["max_s"]
        ok_wh = not wh[max(0, s - int(1.0 * FPS)):s + int(0.5 * FPS)].any()
        ok_oob = not oob[max(0, s - int(p["oob_look_s"] * FPS)):s].any()
        ok_pre = present[max(0, s - pre):s].all()
        ok_motion = (not p["use_ball_motion"]) or (
            np.isfinite(disp) and disp >= p["min_move_m"]
            and not (np.isfinite(step) and step > p["max_step_ms"]))
        if p["require_release"]:
            post = B[e:min(e + rel_look, n)]
            lost = len(post) > 0 and not np.isfinite(post[:, 0]).all()
            ok_rel = lost or (np.isfinite(rel) and rel >= p["release_v"])
        else:
            ok_rel = True
        seeds.append(dict(start=s, end=e, seconds=sec, team=int(kt), seed=(int(s0), int(e0)),
                          keep=bool(ok_len and ok_wh and ok_oob and ok_pre and ok_motion
                                    and ok_rel),
                          whistle=bool(ok_wh), oob=bool(ok_oob), pre=bool(ok_pre),
                          motion=bool(ok_motion), release=bool(ok_rel)))
        if seeds[-1]["keep"]:
            carry[s:e] = True

    # ---- gathered: the keeper takes the ball and the tracker loses it
    # Off by default (see PARAMS): it is the one branch that fires while the ball is *missing*,
    # which is E07's primary D cue, so a false positive there is expensive and a miss is cheap.
    gathered = np.zeros(n, bool)
    geps = []
    i = 0
    while i < n:
        if graded[i] and not present[i]:
            j = i
            while j + 1 < n and graded[j + 1] and not present[j + 1]:
                j += 1
            k = i - 1
            while k >= 0 and not present[k]:
                k -= 1
            if k >= 0 and graded[k]:
                gap_s = (j - i + 1) / FPS
                ret = np.nan
                # Far end of the <= 1 s return window, clamped to the last frame: when the array
                # ends 3-25 frames after the run, min(..., n) used to index present[n]. Measuring
                # over the available tail preserves the "reappears moving fast" test.
                a2, z2 = j + 1, min(j + 1 + int(FPS), n - 1)
                if a2 < n and z2 > a2 + 2 and present[a2] and present[z2]:
                    ret = float(np.linalg.norm(B[z2] - B[a2]) / ((z2 - a2) / FPS))
                ok = (np.isfinite(dmin[k]) and dmin[k] <= p["miss_r"]
                      and gap_s <= p["miss_max_s"]
                      and not oob[max(0, k - int(2 * FPS)):k + 1].any()
                      and not wh[max(0, k - int(2 * FPS)):k + int(2 * FPS)].any()
                      and np.isfinite(vball[k]) and vball[k] <= p["v_ball"]
                      and present[max(0, k - pre):k + 1].all()
                      and (not np.isfinite(ret) or ret >= p["release_v"]))
                geps.append(dict(start=i, end=j, seconds=gap_s, accepted=False,
                                 team=_nearest_team(d[k]), keep=bool(ok),
                                 last_dist=float(dmin[k]) if np.isfinite(dmin[k]) else np.nan,
                                 return_ms=ret))
                if ok:
                    geps[-1]["accepted"] = True
                    if p["include_gathered"]:
                        gathered[i:j + 1] = True
            i = j + 1
        else:
            i += 1

    live_b = carry | gathered
    # weight: a tracked ball at the keeper is stronger evidence than a short missing gap
    weight = np.where(carry, 1.0, np.where(gathered, GATHERED_WEIGHT, 0.0)).astype(np.float32)
    # bridge sub-frame holes and re-derive merged episodes
    for s, e in _runs(~live_b):
        if s > 0 and e < n and live_b[s - 1] and live_b[e] \
                and (e - s) / FPS <= p["merge_gap_s"]:
            live_b[s:e] = True
    live = live_b.astype(np.float32)
    live[~graded] = 0.0
    episodes = []
    for s, e in _runs(live_b):
        kind = "carry" if carry[s:e].sum() >= gathered[s:e].sum() else "gathered"
        seg = np.where(np.isfinite(d[s:e]), d[s:e], np.inf)
        kt = int(np.nanargmin(seg.min(0))) if s < e else 0
        episodes.append(dict(
            start=int(s), end=int(e), seconds=(e - s) / FPS, team=kt, kind=kind,
            release_ms=_release(B, e, n, rel_look),
            min_dist_m=float(np.nanmin(d[s:e, kt])) if np.isfinite(d[s:e, kt]).any() else np.nan))
    info = dict(n_episodes=len(episodes),
                events_source=events_source,
                n_events=int(len(events)),
                n_seed_carry=int(sum(x["keep"] for x in seeds)),
                n_seed_gathered=int(sum(x["keep"] for x in geps)),
                n_gathered_accepted=int(sum(x["accepted"] for x in geps)),
                coverage=float(live.sum() / max(1, graded.sum())),
                slot_rejected=float(1.0 - plaus[graded].mean()) if graded.any() else 0.0,
                neither_plausible=float((~plaus[graded].any(1)).mean()) if graded.any() else 1.0)
    return GkPossession(live, (-LIVE_GAIN * weight).astype(np.float32), episodes, info)


def _nearest_team(row):
    if not np.isfinite(row).any():
        return -1
    return int(np.nanargmin(row))
