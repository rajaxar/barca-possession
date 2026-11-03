"""Label-free dead-ball (D) proxies for BAR-ATH, and the adoption protocol for match-only cues.

This module exists because our strongest inputs (referee audio, broadcast cutaways, camera pan,
Smart Tagging) exist only on the match we are predicting. They can never be selected on an external
dev set, so the temptation is to adopt them by looking at the public 5-minute window -- which
GUARDRAILS 2 forbids. The answer is not "drop them": it is to score them against one shared,
label-free target, with a decision rule written down before the test is run.

TWO PROXIES, BECAUSE ONE BALL STATE CANNOT COVER D
-------------------------------------------------
``proxy()`` needs the ball **missing**; ``present_ball()`` (E42) needs it **tracked and static**.
On this match the ball is tracked on ~66% of true ``D`` frames (Q08), so the missing-ball proxy is
blind to most dead time and the present-ball proxy sees a smaller, different slice. They are
disjoint by construction (a frame cannot be both missing and present) and are meant to be used
together: ``proxy=mode`` in :func:`adoption_report` and ``--proxy {missing,present,both}`` on the
CLI force a caller to say which target a cue was scored against.

THE MISSING-BALL PROXY
----------------------
A frame is proxy-``D`` iff

    the frame is inside a ball-missing run of at least ``MIN_RUN_S`` (2 s)
    AND mean tracked-player speed on the frame is below ``STILL_V`` (1.5 m/s)
    AND the frame is not goalkeeper-in-possession (E38 exclusion, on by default)

with ungraded frames and the published forced-D 2H pre-kickoff block never in the mask
(``structure.forced_dead_mask``; both are D by structure, not by evidence).

The exclusion is imported, never reimplemented: ``info["gk_source"]`` must read
``"gk_possession"``. If that module cannot be **imported**, the missing-ball proxy falls back to a
narrow inline gathered-branch copy under ``gk_source="inline"`` and the present-ball proxy applies
no veto under ``gk_source="unavailable"`` -- in both cases it prints a ``WARNING [dproxy]`` line and
stores the same text in ``info["gk_note"]``. A detector *bug* propagates instead of being
substituted: swallowing every exception is how B10's array-end IndexError hid from these
diagnostics. A consumer that uses the exclusion should assert the source.

Both thresholds are inherited, not fitted:

* **2 s duration floor** -- E25's external-trained classifier gives P(D | missing run) of
  .09 / .39 / .58 / .85 for runs of <=3 / 3-6 / 6-12 / >12 s, and Q08 found the two runs <=1 s in
  the public window carried no signal (0 D frames). 2 s is where P(D) first exceeds a coin flip.
* **1.5 m/s stillness** -- the walk/jog boundary used by I01, E25, E15 and E38; it is the 53rd
  percentile of a half's mean player speed on this match (E15 sec. 1), i.e. a football constant, and
  it reproduces E15's "dead-ish" state exactly, so an E15 cue can be scored against this module
  without a re-definition.
* **12 m/s speed clip** -- a physical guard, not a tuning knob: E16 measured player-speed spikes far
  above a sprint, and one outlier in a 20-player mean would decide the frame. Elite sprint is
  ~10-12 m/s; nothing legitimate is lost.

KNOWN FAILURE MODES (with numbers, so nobody mistakes the proxy for a measurement)
----------------------------------------------------------------------------------
1. **It is blind to the larger half of true D.** On the public window 66% of true ``D`` frames have
   a tracked ball (Q08), and this match keeps the ball visible through most stoppages (ball on the
   spot, rolling, held). Those frames are *never* in the proxy. So the proxy's recall on true D is
   capped well below 1 and a cue whose value is "dead ball with the ball visible" (the referee
   whistle is exactly that: only ~6% of ball-out episodes have a whistle, R06) cannot be certified
   by a frame-agreement ablation against it. That is why the protocol's decision uses the *lead*
   contrast (an onset cue can precede the proxy) and reports the ablation as a secondary number.
2. **Up to ~7% of missing time may be dead but motion-live.** E36: 10.4% of missing time is
   motion-live, and at most 7.2% (4,620 frames) of missing time could be dead while the gate calls
   it live. Concrete case: the 5.9 s public-window gap that is 86.5% ``D`` has classifier P(D)=0.37.
   This is the recall cost, and it is larger than the precision cost below.
3. **0.88% [0, 2.11] of missing time is live but looks dead**, hard upper bound 3.34% (E36, block
   bootstrap over runs). So a player-motion gate on missing time is ~99% precise (96.7% worst case).
   The error is concentrated in *short, slow live passages* (walking dribbles, 5-11 s), i.e. exactly
   the episodes spellF1 punishes twice: one false D breaks the host possession and adds a spurious
   one.
4. **Classifier-vs-proxy disagreement** (measured on the real match by ``scratch/S22/measure.py``;
   the module cannot recompute it without E25's external-trained array; numbers in
   ``MEASURED_AGREEMENT``). 96.8% of proxy frames are called D by E25's classifier (mean P(D) 0.910).
   On missing time the two disagree on **10.9% of frames**: 8.3% is classifier-D but proxy-live
   (the duration floor and stillness gate: the recall cost in (2)), 2.6% is proxy-D but
   classifier-live (the precision cost in (3), measured against a classifier that is itself a
   proxy). Against E36's independent camera, the proxy calls D on 0.70% of missing time that is
   broadcast-verified live (446 frames), inside E36's 0.88% [0, 2.11] band for a motion gate.
5. **11.4% of missing time is unresolved** even with E36's camera plus mechanism rules, so no
   figure here is a label.
6. **The proxy is a *relative* comparator.** It ranks cues; it is not a training target, not a
   score, and cannot validate a threshold on its own. The one absolute check it does carry is the
   published forced-D block, which it must leave alone (see tests).

THE PRESENT-BALL PROXY (``present_ball``, E42)
----------------------------------------------
A frame is present-ball proxy-``D`` iff

    the ball is tracked on the frame
    AND its speed (0.5 s baseline displacement, centered) is <= ``BALL_STILL_V`` (1.0 m/s)
        for a run of at least ``MIN_HOLD_S`` (1.0 s)
    AND mean tracked-player speed on the frame is below ``STILL_V`` (1.5 m/s)
    AND the frame is not goalkeeper-in-possession (E38 carry branch, on by default)

again excluding ungraded frames and the published forced-D block. Every run is classified

* ``placed`` -- no player within ``NEAR_PLAYER_M`` (1.5 m) for most of the run: a ball on the spot,
  a line or a restart mark;
* ``held``   -- a player is within ``NEAR_PLAYER_M`` for most of the run: a ball at someone's feet,
  which is the shape of a player holding the ball during a stoppage *and* of a keeper about to
  distribute. The keeper case is what the E38 exclusion removes; the outfield case cannot be
  separated from live play by tracking alone (see the measured precision by class in
  ``MEASURED_PRESENT``), so it is reported as its own class rather than trusted.

Why each term:

* **1.0 m/s near-rest** -- a ball at rest reads 0.1-0.5 m/s through the ATD's 0.3 s smoothing
  (E18: sigma ~7-9 frames, jitter 1-3 mm), while a passed or dribbled ball is 2-4 m/s even at
  walking pace (S06's coupling curve: ball speed median 1.78 m/s at the slowest player decile,
  the first decile being mostly stopped balls). 1.0 m/s is the top of the "stopped ball" mode and
  below the slowest rolling mode; it is a physics constant, and the 0.5 s baseline makes it
  insensitive to jitter.
* **1.0 s floor** -- the card's floor, and the same "longer than a touch" logic as E25's 2 s
  missing-run floor: a ball that stops for a second while players are still is not a bounce or a
  touch. Sensitivity to 0.5/1.0/2.0 s is in the research note; 1.0 s is the pre-registered value.
* **1.5 m/s collective stillness** -- the same inherited walk/jog boundary as the missing-ball
  proxy (E15/E25/E38), so a cue can be scored against either target without redefinition.
* **E38 exclusion** -- a keeper carrying or holding the ball is live play and is exactly the shape
  that satisfies "ball static/near-static near a player"; it must never become present-ball ``D``.
  The exclusion is imported, not reimplemented; ``info`` records how many candidate frames it vetoed.
* **restart context** -- per run, not a gate: distance to the nearest painted line, to the center
  spot and to either penalty mark, whether the ball is inside a goal area, and the own-half share
  of tracked outfield players (S09's kickoff-formation statistic). ``placed`` runs carry the share
  that are within 2 m of a line, within 9.15 m of the center spot, in a goal area or on a penalty
  mark. Context is reported because it is what distinguishes a goal-kick placement from a ball that
  merely came to rest in open play; it is not used to widen the mask, because a false D costs a
  possession twice and context alone is cheap to satisfy.

Two dead-with-visible-ball shapes are deliberately **not** in the mask:

* **rolling/carried** (ball moving at 1-3 m/s with a player within reach, collective speed low).
  This class is computed and reported as ``slow_runs`` but excluded: E36's three broadcast-verified
  LIVE gaps are walking dribbles at 1.2 m/s mean player speed, and a ball carried off and kicked
  back in (the human's #14) has the same tracking signature. Tracking alone cannot separate them,
  so asserting D here would be the one-eyed instrument again. ``include_rolling=True`` is opt-in.
* **held by a player while static** is *in* the mask but flagged ``held``: it is dead-looking and
  static (a stationary ball at a stationary player's feet), yet the same geometry occurs when a
  keeper holds it (excluded by E38) and, briefly, when a player stands over a live ball. The
  honest statement is that static-held is callable and carried-held is not.

MEASURED_PRESENT (real match, scratch/E42/measure.py) records the proxy's coverage, its
complementarity with the missing-ball proxy, and the independent checks; it is not a label.

THE ADOPTION PROTOCOL
---------------------
``adoption_report(cue)`` takes a per-frame cue score and returns a pre-registered decision. The
statistics are fixed here, before any cue is seen:

* cue *events* are maximal runs of ``cue >= active_thr`` (a binary cue may omit the threshold; a
  continuous cue must declare one, and the report records it);
* the **after window** is ``[onset + 0.5 s, onset + 3.0 s]`` and the **before window** is
  ``[onset - 3.0 s, onset - 0.5 s]``. E15 measured the whistle's D-state contrast at +0.5..+3 s
  (0.509 vs 0.351 random) and showed the cue *leads* tracking by ~2 s; an onset cue must score
  better after itself than before itself, and a cue that merely reacts to a dead ball scores the
  same on both sides;
* **``lead_gain`` = proxy rate in the after window - proxy rate in the before window**, and the
  **decision uses its run-block CI**: adopt as a tier-2 lever iff the 95% CI is entirely above
  zero, otherwise park. A significantly negative lead_gain is also "park", with the reason
  ``cue_is_reactive`` -- it is not a weaker adopt, it is evidence against.
* Two further pre-registered conditions, because a CI alone is not a lever:
  **``lead_gain`` must be at least ``MIN_LEAD_GAIN`` (0.05)**, the full-match spellF1 noise band
  (GUARDRAILS 3: ~0.04), so a significant 1-pp shift cannot be adopted; and the same contrast
  **inside the proxy's domain** (ball-missing frames), ``conditional_lead_gain``, must also be at
  least ``MIN_LEAD_GAIN`` -- otherwise the cue is re-detecting the ball dropout the proxy is built
  from and carries nothing new (reason ``redundant_with_missingness``). E15 rejected the crowd
  envelope on exactly that conditional number (AUC .625 raw, .490 conditional on missingness).
* ``inside`` / ``outside`` (raw proxy rate in the after window vs everywhere else) and the
  **ablation delta** (a frozen ball-state rule with the cue's after windows OR-ed in, scored
  against the proxy over all graded frames; masking the cue removes them) are reported with their
  own CIs. The frozen rule is ``ball missing`` for the missing-ball target and ``ball present and
  static >= 1 s`` (the proxy's own ball half, without its player and keeper gates) for the
  present-ball target, so in both cases it is "the ball state only" -- the thing a cue must add
  to. The ablation is *not* the decision gate: because the missing-ball proxy is blind to
  present-ball D, a real onset cue loses it by construction (the whistle loses 0.009), while a
  state cue that fires for a fifth of the match loses much more (the crowd envelope loses 0.178).

The **proxy kind** is read from ``proxy_obj.info["kind"]`` (``"missing"``, ``"present"`` or
``"both"``) and changes only the domain of the refusal/conditional check and the frozen ablation
rule:

* missing: domain = ball-missing frames (as before E42);
* present: domain = ball-present frames;
* both: domain = graded frames (the union target is defined almost everywhere, so the conditional
  contrast is the raw one; that is stated in the report, not hidden).

Refusals (all before any statistic is computed, so they cannot be negotiated after the fact):
  * ``len(cue) != len(proxy mask)`` -> ValueError (a silent misalignment is the worst failure here);
  * more than ``MAX_NAN_SHARE`` (5%) of graded frames NaN -> ValueError (NaN is not "no evidence";
    a cue that cannot score 5% of the match is not ready). At or below 5%, NaN is scored as 0;
  * infinities -> ValueError;
  * fewer than ``MIN_DOMAIN_FRAMES`` (100, i.e. 4 s) target-domain frames in either window ->
    ValueError: the missing-ball proxy is blind to present-ball D (failure mode 1) and the
    present-ball proxy only sees still-present frames, so a cue that fires outside a proxy's
    domain cannot be scored against it, and a spurious zero would be worse than a refusal.

Blocks for the bootstrap are maximal runs of constant proxy state inside a half, so a draw is a
contiguous stretch of match, not a frame. Seeds are fixed (``SEED``), so a report is reproducible.
"""
import sys
import warnings
from collections import namedtuple

import numpy as np

from .structure import PITCH_L, PITCH_W, structure

FPS = 25.0

# ---------------------------------------------------------------- inherited constants (see docstring)
MIN_RUN_S = 2.0      # E25 gap-length gradient .09/.39/.58/.85; Q08 <=1 s runs carry no D
STILL_V = 1.5        # walk/jog boundary: I01, E25, E15, E38
SPEED_CLIP = 12.0    # physical sprint guard (E16 speed spikes), not a fitted cut

# ---------------------------------------------------------------- present-ball constants (E42)
BALL_STILL_V = 1.0   # m/s: top of the "stopped ball" mode (S06 coupling; E18 jitter), below a roll
MIN_HOLD_S = 1.0     # card's floor: a ball that stops for <1 s is a bounce or a touch, not a stoppage
BALL_MED_F = 5       # median half-width for the per-frame ball speed (0.2 s)
BALL_BASE_S = 0.5    # baseline displacement used as the robust ball speed (E18's smoothing scale)
NEAR_PLAYER_M = 1.5  # ball at someone's feet: E02/I01 proximity radius (E38 use 2 m for a keeper)
ROLL_V = 3.0         # slowest rolling/carried mode ends here (E38's V_BALL, a brisk walk with the ball)
LINE_M = 2.0         # a restart ball placed at a line: taker and ball are within a meter or two
CENTER_M = 9.15      # center circle radius (Laws); a kickoff ball is on/near the spot
SPOT_M = 1.5         # penalty mark is a painted spot; the ball sits on it
GOAL_AREA_M = 5.5    # 6-yard box depth (Laws); goal kicks are placed inside it

# E25's classifier probe P(D | missing-run length), used only as a *strength* for reporting.
# Bins are E25's (orchestrator probe, E07 card): <=3 / 3-6 / 6-12 / >12 s.
GRADE_BINS_S = ((0.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, float("inf")))
GRADE_P = (0.09, 0.39, 0.58, 0.85)

# Measured agreement with the two independent instruments (scratch/S22/measure.py, recorded so a
# consumer can see the proxy's error bars without re-running the classifier).
MEASURED_AGREEMENT = dict(
    # E25 external-trained HGB (scratch/E25/pd_barath_hgb_phys.npy), whole match, graded excl.
    # the published 2H block. Share of *proxy* frames the classifier calls D, and mean P(D).
    classifier_agreement_on_proxy=0.968,
    classifier_mean_pd_on_proxy=0.910,
    # Share of MISSING time where the two disagree (the classifier is itself a proxy):
    classifier_d_proxy_live_share=0.0831,   # recall cost of the duration floor + stillness gate
    proxy_d_classifier_live_share=0.0257,   # precision cost
    # E36's independent broadcast-tracker verdicts (15 LIVE runs = 2,618 frames; 1 DEAD = 260):
    e36_live_frames_called_d=0.170,          # = 0.70% of missing time (E36's yardstick: 0.88%)
    e36_dead_frames_called_d=0.908,
    measured_on="BAR-ATH full match, graded excl. forced 2H block, 2026-10-28 (scratch/S22)",
)

D_PROXY_PARAMS = dict(min_run_s=MIN_RUN_S, still_v=STILL_V, speed_clip=SPEED_CLIP,
                      run_gate="frame",          # "frame" = E15's dead-ish; "run" = whole-gap gate
                      exclude_gk=True,           # E38 keepers are live play, not D
                      gk_include_gathered=True)  # only branch that can overlap a *missing* ball

D_PRESENT_PARAMS = dict(min_hold_s=MIN_HOLD_S, ball_still_v=BALL_STILL_V, still_v=STILL_V,
                        speed_clip=SPEED_CLIP, near_player_m=NEAR_PLAYER_M, roll_v=ROLL_V,
                        base_s=BALL_BASE_S, exclude_gk=True,
                        include_rolling=False)   # opt-in only; E36's live dribbles live there

DProxy = namedtuple("DProxy", "mask runs missing_runs speed info")
# mask: (N,) bool, the per-frame proxy (False on ungraded, forced-D and GK frames)
# runs: list of dict(start, end, frames, seconds, mean_speed, still_share, missing_run, grade)
#       -- end exclusive; one entry per maximal proxy run
# missing_runs: list of dict(start, end, frames, seconds, mean_speed, still_share, qualifies,
#       proxy_frames, proxy_seconds, grade, gk_excluded_frames) -- every missing run, proxy or not
# speed: (N,) float32 mean tracked-player speed (NaN when no player is tracked)
# info: dict with the thresholds used, coverage, the GK-exclusion accounting, and the exact
#       `graded` scope the mask was built on (the protocol reuses it, so they cannot drift apart)

PresentD = namedtuple("PresentD", "mask runs slow_runs vball info")
# mask: (N,) bool, present-ball proxy D (False on ungraded, forced-D and GK frames)
# runs: list of dict(start, end, frames, seconds, mean_speed, ball_speed_med, nearest_player_m,
#       near_player_share, kind "placed"|"held", context dict, placed bool, gk=False)
# slow_runs: same shape for the rolling/carried class (ball 1-3 m/s, players still) -- reported,
#       not in `mask` unless include_rolling=True
# vball: (N,) float32 robust ball speed in m/s (NaN where the ball is missing)
# info: dict(kind="present", thresholds, coverage of graded/present, gk veto accounting, graded)


# ---------------------------------------------------------------- protocol constants (pre-registered)
AFTER_WINDOW_S = (0.5, 3.0)   # E15 sec. 3: the D-state contrast is measured at +0.5..+3 s
BEFORE_WINDOW_S = (-3.0, -0.5)
MAX_NAN_SHARE = 0.05
MIN_DOMAIN_FRAMES = 100      # 4 s of ball-missing frames in *each* window, else refuse
MIN_LEAD_GAIN = 0.05         # >= the full-match spellF1 noise band (~0.04, GUARDRAILS 3)
MIN_CONDITIONAL_LEAD = 0.05  # must still hold inside ball-missing frames, else it is redundant
MAX_ACTIVE_SHARE_WARN = 0.10  # a cue active on >10% of graded frames is a state, not an onset
N_BOOT = 2000
SEED = 0
ALPHA = 0.05


# ================================================================ helpers


def _runs(mask):
    """Maximal True runs as an (K, 2) int array, end exclusive."""
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def mean_player_speed(match, clip=SPEED_CLIP):
    """(N,) mean speed of the tracked players in m/s; NaN when nobody is tracked.

    ``match.V`` is m/s (B03). Untracked slots are NaN and are left out of the mean; speeds are
    clipped to ``[0, clip]`` so one tracking spike cannot decide a frame (physical guard).
    """
    v = np.asarray(match.V, np.float64)
    v = np.where(np.isfinite(v), np.clip(v, 0.0, clip), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are legitimate (no players)
        s = np.nanmean(v, axis=1)
    return s.astype(np.float32)


def _grade(seconds):
    """Duration-graded strength from E25's P(D | run length) probe (prior, not a measurement)."""
    for (lo, hi), p in zip(GRADE_BINS_S, GRADE_P):
        if lo <= seconds < hi:
            return float(p)
    return float(GRADE_P[-1])


def _graded_frames(match, st):
    """Frames the proxy and the protocol are defined on (graded, excl. published forced D)."""
    return np.asarray(match.graded) & ~np.asarray(st.forced_dead_mask)


# ================================================================ goalkeeper exclusion


def _inline_gk(match, st, params):
    """Minimal E38 *gathered-branch* fallback, used only if possession.gk_possession is missing.

    TODO(S22): this duplicates gk_possession.detect's gathered branch and is deliberately narrow --
    a short (<=6 s) ball-missing gap that starts within 2 m of a plausible keeper slot after
    continuously tracked play and ends with a fast return. It is deliberately narrower than E38:
    on BAR-ATH it found 0 of the real detector's 1,252 exclusion frames (B21), so it is a
    degraded-exclusion fallback, not a replacement, and it always reports ``gk_source="inline"``.
    The carry branch cannot overlap the proxy at all (the ball is present by construction), so it
    is not reimplemented here.
    """
    n = len(match.B)
    half, P, B = match.half, match.P, match.B
    present = np.isfinite(B[:, 0])
    out = np.zeros(n, bool)
    for h in (1, 2):
        for t in (0, 1):
            s = int(match.gk[h - 1, t])
            own = 0.0 if st.attack_dir[h - 1, t] > 0 else PITCH_L
            xy = P[:, s]
            ok = np.isfinite(xy[:, 0]) & (np.abs(xy[:, 0] - own) <= 45.0)
            for a, b in _runs((half == h) & ~present):
                dur = (b - a) / FPS
                k = a - 1
                if not (0.6 <= dur <= 6.0) or k < 0 or not ok[k] or not present[k]:
                    continue
                if np.linalg.norm(B[k] - xy[k]) > 2.0 or not present[max(0, k - 75):k + 1].all():
                    continue
                z = b
                if z + 2 >= n or not present[z] or not present[z + 2]:
                    continue
                if np.linalg.norm(B[z + 2] - B[z]) / 2.0 >= 4.0:
                    out[a:b] = True
    out &= _graded_frames(match, st)
    return out


def _gk_fallback_note(target, exc, consequence):
    """Loud, recorded notice that the keeper exclusion is not the real detector.

    Silent substitution is how B10's array-end bug hid: the proxy caught every exception and kept
    returning an exclusion mask from a copy. A missing module is the only legitimate fallback, so
    the fallback prints a status line and stores the same text in ``info["gk_note"]``.
    """
    note = ("%s keeper exclusion is NOT possession.gk_possession (%s: %s); %s"
            % (target, type(exc).__name__, exc, consequence))
    print("WARNING [dproxy] " + note, file=sys.stderr, flush=True)
    return note


def _import_gk():
    """Import E38's detector. A separate seam so the missing-module path is testable."""
    from . import gk_possession
    return gk_possession


def _gk_live(match, st, params):
    """(N,) bool of goalkeeper-in-possession frames, the source label and a fallback note.

    E38's detector is the real thing; the inline fallback is only for a missing module and says so.
    Any other exception from the detector (a bug, not an absent module) propagates: a silent inline
    mask is exactly what hid B10's IndexError from the dev diagnostics.
    """
    if not params["exclude_gk"]:
        return np.zeros(len(match.B), bool), "disabled", None
    try:
        gk = _import_gk()
    except ImportError as exc:
        note = _gk_fallback_note(
            "missing-ball", exc,
            "falling back to the inline gathered-branch copy (gk_source='inline')")
        return _inline_gk(match, st, params), "inline", note
    gp = gk.detect(match, st, events=np.zeros(0),
                   params={"include_gathered": bool(params["gk_include_gathered"])})
    return np.asarray(gp.live) > 0, "gk_possession", None


# ================================================================ the proxy


def proxy(match, st=None, params=None):
    """Compute the label-free D proxy for a ``possession.data.Match``.

    Usage::

        from possession import dproxy, data
        dp = dproxy.proxy(data.load_match())
        dp.mask.sum(), len(dp.runs)          # frames, runs
        dp.runs[0]["seconds"], dp.runs[0]["grade"]
        dp.info["coverage_missing"]          # share of missing time the proxy calls D

    ``params`` overrides ``D_PROXY_PARAMS``. The function is deterministic and reads no label.
    """
    p = dict(D_PROXY_PARAMS, **(params or {}))
    if p["run_gate"] not in ("frame", "run"):
        raise ValueError("run_gate must be 'frame' or 'run'")
    if st is None:
        st = structure(match)
    n = len(match.B)
    half = np.asarray(match.half)
    graded = _graded_frames(match, st)

    missing = graded & ~np.isfinite(match.B[:, 0])
    speed = mean_player_speed(match, p["speed_clip"])
    still = np.isfinite(speed) & (speed < p["still_v"])

    gk_live, gk_source, gk_note = _gk_live(match, st, p)

    # Missing runs: contiguous missing frames inside one half. A half boundary, the forced-D block
    # or any ungraded frame ends a run, so nothing is ever carried across structure. `missing` is
    # already restricted to graded frames, so a run cannot cross the forced block.
    missing_runs = []
    cand = np.zeros(n, bool)
    for a, b in _runs(missing):
        sec = (b - a) / FPS
        qualifies = sec >= p["min_run_s"] - 1e-9
        run_speed = float(np.nanmean(speed[a:b])) if np.isfinite(speed[a:b]).any() else float("nan")
        if not qualifies:
            gate = False
        elif p["run_gate"] == "run":
            gate = bool(np.isfinite(run_speed) and run_speed < p["still_v"])
        else:
            gate = True
        if gate:
            cand[a:b] = True
        missing_runs.append(dict(
            start=int(a), end=int(b), frames=int(b - a), seconds=float(sec),
            mean_speed=run_speed, still_share=float(np.mean(still[a:b])),
            qualifies=bool(qualifies), proxy_frames=0, proxy_seconds=0.0,
            grade=_grade(sec), gk_excluded_frames=int((gk_live[a:b] & still[a:b]).sum())))

    if p["run_gate"] == "frame":
        mask = cand & missing & still
    else:
        mask = cand & missing
    mask &= graded & ~gk_live

    for mr in missing_runs:
        mr["proxy_frames"] = int(mask[mr["start"]:mr["end"]].sum())
        mr["proxy_seconds"] = mr["proxy_frames"] / FPS

    owner = np.full(n, -1, np.int64)
    for k, mr in enumerate(missing_runs):
        owner[mr["start"]:mr["end"]] = k
    runs = []
    for a, b in _runs(mask):
        runs.append(dict(
            start=int(a), end=int(b), frames=int(b - a), seconds=float((b - a) / FPS),
            mean_speed=float(np.nanmean(speed[a:b])),
            still_share=float(np.mean(still[a:b])),
            missing_run=int(owner[a]) if owner[a] >= 0 else None,
            grade=_grade((b - a) / FPS)))

    n_missing = int(missing.sum())
    info = dict(
        kind="missing",   # E42: which ball state this target sees; adoption_report switches on it
        params=p, min_run_s=p["min_run_s"], still_v=p["still_v"], speed_clip=p["speed_clip"],
        run_gate=p["run_gate"], gk_source=gk_source, gk_note=gk_note,
        n_missing_runs=len(missing_runs),
        n_qualifying_runs=int(sum(m["qualifies"] for m in missing_runs)),
        n_missing_frames=n_missing, n_proxy_frames=int(mask.sum()), n_proxy_runs=len(runs),
        coverage_graded=float(mask.sum() / max(1, int(graded.sum()))),
        coverage_missing=float(mask.sum() / max(1, n_missing)),
        missing_share=float(n_missing / max(1, int(graded.sum()))),
        gk_frames=int((gk_live & graded).sum()), gk_in_missing=int((gk_live & missing).sum()),
        gk_vetoed=int(((cand & still if p["run_gate"] == "frame" else cand)
                       & gk_live & graded).sum()),
        median_run_s=float(np.median([r["seconds"] for r in runs])) if runs else 0.0,
        forced_frames_ignored=int(np.asarray(st.forced_dead_mask).sum()),
        graded=np.asarray(graded),   # the exact scope the mask was built on; the protocol reuses it
    )
    return DProxy(mask, runs, missing_runs, speed, info)


# The card calls it "the proxy"; keep both names.
d_proxy = proxy


# ================================================================ the present-ball proxy (E42)


def _robust_ball_speed(B, base_s=BALL_BASE_S):
    """(N,) ball speed in m/s from a centered baseline displacement; NaN when the ball is missing.

    A per-frame difference is dominated by the ATD's 0.3 s smoothing (E18) and by the tracker
    dropping the ball for one frame; a 0.5 s baseline is the scale at which "at rest" and "rolling"
    are actually distinguishable, and it is completely insensitive to sub-frame jitter. Frames whose
    baseline window contains a missing frame are NaN, so a velocity is never bridged across a gap.
    """
    B = np.asarray(B, float)
    n = len(B)
    base = max(1, int(round(float(base_s) * FPS)))
    ok = np.isfinite(B[:, 0])
    c = np.r_[0, np.cumsum(ok)]
    f = np.arange(n)
    lo = np.maximum(0, f - base)
    hi = np.minimum(n, f + base + 1)
    span = hi - 1 - lo
    full = (span >= 2) & (c[hi] - c[lo] == hi - lo)
    v = np.full(n, np.nan)
    idx = np.flatnonzero(full)
    if idx.size:
        v[idx] = np.linalg.norm(B[hi[idx] - 1] - B[lo[idx]], axis=1) / (span[idx] / FPS)
    return v.astype(np.float32)


def _roll_median(x, w):
    """Centered rolling nanmedian, edges padded by the edge value (as gk_possession does)."""
    x = np.asarray(x, float)
    k = max(1, int(w)) // 2
    pad = np.pad(x, k, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1).astype(np.float32)


def nearest_player_m(match, chunk=1 << 14):
    """(N,) distance from the ball to the nearest tracked player in meters; NaN without a ball.

    Chunked so the (N, 46, 2) difference never materialises for the whole match at float64.
    """
    B = np.asarray(match.B, np.float32)
    P = np.asarray(match.P, np.float32)
    n = len(B)
    out = np.full(n, np.nan, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = P[a:b] - B[a:b, None, :]
        d2 = np.einsum("nij,nij->ni", d, d).astype(np.float32)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[a:b] = np.sqrt(np.nanmin(d2, axis=1))
    return out


def pitch_context(match, st):
    """Label-free restart geometry per frame: line, center spot, penalty mark, goal area, formation.

    Returns a dict of (N,) arrays:
      ``line_m``       distance to the nearest painted line (touchline or goal line);
      ``center_m``     distance to the center spot;
      ``spot_m``       distance to the nearer penalty mark;
      ``goal_area``    bool, ball inside either 6-yard box;
      ``own_half``     share of tracked outfield players in their own half (S09's kickoff statistic).

    Nothing here is fitted; the distances are the Laws' marks (9.15 m center circle, 11 m penalty
    spot, 5.5 m goal area) and the lines of the 105.3 x 68 pitch.
    """
    B = np.asarray(match.B, float)
    n = len(B)
    x, y = B[:, 0], B[:, 1]
    line_m = np.minimum.reduce([np.abs(x), np.abs(x - PITCH_L), np.abs(y), np.abs(y - PITCH_W)])
    center_m = np.hypot(x - PITCH_L / 2, y - PITCH_W / 2)
    spot_m = np.minimum(np.hypot(x - 11.0, y - PITCH_W / 2),
                        np.hypot(x - (PITCH_L - 11.0), y - PITCH_W / 2))
    goal_area = (((x <= GOAL_AREA_M) | (x >= PITCH_L - GOAL_AREA_M))
                 & (np.abs(y - PITCH_W / 2) <= 9.16))
    own = np.full(n, np.nan, np.float32)
    half = np.asarray(match.half)
    P = np.asarray(match.P, float)
    for h in (1, 2):
        sel = np.flatnonzero(half == h)
        if sel.size == 0:
            continue
        cnt = np.zeros(sel.size, np.float32)
        tot = np.zeros(sel.size, np.float32)
        for t in (0, 1):
            s = int(match.gk[h - 1, t])
            idx = [i for i in np.where(np.asarray(match.T) == t)[0] if i != s]
            xx = P[sel][:, idx, 0]
            side = float(st.attack_dir[h - 1, t])
            cnt += np.nansum((xx - PITCH_L / 2) * side <= 1.0, axis=1)
            tot += np.sum(np.isfinite(xx), axis=1)
        with np.errstate(all="ignore"):
            own[sel] = cnt / np.maximum(1, tot)
    return dict(line_m=line_m, center_m=center_m, spot_m=spot_m, goal_area=goal_area,
                own_half=own)


def _present_gk_live(match, st, p):
    """E38 keeper-in-possession mask, imported (never reimplemented). Returns (mask, source, note)."""
    if not p["exclude_gk"]:
        return np.zeros(len(match.B), bool), "disabled", None
    try:
        gk = _import_gk()
    except ImportError as exc:  # no inline equivalent: the carry branch has no copy here
        note = _gk_fallback_note(
            "present-ball", exc,
            "no keeper veto is applied (gk_source='unavailable'), so keeper-hold runs stay candidates")
        return np.zeros(len(match.B), bool), "unavailable", note
    gp = gk.detect(match, st, events=np.zeros(0),
                   params={"include_gathered": True})   # gathered cannot fire: ball is present
    return np.asarray(gp.live) > 0, "gk_possession", None


def _run_meta(a, b, vball, speed, near_m, ctx, p):
    """Per-run summary shared by the mask and the reported (non-mask) classes."""
    sl = slice(a, b)
    near = near_m[sl]
    med_near = float(np.nanmedian(near)) if np.isfinite(near).any() else float("nan")
    share_near = float(np.nanmean(near <= p["near_player_m"])) if np.isfinite(near).any() else 0.0
    line_m = float(np.nanmedian(ctx["line_m"][sl]))
    center_m = float(np.nanmedian(ctx["center_m"][sl]))
    spot_m = float(np.nanmedian(ctx["spot_m"][sl]))
    goal_area = bool(np.nanmean(ctx["goal_area"][sl]) > 0.5)
    own_half = float(np.nanmedian(ctx["own_half"][sl]))
    placed_ctx = bool(line_m <= LINE_M or center_m <= CENTER_M or spot_m <= SPOT_M or goal_area)
    return dict(
        start=int(a), end=int(b), frames=int(b - a), seconds=float((b - a) / FPS),
        mean_speed=float(np.nanmean(speed[sl])) if np.isfinite(speed[sl]).any() else float("nan"),
        ball_speed_med=float(np.nanmedian(vball[sl])) if np.isfinite(vball[sl]).any() else float("nan"),
        nearest_player_m=med_near, near_player_share=share_near,
        kind="held" if share_near > 0.5 else "placed",
        context=dict(line_m=line_m, center_m=center_m, spot_m=spot_m,
                     goal_area=goal_area, own_half=own_half),
        placed=placed_ctx)


def present_ball(match, st=None, params=None):
    """Label-free D proxy for **dead play with the ball visible** (E42).

    Usage::

        from possession import dproxy, data
        m = data.load_match()
        dp = dproxy.proxy(m)             # ball-missing target
        pb = dproxy.present_ball(m)      # ball-present, at-rest target
        pb.mask.sum(), len(pb.runs)      # frames, runs
        pb.runs[0]["kind"], pb.runs[0]["placed"], pb.runs[0]["context"]
        pb.info["coverage_present"], pb.info["gk_vetoed"]
        dproxy.adoption_report(cue, m, pb)   # scored against the present-ball target

    The rule and its justification are in the module docstring. The function is deterministic and
    reads no label; ``params`` overrides :data:`D_PRESENT_PARAMS`. A run's ``grade`` is deliberately
    absent: E25's P(D | missing-run length) prior is about *missingness* and does not transfer.
    """
    p = dict(D_PRESENT_PARAMS, **(params or {}))
    if p["near_player_m"] <= 0 or p["min_hold_s"] < 0:
        raise ValueError("near_player_m must be > 0 and min_hold_s >= 0")
    if st is None:
        st = structure(match)
    n = len(match.B)
    graded = _graded_frames(match, st)
    B = np.asarray(match.B, float)
    present = np.isfinite(B[:, 0])
    speed = mean_player_speed(match, p["speed_clip"])
    still = np.isfinite(speed) & (speed < p["still_v"])
    vball = _roll_median(_robust_ball_speed(B, p["base_s"]), BALL_MED_F)
    near_m = nearest_player_m(match)
    ctx = pitch_context(match, st)
    gk_live, gk_source, gk_note = _present_gk_live(match, st, p)

    # ---- the frozen ball-state half of the rule (used as the ablation base for this target)
    ball_rest = graded & present & np.isfinite(vball) & (vball <= p["ball_still_v"])
    base_rule = ball_rest

    # ---- candidate static runs (every one is reported; only those >= min_hold_s reach the mask)
    min_fr = int(round(p["min_hold_s"] * FPS))
    static_runs = []
    cand = np.zeros(n, bool)
    for a, b in _runs(ball_rest):
        qualifies = (b - a) >= min_fr
        if qualifies:
            cand[a:b] = True
        static_runs.append(dict(_run_meta(a, b, vball, speed, near_m, ctx, p),
                                qualifies=bool(qualifies), proxy_frames=0, gk_vetoed_frames=0))

    # ---- the proxy: ball static + collective stillness + not keeper-in-possession
    mask = cand & still & graded & ~gk_live

    # ---- rolling / carried: same stillness context, ball moving 1-3 m/s. Reported, not asserted.
    slow = (graded & present & np.isfinite(vball)
            & (vball > p["ball_still_v"]) & (vball <= p["roll_v"]))
    slow_runs = []
    rolling = np.zeros(n, bool)
    for a, b in _runs(slow):
        if (b - a) < min_fr:
            continue
        rolling[a:b] = True
        slow_runs.append(dict(_run_meta(a, b, vball, speed, near_m, ctx, p), qualifies=True,
                              gk_vetoed_frames=int(gk_live[a:b].sum())))
    rolling_mask = rolling & still & graded & ~gk_live
    if p["include_rolling"]:
        mask = mask | rolling_mask

    for sr in static_runs:
        sr["proxy_frames"] = int(mask[sr["start"]:sr["end"]].sum())
        sr["gk_vetoed_frames"] = int((gk_live[sr["start"]:sr["end"]] & still[sr["start"]:sr["end"]]
                                      & graded[sr["start"]:sr["end"]]).sum())

    runs = [dict(_run_meta(a, b, vball, speed, near_m, ctx, p)) for a, b in _runs(mask)]
    n_present = int((graded & present).sum())
    info = dict(
        kind="present",
        params=p, min_hold_s=p["min_hold_s"], ball_still_v=p["ball_still_v"],
        still_v=p["still_v"], near_player_m=p["near_player_m"],
        gk_source=gk_source, gk_note=gk_note,
        n_static_runs=len(static_runs), n_qualifying_runs=int(sum(s["qualifies"]
                                                                 for s in static_runs)),
        n_present_frames=n_present, n_ball_rest_frames=int(ball_rest.sum()),
        n_proxy_frames=int(mask.sum()), n_proxy_runs=len(runs),
        coverage_graded=float(mask.sum() / max(1, int(graded.sum()))),
        coverage_present=float(mask.sum() / max(1, n_present)),
        present_share=float(n_present / max(1, int(graded.sum()))),
        gk_frames=int((gk_live & graded).sum()),
        gk_vetoed=int((cand & still & gk_live & graded).sum()),
        n_placed_runs=int(sum(r["kind"] == "placed" for r in runs)),
        n_held_runs=int(sum(r["kind"] == "held" for r in runs)),
        placed_context_frames=int(sum(r["frames"] for r in runs if r["placed"])),
        n_slow_runs=len(slow_runs), n_slow_frames=int(rolling_mask.sum()),
        slow_frames_unvetoed=int((rolling & still & graded).sum()),
        slow_in_gk=int((rolling & still & gk_live & graded).sum()),
        slow_near_player_share=(float(np.nanmean([s["near_player_share"] for s in slow_runs]))
                                if slow_runs else 0.0),
        median_run_s=float(np.median([r["seconds"] for r in runs])) if runs else 0.0,
        forced_frames_ignored=int(np.asarray(st.forced_dead_mask).sum()),
        static_runs=static_runs, slow_runs=slow_runs,
        base_rule=base_rule,           # the ablation's "ball state only" rule for this target
        graded=graded,                 # the exact scope the mask was built on
    )
    return PresentD(mask, runs, slow_runs, vball, info)


def both(match, st=None, params=None, present_params=None):
    """The two targets as one mask: union of the missing-ball and present-ball proxies.

    Returns a :class:`DProxy`-shaped object whose ``info["kind"]`` is ``"both"``, so the same
    :func:`adoption_report` call works. ``runs`` are the union's maximal runs re-labeled with
    ``source`` = "missing" | "present" | "mixed"; ``missing_runs`` and ``speed`` come from the
    missing-ball proxy, and ``info`` carries both children's summaries.
    """
    if st is None:
        st = structure(match)
    dm = proxy(match, st, params)
    pb = present_ball(match, st, present_params)
    union = np.asarray(dm.mask, bool) | np.asarray(pb.mask, bool)
    runs = []
    for a, b in _runs(union):
        ms = bool(dm.mask[a:b].any())
        ps = bool(pb.mask[a:b].any())
        runs.append(dict(start=int(a), end=int(b), frames=int(b - a),
                         seconds=float((b - a) / FPS),
                         mean_speed=float(np.nanmean(dm.speed[a:b])),
                         source="mixed" if (ms and ps) else ("missing" if ms else "present")))
    graded = np.asarray(dm.info["graded"], bool)
    base_rule = (np.asarray(dm.info.get("base_rule", np.zeros(len(union), bool)))
                 if "base_rule" in dm.info else
                 (graded & ~np.isfinite(np.asarray(match.B)[:, 0])))
    base_rule = np.asarray(base_rule, bool) | np.asarray(pb.info["base_rule"], bool)
    info = dict(kind="both", params=dm.info["params"], present_params=pb.info["params"],
                still_v=dm.info["still_v"], min_run_s=dm.info["min_run_s"],
                min_hold_s=pb.info["min_hold_s"],
                n_proxy_frames=int(union.sum()), n_proxy_runs=len(runs),
                coverage_graded=float(union.sum() / max(1, int(graded.sum()))),
                coverage_missing=dm.info["coverage_missing"],
                coverage_present=pb.info["coverage_present"],
                median_run_s=float(np.median([r["seconds"] for r in runs])) if runs else 0.0,
                n_missing_frames=int(dm.mask.sum()), n_present_frames=int(pb.mask.sum()),
                overlap_frames=int((dm.mask & pb.mask).sum()),
                missing_info=dm.info, present_info=pb.info,
                base_rule=base_rule, graded=graded)
    return DProxy(union, runs, dm.missing_runs, dm.speed, info)



# ================================================================ the adoption protocol


def _windows(active, window_s):
    """(N,) bool: union of [onset + a, onset + b] over the runs of ``active``."""
    n = len(active)
    out = np.zeros(n, bool)
    lo = int(round(window_s[0] * FPS))
    hi = int(round(window_s[1] * FPS))
    for s, _ in _runs(active):
        a = max(0, s + lo)
        b = min(n, s + hi)
        if b > a:
            out[a:b] = True
    return out


def _blocks(mask, graded, half):
    """(B, 2) half-open blocks: maximal runs of constant proxy state inside a half.

    A block is a contiguous stretch of match with one proxy value; resampling blocks (not frames)
    is the run-level bootstrap E25/E36 use, and it keeps the temporal dependence inside a draw.
    """
    out = []
    for h in (1, 2):
        idx = np.flatnonzero((half == h) & graded)
        if len(idx) == 0:
            continue
        a, z = int(idx[0]), int(idx[-1]) + 1
        x = np.where(graded[a:z], mask[a:z].astype(np.int8), -1)
        brk = np.flatnonzero(np.diff(x) != 0) + 1
        edges = np.r_[0, brk, len(x)]
        for u, v in zip(edges[:-1], edges[1:]):
            if v > u and x[u] >= 0:
                out.append((a + u, a + v))
    return np.array(out, np.int64).reshape(-1, 2)


def _ci(draws, alpha=ALPHA):
    """Percentile CI and a two-sided bootstrap p for a delta whose sign is meaningful."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(low=float("nan"), high=float("nan"), p=float("nan"), significant=False)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(float(np.mean(d <= 0)), float(np.mean(d >= 0))))
    return dict(low=float(lo), high=float(hi), p=float(p), significant=bool(lo > 0 or hi < 0))


def adoption_report(cue, match, proxy_obj=None, active_thr=None, n_boot=N_BOOT, seed=SEED,
                    alpha=ALPHA):
    """Score a match-only cue against the label-free D proxy and return the pre-registered verdict.

    ``cue`` is a per-frame score (higher = more D). A binary cue (dtype bool, or values only 0/1)
    may omit ``active_thr``; any other cue must pass the threshold that was declared *before* the
    test, and it is echoed in the report. ``proxy_obj`` may be a :func:`proxy` result; otherwise it
    is computed from ``match``. ``match`` is required (for graded/half) even with ``proxy_obj``.
    Returns a dict (definitions in the module docstring)::

        r = adoption_report(whistle_p, match=m)
        r["decision"]      # "adopt" | "park"
        r["lead_gain"], r["lead_ci"]
        r["inside"], r["outside"], r["delta"], r["delta_ci"]
        r["ablation_delta"], r["ablation_ci"]
    """
    if match is None:
        raise ValueError("adoption_report needs the Match (graded/half); pass match=...")
    st = structure(match)
    if proxy_obj is None:
        proxy_obj = proxy(match, st)
    mask = np.asarray(proxy_obj.mask, bool)
    n = len(mask)
    cue = np.asarray(cue)
    if cue.ndim != 1 or len(cue) != n:
        raise ValueError("cue shape {} does not match the proxy's {} frames: refusing to align "
                         "silently".format(cue.shape, n))
    if np.isinf(cue).any():
        raise ValueError("cue contains +-inf; a score must be finite or NaN")

    g_stored = proxy_obj.info.get("graded")
    graded = (np.asarray(g_stored, bool) if g_stored is not None and len(g_stored) == n
              else _graded_frames(match, st))
    half = np.asarray(match.half)

    x = cue.astype(np.float64, copy=True)
    nan_share = float(np.mean(~np.isfinite(x[graded]))) if graded.any() else 1.0
    if nan_share > MAX_NAN_SHARE:
        raise ValueError("cue is NaN on {:.1%} of graded frames (limit {:.0%}); a cue that cannot "
                         "score the match is not ready for adoption".format(nan_share,
                                                                           MAX_NAN_SHARE))
    x[~np.isfinite(x)] = 0.0

    binary = bool(cue.dtype == bool or np.all((x == 0.0) | (x == 1.0)))
    if active_thr is None:
        if not binary:
            raise ValueError("continuous cue without active_thr: declare the threshold before the "
                             "test (it is recorded and cannot be chosen after seeing the CI)")
        active = x > 0.5
        thr_used = None
    else:
        active = x >= float(active_thr)
        thr_used = float(active_thr)
    active &= graded

    after = _windows(active, AFTER_WINDOW_S) & graded
    before = _windows(active, BEFORE_WINDOW_S) & graded
    any_win = after | before
    outside = graded & ~any_win

    # The proxy's domain: frames where it *can* be D. Everywhere else it is 0 by construction, so
    # agreement there is a prevalence artefact (and E15 rejected the crowd on exactly this).
    # E42: the target says which ball state it sees; a cue fired outside that state cannot be scored.
    kind = str(proxy_obj.info.get("kind", "missing"))
    if kind not in ("missing", "present", "both"):
        raise ValueError("unknown proxy kind {!r}; expected 'missing', 'present' or 'both'"
                         .format(kind))
    present = graded & np.isfinite(np.asarray(match.B)[:, 0])
    missing = graded & ~np.isfinite(np.asarray(match.B)[:, 0])
    domain = dict(missing=missing, present=present, both=graded)[kind]
    domain_label = dict(missing="ball-missing", present="ball-present", both="graded")[kind]
    n_after = int((after & domain).sum())
    n_before = int((before & domain).sum())
    if n_after < MIN_DOMAIN_FRAMES or n_before < MIN_DOMAIN_FRAMES:
        raise ValueError("cue windows cover only {} / {} {} frames (need {} each): a proxy cannot "
                         "score a cue that does not fire where it can see"
                         .format(n_after, n_before, domain_label, MIN_DOMAIN_FRAMES))

    # ---- descriptive association (inside vs outside), raw: the card's "agreement inside/outside"
    inside_rate = float(mask[after].mean()) if after.any() else float("nan")
    outside_rate = float(mask[outside].mean()) if outside.any() else float("nan")
    delta = inside_rate - outside_rate

    # ---- lead contrast: the pre-registered decision statistic
    before_rate = float(mask[before].mean())
    lead_gain = inside_rate - before_rate

    # ---- redundancy diagnostic: the same contrast *inside the proxy's domain* (ball missing).
    # Everywhere else the proxy is 0 by construction, so that part of the raw contrast is the proxy's
    # own prevalence; E15 rejected the crowd envelope on exactly this conditional number (AUC .490
    # conditional on missingness).
    cond_inside = float(mask[after & domain].mean())
    cond_before = float(mask[before & domain].mean())
    cond_lead = cond_inside - cond_before

    # ---- ablation: cue-free ball-state rule vs the same rule with the cue's after windows OR-ed in
    base = np.asarray(proxy_obj.info.get("base_rule", missing), bool)
    if len(base) != n:
        base = missing
    fused = base | after
    n_graded = max(1, int(graded.sum()))
    agree_base = float(np.sum((base == mask)[graded]) / n_graded)
    agree_fused = float(np.sum((fused == mask)[graded]) / n_graded)
    ablation = agree_fused - agree_base

    def _bal(pred):
        pred, tgt = pred[graded], mask[graded]
        tp = np.sum(pred & tgt); fn = np.sum(~pred & tgt)
        tn = np.sum(~pred & ~tgt); fp = np.sum(pred & ~tgt)
        tpr = tp / (tp + fn) if (tp + fn) else np.nan
        tnr = tn / (tn + fp) if (tn + fp) else np.nan
        return float(np.nanmean([tpr, tnr]))

    # ---- block bootstrap: resample contiguous runs of constant proxy state
    blocks = _blocks(mask, graded, half)
    n_blocks = int(len(blocks))
    draws_delta, draws_lead, draws_abl = [], [], []
    if n_blocks >= 2:
        bmask_a = np.array([np.sum(mask[a:b] & after[a:b]) for a, b in blocks], float)
        bmask_b = np.array([np.sum(mask[a:b] & before[a:b]) for a, b in blocks], float)
        bmask_o = np.array([np.sum(mask[a:b] & outside[a:b]) for a, b in blocks], float)
        bafter = np.array([after[a:b].sum() for a, b in blocks], float)
        bbefore = np.array([before[a:b].sum() for a, b in blocks], float)
        bout = np.array([outside[a:b].sum() for a, b in blocks], float)
        c_after = np.array([np.sum(mask[a:b] & after[a:b] & domain[a:b]) for a, b in blocks], float)
        c_before = np.array([np.sum(mask[a:b] & before[a:b] & domain[a:b]) for a, b in blocks], float)
        c_na = np.array([np.sum(after[a:b] & domain[a:b]) for a, b in blocks], float)
        c_nb = np.array([np.sum(before[a:b] & domain[a:b]) for a, b in blocks], float)
        bbase_ok = np.array([np.sum((base[a:b] == mask[a:b])) for a, b in blocks], float)
        bfuse_ok = np.array([np.sum((fused[a:b] == mask[a:b])) for a, b in blocks], float)
        bgraded = np.array([graded[a:b].sum() for a, b in blocks], float)
        rng = np.random.default_rng(seed)
        draws_cond = []
        for _ in range(int(n_boot)):
            i = rng.integers(0, n_blocks, n_blocks)
            ins = bmask_a[i].sum() / bafter[i].sum() if bafter[i].sum() > 0 else np.nan
            bef = bmask_b[i].sum() / bbefore[i].sum() if bbefore[i].sum() > 0 else np.nan
            out = bmask_o[i].sum() / bout[i].sum() if bout[i].sum() > 0 else np.nan
            g = bgraded[i].sum()
            if g <= 0:
                continue
            draws_delta.append(ins - out)
            draws_lead.append(ins - bef)
            draws_abl.append((bfuse_ok[i].sum() - bbase_ok[i].sum()) / g)
            if c_na[i].sum() > 0 and c_nb[i].sum() > 0:
                draws_cond.append(c_after[i].sum() / c_na[i].sum()
                                  - c_before[i].sum() / c_nb[i].sum())

    delta_ci = _ci(draws_delta, alpha)
    lead_ci = _ci(draws_lead, alpha)
    abl_ci = _ci(draws_abl, alpha)
    cond_ci = _ci(draws_cond, alpha)

    # ---- decision, fixed in advance: the lead CI must be entirely above zero AND the effect must
    # clear the full-match noise band (GUARDRAILS 3: ~0.04 spellF1). A significant 1-pp shift is not
    # a lever, which is exactly the crowd envelope's shape.
    if not (np.isfinite(lead_ci["low"]) and lead_ci["low"] > 0):
        decision = "park"
        reason = ("cue_is_reactive" if np.isfinite(lead_ci["high"]) and lead_ci["high"] < 0
                  else "lead_ci_covers_zero")
    elif lead_gain < MIN_LEAD_GAIN:
        decision, reason = "park", "effect_below_noise_floor"
    elif cond_lead < MIN_CONDITIONAL_LEAD:
        decision, reason = "park", ("redundant_with_missingness" if kind == "missing"
                                    else "redundant_with_ball_state")
    else:
        decision, reason = "adopt", "lead_ci_excludes_zero"

    warnings = []
    if active.sum() / max(1, int(graded.sum())) > MAX_ACTIVE_SHARE_WARN:
        warnings.append("cue_is_state_not_onset")
    if ablation < 0 and kind == "missing":
        warnings.append("ablation_negative_proxy_blind_to_present_ball_D")
    if nan_share > 0:
        warnings.append("cue_had_nans_scored_as_zero")
    if n_blocks < 20:
        warnings.append("few_blocks_ci_fragile")

    return dict(
        decision=decision, reason=reason, warnings=warnings,
        cue=dict(active_thr=thr_used, binary=binary, active_frames=int(active.sum()),
                 active_share=float(active.sum() / max(1, int(graded.sum()))),
                 n_events=int(len(_runs(active))), nan_share=nan_share),
        inside=inside_rate, outside=outside_rate, delta=delta,
        before=before_rate, lead_gain=lead_gain,
        conditional_inside=cond_inside, conditional_before=cond_before,
        conditional_lead_gain=cond_lead, conditional_lead_ci=cond_ci,
        n_after_domain=n_after, n_before_domain=n_before,
        lead_ci=lead_ci, delta_ci=delta_ci,
        ablation_delta=ablation, ablation_ci=abl_ci,
        agreement_masked=agree_base, agreement_fused=agree_fused,
        balanced_masked=_bal(base), balanced_fused=_bal(fused),
        target=dict(kind=kind, domain=domain_label,
                    proxy_coverage_graded=float(proxy_obj.info.get("coverage_graded", np.nan)),
                    proxy_coverage_missing=float(proxy_obj.info.get("coverage_missing", np.nan)),
                    proxy_coverage_present=float(proxy_obj.info.get("coverage_present", np.nan)),
                    graded_frames=int(graded.sum())),
        params=dict(after_window_s=AFTER_WINDOW_S, before_window_s=BEFORE_WINDOW_S,
                    n_boot=int(n_boot), seed=int(seed), alpha=float(alpha),
                    max_nan_share=MAX_NAN_SHARE, min_run_s=proxy_obj.info.get("min_run_s"),
                    still_v=proxy_obj.info.get("still_v")),
        n_blocks=n_blocks,
    )


def format_report(r):
    """One-screen summary of an :func:`adoption_report` (for a card Result or an ADOPTION row)."""
    def ci(c):
        return "[{:+.3f}, {:+.3f}] p={:.3f}".format(c["low"], c["high"], c["p"])

    return "\n".join([
        "cue: {} events, {:.1%} of graded frames active, thr={}".format(
            r["cue"]["n_events"], r["cue"]["active_share"], r["cue"]["active_thr"]),
        "inside {:.3f} vs outside {:.3f}  delta {:+.3f} {}  (domain frames {} / {})".format(
            r["inside"], r["outside"], r["delta"], ci(r["delta_ci"]),
            r["n_after_domain"], r["n_before_domain"]),
        "lead_gain {:+.3f} (before {:.3f})  {}   <- decision statistic; min {:.2f}".format(
            r["lead_gain"], r["before"], ci(r["lead_ci"]), MIN_LEAD_GAIN),
        "conditional lead {:+.3f} (inside domain {:.3f}, before {:.3f})  {}  <- redundancy check".format(
            r["conditional_lead_gain"], r["conditional_inside"], r["conditional_before"],
            ci(r["conditional_lead_ci"])),
        "ablation {:+.4f}  {}  (masked agreement {:.4f}, fused {:.4f})".format(
            r["ablation_delta"], ci(r["ablation_ci"]), r["agreement_masked"], r["agreement_fused"]),
        "DECISION: {} ({}){}".format(r["decision"], r["reason"],
                                     "  warnings: " + ",".join(r["warnings"]) if r["warnings"] else ""),
    ])


def main(argv=None):
    """``python -m possession.dproxy --proxy {missing,present,both} --cue cue.npy [--thr X]``.

    ``--proxy`` has no default on purpose: a caller must state which dead-ball target the cue is
    scored against (E42). ``cue.npy`` is an (N,) per-frame array on the match's frame grid.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="dproxy", description=main.__doc__)
    ap.add_argument("--proxy", required=True, choices=["missing", "present", "both"])
    ap.add_argument("--cue", required=True, help="path to an (N,) .npy cue")
    ap.add_argument("--thr", type=float, default=None, help="activation threshold (continuous cues)")
    a = ap.parse_args(argv)
    from . import data
    m = data.load_match()
    target = dict(missing=proxy, present=present_ball, both=both)[a.proxy](m)
    print(format_report(adoption_report(np.load(a.cue), m, target, active_thr=a.thr)))


if __name__ == "__main__":
    main()
