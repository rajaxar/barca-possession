"""Label-free dead-ball (D) proxy for BAR-ATH, and the adoption protocol for match-only cues.

This module exists because our strongest inputs (referee audio, broadcast cutaways, camera pan,
Smart Tagging) exist only on the match we are predicting. They can never be selected on an external
dev set, so the temptation is to adopt them by looking at the public 5-minute window -- which
GUARDRAILS 2 forbids. The answer is not "drop them": it is to score them against one shared,
label-free target, with a decision rule written down before the test is run.

THE PROXY
---------
A frame is proxy-``D`` iff

    the frame is inside a ball-missing run of at least ``MIN_RUN_S`` (2 s)
    AND mean tracked-player speed on the frame is below ``STILL_V`` (1.5 m/s)
    AND the frame is not goalkeeper-in-possession (E38 exclusion, on by default)

with ungraded frames and the published forced-D 2H pre-kickoff block never in the mask
(``structure.forced_dead_mask``; both are D by structure, not by evidence).

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
  **ablation delta** (a frozen D rule ``ball missing`` with the cue's after windows OR-ed in,
  scored against the proxy over all graded frames; masking the cue removes them) are reported with
  their own CIs. The ablation is *not* the decision gate: because the proxy is blind to present-ball
  D, a real onset cue loses it by construction (the whistle loses 0.009), while a state cue that
  fires for a fifth of the match loses much more (the crowd envelope loses 0.178).

Refusals (all before any statistic is computed, so they cannot be negotiated after the fact):
  * ``len(cue) != len(proxy mask)`` -> ValueError (a silent misalignment is the worst failure here);
  * more than ``MAX_NAN_SHARE`` (5%) of graded frames NaN -> ValueError (NaN is not "no evidence";
    a cue that cannot score 5% of the match is not ready). At or below 5%, NaN is scored as 0;
  * infinities -> ValueError;
  * fewer than ``MIN_DOMAIN_FRAMES`` (100, i.e. 4 s) ball-missing frames in either window ->
    ValueError: the proxy is blind to present-ball D (failure mode 1), so a cue that only fires
    there cannot be scored against it, and a spurious zero would be worse than a refusal.

Blocks for the bootstrap are maximal runs of constant proxy state inside a half, so a draw is a
contiguous stretch of match, not a frame. Seeds are fixed (``SEED``), so a report is reproducible.
"""
import warnings
from collections import namedtuple

import numpy as np

from .structure import PITCH_L, structure

FPS = 25.0

# ---------------------------------------------------------------- inherited constants (see docstring)
MIN_RUN_S = 2.0      # E25 gap-length gradient .09/.39/.58/.85; Q08 <=1 s runs carry no D
STILL_V = 1.5        # walk/jog boundary: I01, E25, E15, E38
SPEED_CLIP = 12.0    # physical sprint guard (E16 speed spikes), not a fitted cut

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

DProxy = namedtuple("DProxy", "mask runs missing_runs speed info")
# mask: (N,) bool, the per-frame proxy (False on ungraded, forced-D and GK frames)
# runs: list of dict(start, end, frames, seconds, mean_speed, still_share, missing_run, grade)
#       -- end exclusive; one entry per maximal proxy run
# missing_runs: list of dict(start, end, frames, seconds, mean_speed, still_share, qualifies,
#       proxy_frames, proxy_seconds, grade, gk_excluded_frames) -- every missing run, proxy or not
# speed: (N,) float32 mean tracked-player speed (NaN when no player is tracked)
# info: dict with the thresholds used, coverage, the GK-exclusion accounting, and the exact
#       `graded` scope the mask was built on (the protocol reuses it, so they cannot drift apart)

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
    continuously tracked play and ends with a fast return. It exists so the proxy keeps its
    exclusion if the module is unavailable, not to replace E38. The carry branch cannot overlap the
    proxy at all (the ball is present by construction), so it is not reimplemented here.
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


def _gk_live(match, st, params):
    """(N,) bool of goalkeeper-in-possession frames, plus the source label.

    E38's detector is the real thing; the inline fallback is only for a missing module and says so.
    """
    if not params["exclude_gk"]:
        return np.zeros(len(match.B), bool), "disabled"
    try:
        from . import gk_possession as gk
        gp = gk.detect(match, st, events=np.zeros(0),
                       params={"include_gathered": bool(params["gk_include_gathered"])})
        return np.asarray(gp.live) > 0, "gk_possession"
    except Exception:  # missing module / unexpected failure: keep the exclusion, never crash
        return _inline_gk(match, st, params), "inline"


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

    gk_live, gk_source = _gk_live(match, st, p)

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
        params=p, min_run_s=p["min_run_s"], still_v=p["still_v"], speed_clip=p["speed_clip"],
        run_gate=p["run_gate"], gk_source=gk_source,
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
    missing = graded & ~np.isfinite(np.asarray(match.B)[:, 0])
    domain = missing
    n_after = int((after & domain).sum())
    n_before = int((before & domain).sum())
    if n_after < MIN_DOMAIN_FRAMES or n_before < MIN_DOMAIN_FRAMES:
        raise ValueError("cue windows cover only {} / {} ball-missing frames (need {} each): the "
                         "proxy is blind to present-ball D, so this cue cannot be scored against it"
                         .format(n_after, n_before, MIN_DOMAIN_FRAMES))

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

    # ---- ablation: cue-free D rule (ball missing) vs the same rule with the cue's after windows
    base = missing
    fused = missing | after
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
        decision, reason = "park", "redundant_with_missingness"
    else:
        decision, reason = "adopt", "lead_ci_excludes_zero"

    warnings = []
    if active.sum() / max(1, int(graded.sum())) > MAX_ACTIVE_SHARE_WARN:
        warnings.append("cue_is_state_not_onset")
    if ablation < 0:
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
        target=dict(proxy_coverage_graded=float(proxy_obj.info.get("coverage_graded", np.nan)),
                    proxy_coverage_missing=float(proxy_obj.info.get("coverage_missing", np.nan)),
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
