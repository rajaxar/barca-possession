"""Degradation simulator (S06): make clean external tracking look like our ATD input.

``degrade_game(game, level=1.0, seed=0)`` returns an :class:`possession.evaluate.Game` whose match
looks like the BAR-ATH tactical ATD to a possession model, with ``ref`` and ``runs`` unchanged.
``degrade(match, level, seed, dead)`` is the same on a bare Match. Steps, in order:

1. **Leak scrub** (always on, independent of ``level``). Every frame that is dead (``dead``, the
   external reference's D frames) or has no native ball is treated as unobserved: IDSSE freezes the
   ball through every D block and teleports it at the restart, Metrica blanks it, and both are label
   leaks. Each such run is rebuilt from a donor: the ball's velocity relative to the players'
   centroid, copied from random alive stretches of the same half, integrated from the last observed
   ball and linearly bridged onto the next one; a speed-capped follower then pulls it towards the
   nearest player, so it sits at players' feet like an alive ball. No frozen ball, NaN or teleport
   marks a D frame. A bounded carry-like wander (S17, ``wobble_*``) adds independent motion, because
   on the ATD a stopped ball is nudged around rather than glued to a walker's foot; its amplitude
   scales with the game's own alive ball pace (``wobble_pace_ref``), like ball speed scales with
   play speed on the ATD.
1b. **Vendor low-pass** (S17/E18). A Gaussian low-pass (σ ≈ 8 frames) over the ball and every player
   track, applied within each half and within present stretches only — never across NaN gaps or half
   boundaries. The ATD ball and players carry this filter (E18), and it is what kills kinematic
   touch detection on the dev set. Smoothing runs before dropout/re-slotting, like a vendor filter
   over continuous tracking; the dropout gap pattern is drawn afterwards and is label-independent.
2. **Ball noise**: a smooth AR(1) offset (decouples Metrica's player-snapped ball from the
   possessor's feet), white jitter at the ATD's mm scale, and short false detections (frame-to-frame
   jumps > 40 m/s).
3. **Ball dropout** in long blocks: alternating present runs and gaps, both drawn from our ATD
   run-length histograms, independent of labels and of position. (ATD presence is lower when play
   is near a goal line, but D also concentrates there, so a spatial dropout bias makes "ball
   missing" predict D; ``drop_zone_gamma`` > 0 re-enables it for ablations, default off.)
4. **Player tracks**: every outfield track is cut into slot segments drawn from the ATD segment
   histogram, each segment lands in a random free slot of the team (ID swaps), some cuts leave a
   short untracked gap (missing players), and ghost duplicates of tracked players appear (>11 per
   team). Goalkeepers keep one slot per team, with short gaps.
5. **Ball s channel**: ATD's unexplained ``BS`` is emulated as random runs of smooth values,
   independent of everything else (the external ``BS`` is the vendor ball speed, a D leak).

Every distribution and rate is fitted to label-free statistics of our own ATD input
(:func:`input_stats` on ``data.load_match()``); no ground truth of any game is read. The fitted
values and the ATD targets are in ``research/S06_degradation_fit.md``. ``level`` scales every
rate and noise amplitude linearly (0 = ball scrub and player re-slotting only, 1 = ATD-like);
only level 1 is fitted.
"""
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from . import data, evaluate, external

FPS = 25
PITCH = (105.0, 68.0)  # all external games (S05 manifest)

# ATD quantiles (0, 5, ..., 100 %) from fit_targets(data.load_match()), graded frames only.
# Ball-missing and ball-present run lengths and outfield slot segment lengths, in frames.
ATD_GAP_Q = (1, 2, 12, 134, 146, 152, 170, 187, 215, 239, 268, 289, 314, 404, 486, 526, 603,
             727, 845, 1270, 5537)
ATD_PRESENT_Q = (16, 40, 54, 105, 115, 137, 174, 213, 235, 300, 345, 458, 571, 614, 706, 758, 940,
                 1222, 1542, 2267, 3667)
ATD_SEG_Q = (8, 42, 65, 92, 140, 194, 244, 317, 427, 618, 873, 1192, 1582, 2140, 2773, 3480, 4367,
             5443, 6819, 9459, 30811)

PARAMS = dict(
    # leak scrub: pull of the rebuilt ball towards the nearest player (0-1), ramp-in at the run
    # edges (frames), and the speed cap of the ball following its target (m/s)
    scrub_pull=0.9, scrub_pull_tau=10, scrub_pull_vmax=20.0,
    # D-ball wobble (S17): independent carry-like wander on rebuilt frames (m/s OU velocity,
    # position leash time constant (frames), position cap (m), edge ramp (frames); 0 = off).
    # The velocity std is scaled to the game's own alive pace (see wobble_pace_ref) so the D
    # ball is not uniformly slow on fast games -- the ATD coupling curve says ball speed scales
    # with play speed. A per-game scalar cannot mark D frames.
    wobble_sigma_v=3.0, wobble_tau_v=20.0, wobble_tau_p=20.0, wobble_cap=3.0, wobble_edge_tau=10.0,
    wobble_pace_ref=6.15, wobble_pace_range=(0.8, 1.3),
    # ball dropout: present-run shortening near the goal lines (see ball_dropout)
    drop_zone_gamma=0.0, drop_zone_ref=0.3,
    # vendor low-pass (E18/S17): Gaussian sigma (frames) on ball and players, fitted to ATD velocity PSD
    smooth_ball_sigma=8.0, smooth_player_sigma=8.0,
    # ball noise: AR(1) offset (m, frames), white jitter (m), false detections (per frame, frames, m)
    ball_ar_tau=50, ball_ar_sigma=0.4, ball_white_sigma=0.0005,
    ball_false_rate=8e-4, ball_false_max_len=3, ball_false_m=(2.0, 15.0),
    # player tracks: slots per team, segment length multiplier (cuts at clean track ends shorten
    # segments), P(untracked gap at a segment cut), mean gap (frames),
    # GK segment length multiplier, ghost starts per team-frame, ghost length multiplier, offsets (m)
    slots_per_team=23, seg_scale=1.25, gap_prob=0.7, gap_mean=320, gk_seg_scale=10.0,
    ghost_rate=1.6e-3, ghost_seg_scale=0.3, ghost_offset_m=(0.5, 3.0), ghost_drift_m=0.5,
    # ball s channel (ATD fit_targets): run starts per frame, run-length quantiles, median, max (m/s)
    bs_rate=1.83e-4, bs_median=1.13, bs_max=7.27,
    bs_run_q=(5, 21, 47, 67, 76, 80, 87, 104, 122, 131, 163, 192, 211, 226, 334, 420, 444, 488, 601,
              952, 7042),
)


# ---------------------------------------------------------------- helpers

def _runs(mask):
    """(k, 2) half-open runs of true values."""
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def _halves(half):
    """[(h, a, b)] half-open frame span of each half present in ``half``."""
    out = []
    for h in np.unique(half):
        if h <= 0:
            continue
        idx = np.flatnonzero(half == h)
        out.append((int(h), int(idx[0]), int(idx[-1]) + 1))
    return out


def _quantile_sampler(q, rng):
    """Draw from the distribution with (equally spaced) quantiles ``q``, log-linear in between."""
    logq = np.log(np.maximum(np.asarray(q, float), 1.0))
    grid = np.linspace(0, 1, len(logq))

    def draw(n=None, scale=1.0):
        u = rng.random(n)
        return np.maximum(1, np.round(np.exp(np.interp(u, grid, logq)) * scale)).astype(np.int64)
    return draw


def _ar1(n, tau, sigma, rng, dims=2):
    """Stationary AR(1) noise with time constant ``tau`` frames and marginal std ``sigma``."""
    if sigma <= 0 or n == 0:
        return np.zeros((n, dims))
    phi = np.exp(-1.0 / tau)
    e = rng.normal(0, sigma * np.sqrt(1 - phi * phi), (n, dims))
    e[0] = rng.normal(0, sigma, dims)
    return lfilter([1.0], [1.0, -phi], e, axis=0)


def _nanmean_rows(x):
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=1)


def _centroid(P):
    """Mean xy of all tracked players per frame, forward/back filled where nobody is tracked."""
    C = _nanmean_rows(P)
    bad = np.isnan(C[:, 0])
    if bad.all():
        return np.zeros_like(C)
    if bad.any():
        idx = np.where(~bad, np.arange(len(C)), 0)
        np.maximum.accumulate(idx, out=idx)
        first = np.flatnonzero(~bad)[0]
        idx[:first] = first
        C = C[idx]
    return C


# ---------------------------------------------------------------- 1. leak scrub

def _wobble_run(L, sigma_v, tau_v, tau_p, cap, edge_tau, rng):
    """Bounded carry-like wander for one rebuilt run (S17): OU velocity (std ``sigma_v`` m/s,
    time constant ``tau_v`` frames), position offset leashed to 0 with time constant ``tau_p``,
    hard-capped at ``cap`` m and ramped to 0 at both run ends. Returns (L, 2)."""
    phi = np.exp(-1.0 / tau_v)
    k = 1.0 / tau_p
    v = np.empty((L, 2))
    v[0] = rng.normal(0, sigma_v, 2)
    for i in range(1, L):
        v[i] = phi * v[i - 1] + rng.normal(0, sigma_v * np.sqrt(1 - phi * phi), 2)
    x = np.empty((L, 2))
    x[0] = v[0] / FPS
    for i in range(1, L):
        x[i] = x[i - 1] + v[i] / FPS - k * x[i - 1]
    norm = np.linalg.norm(x, axis=1)
    x = x * (np.minimum(norm, cap) / np.maximum(norm, 1e-9))[:, None]
    kk = np.arange(L)
    w = np.minimum(1.0, np.minimum(kk + 1, L - kk) / edge_tau)
    return x * w[:, None]


def scrub_ball(B, P, half, invalid, rng, pull=0.0, pull_tau=10, pull_vmax=30.0,
               wobble_sigma_v=0.0, wobble_tau_v=20.0, wobble_tau_p=20.0, wobble_cap=3.0,
               wobble_edge_tau=10.0):
    """Rebuild the ball on ``invalid`` frames from donor alive motion relative to the centroid."""
    out = B.astype(np.float64).copy()
    C = _centroid(P)
    rel = out - C
    for _, a, b in _halves(half):
        bad = invalid[a:b] | np.isnan(out[a:b, 0])
        ok_runs = [(s, e) for s, e in _runs(~bad) if e - s >= 2]
        if not ok_runs:
            continue
        steps = [np.diff(rel[a + s:a + e], axis=0) for s, e in ok_runs]
        lengths = np.array([len(x) for x in steps])
        weights = lengths / lengths.sum()
        n = b - a
        for s, e in _runs(bad):
            L = e - s
            need = L + 1 if e < n else L
            parts, got = [], 0
            while got < need:  # concatenate random donor stretches, length-weighted
                k = rng.choice(len(steps), p=weights)
                st = rng.integers(0, lengths[k])
                take = min(lengths[k] - st, need - got)
                parts.append(steps[k][st:st + take])
                got += take
            dv = np.concatenate(parts)
            if s > 0:
                start = rel[a + s - 1]
                path = start + np.cumsum(dv, axis=0)
                if e < n:  # bridge onto the next observed ball
                    err = path[-1] - rel[a + e]
                    path -= err * (np.arange(1, need + 1) / need)[:, None]
                path = path[:L]
            else:  # half starts unobserved: integrate backwards from the first observed ball
                path = rel[a + e] - np.cumsum(dv[::-1], axis=0)[::-1][:L]
            x = path + C[a + s:a + e]
            if pull > 0:  # alive balls sit at players' feet: follow the nearest player, speed-capped
                d = np.linalg.norm(P[a + s:a + e] - x[:, None, :], axis=-1)
                d = np.where(np.isnan(d), np.inf, d)
                j = d.argmin(axis=1)
                target = x + pull * np.where(np.isfinite(d.min(axis=1))[:, None],
                                             P[a + s:a + e][np.arange(L), j] - x, 0.0)
                y = np.empty_like(x)
                cur = x[0]
                cap = pull_vmax / FPS
                for i in range(L):
                    step = target[i] - cur
                    norm = np.hypot(step[0], step[1])
                    cur = cur + (step if norm <= cap else step * (cap / norm))
                    y[i] = cur
                k = np.arange(L)
                w = np.minimum(1.0, np.minimum(k + 1, L - k) / pull_tau)
                x = x + w[:, None] * (y - x)
            if wobble_sigma_v > 0:  # S17: independent carry-like wander of the rebuilt ball
                x = x + _wobble_run(L, wobble_sigma_v, wobble_tau_v, wobble_tau_p, wobble_cap,
                                    wobble_edge_tau, rng)
            rel[a + s:a + e] = x - C[a + s:a + e]
        out[a:b] = rel[a:b] + C[a:b]
    return out


# ---------------------------------------------------------------- 1b. vendor low-pass

def lowpass(X, half, sigma):
    """Gaussian low-pass (``sigma`` frames) along time within each half and each run of finite
    rows of ``X`` (n, ...); never across a NaN gap or a half boundary."""
    out = np.array(X, np.float64)
    if sigma <= 0:
        return out
    finite = np.isfinite(out.reshape(len(out), -1)).all(axis=1)
    for _, a, b in _halves(half):
        for s, e in _runs(finite[a:b]) + a:
            out[s:e] = gaussian_filter1d(out[s:e], sigma, axis=0, mode="nearest")
    return out


def lowpass_players(P, V, half, sigma):
    """``lowpass`` on every player track (positions and speeds) separately."""
    P2, V2 = P.copy(), V.copy()
    for s in range(P.shape[1]):
        P2[:, s] = lowpass(P[:, s], half, sigma)
        V2[:, s] = lowpass(np.where(np.isnan(P[:, s, 0]), np.nan, V[:, s]), half, sigma)
    return P2, V2


# ---------------------------------------------------------------- 2-3. ball noise and dropout

def ball_noise(B, half, rng, p, level):
    out = B.copy()
    for _, a, b in _halves(half):
        n = b - a
        # the vendor filter smooths tracking errors too: the slow offset gets the same low-pass
        out[a:b] += lowpass(_ar1(n, p["ball_ar_tau"], p["ball_ar_sigma"] * level, rng), np.ones(n),
                            p["smooth_ball_sigma"] * level)
        out[a:b] += rng.normal(0, p["ball_white_sigma"] * level, (n, 2))
        k = rng.poisson(p["ball_false_rate"] * level * n)
        for t in rng.integers(a, b, k):
            d = rng.integers(1, p["ball_false_max_len"] + 1)
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(*p["ball_false_m"])
            out[t:min(t + d, b)] += r * np.array([np.cos(ang), np.sin(ang)])
    return out


def ball_dropout(B, P, half, rng, p, level, pitch=PITCH):
    """NaN the ball in long blocks: alternating present runs and gaps, both drawn from the ATD
    run-length histograms. A present run that starts with the players' centroid far from halfway
    is shortened by exp(-gamma * (dz - dz_ref)), dz = |centroid x - halfway| / half length (ATD
    presence is lower near the goal lines). Each half starts at a random phase of the cycle."""
    out = B.copy()
    if level <= 0:
        return out
    draw_gap = _quantile_sampler(ATD_GAP_Q, rng)
    draw_present = _quantile_sampler(ATD_PRESENT_Q, rng)
    dz = np.abs(_centroid(P)[:, 0] - pitch[0] / 2) / (pitch[0] / 2)
    for _, a, b in _halves(half):
        t = a - int(rng.integers(0, draw_present() + draw_gap()))
        while t < b:
            zone = np.exp(-p["drop_zone_gamma"] * (dz[max(t, a)] - p["drop_zone_ref"]))
            t += int(draw_present(scale=zone / level))
            g = int(draw_gap())
            out[max(t, a):max(min(t + g, b), a)] = np.nan
            t += g
    return out


# ---------------------------------------------------------------- 4. player tracks

def player_tracks(P, V, T, gk, half, rng, p, level):
    """Fragment, swap, drop and duplicate player tracks into a fixed 23 + 23 slot layout."""
    n = len(P)
    k_team = p["slots_per_team"]
    P2 = np.full((n, 2 * k_team, 2), np.nan, np.float32)
    V2 = np.full((n, 2 * k_team), np.nan, np.float32)
    T2 = np.repeat(np.array([0, 1], np.int8), k_team)
    draw_seg = _quantile_sampler(ATD_SEG_Q, rng)
    q_gap = min(1.0, p["gap_prob"] * level)
    for h, a, b in _halves(half):
        for team in (0, 1):
            base = team * k_team
            gk_slot = gk[h - 1, team]
            gk_out = base + k_team - 1
            pieces = []  # (start, stop, source slot)
            for s in np.flatnonzero(T == team):
                for ra, rb in _runs(~np.isnan(P[a:b, s, 0])) + a:
                    t = ra
                    while t < rb:
                        if s == gk_slot:
                            seg = int(draw_seg(scale=p["gk_seg_scale"]))
                        else:
                            seg = int(draw_seg(scale=p["seg_scale"]))
                        stop = min(t + seg, rb)
                        pieces.append((t, stop, s))
                        t = stop
                        if t < rb and rng.random() < q_gap:
                            t += 1 + rng.geometric(1.0 / p["gap_mean"])
            free = np.full(k_team - 1, a)  # next free frame per outfield output slot
            last = {}
            for start, stop, s in sorted(pieces):
                if s == gk_slot:
                    o = gk_out
                else:
                    cand = np.flatnonzero(free <= start)
                    cand = cand[cand != last.get(s, -1)] if len(cand) > 1 else cand
                    if not len(cand):
                        continue
                    j = int(rng.choice(cand))
                    free[j] = stop
                    last[s] = j
                    o = base + j
                P2[start:stop, o] = P[start:stop, s]
                V2[start:stop, o] = V[start:stop, s]
            # ghost duplicates of tracked outfield players
            m = rng.poisson(p["ghost_rate"] * level * (b - a))
            for start in np.sort(rng.integers(a, b, m)):
                live = [j for j in range(k_team - 1) if not np.isnan(P2[start, base + j, 0])]
                cand = np.flatnonzero(free <= start)
                if not live or not len(cand):
                    continue
                src = base + int(rng.choice(live))
                j = int(rng.choice(cand))
                stop = min(start + int(draw_seg(scale=p["ghost_seg_scale"])), b)
                alive = ~np.isnan(P2[start:stop, src, 0])
                stop = start + (int(np.argmin(alive)) if not alive.all() else len(alive))
                if stop <= start:
                    continue
                ang = rng.uniform(0, 2 * np.pi)
                off = rng.uniform(*p["ghost_offset_m"]) * np.array([np.cos(ang), np.sin(ang)])
                off = off + lowpass(_ar1(stop - start, FPS, p["ghost_drift_m"], rng), np.ones(stop - start),
                                    p["smooth_player_sigma"] * level)
                P2[start:stop, base + j] = P2[start:stop, src] + off
                V2[start:stop, base + j] = V2[start:stop, src]
                free[j] = stop
    return P2, V2, T2


# ---------------------------------------------------------------- 5. ball s channel

def ball_s_channel(n, half, rng, p, level):
    BS = np.full(n, np.nan, np.float32)
    draw = _quantile_sampler(p["bs_run_q"], rng)
    for _, a, b in _halves(half):
        k = rng.poisson(p["bs_rate"] * level * (b - a))
        for t in rng.integers(a, b, k):
            stop = min(b, t + int(draw()))
            BS[t:stop] = np.clip(p["bs_median"] * np.exp(_ar1(stop - t, FPS, 0.6, rng, 1)[:, 0]),
                                 0, p["bs_max"])
    return BS


# ---------------------------------------------------------------- API

def degrade(match, level=1.0, seed=0, dead=None, params=None, pitch=PITCH):
    """ATD-like copy of ``match``. ``dead`` (bool per frame) marks frames whose ball is scrubbed
    as a label leak (external D frames); frames with a missing native ball are always scrubbed."""
    p = dict(PARAMS, **(params or {}))
    rng = np.random.default_rng(seed)
    n = len(match.B)
    dead = np.zeros(n, bool) if dead is None else np.asarray(dead, bool)
    wob = p["wobble_sigma_v"]
    if wob > 0 and p.get("wobble_pace_ref"):
        # per-game pace (label-free on our ATD; uses the external D mask like the donor pool does)
        sp = _ball_speed(match.B, match.half)
        alive = ~dead & np.isfinite(match.B[:, 0]) & np.isfinite(sp)
        pace = np.median(sp[alive]) if alive.any() else p["wobble_pace_ref"]
        wob = wob * float(np.clip(pace / p["wobble_pace_ref"], *p["wobble_pace_range"]))
    B = scrub_ball(match.B, match.P, match.half, dead, rng, p["scrub_pull"],
                   p["scrub_pull_tau"], p["scrub_pull_vmax"], wob,
                   p["wobble_tau_v"], p["wobble_tau_p"], p["wobble_cap"], p["wobble_edge_tau"])
    B = lowpass(B, match.half, p["smooth_ball_sigma"] * level)
    B = ball_noise(B, match.half, rng, p, level)
    B = ball_dropout(B, match.P, match.half, rng, p, level, pitch)
    P, V = lowpass_players(match.P, match.V, match.half, p["smooth_player_sigma"] * level)
    P, V, T = player_tracks(P, V, match.T, match.gk, match.half, rng, p, level)
    BS = ball_s_channel(n, match.half, rng, p, level)
    gk = external._goalkeepers(P, T, match.half, pitch[0])
    return data.Match(P=P, V=V, T=T, B=B.astype(np.float32), BS=BS, gk=gk,
                      graded=match.graded, half=match.half)


def degrade_game(game, level=1.0, seed=0, params=None):
    """Degraded copy of an external ``evaluate.Game``; its D frames are scrubbed, ref unchanged."""
    s = [seed, zlib.crc32(game.name.encode())]
    m = degrade(game.match, level, s, np.asarray(game.ref) == "D", params)
    return evaluate.Game(game.name, m, game.ref, game.runs)


# ---------------------------------------------------------------- statistics (label-free)

def _nearest_player_m(P, B):
    """Distance (m) from the ball to the nearest tracked player; NaN without a ball or players."""
    d = np.linalg.norm(P - B[:, None, :], axis=-1)
    out = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def _ball_speed(B, half):
    """Frame-to-frame speed (m/s) into each frame; NaN at half starts and around missing balls."""
    sp = np.full(len(B), np.nan)
    sp[1:] = np.linalg.norm(np.diff(B, axis=0), axis=1) * FPS
    sp[1:][half[1:] != half[:-1]] = np.nan
    return sp


def fit_targets(match):
    """Distributions the simulator samples from, measured on ``match`` (our ATD) graded frames."""
    g = match.graded & (match.half > 0)
    nan = np.isnan(match.B[:, 0])
    q = np.linspace(0, 100, 21)
    gaps, pres, segs, bs_runs = [], [], [], []
    gk = set(np.unique(match.gk).tolist())
    for _, a, b in _halves(np.where(g, match.half, 0)):
        r = _runs(nan[a:b])
        gaps += list(r[:, 1] - r[:, 0])
        r = _runs(~nan[a:b])
        pres += list(r[:, 1] - r[:, 0])
        for s in range(match.P.shape[1]):
            if s not in gk:
                r = _runs(~np.isnan(match.P[a:b, s, 0]))
                segs += list(r[:, 1] - r[:, 0])
        r = _runs(np.isfinite(match.BS[a:b]))
        bs_runs += list(r[:, 1] - r[:, 0])
    bs = match.BS[g & np.isfinite(match.BS)]
    return dict(gap_q=np.percentile(gaps, q), present_q=np.percentile(pres, q), seg_q=np.percentile(segs, q),
                bs_run_q=np.percentile(bs_runs, q), bs_rate=len(bs_runs) / g.sum(),
                bs_median=float(np.median(bs)), bs_max=float(bs.max()))


def input_stats(match, pitch=PITCH):
    """Label-free statistics compared between our ATD and degraded external games (graded frames)."""
    L, W = pitch
    g = match.graded & (match.half > 0)
    nan = np.isnan(match.B[:, 0])
    out = {"ball_missing": nan[g].mean()}
    gaps, pres, segs = [], [], []
    gk = set(np.unique(match.gk).tolist())
    for _, a, b in _halves(np.where(g, match.half, 0)):
        r = _runs(nan[a:b]); gaps += list(r[:, 1] - r[:, 0])
        r = _runs(~nan[a:b]); pres += list(r[:, 1] - r[:, 0])
        for s in range(match.P.shape[1]):
            if s not in gk:
                r = _runs(~np.isnan(match.P[a:b, s, 0])); segs += list(r[:, 1] - r[:, 0])
    gaps, pres, segs = (np.array(x + [0]) for x in (gaps, pres, segs))
    out["gap_median_fr"] = np.median(gaps)
    out["gap_p90_fr"] = np.percentile(gaps, 90)
    out["missing_in_gaps_ge100"] = gaps[gaps >= 100].sum() / g.sum()
    out["present_run_median_fr"] = np.median(pres)
    dz = np.abs(_centroid(match.P)[:, 0] - L / 2)
    out["present_centroid_far"] = (~nan[g & (dz > 20)]).mean()  # centroid > 20 m from halfway
    out["present_centroid_mid"] = (~nan[g & (dz < 10)]).mean()
    cnt = [(~np.isnan(match.P[:, match.T == t, 0])).sum(1)[g] for t in (0, 1)]
    out["players_per_team"] = np.mean([c.mean() for c in cnt])
    out["team_gt11"] = np.mean([(c > 11).mean() for c in cnt])
    out["team_le9"] = np.mean([(c <= 9).mean() for c in cnt])
    out["slot_seg_median_fr"] = np.median(segs)
    out["slot_seg_p90_fr"] = np.percentile(segs, 90)
    sp = _ball_speed(match.B, match.half)
    ok = g & np.isfinite(sp)
    out["ball_speed_median"] = np.median(sp[ok])
    out["ball_speed_p99"] = np.percentile(sp[ok], 99)
    out["jumps40_per_10k"] = (sp[ok] > 40).sum() / ok.sum() * 1e4
    pr = g & ~nan
    Bp = match.B[pr]
    out["oob_share"] = ((Bp[:, 0] < 0) | (Bp[:, 0] > L) | (Bp[:, 1] < 0) | (Bp[:, 1] > W)).mean()
    d = _nearest_player_m(match.P[pr], Bp)
    out["ball_within_0.25m"] = (d < 0.25).mean()
    out["ball_nearest_median_m"] = np.median(d)
    out["bs_present"] = np.isfinite(match.BS[g]).mean()
    return {k: float(v) for k, v in out.items()}


def _auc(x, y):
    """P(score of a positive > score of a negative), ties count half."""
    from scipy.stats import rankdata
    if not len(x) or not len(y):
        return 0.5
    r = rankdata(np.r_[x, y])
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def leak_features(match):
    """Per-frame single features a model could use to read D off a leaky ball channel."""
    sp = _ball_speed(match.B, match.half)
    frozen = np.r_[False, np.all(np.diff(match.B, axis=0) == 0, axis=1)]
    return {"ball_nan": np.isnan(match.B[:, 0]).astype(float),
            "ball_frozen": np.where(np.isnan(match.B[:, 0]), np.nan, frozen.astype(float)),
            "ball_speed": sp,
            "nearest_player_m": _nearest_player_m(match.P, match.B),
            "bs_present": np.isfinite(match.BS).astype(float),
            "players_tracked": (~np.isnan(match.P[:, :, 0])).sum(1).astype(float),
            # reference, not a leak: players genuinely slow down in dead time
            "player_speed_mean": _nanmean_rows(match.V)}


def leak_stats(match, ref, runs, n_null=40, seed=0):
    """Single-feature D-vs-alive AUCs and their null spread under circular shifts of the D mask.

    For each feature: AUC of D vs alive frames (finite values only), and the 95th percentile of
    |AUC - 0.5| when the D mask is rotated by a random offset within each run (same blocks, random
    place, so ball/label autocorrelation is kept). A feature leaks if its |AUC - 0.5| clears that.
    Also the D-end teleport: share of first alive frames after D with speed > 40 m/s vs all alive.
    """
    rng = np.random.default_rng(seed)
    dead = np.asarray(ref) == "D"
    scored = np.zeros(len(dead), bool)
    for a, b in runs:
        scored[a:b] = True
    feats = leak_features(match)
    shifts = []
    for _ in range(n_null):
        m = np.zeros(len(dead), bool)
        for a, b in runs:
            m[a:b] = np.roll(dead[a:b], int(rng.integers((b - a) // 10, (b - a) * 9 // 10)))
        shifts.append(m)
    out = {}
    for name, f in feats.items():
        ok = scored & np.isfinite(f)
        auc = _auc(f[ok & dead], f[ok & ~dead])
        null = [abs(_auc(f[ok & m], f[ok & ~m]) - 0.5) for m in shifts]
        out[name] = dict(auc=auc, null95=float(np.percentile(null, 95)) if null else float("nan"),
                         d_mean=float(np.nanmean(f[ok & dead])),
                         alive_mean=float(np.nanmean(f[ok & ~dead])))
    sp = feats["ball_speed"]
    ends = np.flatnonzero(scored[1:] & dead[:-1] & ~dead[1:]) + 1
    alive = scored & ~dead & np.isfinite(sp)
    e = ends[np.isfinite(sp[ends])]
    out["teleport"] = dict(d_end_fast=float((sp[e] > 40).mean()) if len(e) else 0.0,
                           alive_fast=float((sp[alive] > 40).mean()), n_ends=int(len(e)))
    return out
