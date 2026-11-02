"""Segment decoders over H/A/D evidence (E09): sticky-HMM Viterbi, explicit-duration HSMM, match wrapper.

The primitives take ``E`` of shape ``(n, 3)`` -- one row per bin (5 Hz by default, ``HZ`` bins per
second) with the log-likelihood of each state -- and return one state code per bin
(0 H, 1 A, 2 D):

* :func:`viterbi` -- first-order Viterbi with constant switch costs. ``cost[p, s]`` is subtracted when
  the path moves from ``p`` to ``s``, so a large cost is a sticky prior and a geometric duration
  distribution. This is the S1 arm of E09.
* :func:`hsmm` -- segmental (explicit-duration) Viterbi. Every segment pays a discretised Gamma
  duration log-probability (``beta`` weight), a per-segment constant ``kappa`` and a log transition
  ``lt[p, s]`` between segment states. This is the P1 arm of E09.
* :func:`decode_frames` -- the match-level wrapper: per contiguous graded span, bin the per-frame
  evidence, decode, optionally refine boundaries at 25 fps, strip a detected dead block (the
  published 2H pre-kickoff block, S09's ``structure.forced_dead_mask``) and write D there, and write
  X on ungraded frames. Deterministic; one full match (159k graded frames, ``dmax`` 300 bins) decodes
  in about 3-4 s and 0.1 GB peak RSS.

Nothing here is on the shipped path: ``make_submission`` does not import this module and the default
CSV does not change. E09 found that the explicit-duration decoder does not replace the shipped stack
on dev-v1 (research/E09_result.md): HSMM -0.0090 [-0.0163, -0.0024] vs the shipped stack, 9/10 games
negative. A learned-emission sticky HMM was +0.0084 [-0.0025, +0.0180] (2/10 negative) and is a
tier-2 lever, not a replacement. The module is kept for that lever and for later duration work, in
the same spirit as :mod:`possession.dev_v1` (research tooling).

Both decoders are pure numpy and deterministic. The reference used while tuning (scratch/E09) had a
ctypes twin of the same recurrences; E09 checked the two agree on synthetic sequences and on a full
held-out game (tests/test_hsmm.py covers the properties that matter: spell count, the published
forced block, 25 Hz boundary shift, runtime, determinism).
"""
import numpy as np
from scipy.stats import gamma as _gamma

FPS = 25                                                 # video frames per second
HZ = 5                                                   # bins per second
STRIDE = FPS // HZ                                       # frames per bin
S = 3                                                    # H, A, D
DMAX = 300                                               # 60 s at HZ; R04 section 3.2
_NEG = -1e300
_LD_FLOOR = -1e12                                        # stands in for log 0 so beta * ld stays finite


def switch_cost(lam_ha, lam_d):
    """(3, 3) switch-cost matrix: H<->A costs ``lam_ha``; any move to/from D costs ``lam_d``."""
    c = np.zeros((S, S))
    c[0, 1] = c[1, 0] = lam_ha
    c[0, 2] = c[1, 2] = c[2, 0] = c[2, 1] = lam_d
    return c


def costs_to_logT(cost):
    """Row-normalized log transition matrix with off-diagonals ``exp(-cost)`` and the rest on the diagonal.

    Use with :func:`viterbi` (which subtracts costs) or with a sampler of the same chain.
    """
    T = np.exp(-np.asarray(cost, float))
    np.fill_diagonal(T, 0.0)
    T[np.diag_indices(S)] = np.maximum(1.0 - T.sum(1), 1e-6)
    with np.errstate(divide="ignore"):
        return np.log(T / T.sum(1, keepdims=True))


def viterbi(E, cost):
    """Best state path for per-bin log-likelihoods ``E`` (n, S) and switch costs ``cost`` (S, S)."""
    E = np.asarray(E, float)
    cost = np.asarray(cost, float)
    n = len(E)
    if n == 0:
        return np.zeros(0, np.int8)
    acc = E[0].copy()
    bp = np.zeros((n, S), np.int8)
    for t in range(1, n):
        sc = acc[:, None] - cost                                # sc[p, s]: come from p into s
        bp[t] = sc.argmax(0)                                    # first max: lower state index on ties
        acc = sc.max(0) + E[t]
    st = int(acc.argmax())
    out = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = st
        st = int(bp[t, st])
    return out


def logdur_gamma(params, dmax=DMAX):
    """(S, dmax+1) log P(a segment lasts exactly d bins | state) from Gamma(k, theta) in seconds.

    ``params`` is ``[(k, theta)] * S``; the probability of ``d`` bins is the Gamma mass on
    ``((d-1)/HZ, d/HZ]``. Entry 0 is ``-inf`` (a segment lasts at least one bin). ``dmax`` caps the
    duration support; E09's dev decodes used 600 bins (120 s), the wrapper's default is 300 (60 s).
    """
    edges = np.arange(0, dmax + 1) / HZ
    out = np.full((S, dmax + 1), -np.inf)
    for s, (k, th) in enumerate(params):
        lo = _gamma.cdf(edges[1:] - 1.0 / HZ, k, scale=th)
        hi = _gamma.cdf(edges[1:], k, scale=th)
        out[s, 1:] = np.log(np.maximum(hi - lo, 1e-300))
    return out


def hsmm(E, ld, lt, beta, kappa, dmax=DMAX, cont_penalty=None):
    """Segmental Viterbi over ``E`` (n, S) with duration log-probabilities ``ld`` (S, dmax+1).

    A segment of state ``s`` and length ``d`` contributes ``sum(E[t0:t, s]) + beta * ld[s, d] + kappa``;
    consecutive segments pay ``lt[p, s]``. The first segment enters with a flat ``-log S`` prior.
    Returns one state code per bin.

    ``cont_penalty`` adds R04's continuation tail: if not None, a segment may be followed by another
    segment of the *same* state at this flat log-probability. The output labels are unchanged (adjacent
    same-state chunks are one maximal run), but a state can then last longer than ``dmax`` bins -- needed
    for long dead blocks and long possessions. ``None`` (the E09 dev setting) forbids self-transitions.
    """
    E = np.asarray(E, float)
    n = len(E)
    if n == 0:
        return np.zeros(0, np.int8)
    ld = np.asarray(ld, float)[:, :dmax + 1]
    ld = np.where(np.isfinite(ld), ld, _LD_FLOOR)             # keeps beta = 0 finite
    lt = np.asarray(lt, float)

    C = np.vstack([np.zeros((1, S)), np.cumsum(E, 0)])        # C[t, s] = sum(E[:t, s])
    delta = np.full((n + 1, S), _NEG)
    bprev = np.full((n + 1, S), _NEG)                         # best path value entering s at boundary t
    bps = np.zeros((n + 1, S), np.int8)
    bd = np.ones((n + 1, S), np.int32)
    bprev[0] = -np.log(S)

    for t in range(1, n + 1):
        nd = min(dmax, t)
        t0 = np.arange(t - nd, t)                             # durations nd, nd-1, ..., 1
        d = t - t0
        for s in range(S):
            v = bprev[t0, s] + (C[t, s] - C[t0, s]) + beta * ld[s, d] + kappa
            k = nd - 1 - int(np.argmax(v[::-1]))              # first max scanning d = 1, 2, ...
            delta[t, s] = v[k]
            bd[t, s] = d[k]
        for s in range(S):
            v = delta[t] + lt[:, s]
            if cont_penalty is None:
                v[s] = _NEG
            else:
                v[s] = delta[t, s] + cont_penalty
            p = int(v.argmax())
            bprev[t, s] = v[p]
            bps[t, s] = p
    out = np.empty(n, np.int8)
    t, s = n, int(delta[n].argmax())
    while t > 0:
        d = int(bd[t, s])
        t0 = t - d
        out[t0:t] = s
        s = int(bps[t0, s])
        t = t0
    return out


def _runs(mask):
    """Half-open (start, stop) runs of True in a 1-D bool array."""
    d = np.diff(np.r_[0, np.asarray(mask, np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _bin(E, stride):
    """Sum per-frame log-likelihoods into bins; zero-pad the last bin (neutral in log space)."""
    n = len(E)
    nb = -(-n // stride)
    pad = nb * stride - n
    X = np.vstack([E, np.zeros((pad, S))]) if pad else E
    return X.reshape(nb, stride, S).sum(1)


def _refine(E, path, stride, frames):
    """Move each bin boundary inside +-``frames`` to maximize the per-frame evidence change.

    ``E`` is the per-frame log-likelihood of one graded span, ``path`` the decoded bin labels.
    Only the emission term is used (R04 section 3.3); the duration term is held at its bin-level
    optimum. At most ``stride`` frames of movement (so a boundary cannot jump its neighbor) and
    never across the span ends; ties stay at the bin boundary. Returns one label per frame.
    """
    n = len(E)
    nb = len(path)
    frames = int(min(max(0, frames), stride))
    if nb < 2 or frames == 0:
        return np.repeat(path, stride)[:n]
    C = np.vstack([np.zeros((1, S)), np.cumsum(E, 0)])
    bounds = [0]
    prev = 0
    for k in range(nb - 1):
        f0 = (k + 1) * stride
        nxt = (k + 2) * stride if k + 2 < nb else n
        lo = max(prev + 1, f0 - frames)
        hi = min(nxt - 1, f0 + frames, n - 1)
        lo, hi = min(lo, f0), max(hi, f0)
        a, b = int(path[k]), int(path[k + 1])
        best, best_gain = f0, 0.0
        for f in range(lo, hi + 1):
            if f == f0:
                continue
            if f > f0:
                gain = (C[f, a] - C[f0, a]) - (C[f, b] - C[f0, b])
            else:
                gain = (C[f0, b] - C[f, b]) - (C[f0, a] - C[f, a])
            if gain > best_gain + 1e-12:
                best, best_gain = f, gain
        bounds.append(best)
        prev = best
    bounds.append(n)
    out = np.empty(n, np.int8)
    for i, s in enumerate(path):
        out[bounds[i]:bounds[i + 1]] = s
    return out


def decode_frames(E, half, ld, lt, beta, kappa, *, dmax=DMAX, forced_dead=None,
                  stride=STRIDE, refine_frames=0, cont_penalty=None):
    """Decode per-frame H/A/D evidence into per-frame organizer codes.

    Parameters
    ----------
    E : (N, 3) float
        Per-frame log-likelihood of H, A, D, already temperature-scaled by the caller. Frames may be
        summed into bins internally; a per-bin evidence vector can be supplied by repeating it over
        its frames (the scaling is then absorbed by the caller's temperature).
    half : (N,) int
        1 or 2 on graded frames of that half, 0 elsewhere (``Match.half``). Each contiguous graded
        run is decoded as its own sequence: no path crosses an ungraded hole.
    ld, lt, beta, kappa, dmax
        As in :func:`hsmm`. ``ld`` must have at least ``dmax + 1`` columns (from
        :func:`logdur_gamma`).
    forced_dead : (N,) bool, optional
        Graded frames that are structurally dead, e.g. ``structure.structure(match).forced_dead_mask``
        (the published 2H pre-kickoff block, D-007). These frames are written D and stripped before
        decoding, so a block longer than ``dmax`` is handled without a special duration tail.
    stride : int
        Frames per bin (5 at 25 fps -> 5 Hz).
    refine_frames : int
        If > 0, refine every boundary on the per-frame ``E`` within +- this many frames (capped at
        ``stride``). E09's dev decodes used 0 (5-frame quantization); R04 calls this polish.
    cont_penalty : float, optional
        R04 continuation tail passed to :func:`hsmm`. ``None`` matches E09's dev decodes; a flat
        penalty (e.g. -5.0) lets a state last beyond ``dmax`` bins, which a long half needs because a
        single segment cannot cover it.

    Returns
    -------
    (N,) int8 codes: 0 H, 1 A, 2 D, 3 X. Ungraded frames are X; ``forced_dead`` frames are D.
    """
    E = np.asarray(E, float)
    half = np.asarray(half)
    n = len(E)
    if E.ndim != 2 or E.shape[1] != S:
        raise ValueError("E must be (N, %d)" % S)
    if len(half) != n:
        raise ValueError("half has %d frames, E has %d" % (len(half), n))
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if dmax < 1 or np.shape(ld)[1] < dmax + 1:
        raise ValueError("ld needs at least dmax + 1 columns")
    if forced_dead is not None and len(forced_dead) != n:
        raise ValueError("forced_dead has %d frames, E has %d" % (len(forced_dead), n))

    codes = np.full(n, 3, np.int8)                             # X outside the graded runs
    dead = np.zeros(n, bool) if forced_dead is None else (np.asarray(forced_dead, bool) & (half > 0))
    if dead.any():
        codes[dead] = 2                                        # structural D, never decoded
    for h in np.unique(half):
        if h <= 0:
            continue
        for lo, hi in _runs(half == h):
            for a, b in _runs(~dead[lo:hi]):                   # split around forced blocks
                a, b = a + lo, b + lo
                if b <= a:
                    continue
                Eb = _bin(E[a:b], stride)
                path = hsmm(Eb, ld, lt, beta, kappa, dmax, cont_penalty=cont_penalty)
                if refine_frames:
                    codes[a:b] = _refine(E[a:b], path, stride, refine_frames)
                else:
                    codes[a:b] = np.repeat(path, stride)[:b - a]
    return codes
