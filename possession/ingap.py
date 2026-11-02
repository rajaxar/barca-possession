"""In-gap ball-free decoder (E46 gapdec, shipped as E53), run on top of the D-gated labels.

Baseline+ persists the pre-gap owner through every ball-missing gap, so a hand-over that happens
inside a gap is invisible and one that is seen is placed at the gap end. Inside each ball-missing run
of at least ``MIN_GAP`` frames this module re-decodes the frames labeled H/A with a 2-state HMM over
the ball-free P(home has the ball) of :mod:`possession.ballfree` (5 Hz): emissions are log P(team),
the entry state is pinned to the label before the gap and the exit state to the label after it, and
a switch costs ``LAM`` nats. ``LAM = ln(mean H/A spell in 5 Hz steps)`` comes from the R08 Gamma prior
(k=0.743, theta=13.06 s), not from dev. Decoded frames replace the gap's H/A labels; D and X frames
are never touched, and Baseline+'s spell merge is re-run on each half.

Safety (E53 / Q17): applied to raw Baseline+ on BAR-ATH it changes 12.6% of graded frames, 91% of
them inside E25 dead gaps. It is only safe **after the missingness D gate**, which has already turned
those dead gaps into D so they contain no H/A frame to re-decode. :func:`footprint` and
:func:`check_footprint` are the label-free guard (no public GT): the build must fail if the gated
stack changes more than ``MAX_CHANGED_SHARE`` of the graded frames.

The D gate itself is not part of this module; ``labels`` must already be D-gated. Default on (D08):
:func:`apply` returns its input untouched unless ``enabled`` (module flag ``ENABLED``) is true.
"""
import numpy as np

from . import ballfree
from . import baseline_plus as bp

ENABLED = True                                       # D08: ships on, chained after the D gate only
LAM = float(np.log(0.743 * 13.06 * ballfree.HZ))   # R08 prior, 3.88 nats per switch
MIN_GAP = 50                                         # frames (2 s), E25's ball-missing run convention
PH_CLIP = 0.02
MAX_CHANGED_SHARE = 0.03                             # Q17 footprint bound (measured 0.0116 gated)
MAX_PRESENT_SHARE = 0.05                             # D08: changes on ball-present frames, share of changes
MAX_SPELL_RATIO = 1.36                               # D08: spells after / before
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
NAME = np.array(["H", "A", "D", "X"])


def _runs(mask):
    x = np.asarray(mask, np.int8)
    return np.flatnonzero(np.diff(np.r_[0, x, 0])).reshape(-1, 2)


def decode_gap(ph5, entry, exit_, lam):
    """2-state Viterbi over 5 Hz P(H) samples; states 0=H, 1=A; entry/exit pinned when >= 0."""
    p = np.clip(ph5, PH_CLIP, 1 - PH_CLIP)
    em = np.stack([np.log(p), np.log(1 - p)], 1)
    n = len(p)
    score = em[0].copy()
    if entry >= 0:
        score[1 - entry] -= lam
    back = np.zeros((n, 2), np.int8)
    for t in range(1, n):
        stay, sw = score, score[::-1] - lam
        back[t] = np.where(stay >= sw, [0, 1], [1, 0])
        score = np.maximum(stay, sw) + em[t]
    if exit_ >= 0:
        score[1 - exit_] -= lam
    s = int(np.argmax(score))
    path = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    return path


def gapdec(labels, B, half, ph, lam=LAM, min_gap=MIN_GAP, m=bp.DEFAULT_PARAMS["m"]):
    """Re-decode H/A frames inside ball-missing runs >= ``min_gap``; returns a new label array.

    ``labels``: (N,) H/A/D/X strings, already D-gated. ``B``: (N, 2) ball (NaN = missing).
    ``half``: (N,) 0/1/2. ``ph``: (N,) per-frame ball-free P(H) (``ballfree.predict_frames``).
    """
    c = np.array([CODE[x] for x in labels], np.int8)
    out = c.copy()
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not len(rows):
            continue
        a0, b0 = rows[0], rows[-1] + 1
        for a, b in _runs(np.isnan(B[a0:b0, 0])):
            if b - a < min_gap:
                continue
            a, b = a + a0, b + a0
            live = np.flatnonzero(c[a:b] < 2) + a
            if len(live) < 2 * ballfree.STRIDE:
                continue
            entry = int(c[a - 1]) if a > a0 and c[a - 1] < 2 else -1
            exit_ = int(c[b]) if b < b0 and c[b] < 2 else -1
            q = ph[live[::ballfree.STRIDE]]
            if not np.isfinite(q).all():
                continue
            path = decode_gap(q, entry, exit_, lam)
            out[live] = path[np.minimum((live - live[0]) // ballfree.STRIDE, len(path) - 1)]
        out[a0:b0] = bp._merge_short(out[a0:b0], m)
    return NAME[out]


def footprint(before, after, B, half, dead=None):
    """Label-free change footprint of ``after`` vs ``before`` over graded frames (half > 0).

    ``dead``: optional (N,) bool mask of E25 dead gaps (ball-missing run >= 2 s, mean P(D) > .5).
    Returns changed-frame share of graded, share of changes inside ``dead`` and on ball-present
    frames, and the spell-count delta.
    """
    before, after = np.asarray(before), np.asarray(after)
    graded = np.asarray(half) > 0
    chg = (before != after) & graded
    n = int(chg.sum())
    present = np.isfinite(np.asarray(B)[:, 0])
    return dict(
        graded=int(graded.sum()), changed=n, changed_share_graded=n / max(1, int(graded.sum())),
        changed_in_dead=int((chg & dead).sum()) if dead is not None else None,
        changed_on_ball_present_share=float((chg & present).sum() / max(1, n)),
        spells_before=_n_spells(before, half), spells_after=_n_spells(after, half))


def _n_spells(labels, half):
    n = 0
    for h in (1, 2):
        rows = np.flatnonzero(np.asarray(half) == h)
        if len(rows):
            lab = np.asarray(labels)[rows[0]:rows[-1] + 1]
            ha = np.isin(lab, ("H", "A"))
            n += int(ha[0]) + int((ha[1:] & (lab[1:] != lab[:-1])).sum())
    return n


def check_footprint(fp, max_share=MAX_CHANGED_SHARE):
    """Raise if the changed-frame share exceeds the bound or any change lands in an E25 dead gap."""
    if fp["changed_share_graded"] > max_share:
        raise RuntimeError("ingap footprint %.4f of graded exceeds %.4f; is the labels input D-gated?"
                           % (fp["changed_share_graded"], max_share))
    if fp["changed_in_dead"]:
        raise RuntimeError("ingap changed %d frames inside E25 dead gaps" % fp["changed_in_dead"])
    if fp["changed_on_ball_present_share"] > MAX_PRESENT_SHARE:
        raise RuntimeError("ingap changed %.3f of its changes on ball-present frames (> %.2f)"
                           % (fp["changed_on_ball_present_share"], MAX_PRESENT_SHARE))
    if fp["spells_after"] > MAX_SPELL_RATIO * max(1, fp["spells_before"]):
        raise RuntimeError("ingap spells %d -> %d exceed x%.2f" % (fp["spells_before"], fp["spells_after"], MAX_SPELL_RATIO))
    return fp


def apply(labels, B, half, ph, dead=None, enabled=None, lam=LAM, min_gap=MIN_GAP, check=True):
    """Flag-gated entry point: returns ``labels`` unchanged when disabled; else gapdec + footprint guard.

    ``check=False`` skips the BAR-ATH-calibrated build guard, keeping the label path identical
    (dev harness only).
    """
    if not (ENABLED if enabled is None else enabled):
        return labels
    out = gapdec(labels, B, half, ph, lam=lam, min_gap=min_gap)
    if check:
        check_footprint(footprint(labels, out, B, half, dead))
    return out
