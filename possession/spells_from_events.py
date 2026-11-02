"""Event-first possession (E03): turn ball-touch events into H/A possession spells.

The atomic event is a detected touch (:mod:`possession.touches`).  This module converts the event
sequence into frame labels with one of four rules, so the "last contact vs controlled possession"
question can be decided on external dev games instead of on intuition:

``flip``      every opposing touch flips the owner (pure last contact);
``two_touch`` a challenger takes over only after a second touch or a control run of ``hold_s``
              seconds (a "controlled possession" rule);
``revert``    flip on the touch, but treat it as a deflection and keep the old owner if the old team
              touches again within ``revert_s`` seconds;
``decoder``   soft touch evidence (team, confidence) fed to a two-state Viterbi decoder with a switch
              penalty; the interface E09 can replace with a calibrated duration model.

Two further functions do not rebuild the owner sequence from events at all: :func:`anchor` only
re-times an existing base hand-over to a nearby touch of the new team, and :func:`insert` only adds a
hand-over where the base has none (the operational form of "a single opposing touch counts").

Two literature rules are available as pre/post-processing: a *deflection merge* (a second touch by a
different team within ``merge_s`` seconds is the same play; R03/Bischofberger) and a *duel state*
(both teams within ``duel_r`` meters of the ball -> do not flip on that touch; R03/Vidal-Codina).

D and X frames are never written here: the caller supplies the D detector's labels in ``base`` and
only H/A spans are rebuilt.  Owner state is re-initialized after every D/X run from the first touch
inside the new live span (falling back to the nearest player, then to ``base``), which is the
restart-team rule of the public GT semantics (02_gt_semantics §4).

The module is label-free and reads no match data.  The dev selection driver lives in
``scratch/E03``; :mod:`possession.dev_v1` supplies the harness.
"""
import numpy as np

from . import touches

FPS = 25
RULES = ("flip", "two_touch", "revert", "decoder")
DEFAULT_PARAMS = dict(
    touch=dict(mode="proximity"),   # E02: the ATD ball is kinematics-blind; soft proximity events
    r=1.5,                          # nearest-player radius for the hold / init tests (m)
    duel_r=1.0,                     # R03/Vidal-Codina duel radius (m)
    merge_s=0.5,                    # R03 deflection merge window (s); 0 disables
    hold_s=0.8,                     # control-run length that counts as "controlled" (s)
    revert_s=0.5,                   # deflection revert window (s)
    alpha=6.0,                      # decoder: log-odds per unit touch confidence
    lam=4.0,                        # decoder: nats paid for one owner switch
    beta=0.0,                       # decoder: prior log-odds per frame from the caller's base label
    sigma=3.0,                      # decoder: touch evidence spread (frames, Gaussian sd)
    anchor_w=12,                    # anchor: +-frames searched for a touch of the new team
    anchor_conf=0.0,                # anchor: ignore touches below this confidence
    anchor_only="nearest",          # anchor: "nearest" | "later" (never move a hand-over earlier)
    init_wait_s=1.0,                # first touch this close to a live-span start sets the owner
    init_mode="auto",               # "base": keep the caller's label at the span start (oracle/ablation)
    max_wait_s=2.0,                 # two_touch: how long a challenger may take to qualify
    min_conf=0.0,                   # drop touches below this confidence
    insert_w=12,                    # insert: +-frames around a base hand-over that count as "already there"
    insert_min_s=0.0,               # insert: drop an added hand-over whose new spell would be shorter
)


# --------------------------------------------------------------------------- touch events

def event_table(P, T, B, half, params=None, mode=None):
    """Touch events (DataFrame from :func:`possession.touches.detect_touches`).

    ``mode="proximity"`` is the default because the ATD ball carries no kinematic touch signature
    (E02); IDSSE is run the same way so dev scores reflect what our match can deliver.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    tp = dict(p["touch"])
    if mode is not None:
        tp["mode"] = mode
    tp.setdefault("mode", "proximity")
    df = touches.detect_touches(P, T, B, half, params=tp)
    if len(df) and p["min_conf"] > 0:
        df = df[df.conf >= p["min_conf"]].reset_index(drop=True)
    return df


def merge_deflections(df, merge_s=0.5):
    """R03/Bischofberger: an opposing touch within ``merge_s`` of the previous kept touch is a
    deflection off that player, not a new possession.  Same-team touches are kept (they are repeat
    contacts of the same play)."""
    if len(df) == 0 or merge_s <= 0:
        return df
    keep = np.ones(len(df), bool)
    last_team, last_frame = int(df.team.iloc[0]), int(df.frame.iloc[0])
    for i in range(1, len(df)):
        f, team = int(df.frame.iloc[i]), int(df.team.iloc[i])
        if team != last_team and (f - last_frame) <= merge_s * FPS:
            keep[i] = False
            continue
        last_team, last_frame = team, f
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- geometry helpers

def nearest_series(P, T, B, r=1.5, duel_r=1.0):
    """(nd, nt, duel) at 25 Hz.

    ``nd`` nearest-player distance, ``nt`` its team (-1 if farther than ``r``), ``duel`` True when at
    least one player of each team is within ``duel_r`` meters of the ball (R03's contest state).
    """
    n = len(B)
    nd = np.full(n, np.inf)
    nt = np.full(n, -1, np.int8)
    duel = np.zeros(n, bool)
    ok = ~np.isnan(B[:, 0])
    idx = np.flatnonzero(ok)
    if not len(idx):
        return nd, nt, duel
    d = np.linalg.norm(P[idx] - B[idx][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    nd[idx] = d[np.arange(len(idx)), j]
    nt[idx] = np.where(nd[idx] <= r, T[j], -1).astype(np.int8)
    near = d <= duel_r
    both = np.ones(len(idx), bool)
    for team in (0, 1):
        both &= near[:, T == team].any(axis=1)
    duel[idx] = both
    duel &= ok
    return nd, nt, duel


def _runs(mask):
    x = np.asarray(mask, np.int8)
    e = np.flatnonzero(np.diff(np.r_[0, x, 0]) != 0).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in e]


def _run_len(nt, team, nd, r):
    """Length (frames) of the contiguous run where ``nt == team`` and ``nd <= r`` ending at each frame."""
    ok = (nt == team) & (nd <= r)
    out = np.zeros(len(ok), np.int32)
    c = 0
    for i in range(len(ok)):
        c = c + 1 if ok[i] else 0
        out[i] = c
    return out


# --------------------------------------------------------------------------- spell building

def live_spans(base, half):
    """H/A contiguous spans per half: ``[(a, b, team)]``, half-open."""
    out = []
    x = np.asarray(base)
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        live = np.isin(x[idx], ("H", "A")).astype(np.int8)
        e = np.flatnonzero(np.diff(np.r_[0, live, 0]) != 0).reshape(-1, 2)
        for ra, rb in e:
            a, b = int(idx[ra]), int(idx[rb - 1]) + 1
            out.append((a, b, 0 if x[a] == "H" else 1))
    return out


def _repaint(base, flips, half):
    """Copy ``base`` and re-paint every H/A span with the given (frame, team) flips (boundaries kept)."""
    out = np.array([str(c) for c in base], dtype="<U1")
    for a, b, team in live_spans(base, half):
        keep = sorted((int(f), int(t)) for f, t in flips if a < int(f) < b)
        cur, pos = team, a
        for f, t in keep:
            out[pos:f] = "H" if cur == 0 else "A"
            cur, pos = t, f
        out[pos:b] = "H" if cur == 0 else "A"
    return out


def base_flips(base):
    """Frames where a live H/A run changes team (never across D/X)."""
    x = np.asarray(base)
    live = np.isin(x, ("H", "A"))
    f = np.flatnonzero(live[1:] & live[:-1] & (x[1:] != x[:-1])) + 1
    return [(int(i), 0 if x[i] == "H" else 1) for i in f]


def anchor(base, half, events, params=None):
    """Snap each base hand-over to a touch of the *new* team nearby (soft boundary evidence).

    The event decides *when* the owner changed, not *whether*: the base stack's hand-over is kept and
    only its frame moves.  Public GT places hand-overs at the contact (median offset 0 frames,
    IQR -1..0), so this is the one use of touch events that cannot add a spurious spell.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    evs = [(int(r.frame), int(r.team), float(r.conf)) for r in events.itertuples(index=False)]
    evs = [e for e in evs if e[2] >= p["anchor_conf"]]
    move = []
    for f, team in base_flips(base):
        lo = f if p["anchor_only"] == "later" else f - int(p["anchor_w"])
        cand = [e for e in evs if e[1] == team and lo <= e[0] <= f + int(p["anchor_w"])]
        if not cand:
            continue
        g = min(cand, key=lambda e: abs(e[0] - f))[0]
        if int(g) != f:
            move.append((f, int(g)))
    # apply moves then keep the order inside each span
    flips = dict(base_flips(base))
    for f, g in move:
        flips[g] = flips.pop(f)
    fixed = []
    for a, b, _ in live_spans(base, half):
        sub = sorted((f, t) for f, t in flips.items() if a <= f < b)
        prev = a
        for f, t in sub:
            f = max(f, prev + 1)
            if f >= b:
                f = b - 1
            fixed.append((f, t))
            prev = f
    return _repaint(base, fixed, half)


def insert(base, half, events, params=None):
    """Last-contact insertion: add a hand-over at an accepted opposing touch where ``base`` has none.

    This is the operational form of "a single opposing touch counts": unlike :func:`predict` it never
    re-paints an existing spell, so it can only add a short opposing H/A spell, never move or drop a
    base hand-over.  A touch is ignored if the base has a hand-over within ``insert_w`` frames of it
    (that is the same contact, and re-timing is :func:`anchor`'s job) and it is dropped if the spell it
    would start is shorter than ``insert_min_s`` seconds.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    base = np.asarray(base)
    df = events
    if len(df) and p["min_conf"] > 0:
        df = df[df.conf >= p["min_conf"]]
    df = merge_deflections(df.reset_index(drop=True), p["merge_s"])
    flips = base_flips(base)
    have = np.array([f for f, _ in flips], int)
    w = int(p["insert_w"])
    added = []
    for r in df.itertuples(index=False):
        f, team = int(r.frame), int(r.team)
        if not (0 < f < len(base)) or base[f] not in ("H", "A"):
            continue
        if team == (0 if base[f] == "H" else 1):
            continue
        if len(have) and np.min(np.abs(have - f)) <= w:
            continue
        added.append(f)
        flips.append((f, team))
    if p["insert_min_s"] > 0 and added:
        added_set, minf = set(added), int(round(p["insert_min_s"] * FPS))
        kept = []
        for a, b, _ in live_spans(base, half):
            seq = sorted(x for x in flips if a < x[0] < b)
            for i, (f, team) in enumerate(seq):
                end = seq[i + 1][0] if i + 1 < len(seq) else b
                if f in added_set and end - f < minf:
                    continue
                kept.append((f, team))
        flips = kept
    return _repaint(base, sorted(set(flips)), half)


def touch_logodds(n, events, sigma=3.0, alpha=6.0):
    """Soft (Gaussian-in-time) log-odds for H at every frame from the touch events.

    This is the hand-off to a decoder (E09): one event is evidence spread over +-3 sigma frames,
    never a hard flip.  Returns ``lo`` (positive = H) of length ``n``.
    """
    lo = np.zeros(n)
    sig = max(float(sigma), 0.25)
    for r in events.itertuples(index=False):
        f, team, conf = int(r.frame), int(r.team), float(r.conf)
        lo_f = alpha * conf * (1.0 if team == 0 else -1.0)
        a, b = max(0, f - int(4 * sig)), min(n, f + int(4 * sig) + 1)
        if b <= a:
            continue
        t = np.arange(a, b) - f
        lo[a:b] += lo_f * np.exp(-0.5 * (t / sig) ** 2)
    return lo


def _paint(out, a, b, init, flips):
    """Fill ``out[a:b]`` with ``init`` and apply (frame, team) flips in time order."""
    s = init
    pos = a
    for f, team in flips:
        f = min(max(int(f), a), b)
        if f > pos:
            out[pos:f] = "H" if s == 0 else "A"
        s = int(team)
        pos = f
    if pos < b:
        out[pos:b] = "H" if s == 0 else "A"


def _init_owner(base, a, b, evs, nd, nt, p):
    """Owner at the start of a live span: the first touch if it comes quickly, else nearest player,
    else the supplied base label (restart-team rule; 02_gt_semantics §4)."""
    if p.get("init_mode") == "base":
        lab = base[a]
        return 0 if lab == "H" else 1
    w = int(p["init_wait_s"] * FPS)
    for e in evs:
        if e[0] - a <= w:
            return int(e[1])
    if np.isfinite(nd[a]) and nt[a] >= 0:
        return int(nt[a])
    lab = base[a]
    return 0 if lab == "H" else 1


def _rule_flip(evs, init, a, b, nd, nt, p, duel, base=None):
    flips, s = [], init
    for f, team, conf, dd in evs:
        if dd:
            continue
        if team != s:
            flips.append((f, team))
            s = team
    return flips


def _rule_two_touch(evs, init, a, b, nd, nt, p, duel, base=None):
    flips, s = [], init
    H = int(round(p["hold_s"] * FPS))
    wait = int(p["max_wait_s"] * FPS)
    runlen = p.get("_runlen")
    if runlen is None:
        runlen = {0: _run_len(nt, 0, nd, p["r"]), 1: _run_len(nt, 1, nd, p["r"])}
    pend = None  # [team, count, first_frame]
    for f, team, conf, dd in evs:
        if dd:
            continue
        if team == s:
            pend = None
            continue
        if pend is not None and pend[0] == team:
            pend[1] += 1
        else:
            pend = [team, 1, f]
        hold = False
        if pend[1] < 2:
            lo, hi = pend[2], min(b, pend[2] + wait)
            if hi > lo and np.any(runlen[team][lo:hi] >= H):
                hold = True
        if pend[1] >= 2 or hold:
            flips.append((pend[2], team))   # credit at the first challenging touch
            s = team
            pend = None
    return flips


def _rule_revert(evs, init, a, b, nd, nt, p, duel, base=None):
    flips, s = [], init
    R = int(round(p["revert_s"] * FPS))
    for i, (f, team, conf, dd) in enumerate(evs):
        if dd or team == s:
            continue
        reverted = False
        for g, gt, _, _ in evs[i + 1:]:
            if g - f > R:
                break
            if gt == s:
                reverted = True
                break
        if not reverted:
            flips.append((f, team))
            s = team
    return flips


def _rule_decoder(evs, init, a, b, nd, nt, p, duel, base=None):
    """Two-state Viterbi over *soft* touch evidence: each event is a Gaussian burst of log-odds
    (``alpha * conf``, sd ``sigma`` frames), a switch costs ``lam`` nats and the caller's base label
    adds a per-frame prior (``beta``).  ``beta=0`` is the pure event decoder; ``beta>0`` is the
    soft-evidence fusion (variant d) that E09 can replace with the calibrated HSMM."""
    n = b - a
    lo = np.zeros(n)
    sig = max(float(p["sigma"]), 0.25)
    for f, team, conf, dd in evs:
        if dd:
            continue
        val = p["alpha"] * conf * (1.0 if team == 0 else -1.0)
        f0 = int(f) - a
        aa, bb = max(0, f0 - int(4 * sig)), min(n, f0 + int(4 * sig) + 1)
        if bb > aa:
            t = np.arange(aa, bb) - f0
            lo[aa:bb] += val * np.exp(-0.5 * (t / sig) ** 2)
    if p["beta"] and base is not None:
        lab = np.asarray(base)[a:b]
        lo += p["beta"] * np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))
    em = np.stack([lo, -lo], axis=1)
    lam = p["lam"]
    score = np.zeros(2)
    score[1 - init] -= lam
    back = np.zeros((n, 2), np.int8)
    for t in range(1, n):
        stay = score
        sw = score[::-1] - lam
        take = stay >= sw
        back[t] = np.where(take, [0, 1], [1, 0])
        score = np.maximum(stay, sw) + em[t]
    st = int(np.argmax(score))
    path = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = st
        st = int(back[t, st])
    # convert the path to flips (only where the state changes)
    flips, s = [], init
    ch = np.flatnonzero(np.diff(np.r_[init, path]) != 0)
    for k in ch:
        flips.append((a + int(k), int(path[k])))
    return flips


_RULES = {"flip": _rule_flip, "two_touch": _rule_two_touch, "revert": _rule_revert,
          "decoder": _rule_decoder}


def predict(P, T, B, half, base, rule="flip", params=None, events=None, geom=None):
    """Rebuild H/A labels from touch events; D/X frames are copied from ``base``.

    ``base`` is any length-``n`` H/A/D/X label array (the D detector's output).  ``events`` and
    ``geom`` may be passed in to share work across variants on one cell.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    if rule not in _RULES:
        raise ValueError("unknown rule %r" % (rule,))
    base = np.asarray(base)
    out = np.array([str(x) for x in base], dtype="<U1")
    if events is None:
        events = event_table(P, T, B, half, params=p)
    if geom is None:
        geom = nearest_series(P, T, B, r=p["r"], duel_r=p["duel_r"])
    nd, nt, duel = geom
    evs = np.array([(int(r.frame), int(r.team), float(r.conf), bool(duel[int(r.frame)]))
                    for r in events.itertuples()], dtype=object) if len(events) else np.zeros((0, 4), object)
    if p["merge_s"] > 0 and len(evs):
        # R03 deflection merge on the event list (same logic as merge_deflections on the frame)
        keep, lt, lf = [], int(evs[0][1]), int(evs[0][0])
        for i in range(len(evs)):
            f, team = int(evs[i][0]), int(evs[i][1])
            if i == 0 or not (team != lt and (f - lf) <= p["merge_s"] * FPS):
                keep.append(i)
                lt, lf = team, f
        evs = evs[np.array(keep, int)]
    rule_fn = _RULES[rule]
    if rule == "two_touch":
        # one O(n) pass per team, not one per live span
        p["_runlen"] = {0: _run_len(nt, 0, nd, p["r"]), 1: _run_len(nt, 1, nd, p["r"])}
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        live = np.isin(out[idx], ("H", "A"))
        for ra, rb in _runs(live):
            a, b = int(idx[ra]), int(idx[rb - 1]) + 1   # _runs is half-open in idx-space
            sub = evs[(evs[:, 0] >= a) & (evs[:, 0] < b)] if len(evs) else evs
            init = _init_owner(out, a, b, sub, nd, nt, p)
            flips = rule_fn(sub, init, a, b, nd, nt, p, duel[a:b], base=out)
            _paint(out, a, b, init, flips)
    return out
