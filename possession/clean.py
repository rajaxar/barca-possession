"""Generic, label-free tracking cleaning (E16): duplicate tracks, tracking jumps, ball teleports.

``clean_match`` takes a :class:`possession.data.Match` and returns ``(cleaned_match, info)``. Only
physics and geometry rules are used, with no reference to any frame number, label or event, and
nothing is interpolated or invented: a rule can only *remove* a sample (set it to NaN), never move
one.

* **Duplicate tracks.** Two slots of the same team that are both tracked and closer than
  ``dup_dist`` for at least ``dup_len`` consecutive frames, *while the team has more than
  ``max_team`` slots tracked*, are one player seen twice. (Without the surplus condition the rule
  fires on huddles during stoppages: on the clean IDSSE data it removed 15-20% of dead-ball frames.) Inside that run
  the slot with the shorter life in the half is dropped (the goalkeeper slot is never dropped).
* **Player jumps.** A sample whose implied speed to the previous or next tracked sample of the same
  slot (gap <= ``gap`` frames) exceeds ``player_vmax`` is dropped: both sides of a teleport, and the
  outlier of a spike.
* **Peripheral surplus.** If a team still has more than ``max_team`` tracked slots in a frame, the
  surplus is taken from the slots closest to (or beyond) a touchline / goal line, but only if they
  are within ``edge_margin`` of it (substitutes, staff and ball-kids are tracked at the line; the
  goalkeeper slot is exempt). A surplus that is not peripheral is left and flagged.
* **Ball jumps.** A ball sample whose implied speed from the last accepted sample exceeds
  ``ball_vmax`` is dropped, unless the new position is confirmed by ``confirm`` consecutive
  mutually-consistent samples, in which case it is a genuine relocation (tracker switched to another
  object / the ball was out of sight) and is kept but flagged.

``info["flags"]`` is a per-frame uint8 bitmask (see ``FLAGS``) that marks where something was
removed or where the team tracked-player count is still anomalous after cleaning, so a consumer can
down-weight those frames. All defaults are physical limits (a player cannot exceed ~12 m/s, a kicked
ball ~40 m/s, two teammates cannot occupy one square metre for a second), not fitted values.
"""
import numpy as np

FPS = 25.0
DEFAULT_PARAMS = dict(
    dup_dist=1.0,       # m: same-team slots closer than this ...
    dup_len=25,         # ... for this many consecutive frames (1 s at 25 fps) are duplicates
    player_vmax=12.0,   # m/s: faster than any sprint (record ~12.4 m/s peak)
    ball_vmax=50.0,     # m/s: above any kicked ball (record ~ 40 m/s) with tracking-noise headroom
    gap=5,              # frames: speed is only judged across gaps this short
    confirm=10,         # samples that must agree before a ball relocation is accepted
    max_team=11,        # tracked players per team above which surplus is peripheral-trimmed / flagged
    edge_margin=2.0,    # m inside the pitch lines: only surplus tracks this close to a line are dropped
    min_team=7,         # ... and below which it is flagged
)
FLAGS = dict(ball_jump=1, ball_relocation=2, duplicate=4, player_jump=8, team_extra=16, team_low=32,
             peripheral=64)


def _runs(mask):
    x = np.asarray(mask, np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, x, 0]) != 0)
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def clean_ball(B, half, fps=FPS, params=None):
    """Return ``(B_clean, dropped, relocated)``; the masks are (N,) bool."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    out = np.array(B, dtype=np.float32, copy=True)
    dropped = np.zeros(len(B), bool)
    relocated = np.zeros(len(B), bool)
    for h in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == h)
        idx = rows[np.isfinite(B[rows, 0])]
        if len(idx) < 2:
            continue
        xy = B[idx].astype(np.float64)
        # Frame distance between consecutive tracked samples, used for the speed limit.
        limit = lambda a, b: p["ball_vmax"] * (idx[b] - idx[a]) / fps  # noqa: E731
        last = 0
        for i in range(1, len(idx)):
            if np.linalg.norm(xy[i] - xy[last]) <= limit(last, i):
                last = i
                continue
            # Jump from the last accepted sample: relocation if the next samples agree with i.
            chain, j = 1, i
            while j + 1 < len(idx) and chain < p["confirm"]:
                if np.linalg.norm(xy[j + 1] - xy[j]) > limit(j, j + 1):
                    break
                chain += 1
                j += 1
            if chain >= p["confirm"]:
                relocated[idx[i]] = True
                last = i
            else:
                dropped[idx[i]] = True
    out[dropped] = np.nan
    return out, dropped, relocated


def clean_player_jumps(P, half, fps=FPS, params=None):
    """Drop samples adjacent to an implied speed above ``player_vmax``. Returns ``(P, dropped)``."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    out = np.array(P, dtype=np.float32, copy=True)
    bad = np.zeros(P.shape[:2], bool)
    finite = np.isfinite(P[..., 0])
    for gap in range(1, p["gap"] + 1):
        # Pairs (t, t+gap) with nothing tracked in between: the nearest-neighbour comparison.
        between = np.ones(len(P) - gap, bool)[:, None] & np.ones(P.shape[1], bool)[None]
        for k in range(1, gap):
            between &= ~finite[k:len(P) - gap + k]
        ok = finite[:-gap] & finite[gap:] & between & (half[:-gap] == half[gap:])[:, None]
        speed = np.linalg.norm(P[gap:] - P[:-gap], axis=2) * fps / gap
        jump = ok & (speed > p["player_vmax"])
        bad[:-gap] |= jump
        bad[gap:] |= jump
    out[bad] = np.nan
    return out, bad


def merge_duplicates(P, T, gk, half, params=None):
    """Drop the shorter-lived of two same-team slots that coincide. Returns ``(P, dropped)``."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    out = np.array(P, dtype=np.float32, copy=True)
    dropped = np.zeros(P.shape[:2], bool)
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not len(rows):
            continue
        for team in (0, 1):
            slots = np.flatnonzero(T == team)
            x = out[rows][:, slots, :]  # (n, s, 2) view of the working copy for this half
            life = np.isfinite(x[..., 0]).sum(axis=0)
            guard = int(gk[h - 1, team])

            def surplus():  # a track can only be a duplicate where the team has more than 11 tracked
                return np.isfinite(x[..., 0]).sum(axis=1) > p["max_team"]

            pairs = []
            for a in range(len(slots)):
                for b in range(a + 1, len(slots)):
                    d = np.linalg.norm(x[:, a] - x[:, b], axis=1)
                    close = np.isfinite(d) & (d < p["dup_dist"]) & surplus()
                    if close.sum() >= p["dup_len"]:
                        pairs.append((int(close.sum()), a, b))
            for _, a, b in sorted(pairs, reverse=True):
                d = np.linalg.norm(x[:, a] - x[:, b], axis=1)  # re-evaluated: earlier drops count
                close = np.isfinite(d) & (d < p["dup_dist"]) & surplus()
                drop, keep = (b, a) if (life[b], -b) <= (life[a], -a) else (a, b)
                if slots[drop] == guard:
                    drop, keep = keep, drop
                if slots[drop] == guard:
                    continue
                for start, stop in _runs(close):
                    if stop - start >= p["dup_len"]:
                        sel = rows[start:stop]
                        dropped[sel, slots[drop]] = True
                        out[sel, slots[drop]] = np.nan
                        x[start:stop, drop] = np.nan
                life[drop] = np.isfinite(x[:, drop, 0]).sum()
    return out, dropped


def trim_peripheral(P, T, gk, half, pitch=(105.3, 68.0), params=None):
    """Drop surplus peripheral tracks beyond ``max_team`` per team and frame. Returns ``(P, dropped)``."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    out = np.array(P, dtype=np.float32, copy=True)
    dropped = np.zeros(P.shape[:2], bool)
    length, width = pitch
    for team in (0, 1):
        slots = np.flatnonzero(T == team)
        x = out[:, slots, :]
        inside = np.minimum.reduce([x[..., 0], length - x[..., 0], x[..., 1], width - x[..., 1]])
        score = np.where(np.isfinite(inside), inside, np.inf)
        for h in (1, 2):
            rows = np.flatnonzero(half == h)
            g = np.flatnonzero(slots == int(gk[h - 1, team]))
            if len(g):
                score[np.ix_(rows, g)] = np.inf  # goalkeeper is never trimmed
        surplus = np.isfinite(x[..., 0]).sum(axis=1) - p["max_team"]
        surplus[half == 0] = 0
        order = np.argsort(score, axis=1)
        for k in range(int(max(surplus.max(), 0))):
            rows = np.flatnonzero(surplus > k)
            pick = order[rows, k]
            ok = score[rows, pick] < p["edge_margin"]
            dropped[rows[ok], slots[pick[ok]]] = True
    out[dropped] = np.nan
    return out, dropped


def clean_match(match, fps=FPS, pitch=(105.3, 68.0), params=None):
    """Return ``(cleaned_match, info)``; see the module docstring.

    ``info`` keys: ``flags`` (N,) uint8 bitmask over ``FLAGS``; ``counts`` dict of removed
    samples/frames; ``params`` the parameters used.
    """
    p = dict(DEFAULT_PARAMS, **(params or {}))
    P0 = np.asarray(match.P)
    half = np.asarray(match.half)
    T = np.asarray(match.T)
    n = len(half)

    # Order: jumps first (a teleporting slot must not look like a duplicate), then duplicates.
    P1, pj = clean_player_jumps(P0, half, fps, p)
    P2, dup = merge_duplicates(P1, T, np.asarray(match.gk), half, p)
    P2, per = trim_peripheral(P2, T, np.asarray(match.gk), half, pitch, p)
    B, bj, reloc = clean_ball(np.asarray(match.B), half, fps, p)

    flags = np.zeros(n, np.uint8)
    flags[bj] |= FLAGS["ball_jump"]
    flags[reloc] |= FLAGS["ball_relocation"]
    flags[dup.any(axis=1)] |= FLAGS["duplicate"]
    flags[pj.any(axis=1)] |= FLAGS["player_jump"]
    flags[per.any(axis=1)] |= FLAGS["peripheral"]
    live = half > 0
    for team in (0, 1):
        count = np.isfinite(P2[:, T == team, 0]).sum(axis=1)
        flags[live & (count > p["max_team"])] |= FLAGS["team_extra"]
        flags[live & (count < p["min_team"])] |= FLAGS["team_low"]

    gone = ~np.isfinite(P2[..., 0]) & np.isfinite(P0[..., 0])
    V = np.where(gone, np.nan, match.V).astype(np.asarray(match.V).dtype)
    cleaned = match._replace(P=P2, V=V, B=B) if hasattr(match, "_replace") else _copy(match, P2, V, B)
    g = live
    info = dict(
        flags=flags, params=p,
        counts=dict(
            player_jump_samples=int(pj.sum()), duplicate_samples=int(dup.sum()), peripheral_samples=int(per.sum()),
            ball_jump_samples=int(bj.sum()), ball_relocations=int(reloc.sum()),
            graded_frames=int(g.sum()),
            frames_flagged=int((flags[g] > 0).sum()),
        ),
    )
    return cleaned, info


def _copy(match, P, V, B):
    from types import SimpleNamespace
    d = dict(vars(match))
    d.update(P=P, V=V, B=B)
    return SimpleNamespace(**d)
