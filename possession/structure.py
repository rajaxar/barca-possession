"""Match structure: half blocks, kickoffs, attack direction and the forced-D mask.

Organiser-published structure is stated here as constants with its source (GUARDRAILS 0, D-007).
Everything else (attack direction) is computed from tracking. `kickoff_formation` and the Smart
Tagging comparison in the tests are consistency checks only; they are not inputs.
"""
from collections import namedtuple

import numpy as np

PITCH_L, PITCH_W = 105.3, 68.0  # games.py GAMES["BAR-ATH"]["pitch"]

# Half boundaries as video frames, inclusive: games.half_video_blocks("BAR-ATH").
HALF_BLOCKS = ((1, 11185, 90651), (2, 110808, 190197))

# Half kickoff frames (D-007). 1H = first graded frame; 2H = FAQ.md:38 and starter/load_atd.py:163.
# Graded 2H frames before the kickoff (110808..113741) are all D (FAQ.md:38).
KICKOFFS = {1: 11185, 2: 113742}

Structure = namedtuple("Structure", "halves kickoffs attack_dir forced_dead_mask")
# halves: {h: (first, last)} inclusive video frames; kickoffs: {h: frame};
# attack_dir: (2 halves, 2 teams) int8, +1 = team attacks toward +x (its GK is at low x), -1 otherwise;
# forced_dead_mask: (N,) bool, graded frames before their half's kickoff.


def attack_direction(P, T, gk):
    """(2, 2) int8 attack direction from goalkeeper position: the GK stands near the goal it defends.

    Cross-checked in tests against the outfield centroid of each team.
    """
    out = np.zeros((2, 2), np.int8)
    for h, a, z in HALF_BLOCKS:
        for t in (0, 1):
            x = np.nanmean(P[a:z + 1, gk[h - 1, t], 0])
            out[h - 1, t] = 1 if x < PITCH_L / 2 else -1
    return out


def centroid_direction(P, T, gk):
    """Independent (2, 2) estimate: a team that sits deeper (lower mean outfield x) than its opponent
    over the half defends the low-x goal, so it attacks toward +x."""
    out = np.zeros((2, 2), np.int8)
    for h, a, z in HALF_BLOCKS:
        mx = []
        for t in (0, 1):
            idx = [i for i in np.where(T == t)[0] if i != gk[h - 1, t]]
            mx.append(np.nanmean(P[a:z + 1][:, idx, 0]))
        out[h - 1] = (1, -1) if mx[0] < mx[1] else (-1, 1)
    return out


def structure(match):
    """Structure of the match from a possession.data.Match."""
    n = len(match.B)
    dead = np.zeros(n, bool)
    for h, a, _ in HALF_BLOCKS:
        dead[a:KICKOFFS[h]] = True
    return Structure(halves={h: (a, z) for h, a, z in HALF_BLOCKS}, kickoffs=dict(KICKOFFS),
                     attack_dir=attack_direction(match.P, match.T, match.gk), forced_dead_mask=dead)


def kickoff_formation(match, frame, window=25):
    """Consistency check of a kickoff frame, measured over frames..frame+window.

    Returns dict: own_half_frac (share of tracked outfield players on their own side of the halfway
    line, given the attack direction), ball_to_centre_m (median ball distance to the centre spot, NaN
    if no ball), n_tracked (median tracked players).
    """
    s = structure(match)
    h = int(match.half[frame])
    sl = slice(frame, frame + window)
    P = match.P[sl]
    own, tot = 0, 0
    for t in (0, 1):
        idx = [i for i in np.where(match.T == t)[0] if i != match.gk[h - 1, t]]
        x = P[:, idx, 0]
        side = s.attack_dir[h - 1, t]
        own += np.nansum((x - PITCH_L / 2) * side <= 1.0)  # 1 m slack for the centre circle line
        tot += np.sum(~np.isnan(x))
    ball = match.B[sl]
    d = np.linalg.norm(ball - np.array([PITCH_L / 2, PITCH_W / 2]), axis=1)
    return dict(own_half_frac=float(own / tot),
                ball_to_centre_m=float(np.nanmedian(d)) if np.any(~np.isnan(d)) else float("nan"),
                n_tracked=float(np.median(np.sum(~np.isnan(P[..., 0]), axis=1))))

