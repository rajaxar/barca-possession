"""Match data loader: tracking (ATD), Smart Tagging and broadcast alignment, as numpy arrays.

Everything is indexed by *video frame* 0..N_VIDEO_FRAMES-1 (the submission row). The ATD is already
on the video timeline with frame_id = video_frame + 1 (starter/load_atd.py). Coordinates are metres
on the 105.3 x 68 pitch. Parsed arrays are cached under config.cache_dir(), keyed by file hash.
"""
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np

from . import config
from .structure import HALF_BLOCKS

PITCH_L, PITCH_W = 105.3, 68.0  # games.py GAMES["BAR-ATH"]["pitch"]


_CHUNK = 1 << 24
_SLOTS = {"tactical": 46, "broadcast": 51}
_FRAMES = {"tactical": config.N_VIDEO_FRAMES, "broadcast": 164488}  # games.SOURCES n_video_frames

# Tactical player slots: team 0 = home (first <Team> in the metadata XML), 1 = away.
Match = namedtuple("Match", "P V T B BS gk graded half")
# P (N,46,2) float32 metres, NaN = untracked; V (N,46) m/s; T (46,) int8 team per slot;
# B (N,2) ball xy metres, NaN when missing; BS (N,) ball 's' channel (unexplained, kept separate);
# gk (2 halves, 2 teams) int slot of the goalkeeper (rule: max mean |x - centre|, per half);
# graded (N,) bool; half (N,) int8 with 0 outside the halves, 1 / 2 inside.


def _file_hash(*paths):
    """Content hash of the files. Re-reading 300 MB per call is slow, so a hash is remembered
    against (path, size, mtime) in cache_dir()/hashes.json and reused while that stamp is unchanged."""
    memo_path = os.path.join(config.cache_dir(), "hashes.json")
    try:
        with open(memo_path) as f:
            memo = json.load(f)
    except (OSError, ValueError):
        memo = {}
    h = hashlib.blake2b(digest_size=16)
    dirty = False
    for path in paths:
        st = os.stat(path)
        stamp = [st.st_size, st.st_mtime_ns]
        entry = memo.get(path)
        if not entry or entry["stamp"] != stamp:
            fh = hashlib.blake2b(digest_size=16)
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(_CHUNK)
                    if not chunk:
                        break
                    fh.update(chunk)
            entry = memo[path] = {"stamp": stamp, "hash": fh.hexdigest()}
            dirty = True
        h.update(entry["hash"].encode())
    if dirty:
        tmp = config.tmp_path(memo_path)
        with open(tmp, "w") as f:
            json.dump(memo, f)
        os.replace(tmp, memo_path)
    return h.hexdigest()


def _cached(name, paths, build):
    """Dict of arrays from cache_dir()/name-<hash>.npz, built (and stored) on a miss."""
    key = _file_hash(*paths)
    target = os.path.join(config.cache_dir(), "{}-{}.npz".format(name, key))
    if os.path.isfile(target):
        with np.load(target) as z:
            return {k: z[k] for k in z.files}
    arrays = build()
    tmp = config.tmp_path(target)
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    return arrays


def _team_ids(meta_path):
    """(team names, per-slot team index 0/1). First <Team> in the XML is home."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    ids = [t.get("id") for t in teams]
    names = [t.findtext("Name") for t in teams]
    slot_team = np.array([0 if p.get("teamId") == ids[0] else 1 for p in root.iter("Player")], np.int8)
    return names, slot_team


def _parse_atd(source):
    """Raw ATD -> dict(P, V, T, B, BS) on the video timeline (row i = frame_id i + 1)."""
    n, s = _FRAMES[source], _SLOTS[source]
    _, team = _team_ids(config.asset_path("atd_meta", source))
    if len(team) != s:
        raise ValueError("{} metadata lists {} players, expected {}".format(source, len(team), s))
    with open(config.asset_path("atd_raw", source), "rb") as f:
        raw = f.read()
    text = raw.replace(b"\r", b"").replace(b":", b",").replace(b";", b",").replace(b"\n", b",")
    flat = np.fromstring(text, dtype=np.float64, sep=",")
    width = 1 + 3 * s + 3
    if flat.size != n * width:
        raise ValueError("{} ATD has {} values, expected {}".format(source, flat.size, n * width))
    rows = flat.reshape(n, width)
    if not np.array_equal(rows[:, 0], np.arange(1, n + 1)):
        raise ValueError("{} ATD frame ids are not 1..{}".format(source, n))
    rows = rows[:, 1:].astype(np.float32)
    pl = rows[:, :3 * s].reshape(n, s, 3)
    P = np.stack([pl[..., 0] * PITCH_L, pl[..., 1] * PITCH_W], -1)
    B = np.stack([rows[:, -3] * PITCH_L, rows[:, -2] * PITCH_W], -1)
    P = P.astype(np.float32)
    out = dict(P=P, V=np.ascontiguousarray(pl[..., 2]), T=team,
               B=B.astype(np.float32), BS=np.ascontiguousarray(rows[:, -1]))
    if source == "tactical":
        out["gk"] = goalkeeper_slots(P, team, half_array(n))
    return out


def _arrays(source):
    paths = [config.asset_path("atd_meta", source), config.asset_path("atd_raw", source)]
    return _cached("atd-" + source, paths, lambda: _parse_atd(source))


def half_array(n=config.N_VIDEO_FRAMES):
    half = np.zeros(n, np.int8)
    for h, a, b in HALF_BLOCKS:
        half[a:b + 1] = h
    return half


def goalkeeper_slots(P, T, half):
    """(2 halves, 2 teams) slot index with max mean |x - centre| inside that half.

    Only slots tracked in at least a quarter of the half's frames compete: a slot that appears for
    a few seconds near a goal (e.g. a duplicate) must not win.
    """
    out = np.zeros((2, 2), np.int64)
    for h in (1, 2):
        a, z = next((a, z) for hh, a, z in HALF_BLOCKS if hh == h)
        x = np.abs(P[a:z + 1, :, 0] - PITCH_L / 2)
        with np.errstate(all="ignore"):
            m = np.nanmean(x, axis=0)
        m = np.where(np.isnan(m) | ((~np.isnan(x)).mean(0) < 0.25), -1.0, m)
        for t in (0, 1):
            idx = np.where(T == t)[0]
            out[h - 1, t] = idx[np.argmax(m[idx])]
    return out


def load_match(source="tactical"):
    """Tactical tracking as a Match (see top of module for array layout)."""
    if source != "tactical":
        raise ValueError("load_match supports source='tactical'; use load_broadcast for broadcast")
    a = _arrays("tactical")
    half = half_array(len(a["B"]))
    return Match(P=a["P"], V=a["V"], T=a["T"], B=a["B"], BS=a["BS"],
                 gk=a["gk"], graded=half > 0, half=half)


# ---------------------------------------------------------------- Smart Tagging

# Codes whose Team label is the DEFENDING team; possession belongs to the other team.
_DEFENDER_CODES = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def load_tags():
    """Smart Tagging intervals as a DataFrame in video frames (end exclusive, as starter/load_smart_tagging).

    Columns: id, code, start_frame, end_frame, team (0 home / 1 away / None as tagged),
    attacker_team (team in possession per the code: defender codes flipped, N/A -> None),
    type (set-piece type or None), half (1/2).
    """
    import pandas as pd

    path = config.asset_path("smart_tagging")
    names, _ = _team_ids(config.asset_path("atd_meta"))
    to_team = {name: i for i, name in enumerate(names)}
    rows = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lb.findtext("group"): lb.findtext("text") for lb in inst.findall("label")}
        code = inst.findtext("code")
        team = to_team.get(tags.get("Team"))
        attacker = team
        if team is not None and code in _DEFENDER_CODES:
            attacker = 1 - team
        rows.append(dict(
            id=int(inst.findtext("ID")), code=code,
            start_frame=int(round(float(inst.findtext("start")) * 25)),
            end_frame=int(round(float(inst.findtext("end")) * 25)),
            team=team, attacker_team=attacker, type=tags.get("Type"),
            half={"1st Half": 1, "2nd Half": 2}.get(tags.get("Half"))))
    df = pd.DataFrame(rows)
    for c in ("team", "attacker_team"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


# ---------------------------------------------------------------- Broadcast alignment

def _cloud_score(a, c):
    """Median nearest-neighbour distance (m) between two player clouds, NaN rows ignored."""
    a = a[~np.isnan(a[:, 0])]
    c = c[~np.isnan(c[:, 0])]
    if len(a) < 5 or len(c) < 8:
        return np.nan
    return float(np.median(np.linalg.norm(a[:, None] - c[None], axis=2).min(1)))


def _coarse_lag(Pt, Pb, tac_block, bc_seg, n_samples=60, step=25):
    """Lag k (tactical = broadcast + k) to within ~1 s: cloud score at samples of the broadcast stretch.

    For each candidate k (every `step` frames) the score is the median over the samples that land
    inside the tactical half of the nearest-neighbour distance between the two player clouds; the
    right lag scores far below the others.
    """
    samples = np.linspace(bc_seg[0], bc_seg[1], n_samples).astype(int)
    best = (np.inf, 0)
    for k in range(tac_block[0] - bc_seg[1], tac_block[1] - bc_seg[0] + 1, step):
        tf = samples + k
        ok = (tf >= tac_block[0]) & (tf <= tac_block[1])
        if ok.sum() < 10:
            continue
        sc = [_cloud_score(Pb[i], Pt[j]) for i, j in zip(samples[ok], tf[ok])]
        sc = [x for x in sc if x == x]
        if len(sc) < 10:
            continue
        v = float(np.median(sc))
        if v < best[0]:
            best = (v, k)
    return best[1]


def _refine_lag(Pt, Pb, coarse, bc_seg, tac_block, n_windows=100, radius=60, max_score=0.6):
    """Median frame-exact lag over player-cloud matches at windows spread across the segment.

    Returns (lag, n_matched, n_windows). A window matches when its best 3-frame mean
    nearest-neighbour distance is below `max_score` metres.
    """
    lo, hi = max(bc_seg[0], tac_block[0] - coarse), min(bc_seg[1], tac_block[1] - coarse)
    centres = np.linspace(lo + 500, hi - 500, n_windows).astype(int)
    ks = np.arange(coarse - radius, coarse + radius + 1)
    found = []
    for bf in centres:
        sc = np.full(len(ks), np.nan)
        for j, k in enumerate(ks):
            v = [_cloud_score(Pb[bf + o], Pt[bf + o + k]) for o in (0, 6, 12)
                 if 0 <= bf + o + k < len(Pt)]
            v = [x for x in v if x == x]
            if len(v) >= 2:
                sc[j] = np.mean(v)
        if np.all(np.isnan(sc)):
            continue
        j = int(np.nanargmin(sc))
        if sc[j] < max_score:
            found.append(ks[j])
    if not found:
        raise ValueError("broadcast alignment failed: no matching windows near lag {}".format(coarse))
    return int(np.median(found)), len(found), n_windows


def broadcast_offsets():
    """Per-half lag k with tactical_frame = broadcast_index0 + k, found by matching player clouds.

    Returns {half: dict(offset, matched, windows, bc_range)}. Nothing hard-coded: a coarse lag from
    1 Hz formation summaries, then a frame-exact median over cloud-matched windows.
    """
    def build():
        t = _arrays("tactical")
        b = _arrays("broadcast")
        out = []
        seg = (0, len(b["B"]) - 1)  # each half is searched over the whole broadcast recording
        for h, a, z in HALF_BLOCKS:
            coarse = _coarse_lag(t["P"], b["P"], (a, z), seg)
            lag, matched, windows = _refine_lag(t["P"], b["P"], coarse, seg, (a, z))
            out.append([h, lag, matched, windows, max(a - lag, 0), min(z - lag, seg[1])])
        return dict(offsets=np.array(out, np.int64))

    paths = [config.asset_path(k, s) for s in ("tactical", "broadcast") for k in ("atd_meta", "atd_raw")]
    arr = _cached("bc-offsets-v2", paths, build)["offsets"]
    return {int(h): dict(offset=int(k), matched=int(m), windows=int(w), bc_range=(int(lo), int(hi)))
            for h, k, m, w, lo, hi in arr}


def load_broadcast():
    """Broadcast tracking re-indexed onto the tactical video timeline.

    Returns dict P (N,51,2), B (N,2), T (51,) and `covered` (N,) bool: frames where a broadcast
    record exists within its aligned half. Elsewhere P and B are NaN. Cutaway/replay stretches are
    NOT masked here (they have no tracked players or fail a cloud match; gate downstream).
    """
    b = _arrays("broadcast")
    offs = broadcast_offsets()
    n = config.N_VIDEO_FRAMES
    P = np.full((n, b["P"].shape[1], 2), np.nan, np.float32)
    B = np.full((n, 2), np.nan, np.float32)
    covered = np.zeros(n, bool)
    for h, info in offs.items():
        k = info["offset"]
        lo, hi = max(info["bc_range"][0], -k), min(info["bc_range"][1], n - 1 - k)
        sl_b = slice(lo, hi + 1)
        sl_t = slice(lo + k, hi + 1 + k)
        P[sl_t], B[sl_t], covered[sl_t] = b["P"][sl_b], b["B"][sl_b], True
    return dict(P=P, B=B, T=b["T"], covered=covered)
