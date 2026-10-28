"""External development games in the same array layout as :mod:`possession.data`.

``load_game(id)`` returns an :class:`possession.evaluate.Game`; ``list_games()`` discovers the
available IDs and ``load_games(ids)`` streams them one at a time. Coordinates are metres on a
105 x 68 (or the source's stated) pitch, with the origin at the top left, and all arrays are at
25 Hz. No competition match data or public GT is read here.

Raw data lives outside the solution in ``../external`` (override with ``MTAH_EXTERNAL_DIR``).
Parsed caches live in ``../external/cache`` (override with ``MTAH_EXTERNAL_CACHE_DIR``). Caches are
invalidated by the source files' sizes and mtimes. The external-data warnings in
``research/S05_external_manifest.md`` still apply: this loader does not remove label leaks.
"""
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import config, data, evaluate

FPS = 25
_CACHE_VERSION = 2
KMH_PER_MPS = 3.6  # IDSSE vendor BS (ball speed) is km/h
_IDSSE_SECTION = {"firstHalf": 1, "secondHalf": 2}
_FRAMESET_RE = re.compile(
    r'<FrameSet GameSection="([^"]+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME_RE = re.compile(
    r'<Frame N="(\d+)" T="[^"]*" X="([-+\d.eE]+)" Y="([-+\d.eE]+)"'
    r'(?: Z="([-+\d.eE]+)")?[^>]*? S="([-+\d.eE]+)"[^>]*?M="(\d+)"'
    r'(?: BallPossession="(\d+)" BallStatus="(\d+)")?')
_PITCH_RE = re.compile(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"')
_IDSSE_ID_RE = re.compile(r'(DFL-MAT-[A-Z0-9]+)\.xml$')


def external_dir():
    return os.path.abspath(os.environ.get(
        "MTAH_EXTERNAL_DIR", os.path.join(config.SOLUTION_DIR, "..", "external")))


def cache_dir():
    path = os.environ.get("MTAH_EXTERNAL_CACHE_DIR") or os.path.join(external_dir(), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _idsse_path(game_id, kind):
    paths = glob.glob(os.path.join(external_dir(), "idsse", kind + "*" + game_id + ".xml"))
    if len(paths) != 1:
        raise FileNotFoundError("expected one IDSSE {} file for {}, found {}".format(
            kind, game_id, len(paths)))
    return paths[0]


def list_games(dataset=None):
    """Available canonical game IDs (``idsse:DFL-MAT-...`` and ``metrica:gameN``)."""
    out = []
    if dataset in (None, "idsse"):
        pattern = os.path.join(external_dir(), "idsse", "DFL_04_03_positions_raw_observed_*.xml")
        for path in glob.glob(pattern):
            match = _IDSSE_ID_RE.search(path)
            if match and glob.glob(os.path.join(
                    external_dir(), "idsse", "DFL_02_01_matchinformation_*" + match.group(1) + ".xml")):
                out.append("idsse:" + match.group(1))
    if dataset in (None, "metrica"):
        for number in (1, 2, 3):
            name = "game{}".format(number)
            if os.path.isfile(os.path.join(external_dir(), "metrica", "labels", name + "_labels.csv")):
                out.append("metrica:" + name)
    if dataset not in (None, "idsse", "metrica"):
        raise ValueError("unknown dataset {!r}".format(dataset))
    return sorted(out)


def _source_stamps(paths):
    return [[os.path.abspath(path), os.stat(path).st_size, os.stat(path).st_mtime_ns]
            for path in paths]


def _cached(key, paths, build):
    target = os.path.join(cache_dir(), key + ".npz")
    stamps = json.dumps(_source_stamps(paths), separators=(",", ":"))
    if os.path.isfile(target):
        with np.load(target, allow_pickle=False) as z:
            if (int(z["_cache_version"]) == _CACHE_VERSION and
                    str(z["_source_stamps"]) == stamps):
                return {name: z[name] for name in z.files if not name.startswith("_")}
    arrays = build()
    arrays["_cache_version"] = np.array(_CACHE_VERSION)
    arrays["_source_stamps"] = np.array(stamps)
    tmp = target + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, target)
    return {name: value for name, value in arrays.items() if not name.startswith("_")}


def _goalkeepers(P, T, half, pitch_length):
    """The S03 goalkeeper rule, generalised to arbitrary half boundaries and pitch length."""
    out = np.zeros((2, 2), np.int64)
    for h in (1, 2):
        idx = np.flatnonzero(half == h)
        if not len(idx):
            continue
        x = np.abs(P[idx, :, 0] - pitch_length / 2)
        valid = ~np.isnan(x)
        count = valid.sum(axis=0)
        mean = np.divide(np.nansum(x, axis=0), count, out=np.full(x.shape[1], np.nan),
                         where=count > 0)
        mean = np.where(np.isnan(mean) | (count / len(idx) < 0.25), -1.0, mean)
        for team in (0, 1):
            slots = np.where(T == team)[0]
            out[h - 1, team] = slots[np.argmax(mean[slots])]
    return out


def _mask_runs(mask):
    """Half-open runs of consecutive true values."""
    x = np.asarray(mask, dtype=np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, x, 0]) != 0)
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _game_from_arrays(name, arrays, pitch_length=105.0):
    half = arrays["half"].astype(np.int8, copy=False)
    runs = arrays["runs"] if "runs" in arrays else _mask_runs(half > 0)
    match = data.Match(
        P=arrays["P"], V=arrays["V"], T=arrays["T"], B=arrays["B"], BS=arrays["BS"],
        gk=arrays["gk"], graded=arrays["graded"].astype(bool, copy=False), half=half)
    return evaluate.Game(name, match, arrays["ref"], runs)


def _idsse_meta(game_id):
    path = _idsse_path(game_id, "DFL_02_01_matchinformation_")
    root = ET.parse(path).getroot()
    general = next(root.iter("General"))
    environment = next(root.iter("Environment"))
    return (path, general.get("HomeTeamId"), general.get("GuestTeamId"),
            float(environment.get("PitchX")), float(environment.get("PitchY")))


def _load_idsse(game_id):
    positions = _idsse_path(game_id, "DFL_04_03_positions_raw_observed_")
    meta, home_id, guest_id, pitch_l, pitch_w = _idsse_meta(game_id)

    def build():
        # First pass: discover the union of player slots and read the compact ball/label stream.
        # Player frames are skipped without regex parsing; the second pass fills their arrays.
        balls = {1: [], 2: []}
        players = {}
        section = None
        is_ball = False
        pitch = None
        with open(positions, encoding="utf8") as fh:
            for line in fh:
                if line.startswith("<FrameSet"):
                    match = _FRAMESET_RE.match(line)
                    if not match:
                        raise ValueError("unparseable IDSSE FrameSet in " + positions)
                    section_name, team_id, person_id = match.groups()
                    section = _IDSSE_SECTION[section_name]
                    is_ball = team_id == "BALL"
                    if not is_ball and team_id in (home_id, guest_id):
                        team = 0 if team_id == home_id else 1
                        if person_id in players and players[person_id] != team:
                            raise ValueError("player changes team: " + person_id)
                        players[person_id] = team
                    # Some matches also carry referee FrameSets; Match has team slots only.
                elif line.startswith("<PitchSize"):
                    match = _PITCH_RE.match(line)
                    if match:
                        pitch = tuple(map(float, match.groups()))
                elif is_ball and line.startswith("<Frame "):
                    match = _FRAME_RE.match(line)
                    if not match:
                        raise ValueError("unparseable IDSSE ball frame in " + positions)
                    n, x, y, z, speed, measured, owner, status = match.groups()
                    if owner is None or status is None:
                        raise ValueError("IDSSE ball frame lacks owner/status")
                    balls[section].append((int(n), float(x), float(y), float(z or 0),
                                           float(speed), int(owner), int(status)))
        if pitch is not None and pitch != (pitch_l, pitch_w):
            raise ValueError("IDSSE pitch-size disagreement: {} vs {}".format(pitch, (pitch_l, pitch_w)))
        if not players or not all(balls.values()):
            raise ValueError("incomplete IDSSE position file " + positions)

        player_ids = sorted(players)
        slot = {person_id: i for i, person_id in enumerate(player_ids)}
        lengths = {h: len(balls[h]) for h in (1, 2)}
        offsets = {1: 0, 2: lengths[1]}
        n_total = sum(lengths.values())
        n_slots = len(player_ids)
        P = np.full((n_total, n_slots, 2), np.nan, np.float32)
        B = np.empty((n_total, 2), np.float32)
        BZ = np.empty(n_total, np.float32)
        BS = np.empty(n_total, np.float32)  # vendor speed, not the ATD's unexplained s channel
        ref = np.empty(n_total, dtype="<U1")
        half = np.empty(n_total, np.int8)
        N = np.empty(n_total, np.int32)
        runs = []
        for h in (1, 2):
            rows = balls[h]
            n = len(rows)
            a = offsets[h]
            frame_n = np.array([r[0] for r in rows], np.int32)
            if not np.array_equal(frame_n, np.arange(frame_n[0], frame_n[0] + n)):
                raise ValueError("non-contiguous IDSSE half {} in {}".format(h, game_id))
            xyz = np.array([[r[1], r[2], r[3]] for r in rows], np.float32)
            B[a:a + n] = xyz[:, :2] + np.array([pitch_l / 2, pitch_w / 2], np.float32)
            BZ[a:a + n] = xyz[:, 2]
            BS[a:a + n] = np.array([r[4] for r in rows], np.float32) / KMH_PER_MPS
            owner = np.array([r[5] for r in rows], np.int8)
            status = np.array([r[6] for r in rows], np.int8)
            if not np.isin(owner, (1, 2)).all() or not np.isin(status, (0, 1)).all():
                raise ValueError("unexpected IDSSE owner/status values in " + game_id)
            ref[a:a + n] = np.where(status == 0, "D", np.where(owner == 1, "H", "A"))
            half[a:a + n] = h
            N[a:a + n] = frame_n
            runs.append((a, a + n))

        # Second pass: fill player xy and vendor speed. The absolute XML frame N maps to the
        # contiguous match timeline via each half's first ball-frame number.
        section = None
        current_slot = None
        with open(positions, encoding="utf8") as fh:
            for line in fh:
                if line.startswith("<FrameSet"):
                    match = _FRAMESET_RE.match(line)
                    section_name, team_id, person_id = match.groups()
                    section = _IDSSE_SECTION[section_name]
                    current_slot = slot.get(person_id) if team_id in (home_id, guest_id) else None
                elif current_slot is not None and line.startswith("<Frame "):
                    match = _FRAME_RE.match(line)
                    if not match:
                        raise ValueError("unparseable IDSSE player frame in " + positions)
                    n, x, y, z, speed, measured, owner, status = match.groups()
                    row = offsets[section] + int(n) - N[offsets[section]]
                    P[row, current_slot] = (float(x) + pitch_l / 2, float(y) + pitch_w / 2)

        V = _speed(P, half)  # vendor S is km/h; derive from xy like Metrica so sources share one definition
        T = np.array([players[p] for p in player_ids], np.int8)
        return dict(P=P, V=V, T=T, B=B, BS=BS, BZ=BZ, N=N,
                    gk=_goalkeepers(P, T, half, pitch_l), graded=half > 0, half=half,
                    ref=ref, runs=np.array(runs, np.int64), pitch=np.array([pitch_l, pitch_w]),
                    player_id=np.array(player_ids))

    return _cached("idsse_" + game_id, [positions, meta], build)


def _speed(P, half):
    """Central-difference player speed at 25 Hz, never differencing across a half boundary."""
    V = np.full(P.shape[:2], np.nan, np.float32)
    for h in np.unique(half):
        idx = np.flatnonzero(half == h)
        if len(idx) < 2:
            continue
        a, b = idx[0], idx[-1] + 1
        xy = P[a:b]
        delta = np.empty_like(xy)
        delta[1:-1] = (xy[2:] - xy[:-2]) * (FPS / 2)
        delta[0] = (xy[1] - xy[0]) * FPS
        delta[-1] = (xy[-1] - xy[-2]) * FPS
        V[a:b] = np.linalg.norm(delta, axis=-1)
    return V


def _metrica_paths(number):
    raw = os.path.join(external_dir(), "metrica", "raw")
    labels = os.path.join(external_dir(), "metrica", "labels", "game{}_labels.csv".format(number))
    if number in (1, 2):
        paths = [os.path.join(raw, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(number, side))
                 for side in ("Home", "Away")]
    elif number == 3:
        paths = [os.path.join(raw, "Sample_Game_3_tracking.txt")]
    else:
        raise ValueError("Metrica game must be 1, 2 or 3")
    return paths + [labels]


def _load_metrica(number):
    import pandas as pd

    paths = _metrica_paths(number)

    def build():
        if number in (1, 2):
            sides = []
            frames = []
            for path, side in zip(paths[:2], ("Home", "Away")):
                columns = pd.read_csv(path, skiprows=2, nrows=0).columns
                names = [x for x in columns[3:] if not x.startswith("Unnamed") and x != "Ball"]
                values = pd.read_csv(path, skiprows=3, header=None)
                xy = values.iloc[:, 3:3 + 2 * len(names)].to_numpy(np.float32)
                sides.append(xy.reshape(len(values), len(names), 2))
                frames.append(values[1].to_numpy())
                if side == "Home":
                    ball = values.iloc[:, -2:].to_numpy(np.float32)
                    period = values[0].to_numpy(np.int8)
            if not np.array_equal(frames[0], frames[1]):
                raise ValueError("Metrica home/away frame mismatch in game {}".format(number))
            P = np.concatenate(sides, axis=1)
            T = np.r_[np.zeros(sides[0].shape[1], np.int8), np.ones(sides[1].shape[1], np.int8)]
        else:
            frame_list, player_list, ball_list = [], [], []
            with open(paths[0]) as fh:
                for line in fh:
                    frame, players, ball = line.rstrip("\n").split(":")
                    frame_list.append(int(frame))
                    player_list.append(np.fromstring(players.replace(";", ","), sep=","))
                    ball_list.append(np.fromstring(ball, sep=",")[:2])
            P = np.stack(player_list).reshape(len(frame_list), 22, 2).astype(np.float32)
            ball = np.stack(ball_list).astype(np.float32)
            T = np.r_[np.zeros(11, np.int8), np.ones(11, np.int8)]
            frames = [np.array(frame_list)]
            period = None

        scale = np.array([105.0, 68.0], np.float32)
        P = P * scale
        B = ball * scale
        label = pd.read_csv(paths[-1])
        if not np.array_equal(frames[0], label.frame.to_numpy()):
            raise ValueError("Metrica tracking/label frame mismatch in game {}".format(number))
        half = label.period.to_numpy(np.int8)
        if period is not None and not np.array_equal(period, half):
            raise ValueError("Metrica tracking/label period mismatch in game {}".format(number))
        ref = label.label.to_numpy(dtype="<U1")
        if not np.isin(ref, ("H", "A", "D")).all():
            raise ValueError("unexpected Metrica labels")
        graded = label.seg.to_numpy() != "dead_pre"
        V = _speed(P, half)
        return dict(P=P, V=V, T=T, B=B, BS=np.full(len(B), np.nan, np.float32),
                    gk=_goalkeepers(P, T, half, 105.0), graded=graded, half=half, ref=ref,
                    runs=np.array(_mask_runs(graded), np.int64),
                    pitch=np.array([105.0, 68.0]))

    return _cached("metrica_game{}".format(number), paths, build)


def load_game(game_id):
    """Load one canonical ID as ``evaluate.Game(name, match, ref, runs)``.

    ``match`` is exactly :class:`possession.data.Match`; ``ref`` is a string H/A/D array and
    ``runs`` contains the scored, per-half spans. Metrica pre-kickoff ``dead_pre`` frames are
    retained in the arrays but excluded from ``match.graded`` and ``runs``.
    """
    try:
        dataset, source_id = game_id.split(":", 1)
    except ValueError:
        raise ValueError("game ID must be dataset:source_id, e.g. metrica:game1") from None
    if dataset == "idsse":
        arrays = _load_idsse(source_id)
        return _game_from_arrays(game_id, arrays, float(arrays["pitch"][0]))
    if dataset == "metrica" and re.fullmatch(r"game[123]", source_id):
        arrays = _load_metrica(int(source_id[-1]))
        return _game_from_arrays(game_id, arrays, 105.0)
    raise ValueError("unknown external game ID {!r}".format(game_id))


def pitch(game):
    """(length, width) in metres of a game ID or ``evaluate.Game``, for ``features(..., pitch=...)``."""
    game_id = game if isinstance(game, str) else game.name
    dataset, _, source_id = game_id.partition(":")
    if dataset == "idsse":
        _, _, _, length, width = _idsse_meta(source_id)
        return float(length), float(width)
    if dataset == "metrica":
        return 105.0, 68.0
    raise ValueError("unknown external game ID {!r}".format(game_id))


def load_games(game_ids=None):
    """Generator over games; by default discovers every available IDSSE and Metrica game."""
    for game_id in list_games() if game_ids is None else game_ids:
        yield load_game(game_id)
