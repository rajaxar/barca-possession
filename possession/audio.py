"""Referee whistle on the tactical audio track -> *soft* dead-ball evidence for E07.

The tactical mp4 carries an AAC crowd/referee-mic mix. A referee whistle is two stable narrow
tonal ridges ~400 Hz apart in the 3.4-4.7 kHz band lasting 0.4-1.2 s (R06: measured on the two
published kickoffs, where the whistle ends 0.4-0.7 s before the kickoff frame). Crowd noise is
broadband below ~2.5 kHz; fan whistles are single *swept* ridges. The detector is therefore a
local spectral-contrast test: a frame is a whistle frame if its strongest narrow-band ridge is
>= 20 dB above the local spectral neighbourhood (+-95/155 Hz) and a second ridge at least 235 Hz
away is >= 12 dB up. Nothing here is learned, seeded or tuned on a score; the thresholds are
R06's, chosen from the physics of the sound and the published kickoff spectrograms
(GUARDRAILS 1.0: a league kickoff is structure, not a label).

Outputs:
  * `whistle_events(video)`      -> events at 25 fps (start/end/peak frame + f1/f2/contrast)
  * `crowd_envelope(video)`      -> 0.3-3 kHz crowd loudness at 25 Hz, normalised per half
  * `d_evidence(match, events)`  -> per-frame soft D evidence: a weight in [0, 1], an additive
                                   log-odds increment and its posterior, gated by tracking
                                   stillness. Never a hard D rule (p <= 0.80).

Everything degrades gracefully: no video / no ffmpeg / no audio track -> empty event list with a
`reason`, never an exception. The audio is streamed from an ffmpeg subprocess and reduced frame by
frame, so peak memory is one ~40 MB chunk, not the 500 MB decoded track.

Offline and deterministic, numpy only (no librosa, no ffmpeg-python) plus the ffmpeg *binary*
for decoding; every threshold is a fixed constant, so the same audio always gives the same list.
"""
import hashlib
import json
import os
import shutil
import subprocess
from collections import namedtuple

import numpy as np

from . import config
from .structure import HALF_BLOCKS

# ------------------------------------------------------------------ detector constants (R06)
FPS = 25
SR = 16000  # frame-ok: audio sample rate in Hz, not a video frame
# 16 kHz mono; Nyquist 8 kHz covers the 3.4-4.7 kHz whistle band
N_FFT = 2048               # 128 ms window
HOP = 512                  # 32 ms hop -> 31.25 STFT frames per second
TONE_BAND_HZ = (3300.0, 4900.0)   # search band (a little wider than the peak band)
PEAK_BAND_HZ = (3400.0, 4700.0)   # R06: both ridges live here
CROWD_BAND_HZ = (300.0, 3000.0)   # R06-c: crowd loudness, everything a whistle is not
CONTRAST_REF_HZ = (95.0, 155.0)   # local neighbourhood: just outside a ridge's own skirt
C1_MIN_DB = 20.0           # strongest ridge vs local floor
C2_MIN_DB = 12.0           # second ridge vs local floor
MIN_SEP_HZ = 235.0         # rejects a single ridge's skirts / low-f0 harmonic stacks
MIN_FRAMES = 6             # 6 * 32 ms = 0.19 s of tone; R06: whistles last 0.4-1.2 s
MAX_GAP = 3                # bridge up to 3 silent STFT frames inside one whistle
EPS = 1e-10

Config = namedtuple("Config", "sr n_fft hop tone_band peak_band crowd_band ref_hz c1_db c2_db "
                              "min_sep_hz min_frames max_gap")
CFG = Config(SR, N_FFT, HOP, TONE_BAND_HZ, PEAK_BAND_HZ, CROWD_BAND_HZ, CONTRAST_REF_HZ,
             C1_MIN_DB, C2_MIN_DB, MIN_SEP_HZ, MIN_FRAMES, MAX_GAP)

# Bumped whenever the numbers above (or the STFT) change: it is part of every cache key, so a
# stale cached feature block can never leak into a run with new thresholds.
DETECTOR_VERSION = "e15-1"

# Feature arrays at the STFT rate (t: seconds, centre of the window).
Feat = namedtuple("Feat", "t c1 f1 c2 f2 crowd_db")
# One whistle is a row of EVENT_DTYPE below: 0-based video frames (submission rows), 25 fps, so
# frame f is at t = f / 25 s (organiser convention: frame_id = video_frame + 1, R06 §1).
Track = namedtuple("Track", "events ok reason video")
# Soft evidence. p / log_odds are (N,) float32; gate / onset_frame are one value per event.
Evidence = namedtuple("Evidence", "p log_odds gate onset_frame events gain")

EVENT_DTYPE = np.dtype([("start_frame", "i4"), ("end_frame", "i4"), ("peak_frame", "i4"),
                        ("dur_s", "f4"), ("f1_hz", "f4"), ("f2_hz", "f4"), ("sep_hz", "f4"),
                        ("c1_db", "f4"), ("c2_db", "f4"), ("f1_sd_hz", "f4"),
                        ("sep_sd_hz", "f4"), ("n_stft", "i4")])

# Shape and strength of the soft D evidence (see d_evidence).
#   weight w(f) = gate * shape in [0, 1]: 1 = "this frame is a whistle stop, full strength".
#   log_odds    = LOG_ODDS_GAIN * w, the increment E07 may ADD to its own D log-odds.
#   p           = sigmoid(log_odds), the posterior if the whistle were the only evidence.
# LOG_ODDS_GAIN = 1.4 is the strongest single-cue nudge we are willing to inject (p <= 0.80, so it
# can never alone decide D). E25's external-trained P(D) rises from 0.49 to 0.71 three seconds after
# a whistle, i.e. +0.94 log-odds for a *typical* whistle whose gate is ~0.5: 1.4 * 0.5 * 0.6 = 0.42
# at 3 s and 0.7 for a full-gate whistle. E07 should keep its own scale and recalibrate on dev.
LOG_ODDS_GAIN = 1.4
P_CAP = float(1.0 / (1.0 + np.exp(-LOG_ODDS_GAIN)))   # 0.802
BOOST_S = 1.0              # full-strength plateau after the whistle onset
BOOST_DECAY_S = 4.0        # exponential decay constant after the plateau
BOOST_HORIZON_S = 10.0     # no contribution beyond this
GATE_WINDOW_S = (0.5, 3.0)  # tracking state is measured here, after the whistle
STILL_SPEED_MS = 1.5       # walk/jog boundary; the same cut E25 and R06 used (physical, not fitted)


# ------------------------------------------------------------------ ffmpeg / audio input
def ffmpeg_bin():
    """The ffmpeg CLI, or None. MTAH_FFMPEG overrides the lookup (tests use it to force absence)."""
    env = os.environ.get("MTAH_FFMPEG")
    if env:
        return env if (os.path.isfile(env) and os.access(env, os.X_OK)) else None
    return shutil.which("ffmpeg")


def probe(video):
    """(info dict, reason). Cheap container probe; never raises."""
    ff = ffmpeg_bin()
    if not ff:
        return None, "no_ffmpeg"
    if not video or not os.path.isfile(video):
        return None, "no_video"
    ffprobe = os.path.join(os.path.dirname(ff), "ffprobe")
    exe = ffprobe if os.path.isfile(ffprobe) else shutil.which("ffprobe")
    if not exe:
        return {"audio": None}, "no_ffprobe"
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels,start_time,duration:format=duration",
             "-of", "json", video],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {"audio": None}, "no_ffprobe"
    try:
        blob = json.loads(out.stdout or "{}")
    except ValueError:
        return {"audio": None}, "no_ffprobe"
    streams = blob.get("streams") or []
    info = {"audio": streams[0] if streams else None,
            "duration": float((blob.get("format") or {}).get("duration") or 0.0) or None}
    return info, ("ok" if info["audio"] else "no_audio")


def pcm_chunks(video, start=0.0, duration=None, chunk=1 << 20, sr=SR):
    """Yield float32 mono PCM blocks from the video's first audio stream (ffmpeg CLI, streamed).

    `start`/`duration` are seconds of *audio* time. Raises RuntimeError only if ffmpeg fails before
    producing any audio (the caller turns that into a graceful skip). A failure *after* some audio
    was decoded yields what was decoded, so the event list covers only that part of the track.
    """
    ff = ffmpeg_bin()
    if not ff:
        raise RuntimeError("no_ffmpeg")
    cmd = [ff, "-nostdin", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", "%.3f" % start]
    if duration:
        cmd += ["-t", "%.3f" % duration]
    cmd += ["-i", video, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    got = False
    try:
        while True:
            raw = proc.stdout.read(chunk * 4)
            if not raw:
                break
            got = True
            yield np.frombuffer(raw, dtype="<f4").copy()
    finally:
        proc.stdout.close()
        err = proc.stderr.read() if proc.stderr else b""
        proc.stderr.close()
        code = proc.wait()
    if code != 0 and not got:
        raise RuntimeError("ffmpeg_failed: " + (err.decode("utf-8", "replace").strip()[:200] or code))


# ------------------------------------------------------------------ chunked STFT -> features
def _contrast(L, ref_bins):
    """Level minus the louder of the two neighbours at +-d bins, for each d; -inf at the edges."""
    floor = np.float32(-120.0)
    ref = np.full_like(L, floor)
    n = L.shape[1]
    for d in ref_bins:
        if d <= 0 or d >= n:
            continue
        a = np.full_like(L, floor)
        a[:, :n - d] = L[:, d:]
        np.maximum(ref, a, out=ref)
        b = np.full_like(L, floor)
        b[:, d:] = L[:, :-d]
        np.maximum(ref, b, out=ref)
    return L - ref


def frame_features(x, cfg=CFG):
    """Per-frame whistle + crowd features for a block of raw frames, shape (n, n_fft)."""
    win = np.hanning(cfg.n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * win, axis=1))
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / cfg.sr)
    df = cfg.sr / cfg.n_fft
    tone = (freqs >= cfg.tone_band[0] - 1e-6) & (freqs <= cfg.tone_band[1] + 1e-6)
    peak = (freqs >= cfg.peak_band[0] - 1e-6) & (freqs <= cfg.peak_band[1] + 1e-6)
    crowd = (freqs >= cfg.crowd_band[0]) & (freqs <= cfg.crowd_band[1])
    db = 20.0 * np.log10(spec + EPS)
    L = db[:, tone].astype(np.float32)
    ref_bins = [int(round(hz / df)) for hz in cfg.ref_hz]
    peak_in_tone = peak[tone]
    C = _contrast(L, ref_bins)[:, peak_in_tone]
    fp = freqs[peak]
    k1 = C.argmax(1)
    c1 = C.max(1)
    C2 = C.copy()
    rows = np.arange(len(C))
    m = int(round(cfg.min_sep_hz / df))
    for off in range(-m, m + 1):
        C2[rows, np.clip(k1 + off, 0, C.shape[1] - 1)] = -120.0
    k2 = C2.argmax(1)
    crowd_db = 10.0 * np.log10((spec[:, crowd] ** 2).mean(1) + EPS)
    return Feat(t=None, c1=c1, f1=fp[k1].astype(np.float32), c2=C2.max(1),
                f2=fp[k2].astype(np.float32), crowd_db=crowd_db.astype(np.float32))


def features_from_chunks(chunks, cfg=CFG):
    """Per-frame Feat for an iterable of float32 PCM blocks (the STFT is chunked, features are not)."""
    parts = []
    buf = np.zeros(0, np.float32)
    for chunk in chunks:
        buf = np.concatenate((buf, chunk)) if buf.size else np.asarray(chunk, np.float32)
        n = (buf.size - cfg.n_fft) // cfg.hop + 1
        if n <= 0:
            continue
        idx = np.arange(n, dtype=np.int32)[:, None] * cfg.hop + np.arange(cfg.n_fft, dtype=np.int32)
        parts.append(frame_features(buf[idx], cfg))
        buf = buf[n * cfg.hop:]
    if buf.size >= cfg.n_fft:
        n = (buf.size - cfg.n_fft) // cfg.hop + 1
        idx = np.arange(n, dtype=np.int32)[:, None] * cfg.hop + np.arange(cfg.n_fft, dtype=np.int32)
        parts.append(frame_features(buf[idx], cfg))
    if not parts:  # no full frame at all (very short or silent track)
        z = np.zeros(0, np.float32)
        return Feat(t=z, c1=z, f1=z, c2=z, f2=z, crowd_db=z)
    cat = {f: np.concatenate([p[i] for p in parts]) for i, f in enumerate(Feat._fields) if f != "t"}
    n = len(cat["c1"])
    t = (np.arange(n, dtype=np.float64) * cfg.hop + cfg.n_fft / 2.0) / cfg.sr
    return Feat(t=t, **cat)


def analyse(video, start=0.0, duration=None, cfg=CFG, cache=True):
    """Stream the audio from the video and return Feat at the STFT rate (t = window centre, s).

    The per-frame features are cached under config.cache_dir() (keyed by the video's size+mtime,
    the detector version and the slice), so the one-off decode of the full match happens once.
    """
    video = str(video)
    extra = dict(start=round(float(start), 3),
                 duration=None if duration is None else round(float(duration), 3),
                 cfg=[list(x) if isinstance(x, tuple) else x for x in cfg])
    path = _cache_path(video, "feat", extra)
    if cache and os.path.isfile(path):
        try:
            with np.load(path) as z:
                return Feat(t=z["t"], c1=z["c1"], f1=z["f1"], c2=z["c2"], f2=z["f2"],
                            crowd_db=z["crowd_db"])
        except (OSError, ValueError, KeyError):
            pass
    feat = features_from_chunks(pcm_chunks(video, start=start, duration=duration, sr=cfg.sr), cfg)
    if cache and len(feat.t):
        try:
            _save(path, t=feat.t, c1=feat.c1, f1=feat.f1, c2=feat.c2, f2=feat.f2,
                  crowd_db=feat.crowd_db)
        except OSError:
            pass
    return feat


# ------------------------------------------------------------------ events
def _runs(mask, max_gap):
    """(first, last) index pairs of True runs, bridging False gaps <= max_gap."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        last = i
        j = i
        while j + 1 < n and j + 1 - last <= max_gap:
            j += 1
            if mask[j]:
                last = j
        out.append((i, last))
        i = last + 1
    return out


def detect(feat, cfg=CFG, t0=0.0):
    """Whistle events from Feat. Returns a structured array sorted by start frame.

    `t0` is the absolute audio time of `feat.t[0]` (non-zero when the audio was read from a slice),
    so the emitted frames are always indices into the full video.
    """
    empty = np.zeros(0, dtype=EVENT_DTYPE)
    n = len(feat.c1)
    if n == 0:
        return empty
    hits = (feat.c1 > cfg.c1_db) & (feat.c2 > cfg.c2_db)
    rows = []
    half_win = cfg.n_fft / (2.0 * cfg.sr)
    for a, b in _runs(hits, cfg.max_gap):
        if b - a + 1 < cfg.min_frames:
            continue
        sl = slice(a, b + 1)
        # Order the pair low/high first: either ridge can be the stronger one frame to frame, so
        # stability must be measured on the sorted pair, not on "strongest"/"second".
        lo_hz = np.minimum(feat.f1[sl], feat.f2[sl])
        hi_hz = np.maximum(feat.f1[sl], feat.f2[sl])
        f1 = float(np.median(lo_hz))
        f2 = float(np.median(hi_hz))
        sep = f2 - f1
        t0a = float(feat.t[a] + t0 - half_win)
        t1a = float(feat.t[b] + t0 + half_win)
        k = a + int(np.argmax(feat.c1[sl]))
        rows.append((int(round(t0a * FPS)) + 1, int(round(t1a * FPS)) + 1,
                     int(round(float(feat.t[k] + t0) * FPS)) + 1, t1a - t0a, f1, f2, sep,
                     float(feat.c1[a:b + 1].max()), float(feat.c2[a:b + 1].max()),
                     float(np.std(lo_hz)), float(np.std((hi_hz - lo_hz))),
                     b - a + 1))
    if not rows:
        return empty
    ev = np.array(rows, dtype=EVENT_DTYPE)
    return ev[np.argsort(ev["start_frame"], kind="stable")]


# ------------------------------------------------------------------ cache helpers
def _video_key(video, extra):
    st = os.stat(video)
    h = hashlib.blake2b(digest_size=12)
    h.update(("%d:%d:%s" % (st.st_size, st.st_mtime_ns, DETECTOR_VERSION)).encode())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _cache_path(video, kind, extra):
    return os.path.join(config.cache_dir(), "audio_%s_%s.npz" % (kind, _video_key(video, extra)))


def _save(path, **arrays):
    tmp = path[:-4] + ".tmp%d.npz" % os.getpid()
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ public entry points
def whistle_events(video=None, start=0.0, duration=None, cfg=CFG, cache=True):
    """Whistle events for the tactical audio, at 25 fps. Missing video / ffmpeg / audio -> ok=False.

    Video frames follow the organisers' convention: time of frame f (0-based) is f / 25 s
    (R06; the ATD frame_id is video_frame + 1).
    """
    video = config.asset_path("video") if video is None else video
    if video is None or str(video).endswith(os.sep):
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_video", video)
    video = str(video)
    if not ffmpeg_bin():
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_ffmpeg", video)
    if not os.path.isfile(video):
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_video", video)
    _, reason = probe(video)
    if reason == "no_audio":  # container has no audio stream at all: skip without decoding
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_audio", video)
    extra = dict(start=round(float(start), 3),
                 duration=None if duration is None else round(float(duration), 3))
    path = _cache_path(video, "ev", extra)
    if cache and os.path.isfile(path):
        try:
            with np.load(path) as z:
                return Track(z["events"], True, "ok", video)
        except (OSError, ValueError, KeyError):
            pass
    try:
        feat = analyse(video, start=start, duration=duration, cfg=cfg, cache=cache)
    except RuntimeError as exc:
        why = str(exc)
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False,
                     "no_audio" if "ffmpeg_failed" in why else why, video)
    ev = detect(feat, cfg, t0=float(start or 0.0))
    if cache:
        try:
            _save(path, events=ev)
        except OSError:
            pass
    return Track(ev, True, "ok", video)


def crowd_from_feat(feat, n_frames=None, t0=0.0):
    """(N,) crowd loudness at 25 Hz, robust-z normalised within each half (R06-c)."""
    n_frames = config.N_VIDEO_FRAMES if n_frames is None else int(n_frames)
    out = np.full(n_frames, np.nan, np.float32)
    if len(feat.t):
        tv = np.arange(n_frames) / float(FPS)
        out = np.interp(tv, feat.t + t0, feat.crowd_db,
                        left=np.nan, right=np.nan).astype(np.float32)
    for h, a, z in HALF_BLOCKS:
        lo, hi = a - 1, min(z, n_frames)
        seg = out[lo:hi]
        good = np.isfinite(seg)
        if good.sum() < 100:
            continue
        med = float(np.median(seg[good]))
        mad = float(np.median(np.abs(seg[good] - med))) * 1.4826
        seg[good] = (seg[good] - med) / (mad if mad > 1e-6 else 1.0)
    out[:HALF_BLOCKS[0][1] - 1] = np.nan
    out[HALF_BLOCKS[0][2]:HALF_BLOCKS[1][1] - 1] = np.nan
    out[min(HALF_BLOCKS[1][2], n_frames):] = np.nan
    return out


def crowd_envelope(video=None, start=0.0, duration=None, cfg=CFG, n_frames=None, cache=True):
    """(N,) crowd loudness at 25 Hz, robust-z normalised *within each half* (R06-c).

    N defaults to config.N_VIDEO_FRAMES. NaN outside the two halves and wherever the audio is
    missing. This is a diagnostic candidate only: E25 found crowd loudness does not separate D
    (AUC .486), and E15 confirms it label-free (research/E15_result.md).
    """
    video = config.asset_path("video") if video is None else video
    n_frames = config.N_VIDEO_FRAMES if n_frames is None else int(n_frames)
    if not video or not os.path.isfile(str(video)) or not ffmpeg_bin():
        return np.full(n_frames, np.nan, np.float32)
    video = str(video)
    path = _cache_path(video, "crowd", dict(start=round(float(start), 3), n=n_frames,
                                            duration=None if duration is None else round(float(duration), 3)))
    if cache and os.path.isfile(path):
        try:
            with np.load(path) as z:
                return z["env"]
        except (OSError, ValueError, KeyError):
            pass
    try:
        feat = analyse(video, start=start, duration=duration, cfg=cfg, cache=cache)
    except RuntimeError:
        return np.full(n_frames, np.nan, np.float32)
    out = crowd_from_feat(feat, n_frames, t0=float(start or 0.0))
    if cache:
        try:
            _save(path, env=out.astype(np.float32))
        except OSError:
            pass
    return out


# ------------------------------------------------------------------ soft D evidence
def _mean_speed(match):
    """Per-frame mean player speed in m/s (clipped, NaN where no player is tracked).

    E25's statistic: mean over tracked slots of the speed channel, clipped at 12 m/s. It is a
    property of the tracking signal, not of this match's frames.
    """
    ok = np.isfinite(match.P[..., 0])
    v = np.where(ok, np.clip(np.nan_to_num(match.V), 0.0, 12.0), np.nan)
    return np.nanmean(v, axis=1)


def _shape(dt):
    """Evidence shape (seconds after the whistle onset): plateau, exponential decay, hard stop."""
    if dt < 0.0 or dt > BOOST_HORIZON_S:
        return 0.0
    if dt <= BOOST_S:
        return 1.0
    return float(np.exp(-(dt - BOOST_S) / BOOST_DECAY_S))


def d_evidence(match, events=None, video=None, gain=LOG_ODDS_GAIN):
    """Soft per-frame D evidence from whistle events, gated by tracking stillness.

      weight w[f] = gate(event) * shape(f - onset)
        shape: 1 for 0..1 s after the whistle, then exp(-(dt-1)/4 s), 0 after 10 s.
        gate : fraction of frames in GATE_WINDOW_S (0.5-3 s after the whistle peak) where the ball
               is missing AND the mean player speed < STILL_SPEED_MS. That is the state E25
               measured at 0.51 of whistle times vs 0.35 of random times; a fan whistle blown
               during end-to-end play gets gate ~0.
      log_odds[f] = gain * w[f]      additive evidence for E07's D score (never a rule)
      p[f]        = sigmoid(log_odds[f])   the posterior with the whistle as the only evidence
    Everything is 0 away from a whistle; p <= sigmoid(gain) = 0.80, so audio alone can never
    declare D. E07 should add `log_odds` to its own D log-odds and keep its duration model.
    """
    n = len(match.B)
    weight = np.zeros(n, np.float32)
    gate = np.zeros(0, np.float32)
    onset = np.zeros(0, np.int32)
    if events is None:
        events = whistle_events(video).events
    if len(events) == 0:
        return Evidence(np.zeros(n, np.float32), np.zeros(n, np.float32), gate, onset, events, gain)
    speed = _mean_speed(match)
    missing = ~np.isfinite(match.B[:, 0])
    still = np.isfinite(speed) & (speed < STILL_SPEED_MS)
    dead_ish = (missing & still).astype(np.float32)
    lo = int(round(GATE_WINDOW_S[0] * FPS))
    hi = int(round(GATE_WINDOW_S[1] * FPS))
    gate = np.zeros(len(events), np.float32)
    for i, ev in enumerate(events):
        a = max(int(ev["peak_frame"]) + lo, 0)
        b = min(int(ev["peak_frame"]) + hi, n)
        gate[i] = float(dead_ish[a:b].mean()) if b > a else 0.0
    onset = events["start_frame"].astype(np.int32)
    for i, s in enumerate(onset):
        a = max(int(s), 0)
        b = min(int(s) + int(BOOST_HORIZON_S * FPS), n)
        if b <= a:
            continue
        dt = (np.arange(a, b) - int(s)) / float(FPS)
        shape = np.where(dt <= BOOST_S, 1.0,
                         np.exp(-np.maximum(dt - BOOST_S, 0.0) / BOOST_DECAY_S)).astype(np.float32)
        np.maximum(weight[a:b], gate[i] * shape, out=weight[a:b])
    log_odds = (gain * weight).astype(np.float32)
    p = (1.0 / (1.0 + np.exp(-log_odds.astype(np.float64)))).astype(np.float32)
    p[weight <= 0] = 0.0
    return Evidence(p, log_odds, gate, onset, events, gain)
