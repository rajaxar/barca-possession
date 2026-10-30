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

Everything degrades gracefully: no video / no ffmpeg / no audio stream / a truncated, failed or
timed-out decode -> empty event list with a `reason` (`no_video`, `no_ffmpeg`, `no_audio`,
`popen_failed`, `ffmpeg_failed`, `partial_decode`, `timeout`), never an exception and never a
plausible-looking partial list. The audio is streamed from an ffmpeg subprocess whose stderr is
drained on a thread and which is killed after a wall-clock timeout; features are reduced frame by
frame, so peak memory is ~300 MB (one 2048x1025 complex STFT plus its index array), not the
500 MB decoded track.

Offline and deterministic, numpy only (no librosa, no ffmpeg-python) plus the ffmpeg *binary*
for decoding; every threshold is a fixed constant, so the same audio always gives the same list.
The STFT grid is anchored to absolute audio time (`start` multiples of `hop`), so a sliced call
returns the same frames as the full run over the same interval (S26/F6).
"""
import errno
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
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
MAX_GAP = 3                # _runs bridges max_gap-1 = 2 silent STFT frames = 64 ms inside a whistle
EPS = 1e-10

Config = namedtuple("Config", "sr n_fft hop tone_band peak_band crowd_band ref_hz c1_db c2_db "
                              "min_sep_hz min_frames max_gap")
CFG = Config(SR, N_FFT, HOP, TONE_BAND_HZ, PEAK_BAND_HZ, CROWD_BAND_HZ, CONTRAST_REF_HZ,
             C1_MIN_DB, C2_MIN_DB, MIN_SEP_HZ, MIN_FRAMES, MAX_GAP)

# Bumped whenever the numbers above (or the STFT) change: it is part of every cache key, so a
# stale cached feature block can never leak into a run with new thresholds.
DETECTOR_VERSION = "e15-1"
# Bumped whenever anything else about this file changes: the feature computation, `detect`, the
# frame mapping or the cache layout. Thresholds are DETECTOR_VERSION's business; both are part of
# every cache key, so a code change cannot reuse arrays computed by older code (S26/F5).
MODULE_VERSION = "s26-1"

# Decode guards (S26/F1-F2). The full match decodes in ~35-40 s here, so 300 s is ~8x that before
# we give up; MTAH_AUDIO_TIMEOUT_S overrides it (tests use a 1 s timeout on a hung shim).
AUDIO_TIMEOUT_S = 300.0
# |expected - decoded| samples tolerated before a decode is called partial. The AAC edit list and
# the container-duration rounding cost ~0.09 s on the full match; 1 s is 10x that and still flags
# any truncation that could hide or move a whistle.
SAMPLE_TOL_S = 1.0

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
class AudioUnavailable(RuntimeError):
    """A decode that could not be completed; `reason` is a stable token for `Track.reason`."""

    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


def _timeout_s():
    """Wall-clock limit for one decode, MTAH_AUDIO_TIMEOUT_S overridable."""
    try:
        return float(os.environ.get("MTAH_AUDIO_TIMEOUT_S", AUDIO_TIMEOUT_S))
    except (TypeError, ValueError):
        return AUDIO_TIMEOUT_S


def _drain(pipe, sink, limit=1 << 16):
    """Read a pipe to EOF on a worker thread, keeping at most `limit` bytes for diagnostics."""
    try:
        while True:
            block = pipe.read(1 << 16)
            if not block:
                break
            if len(sink) < limit:
                sink.extend(block[:limit - len(sink)])
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


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


def pcm_chunks(video, start=0.0, duration=None, chunk=1 << 20, sr=SR,
               expect_samples=None, timeout=None):
    """Yield float32 mono PCM blocks from the video's first audio stream (ffmpeg CLI, streamed).

    `start`/`duration` are seconds of *audio* time. Raises `AudioUnavailable` (never a bare
    exception) if ffmpeg cannot be spawned, if the decode exceeds `timeout` wall-clock seconds
    (the child is killed), if the child exits non-zero, or if fewer samples than
    `expect_samples` (from the container duration) arrived. A truncated decode must never look
    like a complete one: the caller turns every failure into `ok=False` and caches nothing.
    """
    ff = ffmpeg_bin()
    if not ff:
        raise AudioUnavailable("no_ffmpeg")
    timeout = _timeout_s() if timeout is None else float(timeout)
    cmd = [ff, "-nostdin", "-loglevel", "error"]
    if start and start > 0:
        cmd += ["-ss", "%.3f" % start]
    if duration:
        cmd += ["-t", "%.3f" % duration]
    cmd += ["-i", video, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:  # e.g. Exec format error, vanished or unreadable binary
        raise AudioUnavailable("no_ffmpeg" if exc.errno == errno.ENOENT else "popen_failed",
                               str(exc))
    # Drain stderr on its own thread: with >64 KB of ffmpeg errors an undrained pipe blocks the
    # child, which stops producing stdout and deadlocks the reader (S26/F2).
    err = bytearray()
    drain = threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True)
    drain.start()
    killed = threading.Event()

    def _kill():  # watchdog: a hung decode is worse than a failed one
        killed.set()
        try:  # the child leads its own process group (start_new_session), so kill its helpers too
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            try:
                proc.kill()
            except OSError:
                pass

    timer = None
    if timeout and timeout > 0:
        timer = threading.Timer(timeout, _kill)
        timer.daemon = True
        timer.start()
    got = 0
    tail = b""
    try:
        while True:
            raw = proc.stdout.read(chunk * 4)
            if not raw:
                break
            raw = tail + raw
            n = len(raw) // 4          # a short final read must not break the sample stride
            tail = raw[n * 4:]
            if not n:
                continue
            got += n
            yield np.frombuffer(raw[:n * 4], dtype="<f4").copy()
        proc.wait()
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        if timer is not None:
            timer.cancel()
        if proc.poll() is None:  # consumer stopped early: never leave a decode running
            _kill()
        code = proc.wait()
        drain.join(timeout=5.0)
    if killed.is_set():
        raise AudioUnavailable("timeout", "no audio in %.0f s" % timeout)
    if code != 0 and got == 0:
        raise AudioUnavailable("ffmpeg_failed",
                               err.decode("utf-8", "replace").strip()[:200] or str(code))
    if expect_samples is not None:
        short = int(expect_samples) - got
        if short > int(round(SAMPLE_TOL_S * sr)):
            raise AudioUnavailable("partial_decode", "%d of %d samples (%.2f s missing)"
                                   % (got, int(expect_samples), short / float(sr)))
    if code != 0:
        raise AudioUnavailable("ffmpeg_failed",
                               err.decode("utf-8", "replace").strip()[:200] or str(code))


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


def features_from_chunks(chunks, cfg=CFG, first_sample=0):
    """Per-frame Feat for an iterable of float32 PCM blocks (the STFT is chunked, features are not).

    `first_sample` is the absolute index (in the source stream, from sample 0) of the first sample
    in `chunks`, so `feat.t` is always *absolute* audio time in seconds. A slice decoded from
    `start` therefore lands on the same windows as the full run (S26/F6).
    """
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
    t = (int(first_sample) + np.arange(n, dtype=np.float64) * cfg.hop
         + cfg.n_fft / 2.0) / cfg.sr
    return Feat(t=t, **cat)


def _audio_duration(info):
    """Stream duration if ffprobe reported one, else the container duration, else None."""
    if not info:
        return None
    audio = info.get("audio") or {}
    for value in (audio.get("duration"), info.get("duration")):
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _grid(start, sr, hop):
    """(first_sample, skip): drop `skip` decoded samples so windows sit on absolute hop multiples.

    ffmpeg's `-ss` returns audio that starts at ~round(start*sr); skipping the remainder of the hop
    puts every STFT window at the same absolute sample as in the full run (S26/F6).
    """
    off = int(round(max(0.0, float(start or 0.0)) * sr))
    skip = (-off) % hop
    return off + skip, skip


def _drop(chunks, k):
    """Drop the first `k` samples from a chunk stream."""
    for chunk in chunks:
        if k > 0:
            if chunk.size <= k:
                k -= chunk.size
                continue
            chunk = chunk[k:]
            k = 0
        yield chunk


def analyse(video, start=0.0, duration=None, cfg=CFG, cache=True, info=None, timeout=None):
    """Stream the audio from the video and return Feat at the STFT rate (t = absolute seconds).

    The per-frame features are cached under config.cache_dir(), keyed by a content fingerprint of
    the video, the module/detector versions, `cfg` and the slice. A partial or failed decode raises
    `AudioUnavailable` *before* anything is cached, so a truncated run cannot poison the cache.
    """
    video = str(video)
    sr, hop = cfg.sr, cfg.hop
    if info is None:
        info, _ = probe(video)
    dur = _audio_duration(info)
    first, skip = _grid(start, sr, hop)
    expect = None
    if dur is not None:
        span = dur - max(0.0, float(start or 0.0))
        if duration:
            span = min(span, float(duration))
        expect = int(round(max(0.0, span) * sr))
    extra = dict(start=round(float(start), 3),
                 duration=None if duration is None else round(float(duration), 3),
                 first_sample=first, skip=skip)
    path = _cache_path(video, "feat", extra, cfg) if cache else None
    if path and os.path.isfile(path):
        try:
            with np.load(path) as z:
                return Feat(t=z["t"], c1=z["c1"], f1=z["f1"], c2=z["c2"], f2=z["f2"],
                            crowd_db=z["crowd_db"])
        except (OSError, ValueError, KeyError):
            pass
    chunks = _drop(pcm_chunks(video, start=start, duration=duration, sr=sr,
                              expect_samples=expect, timeout=timeout), skip)
    feat = features_from_chunks(chunks, cfg, first_sample=first)
    if path and len(feat.t):
        try:
            _save(path, t=feat.t, c1=feat.c1, f1=feat.f1, c2=feat.c2, f2=feat.f2,
                  crowd_db=feat.crowd_db)
        except OSError:
            pass
    return feat


# ------------------------------------------------------------------ events
def _runs(mask, max_gap):
    """(first, last) index pairs of True runs, bridging False gaps of at most `max_gap - 1` frames.

    So max_gap=3 bridges 1- and 2-frame gaps (32-64 ms at 25 fps) but splits at 3.
    """
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

    `feat.t` is absolute audio time (see `features_from_chunks`), so `t0` only exists for callers
    that built a Feat on a relative clock; the emitted frames are always submission rows:
    frame f is at t = f / 25 s (the ATD frame_id is f + 1), 0-based, never negative.
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
        rows.append((max(0, int(round(t0a * FPS))), max(0, int(round(t1a * FPS))),
                     max(0, int(round(float(feat.t[k] + t0) * FPS))), t1a - t0a, f1, f2, sep,
                     float(feat.c1[a:b + 1].max()), float(feat.c2[a:b + 1].max()),
                     float(np.std(lo_hz)), float(np.std((hi_hz - lo_hz))),
                     b - a + 1))
    if not rows:
        return empty
    ev = np.array(rows, dtype=EVENT_DTYPE)
    return ev[np.argsort(ev["start_frame"], kind="stable")]


# ------------------------------------------------------------------ cache helpers
def _content_fingerprint(video, mb=1 << 20):
    """size + mtime_ns + ctime_ns + a hash of the first/last MB of the file (S26/F5).

    mtime alone is not identity: an in-place remux/remux-patch can keep size *and* mtime. ctime
    changes on any write, and the head/tail hash catches a swapped file whose timestamps were
    copied. (An edit confined to the middle with both timestamps forged is the residual gap; it
    needs write access plus deliberate timestamp forgery.)
    """
    try:
        st = os.stat(video)
    except OSError:
        return "missing"
    h = hashlib.blake2b(digest_size=8)
    try:
        with open(video, "rb") as f:
            h.update(f.read(mb))
            if st.st_size > mb:
                f.seek(st.st_size - mb)
                h.update(f.read(mb))
    except OSError:
        pass
    return "%d:%d:%d:%s" % (st.st_size, st.st_mtime_ns, st.st_ctime_ns, h.hexdigest())


def _cfg_key(cfg):
    return [list(x) if isinstance(x, tuple) else x for x in cfg]


def _video_key(video, kind, extra, cfg=CFG):
    """Cache identity: module/detector versions, kind, content fingerprint, cfg and the slice."""
    h = hashlib.blake2b(digest_size=12)
    h.update(("%s:%s:%s" % (MODULE_VERSION, DETECTOR_VERSION, kind)).encode())
    h.update(_content_fingerprint(video).encode())
    h.update(json.dumps(_cfg_key(cfg), sort_keys=True, default=str).encode())
    h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _cache_path(video, kind, extra, cfg=CFG):
    return os.path.join(config.cache_dir(), "audio_%s_%s.npz" % (kind, _video_key(video, kind,
                                                                                  extra, cfg)))


def _save(path, **arrays):
    tmp = config.tmp_path(path)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ public entry points
def whistle_events(video=None, start=0.0, duration=None, cfg=CFG, cache=True, timeout=None):
    """Whistle events for the tactical audio, at 25 fps. Missing video / ffmpeg / audio -> ok=False.

    Video frames follow the organisers' convention: time of frame f (0-based) is f / 25 s
    (R06; the ATD frame_id is video_frame + 1). Never raises: a failed, truncated or timed-out
    decode returns `ok=False` with a token reason and no events, and nothing is cached.
    """
    video = config.asset_path("video") if video is None else video
    if video is None or str(video).endswith(os.sep):
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_video", video)
    video = str(video)
    if not ffmpeg_bin():
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_ffmpeg", video)
    if not os.path.isfile(video):
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_video", video)
    info, reason = probe(video)
    if reason == "no_audio":  # container has no audio stream at all: skip without decoding
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "no_audio", video)
    extra = dict(start=round(float(start), 3),
                 duration=None if duration is None else round(float(duration), 3))
    path = _cache_path(video, "ev", extra, cfg) if cache else None
    if path and os.path.isfile(path):
        try:
            with np.load(path) as z:
                return Track(z["events"], True, "ok", video)
        except (OSError, ValueError, KeyError):
            pass
    try:
        feat = analyse(video, start=start, duration=duration, cfg=cfg, cache=cache, info=info,
                       timeout=timeout)
        ev = detect(feat, cfg)
    except AudioUnavailable as exc:
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, exc.reason, video)
    except Exception:  # the contract is "never an exception"; unanticipated => no audio evidence
        return Track(np.zeros(0, dtype=EVENT_DTYPE), False, "audio_error", video)
    if path:
        try:
            _save(path, events=ev)
        except OSError:
            pass
    return Track(ev, True, "ok", video)


def availability(video=None):
    """Why audio is (or is not) usable, without decoding: ok | no_video | no_ffmpeg | no_audio."""
    try:
        video = config.asset_path("video") if video is None else video
        if not video or not os.path.isfile(str(video)):
            return "no_video"
        if not ffmpeg_bin():
            return "no_ffmpeg"
        _, reason = probe(str(video))
        return reason
    except Exception:
        return "audio_error"


def status_line(video=None, track=None):
    """One-line run-log record, e.g. `audio: ok` / `audio: no_ffmpeg` (S26)."""
    reason = track.reason if track is not None else availability(video)
    return "audio: " + (reason or "ok")


def crowd_from_feat(feat, n_frames=None, t0=0.0):
    """(N,) crowd loudness at 25 Hz, robust-z normalised within each half (R06-c).

    `t0` shifts a Feat built on a relative clock; `analyse` returns absolute times, so it is 0.
    """
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
    extra = dict(start=round(float(start), 3), n=n_frames,
                 duration=None if duration is None else round(float(duration), 3))
    path = _cache_path(video, "crowd", extra, cfg)
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
    out = crowd_from_feat(feat, n_frames, t0=0.0)  # feat.t is already absolute time
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
    Everything is 0 away from a whistle. `gain` is clamped to [0, LOG_ODDS_GAIN], so
    log_odds <= 1.40 and p <= 0.80 for *any* argument: audio alone can never declare D, and
    `p` is not a calibrated posterior (it is 0, not 0.5, off-event). E07 should add `log_odds`
    to its own D log-odds and keep its duration model; the effective gain is returned.
    """
    gain = float(min(max(float(gain), 0.0), LOG_ODDS_GAIN))
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
    log_odds = np.clip(gain * weight, 0.0, LOG_ODDS_GAIN).astype(np.float32)
    p = (1.0 / (1.0 + np.exp(-log_odds.astype(np.float64)))).astype(np.float32)
    p[weight <= 0] = 0.0
    return Evidence(p, log_odds, gate, onset, events, gain)
