"""Tests for the whistle detector: synthetic-signal behaviour, 25 fps frame mapping, graceful
degradation (no mp4 / no ffmpeg / no audio track), determinism, and the soft-P(D) contract.

No match data is needed for the core tests; the two integration tests skip when the tactical mp4 or
ffmpeg is absent, so `make test` works on a bare checkout.
"""
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

from possession import audio


# ------------------------------------------------------------------ synthetic audio
SR = audio.SR


def _two_tones(dur=4.0, onset=1.5, length=0.6, f=(3900.0, 4300.0), level=0.25, seed=0):
    """Crowd-like noise plus an optional two-tone (or single-tone) burst: a synthetic whistle."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    x = 0.02 * rng.standard_normal(n).astype(np.float32)
    # a little low-frequency crowd rumble, i.e. energy where the detector must NOT look
    x += (0.01 * np.sin(2 * np.pi * 400 * np.arange(n) / SR)).astype(np.float32)
    a, b = int(onset * SR), int((onset + length) * SR)
    t = np.arange(b - a) / SR
    env = np.hanning(b - a).astype(np.float32)
    for f0 in f:
        x[a:b] += (level * env * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    return x


def detect_array(x, cfg=audio.CFG):
    return audio.detect(audio.features_from_chunks([x], cfg), cfg)


def write_wav(path, x, sr=SR):
    """16-bit mono WAV of a float32 signal: a real file for ffmpeg, exact and cheap to build."""
    pcm = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


class SyntheticDetector(unittest.TestCase):
    def test_two_tone_burst_is_one_event(self):
        ev = detect_array(_two_tones())
        self.assertEqual(len(ev), 1, msg=str(ev))
        e = ev[0]
        self.assertLess(abs(e["f1_hz"] - 3900), 60)
        self.assertLess(abs(e["f2_hz"] - 4300), 60)
        self.assertGreater(e["sep_hz"], 300)
        self.assertGreater(e["c1_db"], audio.C1_MIN_DB)
        self.assertGreater(e["c2_db"], audio.C2_MIN_DB)

    def test_single_tone_is_rejected(self):
        """One ridge = fan whistle / speaker tone; the two-tone rule must drop it."""
        self.assertEqual(len(detect_array(_two_tones(f=(4100.0,)))), 0)

    def test_short_blip_is_rejected(self):
        """0.1 s of tone is shorter than MIN_FRAMES of a real whistle (0.4-1.2 s)."""
        self.assertEqual(len(detect_array(_two_tones(length=0.1))), 0)

    def test_low_frequency_tone_is_rejected(self):
        """Energy below the band (crowd, chants) must not fire."""
        self.assertEqual(len(detect_array(_two_tones(f=(900.0, 1300.0)))), 0)

    def test_event_frames_are_25fps(self):
        x = _two_tones(onset=1.5, length=0.6)
        e = detect_array(x)[0]
        # frame f (0-based) is at f/25 s; the event must start at the burst, not at the file start
        self.assertLessEqual(abs(e["start_frame"] - (1.5 * 25)), 3)
        self.assertLessEqual(abs(e["end_frame"] - (1.5 + 0.6) * 25), 4)
        self.assertLessEqual(e["start_frame"], e["peak_frame"])
        self.assertLessEqual(e["peak_frame"], e["end_frame"])
        self.assertLessEqual(abs(e["dur_s"] - 0.6), 0.12)

    def test_two_runs_identical(self):
        x = _two_tones(seed=7)
        a, b = detect_array(x), detect_array(x)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(list(a["start_frame"]), list(b["start_frame"]))

    def test_chunking_does_not_change_events(self):
        """The detector streams ffmpeg output in ~5 s blocks; block boundaries must not matter."""
        x = _two_tones(dur=12.0, onset=3.0, seed=3)
        whole = detect_array(x)
        pieces = detect_array(np.concatenate([x[:SR * 2], x[SR * 2:SR * 7], x[SR * 7:]]))
        self.assertEqual(whole.tobytes(), pieces.tobytes())
        self.assertGreaterEqual(len(whole), 1)


class GracefulSkip(unittest.TestCase):
    def test_missing_video(self):
        tr = audio.whistle_events("/nonexistent/dir/BAR-ATH_tactical.mp4")
        self.assertFalse(tr.ok)
        self.assertEqual(tr.reason, "no_video")
        self.assertEqual(len(tr.events), 0)
        self.assertEqual(tr.events.dtype, audio.EVENT_DTYPE)

    def test_no_ffmpeg(self):
        old = os.environ.get("MTAH_FFMPEG")
        os.environ["MTAH_FFMPEG"] = "/nonexistent/ffmpeg"
        try:
            self.assertIsNone(audio.ffmpeg_bin())
            tr = audio.whistle_events("/nonexistent/dir/x.mp4")
            self.assertFalse(tr.ok)
            self.assertEqual(tr.reason, "no_ffmpeg")
            self.assertEqual(len(tr.events), 0)
        finally:
            if old is None:
                os.environ.pop("MTAH_FFMPEG", None)
            else:
                os.environ["MTAH_FFMPEG"] = old

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_video_without_audio_track(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "silent.mp4")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc=size=64x64:rate=25:duration=1", "-pix_fmt", "yuv420p",
                            "-y", path], check=True)
            tr = audio.whistle_events(path, cache=False)
            self.assertFalse(tr.ok)
            self.assertEqual(tr.reason, "no_audio")
            self.assertEqual(len(tr.events), 0)

    def test_crowd_envelope_missing_video_is_nan(self):
        env = audio.crowd_envelope("/nonexistent/dir/x.mp4", n_frames=1000)
        self.assertEqual(env.shape, (1000,))
        self.assertTrue(np.all(np.isnan(env)))


class SoftEvidence(unittest.TestCase):
    """P(D) is a bounded prior, never a decision, and it is gated by tracking stillness."""

    def _make(self, n=1500, still=(40, 110), missing=(40, 110)):
        from possession.data import Match
        P = np.zeros((n, 4, 2), np.float32)
        P[..., 0] = np.linspace(0, 100, n)[:, None]
        P[..., 1] = np.linspace(0, 60, n)[:, None]
        V = np.full((n, 4), 2.5, np.float32)
        V[still[0]:still[1]] = 0.4
        B = np.tile(np.array([52.0, 34.0], np.float32), (n, 1))
        B[missing[0]:missing[1]] = np.nan
        half = np.zeros(n, np.int8)
        half[300:] = 1
        return Match(P=P, V=V, T=np.zeros(4, np.int8), B=B, BS=np.zeros(n, np.float32),
                     gk=np.zeros((2, 2), np.int8), graded=half > 0, half=half)

    def _events(self, start=25, peak=27, end=40):
        ev = np.zeros(1, dtype=audio.EVENT_DTYPE)
        ev[0] = (start, peak, end, 0.6, 3900.0, 4300.0, 400.0, 30.0, 20.0, 10.0, 10.0, 18)
        return ev

    def test_bounded_and_zero_away(self):
        m = self._make()
        ev = audio.d_evidence(m, self._events())
        self.assertLessEqual(float(ev.p.max()), audio.P_CAP + 1e-6)
        self.assertGreater(float(ev.p.max()), 0.0)
        self.assertLessEqual(float(ev.log_odds.max()), audio.LOG_ODDS_GAIN + 1e-6)
        self.assertTrue(np.all(ev.p[:25] == 0))
        self.assertTrue(np.all(ev.p[int(25 + audio.BOOST_HORIZON_S * 25) + 2:] == 0))
        self.assertTrue(np.all(ev.log_odds[:25] == 0))

    def test_evidence_decays(self):
        ev = audio.d_evidence(self._make(still=(40, 400), missing=(40, 400)),
                              self._events(start=25, peak=27, end=40))
        lags = [int((d * 25) + 25) for d in (0, 2, 5, 9)]
        vals = [float(ev.log_odds[f]) for f in lags]
        self.assertGreater(vals[0], vals[1])
        self.assertGreater(vals[1], vals[2])
        self.assertGreater(vals[2], vals[3])

    def test_gate_needs_stillness(self):
        ev = self._events()
        live = audio.d_evidence(self._make(still=(200, 260), missing=(200, 260)), ev)
        stopped = audio.d_evidence(self._make(still=(40, 110), missing=(40, 110)), ev)
        self.assertLess(float(live.gate[0]), 0.05)
        self.assertGreater(float(stopped.gate[0]), 0.9)
        self.assertGreater(float(stopped.log_odds.max()), float(live.log_odds.max()))

    def test_no_events_means_no_evidence(self):
        m = self._make()
        empty = np.zeros(0, dtype=audio.EVENT_DTYPE)
        ev = audio.d_evidence(m, empty)
        self.assertEqual(float(ev.p.max()), 0.0)
        self.assertEqual(float(ev.log_odds.max()), 0.0)


class RealAudio(unittest.TestCase):
    """Integration on a 40 s slice of the tactical mp4: determinism of the shipped path and the
    published 1H kickoff whistle. Skipped when the mp4 or ffmpeg is absent."""

    @classmethod
    def setUpClass(cls):
        from possession import config
        cls.video = config.asset_path("video")
        if not audio.ffmpeg_bin() or not os.path.isfile(cls.video):
            raise unittest.SkipTest("tactical mp4 or ffmpeg not available")

    def test_first_half_whistle_and_determinism(self):
        from possession import structure
        tr = audio.whistle_events(self.video, start=430.0, duration=25.0, cache=False)
        self.assertTrue(tr.ok, tr.reason)
        # R06: the 1H whistle ends 0.4-0.7 s before the published kickoff frame
        ko = structure.KICKOFFS[1]
        ends = tr.events["end_frame"]
        self.assertTrue(np.any((ends >= ko - 25) & (ends <= ko)), str(tr.events))  # absolute frames
        again = audio.whistle_events(self.video, start=430.0, duration=25.0, cache=False)
        self.assertEqual(tr.events.tobytes(), again.events.tobytes())


class S26Fixtures(unittest.TestCase):
    """Helpers for the S26 regression tests: quiet WAV fixtures and fake ffmpeg binaries."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="s26audio")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self._env = {}
        for name in ("MTAH_CACHE_DIR", "MTAH_FFMPEG", "MTAH_AUDIO_TIMEOUT_S"):
            self._env[name] = os.environ.get(name)

    def tearDown(self):
        for name, val in self._env.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val

    def wav(self, name="clip.wav", **kw):
        path = os.path.join(self.dir, name)
        write_wav(path, _two_tones(**kw))
        return path

    def setenv(self, **kw):
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def shim(self, name, body, real_ffmpeg=True):
        """A fake ffmpeg at `name`; `body` runs before exec'ing the real binary (if asked)."""
        path = os.path.join(self.dir, name)
        lines = ["#!/bin/sh"]
        if body:
            lines.append(body)
        if real_ffmpeg:
            lines.append('exec %s "$@"' % shlex.quote(shutil.which("ffmpeg") or "/usr/bin/false"))
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
        return path


needs_ffmpeg = unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")


class DecodeContract(S26Fixtures):
    """S26/F1-F3: a decode that is partial, hung or unspawnable must fail closed, never look ok."""

    @needs_ffmpeg
    def test_partial_decode_fails_and_caches_nothing(self):
        """F1: a decode that stops at 4 s of the 10 s the container promises is not a success.

        The fixture is a synthetic 10 s WAV (a real whistle at 2 s) decoded through a shim that
        keeps only the first 4 s of PCM and exits 0 -- the Q09 case A shape, where the *only*
        signal is the sample count (an exit code cannot see it). Nothing may be cached.
        """
        path = self.wav(dur=10.0, onset=2.0)
        keep = 4 * audio.SR * 4
        real = shlex.quote(shutil.which("ffmpeg"))
        ff = self.shim("ffmpeg_truncating",
                       '%s "$@" | head -c %d; exit 0' % (real, keep), real_ffmpeg=False)
        self.setenv(MTAH_FFMPEG=ff, MTAH_CACHE_DIR=os.path.join(self.dir, "cache"))
        tr = audio.whistle_events(path, cache=True)
        self.assertFalse(tr.ok)
        self.assertEqual(tr.reason, "partial_decode", tr.reason)
        self.assertEqual(len(tr.events), 0)
        self.assertEqual(tr.events.dtype, audio.EVENT_DTYPE)
        leftovers = [f for f in os.listdir(os.environ["MTAH_CACHE_DIR"]) if f.endswith(".npz")]
        self.assertEqual(leftovers, [], "a failed decode must not write the cache")

    @needs_ffmpeg
    def test_partial_decode_with_nonzero_exit_is_still_partial(self):
        """F1: short output plus exit 1 is reported as `partial_decode`, not as a plain failure."""
        path = self.wav(dur=10.0, onset=2.0)
        keep = 4 * audio.SR * 4
        real = shlex.quote(shutil.which("ffmpeg"))
        ff = self.shim("ffmpeg_truncating_fail",
                       '%s "$@" | head -c %d; echo "decoder died" >&2; exit 1' % (real, keep),
                       real_ffmpeg=False)
        self.setenv(MTAH_FFMPEG=ff, MTAH_CACHE_DIR=os.path.join(self.dir, "cache"))
        tr = audio.whistle_events(path, cache=True)
        self.assertFalse(tr.ok)
        self.assertEqual(tr.reason, "partial_decode", tr.reason)
        self.assertEqual(len(tr.events), 0)

    @needs_ffmpeg
    def test_decode_failure_before_output_is_a_reason(self):
        """F1/F3: a child that emits nothing and exits 1 -> ok=False, no exception."""
        path = self.wav()
        ff = self.shim("ffmpeg_immediate_fail", "echo 'no such stream' >&2; exit 1",
                       real_ffmpeg=False)
        self.setenv(MTAH_FFMPEG=ff)
        tr = audio.whistle_events(path, cache=False)
        self.assertFalse(tr.ok)
        self.assertEqual(tr.reason, "ffmpeg_failed", tr.reason)
        self.assertEqual(len(tr.events), 0)

    @needs_ffmpeg
    def test_stderr_flood_does_not_deadlock(self):
        """F2: 200 KB of stderr before any PCM used to fill the pipe and hang forever."""
        path = self.wav(dur=8.0, onset=3.0)
        flood = "%s -c \"import sys; sys.stderr.write('x' * 200000); sys.stderr.flush()\"" % (
            shlex.quote(sys.executable))
        ff = self.shim("ffmpeg_stderrflood", flood)
        self.setenv(MTAH_FFMPEG=ff)
        t0 = time.time()
        tr = audio.whistle_events(path, cache=False)
        self.assertLess(time.time() - t0, 60.0)
        self.assertTrue(tr.ok, tr.reason)  # the decode itself is fine, only stderr was noisy
        self.assertEqual(len(tr.events), 1)

    @needs_ffmpeg
    def test_hung_decode_times_out(self):
        """F2: the child is killed at the wall-clock limit and reported as `timeout`."""
        path = self.wav()
        ff = self.shim("ffmpeg_hang", "sleep 60", real_ffmpeg=False)
        self.setenv(MTAH_FFMPEG=ff, MTAH_AUDIO_TIMEOUT_S="1")
        t0 = time.time()
        tr = audio.whistle_events(path, cache=False)
        self.assertLess(time.time() - t0, 15.0)
        self.assertFalse(tr.ok)
        self.assertEqual(tr.reason, "timeout", tr.reason)
        self.assertEqual(len(tr.events), 0)

    @needs_ffmpeg
    def test_unspawnable_binary_degrades(self):
        """F3: an executable file that is not a binary makes Popen raise OSError; catch it."""
        path = self.wav()
        ff = os.path.join(self.dir, "ffmpeg_not_a_binary")
        with open(ff, "w") as f:
            f.write("this is not a program\n")
        os.chmod(ff, 0o755)
        self.setenv(MTAH_FFMPEG=ff)
        tr = audio.whistle_events(path, cache=False)  # must not raise
        self.assertFalse(tr.ok)
        self.assertIn(tr.reason, ("popen_failed", "no_ffmpeg"), tr.reason)
        self.assertEqual(len(tr.events), 0)

    def test_status_line_never_raises(self):
        """The run log line exists for every state, including "no video at all"."""
        self.assertEqual(audio.status_line("/nonexistent/x.mp4"), "audio: no_video")
        self.setenv(MTAH_FFMPEG="/nonexistent/ffmpeg")
        self.assertEqual(audio.status_line("/nonexistent/x.mp4"), "audio: no_video")
        ev = np.zeros(0, dtype=audio.EVENT_DTYPE)
        self.assertEqual(audio.status_line(track=audio.Track(ev, False, "timeout", "v")),
                         "audio: timeout")


class FrameMapping(S26Fixtures):
    """S26/F4: submission row f is at t = f/25 s; no +1, never negative."""

    def _feat(self, t0=1.0, n=30):
        t = t0 + np.arange(n) * 0.032
        return audio.Feat(t=t, c1=np.full(n, 40.0, np.float32), f1=np.full(n, 3900.0, np.float32),
                          c2=np.full(n, 30.0, np.float32), f2=np.full(n, 4300.0, np.float32),
                          crowd_db=np.zeros(n, np.float32))

    def test_emitted_rows_are_round_of_t_times_25(self):
        feat = self._feat()
        ev = audio.detect(feat)
        self.assertEqual(len(ev), 1)
        half = audio.N_FFT / (2.0 * audio.SR)
        e = ev[0]
        self.assertEqual(int(e["start_frame"]), int(round((feat.t[0] - half) * audio.FPS)))
        self.assertEqual(int(e["end_frame"]), int(round((feat.t[-1] + half) * audio.FPS)))
        self.assertEqual(int(e["peak_frame"]), int(round(feat.t[0] * audio.FPS)))
        self.assertEqual(int(e["start_frame"]), 23)   # not 24: the ATD frame_id would be 24
        self.assertLess(int(e["start_frame"]), int(e["end_frame"]))

    def test_rows_are_never_negative(self):
        ev = audio.detect(self._feat(t0=0.0))
        self.assertEqual(len(ev), 1)
        self.assertGreaterEqual(int(ev[0]["start_frame"]), 0)

    @needs_ffmpeg
    def test_synthetic_burst_frame_is_zero_based(self):
        x = _two_tones(onset=1.5, length=0.6)
        e = detect_array(x)[0]
        half = audio.N_FFT / (2.0 * audio.SR)
        # the envelope crosses threshold a little after the burst onset; either way the event can
        # never be a row *after* the burst itself (the old +1 put it at least one row late)
        self.assertLessEqual(int(e["start_frame"]), int(1.5 * audio.FPS) + 1)
        self.assertLessEqual(abs(int(e["start_frame"]) - round((1.5 - half) * audio.FPS)), 4)


class CacheIdentity(S26Fixtures):
    """S26/F5: cfg, module version and content all belong to the key."""

    @needs_ffmpeg
    def test_cfg_is_part_of_the_key(self):
        path = self.wav(dur=8.0, onset=3.0)
        extra = dict(start=0.0, duration=None)
        a = audio._video_key(path, "ev", extra, audio.CFG)
        b = audio._video_key(path, "ev", extra, audio.CFG._replace(min_frames=7))
        self.assertNotEqual(a, b)

    @needs_ffmpeg
    def test_replaced_content_with_restored_mtime_changes_the_key(self):
        path = self.wav(dur=4.0)
        extra = dict(start=0.0, duration=None)
        before = audio._video_key(path, "ev", extra, audio.CFG)
        st = os.stat(path)
        with open(path, "r+b") as f:  # same size, different bytes, mtime restored
            f.seek(200)
            f.write(b"\x00\x01\x02\x03")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(os.stat(path).st_size, st.st_size)
        self.assertEqual(os.stat(path).st_mtime_ns, st.st_mtime_ns)
        self.assertNotEqual(audio._video_key(path, "ev", extra, audio.CFG), before)
        self.assertIn(str(st.st_size), audio._content_fingerprint(path))

    @needs_ffmpeg
    def test_non_default_cfg_cannot_poison_the_event_cache(self):
        path = self.wav(dur=8.0, onset=3.0)
        self.setenv(MTAH_CACHE_DIR=os.path.join(self.dir, "cache"))
        quiet = audio.whistle_events(path, cfg=audio.CFG._replace(min_frames=1000), cache=True)
        self.assertEqual(len(quiet.events), 0)
        default = audio.whistle_events(path, cache=True)
        self.assertTrue(default.ok)
        self.assertEqual(len(default.events), 1)


class SliceInvariance(S26Fixtures):
    """S26/F6: the analysis window start must not move an event (absolute-time STFT grid)."""

    @needs_ffmpeg
    def test_slice_equals_full_run(self):
        path = self.wav(dur=10.0, onset=3.0, length=0.6, seed=11)
        full = audio.whistle_events(path, cache=False)
        self.assertTrue(full.ok, full.reason)
        self.assertEqual(len(full.events), 1)
        want = full.events[0]
        for start in (0.0, 1.0, 2.0, 2.3, 2.5):
            tr = audio.whistle_events(path, start=start, duration=5.0, cache=False)
            self.assertTrue(tr.ok, tr.reason)
            self.assertEqual(len(tr.events), 1, "start=%s" % start)
            got = tr.events[0]
            for field in ("start_frame", "end_frame", "peak_frame"):
                self.assertEqual(int(got[field]), int(want[field]),
                                 "%s moved with start=%s" % (field, start))


class CacheAndDocs(S26Fixtures):
    """S26/F7-F9: the evidence cap, the documented gap rule, the documented memory."""

    def test_gain_is_clamped(self):
        from possession.data import Match
        n = 400
        P = np.zeros((n, 4, 2), np.float32)
        P[..., 0] = np.linspace(0, 100, n)[:, None]
        V = np.full((n, 4), 0.4, np.float32)
        B = np.full((n, 2), np.nan, np.float32)
        m = Match(P=P, V=V, T=np.zeros(4, np.int8), B=B, BS=np.zeros(n, np.float32),
                  gk=np.zeros((2, 2), np.int8), graded=np.ones(n, bool), half=np.zeros(n, np.int8))
        ev = np.zeros(1, dtype=audio.EVENT_DTYPE)
        ev[0] = (10, 12, 15, 0.6, 3900.0, 4300.0, 400.0, 30.0, 20.0, 10.0, 10.0, 18)
        for gain in (2.0, 5.0, 100.0):
            out = audio.d_evidence(m, ev, gain=gain)
            self.assertLessEqual(float(out.p.max()), audio.P_CAP + 1e-6)
            self.assertLessEqual(float(out.log_odds.max()), audio.LOG_ODDS_GAIN + 1e-6)
            self.assertEqual(out.gain, audio.LOG_ODDS_GAIN)
        zero = audio.d_evidence(m, ev, gain=-3.0)
        self.assertEqual(float(zero.log_odds.max()), 0.0)

    def test_max_gap_documented_semantics(self):
        """F8: MAX_GAP=3 bridges 2 silent frames (64 ms), not 3 (96 ms)."""
        T, F = True, False
        self.assertEqual(audio._runs([T, F, T], 3), [(0, 2)])
        self.assertEqual(audio._runs([T, F, F, T], 3), [(0, 3)])
        self.assertEqual(audio._runs([T, F, F, F, T], 3), [(0, 0), (4, 4)])

    def test_memory_claim_is_corrected(self):
        """F9: the docstring must not repeat the old ~40 MB peak-memory claim."""
        doc = audio.__doc__ or ""
        self.assertIn("300 MB", doc)
        self.assertNotIn("40 MB", doc)


class SubmissionLog(S26Fixtures):
    """The run log carries one `audio: ok/reason` line and can never fail the build."""

    def test_audio_line_degrades(self):
        from possession import make_submission
        self.setenv(MTAH_FFMPEG="/nonexistent/ffmpeg")
        line = make_submission.audio_status_line()
        self.assertTrue(line.startswith("audio: "), line)
        self.assertNotIn("\n", line)


if __name__ == "__main__":
    unittest.main()
