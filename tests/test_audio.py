"""Tests for the whistle detector: synthetic-signal behaviour, 25 fps frame mapping, graceful
degradation (no mp4 / no ffmpeg / no audio track), determinism, and the soft-P(D) contract.

No match data is needed for the core tests; the two integration tests skip when the tactical mp4 or
ffmpeg is absent, so `make test` works on a bare checkout.
"""
import os
import shutil
import subprocess
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
