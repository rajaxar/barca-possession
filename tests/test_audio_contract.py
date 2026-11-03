"""Fast contract test for `audio.require_ok`: a failed decode must never read as "no whistles".

`whistle_events` is allowed to degrade -- a missing video, ffmpeg or audio stream, a failed or
truncated decode, a timeout -- but it reports that as `ok=False` plus a token reason and an empty
event list.  `require_ok` is the one-line contract every consumer uses instead of reading
`.events` blind: it raises with that reason, or returns the events unchanged.  No match data and
no ffmpeg binary are needed here.
"""
import unittest

import numpy as np

from possession import audio


def _track(n=1, ok=True, reason="ok", video="BAR-ATH_tactical.mp4"):
    ev = np.zeros(n, dtype=audio.EVENT_DTYPE)
    if n:
        ev[0] = (100, 116, 104, 0.6, 3900.0, 4300.0, 400.0, 30.0, 20.0, 10.0, 10.0, 18)
    return audio.Track(ev, ok, reason, video)


class RequireOk(unittest.TestCase):
    def test_failed_decode_raises_with_the_reason(self):
        tr = _track(n=0, ok=False, reason="partial_decode")
        with self.assertRaises(audio.AudioUnavailable) as ctx:
            audio.require_ok(tr, "E38 keeper cue")
        self.assertEqual(ctx.exception.reason, "partial_decode")
        self.assertIn("partial_decode", str(ctx.exception))
        self.assertIn("E38 keeper cue", str(ctx.exception))

    def test_every_failure_token_raises(self):
        """Each degradation reason is a refusal, never an empty-but-ok list."""
        for reason in ("no_video", "no_ffmpeg", "no_audio", "no_ffprobe", "popen_failed",
                       "ffmpeg_failed", "partial_decode", "timeout", "audio_error"):
            tr = _track(n=0, ok=False, reason=reason)
            with self.assertRaises(audio.AudioUnavailable, msg=reason) as ctx:
                audio.require_ok(tr, "diagnostic")
            self.assertEqual(ctx.exception.reason, reason)

    def test_real_failed_track_raises(self):
        """The helper is wired to the real producer, not only to a hand-built namedtuple."""
        tr = audio.whistle_events("/nonexistent/dir/BAR-ATH_tactical.mp4")
        self.assertFalse(tr.ok)
        with self.assertRaises(audio.AudioUnavailable) as ctx:
            audio.require_ok(tr, "diagnostic")
        self.assertEqual(ctx.exception.reason, "no_video")

    def test_success_returns_the_events_and_keeps_the_source(self):
        tr = _track()
        before = (tr.ok, tr.reason, tr.video)
        got = audio.require_ok(tr, "keeper cue")
        self.assertIs(got, tr.events, "the events array must be passed through, not copied")
        self.assertEqual(len(got), 1)
        self.assertEqual(int(got[0]["start_frame"]), 100)
        # provenance survives: the track still names its source and its successful status
        self.assertEqual((tr.ok, tr.reason, tr.video), before)
        self.assertEqual(tr.video, "BAR-ATH_tactical.mp4")

    def test_empty_success_is_not_a_failure(self):
        """A genuinely silent track (ok, zero events) must not raise: 0 is a real answer."""
        tr = _track(n=0, ok=True, reason="ok")
        got = audio.require_ok(tr, "silent check")
        self.assertEqual(len(got), 0)
        self.assertEqual(got.dtype, audio.EVENT_DTYPE)


if __name__ == "__main__":
    unittest.main()
