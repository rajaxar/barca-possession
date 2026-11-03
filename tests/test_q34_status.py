"""Q34: every silent substitution that can change a reported number must record a visible status.

The audit is `research/Q34_silent_substitutions.md`. Two earlier bugs were quiet substitutions,
not crashes: E14's matched-tuple swap (which produced the retired 2-of-474 count) and B21's inline keeper copy (0 of 1,252 frames).
These tests pin the status for the five paths the audit classified as (b) -- a substitution that
reaches a reported number:

1. ``gk_possession.detect`` records where its whistle events came from (audio / audio failure /
   caller-supplied), so "0 episodes because no audio" cannot be read as "0 whistles".
2. ``audio.d_evidence`` carries the same source on its ``Evidence``.
3. ``walkthrough_numbers.check`` compares *read* (digest-parsed) numbers too, so a corrected digest
   cannot leave a stale figure in the notebook with `--check` green.
4. ``dproxy.adoption_report`` records the ablation's base-rule source instead of silently using the
   raw missing-ball mask for a proxy object that does not declare one.
5. ``dev_v1.reference_results`` logs a cache hit with the code provenance, so a stale reference
   table is visible in the run log.

None of these changes a label, a stage, or the submission CSV.
"""
import contextlib
import importlib.util
import io
import json
import os
import pickle
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

from possession import audio, dev_v1, dproxy
from possession import gk_possession as gk
from possession.data import Match
from possession.structure import PITCH_L, PITCH_W, Structure

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 1200


def _match(n=N):
    """Synthetic ATD-shaped Match: two keepers, 20 outfielders walking, no tracked ball."""
    P = np.full((n, 46, 2), np.nan, np.float32)
    V = np.full((n, 46), np.nan, np.float32)
    T = np.array([0] * 23 + [1] * 23, np.int8)
    half = np.ones(n, np.int8)
    for f in range(n):
        P[f, 22] = (5.0, PITCH_W / 2)
        P[f, 45] = (PITCH_L - 5.0, PITCH_W / 2)
        V[f, 22] = V[f, 45] = 0.0
        for i in range(11):
            P[f, i] = (20 + 4 * i, 5 + 3 * i)
            P[f, 23 + i] = (85 - 4 * i, 5 + 3 * i)
            V[f, i] = V[f, 23 + i] = 1.2
    B = np.full((n, 2), np.nan, np.float32)  # missing ball: the missing-ball proxy's domain
    return Match(P, V, T, B, np.zeros(n, np.float32), np.array([[22, 45], [22, 45]]),
                 half > 0, half)


def _st(n=N):
    return Structure(halves={1: (0, n - 1)}, kickoffs={1: 0},
                     attack_dir=np.array([[1, -1], [-1, 1]], np.int8),
                     forced_dead_mask=np.zeros(n, bool))


def _empty_events():
    return np.zeros(0, dtype=audio.EVENT_DTYPE)


class GkEventsSource(unittest.TestCase):
    """(b1) gk_possession must say whether its whistle input was the audio or a failure."""

    def _detect(self, whistle):
        m = _match()
        with mock.patch.object(audio, "whistle_events", whistle):
            return gk.detect(m, st=_st())

    def test_detector_error_is_recorded_not_hidden(self):
        def boom(*a, **k):
            raise RuntimeError("decoder exploded")
        gp = self._detect(boom)
        self.assertEqual(gp.info["events_source"], "audio_error")
        self.assertEqual(gp.info["n_events"], 0)
        # The detector still produced a usable result -- the point is that it is labelled.
        self.assertIsInstance(gp.info["n_episodes"], int)

    def test_audio_absent_keeps_the_audio_tracks_own_token(self):
        gp = self._detect(lambda *a, **k: audio.Track(_empty_events(), False, "no_ffmpeg", None))
        self.assertEqual(gp.info["events_source"], "no_ffmpeg")
        self.assertEqual(gp.info["n_events"], 0)

    def test_real_audio_path_is_recorded_as_audio(self):
        gp = self._detect(lambda *a, **k: audio.Track(_empty_events(), True, "ok", None))
        self.assertEqual(gp.info["events_source"], "audio")

    def test_caller_supplied_events_are_labelled_given(self):
        gp = gk.detect(_match(), st=_st(), events=_empty_events())
        self.assertEqual(gp.info["events_source"], "given")
        self.assertEqual(gp.info["n_events"], 0)

    def test_source_is_visible_next_to_the_episode_count(self):
        """The number a diagnostic reports (n_episodes) sits beside the input it was measured with."""
        gp = self._detect(lambda *a, **k: audio.Track(_empty_events(), False, "no_video", None))
        self.assertIn("events_source", gp.info)
        self.assertIn("n_events", gp.info)
        self.assertEqual(gp.info["events_source"], "no_video")


class DEvidenceSource(unittest.TestCase):
    """(b2) audio.d_evidence: zero evidence must say whether there was no audio or no whistle."""

    def test_missing_audio_is_visible_on_the_evidence(self):
        m = _match()
        with mock.patch.object(audio, "whistle_events",
                               lambda *a, **k: audio.Track(_empty_events(), False, "no_ffmpeg", None)):
            ev = audio.d_evidence(m)
        self.assertEqual(ev.source, "no_ffmpeg")
        self.assertEqual(float(np.abs(ev.p).max()), 0.0)

    def test_given_events_are_labelled_given(self):
        ev = audio.d_evidence(_match(), events=_empty_events())
        self.assertEqual(ev.source, "given")

    def test_old_six_field_construction_still_works(self):
        ev = audio.Evidence(np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(0, np.float32),
                            np.zeros(0, np.int32), _empty_events(), 1.0)
        self.assertEqual(ev.source, "given")


def _load_walkthrough_numbers():
    path = os.path.join(SOLUTION, "scripts", "walkthrough_numbers.py")
    spec = importlib.util.spec_from_file_location("q34_walkthrough_numbers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ManifestCheckCoversReadNumbers(unittest.TestCase):
    """(b3) `--check` must fail when a digest-parsed number moves, not only a recomputed one."""

    def setUp(self):
        self.wn = _load_walkthrough_numbers()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "numbers.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _entry(self, value, recomputed, source="research/E99_floor_recheck.md"):
        return {"value": value, "display": str(value), "source": source, "command": "",
                "recomputed": recomputed, "tolerance": 0.0}

    def _write(self, entries):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"schema": 1, "entries": entries}, fh)

    def test_changed_read_number_fails_the_check(self):
        old = {"floor_tp": self._entry(2.0, False)}
        self._write(old)
        new = {"entries": {"floor_tp": self._entry(84.0, False)}}
        with self.assertRaises(SystemExit) as cm:
            self.wn.check(self.path, new)
        msg = str(cm.exception)
        self.assertIn("stored numbers drifted", msg)
        self.assertIn("read: research/E99_floor_recheck.md", msg)

    def test_unchanged_manifest_passes(self):
        old = {"read_num": self._entry(0.0208, False), "recomp": self._entry(370.0, True, "csv")}
        self._write(old)
        new = {"entries": old}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.wn.check(self.path, new)
        self.assertIn("no drift", buf.getvalue())
        self.assertIn("1 read", buf.getvalue())


class DproxyBaseRuleSource(unittest.TestCase):
    """(b4) the ablation base rule must not be substituted without saying so."""

    def _report(self, declare_base):
        n = N
        m = _match(n)
        cue = np.zeros(n, bool)
        for start in (300, 600, 900):
            cue[start:start + 25] = True
        info = dict(kind="missing", graded=m.graded, min_run_s=2.0, still_v=1.5,
                    coverage_missing=1.0)
        if declare_base:
            info["base_rule"] = cue.copy()
        proxy_obj = dproxy.DProxy(cue.copy(), [], [], np.zeros(n, np.float32), info)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return dproxy.adoption_report(cue, match=m, proxy_obj=proxy_obj, n_boot=20)

    def test_missing_base_rule_is_recorded_and_warned(self):
        r = self._report(declare_base=False)
        self.assertEqual(r["base_rule_source"], "fallback_missing_mask")
        self.assertIn("ablation_base_rule_not_from_proxy_info", r["warnings"])
        self.assertIn("base rule: fallback_missing_mask", dproxy.format_report(r))

    def test_declared_base_rule_is_not_flagged(self):
        r = self._report(declare_base=True)
        self.assertEqual(r["base_rule_source"], "proxy_info")
        self.assertNotIn("ablation_base_rule_not_from_proxy_info", r["warnings"])
        self.assertNotIn("base rule:", dproxy.format_report(r))


class ReferenceCacheHitIsLogged(unittest.TestCase):
    """(b5) a dev-v1 reference cache hit must print the file and the code provenance."""

    def test_cache_hit_logs_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "reference_q34test.pkl"), "wb") as fh:
                pickle.dump({"smooth50": "sentinel"}, fh)
            with mock.patch.object(dev_v1, "_dir", lambda: tmp), \
                    mock.patch.object(dev_v1, "_stamp", lambda: "q34test"):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    res = dev_v1.reference_results()
        out = buf.getvalue()
        self.assertEqual(res, {"smooth50": "sentinel"})
        self.assertIn("reference cache HIT", out)
        self.assertIn("reference_q34test.pkl", out)
        self.assertIn("dev_v1: possession=", out)
        self.assertIn("git=", out)


if __name__ == "__main__":
    unittest.main()
