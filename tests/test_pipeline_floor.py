"""B14: pin the short-spell floor to **every** return path of ``pipeline.build_labels``.

D02/Q27 found the failure mode that this test exists for: the floor written but never wired (E14 ->
B01), and the risk that a later patch moves ``postprocess.apply`` inside ``if dgate.ENABLED:`` so the
gate-off and add-ons-off builds silently lose it again.  A hand check does not survive the next patch.

The synthetic match and pass-through stages below keep the test label-free, model-free and fast: they
exercise the pipeline's control flow, not the football stages.  The floor itself is the real
``postprocess.apply``; the only synthetic thing is the sequence it receives.
"""
import contextlib
import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from possession import baseline_plus, dgate, growd, ingap, pipeline, postprocess, structure, subgap

SOLUTION = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHIPPED_CSV = os.path.join(SOLUTION, "outputs", "submission.csv")

# A known sub-floor spell: 10 frames = 0.4 s, well under postprocess.DEFAULT["min_spell"] (38 f).
LONG, SHORT = 100, 10
LABELS = (["H"] * LONG + ["A"] * SHORT + ["H"] * LONG + ["D"] * 190)
N = len(LABELS)
SHORT_SPAN = slice(LONG, LONG + SHORT)


def _fake_match():
    """Only the attributes ``build_labels`` touches (``half``) plus those its call sites read eagerly."""
    z = np.zeros((N, 2))
    return SimpleNamespace(
        half=np.ones(N, np.int8),
        B=z,
        P=np.zeros((N, 2)),
        V=np.zeros((N, 2)),
        T=np.zeros(N),
    )


@contextlib.contextmanager
def _stages(labels=LABELS):
    """Replace every stage with a pass-through; return the list of stage calls actually made."""
    calls = []
    z = np.zeros(N)
    dead = np.zeros(N, bool)

    def dgate_apply(lab, *a, **k):
        calls.append("dgate")
        return np.asarray(lab).copy(), dead, {}

    def growd_apply(lab, *a, **k):
        calls.append("growd")
        return np.asarray(lab).copy(), {}

    def ingap_apply(lab, *a, **k):
        calls.append("ingap")
        return np.asarray(lab).copy()

    def subgap_apply(lab, *a, **k):
        calls.append("subgap")
        return np.asarray(lab).copy(), {}

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(structure, "structure",
                                              return_value=SimpleNamespace(forced_dead_mask=dead)))
        stack.enter_context(mock.patch.object(baseline_plus, "predict",
                                              return_value=np.array(labels).copy()))
        stack.enter_context(mock.patch.object(dgate, "apply", side_effect=dgate_apply))
        stack.enter_context(mock.patch.object(growd, "apply", side_effect=growd_apply))
        stack.enter_context(mock.patch.object(pipeline, "ingap_ph", return_value=z))
        stack.enter_context(mock.patch.object(ingap, "apply", side_effect=ingap_apply))
        stack.enter_context(mock.patch.object(ingap, "footprint", return_value={}))
        stack.enter_context(mock.patch.object(dgate, "frame_pd", return_value=z))
        stack.enter_context(mock.patch.object(subgap, "apply", side_effect=subgap_apply))
        yield calls


def _build(flags=()):
    """Run ``build_labels`` with the given module flags temporarily changed."""
    with _stages() as calls, contextlib.ExitStack() as stack:
        for module, flag in flags:
            stack.enter_context(mock.patch.object(module, flag[0], flag[1]))
        out = pipeline.build_labels(match=_fake_match(), models=("dm", "cols", "bf"),
                                    log=None, guard=False, forced=np.zeros(N, bool))
    return np.asarray(out), calls


class FloorOnEveryReturnPath(unittest.TestCase):
    """The floor must be the last label write whichever branch produced the labels."""

    def _assert_floor(self, out):
        np.testing.assert_array_equal(out[SHORT_SPAN], ["D"] * SHORT,
                                      "sub-1.5 s spell was not absorbed to D (floor skipped)")

    def test_default_flags(self):
        out, calls = _build()
        self._assert_floor(out)
        self.assertEqual(calls, ["dgate", "growd", "ingap", "subgap"])

    def test_gate_disabled(self):
        out, calls = _build([(dgate, ("ENABLED", False))])
        self._assert_floor(out)
        self.assertEqual(calls, [], "gate-off path ran a stage it should skip")
        np.testing.assert_array_equal(out[LONG + SHORT:2 * LONG + SHORT], ["H"] * LONG)  # nothing else moved

    def test_add_ons_disabled(self):
        out, calls = _build([(growd, ("ENABLED", False)), (ingap, ("ENABLED", False)),
                             (subgap, ("ENABLED", False))])
        self._assert_floor(out)
        # dgate still owns the branch.  growd.apply is still *called* (pipeline invokes it
        # unconditionally inside the gate branch) but its ENABLED flag makes it a no-op; ingap and
        # subgap are skipped by the pipeline's own flags.
        self.assertEqual(calls, ["dgate", "growd"])

    def test_postprocess_disabled(self):
        out, calls = _build([(postprocess, ("ENABLED", False))])
        np.testing.assert_array_equal(out[SHORT_SPAN], ["A"] * SHORT,
                                      "floor ran with postprocess.ENABLED = False")
        self.assertEqual(calls, ["dgate", "growd", "ingap", "subgap"])


class MergeDuplicationIsVisible(unittest.TestCase):
    """E83: the m=20 merge exists twice.  ``ingap.gapdec`` re-applies Baseline+'s merge *after* the
    in-gap decode, so on the shipped path the second copy is the one that decides; the add-ons-off
    path falls back to Baseline+'s copy.  Pinned here, not refactored (B14 scope).
    """

    def test_ingap_reapplies_the_same_merge(self):
        m = inspect.signature(ingap.gapdec).parameters["m"].default
        self.assertEqual(m, baseline_plus.DEFAULT_PARAMS["m"])
        self.assertEqual(m, 20)

    def test_gapdec_merges_even_with_no_missing_ball_gaps(self):
        # Baseline+'s merge is redundant when ingap runs: gapdec re-merges each half at line 92,
        # unconditionally, not only inside the ball-missing runs it re-decodes.
        labels = np.array(["H"] * LONG + ["A"] * SHORT + ["H"] * LONG)
        match = _fake_match()
        match.half = np.ones(len(labels), np.int8)
        match.B = np.zeros((len(labels), 2))
        out = ingap.gapdec(labels, match.B, match.half, np.zeros(len(labels)))
        np.testing.assert_array_equal(out[LONG:LONG + SHORT], ["H"] * SHORT)

    def test_baseline_merge_is_idempotent(self):
        # Re-running the same merge cannot change an answer the first pass already produced, which is
        # why gapdec's second copy is safe; the fixture also proves this merge actually does something.
        lab = np.array([0] * LONG + [1] * SHORT + [0] * LONG, np.int8)
        once = baseline_plus._merge_short(lab, 20)
        np.testing.assert_array_equal(baseline_plus._merge_short(once, 20), once)
        self.assertFalse(np.array_equal(once, lab), "fixture is not short enough to merge")


class ShippedShape(unittest.TestCase):
    """Cheap label-only check of the frozen artifact: 370 spells / 138 D stretches / 40,462 X."""

    def test_shipped_csv_shape(self):
        if not os.path.isfile(SHIPPED_CSV):
            self.skipTest("outputs/submission.csv absent")
        with open(SHIPPED_CSV) as fh:
            self.assertEqual(fh.readline().strip(), "frame,label")
            labels = np.array([line.rsplit(",", 1)[1].strip() for line in fh])
        self.assertEqual(len(labels), 199319)
        self.assertEqual(int((labels == "X").sum()), 40462)

        # X is written only outside the graded halves, so the graded runs are the non-X stretches.
        mask = (labels != "X").astype(np.int8)
        edges = np.diff(np.r_[0, mask, 0])
        runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        self.assertEqual(postprocess.n_spells(labels, runs), 370)
        self.assertEqual(sum(1 for _, _, c in postprocess.segments(labels, runs) if c == 2), 138)


if __name__ == "__main__":
    unittest.main()
