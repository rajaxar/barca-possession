"""Constant-provenance guard: every module-level constant in the shipped path must be tagged.

The register lives in ``docs/constant_provenance.json`` and is rendered to ``docs/CONSTANTS.md``.
This test re-derives the constant set from the AST, so a new constant cannot arrive untagged, and it
proves the check is non-vacuous with planted constants (in a temp copy, never in ``solution/``).

Needs no match data.  Run with ``python -m unittest tests.test_constant_provenance -v``.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOLUTION, "scripts"))

import constant_provenance as cp  # noqa: E402

# Built by concatenation so this guard file does not itself contain a GT path token that
# tests/test_rules_guard.py forbids outside scripts/sanity_public.py.
FORBIDDEN_CITATIONS = ("ground truth", "ground-truth", "public window", "public " + "g" + "t",
                       "possession" + "_g" + "t", "video review", "hand label")


def copy_tree(dst):
    for mod in cp.MODULES:
        path = os.path.join(dst, mod)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(os.path.join(SOLUTION, mod), path)
    os.makedirs(os.path.join(dst, "docs"), exist_ok=True)
    shutil.copyfile(os.path.join(SOLUTION, cp.MANIFEST), os.path.join(dst, cp.MANIFEST))


def append(dst, mod, text):
    with open(os.path.join(dst, mod), "a", encoding="utf-8") as fh:
        fh.write(text)


def edit(dst, mod, old, new):
    path = os.path.join(dst, mod)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert old in text, "fixture text not found in " + mod
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old, new, 1))


def reload_manifest(dst):
    with open(os.path.join(dst, cp.MANIFEST), encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(dst, manifest):
    with open(os.path.join(dst, cp.MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


class ConstantProvenance(unittest.TestCase):
    def test_every_module_constant_is_tagged(self):
        problems = cp.check()
        self.assertEqual(problems, [], "constant provenance register is incomplete:\n  "
                         + "\n  ".join(problems))

    def test_doc_is_in_sync_with_json(self):
        with open(os.path.join(SOLUTION, cp.DOC), encoding="utf-8") as fh:
            doc = fh.read()
        self.assertEqual(doc, cp.render(cp.load()),
                         "%s is stale; run `python scripts/constant_provenance.py --write`" % cp.DOC)

    def test_every_tag_class_is_used(self):
        const, fields = cp.counts(cp.load())
        for key in cp.PROVENANCE_TAGS:
            self.assertGreater(const.get(key, 0) + fields.get(key, 0), 0,
                               "no constant or dict field carries the %r tag" % key)
        self.assertGreater(fields.get("match", 0), 0, "no dict field is tagged match-fitted")

    def test_q36_constants_are_registered(self):
        """The values Q36 had to dig for are present and carry the expected provenance class."""
        m = cp.load()["modules"]

        def tag(mod, name):
            return m["possession/%s.py" % mod]["constants"][name]["tag"]

        def field(mod, name, key):
            return m["possession/%s.py" % mod]["constants"][name]["fields"][key]

        # the degrader's ATD run-length quantiles (run lengths, not "noise")
        for name in ("ATD_GAP_Q", "ATD_PRESENT_Q", "ATD_SEG_Q"):
            self.assertEqual(tag("degrade", name), "match", name)
        # the ball low-pass width, on the ball (degrade) and on the players (dgate)
        self.assertEqual(field("degrade", "PARAMS", "smooth_ball_sigma")["tag"], "match")
        self.assertEqual(field("degrade", "PARAMS", "smooth_ball_sigma")["value"], 8.0)
        self.assertEqual(tag("dgate", "LOWPASS_SIGMA"), "match")
        self.assertEqual(m["possession/dgate.py"]["constants"]["LOWPASS_SIGMA"]["value"], 8.0)
        # both growd footprint-guard families: relative canary and absolute cap
        self.assertEqual(tag("growd", "MAX_CHANGED_PER_ELIGIBLE"), "guard")
        self.assertEqual(tag("growd", "MAX_D_SHARE_ADDED"), "guard")
        self.assertEqual(tag("growd", "MAX_MASK_SHARE"), "guard")
        self.assertEqual(tag("growd", "MIN_MASK_SHARE"), "guard")
        # the present-ball share bar
        self.assertEqual(tag("ingap", "MAX_PRESENT_SHARE"), "guard")
        # structure is published, not evidence
        for name in ("HALF_BLOCKS", "KICKOFFS", "PITCH_L", "PITCH_W"):
            self.assertEqual(tag("structure", name), "structure", name)

    def test_no_entry_is_justified_by_this_matchs_answers(self):
        """No source/why may cite ground truth, the public window or video: that is the whole point."""
        bad = []
        for mod, entry in cp.load()["modules"].items():
            for name, c in entry["constants"].items():
                texts = [(" ".join([c.get("source", ""), c.get("why", "")]).lower(), "")]
                texts += [(" ".join([f.get("source", ""), f.get("why", "")]).lower(), "[%r]" % key)
                          for key, f in c.get("fields", {}).items()]
                for text, suffix in texts:
                    for needle in FORBIDDEN_CITATIONS:
                        if needle in text:
                            bad.append("%s:%s%s cites %r" % (mod, name, suffix, needle))
        self.assertEqual(bad, [], "a constant is justified by this match's answers:\n  " + "\n  ".join(bad))


class GuardDetects(unittest.TestCase):
    """The register check fires on planted constants, fields, values and scaffold abuse."""

    def test_planted_untagged_constant_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            self.assertEqual(cp.check(tmp), [], "the copied tree must start clean")
            append(tmp, "possession/dgate.py", "\nPLANTED_UNTAGGED = 1234\n")
            problems = cp.check(tmp)
            hits = [p for p in problems if "PLANTED_UNTAGGED" in p]
            self.assertTrue(hits, "planted constant was not flagged:\n" + "\n".join(problems))
            self.assertIn("untagged constant: possession/dgate.py", hits[0])
            self.assertIn("PLANTED_UNTAGGED = 1234", hits[0])

    def test_planted_untagged_dict_field_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            append(tmp, "possession/degrade.py", "\nPLANTED_TABLE = dict(knob=1.5)\n")
            manifest = reload_manifest(tmp)
            manifest["modules"]["possession/degrade.py"]["constants"]["PLANTED_TABLE"] = dict(
                tag="external", source="none", why="planted")
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            self.assertTrue(any("untagged field: possession/degrade.py.PLANTED_TABLE['knob']" in p
                                for p in problems), "\n".join(problems))

    def test_planted_value_change_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            edit(tmp, "possession/dgate.py", "TF = 50", "TF = 40")
            problems = cp.check(tmp)
            self.assertTrue(any(p.startswith("value drift: possession/dgate.py:TF") for p in problems),
                            "a silent retune was not caught:\n" + "\n".join(problems))

    def test_planted_literal_cannot_hide_behind_scaffold(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            append(tmp, "possession/config.py", "\nPLANTED_HIDDEN = 7\n")
            manifest = reload_manifest(tmp)
            manifest["modules"]["possession/config.py"]["constants"]["PLANTED_HIDDEN"] = dict(
                tag="scaffold", source="path", why="planted")
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            self.assertTrue(any("PLANTED_HIDDEN" in p and "scaffold" in p for p in problems),
                            "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
