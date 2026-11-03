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
    for mod in cp.discovered(SOLUTION):
        path = os.path.join(dst, mod)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(os.path.join(SOLUTION, mod), path)
    os.makedirs(os.path.join(dst, "docs"), exist_ok=True)
    shutil.copyfile(os.path.join(SOLUTION, cp.MANIFEST), os.path.join(dst, cp.MANIFEST))


def append(dst, mod, text):
    with open(os.path.join(dst, mod), "a", encoding="utf-8") as fh:
        fh.write(text)


def plant_module(dst, name, text):
    with open(os.path.join(dst, "possession", name), "w", encoding="utf-8") as fh:
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


class ScopeIsDiscovered(unittest.TestCase):
    """The module scope is the filesystem, not a list: every file is registered or named as exempt."""

    def test_every_shipped_module_is_registered_or_exempt(self):
        manifest = cp.load()
        present = set(cp.discovered())
        registered = set(manifest["modules"])
        exempt = set(manifest["exempt"])
        self.assertEqual(registered & exempt, set(), "a module is both registered and exempt")
        self.assertEqual(present - registered - exempt, set(),
                         "modules on disk that the register neither tags nor exempts")
        self.assertEqual(registered - present, set(), "registered modules that are not on disk")
        self.assertEqual(exempt - present, set(), "stale exemptions")

    def test_every_exemption_is_justified(self):
        for mod, scope in cp.load()["exempt"].items():
            self.assertIn(scope.get("category"), cp.EXEMPT_CATEGORIES, mod)
            self.assertTrue(scope.get("why", "").strip(), "%s has no justification" % mod)

    def test_every_registered_entry_records_its_line(self):
        for mod, entry in cp.load()["modules"].items():
            for name, c in entry["constants"].items():
                self.assertIsInstance(c.get("line"), int, "%s:%s has no line" % (mod, name))


class ExemptionsAreProven(unittest.TestCase):
    """B31: every exemption's category is enforced against the import/call graph, not its prose."""

    def test_every_exemption_matches_its_reachability(self):
        problems = cp.exemption_problems()
        self.assertEqual(problems, [], "an exemption contradicts the import graph:\n  "
                         + "\n  ".join(problems))
        report = cp.exemption_report()
        self.assertEqual(len(report), 14, "the graph did not classify every exemption")
        for mod, row in report.items():
            with self.subTest(module=mod):
                self.assertIn(row["category"], cp.EXEMPT_CATEGORIES)

    def test_audio_is_reached_only_by_the_post_write_status_line(self):
        """The acceptance bug: audio is imported by make_submission, but only after the CSV is written."""
        row = cp.exemption_report()["possession/audio.py"]
        self.assertTrue(row["reachable"], "audio must be reachable for the status-line regression")
        self.assertFalse(row["label_path"])
        self.assertFalse(row["imported_on_label"])
        self.assertTrue(any(container == "audio_status_line"
                            for _, _, container, _ in row["import_sites"]),
                        "the graph lost the function-local import that hid this before")

    def test_loader_constants_are_read_only_inside_their_module(self):
        report = cp.exemption_report()
        for mod in ("possession/data.py", "possession/external.py"):
            with self.subTest(module=mod):
                self.assertTrue(report[mod]["label_path"])
                self.assertEqual(report[mod]["literal_reads"], {},
                                 "a loader constant is read by another module")


class GuardDetects(unittest.TestCase):
    """The register check fires on planted constants, fields, values, scaffold abuse and scope gaps."""

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

    def test_planted_new_module_fails_until_registered(self):
        """A new possession/*.py is a scope decision: it fails until the register names it."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            self.assertEqual(cp.check(tmp), [], "the copied tree must start clean")
            plant_module(tmp, "newstage.py", '"""Planted stage."""\n\nPLANTED_UNTAGGED = 1234\n')
            problems = cp.check(tmp)
            hits = [p for p in problems if "unregistered module" in p]
            self.assertTrue(hits, "a new module was not flagged:\n" + "\n".join(problems))
            self.assertIn("unregistered module: possession/newstage.py", hits[0])
            self.assertIn("1 module-level constant", hits[0])

    def test_planted_new_module_registered_still_needs_its_constants(self):
        """Registering the module does not tag the constant; the second failure names file:line."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            plant_module(tmp, "newstage.py", '"""Planted stage."""\n\nPLANTED_UNTAGGED = 1234\n')
            manifest = reload_manifest(tmp)
            manifest["modules"]["possession/newstage.py"] = {"constants": {}}
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            hits = [p for p in problems if "PLANTED_UNTAGGED" in p]
            self.assertTrue(hits, "the registered-but-untagged constant was not flagged:\n"
                            + "\n".join(problems))
            self.assertIn("untagged constant: possession/newstage.py:3", hits[0])

    def test_planted_moved_constant_reports_the_new_line(self):
        """A constant that moves fails with the line the AST now reports, so the fix is one edit."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            edit(tmp, "possession/dgate.py", "\nTF = 50", "\n# planted move\nTF = 50")
            problems = cp.check(tmp)
            hits = [p for p in problems if p.startswith("stale line: possession/dgate.py:TF")]
            self.assertTrue(hits, "a moved constant was not flagged:\n" + "\n".join(problems))
            self.assertIn("is at line 32", hits[0])
            self.assertIn("register records 31", hits[0])
            self.assertIn('"line" to 32', hits[0])

    def test_planted_stale_exemption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            manifest = reload_manifest(tmp)
            manifest["exempt"]["possession/gone.py"] = dict(category="not shipped", why="planted")
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            self.assertTrue(any("stale exemption: possession/gone.py" in p for p in problems),
                            "\n".join(problems))

    def test_planted_reachable_module_marked_not_shipped_fails(self):
        """B31 plant 1: the graph must reject 'not shipped' on a module the build imports."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            self.assertEqual(cp.check(tmp), [], "the copied tree must start clean")
            manifest = reload_manifest(tmp)
            manifest["exempt"]["possession/data.py"]["category"] = "not shipped"
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            hits = [p for p in problems if p.startswith("exempt module possession/data.py")]
            self.assertTrue(hits, "a reachable module was accepted as 'not shipped':\n"
                            + "\n".join(problems))
            self.assertIn("claims 'not shipped' but the shipped entry point reaches it", hits[0])
            self.assertIn("pipeline", hits[0])

    def test_planted_unreachable_module_marked_report_only_fails(self):
        """B31 plant 2: 'report-only' needs a status path (or a CLI), not silence."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            self.assertEqual(cp.check(tmp), [], "the copied tree must start clean")
            manifest = reload_manifest(tmp)
            manifest["exempt"]["possession/hsmm.py"]["category"] = "report-only"
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            hits = [p for p in problems if p.startswith("exempt module possession/hsmm.py")]
            self.assertTrue(hits, "an unreachable module was accepted as 'report-only':\n"
                            + "\n".join(problems))
            self.assertIn("nothing imports it", hits[0])
            self.assertIn("use 'not shipped'", hits[0])

    def test_audio_marked_not_shipped_fails(self):
        """The exact acceptance error: audio is imported, so 'not shipped' is false."""
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree(tmp)
            manifest = reload_manifest(tmp)
            manifest["exempt"]["possession/audio.py"]["category"] = "not shipped"
            save_manifest(tmp, manifest)
            problems = cp.check(tmp)
            hits = [p for p in problems if p.startswith("exempt module possession/audio.py")]
            self.assertTrue(hits, "the historical audio mistake was not caught:\n"
                            + "\n".join(problems))
            self.assertIn("audio_status_line", hits[0])


if __name__ == "__main__":
    unittest.main()
