"""B28: the two compliance surfaces run under ``make test`` as well as ``make freeze-check``.

The public page is an optional deliverable, so its test skips cleanly when the HTML is absent -- the
same rule the freeze target follows. The manifest is always checked: it is tracked, so a shape
regression there is a regression in the repository.
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SOLUTION, "scripts"))

import freeze_surfaces as fs  # noqa: E402

HAVE_PAGE = os.path.isfile(fs.PUBLIC_HTML)


class ManifestShape(unittest.TestCase):
    def _copy(self, tmp):
        path = os.path.join(tmp, "numbers.json")
        with open(fs.MANIFEST, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc, path

    def _write(self, doc, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
            fh.write("\n")

    def test_current_manifest_is_numbers_and_strings(self):
        report = fs.check_manifest()
        self.assertTrue(report["tracked"], report)
        self.assertGreater(report["entries"], 0)
        with open(fs.MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)
        for entry in manifest["entries"].values():
            self.assertNotIsInstance(entry["value"], (list, dict))

    def test_coordinate_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc, path = self._copy(tmp)
            key = sorted(doc["entries"])[0]
            doc["entries"][key]["value"] = [[12.31, 45.62], [12.44, 45.80]]
            self._write(doc, path)
            with self.assertRaises(fs.SurfaceFailure) as caught:
                fs.check_manifest(path)
            message = str(caught.exception)
            self.assertIn("entries.%s.value is a list" % key, message)
            self.assertIn("coordinate array is match data", message)

    def test_non_public_figure_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc, path = self._copy(tmp)
            key = sorted(doc["entries"])[0]
            doc["entries"][key]["note"] = "see outputs/frame_11185.png"
            self._write(doc, path)
            with self.assertRaises(fs.SurfaceFailure) as caught:
                fs.check_manifest(path)
            self.assertIn("outputs/frame_11185.png", str(caught.exception))
            self.assertIn("public placeholder", str(caught.exception))

    def test_unknown_figure_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc, path = self._copy(tmp)
            key = sorted(doc["entries"])[0]
            doc["entries"][key]["note"] = "see fig_a_frame_from_our_match"
            self._write(doc, path)
            with self.assertRaises(fs.SurfaceFailure) as caught:
                fs.check_manifest(path)
            self.assertIn("fig_a_frame_from_our_match", str(caught.exception))


class AbsentPage(unittest.TestCase):
    def test_absent_page_is_a_clean_skip(self):
        report = fs.check_public_page(os.path.join(SOLUTION, "notebooks", "no-such-page.html"))
        self.assertEqual(report["status"], "absent")
        self.assertEqual(report["checks"], 0)


@unittest.skipUnless(HAVE_PAGE, "public walkthrough page not built")
class PublicPage(unittest.TestCase):
    def test_current_page_passes_every_audit_check(self):
        report = fs.check_public_page()
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["checks"], fs.AUDIT_CHECKS)

    def test_planted_frame_number_fails_the_audit(self):
        with open(fs.PUBLIC_HTML, encoding="utf-8") as fh:
            doc = fh.read()
        planted = doc.replace(
            "</main>",
            "<p>the ball leaves frame 11185 here</p></main>", 1)
        with tempfile.TemporaryDirectory() as tmp:
            victim = os.path.join(tmp, "planted.html")
            with open(victim, "w", encoding="utf-8") as fh:
                fh.write(planted)
            with self.assertRaises(fs.SurfaceFailure) as caught:
                fs.check_public_page(victim)
            message = str(caught.exception)
            self.assertIn("no-frame-numbers", message)
            self.assertIn("11185", message)


if __name__ == "__main__":
    unittest.main()
