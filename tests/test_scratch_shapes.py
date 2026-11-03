"""Static scratch guard: the two shapes that silently substituted a plausible number (Q35/B27).

Scope: ``scratch/*.py`` and ``scratch/*/*.py``, excluding vendored copies (``pkg/``, ``repo/``,
``clone/``, ``snapshot*``, ``mutant/``, ``perturbed/``, ``site-packages/``, ``.venv/``) -- those are
frozen snapshots of the shipped tree, not live scripts.  The guard is a tripwire, not a refactor:
every hit prints ``file:line``, the shape and the fix, so the next wrapper cannot reintroduce what
Q35 fixed by hand.

Shape 1: element 0 of a ``spell_diff(...)["matched"]`` tuple is the **GT** span and element 1 is the
prediction (``evaluate.py:138``).  The exact historical bug destructures the first element inline,
``for (a, b, c), _g, _v in d["matched"]``, and tests predicted spells against the GT set -- the
``2 of 474`` count, corrected by E99 to 84/464 (86/474) at .3 and 43 at .5.  The shape is allowed
only in the two historical artifacts in :data:`MATCH_ALLOW`, which keep it to document the bug.

Shape 2: ``.events`` read from a ``whistle_events`` track without an ``ok`` check.  A failed decode
returns an empty event list with ``ok=False``; reading ``.events`` turns "no audio" into a
plausible "no whistles".  The fix is ``audio.require_ok(track, what)`` (or an explicit
``assert track.ok``).  The Q09 failure/cache harnesses are allowed: there the ``ok`` flag *is* the
measurement, not a guard.

Run with ``python -m unittest tests.test_scratch_shapes``.
"""
import ast
import glob
import os
import re
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)
ROOT = os.path.dirname(SOLUTION)

# A for-target that destructures the first thing it unpacks: the E14/E99 element-0 shape.
MATCH_TARGET = re.compile(r"\bfor\s*\([^()]*\)\s*,\s*[\w.]+\s*,\s*[\w.]+\s+in\b")
# `tr = ... whistle_events(...)` binds a Track to `tr`; a direct `whistle_events(...).events`
# is caught on its own line.
TRACK_ASSIGN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*[^\n#]*whistle_events\s*\(")
EVENTS_ATTR = re.compile(r"\b([A-Za-z_]\w*)\.events\b")
OK_NEARBY = re.compile(
    r"\brequire_ok\s*\(|"
    r"\bassert\s+[\w.]*\.ok\b|"
    r"\bif\s+not\s+[\w.]*\.ok\b|"
    r"\bif\s+[\w.]*\.ok\b|"
    r"\braise\s+[^\n]*\.reason\b")

SKIP_DIRS = {"pkg", "repo", "clone", "mutant", "perturbed", "site-packages", ".venv",
             "__pycache__"}

MATCH_ALLOW = {
    os.path.join("scratch", "E14", "short_match.py"):
        "historical artifact: E99 corrected the swap (84/464 at .3, 43 at .5; 86/474 on these "
        "labels) -- kept as found, never rerun for a number",
    os.path.join("scratch", "E99", "analyze.py"):
        "deliberate reproduction of E14's undercount; mechanism.py carries the corrected "
        "element-1 keying",
}
EVENTS_ALLOW = {
    os.path.join("scratch", "Q09", "ffmpeg_tests.py"):
        "Q09 failure harness: tr.ok/reason IS the observation; a forced failure must not raise",
    os.path.join("scratch", "Q09", "cache_tests.py"):
        "Q09 cache harness: an empty or stale event list is the measurement",
}


class Hit:
    __slots__ = ("path", "line", "shape", "text", "fix")

    def __init__(self, path, line, shape, text, fix):
        self.path = path
        self.line = line
        self.shape = shape
        self.text = text
        self.fix = fix

    def __str__(self):
        return '{}:{}: [{}] "{}" -> {}'.format(self.path, self.line, self.shape,
                                               self.text.strip(), self.fix)


def _vendored(rel):
    parts = rel.split(os.sep)[1:-1]  # below scratch/, above the file
    return any(p in SKIP_DIRS or p.startswith("snapshot") for p in parts)


def default_files(root=ROOT):
    """`scratch/*.py` + `scratch/*/*.py`, relative to `root`, without vendored snapshots."""
    rels = []
    for pattern in (os.path.join("scratch", "*.py"), os.path.join("scratch", "*", "*.py")):
        for path in glob.glob(os.path.join(root, pattern)):
            rel = os.path.relpath(path, root)
            if _vendored(rel):
                continue
            rels.append(rel)
    return sorted(set(rels))


def track_lines(lines):
    """First assignment line of every name bound from a `whistle_events(...)` call."""
    out = {}
    for i, line in enumerate(lines, 1):
        m = TRACK_ASSIGN.match(line)
        if m and m.group(1) not in out:
            out[m.group(1)] = i
    return out


def string_lines(text):
    """Line numbers inside a string literal: prose and embedded snippets are not code.

    Multi-line literals (docstrings, embedded scripts) are skipped whole.  A one-line literal is
    skipped only when it is the entire stripped line -- a line such as ``d["matched"]`` is code.
    """
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if (node.end_lineno or node.lineno) > node.lineno:
            out.update(range(node.lineno, node.end_lineno + 1))
            continue
        raw = lines[node.lineno - 1].strip()
        if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
            out.add(node.lineno)
    return out


def scan_text(text, relpath="scratch/probe.py"):
    """Hits in one file's text; ``relpath`` only labels the report and the allowlists."""
    lines = text.splitlines()
    tracks = track_lines(lines)
    in_string = string_lines(text)
    hits = []
    for i, line in enumerate(lines, 1):
        if i in in_string:
            continue
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # shape 1: inline destructuring of element 0 of a matched tuple
        if relpath not in MATCH_ALLOW and MATCH_TARGET.search(line):
            near = " ".join(lines[max(0, i - 1):min(len(lines), i + 2)])
            if "matched" in near:
                hits.append(Hit(relpath, i, "matched[0]", line,
                                "element 0 is the GT span; use element 1 (or key GT-side "
                                "explicitly, as E99/mechanism.py and E88/validate.py do)"))
        # shape 2: `.events` from a whistle track without ok/require_ok nearby
        if relpath in EVENTS_ALLOW:
            continue
        direct = "whistle_events" in line and ".events" in line
        bases = [m.group(1) for m in EVENTS_ATTR.finditer(line) if m.group(1) in tracks]
        if not (direct or bases):
            continue
        if bases:
            # anchored anywhere between the track assignment and this use (plus a look-ahead),
            # so one `require_ok`/`assert track.ok` at the top covers every later cue
            first = min(tracks[b] for b in bases)
            window = "\n".join(lines[first - 1:min(len(lines), i + 8)])
        else:
            window = "\n".join(lines[max(0, i - 9):min(len(lines), i + 8)])
        if OK_NEARBY.search(window):
            continue
        hits.append(Hit(relpath, i, "events-without-ok", line,
                        "use audio.require_ok(track, what) or assert track.ok; a failed decode "
                        "must not read as zero whistles"))
    return hits


def scan(root=ROOT, rels=None):
    rels = default_files(root) if rels is None else rels
    hits = []
    for rel in rels:
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        hits += scan_text(text, rel)
    return hits


def format_hits(hits):
    if not hits:
        return "no scratch shape hits"
    return ("scratch shape(s) found (guard; see tests/test_scratch_shapes.py):\n  "
            + "\n  ".join(str(h) for h in hits))


class ScratchShapes(unittest.TestCase):
    def test_current_scratch_has_no_bad_shapes(self):
        hits = scan()
        self.assertEqual(hits, [], format_hits(hits))

    def test_allowlisted_artifacts_are_named(self):
        """The allowlists are explicit, not a blanket skip of the two files' directories."""
        self.assertIn(os.path.join("scratch", "E14", "short_match.py"), MATCH_ALLOW)
        self.assertIn(os.path.join("scratch", "E99", "analyze.py"), MATCH_ALLOW)
        self.assertIn("E99", MATCH_ALLOW[os.path.join("scratch", "E14", "short_match.py")])
        self.assertIn(os.path.join("scratch", "Q09", "ffmpeg_tests.py"), EVENTS_ALLOW)


class ScratchShapesDetect(unittest.TestCase):
    """The guard is non-vacuous: it fails on planted copies of both shapes."""

    def test_planted_violations_fail_in_a_scratch_copy(self):
        rels = default_files()
        with tempfile.TemporaryDirectory() as tmp:
            for rel in rels:
                dst = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, rel), dst)
            self.assertEqual(scan(tmp, rels), [], "scratch copy must start clean")

            victim = os.path.join(tmp, "scratch", "Q99", "probe.py")
            os.makedirs(os.path.dirname(victim), exist_ok=True)
            with open(victim, "w", encoding="utf-8") as fh:
                fh.write(
                    "import numpy as np\n"
                    "from possession import audio, evaluate\n"
                    "d = evaluate.spell_diff(pred, ref, runs, iou=0.3)\n"
                    "mp = {(a, b) for (a, b, c), _g, _v in d['matched']}\n"
                    "tr = audio.whistle_events()\n"
                    "ev = tr.events\n")

            hits = scan(tmp, rels + [os.path.join("scratch", "Q99", "probe.py")])
            planted = [h for h in hits if h.path == os.path.join("scratch", "Q99", "probe.py")]
            self.assertTrue(planted, "planted shapes were not flagged:\n" + format_hits(hits))
            shapes = {h.shape for h in planted}
            self.assertEqual(shapes, {"matched[0]", "events-without-ok"}, format_hits(planted))
            self.assertTrue(any(h.line == 4 for h in planted), format_hits(planted))
            self.assertTrue(any(h.line == 6 for h in planted), format_hits(planted))

    def test_ok_check_or_require_ok_makes_the_events_shape_clean(self):
        for body in ("assert tr.ok, tr.reason\n    ev = tr.events\n",
                     "ev = audio.require_ok(tr, 'cue')\n",
                     "if not tr.ok:\n    raise RuntimeError(tr.reason)\nev = tr.events\n"):
            text = "tr = audio.whistle_events()\n" + body
            self.assertEqual(scan_text(text), [], "not accepted:\n" + text)

    def test_events_shape_is_not_flagged_without_a_track(self):
        self.assertEqual(scan_text("x = box.events\n"), [])

    def test_element_one_keying_is_clean(self):
        self.assertEqual(scan_text(
            "d = evaluate.spell_diff(p, r, runs, iou=0.3)\n"
            "pred = {(p[0], p[1]) for _g, p, _v in d['matched']}\n"), [])


if __name__ == "__main__":
    unittest.main()
