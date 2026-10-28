"""Static rules guard (GUARDRAILS 1 and 2). Needs no match data; fails on patterns that would break a rule.

* The public-window GT is read only by scripts/sanity_public.py.
* No game-frame literal (11185..190197) in possession/ outside structure.py, unless the statement carries
  a `# frame-ok: <citation or reason>` comment.
* No network imports in possession/ or scripts/.
* No data files (by extension) and nothing over 200 KB among tracked or not-ignored files.
"""
import ast
import os
import re
import subprocess
import unittest

SOLUTION = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SKIP_DIRS = {".venv", ".cache", ".git", "__pycache__", "scratch", ".omc", "possession.egg-info"}
GT_ALLOWED = {os.path.join("scripts", "sanity_public.py"), os.path.join("tests", "test_rules_guard.py")}
FRAME_LO, FRAME_HI = 11185, 190197
FRAME_ALLOWED = {os.path.join("possession", "structure.py")}
FRAME_MARK = "frame-ok:"
# Reviewed hits in files this guard's author does not own; replace by an in-file `# frame-ok:` marker when touched.
FRAME_REVIEWED = {
    (os.path.join("possession", "data.py"), 164488),  # broadcast n_video_frames (games.py:87), a length, not a game event
    (os.path.join("possession", "degrade.py"), 30811),  # ATD_SEG_Q top quantile: a segment length, not a frame index
}
NETWORK = {"requests", "urllib", "urllib2", "urllib3", "http", "httplib", "httpx", "socket", "aiohttp",
           "openai", "anthropic", "ftplib", "smtplib", "websocket", "websockets", "paramiko", "grpc"}
DATA_EXT = {".csv", ".npz", ".npy", ".pkl", ".pickle", ".parquet", ".mp4", ".avi", ".mov", ".mkv",
            ".h5", ".hdf5", ".feather", ".xlsx", ".tsv"}
MAX_BYTES = 200 * 1024
GT_RE = re.compile(r"possession_gt|_gt_public|_gt\.csv|evaluation[/\\]gt|[\"']gt[\"']")


def py_files(*subdirs):
    for sub in subdirs:
        for root, dirs, files in os.walk(os.path.join(SOLUTION, sub)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in sorted(files):
                if f.endswith(".py"):
                    yield os.path.join(root, f)


def read(path):
    with open(path) as fh:
        return fh.read()


def rel(path):
    return os.path.relpath(path, SOLUTION)


def frame_literals(src):
    """(lineno, value) of unmarked int literals in the frame range; marker on any line of the statement."""
    tree = ast.parse(src)
    lines = src.splitlines()
    found = []

    def visit(node, stmt):
        if isinstance(node, ast.stmt):
            stmt = node
        if isinstance(node, ast.Constant) and type(node.value) is int and FRAME_LO <= node.value <= FRAME_HI:
            span = lines[stmt.lineno - 1:stmt.end_lineno] if stmt is not None else []
            if not any(FRAME_MARK in ln for ln in span):
                found.append((node.lineno, node.value))
        for child in ast.iter_child_nodes(node):
            visit(child, stmt)

    visit(tree, None)
    return found


def network_imports(src):
    hits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        hits += [(node.lineno, n) for n in names if n.split(".")[0] in NETWORK]
    return hits


def gt_hits(src):
    return [(i, ln.strip()) for i, ln in enumerate(src.splitlines(), 1) if GT_RE.search(ln)]


def shipped_files(root):
    """Tracked plus untracked-not-ignored files (what the next commit could contain); None without git."""
    try:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=root,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [f for f in out.splitlines() if f]


def data_file_problems(root, files):
    bad = []
    for f in files:
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        if os.path.splitext(f)[1].lower() in DATA_EXT:
            bad.append("{}: data extension".format(f))
        elif os.path.getsize(p) > MAX_BYTES:
            bad.append("{}: {} bytes > {}".format(f, os.path.getsize(p), MAX_BYTES))
    return bad


class RulesGuard(unittest.TestCase):
    def test_public_gt_only_in_sanity_public(self):
        bad = []
        for p in py_files("possession", "scripts", "tests"):
            if rel(p) in GT_ALLOWED:
                continue
            bad += ["{}:{}: {}".format(rel(p), n, t) for n, t in gt_hits(read(p))]
        self.assertEqual(bad, [], "public GT referenced outside scripts/sanity_public.py (GUARDRAILS 2)")

    def test_no_unmarked_frame_literals(self):
        bad = []
        for p in py_files("possession"):
            if rel(p) in FRAME_ALLOWED:
                continue
            bad += ["{}:{}: {}".format(rel(p), n, v) for n, v in frame_literals(read(p))
                    if (rel(p), v) not in FRAME_REVIEWED]
        self.assertEqual(bad, [], "game-frame literal outside structure.py; cite it and add `# %s` "
                                  "only if it is organiser-published structure (GUARDRAILS 1.0)" % FRAME_MARK)

    def test_no_network_imports(self):
        bad = []
        for p in py_files("possession", "scripts"):
            bad += ["{}:{}: import {}".format(rel(p), n, m) for n, m in network_imports(read(p))]
        self.assertEqual(bad, [], "network import (GUARDRAILS 3)")

    def test_no_data_files_or_large_files(self):
        files = shipped_files(SOLUTION)
        if files is None:
            self.skipTest("not a git checkout")
        self.assertEqual(data_file_problems(SOLUTION, files), [], "data/large file would be committed (GUARDRAILS 1.4)")


class GuardDetects(unittest.TestCase):
    """The checks themselves flag planted violations (in-memory and in a temp dir, never in solution/)."""

    def test_frame_literal_flagged_and_markable(self):
        self.assertEqual(frame_literals("X = 50000\n"), [(1, 50000)])
        self.assertEqual(frame_literals("X = (1,\n 50000)  # frame-ok: FAQ.md:38\n"), [])
        self.assertEqual(frame_literals("X = 5000\nY = 199319\n"), [])

    def test_network_import_flagged(self):
        self.assertEqual(network_imports("import socket\nfrom urllib.request import urlopen\nimport os\n"),
                         [(1, "socket"), (2, "urllib.request")])

    def test_gt_reference_flagged(self):
        self.assertTrue(gt_hits("p = 'evaluation/gt/BAR-ATH_possession_gt_public.csv'"))
        self.assertTrue(gt_hits("os.path.join(e, \"gt\", f)"))
        self.assertFalse(gt_hits("n_gt_spells = 3"))

    def test_data_files_flagged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for name, body in (("a.csv", "x"), ("big.txt", "x" * (MAX_BYTES + 1)), ("ok.py", "x")):
                with open(os.path.join(d, name), "w") as fh:
                    fh.write(body)
            bad = data_file_problems(d, ["a.csv", "big.txt", "ok.py"])
        self.assertEqual(len(bad), 2)


if __name__ == "__main__":
    unittest.main()
