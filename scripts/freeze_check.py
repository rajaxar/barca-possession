"""Assert the D-029 freeze in one command; fail loudly on any drift, never repair it.

Anchors: work/DECISIONS.md D-029, work/cards/D05-freeze-final-configuration.md, B18.

    cd solution && make freeze-check

For the planted proofs only, ``python scripts/freeze_check.py --only constant-provenance
--provenance-root scratch/B29/planted`` runs that one check against a scratch tree, and
``--only environment --requirements ../scratch/B32/tampered-requirements.txt`` runs the environment
check against a scratch pin file (whose frozen-record hash therefore differs -- that is the B32
proof).  Both print PARTIAL and are never a freeze verdict.

Checks, in order:

1. git identity -- HEAD, the local tag ``final-candidate`` and the files that differ.  A post-freeze
   ``HEAD`` is a **warning**, not a failure: D-029 allows a docs-only follow-up while the rebuilt
   CSV is still ``ee7b5b6b...46d8``, and the sha check below is the gate that decides.
2. the frozen environment -- D-029 freezes Python **3.13.5** and **34 hash-pinned requirements**.  The
   pin file is also the comparison input, so B32 freezes *its* bytes: ``requirements.txt``'s sha256
   and ``pyproject.toml``'s canonical dependency section (the ``>=3.13`` floor and the setuptools
   build pin) must match constants in this script before anything is compared.  A changed record
   fails as an S02/D-029 reopening, not as environment drift.  Then the interpreter's
   ``major.minor.micro`` must match, and every package in ``requirements.txt`` must be
   installed in the active environment at the pinned version, compared by normalized name + version
   through ``importlib.metadata`` (never a text diff).  A missing pin counts as a mismatch.  It
   prints the record fingerprints and one line -- Python version, pins checked, mismatches -- and
   runs before the builds, so a drifted numpy/scikit-learn fails in ~0.2 s instead of after three
   minutes.  **Hard failure by default**; ``--allow-env-drift`` is the documented opt-out for a
   reviewer's knowingly different machine (it still prints every mismatch and says the run is not
   an environment verdict).  It does **not** cover a changed record: regenerating the pins stays a
   hard failure.
3. ``scripts/constant_provenance.py --check`` (E113/B29/B30): every module-level constant and every
   string-keyed dict field in a registered module carries exactly one provenance tag in
   ``docs/constant_provenance.json``, no value has drifted from the register, ``scaffold`` hides
   no literal, every entry's ``line`` matches the AST, and every ``possession/*.py`` is either
   registered or named as exempt with a reason (the scope is discovered, not listed).  It runs
   before the builds, so a silent retune that leaves the CSV untouched fails in about a tenth of a
   second instead of after three minutes.  A missing register is a clean skip:
   the register is tracked, but a tree that ships only the build must still be checkable.
3. two independent ``possession.make_submission`` builds -- the first at the canonical
   ``outputs/submission.csv``, the second in ``scratch/B18/`` -- compared byte for byte.
4. sha256 of the rebuilt CSV equals the frozen value.
5. the organizer validator says Valid with 199,319 rows (and its label counts agree with ours).
6. 370 spells / 138 D stretches / 40,462 X, counted here from the CSV itself.
7. ``tests.test_rules_guard`` (GUARDRAILS 1-3) and ``tests.test_prose_guard``, no skips.
8. ``scripts/walkthrough_numbers.py --check`` (the merged B22): a docs-only commit may not move a
   published number without the number manifest noticing.
9. the keeper-exclusion contract on a tiny synthetic match (B25/B26): ``gk_possession`` imports, both
   dproxy targets report it, a missing module raises ``MissingGkPossession``, and the explicit
   ``allow_missing=True`` opt-in records ``unavailable`` while vetoing nothing.
10. the Q34 provenance statuses (the merged B27): a number report carries ``events_source``, the
    adoption report carries ``base_rule_source``, and a reference log's git provenance is not
    ``"unknown"`` -- so a broken install or a stale digest fails here, not only in a diagnostic.
11. the numbers manifest's shape (B28): ``notebooks/numbers.json`` is tracked by git, holds only
    numbers and strings (never an array -- a coordinate array is match data), and names only public
    figures/placeholders or local-only paths under ``notebooks/figs/``.
12. the public walkthrough page's publication audit (B28, the merged P04): if
    ``notebooks/possession_walkthrough_public.html`` is present, all 13 checks must pass; if it is
    absent the check is a clean skip, because the page is only built before a release (D-026).
    The audit starts in its own process just before the validator and is joined here, so its ~2 s
    cost is hidden behind checks 6-12; it never overlaps the two builds or the sha assertion.

A failure prints the specific mismatch (got X, the freeze says Y) and exits non-zero.  Nothing in
this script edits a source file, the CSV or the manifest; ``outputs/submission.csv`` is only
rewritten by build 1, exactly as ``make submission`` writes it.
"""
import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from importlib import metadata

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION)
SCRATCH = os.path.join(ROOT, "scratch", "B18")

FROZEN_SHA = "ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8"
FROZEN_ROWS = 199319
FROZEN_SPELLS = 370
FROZEN_D_STRETCHES = 138
FROZEN_X = 40462
FROZEN_LABELS = collections.OrderedDict([("H", 50478), ("A", 35720), ("D", 72659), ("X", 40462)])
FROZEN_PYTHON = "3.13.5"
FROZEN_PINS = 34
# B32: the pin file is the freeze record, so its bytes are frozen too.  `requirements.txt` is both
# the record and the environment check's input; without this anchor, edit a pin + reinstall and both
# sides move together (34 pins, 0 mismatches, a changed environment).  A deliberate regeneration is
# an S02/D-029 reopening and must update these constants.
FROZEN_REQUIREMENTS_SHA = "f35b34af857bf4bd773b62a22e71385c3d154a5aa0235ea9f102fb3630c98092"
# pyproject.toml's dependency section only: the `>=3.13` floor, the setuptools build pin and the
# (empty) runtime dependency list, fingerprinted canonically so formatting cannot move it.
FROZEN_PYPROJECT_DEPS_SHA = "bcf074bbcf6d4721af931774185f4078e385302bbaedb0a18c7eb80d29123233"
TAG = "final-candidate"


class Failure(Exception):
    """A named freeze assertion failed; carries the message printed to the operator."""


def tail(text, lines=25):
    rows = [r for r in text.splitlines() if r.strip()]
    return "\n".join("    " + r for r in rows[-lines:]) if rows else "    (no output)"


def run(cmd, *, cwd, log=None):
    """Run a command, tee its combined output to ``log``; raise Failure with the tail on non-zero."""
    started = time.time()
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    elapsed = time.time() - started
    if log is not None:
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n\n" + proc.stdout)
    if proc.returncode != 0:
        where = " (full log: %s)" % log if log else ""
        raise Failure("command failed with exit %d%s:\n  $ %s\n%s" % (
            proc.returncode, where, " ".join(cmd), tail(proc.stdout)))
    return proc.stdout, elapsed


def expect(name, actual, wanted, extra=""):
    if actual != wanted:
        raise Failure("%s mismatch: rebuilt value is %r, the D-029 freeze says %r.%s"
                      % (name, actual, wanted, ("\n  " + extra) if extra else ""))
    print("  ok   %-22s %s" % (name, actual))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, check=True):
    proc = subprocess.run(["git", "-C", SOLUTION] + list(args), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise Failure("git %s failed: %s" % (" ".join(args), proc.stderr.strip()
                                             or proc.stdout.strip()))
    return proc


def short(sha):
    return sha[:12] if sha else "(none)"


def classify(path):
    if path.startswith("possession/"):
        return "label-path"
    if path.startswith("docs/") or path.endswith((".md", ".txt")):
        return "docs"
    return "build/test/script"


def git_identity():
    """Print HEAD vs the freeze tag; warn (never fail) on a post-freeze HEAD."""
    print("== 1/13 git identity")
    if git("rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise Failure("not a git checkout: the freeze record names a commit and tag, so the identity "
                      "cannot be checked. Run this from the `solution/` repository.")

    head = git("rev-parse", "HEAD").stdout.strip()
    tag = git("rev-parse", "--verify", TAG + "^{commit}", check=False)
    if tag.returncode != 0:
        raise Failure("the local tag %r is missing; without it the frozen commit cannot be identified "
                      "(D-029 records tag %r at f673096c...)." % (TAG, TAG))
    tag_commit = tag.stdout.strip()

    status = git("status", "--porcelain").stdout.strip().splitlines()
    dirty_text = ""
    if status:
        dirty_text = " (+ %d uncommitted path(s) in the build tree)" % len(status)

    if head == tag_commit:
        print("  ok   HEAD == %s == tag %s" % (short(head), TAG))
    else:
        ahead = git("merge-base", "--is-ancestor", tag_commit, head, check=False).returncode == 0
        print("  warn HEAD %s%s is %s tag %s (%s)" % (
            short(head), dirty_text, "ahead of" if ahead else "not a descendant of",
            short(tag_commit), TAG))
        diffs = [p for p in git("diff", "--name-only", tag_commit + ".." + head).stdout.splitlines()
                 if p]
        if diffs:
            grouped = collections.OrderedDict()
            for path in diffs:
                grouped.setdefault(classify(path), []).append(path)
            for kind, paths in grouped.items():
                shown = ", ".join(paths[:8]) + (" ..." if len(paths) > 8 else "")
                print("       %-16s %s" % (kind + ":", shown))
            if "label-path" in grouped:
                print("       NOTE: possession/ differs from the tag. D-029 allows any post-freeze")
                print("       commit only while the rebuilt CSV is still %s; the sha" % short(FROZEN_SHA))
                print("       assertion below decides. A docstring/comment change must leave it fixed.")
            elif ahead:
                print("       Only non-label-path files differ (docs/prose/tests/scripts); D-029's")
                print("       docs-only condition holds if the rebuild below is still %s."
                      % short(FROZEN_SHA))
        if status:
            print("       WARNING: uncommitted changes are in the working tree, so the build below")
            print("       reflects the files on disk, not necessarily commit %s." % short(head))
    return head, tag_commit, status


def canonical_name(name):
    """PEP 503 normalization: ``scikit_learn``, ``scikit-learn`` and ``Scikit.Learn`` are one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path):
    """Pinned ``(canonical name -> (name, version))`` from a hash-locked requirements file.

    Line continuations are joined, trailing ``--hash`` options are ignored, and an unreadable
    requirement raises rather than being skipped: silently dropping a pin would make the check
    vacuous.  Nothing here compares file text with the environment.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read().replace("\\\r\n", " ").replace("\\\n", " ")
    pins = collections.OrderedDict()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;]+)",
                         line)
        if not match:
            raise Failure("cannot read a requirement in %s near %r; every line must be "
                          "name==version with optional --hash options (refusing to skip a pin)."
                          % (path, line[:90]))
        name, version = match.group(1), match.group(2)
        key = canonical_name(name)
        if key in pins:
            raise Failure("duplicate pin for %s in %s; one version per package, not two."
                          % (name, path))
        pins[key] = (name, version)
    return pins


def dependency_section(path):
    """Canonical ``pyproject.toml`` dependency section and its sha256 (B32).

    Only the declarations that decide the environment are read: the build requirement, the Python
    floor and the runtime dependency list.  Formatting and comment edits cannot move the
    fingerprint; a changed floor or build pin does.
    """
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    section = collections.OrderedDict([
        ("build-system.requires", data.get("build-system", {}).get("requires")),
        ("project.dependencies", data.get("project", {}).get("dependencies")),
        ("project.requires-python", data.get("project", {}).get("requires-python")),
    ])
    blob = json.dumps(section, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return section, hashlib.sha256(blob).hexdigest()


def check_frozen_record(path):
    """B32: the pin file is the freeze record, so its bytes are frozen too.

    B23 compares the installed set against the pins, but ``requirements.txt`` is also that
    comparison's input: edit a pin, reinstall the matching version, and both sides move together --
    the check reports 0 mismatches on a changed environment.  The sha256 below is the anchor the
    file itself cannot move, so regenerating the pins is a deliberate act.  Hard failure:
    ``--allow-env-drift`` does not apply, because a changed record is a freeze change, not a
    difference between machines.
    """
    digest = sha256(path)
    rel = os.path.relpath(path)
    if digest != FROZEN_REQUIREMENTS_SHA:
        raise Failure(
            "the pin file itself changed: %s is sha256 %s,\n"
            "  but the D-029 freeze (B32) records solution/requirements.txt as %s.\n"
            "  requirements.txt is the freeze record, not just the comparison input: edit a pin and\n"
            "  reinstall the matching version and the pins and the installed set move together, so\n"
            "  the installed-vs-pinned comparison still reports 0 mismatches on a changed\n"
            "  environment. Regenerating the pins is an S02/D-029 reopening, not a routine edit: it\n"
            "  needs a new decision, a re-run of Q04, and a re-checked CSV sha256. The freeze is\n"
            "  %s today,\n"
            "  and it is not assumed to survive the reinstall. If that is the intent, update\n"
            "  FROZEN_REQUIREMENTS_SHA deliberately; nothing was repaired by this check."
            % (rel, digest, FROZEN_REQUIREMENTS_SHA, FROZEN_SHA))
    print("  ok   record             %s sha256 %s (frozen)" % (rel, short(digest)))

    pyproject = os.path.join(SOLUTION, "pyproject.toml")
    if not os.path.isfile(pyproject):
        raise Failure("missing %s; the D-029 Python floor and build pin cannot be asserted (B32)."
                      % pyproject)
    section, section_sha = dependency_section(pyproject)
    if section_sha != FROZEN_PYPROJECT_DEPS_SHA:
        raise Failure(
            "pyproject.toml's dependency section changed: %s is sha256 %s,\n"
            "  but the D-029 freeze (B32) records it as %s.\n"
            "  The section is %s.\n"
            "  The Python floor and the setuptools build pin are part of the frozen environment.\n"
            "  Changing them is an S02/D-029 reopening, not a routine edit: it needs a new decision,\n"
            "  a re-run of Q04, and a re-checked CSV sha256 (the freeze is %s today). If that is the\n"
            "  intent, update FROZEN_PYPROJECT_DEPS_SHA deliberately; nothing was repaired."
            % (os.path.relpath(pyproject), section_sha, FROZEN_PYPROJECT_DEPS_SHA,
               json.dumps(section, sort_keys=True), FROZEN_SHA))
    print("  ok   record             pyproject deps sha256 %s (floor %s, build %s)"
          % (short(section_sha), section["project.requires-python"],
             ", ".join(section["build-system.requires"] or [])))


def check_environment(requirements=None, allow_drift=False):
    """B23: D-029's Python 3.13.5 and 34 hash-pinned requirements are part of the freeze.

    Compare by normalized package name + version through ``importlib.metadata`` -- the same
    semantics ``pip check`` uses -- not by a text diff of ``requirements.txt``.  A missing pin is a
    mismatch, and a shrunken/duplicated pin file fails before any comparison, so the check cannot
    pass vacuously.  B32 anchors the record itself: the pin file's sha256 and ``pyproject.toml``'s
    dependency section must match frozen constants, so a changed pin file fails even when the
    installed set was reinstalled to match it.  It is offline, installs and repairs nothing, and
    runs before the builds so drift fails in ~0.2 s rather than after three minutes.  Hard failure
    by default; the documented ``--allow-env-drift`` opt-out still prints every mismatch, then
    continues with a warning that the run is not an environment verdict.  That flag never covers a
    changed record.
    """
    print("== 2/13 frozen environment")
    path = os.path.abspath(requirements or os.path.join(SOLUTION, "requirements.txt"))
    if not os.path.isfile(path):
        raise Failure("requirements file not found at %s; the D-029 environment cannot be asserted."
                      % path)
    check_frozen_record(path)
    pins = read_pins(path)
    if len(pins) != FROZEN_PINS:
        raise Failure("%s holds %d pins, but the D-029 freeze names %d hash-pinned requirements. "
                      "A pin was added, removed or merged; that is a freeze change, not environment "
                      "drift, so --allow-env-drift does not cover it."
                      % (path, len(pins), FROZEN_PINS))

    python_now = ".".join(str(part) for part in sys.version_info[:3])
    drift = []
    if python_now != FROZEN_PYTHON:
        drift.append("python: installed %s, the D-029 freeze says %s" % (python_now, FROZEN_PYTHON))
    for name, wanted in pins.values():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != wanted:
            drift.append("package %s: installed %s, %s pins %s"
                         % (name, installed or "(not installed)", os.path.basename(path), wanted))

    label = "ok  " if not drift else ("warn" if allow_drift else "FAIL")
    print("  %s environment        Python %s, %d pins checked, %d mismatch%s"
          % (label, python_now, len(pins), len(drift), "" if len(drift) == 1 else "es"))
    if not drift:
        if allow_drift:
            print("       --allow-env-drift given, but nothing drifted; the flag changed nothing.")
        return

    for line in drift:
        print("       " + line)
    if allow_drift:
        print("       --allow-env-drift: continuing on a knowingly different machine. This run is")
        print("       not an environment verdict; the CSV/sha assertions below still decide the build.")
        return

    raise Failure(
        "environment drift -- the active interpreter does not match the D-029 freeze:\n  "
        + "\n  ".join(drift)
        + "\n  Run this target with solution/.venv/bin/python built from requirements.txt\n"
          "  (Q04: Python %s, %d hash pins), or pass --allow-env-drift on a reviewer's knowingly\n"
          "  different machine (the CSV assertions then still run). Nothing is installed or repaired."
          % (FROZEN_PYTHON, FROZEN_PINS))


def check_constants(root=None):
    """E113/B29: the shipped constants and dict fields agree with their provenance register.

    Delegates to ``scripts/constant_provenance.py --check`` so there is exactly one implementation of
    the rule, shared with ``make test``.  It runs before the builds: a constant retune that changes
    provenance without changing the CSV would otherwise pass every other check on this path.
    """
    print("== 3/13 constant provenance")
    root = SOLUTION if root is None else os.path.abspath(root)
    register = os.path.join(root, "docs", "constant_provenance.json")
    if not os.path.isfile(register):
        print("  skip provenance        register absent (%s); nothing to compare"
              % os.path.relpath(register, root))
        return
    script = os.path.join(SOLUTION, "scripts", "constant_provenance.py")
    if not os.path.isfile(script):
        raise Failure("missing %s; without it the provenance register cannot be checked (E113)."
                      % script)
    os.makedirs(SCRATCH, exist_ok=True)
    log = os.path.join(SCRATCH, "constant_provenance.log")
    try:
        out, elapsed = run([sys.executable, script, "--check", "--root", root],
                           cwd=SOLUTION, log=log)
    except Failure:
        fails = []
        if os.path.isfile(log):
            with open(log, encoding="utf-8") as fh:
                fails = [line[5:] for line in fh.read().splitlines() if line.startswith("FAIL ")]
        if not fails:
            raise  # the command itself failed; run() already carries its output tail
        more = "" if len(fails) <= 12 else "\n  ... and %d more in %s" % (len(fails) - 12, log)
        raise Failure(
            "constant provenance does not match the register (%s):\n  %s%s\n"
            "  Every shipped constant and every string-keyed dict field carries exactly one tag in\n"
            "  %s. A retune that moves provenance without moving the CSV fails here instead of\n"
            "  passing the sha. Update the register and docs/CONSTANTS.md, or revert the value; a\n"
            "  constant is never justified by this match's answers."
            % (os.path.relpath(register, SOLUTION), "\n  ".join(fails[:12]), more,
               os.path.relpath(register, root)))

    match = re.search(r"^constant provenance: (\d+) constants, (\d+) dict fields, (\d+) problem",
                      out, re.M)
    if not match:
        raise Failure("constant_provenance.py --check printed no summary line:\n%s" % tail(out))
    constants, fields, problems = (int(group) for group in match.groups())
    if problems:
        raise Failure("constant provenance: %d problem(s) but the check exited 0; refusing to pass."
                      % problems)
    if constants == 0:
        raise Failure("constant provenance checked 0 constants; an empty scan is not a pass.")
    print("  ok   provenance         %d constants, %d dict fields, %d problems (%.1f s)"
          % (constants, fields, problems, elapsed))


def build_twice():
    """Rebuild the submission twice and compare the bytes."""
    print("== 4/13 two builds, byte-identical")
    os.makedirs(SCRATCH, exist_ok=True)
    canonical = os.path.join(SOLUTION, "outputs", "submission.csv")
    second = os.path.join(SCRATCH, "build2.csv")
    logs = (os.path.join(SCRATCH, "build1.log"), os.path.join(SCRATCH, "build2.log"))

    built = []
    for n, out in enumerate((canonical, second), 1):
        print("  ..   build %d -> %s" % (n, os.path.relpath(out, SOLUTION)))
        _, elapsed = run([sys.executable, "-m", "possession.make_submission", "--out", out],
                         cwd=SOLUTION, log=logs[n - 1])
        if not os.path.isfile(out):
            raise Failure("build %d exited 0 but wrote no file at %s (log: %s)"
                          % (n, out, logs[n - 1]))
        built.append(out)
        print("  ok   build %d finished in %.1f s, %d bytes"
              % (n, elapsed, os.path.getsize(out)))

    a, b = (sha256(p) for p in built)
    if a != b:
        raise Failure("the two builds are not byte-identical: build 1 %s vs build 2 %s.\n"
                      "  The build is nondeterministic; that alone breaks GUARDRAILS 1.4 (one command, "
                      "byte-identical).\n  Compare %s and %s; logs: %s, %s"
                      % (a, b, built[0], built[1], logs[0], logs[1]))
    print("  ok   cmp                builds identical (%s)" % short(a))
    return built[0], a


def check_sha(path, sha):
    print("== 5/13 frozen sha256")
    if sha != FROZEN_SHA:
        raise Failure(
            "sha256 mismatch: the rebuilt CSV is %s\n"
            "  but the D-029 freeze is %s.\n"
            "  The two builds were byte-identical, so the tree is deterministic; the code, data or\n"
            "  environment no longer reproduces the frozen artifact. Nothing was changed by this\n"
            "  check. Investigate `git -C solution diff --stat %s..HEAD` (and any uncommitted\n"
            "  changes), then the build logs in %s."
            % (sha, FROZEN_SHA, TAG, SCRATCH))
    print("  ok   sha256             %s" % sha)


def check_validator(path):
    """Run the organizer's validator on the rebuilt CSV; assert Valid + 199,319 rows."""
    print("== 6/13 organizer validator")
    repo = os.path.abspath(os.environ.get(
        "MTAH_REPO_DIR", os.path.join(ROOT, "more-than-a-hack-2027")))
    validator = os.path.join(repo, "evaluation", "validate_submission.py")
    if not os.path.isfile(validator):
        raise Failure("organizer validator not found at %s. Set MTAH_REPO_DIR to the organizer "
                      "repository (D-029 validates with evaluation/validate_submission.py)." % validator)
    out, _ = run([sys.executable, validator, path], cwd=SOLUTION,
                 log=os.path.join(SCRATCH, "validator.log"))

    if "Valid." not in out:
        raise Failure("the organizer validator did not say Valid:\n%s" % tail(out))
    match = re.search(r"^\s+(\d+) rows, frame 0\.\.(\d+)\s*$", out, re.M)
    if not match:
        raise Failure("the organizer validator printed no row/frame count; cannot assert %d rows:\n%s"
                      % (FROZEN_ROWS, tail(out)))
    rows, last = int(match.group(1)), int(match.group(2))
    expect("validator rows", rows, FROZEN_ROWS,
           "the validator read %s; D-029 freezes 199,319 rows (frames 0..199318)." % path)
    expect("validator last frame", last, FROZEN_ROWS - 1)

    counted = {}
    for label, value in re.findall(r"^\s+([HADX])\s+([\d,]+)\s+[\d.]+%\s*$", out, re.M):
        counted[label] = int(value.replace(",", ""))
    if counted:
        for label, wanted in FROZEN_LABELS.items():
            if label in counted:
                expect("validator %s count" % label, counted[label], wanted,
                       "the validator's own label counts disagree with the freeze.")
    print("  ok   verdict            Valid")
    return rows


def read_labels(path):
    """Labels from the CSV, with the frame column verified to be 0..n-1."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] != ["frame", "label"]:
        raise Failure("unexpected CSV header %r in %s (expected ['frame', 'label'])"
                      % (rows[0] if rows else None, path))
    labels = []
    for i, row in enumerate(rows[1:]):
        if len(row) != 2 or row[0] != str(i):
            raise Failure("row %d of %s is %r; expected frame %d" % (i + 1, path, row, i))
        if row[1] not in ("H", "A", "D", "X"):
            raise Failure("row %d of %s has label %r; the frozen file uses only H/A/D/X"
                          % (i + 1, path, row[1]))
        labels.append(row[1])
    return labels


def maximal_D_runs(labels):
    runs, current = 0, 0
    for label in labels:
        if label == "D":
            current += 1
            continue
        if current:
            runs += 1
            current = 0
    return runs + (1 if current else 0)


def check_counts(path):
    """Count spells / D stretches / X from the CSV itself (not from the manifest)."""
    print("== 7/13 frozen shape")
    labels = read_labels(path)
    expect("rows", len(labels), FROZEN_ROWS)

    counts = collections.Counter(labels)
    spells, previous = 0, None
    for label in labels:
        if label in ("H", "A") and label != previous:
            spells += 1  # a maximal H/A run; D splits it
        previous = label
    d_runs = maximal_D_runs(labels)

    expect("spells", spells, FROZEN_SPELLS,
           "a spell is a maximal H/A run; D/X split it. Counted from %s." % path)
    expect("D stretches", d_runs, FROZEN_D_STRETCHES)
    expect("X frames", counts.get("X", 0), FROZEN_X)
    for label, wanted in FROZEN_LABELS.items():
        expect("%s frames" % label, counts.get(label, 0), wanted)
    return counts, spells, d_runs


def check_rules_guard():
    print("== 8/13 tests.test_rules_guard + tests.test_prose_guard")
    out, elapsed = run([sys.executable, "-m", "unittest",
                        "tests.test_rules_guard", "tests.test_prose_guard"],
                       cwd=SOLUTION, log=os.path.join(SCRATCH, "rules_guard.log"))
    match = re.search(r"^Ran (\d+) tests?", out, re.M)
    if not match:
        raise Failure("test_rules_guard printed no test count:\n%s" % tail(out))
    ran = int(match.group(1))
    if ran == 0:
        raise Failure("test_rules_guard ran 0 tests; the rules guard is not actually checking anything.")
    if re.search(r"skipped=", out):
        raise Failure("test_rules_guard skipped a test:\n%s\n"
                      "  A skipped rules guard is not a passing rules guard." % tail(out))
    print("  ok   %d tests in %.1f s, no skips" % (ran, elapsed))


def check_numbers():
    """Merged B22: a post-freeze commit may not move a published number unnoticed."""
    print("== 9/13 walkthrough number manifest")
    script = os.path.join(SOLUTION, "scripts", "walkthrough_numbers.py")
    if not os.path.isfile(script):
        raise Failure("missing %s; the number manifest is the docs-drift gate (B22)." % script)
    out, _ = run([sys.executable, script, "--check"], cwd=SOLUTION,
                 log=os.path.join(SCRATCH, "walkthrough_numbers.log"))
    if "no drift" not in out:
        raise Failure("walkthrough_numbers.py --check did not report 'no drift':\n%s" % tail(out))
    print("  ok   %s" % out.strip().splitlines()[-1])


def _synthetic_match(n=600):
    """Tiny ATD-shaped Match for the contract checks: keepers, walking outfielders, half a ball.

    Ball present for the first ``n // 2`` frames, missing after: that gives the missing-ball proxy a
    qualifying gap and the present-ball proxy a graded ball-present stretch. No match data is read.
    """
    import numpy as np
    from possession.data import Match
    from possession.structure import PITCH_L, PITCH_W, Structure

    present_frames = n // 2
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
    B = np.full((n, 2), np.nan, np.float32)
    B[:present_frames] = (PITCH_L / 2, PITCH_W / 2)
    match = Match(P, V, T, B, np.zeros(n, np.float32),
                  np.array([[22, 45], [22, 45]]), np.ones(n, bool), half)
    st = Structure(halves={1: (0, n - 1)}, kickoffs={1: 0},
                   attack_dir=np.array([[1, -1], [-1, 1]], np.int8),
                   forced_dead_mask=np.zeros(n, bool))
    return match, st


def check_keeper_contract():
    """B25/B26: the keeper exclusion is imported, named, and never substituted for.

    Mirrors ``tests/test_dproxy.py``'s ``GkSourceContract`` on a synthetic match: if ``dproxy``
    silently loses ``gk_possession`` this fails, instead of only a later diagnostic noticing. The
    check is a diagnostic contract only -- ``gk_possession`` stays out of the build (D-028/D-029).
    """
    print("== 10/13 keeper exclusion contract")
    import numpy as np
    from unittest import mock

    from possession import dproxy, gk_possession  # noqa: F401 - the import is the first assertion

    match, st = _synthetic_match()
    dp = dproxy.proxy(match, st, dict(exclude_gk=True))
    pb = dproxy.present_ball(match, st, dict(exclude_gk=True))
    for name, info in (("proxy", dp.info), ("present_ball", pb.info)):
        if info.get("gk_source") != "gk_possession" or info.get("gk_note") is not None:
            raise Failure("keeper contract: %s reports gk_source=%r, gk_note=%r; the freeze requires "
                          "the imported detector and no note (B25)."
                          % (name, info.get("gk_source"), info.get("gk_note")))

    with mock.patch.object(dproxy, "_import_gk", side_effect=ImportError("no gk module")):
        for name, fn in (("proxy", dproxy.proxy), ("present_ball", dproxy.present_ball)):
            try:
                fn(match, st, dict(exclude_gk=True))
            except dproxy.MissingGkPossession as exc:
                if "allow_missing=True" not in str(exc):
                    raise Failure("keeper contract: %s raised MissingGkPossession without naming the "
                                  "allow_missing=True opt-in: %s" % (name, exc))
            else:
                raise Failure("keeper contract: %s ran without possession.gk_possession and did not "
                              "raise MissingGkPossession (B25 deleted every substitute)." % name)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            dp_opt = dproxy.proxy(match, st, dict(exclude_gk=True, allow_missing=True))
            pb_opt = dproxy.present_ball(match, st, dict(exclude_gk=True, allow_missing=True))

    dp_off = dproxy.proxy(match, st, dict(exclude_gk=False))
    pb_off = dproxy.present_ball(match, st, dict(exclude_gk=False))
    for name, got, off in (("proxy", dp_opt, dp_off), ("present_ball", pb_opt, pb_off)):
        info = got.info
        if info.get("gk_source") != "unavailable":
            raise Failure("keeper contract: %s with allow_missing=True reports gk_source=%r, not "
                          "'unavailable'." % (name, info.get("gk_source")))
        if "allow_missing=True" not in str(info.get("gk_note")):
            raise Failure("keeper contract: %s with allow_missing=True does not record the opt-in in "
                          "gk_note: %r" % (name, info.get("gk_note")))
        if info.get("gk_frames") != 0 or info.get("gk_vetoed") != 0:
            raise Failure("keeper contract: %s with allow_missing=True invented a keeper mask "
                          "(gk_frames=%r, gk_vetoed=%r)."
                          % (name, info.get("gk_frames"), info.get("gk_vetoed")))
        if not np.array_equal(np.asarray(got.mask, bool), np.asarray(off.mask, bool)):
            raise Failure("keeper contract: %s with allow_missing=True vetoes frames it cannot see "
                          "(its mask differs from exclude_gk=False)." % name)
    if "WARNING [dproxy]" not in stderr.getvalue():
        raise Failure("keeper contract: the allow_missing=True opt-in printed no WARNING [dproxy] line.")
    print("  ok   keeper contract      gk_possession imported; both proxies 'gk_possession'/no note; "
          "ImportError -> MissingGkPossession; opt-in 'unavailable', 0 vetoed")


def check_provenance():
    """Q34/B27: every status that travels with a reported number is asserted on the freeze path.

    Mirrors ``tests/test_q34_status.py``: the number report names where its whistle events came from,
    the adoption report names its ablation base rule, and a reference log names a real git commit
    (never ``unknown``). A broken install or stale digest fails the freeze instead of a diagnostic.
    """
    print("== 11/13 provenance statuses")
    import numpy as np
    from unittest import mock

    from possession import audio, dproxy, gk_possession

    match, st = _synthetic_match()
    events = np.zeros(0, dtype=audio.EVENT_DTYPE)
    with mock.patch.object(audio, "whistle_events",
                           lambda *a, **k: audio.Track(events, False, "no_ffmpeg", None)):
        gp = gk_possession.detect(match, st)
    if gp.info.get("events_source") != "no_ffmpeg" or gp.info.get("n_events") != 0:
        raise Failure("provenance: the keeper number report reads events_source=%r, n_events=%r; a "
                      "missing audio track must be named 'no_ffmpeg' (Q34 b1)."
                      % (gp.info.get("events_source"), gp.info.get("n_events")))

    pb = dproxy.present_ball(match, st, dict(exclude_gk=True))
    cue = np.zeros(len(match.B), bool)
    for start in (100, 180, 250):
        cue[start:start + 25] = True
    report = dproxy.adoption_report(cue, match, proxy_obj=pb, n_boot=20)
    if report.get("base_rule_source") != "proxy_info":
        raise Failure("provenance: the adoption report reads base_rule_source=%r; a proxy built by "
                      "present_ball must declare its own base rule (Q34 b4)."
                      % report.get("base_rule_source"))

    # Q34 b5: a reference-cache hit logs the code that wrote the table. Call the real provenance()
    # but skip two imports that do not feed its git/path fields: dev_v1 imports the pandas-backed
    # evaluate harness, and params() imports the scipy degrader only to print two constants.
    import types
    with mock.patch.dict(sys.modules,
                         {"possession.evaluate": types.ModuleType("possession.evaluate")}):
        from possession import dev_v1
        with mock.patch.object(dev_v1, "params", lambda: {}):
            prov = dev_v1.provenance()
    if "git=" not in prov or "git=unknown" in prov:
        raise Failure("provenance: dev_v1.provenance() has no real git hash (%r); a reference-cache "
                      "hit would be logged without usable provenance (Q34 b5)." % prov)
    git_hash = prov.split("git=", 1)[1].split()[0]
    print("  ok   provenance          events_source=no_ffmpeg; base_rule_source=proxy_info; "
          "git=%s" % git_hash)


def _surfaces():
    """Import the B28 surface checks without requiring ``scripts`` to be a package."""
    scripts = os.path.join(SOLUTION, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import freeze_surfaces  # noqa: E402 - local helper, no match data
    return freeze_surfaces


def check_manifest_shape():
    """B28: the tracked number manifest holds numbers and strings, never an array of coordinates."""
    print("== 12/13 numbers manifest shape")
    surfaces = _surfaces()
    try:
        report = surfaces.check_manifest()
    except surfaces.SurfaceFailure as exc:
        raise Failure(str(exc))
    print("  ok   manifest           %d entries, %d figure ref(s), tracked by git (%s)"
          % (report["entries"], report["figure_refs"], report["manifest"]))


def _public_page_start():
    """Start the public-page audit now so checks 6-12 run while it does (B28)."""
    return _surfaces().start_public_page()


def check_public_page(proc=None):
    """B28/P04: if the public page exists, its 13-check publication audit must pass; else skip.

    ``proc`` is the audit process started just before the validator; joining it here hides the
    audit's ~2 s behind checks 6-12 without ever overlapping the two builds.
    """
    print("== 13/13 public walkthrough page")
    surfaces = _surfaces()
    try:
        report = surfaces.check_public_page(proc=proc)
    except surfaces.SurfaceFailure as exc:
        raise Failure(str(exc))
    if report["status"] == "absent":
        print("  skip public page      not built (%s); optional deliverable, nothing to audit"
              % os.path.relpath(report["html"], SOLUTION))
        return
    print("  ok   public page        %d/%d audit checks, no match-derived figure or text (%s)"
          % (report["checks"], surfaces.AUDIT_CHECKS, os.path.relpath(report["html"], SOLUTION)))


def report_failure(error, title):
    """Print a Failure the way the operator must read it; a partial run says so in the title."""
    sys.stdout.flush()
    print("\n%s FAILED" % title, file=sys.stderr)
    for line in str(error).splitlines():
        print("  " + line if line else "", file=sys.stderr)
    print("\nNothing was repaired; the mismatch above is the finding.", file=sys.stderr)
    return 1


def main(argv=None):
    global SCRATCH
    ap = argparse.ArgumentParser(description="Assert the D-029 freeze; fail loudly, never repair.")
    ap.add_argument("--scratch", default=SCRATCH,
                    help="directory for the second build and logs (default: %(default)s)")
    ap.add_argument("--only", choices=("constant-provenance", "environment"), default=None,
                    help="run only this fast check and exit; for the B29/B23 proofs and debugging, "
                         "prints PARTIAL and is never a freeze verdict")
    ap.add_argument("--provenance-root", default=None,
                    help="with --only constant-provenance, check a scratch tree's modules and "
                         "docs/constant_provenance.json instead of solution/")
    ap.add_argument("--requirements", default=None,
                    help="with --only environment, read this hash-locked pin file instead of "
                         "solution/requirements.txt (planted proof; its bytes will not match the "
                         "frozen record hash and the check says so; installs nothing)")
    ap.add_argument("--allow-env-drift", action="store_true",
                    help="downgrade a Python/package version mismatch to a warning for a reviewer's "
                         "knowingly different machine; every mismatch is still printed and the CSV "
                         "assertions still run (default: hard failure)")
    args = ap.parse_args(argv)
    SCRATCH = os.path.abspath(args.scratch)

    if args.requirements and args.only != "environment":
        ap.error("--requirements is only valid with --only environment")

    if args.only:
        print("PARTIAL freeze check -- only %r; this is not a freeze verdict." % args.only)
        try:
            if args.only == "environment":
                check_environment(args.requirements, args.allow_env_drift)
            else:
                check_constants(args.provenance_root)
        except Failure as error:
            return report_failure(error, "PARTIAL CHECK")
        print("\nPARTIAL CHECK PASSED -- run `make freeze-check` for the freeze verdict")
        return 0

    print("make freeze-check -- D-029 frozen artifact %s" % short(FROZEN_SHA))
    try:
        head, tag_commit, status = git_identity()

        # The build imports `possession` from the working directory; prove it is this tree
        # (GUARDRAILS §4, the Q17 snapshot trap).
        probe, _ = run([sys.executable, "-c", "import possession; print(possession.__file__)"],
                       cwd=SOLUTION)
        imported = probe.strip()
        if not os.path.abspath(imported).startswith(SOLUTION + os.sep):
            raise Failure("the build would import possession from %s, not this tree (%s); "
                          "refusing to call that a freeze check." % (imported, SOLUTION))
        print("  ok   import             %s" % imported)

        check_environment(allow_drift=args.allow_env_drift)
        check_constants()
        path, sha = build_twice()
        check_sha(path, sha)
        page_proc = _public_page_start()
        check_validator(path)
        counts, spells, d_runs = check_counts(path)
        check_rules_guard()
        check_numbers()
        check_keeper_contract()
        check_provenance()
        check_manifest_shape()
        check_public_page(page_proc)
    except Failure as error:
        return report_failure(error, "FREEZE CHECK")

    print("\nFREEZE CHECK PASSED")
    print("  commit        %s%s" % (head, "" if head == tag_commit else
                                    " (tag %s -> %s)" % (TAG, short(tag_commit))))
    print("  sha256        %s" % sha)
    print("  rows          {:,}".format(FROZEN_ROWS))
    print("  spells        {}".format(spells))
    print("  D stretches   {}".format(d_runs))
    print("  X frames      {:,}".format(counts.get("X", 0)))
    print("  labels        H {:,} / A {:,} / D {:,} / X {:,}".format(
        counts.get("H", 0), counts.get("A", 0), counts.get("D", 0), counts.get("X", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
