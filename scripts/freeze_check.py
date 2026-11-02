"""Assert the D-029 freeze in one command; fail loudly on any drift, never repair it.

Anchors: work/DECISIONS.md D-029, work/cards/D05-freeze-final-configuration.md, B18.

    cd solution && make freeze-check

Checks, in order:

1. git identity -- HEAD, the local tag ``final-candidate`` and the files that differ.  A post-freeze
   ``HEAD`` is a **warning**, not a failure: D-029 allows a docs-only follow-up while the rebuilt
   CSV is still ``ee7b5b6b...46d8``, and the sha check below is the gate that decides.
2. two independent ``possession.make_submission`` builds -- the first at the canonical
   ``outputs/submission.csv``, the second in ``scratch/B18/`` -- compared byte for byte.
3. sha256 of the rebuilt CSV equals the frozen value.
4. the organizer validator says Valid with 199,319 rows (and its label counts agree with ours).
5. 370 spells / 138 D stretches / 40,462 X, counted here from the CSV itself.
6. ``tests.test_rules_guard`` (GUARDRAILS 1-3), no skips.
7. ``scripts/walkthrough_numbers.py --check`` (the merged B22): a docs-only commit may not move a
   published number without the number manifest noticing.

A failure prints the specific mismatch (got X, the freeze says Y) and exits non-zero.  Nothing in
this script edits a source file, the CSV or the manifest; ``outputs/submission.csv`` is only
rewritten by build 1, exactly as ``make submission`` writes it.
"""
import argparse
import collections
import csv
import hashlib
import os
import re
import subprocess
import sys
import time

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION)
SCRATCH = os.path.join(ROOT, "scratch", "B18")

FROZEN_SHA = "ee7b5b6bdf7404e82696654ce01b9d823e5b54f149028dcfdeaa5626cf3646d8"
FROZEN_ROWS = 199319
FROZEN_SPELLS = 370
FROZEN_D_STRETCHES = 138
FROZEN_X = 40462
FROZEN_LABELS = collections.OrderedDict([("H", 50478), ("A", 35720), ("D", 72659), ("X", 40462)])
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
    print("== 1/7 git identity")
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


def build_twice():
    """Rebuild the submission twice and compare the bytes."""
    print("== 2/7 two builds, byte-identical")
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
    print("== 3/7 frozen sha256")
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
    print("== 4/7 organizer validator")
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
    print("== 5/7 frozen shape")
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
    print("== 6/7 tests.test_rules_guard")
    out, elapsed = run([sys.executable, "-m", "unittest", "tests.test_rules_guard"],
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
    print("== 7/7 walkthrough number manifest")
    script = os.path.join(SOLUTION, "scripts", "walkthrough_numbers.py")
    if not os.path.isfile(script):
        raise Failure("missing %s; the number manifest is the docs-drift gate (B22)." % script)
    out, _ = run([sys.executable, script, "--check"], cwd=SOLUTION,
                 log=os.path.join(SCRATCH, "walkthrough_numbers.log"))
    if "no drift" not in out:
        raise Failure("walkthrough_numbers.py --check did not report 'no drift':\n%s" % tail(out))
    print("  ok   %s" % out.strip().splitlines()[-1])


def main(argv=None):
    global SCRATCH
    ap = argparse.ArgumentParser(description="Assert the D-029 freeze; fail loudly, never repair.")
    ap.add_argument("--scratch", default=SCRATCH,
                    help="directory for the second build and logs (default: %(default)s)")
    args = ap.parse_args(argv)
    SCRATCH = os.path.abspath(args.scratch)

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

        path, sha = build_twice()
        check_sha(path, sha)
        check_validator(path)
        counts, spells, d_runs = check_counts(path)
        check_rules_guard()
        check_numbers()
    except Failure as error:
        sys.stdout.flush()
        print("\nFREEZE CHECK FAILED", file=sys.stderr)
        for line in str(error).splitlines():
            print("  " + line if line else "", file=sys.stderr)
        print("\nNothing was repaired; the mismatch above is the finding.", file=sys.stderr)
        return 1

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
