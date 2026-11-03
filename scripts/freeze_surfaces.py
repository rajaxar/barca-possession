"""The two compliance surfaces of the freeze, asserted in one place (B28).

Both surfaces sit on the publication boundary, so they belong on the freeze path rather than in a
manual script:

* **the public walkthrough page** -- ``notebooks/possession_walkthrough_public.html``. Its audit
  (``scripts/audit_walkthrough_public.py``) re-renders the public figure set, hashes every embedded
  payload and greps the visible text for frame numbers, clock times, match paths and card ids. A
  present page must pass every check; an absent page is a clean skip, because the page is an
  optional deliverable that is only built before a release (D-026).
* **the numbers manifest** -- ``notebooks/numbers.json``. It is tracked, so it is the one place a
  published number can be smuggled into the repo. The shape check asserts it holds numbers and
  strings only (never an array -- a coordinate array is match data, GUARDRAILS 1.3), that every
  figure it names is a public figure/placeholder or a local-only path under ``notebooks/figs/``,
  and that the file is tracked by git (a number that is not committed is not reviewable).

``scripts/freeze_check.py`` runs both (checks 12-13) and ``tests/test_freeze_surfaces.py`` runs them
under ``make test``. Neither reads match data and neither writes anything.

    cd solution
    .venv/bin/python scripts/freeze_surfaces.py            # both, human-readable
    .venv/bin/python scripts/freeze_surfaces.py --json /tmp/surfaces.json
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(SOLUTION, "notebooks", "numbers.json")
PUBLIC_HTML = os.path.join(SOLUTION, "notebooks", "possession_walkthrough_public.html")
WALKTHROUGH = os.path.join(SOLUTION, "scripts", "build_walkthrough.py")

#: The audit prints one PASS/FAIL line per check. A missing check is a gap, not a pass.
AUDIT_CHECKS = 13

TOP_KEYS = {"entries", "generated_by", "schema", "shipped_stack"}
ENTRY_REQUIRED = {"value", "display", "source", "command", "recomputed", "tolerance"}
ENTRY_OPTIONAL = {"note"}
SCALARS = (str, int, float, bool, type(None))

FIGURE_NAME = re.compile(r"(?<![\w/.-])(fig_[a-z0-9_]+)(?![\w])")
FIGURE_PATH = re.compile(
    r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:png|gif|jpe?g|mp4|webm|svg))")
PUBLIC_PATH = re.compile(r"^(?:\.\./)?(?:notebooks/)?figs/public/[\w.-]+\.(?:png|gif|svg)$")
LOCAL_PATH = re.compile(r"^(?:\.\./)?(?:notebooks/)?figs/([\w.-]+)\.(?:png|gif|mp4|webm)$")


class SurfaceFailure(Exception):
    """A compliance surface failed; the message names the file, the field and the rule."""


# ----------------------------------------------------------------------------- figure registry

def figure_registry():
    """``build_walkthrough.FIGURES`` read as a literal, without importing numpy/matplotlib.

    name -> (section, local_only, caption). Parsing the source keeps this check independent of the
    exporter's imports and costs microseconds on the freeze path.
    """
    with open(WALKTHROUGH, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=WALKTHROUGH)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "FIGURES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SurfaceFailure("cannot find the FIGURES registry in %s; the manifest figure check has "
                         "nothing to compare against." % WALKTHROUGH)


# ----------------------------------------------------------------------------- manifest shape

def _strings(node, path):
    """Yield (path, value) for every string anywhere in the document."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(value, "%s.%s" % (path, key) if path else str(key))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _strings(value, "%s[%d]" % (path, i))


def _first_sequence(node, path):
    """The first list/tuple in the document, as (path, value), or None."""
    if isinstance(node, (list, tuple)):
        return path, node
    if isinstance(node, dict):
        for key, value in node.items():
            found = _first_sequence(value, "%s.%s" % (path, key) if path else str(key))
            if found:
                return found
    return None


def _tracked(relpath):
    proc = subprocess.run(["git", "-C", SOLUTION, "ls-files", "--error-unmatch", relpath],
                          capture_output=True, text=True)
    return proc.returncode == 0


def check_manifest(manifest=None):
    """Shape-check ``notebooks/numbers.json``; return a summary dict or raise SurfaceFailure."""
    manifest = os.path.abspath(manifest or MANIFEST)
    rel = os.path.relpath(manifest, SOLUTION)
    if not os.path.isfile(manifest):
        raise SurfaceFailure("manifest shape: %s is missing. Every published number lives there "
                             "(D-026); run scripts/walkthrough_numbers.py." % rel)
    try:
        with open(manifest, encoding="utf-8") as fh:
            doc = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SurfaceFailure("manifest shape: %s is not valid JSON (%s)." % (rel, exc))

    if not isinstance(doc, dict):
        raise SurfaceFailure("manifest shape: %s is a %s, not the object {entries, ...}."
                             % (rel, type(doc).__name__))
    extra_top = sorted(set(doc) - TOP_KEYS)
    if extra_top:
        raise SurfaceFailure("manifest shape: %s has unexpected top-level key(s) %s; the schema is "
                             "%s." % (rel, extra_top, sorted(TOP_KEYS)))
    if not isinstance(doc.get("entries"), dict) or not doc["entries"]:
        raise SurfaceFailure("manifest shape: %s has no non-empty 'entries' object." % rel)

    # 1. numbers and strings only: an array anywhere is match data, not a published number.
    found = _first_sequence(doc, "")
    if found:
        where, value = found
        raise SurfaceFailure(
            "manifest shape: %s %s is a %s with %d item(s), not a scalar.\n"
            "  The manifest is a table of numbers (D-026); a coordinate array is match data and may "
            "never be tracked (GUARDRAILS 1.3/1.4).\n"
            "  Remove it and keep the derived number only, with its source and command."
            % (rel, where or "<root>", type(value).__name__, len(value)))

    entries = doc["entries"]
    figure_refs = 0
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            raise SurfaceFailure("manifest shape: %s entry %r is a %s, not an object."
                                 % (rel, key, type(entry).__name__))
        missing = sorted(ENTRY_REQUIRED - set(entry))
        if missing:
            raise SurfaceFailure("manifest shape: %s entry %r is missing %s; every number needs its "
                                 "value, display, source, command, recomputed flag and tolerance."
                                 % (rel, key, missing))
        extra = sorted(set(entry) - ENTRY_REQUIRED - ENTRY_OPTIONAL)
        if extra:
            raise SurfaceFailure("manifest shape: %s entry %r has unexpected key(s) %s."
                                 % (rel, key, extra))
        value = entry["value"]
        if not isinstance(value, SCALARS) or isinstance(value, complex):
            raise SurfaceFailure("manifest shape: %s entry %r value is a %s, not a scalar/string."
                                 % (rel, key, type(value).__name__))
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise SurfaceFailure("manifest shape: %s entry %r value is %r, not a finite number."
                                 % (rel, key, value))
        for field in ("display", "source", "command"):
            if not isinstance(entry[field], str):
                raise SurfaceFailure("manifest shape: %s entry %r %s is a %s, not a string."
                                     % (rel, key, field, type(entry[field]).__name__))
        if not isinstance(entry["recomputed"], bool):
            raise SurfaceFailure("manifest shape: %s entry %r recomputed is %r, not a boolean."
                                 % (rel, key, entry["recomputed"]))
        if not isinstance(entry["tolerance"], (int, float)) or isinstance(entry["tolerance"], bool):
            raise SurfaceFailure("manifest shape: %s entry %r tolerance is %r, not a number."
                                 % (rel, key, entry["tolerance"]))
        if "note" in entry and not isinstance(entry["note"], str):
            raise SurfaceFailure("manifest shape: %s entry %r note is not a string." % (rel, key))

    # 2. every figure it names is published as a placeholder/synthetic, or is a local-only path.
    registry = figure_registry()
    for path, text in _strings(doc, ""):
        for name in FIGURE_NAME.findall(text):
            figure_refs += 1
            if name not in registry:
                raise SurfaceFailure(
                    "manifest shape: %s %s names figure %r, which is not in build_walkthrough."
                    "FIGURES.\n  Every figure the manifest names must be a public figure/placeholder "
                    "or a local-only path under notebooks/figs/ (D-026)." % (rel, path, name))
        for named in FIGURE_PATH.findall(text):
            figure_refs += 1
            if PUBLIC_PATH.match(named):
                continue
            match = LOCAL_PATH.match(named)
            if match and match.group(1) in registry:
                continue
            raise SurfaceFailure(
                "manifest shape: %s %s names the path %r.\n  A figure path in the manifest must be "
                "a public placeholder (notebooks/figs/public/*.png) or a local-only figure under "
                "notebooks/figs/ (D-026); anything else is an untracked or unpublishable render."
                % (rel, path, named))

    # 3. tracked status is intentional: a number that is not in git is not reviewable.
    if not _tracked(rel):
        raise SurfaceFailure(
            "manifest shape: %s is not tracked by git.\n  The manifest is the record of every "
            "published number; an untracked copy can change without review (D-026). Run "
            "`git add %s`." % (rel, rel))

    return {"manifest": rel, "entries": len(entries), "figure_refs": figure_refs, "tracked": True}


# ----------------------------------------------------------------------------- public page

def start_public_page(html=None):
    """Start the publication audit in its own process; ``None`` when the page has not been built.

    A subprocess keeps the audit's stdout out of this process's stream, so the freeze report cannot
    lose a line to it, and the audit runs exactly as an operator would run it.
    """
    html = os.path.abspath(html or PUBLIC_HTML)
    if not os.path.isfile(html):
        return None
    return subprocess.Popen(
        [sys.executable, os.path.join(SOLUTION, "scripts", "audit_walkthrough_public.py"),
         "--html", html],
        cwd=SOLUTION, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def check_public_page(html=None, proc=None):
    """Run (or join) the publication audit; ``status='absent'`` is a clean skip, ``'pass'`` is 13/13."""
    html = os.path.abspath(html or PUBLIC_HTML)
    if not os.path.isfile(html):
        if proc is not None:
            proc.kill()
        return {"status": "absent", "html": html, "checks": 0, "failed": []}
    if proc is None:
        proc = start_public_page(html)
    output, _ = proc.communicate()
    lines = [line for line in (output or "").splitlines()
             if line.startswith(("PASS ", "FAIL "))]
    failed = [line for line in lines if line.startswith("FAIL ")]
    if proc.returncode != 0 or failed:
        raise SurfaceFailure(
            "public page audit failed (%d/%d checks passed, %s):\n%s\n"
            "  The page embeds or names something the public profile may not carry (D-002/D-026).\n"
            "  Nothing was changed; re-export with scripts/build_walkthrough_html.py --profile public."
            % (len(lines) - len(failed), len(lines), html,
               "\n".join("  " + line for line in (failed or lines[-3:] or
                                                 ["(no PASS/FAIL line; exit %d)" % proc.returncode]))))
    if len(lines) != AUDIT_CHECKS:
        raise SurfaceFailure(
            "public page audit printed %d checks, expected %d. A missing check is a gap, not a pass "
            "(B28); update AUDIT_CHECKS in scripts/freeze_surfaces.py when the audit changes."
            % (len(lines), AUDIT_CHECKS))
    return {"status": "pass", "html": html, "checks": len(lines), "failed": []}


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description="Assert the public page and the numbers manifest.")
    ap.add_argument("--manifest", default=MANIFEST, help="numbers manifest (default: %(default)s)")
    ap.add_argument("--html", default=PUBLIC_HTML, help="public page (default: %(default)s)")
    ap.add_argument("--json", help="write a machine-readable report here")
    args = ap.parse_args(argv)

    report, failed = {}, False
    for name, fn, kw in (("manifest", check_manifest, {"manifest": args.manifest}),
                         ("public_page", check_public_page, {"html": args.html})):
        try:
            report[name] = fn(**kw)
        except SurfaceFailure as exc:
            failed = True
            report[name] = {"error": str(exc)}
            print("FAIL %s\n  %s" % (name, str(exc).replace("\n", "\n  ")))

    manifest = report["manifest"]
    if "error" not in manifest:
        print("PASS manifest          %d entries, %d figure ref(s), tracked by git (%s)"
              % (manifest["entries"], manifest["figure_refs"], manifest["manifest"]))
    page = report["public_page"]
    if "error" not in page:
        print("PASS public-page       %s" % (
            "%d/%d checks (%s)" % (page["checks"], AUDIT_CHECKS, os.path.relpath(page["html"], SOLUTION))
            if page["status"] == "pass" else
            "skipped, page absent (%s)" % os.path.relpath(page["html"], SOLUTION)))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"pass": not failed, "surfaces": report}, fh, indent=1)
            fh.write("\n")
    print("RESULT %s" % ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
