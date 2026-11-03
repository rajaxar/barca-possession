"""Constant-provenance register: discover every shipped module, then check its constants.

This is the machine that answers the advisors' question *"is any constant trained, fitted or tuned on
the wrong data?"* without an auditor reading prose.  ``docs/constant_provenance.json`` tags every
module-level assignment in the shipped decision path with exactly one provenance class; this module
re-derives the assignment set from the AST (so a new constant cannot arrive untagged), validates the
manifest, and renders ``docs/CONSTANTS.md`` from it.

The module scope is **discovered, not listed in this file**: every ``possession/*.py`` except
``__init__.py`` must appear in the register either under ``modules`` (registered; every constant
tagged) or under ``exempt`` (named with a category and a justification).  A new module that is in
neither fails, so a constant cannot arrive in a file nobody remembered to add to a list.

Every entry also records the AST line of its assignment.  A constant that moves fails with the line
the AST now reports, so refreshing a citation is a one-number edit.

    python scripts/constant_provenance.py --check     # non-zero exit and a list on any problem
    python scripts/constant_provenance.py --write     # regenerate docs/CONSTANTS.md from the JSON

Needs no match data and no third-party package.  The unittest is ``tests/test_constant_provenance.py``.
"""
import argparse
import ast
import json
import os
import sys

SOLUTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join("docs", "constant_provenance.json")
DOC = os.path.join("docs", "CONSTANTS.md")
SCHEMA = 2
PACKAGE = "possession"

# The categories an exemption may claim.  A registered module has one tag per constant; an exempt
# module has one category plus a written justification, so the gap is named rather than accidental.
EXEMPT_CATEGORIES = (
    "not shipped",       # research arm, renderer, closed decoder: nothing on the CSV path imports it
    "report-only",       # runs against this match but only prints a diagnostic; never writes a label
    "loader/parser",     # reads/reshapes provider data; parses arrays, holds no decision value
    "perception",        # builds features from tracking; feeds a model, not a label directly
    "dev harness",       # scoring/evaluation machinery; defines the measurement, not the build
)

# The five provenance classes the card asks for, plus one explicit non-provenance class.  `scaffold`
# is restricted by check() to computed objects (paths, type aliases), so a literal cannot hide there.
TAGS = {
    "external": dict(
        label="external (card)",
        definition="Set on external games/matches or by a recorded project decision and cited to a card "
                   "(E/R/S/Q, D-0xx). Never selected on this match's ground truth, video or public window.",
    ),
    "structure": dict(
        label="published structure (D-007)",
        definition="Published by the organizers (D-007, FAQ, games.py, starter kit): half boundaries, "
                   "kickoffs, match length, pitch, label encoding, asset names, frame rate. Structure is "
                   "not evidence about what happened.",
    ),
    "physical": dict(
        label="physical/Laws",
        definition="A physical, physiological or Laws-of-the-Game fact (pitch geometry, sprint speed, "
                   "walk/jog boundary, restart geometry); not fitted to any dataset.",
    ),
    "match": dict(
        label="label-free match diagnostic",
        definition="Measured or fitted on THIS match's tracking with no labels and no video: ATD run-length "
                   "quantiles, the low-pass widths, the simulator's fitted table, report-only measurements. "
                   "This is the one class an auditor must read carefully.",
    ),
    "video": dict(
        label="video-derived (report-only)",
        definition="An agreement rate or diagnostic computed against an automatic verdict derived from the "
                   "broadcast video. Report-only: it never enters the build, a label path or a shipped decision.",
    ),
    "guard": dict(
        label="guard",
        definition="A defensive bound, canary, clip or pre-registered decision rule. It can fail the build "
                   "or clip a value; it never selects a label and is never tuned on this match's answers.",
    ),
    "scaffold": dict(
        label="scaffold (not provenance)",
        definition="A filesystem path or a type alias: a computed object with no decision value. check() "
                   "refuses this tag on a literal, so a number cannot be hidden here.",
    ),
}
TAG_KEYS = tuple(TAGS)
PROVENANCE_TAGS = tuple(k for k in TAG_KEYS if k != "scaffold")


# ------------------------------------------------------------------ discovery
class Assignment:
    __slots__ = ("name", "lineno", "value", "single", "index")

    def __init__(self, name, lineno, value, single, index=None):
        self.name = name
        self.lineno = lineno
        self.value = value
        self.single = single
        self.index = index


def _flatten_target(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for elt in node.elts:
            out += _flatten_target(elt)
        return out
    return []


def _local_defs(tree):
    return {n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def discovered(root=SOLUTION):
    """Every ``possession/*.py`` on disk except ``__init__``, sorted.  No hardcoded module list.

    ``__init__.py`` is excluded because it is a package marker: a constant there would be an import
    side effect, and ``__init__`` is not a module anyone edits to ship a value.
    """
    folder = os.path.join(root, PACKAGE)
    return ["%s/%s" % (PACKAGE, name) for name in sorted(os.listdir(folder))
            if name.endswith(".py") and name != "__init__.py"]


def module_assignments(path):
    """Module-level assignments in ``path``, by AST.

    Includes tuple/multiple targets (``FPS, STRIDE = 25, 5``), annotated and augmented assignments, and
    assignments inside a module-level if/try/for.  Excludes dunder names, function-local variables, and
    aliases of a callable defined in the same module (``d_proxy = proxy`` is code, not a constant).
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    defs = _local_defs(tree)
    found = []

    def walk(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [n for t in targets for n in _flatten_target(t)]
                names = [n for n in names if not (n.startswith("__") and n.endswith("__"))]
                if (isinstance(node.value, ast.Name) and node.value.id in defs
                        and len(names) == 1):
                    continue  # a callable alias, not a constant
                # A flat tuple/list target pairs element i with RHS element i (``FPS, STRIDE = 25, 5``).
                index = None
                if len(targets) == 1 and isinstance(targets[0], (ast.Tuple, ast.List)):
                    flat = [n for n in _flatten_target(targets[0])]
                    if (len(flat) == len(names) and isinstance(node.value, (ast.Tuple, ast.List))
                            and len(node.value.elts) == len(flat)):
                        index = {n: i for i, n in enumerate(names)}
                for name in names:
                    found.append(Assignment(name, node.lineno, node.value, len(names) == 1,
                                            None if index is None else index[name]))
            elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
                for attr in ("body", "orelse", "finalbody"):
                    walk(getattr(node, attr, []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    walk(handler.body)
    walk(tree.body)
    return found


def dict_fields(node):
    """String keys of a dict literal or a ``dict(...)`` call; [] for anything else."""
    if isinstance(node, ast.Dict):
        keys = []
        for key in node.keys:
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                return []
            keys.append(key.value)
        return keys
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if any(isinstance(a, ast.Starred) for a in node.args):
            return []
        return [k.arg for k in node.keywords if k.arg]
    return []


def field_value(node, key):
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
        return None
    for kw in getattr(node, "keywords", []):
        if kw.arg == key:
            return kw.value
    return None


def _string_leaves(node):
    """True only when every leaf of the value is a string literal (a pure name table)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Dict):
        return all(_string_leaves(k) and _string_leaves(v) for k, v in zip(node.keys, node.values))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_string_leaves(e) for e in node.elts)
    return False


def required_fields(node):
    """Fields that must be tagged: string-keyed dicts carrying a number or a non-literal value.

    Pure name tables (``config.ASSET_CANDIDATES``) and integer-keyed maps (``structure.KICKOFFS``)
    are single constants and need no field rows.
    """
    keys = dict_fields(node)
    if not keys:
        return []
    values = node.values if isinstance(node, ast.Dict) else [k.value for k in node.keywords]
    if all(_string_leaves(v) for v in values):
        return []
    return keys


def literal(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def snapshot(assignment):
    """JSON-safe literal value of a name, or None when it is computed."""
    value = None
    if assignment.single:
        value = literal(assignment.value)
    elif assignment.index is not None and isinstance(assignment.value, (ast.Tuple, ast.List)):
        value = literal(assignment.value.elts[assignment.index])
    return json.loads(json.dumps(value)) if value is not None else None


def _fmt(value):
    return repr(value) if not isinstance(value, list) else repr(tuple(value))


# ------------------------------------------------------------------ manifest
def load(root=SOLUTION):
    with open(os.path.join(root, MANIFEST), encoding="utf-8") as fh:
        return json.load(fh)


def audit(root=SOLUTION):
    """``(problems, stats)`` for the register at ``root``; empty problems means every constant is tagged.

    ``stats`` counts what was actually inspected: modules discovered on disk, registered, exempt, and
    the module-level constants and dict fields re-derived from the AST.  The freeze check prints them,
    so a silently empty scan cannot look like a pass.
    """
    problems = []
    stats = dict(constants=0, fields=0, modules=0, exempt=0, discovered=0)
    try:
        manifest = load(root)
    except (OSError, ValueError) as exc:
        return ["cannot read %s: %s" % (MANIFEST, exc)], stats
    if manifest.get("schema") != SCHEMA:
        problems.append("schema is %r, expected %r" % (manifest.get("schema"), SCHEMA))
    modules = manifest.get("modules", {})
    exempt = manifest.get("exempt", {})
    present = discovered(root)
    stats["discovered"] = len(present)
    for mod in present:
        if mod in exempt:
            stats["exempt"] += 1
            scope = exempt[mod]
            if mod in modules:
                problems.append("module is both registered and exempt: %s -- keep exactly one" % mod)
            if scope.get("category") not in EXEMPT_CATEGORIES:
                problems.append("exempt module %s has category %r, expected one of %s"
                                % (mod, scope.get("category"), list(EXEMPT_CATEGORIES)))
            if not scope.get("why"):
                problems.append("exempt module %s has no written justification" % mod)
            continue
        if mod not in modules:
            count = len(module_assignments(os.path.join(root, mod)))
            problems.append("unregistered module: %s (%d module-level constant%s) -- add it to "
                            "\"modules\" in %s with a tag per constant, or name it in \"exempt\" "
                            "with a category and a justification"
                            % (mod, count, "" if count == 1 else "s", MANIFEST))
            continue
        entries = modules[mod].get("constants", {})
        assigns = module_assignments(os.path.join(root, mod))
        stats["modules"] += 1
        seen = set()
        for a in assigns:
            stats["constants"] += 1
            seen.add(a.name)
            entry = entries.get(a.name)
            if entry is None:
                problems.append("untagged constant: %s:%d %s = %s -- add an entry to %s"
                                % (mod, a.lineno, a.name, _fmt(literal(a.value)), MANIFEST))
                continue
            tag = entry.get("tag")
            if tag not in TAG_KEYS:
                problems.append("%s:%s has tag %r, not one of %s" % (mod, a.name, tag, list(TAG_KEYS)))
            if tag == "scaffold" and not isinstance(a.value, ast.Call):
                problems.append("%s:%s is tagged scaffold but is not a computed object; "
                                "scaffold may not hide a literal" % (mod, a.name))
            for key in ("source", "why"):
                if not entry.get(key):
                    problems.append("%s:%s has no %s" % (mod, a.name, key))
            line = entry.get("line")
            if not isinstance(line, int):
                problems.append("missing line: %s:%s has no integer \"line\" in the register "
                                "(the AST says line %d) -- add \"line\": %d"
                                % (mod, a.name, a.lineno, a.lineno))
            elif line != a.lineno:
                problems.append("stale line: %s:%s is at line %d, register records %d -- set "
                                "\"line\" to %d in %s"
                                % (mod, a.name, a.lineno, line, a.lineno, MANIFEST))
            expected = entry.get("value")
            actual = snapshot(a)
            if expected is not None and actual != expected:
                problems.append("value drift: %s:%s is %r, manifest records %r"
                                % (mod, a.name, actual, expected))
            fields = entry.get("fields", {})
            want = required_fields(a.value)
            stats["fields"] += len(want)
            for key in want:
                if key not in fields:
                    problems.append("untagged field: %s.%s[%r] -- add a field entry to %s"
                                    % (mod, a.name, key, MANIFEST))
                    continue
                fentry = fields[key]
                if fentry.get("tag") not in TAG_KEYS:
                    problems.append("%s.%s[%r] has tag %r" % (mod, a.name, key, fentry.get("tag")))
                if not fentry.get("source"):
                    problems.append("%s.%s[%r] has no source" % (mod, a.name, key))
                fexp = fentry.get("value")
                factual = literal(field_value(a.value, key))
                if fexp is not None and factual is not None and json.loads(json.dumps(factual)) != fexp:
                    problems.append("value drift: %s.%s[%r] is %r, manifest records %r"
                                    % (mod, a.name, key, factual, fexp))
            for key in fields:
                if key not in want:
                    problems.append("stale field entry: %s.%s[%r] is not a tagged field of the constant"
                                    % (mod, a.name, key))
        for name in entries:
            if name not in seen:
                problems.append("stale entry: %s:%s is no longer a module constant" % (mod, name))
    for mod in modules:
        if mod not in present:
            problems.append("registered module is not on disk: %s -- remove it from \"modules\" or "
                            "restore the file" % mod)
    for mod in exempt:
        if mod not in present:
            problems.append("stale exemption: %s is not on disk -- remove it from \"exempt\"" % mod)
    return problems, stats


def check(root=SOLUTION):
    """All problems with the register at ``root``; an empty list means every constant is tagged."""
    return audit(root)[0]


# ------------------------------------------------------------------ rendering
def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def _short(value):
    if isinstance(value, (list, tuple)):
        if len(value) > 6:
            return "(%s, %s, %s, ..., %s)" % (value[0], value[1], value[2], value[-1])
        return "(" + ", ".join(repr(v) for v in value) + ")"
    if isinstance(value, dict):
        items = list(value.items())
        body = ", ".join("%s: %s" % (k, _short(v)) for k, v in items[:4])
        return "{" + body + (", ..." if len(items) > 4 else "") + "}"
    if value is None:
        return "computed"
    return repr(value)


def counts(manifest):
    const, fields = {}, {}
    for entry in manifest["modules"].values():
        for c in entry["constants"].values():
            const[c["tag"]] = const.get(c["tag"], 0) + 1
            for f in c.get("fields", {}).values():
                fields[f["tag"]] = fields.get(f["tag"], 0) + 1
    return const, fields


def render(manifest):
    const, fields = counts(manifest)
    n_const = sum(const.values())
    n_fields = sum(fields.values())
    exempt = manifest.get("exempt", {})
    out = []
    add = out.append
    add("# CONSTANTS — provenance of every module-level constant in the shipped path")
    add("")
    add("Generated from [`constant_provenance.json`](constant_provenance.json) by "
        "`scripts/constant_provenance.py --write`. **Do not edit this file by hand**; edit the JSON "
        "and run the script. The machine check is `tests/test_constant_provenance.py`.")
    add("")
    add("## The question this answers")
    add("")
    add("> Is any constant trained, fitted, tuned or calibrated on the wrong data?")
    add("")
    add("Every module-level assignment in a registered module carries exactly one tag below. The tag "
        "says what the constant may be justified by. `match` is the only class that is measured on "
        "this match, and it is **label-free**: no possession label, no hand label and no video enters "
        "it. No constant in the register is tagged with this match's ground truth or with the public "
        "5 minutes, because no such constant exists (Q36/Q37). Every other module is exempt with a "
        "reason in [Scope](#scope), including the loader/parser boundary still to be registered.")
    add("")
    add("## Tag vocabulary")
    add("")
    add("| tag | meaning |")
    add("|---|---|")
    for key in TAG_KEYS:
        add("| `%s` | %s |" % (TAGS[key]["label"], _cell(TAGS[key]["definition"])))
    add("")
    add("## Counts")
    add("")
    add("| tag | constants | dict fields |")
    add("|---|---:|---:|")
    for key in TAG_KEYS:
        add("| `%s` | %d | %d |" % (key, const.get(key, 0), fields.get(key, 0)))
    add("| **total** | **%d** | **%d** |" % (n_const, n_fields))
    add("")
    add("Dict-valued constants (the degrader's `PARAMS`, the proxies' parameter sets) expose their keys "
        "as field rows: the container carries a role tag and every value carries its own.")
    add("")
    add("## Scope")
    add("")
    add("The scope is **discovered, not listed in code**: every `possession/*.py` except `__init__.py` "
        "is either registered above (every module-level assignment carries a tag) or exempted below "
        "with a reason. `scripts/constant_provenance.py` fails on a module in neither, on a registered "
        "constant that is not at the line recorded here, and on a stale exemption. A new module "
        "therefore cannot pass unseen, and a new constant cannot arrive untagged.")
    add("")
    add("Registered (%d): the stage writers (`pipeline` and `make_submission` declare no constant), the "
        "training-data simulator, `config`, the modules a shipped stage reads values from "
        "(`dproxy`, `ballfree`), and `features`, whose columns feed the ball-free model."
        % sum(1 for m in manifest["modules"]))
    add("")
    add("Exempt (%d), each named with its justification:" % len(exempt))
    add("")
    add("| module | category | why it holds no shipped decision value |")
    add("|---|---|---|")
    for mod in sorted(exempt):
        scope = exempt[mod]
        add("| `%s` | %s | %s |" % (mod, scope.get("category"), _cell(scope.get("why"))))
    add("")
    add("`features` is registered rather than exempt because `ballfree` reads its columns: it is the "
        "one E113 perception module on a label path, and B30 folded it in so no exempt module feeds a "
        "shipped stage. What remains is E113's loader/parser boundary — `data`, `external` and "
        "`evaluate` — named above and left to a later card.")
    add("")
    add("## Fitted on this match without labels")
    add("")
    add("These are the constants Q36 had to dig for. None is trained on a label or a video; each is a "
        "statistic of our own tracking. They are the boundary the advisors should rule on if they read "
        "\"parameters must be tuned on external data\" as covering every number in the build.")
    add("")
    match_rows = [(mod, name, c) for mod, m in manifest["modules"].items()
                  for name, c in m["constants"].items() if c["tag"] == "match"]
    match_rows.sort()
    add("| constant | what it is | source |")
    add("|---|---|---|")
    for mod, name, c in match_rows:
        add("| `%s.%s` | %s | %s |" % (os.path.splitext(os.path.basename(mod))[0], name,
                                       _cell(c["why"]), _cell(c["source"])))
    for mod, name, c in match_rows:
        for fname, f in c.get("fields", {}).items():
            if f["tag"] != "match":
                continue
            add("| `%s.%s[%r]` | %s | %s |"
                % (os.path.splitext(os.path.basename(mod))[0], name, fname,
                   _cell(f["why"]), _cell(f["source"])))
    add("")
    add("Explicitly named: the degrader's ATD **run-length quantiles** (`ATD_GAP_Q`, `ATD_PRESENT_Q`, "
        "`ATD_SEG_Q` — segment/gap lengths, not noise), the **ball low-pass width** "
        "(`degrade.PARAMS['smooth_ball_sigma'] = 8`; `dgate.LOWPASS_SIGMA = 8` is the separate player "
        "filter applied to the external training tracks), and the footprint guards below.")
    add("")
    add("## Guard families")
    add("")
    add("No guard selects a label: each raises, clips or refuses, so a guard can only stop a bad build.")
    add("")
    add("| family | constants | role |")
    add("|---|---|---|")
    add("| dgate added-D share | `dgate.MIN_D_SHARE_ADDED`, `MAX_D_SHARE_ADDED` | two-sided canary on added D |")
    add("| growd relative footprint | `growd.MAX_CHANGED_PER_ELIGIBLE` | changed frames ≤ 1.10× the graded "
        "mask frames the input left live — the canary that does not move when an upstream stage changes |")
    add("| growd absolute cap | `growd.MAX_D_SHARE_ADDED` | runaway cap only (BAR-ATH .02895, dev max .126) |")
    add("| growd mask share | `growd.MIN_MASK_SHARE`, `MAX_MASK_SHARE` | two-sided bound on growd's own grown mask |")
    add("| growd flickers | `growd.MAX_FLICKER_ADDED` | added <1 s spells |")
    add("| ingap present-ball share | `ingap.MAX_PRESENT_SHARE` | **present-ball share bar**: ≤5% of ingap's "
        "changes may land on ball-present frames |")
    add("| ingap footprint | `ingap.MAX_CHANGED_SHARE`, `MAX_SPELL_RATIO`, `PH_CLIP` | changed share, spell ratio, log clip |")
    add("| subgap added-D share | `subgap.MIN_ADDED`, `MAX_ADDED` | two-sided added-D bound |")
    add("| leak guards | `ballfree.FORBIDDEN_PREFIXES` | columns the ball-free model may never read |")
    add("| adoption protocol | `dproxy.MAX_NAN_SHARE`, `MIN_DOMAIN_FRAMES`, `MIN_LEAD_GAIN`, "
        "`MIN_CONDITIONAL_LEAD`, `MAX_ACTIVE_SHARE_WARN`, `ALPHA` | pre-registered decision rule |")
    add("")
    add("## Every constant")
    add("")
    for mod in manifest["modules"]:
        entry = manifest["modules"][mod]
        add("### `%s`" % mod)
        add("")
        if entry.get("note"):
            add(entry["note"])
            add("")
        add("| constant | value | tag | source | why |")
        add("|---|---|---|---|---|")
        for name, c in entry["constants"].items():
            add("| `%s` | `%s` | `%s` | %s | %s |"
                % (name, _short(c.get("value")), c["tag"], _cell(c["source"]), _cell(c["why"])))
            for fname, f in c.get("fields", {}).items():
                add("| `%s[%r]` | `%s` | `%s` | %s | %s |"
                    % (name, fname, _short(f.get("value")), f["tag"], _cell(f["source"]), _cell(f["why"])))
        add("")
    add("## Adding a constant")
    add("")
    add("1. Add module-level assignment (any target form).")
    add("2. Add an entry to `docs/constant_provenance.json` under that module with `tag`, `source`, "
        "`why` and the AST `line`.")
    add("3. If it is a string-keyed dict carrying numbers, add a field entry per key.")
    add("4. Run `python scripts/constant_provenance.py --write` then `make test`; an untagged name, an "
        "untagged field, a stale entry, a stale line or a changed value fails "
        "`tests/test_constant_provenance.py`.")
    add("")
    add("A **new module** is a scope decision, not an automatic pass: add `\"modules\"` entry with a "
        "tag per constant, or an `\"exempt\"` entry with a category and a written justification. Until "
        "one exists, the check fails with `unregistered module: possession/<file>.py`.")
    add("")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 and list every problem")
    ap.add_argument("--write", action="store_true", help="regenerate docs/CONSTANTS.md from the JSON")
    ap.add_argument("--root", default=SOLUTION, help="repository root (default: solution/)")
    args = ap.parse_args(argv)
    if args.write:
        manifest = load(args.root)
        with open(os.path.join(args.root, DOC), "w", encoding="utf-8") as fh:
            fh.write(render(manifest))
        const, fields = counts(manifest)
        print("wrote %s: %d constants, %d dict fields, %s" % (DOC, sum(const.values()), sum(fields.values()),
                                              ", ".join("%s %d+%d" % (k, const.get(k, 0), fields.get(k, 0)) for k in TAG_KEYS)))
    if args.check or not args.write:
        problems, stats = audit(args.root)
        for p in problems:
            print("FAIL", p)
        print("constant provenance: %d constants, %d dict fields, %d problem(s)"
              % (stats["constants"], stats["fields"], len(problems)))
        print("scope: %d modules discovered, %d registered, %d exempt"
              % (stats["discovered"], stats["modules"], stats["exempt"]))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
