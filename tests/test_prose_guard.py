"""Prose guard: fail the build on known-retired claims reappearing in shipped prose or docstrings.

This is a tripwire for an explicit, short list of claims the project has already corrected -- not a
semantic checker.  A hit reports ``file:line``, the matched phrase, the fact to quote instead and the
card that retired the claim, so the fix is obvious.  Extend :data:`RULES` when a new claim is retired.

Scope (shipped surface only): ``solution/**/*.py``, ``solution/docs/*.md``,
``solution/notebooks/*.ipynb`` (figures are not scanned), ``work/HUMAN_NOTES.md`` and
``research/Z01_final_report.md`` (the final report is prose the human will quote, and unlike the dated
digests it carries no correction banners).
``research/*.md`` is deliberately not enforced: those are dated digests that quote withdrawn claims
inside correction banners, so enforcing there would require the semantic reading this guard avoids.

Exemptions, by path and pattern only:
* both "Verbatim answers" blocks in ``work/HUMAN_NOTES.md`` -- the human's own words;
* any ``>`` blockquote whose block says it is attributed to the human/observer.

Needs no third-party package and reads text only.  Run with ``python -m unittest tests.test_prose_guard``.
"""
import glob
import os
import re
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)
ROOT = os.path.dirname(SOLUTION)
SELF = os.path.abspath(__file__)
HUMAN_NOTES = os.path.join("work", "HUMAN_NOTES.md")
Z01_REPORT = os.path.join("research", "Z01_final_report.md")

# Retired claims.  pattern: regex; applies: optional predicate on the line (context that makes the
# phrase the retired claim, e.g. a sha is only retired when presented as the shipped one); allowed:
# optional predicate that accepts an explicit historical/corrective framing.
CORRECTIVE = re.compile(
    r"correct(?:ed|ion)|withdrawn|retired|superseded|was wrong|never call|not\s+[\"']|"
    r"instead of|pre-(?:E66|B01|floor)|no longer",
    re.I)


RULES = [
    dict(
        name="E14 span-swap count",
        pattern=re.compile(r"\b2 of 474\b|\b474 of 476\b"),
        needles=("474",),
        fact="the corrected count is 84 of 464 sub-1.5 s spells at IoU >= .3 (43 at .5); "
             "the floor is a precision/recall trade",
        allowed=lambda line: bool(CORRECTIVE.search(line)),
        card="E99",
    ),
    dict(
        name="floor 'unmatchable' mechanism",
        pattern=re.compile(r"\bunmatchable\b|\bprun\w*\b[^.\n]{0,50}\b(?:cannot|can't|can not)\s+match\b"
                           r"|\bspells?\b[^.\n]{0,30}\bcannot\s+match\b", re.I),
        needles=("unmatchable", "match"),
        fact="the floor deletes ~84 real short possessions to remove ~380 false positives; "
             "it is a precision/recall trade (E99/B17), not free pruning of unmatchable scatter",
        allowed=lambda line: bool(CORRECTIVE.search(line)),
        card="E99/B17",
    ),
    dict(
        name="8255d2d1 presented as the shipped sha",
        pattern=re.compile(r"\b8255d2d1\b"),
        needles=("8255d2d1",),
        allowed=lambda line: bool(re.search(
            r"\bE66\b|\bQ26\b|pre-freeze|pre-B01|historical|superseded",
            line, re.I)),
        fact="the shipped sha is ee7b5b6b...46d8 (D-029); 8255d2d1...5e61 is E66/pre-freeze only",
        card="D-029",
    ),
    dict(
        name="fe37c459 called floor-off",
        pattern=re.compile(r"\bfe37c459\b"),
        needles=("fe37c459",),
        applies=lambda line: bool(re.search(r"floor[\s_-]*(?:off|disabled)|floor\s+is\s+off", line, re.I)),
        allowed=lambda line: bool(re.search(r"\bB04\b|add-?ons?|grown|sub-?gap", line, re.I)),
        fact="fe37c459...8b02 is B04 (grown D + sub-gap AND floor off); floor-off alone "
             "is 8255d2d1...5e61 (E66), per Q27",
        card="Q27",
    ),
    dict(
        name="mixed clock denominator for ball-in-play",
        pattern=re.compile(r"ours\s+54\.3\s*%|z\s*=?\s*-0\.6\s*sd|-0\.6\s*sd\s+on\s+rate", re.I),
        needles=("54.3", "-0.6"),
        allowed=lambda line: bool(CORRECTIVE.search(line)),
        fact="the like-for-like figure is 55.28% of the 103.95-min playing clock, z -0.46 (E105); "
             "54.3% is live-of-graded against an external playing-clock share",
        card="E105",
    ),
    dict(
        name=".456/.712 public-window numbers",
        pattern=re.compile(r"(?<![\d.])\.(?:456|712)(?![\d])"),
        needles=(".456", ".712"),
        applies=lambda line: bool(re.search(
            r"public|spellf1|macro|score|baseline|window|\u2192|->|retired|withdrawn|pre-freeze",
            line, re.I)),
        allowed=lambda line: bool(re.search(
            r"\bnot\b|\bnever\b|retired|withdrawn|pre-freeze|gone|stale|\u2192|->|superseded", line, re.I)),
        fact="the public window is spellF1 .4686 / macroF1 .7199 (D-029); .456/.712 is the "
             "retired B04/pre-floor value",
        card="D-029",
    ),
]

SCAN_GLOBS = [
    os.path.join("solution", "*.py"),
    os.path.join("solution", "possession", "**", "*.py"),
    os.path.join("solution", "scripts", "**", "*.py"),
    os.path.join("solution", "tests", "**", "*.py"),
    os.path.join("solution", "docs", "*.md"),
    os.path.join("solution", "notebooks", "*.ipynb"),
]
EXTRA_FILES = [HUMAN_NOTES, Z01_REPORT]
SKIP_PARTS = {"__pycache__", ".venv", ".git", ".cache", "possession.egg-info"}

VERBATIM_HEADING = re.compile(r"^#{1,6}\s+.*verbatim", re.I)
VERBATIM_END = re.compile(r"^(?:#{1,6}\s|---\s|(?:hat|What) I read)", re.I)
HUMAN_ATTRIBUTION = re.compile(r"\b(human|observer|entrant)\b", re.I)


class Hit:
    __slots__ = ("path", "line", "phrase", "fact", "card", "rule")

    def __init__(self, path, line, phrase, fact, card, rule):
        self.path = path
        self.line = line
        self.phrase = phrase
        self.fact = fact
        self.card = card
        self.rule = rule

    def __str__(self):
        return '{}:{}: "{}" -> {} [{}]'.format(
            self.path, self.line, self.phrase, self.fact, self.card)


def default_files(root=ROOT):
    """Paths relative to ``root`` for the scanned surface, sorted, without the guard itself."""
    rels = []
    for pattern in SCAN_GLOBS:
        for p in glob.glob(os.path.join(root, pattern), recursive=True):
            rel = os.path.relpath(p, root)
            if os.path.abspath(p) == SELF or set(rel.split(os.sep)) & SKIP_PARTS:
                continue
            rels.append(rel)
    for rel in EXTRA_FILES:
        if os.path.isfile(os.path.join(root, rel)):
            rels.append(rel)
    return sorted(set(rels))


def verbatim_spans(lines):
    """(first, last) 1-based line numbers of the human's 'Verbatim answers' blocks."""
    spans, start = [], None
    for i, line in enumerate(lines, 1):
        if VERBATIM_HEADING.match(line):
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif start is not None and VERBATIM_END.match(line):
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(lines)))
    return spans


def quote_block(lines, lineno):
    """The contiguous ``>`` block containing the 1-based line, or '' if it is not a blockquote."""
    i = lineno - 1
    if not (0 <= i < len(lines) and lines[i].lstrip().startswith(">")):
        return ""
    lo = i
    while lo > 0 and lines[lo - 1].lstrip().startswith(">"):
        lo -= 1
    hi = i
    while hi + 1 < len(lines) and lines[hi + 1].lstrip().startswith(">"):
        hi += 1
    return " ".join(lines[lo:hi + 1])


def is_exempt(relpath, lineno, line, lines, spans):
    if relpath == HUMAN_NOTES and any(a <= lineno <= b for a, b in spans):
        return True
    block = quote_block(lines, lineno)
    return bool(block and HUMAN_ATTRIBUTION.search(block))


def scan_text(text, relpath="scratch/probe.md"):
    lines = text.splitlines()
    spans = verbatim_spans(lines) if relpath == HUMAN_NOTES else []
    hits = []
    for i, line in enumerate(lines, 1):
        if is_exempt(relpath, i, line, lines, spans):
            continue
        low = line.lower()
        for rule in RULES:
            if rule.get("needles") and not any(n in low for n in rule["needles"]):
                continue
            if "applies" in rule and not rule["applies"](line):
                continue
            if "allowed" in rule and rule["allowed"](line):
                continue
            for m in rule["pattern"].finditer(line):
                hits.append(Hit(relpath, i, m.group(0), rule["fact"], rule["card"], rule["name"]))
    return hits


def scan(root=ROOT, rels=None):
    rels = default_files(root) if rels is None else rels
    hits = []
    for rel in rels:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        hits += scan_text(text, rel)
    return hits


def format_hits(hits):
    if not hits:
        return "no retired claims"
    return ("retired claim(s) found (prose guard; extend RULES in tests/test_prose_guard.py):\n  "
            + "\n  ".join(str(h) for h in hits))


class ProseGuard(unittest.TestCase):
    def test_current_tree_has_no_retired_claims(self):
        hits = scan()
        self.assertEqual(hits, [], format_hits(hits))


class ProseGuardDetects(unittest.TestCase):
    """The guard is non-vacuous: it fails on planted violations and honours the exemptions."""

    def test_planted_violation_fails_in_scratch_copy(self):
        rels = default_files()
        with tempfile.TemporaryDirectory() as tmp:
            for rel in rels:
                dst = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(os.path.join(ROOT, rel), dst)
            self.assertEqual(scan(tmp, rels), [], "scratch copy must start clean")

            victim = os.path.join(tmp, "solution", "possession", "postprocess.py")
            with open(victim, "a", encoding="utf-8") as fh:
                fh.write('\n\ndef _planted():\n    """2 of 474 short spells cannot match."""\n')

            hits = scan(tmp, rels)
            planted = [h for h in hits if h.path == os.path.join("solution", "possession", "postprocess.py")]
            self.assertTrue(planted, "planted '2 of 474' was not flagged:\n" + format_hits(hits))
            self.assertTrue(any(h.phrase == "2 of 474" for h in planted), format_hits(planted))
            self.assertIn("E99", format_hits(planted))
            self.assertIn("84 of 464", format_hits(planted))

    def test_each_rule_flags_its_claim(self):
        cases = [
            ("2 of 474", "E99"),
            ("474 of 476", "E99"),
            ("the floor is pruning spells that cannot match", "E99/B17"),
            ("free pruning of unmatchable scatter", "E99/B17"),
            ("shipped sha 8255d2d1...5e61", "D-029"),
            ("floor off reproduces fe37c459...8b02", "Q27"),
            ("public spellF1 .456 / macroF1 .712", "D-029"),
        ]
        for text, card in cases:
            hits = scan_text(text)
            self.assertTrue(hits, "not flagged: " + text)
            self.assertTrue(any(h.card == card for h in hits), "{} -> {}".format(text, format_hits(hits)))

    def test_historical_and_corrective_framing_is_allowed(self):
        self.assertEqual(scan_text("E66's sha 8255d2d1...5e61 was the pre-freeze build"), [])
        self.assertEqual(scan_text("fe37c459...8b02 is B04: add-ons and floor off"), [])
        self.assertEqual(scan_text("the public window is now .4686/.7199, not .456/.712"), [])
        # an unrelated probability is not the public-window number
        self.assertEqual(scan_text("rng.binomial(100, .456)"), [])

    def test_human_verbatim_blocks_are_exempt(self):
        with open(os.path.join(ROOT, HUMAN_NOTES), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        spans = verbatim_spans(lines)
        self.assertEqual(len(spans), 2, "expected both 'Verbatim answers' blocks, got {}".format(spans))
        first = next(i for i, ln in enumerate(lines, 1) if "from the corner" in ln)
        self.assertTrue(is_exempt(HUMAN_NOTES, first, lines[first - 1], lines, spans))
        # the orchestrator's interpretation below the block is not the human's words
        interp = next(i for i, ln in enumerate(lines, 1) if "Consistent with our model" in ln)
        self.assertFalse(is_exempt(HUMAN_NOTES, interp, lines[interp - 1], lines, spans))

    def test_attributed_blockquote_is_exempt(self):
        self.assertEqual(scan_text("> 2 of 474\n> -- the human, watch notes\n"), [])
        self.assertEqual(len(scan_text("> 2 of 474\n")), 1, "an unattributed quote must still be flagged")

    def test_reports_file_line_phrase_and_replacement(self):
        hit = scan_text("x\n2 of 474\n")[0]
        self.assertEqual((hit.line, hit.phrase), (2, "2 of 474"))
        self.assertIn("84 of 464", str(hit))
        self.assertIn("E99", str(hit))


if __name__ == "__main__":
    unittest.main()
