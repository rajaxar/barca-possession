"""Build ``notebooks/numbers.json``: every number the walkthrough quotes, with its provenance.

Two kinds of entry:

* **recomputed** -- measured from the live code right now (the frozen dev-v1 reference table, the
  current submission's spell/D morphology, the synthetic spellF1 demonstration).  ``--check``
  rebuilds these and fails when one has moved beyond its recorded tolerance.
* **read** -- an expensive result taken from a research digest.  The value is *parsed out of the
  digest* (never typed into the notebook), and ``--check`` fails if the pattern no longer matches
  **or if the parsed value has moved** (a corrected digest must regenerate the manifest, Q34).

Each entry is ``{"value", "display", "source", "command", "recomputed", "tolerance"}``.
The notebook builder substitutes ``{key}`` from this file, so no number in the prose is hand-typed.

    cd solution
    .venv/bin/python scripts/walkthrough_numbers.py              # write notebooks/numbers.json
    .venv/bin/python scripts/walkthrough_numbers.py --check      # fail on drift
    .venv/bin/python scripts/walkthrough_numbers.py --show gate_dev

Reads the match for the morphology entries; never reads or scores the public answers.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import sys

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION_DIR)
sys.path.insert(0, SOLUTION_DIR)

DEFAULT_OUT = os.path.join(SOLUTION_DIR, "notebooks", "numbers.json")
SHIPPED_STACK = "3a4e368"
FPS = 25


# ----------------------------------------------------------------------------- formatting

def _score(x):
    s = "%.4f" % float(x)
    return s[1:] if s.startswith("0.") else ("-." + s[2:] if s.startswith("-0.") else s)


def _delta(x):
    s = "%+.4f" % float(x)
    return s.replace("+0.", "+.").replace("-0.", "-.")


def _pct(x, nd=0):
    return ("%." + str(nd) + "f%%") % (100.0 * float(x))


def _score2(x):
    s = "%.2f" % float(x)
    return s[1:] if s.startswith("0.") else s


def _num(x, nd=2):
    return ("%." + str(nd) + "f") % float(x)


def _trim(x):
    s = _delta(x)
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return s


class Manifest:
    def __init__(self):
        self.e = {}
        self.live = {}

    def add(self, key, value, *, source, command, recomputed, tolerance=0.0,
            display=None, note=None, register=True):
        if key in self.e:
            raise KeyError("duplicate number key %r" % key)
        entry = {"value": value, "display": display if display is not None else str(value),
                 "source": source, "command": command, "recomputed": bool(recomputed),
                 "tolerance": float(tolerance)}
        if note:
            entry["note"] = note
        self.e[key] = entry
        if register:
            self.live[key] = value

    def add_range(self, key, value, lo, hi, *, fmt=_delta, source, command, recomputed,
                  tolerance=0.0, note=None):
        """One entry whose display carries the interval: ``+.0728 [+.0649, +.0807]``."""
        self.add(key, value, source=source, command=command, recomputed=recomputed,
                 tolerance=tolerance, display="%s [%s, %s]" % (fmt(value), fmt(lo), fmt(hi)),
                 note=note)


def _read(path, pattern, *, group=1, cast=float, source=None):
    """First capture of ``pattern`` in ``path``; raises with the file name when absent."""
    full = os.path.join(ROOT, path)
    text = open(full, encoding="utf-8").read()
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError("number pattern not found in %s: %r" % (path, pattern))
    raw = m.group(group)
    norm = str(raw).replace("−", "-").replace("–", "-")
    if cast is not float:
        try:
            return cast(norm), raw
        except (TypeError, ValueError):
            pass
    if norm.endswith("%"):
        return float(norm[:-1]) / 100.0, raw
    return cast(norm), raw


def _grep(path, pattern):
    return "grep -n -m1 '%s' %s" % (pattern.replace("'", "'\\''"), path)


# ----------------------------------------------------------------------------- recomputed

def _dev_ref(m):
    """The four frozen dev-v1 reference rows and their marginal deltas (D-021 / S40)."""
    from possession import dev_v1
    refs = dev_v1.reference_results()
    names = {"smooth50": "smooth50", "baseline_plus": "baseline_plus", "gate": "gate", "ship": "ship"}
    vals = {k: float(refs[k].summary()["headline"]) for k in names}
    cmd = "cd solution && .venv/bin/python -c \"from possession import dev_v1; r=dev_v1.reference_results(); print({k: round(v.summary()['headline'],4) for k,v in r.items()})\""
    for k, label in (("smooth50", "The organizers' 2-second baseline (smooth50)."),
                     ("baseline_plus", "Baseline+: nearest player within the radius, else hold, with hysteresis and merge."),
                     ("gate", "Baseline+ and the missing-ball stoppage rule."),
                     ("ship", "The shipped file: gate then the in-gap decoder.")):
        m.add("%s_dev" % k, round(vals[k], 4), source="solution/possession/dev_v1.py (reference_results)",
              command=cmd, recomputed=True, tolerance=5e-5, display=_score(vals[k]), note=label)

    def cmp(a, b):
        c = dev_v1.compare_dev_v1(a, b)
        h = c["headline"]
        return float(h["delta"]), float(h["ci"][0]), float(h["ci"][1]), int(h["neg"])

    d, lo, hi, neg = cmp("baseline_plus", "smooth50")
    m.add_range("baseline_plus_over_smooth50", round(d, 4), round(lo, 4), round(hi, 4),
                source="solution/possession/dev_v1.py (compare_dev_v1)", command=cmd, recomputed=True,
                tolerance=5e-5, note="games with a negative delta: %d/10" % neg)
    d, lo, hi, neg = cmp("gate", "baseline_plus")
    m.add_range("gate_over_baseline_plus", round(d, 4), round(lo, 4), round(hi, 4),
                source="solution/possession/dev_v1.py (compare_dev_v1)", command=cmd, recomputed=True,
                tolerance=5e-5, note="games with a negative delta: %d/10" % neg)
    d, lo, hi, neg = cmp("ship", "gate")
    m.add_range("ingap_over_gate", round(d, 4), round(lo, 4), round(hi, 4),
                source="solution/possession/dev_v1.py (compare_dev_v1)", command=cmd, recomputed=True,
                tolerance=5e-5, note="games with a negative delta: %d/10" % neg)
    d, lo, hi, neg = cmp("ship", "smooth50")
    m.add_range("ship_over_smooth50", round(d, 4), round(lo, 4), round(hi, 4),
                source="solution/possession/dev_v1.py (compare_dev_v1)", command=cmd, recomputed=True,
                tolerance=5e-5, note="games with a negative delta: %d/10" % neg)

    # The frozen D-028 row is one run of the whole stack (B01, re-run by Q28); it is not a sum of
    # components and is not in the cached reference table, so it is read from the digest.
    def from_digest(key, pattern, display, note, cast=float):
        value, _ = _read("research/B01_result.md", pattern, cast=cast)
        m.add(key, value, source="research/B01_result.md", recomputed=False,
              command=_grep("research/B01_result.md", pattern), display=display, note=note)

    from_digest("master_dev", r"master \*\*(\.4247)\*\*", ".4247",
                "the frozen D-028 stack, one run of the whole pipeline")
    d, _ = _read("research/B01_result.md", r"master − ship \| \*\*(\+\.0208)\*\*", cast=float)
    lo, _ = _read("research/B01_result.md", r"master − ship \| \*\*\+\.0208\*\* \| \[\+(\.0169),", cast=float)
    hi, _ = _read("research/B01_result.md", r"master − ship \| \*\*\+\.0208\*\* \| \[\+\.0169, \+(\.0250)\]", cast=float)
    m.add_range("master_over_ship", round(d, 4), round(lo, 4), round(hi, 4),
                source="research/B01_result.md", command=_grep("research/B01_result.md", "master − ship"),
                recomputed=False, note="frozen master − ship, 0/10 games negative")
    for suffix, v in (("lo", lo), ("hi", hi)):
        m.add("master_over_ship_" + suffix, round(v, 4), source="research/B01_result.md",
              command=_grep("research/B01_result.md", "master − ship"), recomputed=False,
              display=_delta(v))
    d, _ = _read("research/B01_result.md", r"master − Baseline\+ \| \*\*(\+\.0988)\*\*", cast=float)
    lo, _ = _read("research/B01_result.md", r"master − Baseline\+ \| \*\*\+\.0988\*\* \| \[\+(\.0909),", cast=float)
    hi, _ = _read("research/B01_result.md", r"master − Baseline\+ \| \*\*\+\.0988\*\* \| \[\+\.0909, \+(\.1068)\]", cast=float)
    m.add_range("master_over_baseline_plus", round(d, 4), round(lo, 4), round(hi, 4),
                source="research/B01_result.md", command=_grep("research/B01_result.md", "master − Baseline+"),
                recomputed=False, note="frozen master − Baseline+, 0/10 games negative")
    for suffix, v in (("lo", lo), ("hi", hi)):
        m.add("master_over_baseline_plus_" + suffix, round(v, 4), source="research/B01_result.md",
              command=_grep("research/B01_result.md", "master − Baseline+"), recomputed=False,
              display=_delta(v))
    m.add("master_neg", "0/10", source="research/B01_result.md", recomputed=False,
          command=_grep("research/B01_result.md", "master − ship"), display="0/10",
          note="games in which master loses to ship")


def _runs(labels, want):
    """Number of maximal runs whose label is in ``want`` (a later team counts as a new run)."""
    n, prev = 0, None
    for label in labels:
        if label in want and label != prev:
            n += 1
        prev = label
    return n


def _submission():
    """Counts and sha straight from the frozen CSV, so no number can drift from the artifact."""
    path = os.path.join(SOLUTION_DIR, "outputs", "submission.csv")
    if not os.path.isfile(path):
        raise RuntimeError("missing %s; rebuild the submission first" % path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["frame", "label"]:
        raise RuntimeError("unexpected CSV header in %s: %r" % (path, rows[0]))
    labels = [r[1] for r in rows[1:]]
    if len(labels) != 199319:
        raise RuntimeError("unexpected row count in %s: %d" % (path, len(labels)))
    counts = collections.Counter(labels)
    d_run_lengths, run = [], 0
    for label in labels:
        if label == "D":
            run += 1
        elif run:
            d_run_lengths.append(run)
            run = 0
    if run:
        d_run_lengths.append(run)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return path, labels, counts, d_run_lengths, sha


def _match_numbers(m):
    """Match-level morphology computed from ``outputs/submission.csv`` itself."""
    import numpy as np
    from possession import data

    match = data.load_match()
    B = np.asarray(match.B)
    half = np.asarray(match.half)
    graded = half > 0

    def add(key, value, display, *, source, command, recomputed, note=None, tol=0.0):
        m.add(key, value, source=source, command=command, recomputed=recomputed,
              tolerance=tol, display=display, note=note)

    miss = ~np.isfinite(B[:, 0])
    add("missing_share_graded", round(float(miss[graded].mean()), 4),
        _pct(float(miss[graded].mean()), 1),
        source="solution/possession/data.py (ball channel)", recomputed=True,
        command="cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show missing_share_graded",
        note="share of graded frames with no tracked ball", tol=1e-4)

    path, labels, counts, d_runs, sha = _submission()
    n = {k: counts.get(k, 0) for k in "HADX"}
    n_graded = n["H"] + n["A"] + n["D"]
    n_all = n_graded + n["X"]
    spells = _runs(labels, "HA")
    cmd = "cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show "
    src = "solution/outputs/submission.csv (counted row by row)"
    add("all_frames", n_all, "{:,}".format(n_all), source=src, recomputed=True,
        command="wc -l < solution/outputs/submission.csv", note="rows including X")
    add("graded_frames", n_graded, "{:,}".format(n_graded), source=src, recomputed=True,
        command=cmd + "graded_frames", note="H + A + D")
    for key, label in (("count_h", "H"), ("count_a", "A"), ("count_d", "D"), ("count_x", "X")):
        add(key, n[label], "{:,}".format(n[label]), source=src, recomputed=True,
            command=cmd + key, note="rows labelled %s" % label)
    add("pred_d_share_graded", round(n["D"] / n_graded, 4), _score(n["D"] / n_graded),
        source=src, recomputed=True, command=cmd + "pred_d_share_graded",
        note="D frames over graded frames")
    add("pred_spells", spells, str(spells), source=src, recomputed=True,
        command=cmd + "pred_spells", note="maximal H/A runs; D splits them")
    add("d_stretches", len(d_runs), str(len(d_runs)), source=src, recomputed=True,
        command=cmd + "d_stretches", note="maximal D runs")
    add("d_runs_ge_15s", sum(1 for r in d_runs if r >= 38), str(sum(1 for r in d_runs if r >= 38)),
        source=src, recomputed=True, command=cmd + "d_runs_ge_15s",
        note="D runs of at least 38 frames (1.52 s)")
    add("ball_in_play_min", round((n["H"] + n["A"]) / FPS / 60.0, 2),
        _num((n["H"] + n["A"]) / FPS / 60.0, 2) + " min",
        source=src, recomputed=True, command=cmd + "ball_in_play_min",
        note="H + A frames at 25 fps")
    add("graded_min", round(n_graded / FPS / 60.0, 1), _num(n_graded / FPS / 60.0, 1) + " min",
        source=src, recomputed=True, command=cmd + "graded_min",
        note="graded frames at 25 fps")
    add("csv_sha8", sha[:8], sha[:8] + "…" + sha[-4:], source=src, recomputed=True,
        command="shasum -a 256 solution/outputs/submission.csv",
        note="head and tail of the sha256 of the frozen file")


def _practice_shape(m):
    """The shape clue's size, measured on one practice game (external tracking, no match data)."""
    import numpy as np
    from possession import dev_v1, external

    clean = external.load_game("metrica:game1")
    match = dev_v1.degraded_match(clean, 0, 1.0, 0.12)
    P = np.asarray(match.P)
    B = np.asarray(match.B)
    T = np.asarray(match.T)
    ref = np.asarray(clean.ref)

    widths = []
    for team in (0, 1):
        X = P[:, T == team, 1]
        ok = np.isfinite(X)
        mean = np.where(ok, X, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
        var = np.where(ok, (X - mean[:, None]) ** 2, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
        widths.append(np.sqrt(var))
    widths = np.array(widths).T

    present = np.isfinite(B[:, 0])
    D = np.hypot(P[:, :, 0] - B[:, 0, None], P[:, :, 1] - B[:, 1, None])
    ok = present & np.isfinite(D).any(1)
    near = np.full(len(D), -1)
    for i in np.where(ok)[0]:
        near[i] = T[np.nanargmin(D[i])]
    home = int(np.mean(near[ok & (ref == "H")] == 0) < 0.5)
    own = np.where(ref == "H", home, 1 - home)
    diff = widths[np.arange(len(widths)), own] - widths[np.arange(len(widths)), 1 - own]
    live = ok & np.isin(ref, ["H", "A"])
    dead_gaps = []
    run = 0
    for i in range(len(present) + 1):
        if i < len(present) and not present[i]:
            run += 1
            continue
        if run >= 50:
            vals = diff[i - run:i]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 10:
                dead_gaps.append(float(np.mean(vals)))
        run = 0
    diff_live = float(np.nanmean(diff[live]))
    own_width = float(np.nanmean(widths[np.arange(len(widths)), own][live]))
    def_width = float(np.nanmean(widths[np.arange(len(widths)), 1 - own][live]))
    cmd = "cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show "
    for key, value, display, note in (
            ("shape_width_own_m", own_width, _num(own_width, 1) + " m", "possession team, live frames"),
            ("shape_width_def_m", def_width, _num(def_width, 1) + " m", "defending team, live frames"),
            ("shape_width_diff_m", diff_live, _num(diff_live, 1) + " m", "own minus defending"),
            ("shape_dead_gap_m", float(np.mean(dead_gaps)),
             _num(float(np.mean(dead_gaps)), 1).replace("-", "−") + " m",
             "same difference inside dead ball-missing gaps")):
        m.add(key, round(value, 2), source="solution/possession/external.py, metrica:game1",
              command=cmd + key, recomputed=True, tolerance=0.02, display=display, note=note)


def _metric_demo(m):
    """The section-2 synthetic demonstration, scored by the organizer's own metric."""
    import numpy as np
    from possession import evaluate

    n = 500
    ref = np.full(n, "D", dtype="<U1")
    ref[:250] = "H"

    def pred(parts):
        p = np.full(n, "D", dtype="<U1")
        for a, b, lab in parts:
            p[a:b] = lab
        return p

    cases = {
        "perfect": [(0, 250, "H")],
        "split2": [(0, 125, "H"), (125, 150, "D"), (150, 250, "H")],
        "split3": [(0, 80, "H"), (80, 105, "D"), (105, 170, "H"), (170, 195, "D"), (195, 250, "H")],
        "split4": [(0, 60, "H"), (60, 80, "D"), (80, 120, "H"), (120, 140, "D"),
                   (140, 180, "H"), (180, 200, "D"), (200, 250, "H")],
        "missed": [],
    }
    cmd = "cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show demo_split2"
    for name, parts in cases.items():
        score = float(evaluate.score(pred(parts), ref)["spell_f1_mean"])
        key = "demo_%s" % name
        m.add(key, round(score, 4), source="organizer metric, synthetic labels",
              command="cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show " + key,
              recomputed=True, tolerance=1e-9, display=_score2(score),
              note="synthetic 10 s true H spell; this scenario only")
    for key, value in (("demo_iou_low", 0.3), ("demo_iou_high", 0.7)):
        m.add(key, value, source="the organizer's spellF1 band (SPELL_IOUS)",
              command="grep -n 'SPELL_IOUS' more-than-a-hack-2027/evaluation/possession_metrics.py",
              recomputed=False, display="%.1f" % value,
              note="the score averages F1 over these IoU thresholds")
    # IoU of each fragment against the 250-frame true spell.
    for n_frag, key in ((2, "demo_iou2"), (3, "demo_iou3"), (4, "demo_iou4")):
        length = int(round(250.0 / n_frag))
        iou = length / 250.0
        m.add(key, round(iou, 4), source="synthetic labels (one true 250-frame spell)",
              command="cd solution && .venv/bin/python scripts/walkthrough_numbers.py --show " + key,
              recomputed=True, tolerance=1e-9, display="%.2f" % iou,
              note="IoU of one equal fragment against the whole true spell")


def _constants(m):
    m.add("fps", FPS, source="solution/possession/evaluate.py: FPS", command="grep -n '^FPS' solution/possession/evaluate.py",
          recomputed=True, display="25 Hz")
    m.add("n_labels", 3, source="work/BIGPICTURE.md",
          command="grep -n 'three things' work/BIGPICTURE.md", recomputed=False, display="three")
    m.add("public_minutes", 5, source="work/GUARDRAILS.md §2", command="grep -n 'public 5 minutes' work/GUARDRAILS.md",
          recomputed=False, display="five")
    m.add("public_gt_spells", 17, source="work/DECISIONS.md D-013", command="grep -n '17 GT spells' work/DECISIONS.md",
          recomputed=False, display="17")
    m.add("practice_games", 10, source="work/DECISIONS.md D-022", command="grep -n '10 games' work/DECISIONS.md",
          recomputed=False, display="10")
    m.add("practice_seeds", 3, source="work/DECISIONS.md D-022", command="grep -n 'seeds' work/DECISIONS.md",
          recomputed=False, display="three")
    m.add("practice_idsse", 7, source="work/DECISIONS.md D-022", command="grep -n '7 IDSSE' work/DECISIONS.md",
          recomputed=False, display="seven")
    m.add("practice_metrica", 3, source="work/DECISIONS.md D-022", command="grep -n '3 Metrica' work/DECISIONS.md",
          recomputed=False, display="three")
    m.add("skillcorner_games", 20, source="work/DECISIONS.md D-022", command="grep -n '20 games' work/DECISIONS.md",
          recomputed=False, display="20")
    m.add("nearest_radius_m", 2.5, source="solution/possession/baseline_plus.py DEFAULT_PARAMS",
          command="grep -n 'DEFAULT_PARAMS' solution/possession/baseline_plus.py",
          recomputed=False, display="2.5 m")


def _digest(m):
    """Expensive results, read out of the research digest (never hand-typed into prose)."""
    def one(key, path, pattern, *, cast=float, display=None, note=None):
        value, raw = _read(path, pattern, cast=cast)
        m.add(key, value, source=path, command=_grep(path, pattern), recomputed=False,
              display=display if display is not None else raw, note=note)

    def rng(key, path, pattern_lo, pattern_hi, *, fmt=_delta, note=None, scale=1.0):
        lo, _ = _read(path, pattern_lo, cast=float)
        hi, _ = _read(path, pattern_hi, cast=float)
        lo, hi = lo * scale, hi * scale
        m.add(key, round((lo + hi) / 2.0, 4), source=path,
              command=_grep(path, pattern_lo), recomputed=False,
              display="%s to %s" % (fmt(lo), fmt(hi)), note=note)

    # -- section 1 / 3: the data we are given
    one("noise_band", "solution/docs/GLOSSARY.md", r"below about (0\.04) on the full match")
    m.e["noise_band"]["display"] = "0.04"
    one("practice_easier", "solution/docs/HOW_WE_WORK.md", r"about (\.06) easier")
    m.e["practice_easier"]["display"] = ".06"
    one("smooth50_match", "solution/docs/WHAT_WE_TRIED.md", r"against (\.243) on the real match")
    m.e["smooth50_match"]["display"] = ".243"
    one("visible_share_lo", "solution/docs/HOW_WE_WORK.md", r"between (7%) and 23% of stoppages")
    m.e["visible_share_lo"]["display"] = "7%"
    one("visible_share_hi", "solution/docs/HOW_WE_WORK.md", r"between 7% and (23%) of stoppages")
    m.e["visible_share_hi"]["display"] = "23%"
    one("practice_missing_share", "research/E76_regrid.md", r"missing share (\.373) at primary")
    m.e["practice_missing_share"]["display"] = "37.3%"
    one("practice_live_missing_old", "work/FINDINGS.md", r"P\(ball missing \| live\) dev (\.42)",
        display="42%")
    one("practice_live_missing", "work/FINDINGS.md",
        r"P\(NaN\|live\) \.43 → (\.044)")
    m.e["practice_live_missing"]["display"] = "4.4%"
    one("practice_long_gap_live", "work/FINDINGS.md",
        r"long-gap live share \.35-\.39 → (\.021)")
    m.e["practice_long_gap_live"]["display"] = "2.1%"
    one("ours_long_gap_live", "work/FINDINGS.md", r"live share dev 35-39% vs (≤ ~4)%",
        cast=lambda s: 0.04, display="≤ ~4%")
    one("ours_live_missing", "work/FINDINGS.md", r"P\(NaN\|live\) \.43 → \.044 \(~(\.03)\)",
        cast=lambda s: 0.03, display="~3%")
    one("baseline_plus_old", "work/FINDINGS.md", r"\*\*Baseline\+ (\.216) → \.326", display=".216")
    one("baseline_plus_new", "work/FINDINGS.md", r"Baseline\+ \.216 → (\.326)", display=".326")
    one("ingap_old", "work/FINDINGS.md",
        r"in-gap decoder \+\.082 → (\+\.003\.\.\+\.016)", cast=lambda s: 0.082, display="+.082")
    one("ingap_new", "work/FINDINGS.md", r"in-gap decoder \+\.0816 → (\+\.006)", display="+.006")

    # -- section 1: why nearest is not owner
    one("flip_recall_clean", "work/FINDINGS.md", r"nearest-player cue catches \*\*(\.764)\*\*")
    m.e["flip_recall_clean"]["display"] = ".764"
    one("flip_recall_degraded", "work/FINDINGS.md", r"vs \*\*(\.482)\*\* on the degraded copy")
    m.e["flip_recall_degraded"]["display"] = ".482"
    one("unresolvable_share", "work/FINDINGS.md", r"~(13\.5%) of all flips unresolvable")
    m.e["unresolvable_share"]["display"] = "13.5%"

    # -- section 3: the two practice-data flaws
    one("flaw_texture_auc", "work/FINDINGS.md", r"texture-only LOGO 1\.000 → (\.92)")
    m.e["flaw_texture_auc"]["display"] = ".92"
    one("flaw_texture_auc_heldout", "work/FINDINGS.md", r"give texture (\.907)")
    m.e["flaw_texture_auc_heldout"]["display"] = ".907"
    one("flaw_target_auc", "work/FINDINGS.md", r"target (\.80) not met")
    m.e["flaw_target_auc"]["display"] = ".80"

    # -- section 4: shipped add-ons
    one("grown_delta", "research/E76_regrid.md", r"E65 grown D, R25 cap30 \| (\+\.0055)")
    m.e["grown_delta"]["display"] = "+.0055"
    m.e["grown_delta"]["value"] = 0.0055
    one("grown_ci_low", "research/E76_regrid.md", r"R25 cap30 \| \+\.0055 \[\+\.0159\] \| \[(\.0044),")
    m.e["grown_ci_low"]["display"] = ".0044"
    one("grown_ci_high", "research/E76_regrid.md", r"R25 cap30 \| \+\.0055 \[\+\.0159\] \| \[\.0044,(\.0066)\]")
    m.e["grown_ci_high"]["display"] = ".0066"
    one("subgap_delta", "research/E76_regrid.md", r"E69 sub-gap gate overlay \| (\+\.0044)")
    m.e["subgap_delta"]["display"] = "+.0044"
    one("subgap_ci_low", "research/E76_regrid.md", r"E69 sub-gap gate overlay \| \+\.0044 \[\+\.0032\] \| \[(\.0033),")
    m.e["subgap_ci_low"]["display"] = ".0033"
    one("subgap_ci_high", "research/E76_regrid.md", r"E69 sub-gap gate overlay \| \+\.0044 \[\+\.0032\] \| \[\.0033,(\.0056)\]")
    m.e["subgap_ci_high"]["display"] = ".0056"
    one("postprocess_delta", "research/E14_result.md", r"(\+\.0099) spellF1 \[\+\.0063, \+\.0138\]", cast=lambda s: 0.0099,
        display="+.0099")
    one("postprocess_delta_current", "work/FINDINGS.md",
        r"absolute value on the current stack is \*\*(\+\.0113)\*\*")
    m.e["postprocess_delta_current"]["display"] = "+.0113"
    one("postprocess_ci_low", "research/E14_result.md", r"\[\+(\.0063), \+\.0138\]")
    m.e["postprocess_ci_low"]["display"] = ".0063"
    one("postprocess_ci_high", "research/E14_result.md", r"\[\+\.0063, \+(\.0138)\]")
    m.e["postprocess_ci_high"]["display"] = ".0138"
    rng("expected_gate", "research/E76_regrid.md", r"D gate: (\+\.06)\.\.\+\.08",
        r"D gate: \+\.06\.\.(\+\.08)", fmt=_trim, note="expected on our match")
    rng("expected_ingap", "research/E76_regrid.md", r"Ingap: ~ (\+\.002)\.\.\+\.005",
        r"Ingap: ~ \+\.002\.\.(\+\.005)", fmt=_trim, note="expected on our match")
    rng("expected_grown", "solution/docs/WHAT_WE_TRIED.md", r"We expect (\+\.001) to \+\.005 on our match",
        r"We expect \+\.001 to (\+\.005) on our match", fmt=_trim, note="expected on our match")
    one("expected_subgap", "solution/docs/WHAT_WE_TRIED.md", r"We expect about (\+\.002) on our match")
    m.e["expected_subgap"]["display"] = "+.002"
    rng("expected_postprocess", "work/FINDINGS.md", r"expected (\+\.006)\.\.\+\.009",
        r"expected \+\.006\.\.(\+\.009)", fmt=_trim,
        note="short-spell floor, not wired")

    # -- section 5
    one("e25_dead_share", "work/FINDINGS.md", r"flags \*\*(85\.6%)", cast=lambda s: 0.856)
    m.e["e25_dead_share"]["display"] = "85.6%"
    one("e25_ci_low", "work/FINDINGS.md", r"85\.6% \[(80\.8), 89\.2\]", cast=lambda s: float(s) / 100.0)
    m.e["e25_ci_low"]["display"] = "80.8%"
    one("e25_ci_high", "work/FINDINGS.md", r"85\.6% \[80\.8, (89\.2)\]", cast=lambda s: float(s) / 100.0)
    m.e["e25_ci_high"]["display"] = "89.2%"
    one("e36_dead_share", "work/FINDINGS.md", r"independent route puts \*\*(84\.5%)")
    m.e["e36_dead_share"]["display"] = "84.5%"
    one("gate_pd_missing_low", "work/FINDINGS.md", r"corrected P\(D\|missing\) \*\*(0\.93)", cast=lambda s: 0.93)
    m.e["gate_pd_missing_low"]["display"] = ".93"
    one("gate_pd_missing_high", "work/FINDINGS.md", r"corrected P\(D\|missing\) \*\*0\.93-(0\.99)", cast=lambda s: 0.99)
    m.e["gate_pd_missing_high"]["display"] = ".99"
    one("shape_correct", "solution/docs/WHAT_WE_TRIED.md", r"Correct (87%) of the time")
    m.e["shape_correct"]["display"] = "87%"
    one("shape_auc_idsse", "research/E04_result.md", r"home-minus-away team width is (0\.88)")
    m.e["shape_auc_idsse"]["display"] = ".88"
    one("shape_auc_metrica", "research/E04_result.md", r"0\.88 \(IDSSE\) and (0\.90)")
    m.e["shape_auc_metrica"]["display"] = ".90"
    one("eligible_ours", "work/FINDINGS.md", r"eligible pool (3\.9%)")
    m.e["eligible_ours"]["display"] = "3.9%"
    rng("eligible_dev", "work/FINDINGS.md", r"vs (17)-44%",
        r"vs 17-(44)%", fmt=lambda x: "%.0f%%" % (100.0 * x), scale=0.01,
        note="dev eligible share")

    # -- section 6: what we dropped
    one("kick_recall_low", "solution/docs/WHAT_WE_TRIED.md", r"Recall (75) to 83% on clean data", cast=lambda s: 0.75,
        display="75%")
    one("kick_recall_high", "solution/docs/WHAT_WE_TRIED.md", r"Recall 75 to (83%) on clean data")
    m.e["kick_recall_high"]["display"] = "83%"
    one("coupling_low", "solution/docs/WHAT_WE_TRIED.md", r"cost −(0\.002) to −0\.011", cast=lambda s: -0.002,
        display="−.002")
    one("coupling_high", "solution/docs/WHAT_WE_TRIED.md", r"cost −0\.002 to −(0\.011)", cast=lambda s: -0.011,
        display="−.011")
    one("video_recall", "solution/docs/WHAT_WE_TRIED.md", r"found it on (36%) of frames")
    m.e["video_recall"]["display"] = "36%"
    one("video_false", "solution/docs/WHAT_WE_TRIED.md", r"fired on (82%) of frames")
    m.e["video_false"]["display"] = "82%"
    rng("ball_px", "solution/docs/WHAT_WE_TRIED.md", r"ball is (8) to 10 pixels wide",
        r"ball is 8 to (10) pixels wide", fmt=lambda x: "%.0f px" % x)
    one("hsmm_delta", "work/FINDINGS.md", r"\*\*HSMM (\.3949), −\.0090", cast=lambda s: -0.0090,
        display="−.0090")
    one("hsmm_ci_low", "work/FINDINGS.md", r"−\.0090 \[(−\.0163),−\.0024\]")
    m.e["hsmm_ci_low"]["display"] = "−.0163"
    one("hsmm_ci_high", "work/FINDINGS.md", r"−\.0090 \[−\.0163,(−\.0024)\]")
    m.e["hsmm_ci_high"]["display"] = "−.0024"
    one("hmm_delta", "work/FINDINGS.md", r"HMM (\.4123), \+\.0084", cast=lambda s: 0.0084,
        display=".4123")
    m.e["hmm_delta"]["display"] = "+.0084"
    one("hmm_ci_low", "work/FINDINGS.md", r"\+\.0084 \[(−\.0025),\+\.0180\]")
    m.e["hmm_ci_low"]["display"] = "−.0025"
    one("hmm_ci_high", "work/FINDINGS.md", r"\+\.0084 \[−\.0025,(\+\.0180)\]")
    m.e["hmm_ci_high"]["display"] = "+.0180"
    one("restart_team_pp", "solution/docs/WHAT_WE_TRIED.md", r"changed it by (−0\.4) points",
        cast=lambda s: -0.4, display="−0.4 points")
    one("split_delta", "research/E76_regrid.md", r"E72 split D blocks \| (-?\.0011)")
    m.e["split_delta"]["display"] = "−.0011"
    one("panns_synthetic", "solution/docs/WHAT_WE_TRIED.md", r"synthetic whistle tone at (\.115)")
    m.e["panns_synthetic"]["display"] = ".115"
    one("panns_real", "solution/docs/WHAT_WE_TRIED.md", r"real whistles at about (\.001)")
    m.e["panns_real"]["display"] = ".001"
    one("crowd_lead", "work/FINDINGS.md", r"crowd parks\*\* \(lead (\+\.027)")
    m.e["crowd_lead"]["display"] = "+.027"

    # -- section 7: the frozen results and the public window (report only)
    one("public_ship_spellf1", "research/B01_result.md", r"spellF1 (\.4686), macroF1 \.7199")
    m.e["public_ship_spellf1"]["display"] = ".4686"
    one("public_ship_macro", "research/B01_result.md", r"spellF1 \.4686, macroF1 (\.7199)")
    m.e["public_ship_macro"]["display"] = ".7199"
    one("public_ship_spells", "research/B01_result.md", r"macroF1 \.7199, (18) predicted vs 17 GT", cast=int)
    m.e["public_ship_spells"]["display"] = "18"
    for name, spell_pat, macro_pat, count_pat, n in (
            ("smooth125", r"ours \.4686,\s*smooth125 (\.4242),", r"\| smooth125 \| \.4242 \| (\.5812) \| 16/17",
             r"\| smooth125 \| \.4242 \| \.5812 \| (16)/17", 16),
            ("smooth50", r"smooth125 \.4242,\s*smooth50 (\.3256),",
             r"smooth50 \*\(published floor\)\* \| \.3256 \| (\.6207) \| 26/17",
             r"smooth50 \*\(published floor\)\* \| \.3256 \| \.6207 \| (26)/17", 26),
            ("smooth25", r"smooth50 \.3256,\s*smooth25 (\.2868),",
             r"\| smooth25 \| \.2868 \| (\.6200) \| 36/17",
             r"\| smooth25 \| \.2868 \| \.6200 \| (36)/17", 36),
            ("raw", r"smooth25 \.2868,\s*raw (\.0378);", r"\| raw \| \.0378 \| (\.5972) \| 131/17",
             r"\| raw \| \.0378 \| \.5972 \| (131)/17", 131)):
        one("public_%s_spellf1" % name, "research/Z01_final_report.md", spell_pat)
        one("public_%s_macro" % name, "work/DECISIONS.md", macro_pat)
        one("public_%s_spells" % name, "work/DECISIONS.md", count_pat, cast=int)
        m.e["public_%s_spellf1" % name]["display"] = _score(m.e["public_%s_spellf1" % name]["value"])
        m.e["public_%s_macro" % name]["display"] = _score(m.e["public_%s_macro" % name]["value"])
        m.e["public_%s_spells" % name]["display"] = str(n)
    # the on-match size of the one gain that is new in the frozen build
    rng("expected_master", "research/Z01_final_report.md",
        r"master − ship ≈ (\+\.011)…\+\.016", r"master − ship ≈ \+\.011…(\+\.016)", fmt=_trim,
        note="reach-scaled, not the dev number")

    # -- section 5: the second instrument's interval (E36)
    one("e36_live_low", "research/E36_gap_mechanisms.md",
        r"verified-live share of missing time \| 4\.09% \| \[(2\.05)%, 6\.92%\]",
        cast=lambda s: float(s) / 100.0)
    m.e["e36_live_low"]["display"] = "2.1%"
    one("e36_live_high", "research/E36_gap_mechanisms.md",
        r"verified-live share of missing time \| 4\.09% \| \[2\.05%, (6\.92)%\]",
        cast=lambda s: float(s) / 100.0)
    m.e["e36_live_high"]["display"] = "6.9%"
    one("e36_unresolved", "research/E36_gap_mechanisms.md",
        r"unresolved \| 42 \| 7,332 \| (11\.4)%", cast=lambda s: float(s) / 100.0)
    m.e["e36_unresolved"]["display"] = "11.4%"
    dead = m.e["e36_dead_share"]["value"]
    live_lo = m.e["e36_live_low"]["value"]
    live_hi = m.e["e36_live_high"]["value"]
    lo = dead - (live_hi - live_lo) / 2.0
    hi = dead + (live_hi - live_lo) / 2.0
    m.add("e36_dead_low", round(lo, 4),
          source="research/E36_gap_mechanisms.md (verified-live CI carried onto the dead share)",
          command=_grep("research/E36_gap_mechanisms.md", "verified-live share"),
          recomputed=False, display=_pct(lo), note="unresolved share held fixed at 11.4%")
    m.add("e36_dead_high", round(hi, 4),
          source="research/E36_gap_mechanisms.md (verified-live CI carried onto the dead share)",
          command=_grep("research/E36_gap_mechanisms.md", "verified-live share"),
          recomputed=False, display=_pct(hi), note="unresolved share held fixed at 11.4%")

    # -- section 6: the dropped arms in one chart
    one("mbr_delta", "research/E26_mbr.md", r"\| b MBR \|.*?\*\*(-\.0074)")
    m.e["mbr_delta"]["display"] = "−.0074"
    one("viterbi_delta", "research/E26_mbr.md", r"\| a Viterbi \|.*?\*\*(-\.0114)")
    m.e["viterbi_delta"]["display"] = "−.0114"

    # -- section 8: dev blind spots and the two evidence limits
    one("missed_share", "research/E46_missed_spells.md",
        r"so \*\*(78\.9%) of GT spells are missed", cast=lambda s: float(s) / 100.0)
    m.e["missed_share"]["display"] = "79%"
    one("handover_tail", "research/E83_handover_scatter.md",
        r"\((46\.9)% of matched\)", cast=lambda s: float(s) / 100.0)
    m.e["handover_tail"]["display"] = "46.9%"
    one("handover_tail_low", "research/E83_handover_scatter.md",
        r"matched, .*?\| \*\*12,554 \(46\.9% of matched\)\*\* \| \[(43\.9), 49\.5\]",
        cast=lambda s: float(s) / 100.0)
    m.e["handover_tail_low"]["display"] = "43.9%"
    one("handover_tail_high", "research/E83_handover_scatter.md",
        r"matched, .*?\| \*\*12,554 \(46\.9% of matched\)\*\* \| \[43\.9, (49\.5)\]",
        cast=lambda s: float(s) / 100.0)
    m.e["handover_tail_high"]["display"] = "49.5%"
    one("handover_early", "research/E83_handover_scatter.md",
        r"\| early > 1 s .*?\| (31\.2)% of matched", cast=lambda s: float(s) / 100.0)
    m.e["handover_early"]["display"] = "31.2%"
    one("handover_late", "research/E83_handover_scatter.md",
        r"\| late > 1 s .*?\| (15\.7)% of matched", cast=lambda s: float(s) / 100.0)
    m.e["handover_late"]["display"] = "15.7%"
    one("handover_cue_share", "research/E83_handover_scatter.md",
        r"a1\+a2 are (93\.4)% and \+\.0796", cast=lambda s: float(s) / 100.0)
    m.e["handover_cue_share"]["display"] = "93.4%"
    one("handover_oracle", "research/E83_handover_scatter.md", r"a1\+a2 are 93\.4% and (\+\.0796)\.")
    m.e["handover_oracle"]["display"] = "+.0796"
    one("stage_moved", "research/E83_handover_scatter.md",
        r"only 87 \((0\.69)%\) are moved ≥0\.5 s", cast=lambda s: float(s) / 100.0)
    m.e["stage_moved"]["display"] = "0.69%"
    one("contact_s", "research/E83_handover_scatter.md", r"~(0\.16) s")
    m.e["contact_s"]["display"] = "0.16 s"
    one("metrica_d_spread", "work/DECISIONS.md", r"with (20)x spread in its own present-D share", cast=int)
    m.e["metrica_d_spread"]["display"] = "20×"
    one("sliver_gaps", "work/DECISIONS.md", r"~(\d+) gaps of 2-4 s per game on dev", cast=int)
    m.e["sliver_gaps"]["display"] = "10"
    one("sliver_share", "work/DECISIONS.md", r"~\.(\d+)% of graded; inflates ingap",
        cast=lambda s: float(s) / 100.0)
    m.e["sliver_share"]["display"] = "0.5%"
    one("anchor_recall_knob0", "work/DECISIONS.md", r"\*\*(\.19) \(knob 0\) / \.085 \(knob 1\)\*\*")
    m.e["anchor_recall_knob0"]["display"] = ".19"
    one("anchor_recall_knob1", "work/DECISIONS.md", r"\.19 \(knob 0\) / (\.085) \(knob 1\)")
    m.e["anchor_recall_knob1"]["display"] = ".085"
    one("anchor_recall_clean", "work/DECISIONS.md", r"on dev-v1 vs \*\*(\.45) clean\*\*")
    m.e["anchor_recall_clean"]["display"] = ".45"
    one("gap_edge_sd_s", "work/DECISIONS.md", r"at quarter strength \(sd ~(1) s\)")
    m.e["gap_edge_sd_s"]["display"] = "1 s"
    one("gap_edge_spread_min", "work/DECISIONS.md", r"is (2)-4x shipped", cast=int)
    m.e["gap_edge_spread_min"]["display"] = "2×"
    one("gap_edge_spread_max", "work/DECISIONS.md", r"is 2-(4)x shipped", cast=int)
    m.e["gap_edge_spread_max"]["display"] = "4×"
    one("geom_nearest_dev", "research/E78_geometry_harshness.md",
        r"median nearest (2\.29) vs 2\.35")
    m.e["geom_nearest_dev"]["display"] = "2.29 m"
    one("geom_nearest_match", "research/E78_geometry_harshness.md",
        r"median nearest 2\.29 vs (2\.35)")
    m.e["geom_nearest_match"]["display"] = "2.35 m"

    # -- reading copy: what changed since the freeze (P05). Every figure is parsed from the digest;
    # the exact command is a one-line grep, and --check fails if the digest or the value moves.
    one("ingap_marginal", "research/E112_ingap_marginal.md",
        r"in-gap marginal \(headline, 6 primary\)\*\* \| \*\*(\+\.0017)",
        display="+.0017", note="current-stack leave-one-out marginal, three seeds")
    one("ingap_marginal_lo", "research/E112_ingap_marginal.md",
        r"in-gap marginal \(headline, 6 primary\)\*\* \| \*\*\+\.0017 \[\+(\.0004),",
        display="+.0004", note="lower end of the game-cluster interval")
    one("ingap_marginal_hi", "research/E112_ingap_marginal.md",
        r"\+\.0004, \+(\.0030)\]\*\*, 3/10 games negative",
        display="+.0030", note="upper end of the game-cluster interval")
    one("ingap_seed_lo", "research/E112_ingap_marginal.md",
        r"max−min: (−\.0009) to \+\.0031", display="−.0009",
        note="lowest of the three frozen degrade draws")
    one("ingap_seed_hi", "research/E112_ingap_marginal.md",
        r"max−min: −\.0009 to (\+\.0031)", display="+.0031",
        note="highest of the three frozen degrade draws")
    one("sign_threshold", "research/E118_seed_cluster.md",
        r"Threshold: X = (\.004)\.", display=".004",
        note="a dev delta below this is quoted as level with zero")
    one("ingap_two_level_lo", "research/E118_seed_cluster.md",
        r"3-draw \[(−\.0011),\+\.0041\]", display="−.0011",
        note="two-level (game x draw) interval, frozen three seeds")
    one("ingap_two_level_hi", "research/E118_seed_cluster.md",
        r"3-draw \[−\.0011,(\+\.0041)\]", display="+.0041",
        note="two-level (game x draw) interval, frozen three seeds")

    one("headline_six_draw", "research/E103_seed_stability.md",
        r"6-seed pooled\*\* \| \*\*\.4240\*\* \| \*\*\.4037\*\* \| \*\*(\+\.0203)\*\*",
        display="+.0203", note="master - ship over six degradation draws")
    one("headline_six_draw_lo", "research/E103_seed_stability.md",
        r"6-seed pooled\*\* \| \*\*\.4240\*\* \| \*\*\.4037\*\* \| \*\*\+\.0203\*\* \| \*\*\[\+(\.0166),",
        display="+.0166", note="lower end of the six-draw interval")
    one("headline_six_draw_hi", "research/E103_seed_stability.md",
        r"6-seed pooled\*\* \| \*\*\.4240\*\* \| \*\*\.4037\*\* \| \*\*\+\.0203\*\* \| \*\*\[\+\.0166, \+(\.0241)\]",
        display="+.0241", note="upper end of the six-draw interval")
    one("level_clean", "research/E82_level_robustness.md",
        r"At (0\.8)x", display="0.8×", note="cleaner severity perturbation")
    one("level_messy", "research/E82_level_robustness.md",
        r"at (1\.2)x", display="1.2×", note="messier severity perturbation")
    one("level_margin_08", "research/E82_level_robustness.md",
        r"\| \*\*master − ship\*\* \| \*\*(\+\.0195) \[\+(\.0145),\+(\.0254)\]\*\*",
        display="+.0195", note="smallest adjacent margin, cleaner level, seed 0")

    one("skillcorner_master_lo", "research/E104_dev1b_confirmation.md",
        r"master − Baseline\+ \+(\.0361)…\+\.0567", display="+.0361",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_master_hi", "research/E104_dev1b_confirmation.md",
        r"master − Baseline\+ \+\.0361…(\+\.0567)", display="+.0567",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_ship_lo", "research/E104_dev1b_confirmation.md",
        r"ship − Baseline\+\s+\+(\.0215)…", display="+.0215",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_ship_hi", "research/E104_dev1b_confirmation.md",
        r"ship − Baseline\+\s+\+\.0215…(\+\.0407)", display="+.0407",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_master_ship_lo", "research/E104_dev1b_confirmation.md",
        r"master − ship \+(\.0092)…\+\.0166", display="+.0092",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_master_ship_hi", "research/E104_dev1b_confirmation.md",
        r"master − ship \+\.0092…(\+\.0166)", display="+.0166",
        note="cross-rule range on 20 held-out SkillCorner games")
    one("skillcorner_bar", "research/E104_dev1b_confirmation.md",
        r"D-023's ≈ (\.02) bar", display="0.02",
        note="minimum pooled delta for a held-out claim (D-023)")

    one("playing_clock_min", "research/E105_inplay_rate.md",
        r"live time is \*\*57\.47 min of a (103\.95)-min playing clock",
        display="103.95 min", note="graded frames minus the published 2H block, at 25 fps")
    one("inplay_rate", "research/E105_inplay_rate.md",
        r"playing clock =\s+(55\.28)%",
        cast=lambda s: round(float(s) / 100.0, 4), display="55.28%",
        note="live share of the playing clock")
    one("external_inplay_rate", "research/E105_inplay_rate.md",
        r"games average \*\*55\.56 ± 6\.37 live min and (58\.21) ± (6\.38)% share\*\*",
        cast=lambda s: round(float(s) / 100.0, 4), display="58.21%",
        note="mean live share of the ten external games")
    one("external_inplay_sd", "research/E105_inplay_rate.md",
        r"games average \*\*55\.56 ± 6\.37 live min and 58\.21 ± (6\.38)% share\*\*",
        cast=lambda s: round(float(s) / 100.0, 4), display="6.38%",
        note="between-game sd of the external live share")
    one("inplay_z", "research/E105_inplay_rate.md",
        r"live share of the playing clock\*\* \| \*\*55\.28%\*\* \| \*\*58\.21 ± 6\.38%\*\* \| \*\*(−0\.46)\*\*",
        display="−0.46", note="z on the rate axis; the raw-minute axis is confounded")

    one("attrib_floor", "research/E98_attribution.md",
        r"\| \*\*floor\*\* \| \.4144 \| \*\*(\+\.0114)\*\*", display="+.0114",
        note="leave-one-out marginal at master, seed 0")
    one("attrib_subgap", "research/E98_attribution.md",
        r"\| \*\*sub-gap\*\* \| \.4195 \| \*\*(\+\.0062)\*\*", display="+.0062",
        note="leave-one-out marginal at master, seed 0")
    one("attrib_grown", "research/E98_attribution.md",
        r"\| \*\*grown D\*\* \| \.4213 \| \*\*(\+\.0045)\*\*", display="+.0045",
        note="leave-one-out marginal at master, seed 0")
    one("attrib_ingap_seed0", "research/E98_attribution.md",
        r"\| \*\*in-gap\*\* \| \.4267 \| \*\*(−\.0009)\*\*", display="−.0009",
        note="leave-one-out marginal at master, seed 0 (superseded by E112)")
    one("attrib_sum", "research/E98_attribution.md",
        r"grown D \+ sub-gap \+ floor\*\* \| \*\*(\+\.0221)\*\*", display="+.0221",
        note="sum of the three post-ship marginals")
    one("attrib_interaction", "research/E98_attribution.md",
        r"three post-`ship` stages\)\*\* \| \*\*(−\.0013)\*\*", display="−.0013",
        note="sum of three marginals minus the measured combined run")


def build():
    m = Manifest()
    _constants(m)
    _match_numbers(m)
    _practice_shape(m)
    _metric_demo(m)
    _dev_ref(m)
    _digest(m)
    return {
        "schema": 1,
        "shipped_stack": SHIPPED_STACK,
        "generated_by": "solution/scripts/walkthrough_numbers.py",
        "entries": {k: m.e[k] for k in sorted(m.e)},
    }


# ----------------------------------------------------------------------------- cli

def _flatten(entries):
    return {k: v["value"] for k, v in entries.items()}


def check(path, new):
    if not os.path.isfile(path):
        raise SystemExit("--check: %s does not exist; run without --check first" % path)
    old = json.load(open(path))["entries"]
    missing = sorted(set(new["entries"]) - set(old))
    extra = sorted(set(old) - set(new["entries"]))
    if missing or extra:
        raise SystemExit("--check: keys changed (missing %s, extra %s)" % (missing, extra))
    bad = []
    for key, entry in new["entries"].items():
        # Every entry is compared, not only the recomputed ones. A "read" entry is parsed out of a
        # research digest; if that digest is corrected, the stored number is stale, and the notebook
        # that substitutes from numbers.json would keep quoting the old figure. Checking only
        # recomputed entries let that happen with `--check` still green (Q34).
        a, b = old[key]["value"], entry["value"]
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                bad.append("%s: stored %r, now %r" % (key, a, b))
            continue
        a, b = float(a), float(b)
        if abs(a - b) > max(float(entry["tolerance"]), 1e-12):
            where = "" if entry["recomputed"] else " (read: %s)" % entry.get("source", "?")
            bad.append("%s%s: stored %.6g, now %.6g (tol %g)"
                       % (key, where, a, b, entry["tolerance"]))
    if bad:
        raise SystemExit("--check: stored numbers drifted:\n  " + "\n  ".join(bad))
    print("numbers.json OK: %d keys (%d recomputed, %d read), no drift" % (
        len(new["entries"]), sum(1 for e in new["entries"].values() if e["recomputed"]),
        sum(1 for e in new["entries"].values() if not e["recomputed"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="fail if a recomputed value drifted")
    ap.add_argument("--show", metavar="KEY", help="print one entry as JSON and exit")
    args = ap.parse_args(argv)

    new = build()
    if args.show:
        entry = new["entries"].get(args.show)
        if entry is None:
            raise SystemExit("unknown key %r (%d keys)" % (args.show, len(new["entries"])))
        print(json.dumps({args.show: entry}, indent=2))
        return 0
    if args.check:
        check(args.out, new)
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote %s (%d keys)" % (args.out, len(new["entries"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
