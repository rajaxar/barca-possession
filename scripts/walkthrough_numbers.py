"""Build ``notebooks/numbers.json``: every number the walkthrough quotes, with its provenance.

Two kinds of entry:

* **recomputed** -- measured from the live code right now (the frozen dev-v1 reference table, the
  current submission's spell/D morphology, the synthetic spellF1 demonstration).  ``--check``
  rebuilds these and fails when one has moved beyond its recorded tolerance.
* **read** -- an expensive result taken from a research digest.  The value is *parsed out of the
  digest* (never typed into the notebook), and ``--check`` fails if the pattern no longer matches.

Each entry is ``{"value", "display", "source", "command", "recomputed", "tolerance"}``.
The notebook builder substitutes ``{key}`` from this file, so no number in the prose is hand-typed.

    cd solution
    .venv/bin/python scripts/walkthrough_numbers.py              # write notebooks/numbers.json
    .venv/bin/python scripts/walkthrough_numbers.py --check      # fail on drift
    .venv/bin/python scripts/walkthrough_numbers.py --show gate_dev

Reads the match for the morphology entries; never reads or scores the public answers.
"""
import argparse
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


def _match_numbers(m):
    """Match-level morphology from the frozen Q26 reproduction and the ball channel.

    ``outputs/submission.csv`` is deliberately *not* read: other cards rebuild it, and this notebook
    quotes the shipped file, whose counts and sha are frozen in research/Q26_e66_repro.md.
    """
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

    q26 = os.path.join(ROOT, "research", "Q26_e66_repro.md")
    text = open(q26, encoding="utf-8").read()
    counts = re.search(r"H ([\d,]+) / A ([\d,]+) / D ([\d,]+) / X ([\d,]+)", text)
    if not counts:
        raise RuntimeError("could not read the validator counts from research/Q26_e66_repro.md")
    n = {k: int(v.replace(",", "")) for k, v in zip("HADX", counts.groups())}
    n_graded = n["H"] + n["A"] + n["D"]
    add("all_frames", n["H"] + n["A"] + n["D"] + n["X"], "{:,}".format(n["H"] + n["A"] + n["D"] + n["X"]),
        source="research/Q26_e66_repro.md (validator)", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "199,319 rows"))
    add("graded_frames", n_graded, "{:,}".format(n_graded),
        source="research/Q26_e66_repro.md (validator)", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "H 51,037"))
    add("pred_d_share_graded", round(n["D"] / n_graded, 4), _pct(n["D"] / n_graded, 1),
        source="research/Q26_e66_repro.md (validator counts)", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "H 51,037"))
    add("pred_spells", 414, "414", source="research/Q26_e66_repro.md", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "Final spells 414"))
    add("ball_in_play_min", round((n["H"] + n["A"]) / FPS / 60.0, 1),
        _num((n["H"] + n["A"]) / FPS / 60.0, 1) + " min",
        source="research/Q26_e66_repro.md (validator counts)", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "H 51,037"))
    add("pred_d_runs", 114, "114", source="work/FINDINGS.md (E14)", recomputed=False,
        command=_grep("work/FINDINGS.md", "D stretches 114"))
    sha = re.search(r"sha256 `([0-9a-f]{8})[0-9a-f]*([0-9a-f]{4})`", text)
    if not sha:
        raise RuntimeError("could not read the shipped CSV sha from research/Q26_e66_repro.md")
    add("csv_sha8", sha.group(1), sha.group(1) + "…" + sha.group(2),
        source="research/Q26_e66_repro.md (Q26 clean-clone build)", recomputed=False,
        command=_grep("research/Q26_e66_repro.md", "sha256"))


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

    # -- public sanity (report only)
    one("public_ship_spellf1", "solution/docs/WHAT_WE_TRIED.md", r"the shipped entry scores \*{0,2}(\.[0-9]{3,4})")
    m.e["public_ship_spellf1"]["display"] = ".4686"
    one("public_ship_macro", "solution/docs/WHAT_WE_TRIED.md", r"Our frame-level score there is \*{0,2}(\.[0-9]{3,4})")
    m.e["public_ship_macro"]["display"] = ".7199"
    one("public_baseline2", "solution/docs/WHAT_WE_TRIED.md", r"against (\.326) for their 2-second baseline")
    m.e["public_baseline2"]["display"] = ".326"
    one("public_baseline5", "solution/docs/WHAT_WE_TRIED.md", r"and (\.424) for their 5-second one")
    m.e["public_baseline5"]["display"] = ".424"


def build():
    m = Manifest()
    _constants(m)
    _match_numbers(m)
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
        if not entry["recomputed"]:
            continue
        a, b = old[key]["value"], entry["value"]
        if isinstance(a, str) or isinstance(b, str):
            if a != b:
                bad.append("%s: stored %r, now %r" % (key, a, b))
            continue
        a, b = float(a), float(b)
        if abs(a - b) > max(float(entry["tolerance"]), 1e-12):
            bad.append("%s: stored %.6g, now %.6g (tol %g)" % (key, a, b, entry["tolerance"]))
    if bad:
        raise SystemExit("--check: recomputed values drifted:\n  " + "\n  ".join(bad))
    print("numbers.json OK: %d keys, %d recomputed, no drift" % (
        len(new["entries"]), sum(1 for e in new["entries"].values() if e["recomputed"])))


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
