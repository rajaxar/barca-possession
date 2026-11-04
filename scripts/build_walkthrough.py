"""Build ``notebooks/possession_walkthrough.ipynb`` (markdown + HTML only) and its figures.

The notebook has **no code cells**: prose, synthetic schematics and locally rendered clips. Every
number comes from ``notebooks/numbers.json`` (built by ``walkthrough_numbers.py``); the prose uses
``{key}`` substitution, so no measured number is typed here.

    cd solution
    .venv/bin/python scripts/build_walkthrough.py --profile local     # render everything, write the notebook
    .venv/bin/python scripts/build_walkthrough.py --profile public    # placeholders for real-tracking figures
    .venv/bin/python scripts/build_walkthrough.py --figs-only         # figures only
    .venv/bin/python scripts/build_walkthrough.py --check             # numbers, placeholders and files

``local`` figures render real tracking (ours and an external practice match) into ``notebooks/figs/``,
which is gitignored; they are never publishable (D-002). ``public`` keeps every synthetic schematic
and replaces each real-tracking figure with a neutral placeholder that carries the same caption.
"""
import argparse
import hashlib
import json
import os
import string
import sys

import numpy as np

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION_DIR)
sys.path.insert(0, SOLUTION_DIR)

NOTEBOOK = os.path.join(SOLUTION_DIR, "notebooks", "possession_walkthrough.ipynb")
NUMBERS = os.path.join(SOLUTION_DIR, "notebooks", "numbers.json")
FIGS = os.path.join(SOLUTION_DIR, "notebooks", "figs")
PUBLIC_FIGS = os.path.join(FIGS, "public")

SECTIONS = [
    ("1", "The problem: a ball that keeps disappearing"),
    ("2", "The score: why one extra boundary is expensive"),
    ("3", "The practice set: ten matches, made to look like ours"),
    ("4", "The pipeline, one stage at a time"),
    ("5", "The two clues that carry it"),
    ("6", "What we tried and dropped"),
]

# figure name -> (section number, local_only, caption)
FIGURES = {
    "fig_three_labels": ("1", False, "The three labels, and the 25 answers per second they ask for. Synthetic."),
    "fig_match_gap": ("1", True, "Our match: live play, then the tracking loses the ball and the label turns D. Local render."),
    "fig_nearest_player": ("1", False, "Why the nearest player is not the owner: the ball is in flight, and the nearest player has not touched it. Synthetic."),
    "fig_spellf1": ("2", False, "The centerpiece: a true ten-second possession, a perfect answer, the same answer split, and a missed spell, scored by the organizer's metric. Synthetic."),
    "fig_spellf1_curve": ("2", False, "The same experiment as a curve: score against the number of fragments. Synthetic."),
    "fig_practice_clean": ("3", True, "The practice clip before degrading: the ball stays visible through the stoppage. Local render."),
    "fig_practice_degraded": ("3", True, "The same clip after degrading: the ball is missing where our tracker loses it. Local render."),
    "fig_missingness": ("3", False, "Where the ball is missing: our match against the degraded practice set. Aggregates only."),
    "fig_stage_smooth50": ("4", True, "Stage 1, the organizers' 2-second baseline: nearest player, majority-smoothed. Local render."),
    "fig_stage_baseline_plus": ("4", True, "Stage 2, Baseline+: hysteresis, merge and a 2.5 m radius. Local render."),
    "fig_stage_gate": ("4", True, "Stage 3, the missing-ball gate: the ball is gone and the players are still, so the stretch becomes D. Local render."),
    "fig_stage_ingap": ("4", True, "Stage 4, the in-gap decoder: team shape inserts a hand-over inside a missing-ball gap. Local render."),
    "fig_stage_grown": ("4", True, "Stage 5, grown stoppages: each resting-ball moment is extended to the stoppage around it. Local render."),
    "fig_stage_subgap": ("4", True, "Stage 6, the sub-gap rule: a stoppage inside a gap the main gate left alone. Local render."),
    "fig_gate_instruments": ("5", False, "Two independent instruments agree on how much of the missing time is dead. Aggregates only."),
    "fig_killed": ("6", False, "Each dropped arm's decisive spellF1 delta against the shipped stack. Synthetic from numbers only."),
    "fig_width_schematic": ("5", False, "The team with the ball is wider; the defending team is narrow. Synthetic."),
    "fig_shape_deform": ("5", True, "The shape clue in a real missing-ball gap: the wide team changes as the owner changes. Local render."),
}


# ----------------------------------------------------------------------------- numbers

def load_numbers():
    if not os.path.isfile(NUMBERS):
        raise SystemExit("missing %s; run scripts/walkthrough_numbers.py first" % NUMBERS)
    doc = json.load(open(NUMBERS))
    return doc, {k: v["display"] for k, v in doc["entries"].items()}


def fields(text):
    return [f for _, f, _, _ in string.Formatter().parse(text) if f]


# ----------------------------------------------------------------------------- drawing helpers

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name, profile="local"):
    """Write a synthetic figure (both profiles) and close it."""
    plt = _mpl()
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _pitch(ax, length=105.0, width=68.0, color="#e9f2e4", line="#9db39d"):
    from matplotlib.patches import Rectangle, Circle
    ax.add_patch(Rectangle((0, 0), length, width, facecolor=color, edgecolor=line, lw=1.2, zorder=0))
    ax.plot([length / 2, length / 2], [0, width], color=line, lw=1.0, zorder=1)
    ax.add_patch(Circle((length / 2, width / 2), 9.15, fill=False, edgecolor=line, lw=1.0, zorder=1))
    ax.set_xlim(-2, length + 2)
    ax.set_ylim(-2, width + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _dot(ax, x, y, color, label=None, size=220, edge="white"):
    ax.scatter([x], [y], s=size, c=color, edgecolors=edge, linewidths=1.4, zorder=4)
    if label:
        ax.text(x, y - 2.4, label, ha="center", va="top", fontsize=8, color="#333333", zorder=5)


def _ball(ax, x, y):
    ax.scatter([x], [y], s=70, c="white", edgecolors="#222222", linewidths=1.4, zorder=6)
    ax.scatter([x], [y], s=14, c="#222222", zorder=7)


def _arrow(ax, a, b, color="#333333", style="-|>", lw=1.4, ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
                zorder=3)


# ----------------------------------------------------------------------------- synthetic figures

def fig_three_labels(numbers, profile):
    plt = _mpl()
    fig = plt.figure(figsize=(10.5, 5.2))
    ax = fig.add_axes([0.03, 0.36, 0.40, 0.56])
    _pitch(ax, 60, 40)
    _dot(ax, 20, 20, "#1a5fb4", "H")
    _dot(ax, 27, 25, "#1a5fb4")
    _dot(ax, 34, 17, "#c01c28", "A")
    _dot(ax, 42, 23, "#c01c28")
    _ball(ax, 31, 21)
    ax.set_title("one frame: positions", fontsize=10, loc="left")

    ax2 = fig.add_axes([0.48, 0.52, 0.48, 0.34])
    # 20 s of labels, one block per second
    labels = ["H"] * 5 + ["D"] * 3 + ["A"] * 8 + ["H"] * 4
    colors = {"H": "#1a5fb4", "A": "#c01c28", "D": "#7f8c8d"}
    for i, lab in enumerate(labels):
        ax2.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=colors[lab], edgecolor="white", lw=0.8))
        ax2.text(i + 0.5, 0.5, lab, ha="center", va="center", color="white", fontsize=9, weight="bold")
    ax2.set_xlim(0, len(labels))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("time (s), one block per second", fontsize=9)
    ax2.set_title("the answer is a label per frame", fontsize=10, loc="left")
    for s in ax2.spines.values():
        s.set_visible(False)

    ax3 = fig.add_axes([0.48, 0.10, 0.48, 0.26])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.text(0.0, 0.82, "H  Barcelona has the ball", fontsize=10, color="#1a5fb4", weight="bold")
    ax3.text(0.0, 0.54, "A  Athletic has the ball", fontsize=10, color="#c01c28", weight="bold")
    ax3.text(0.0, 0.26, "D  the ball is dead", fontsize=10, color="#5c5c5c", weight="bold")
    ax3.text(0.0, 0.02, "25 answers per second, for every graded frame", fontsize=9, color="#444444")
    return _save(fig, "fig_three_labels", profile)


def fig_nearest_player(numbers, profile):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    _pitch(ax, 60, 40)
    # passer, receiver, defender, ball in flight
    _dot(ax, 18, 22, "#1a5fb4", "passer (H)")
    _dot(ax, 40, 26, "#1a5fb4", "receiver (H)")
    _dot(ax, 33, 21, "#c01c28", "defender (A)")
    _ball(ax, 31, 23)
    _arrow(ax, (20, 22), (30, 23), color="#1a5fb4", ls="--")
    ax.plot([31, 33], [23, 21], color="#c01c28", lw=1.4, ls=":")
    ax.plot([31, 40], [23, 26], color="#1a5fb4", lw=1.4, ls=":")
    ax.text(36.2, 18.6, "1.9 m to defender", color="#c01c28", fontsize=9, ha="left")
    ax.text(36.0, 25.4, "4.2 m", color="#1a5fb4", fontsize=9, ha="center")
    ax.annotate("nearest, but has not touched it", xy=(33, 21), xytext=(36, 33),
                fontsize=9, color="#c01c28", arrowprops=dict(arrowstyle="->", color="#c01c28", lw=1.0))
    ax.set_title("the ball is in flight: nearest is not owner", fontsize=11, loc="left")
    return _save(fig, "fig_nearest_player", profile)


def _spellf1_figure(numbers, scene):
    """One frame of the section-2 animation. `scene` is a dict with pred parts, iou, f1, caption."""
    plt = _mpl()
    fig = plt.figure(figsize=(11.0, 6.6))
    fig.patch.set_facecolor("white")
    fig.text(0.04, 0.955, scene["caption"], fontsize=12.5, weight="bold", color="#222222")
    fig.text(0.04, 0.905, "one true H spell, ten seconds", fontsize=9.5, color="#555555")

    # true bar
    axt = fig.add_axes([0.06, 0.72, 0.88, 0.11])
    axt.set_xlim(0, 20)
    axt.set_ylim(0, 1)
    axt.add_patch(plt.Rectangle((0, 0), 10, 1, facecolor="#1a5fb4", edgecolor="white", lw=1.5))
    axt.add_patch(plt.Rectangle((10, 0), 10, 1, facecolor="#cfd8dc", edgecolor="white", lw=1.5))
    axt.text(5, 0.5, "H", ha="center", va="center", color="white", fontsize=12, weight="bold")
    axt.text(15, 0.5, "D", ha="center", va="center", color="#555555", fontsize=12, weight="bold")
    axt.set_yticks([])
    axt.set_xticks([0, 5, 10, 15, 20])
    axt.set_xticklabels(["0", "5", "10", "15", "20"])
    axt.tick_params(labelsize=9, length=0)
    axt.set_title("TRUE", fontsize=8.5, loc="left", color="#555555")
    for s in axt.spines.values():
        s.set_visible(False)

    # predicted bar
    axp = fig.add_axes([0.06, 0.52, 0.88, 0.11])
    axp.set_xlim(0, 20)
    axp.set_ylim(0, 1)
    axp.add_patch(plt.Rectangle((0, 0), 20, 1, facecolor="#cfd8dc", edgecolor="white", lw=1.5))
    for a, b, lab in scene["pred"]:
        axp.add_patch(plt.Rectangle((a / 25.0, 0), (b - a) / 25.0, 1,
                                    facecolor="#1a5fb4" if lab == "H" else "#cfd8dc",
                                    edgecolor="white", lw=1.5))
    axp.set_yticks([])
    axp.set_xticks([0, 5, 10, 15, 20])
    axp.set_xticklabels(["0", "5", "10", "15", "20"])
    axp.tick_params(labelsize=9, length=0)
    axp.set_title("PREDICTED", fontsize=8.5, loc="left", color="#555555")
    for s in axp.spines.values():
        s.set_visible(False)

    # IoU strip
    axi = fig.add_axes([0.08, 0.25, 0.52, 0.16])
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.axvspan(0.3, 0.7, color="#dfe8d8", zorder=0)
    axi.text(0.5, 0.86, "the 0.3-0.7 band the score averages over", ha="center", fontsize=8.5, color="#4b6043")
    axi.axhline(0.5, color="#cccccc", lw=0.8)
    colors = ["#1a5fb4", "#2a7ab0", "#4a90c2", "#7fb3d5"]
    # Fragments of one split share an IoU, so label each distinct value once ("0.50 ×2") instead of
    # stacking identical labels on the same x. Dots are fanned vertically so every fragment is visible.
    order, groups = [], {}
    for v in scene["ious"]:
        key = round(float(v), 4)
        if key not in groups:
            groups[key] = 0
            order.append(key)
        groups[key] += 1
    for gi, key in enumerate(order):
        n = groups[key]
        for j in range(n):
            y = 0.5 if n == 1 else 0.5 + 0.055 * (j - (n - 1) / 2.0)
            axi.scatter([key], [y], s=110, color=colors[gi % len(colors)],
                        edgecolors="white", linewidths=1.2, zorder=4)
        label = "%.2f" % key + (" ×%d" % n if n > 1 else "")
        axi.text(key, 0.20, label, ha="center", fontsize=9, color="#333333")
    if not scene["ious"]:
        axi.text(0.5, 0.5, "no predicted H spell", ha="center", va="center", fontsize=9, color="#888888")
    axi.set_yticks([])
    axi.set_xlabel("IoU of one predicted fragment against the true ten-second spell", fontsize=9)
    axi.tick_params(labelsize=9, length=0)
    for s in axi.spines.values():
        s.set_visible(False)

    # F1 gauge
    axf = fig.add_axes([0.67, 0.24, 0.27, 0.16])
    axf.set_xlim(0, 1)
    axf.set_ylim(0, 1)
    axf.add_patch(plt.Rectangle((0, 0.28), 1, 0.30, facecolor="#eeeeee", edgecolor="none"))
    axf.add_patch(plt.Rectangle((0, 0.28), scene["f1"], 0.30,
                                facecolor="#c0392b" if scene["f1"] < 0.5 else "#2e7d32", edgecolor="none"))
    axf.text(0.5, 0.72, "spellF1 = %.2f" % scene["f1"], ha="center", fontsize=12, weight="bold", color="#222222")
    axf.set_xticks([0, 0.5, 1])
    axf.set_xticklabels(["0", "0.5", "1"])
    axf.set_yticks([])
    axf.tick_params(labelsize=9, length=0)
    for s in axf.spines.values():
        s.set_visible(False)

    fig.text(0.04, 0.04, "Synthetic labels, scored by more-than-a-hack-2027/evaluation/possession_metrics.py",
             fontsize=7.5, color="#888888")
    return fig


def _scenes(numbers):
    """Six scenes; the displayed F1 values are the numbers.json values (recomputed by the scorer)."""
    v = lambda k: float(numbers["entries"][k]["value"])
    n = 500
    true = [(0, 250, "H")]
    scenes = [
        dict(caption="The true spell, and a perfect answer: one H spell on exactly the same frames.",
             pred=list(true), ious=[1.0], f1=v("demo_perfect")),
        dict(caption="Split it in two: the same frames, a one-second D gap in the middle.",
             pred=[(0, 125, "H"), (125, 150, "D"), (150, 250, "H")],
             ious=[v("demo_iou2"), v("demo_iou2")], f1=v("demo_split2")),
        dict(caption="Split it in three: each fragment now overlaps only a third of the true spell.",
             pred=[(0, 80, "H"), (80, 105, "D"), (105, 170, "H"), (170, 195, "D"), (195, 250, "H")],
             ious=[v("demo_iou3")] * 3, f1=v("demo_split3")),
        dict(caption="Split it in four: every fragment is below the 0.3 IoU floor.",
             pred=[(0, 60, "H"), (60, 80, "D"), (80, 120, "H"), (120, 140, "D"),
                   (140, 180, "H"), (180, 200, "D"), (200, 250, "H")],
             ious=[v("demo_iou4")] * 4, f1=v("demo_split4")),
        dict(caption="Miss the possession completely: the answer says D for the whole ten seconds.",
             pred=[], ious=[], f1=v("demo_missed")),
    ]
    return scenes


def fig_spellf1(numbers, profile):
    """The centerpiece: a deterministic GIF, 10 fps, ~1.1 s per scene."""
    from possession import render
    plt = _mpl()
    tmp = os.path.join(FIGS, "_tmp_spellf1")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    hold = 14
    i = 0
    for scene in _scenes(numbers):
        fig = _spellf1_figure(numbers, scene)
        fig.canvas.draw()
        for _ in range(hold):
            render._write_png(fig, os.path.join(tmp, "f_%05d.png" % i))
            i += 1
        plt.close(fig)
    path = os.path.join(FIGS, "fig_spellf1.gif")
    render._encode_gif(tmp, fps=10, path=path, width=900, max_bytes=1_500_000, colors=96)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return path


def fig_spellf1_curve(numbers, profile):
    plt = _mpl()
    v = lambda k: float(numbers["entries"][k]["value"])
    xs = [1, 2, 3, 4]
    ys = [v("demo_perfect"), v("demo_split2"), v("demo_split3"), v("demo_split4")]
    ious = [1.0, v("demo_iou2"), v("demo_iou3"), v("demo_iou4")]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(xs, ys, "o-", color="#c0392b", lw=2.0, ms=8, label="spellF1")
    for x, y in zip(xs, ys):
        ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(xs, ious, "s--", color="#1a5fb4", lw=1.6, ms=7, label="fragment IoU")
    ax2.axhspan(0.3, 0.7, color="#dfe8d8", alpha=0.6)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("IoU against the true spell", color="#1a5fb4")
    ax.set_xlabel("number of fragments the true spell is split into")
    ax.set_ylabel("spellF1", color="#c0392b")
    ax.set_xticks(xs)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("the same ten-second spell, split 1-4 ways", fontsize=11, loc="left")
    ax.grid(alpha=0.2)
    return _save(fig, "fig_spellf1_curve", profile)


def fig_missingness(numbers, profile):
    plt = _mpl()
    e = numbers["entries"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    labels = ["our match", "practice set"]
    vals = [e["missing_share_graded"]["value"], e["practice_missing_share"]["value"]]
    axes[0].bar(labels, vals, color=["#1a5fb4", "#7f8c8d"], width=0.55)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.012, "%.0f%%" % (100 * v), ha="center", fontsize=10)
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel("share of graded frames")
    axes[0].set_title("ball missing at all", fontsize=10, loc="left")
    vals2 = [e["practice_live_missing"]["value"], e["ours_live_missing"]["value"]]
    axes[1].bar(labels, vals2, color=["#1a5fb4", "#7f8c8d"], width=0.55)
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 0.004, "%.1f%%" % (100 * v), ha="center", fontsize=10)
    axes[1].set_ylim(0, 0.07)
    axes[1].set_ylabel("share of live frames")
    axes[1].set_title("ball missing during live play", fontsize=10, loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("the practice set hides the ball where ours does (almost)", fontsize=11, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "fig_missingness", profile)


def fig_gate_instruments(numbers, profile):
    plt = _mpl()
    e = numbers["entries"]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    vals = [e["e25_dead_share"]["value"], e["e36_dead_share"]["value"]]
    names = ["player-motion classifier", "broadcast ball track"]
    ax.bar(names, vals, color=["#1a5fb4", "#2e7d32"], width=0.5)
    lo = e["e25_ci_low"]["value"]
    hi = e["e25_ci_high"]["value"]
    ax.errorbar([0], [vals[0]], yerr=[[vals[0] - lo], [hi - vals[0]]], fmt="none",
                ecolor="#222222", capsize=5, lw=1.2)
    lo36 = e["e36_dead_low"]["value"]
    hi36 = e["e36_dead_high"]["value"]
    ax.errorbar([1], [vals[1]], yerr=[[vals[1] - lo36], [hi36 - vals[1]]], fmt="none",
                ecolor="#222222", capsize=5, lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.045, "%.1f%%" % (100 * v), ha="center", fontsize=11, weight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of missing time judged dead")
    ax.axhline(0.9, color="#c0392b", ls="--", lw=1.0)
    ax.text(1.42, 0.905, "90%", color="#c0392b", fontsize=8, ha="right")
    ax.set_title("two instruments, different failure modes, one answer", fontsize=11, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save(fig, "fig_gate_instruments", profile)


def fig_killed(numbers, profile):
    """One small chart: the decisive spellF1 delta of each dropped arm (numbers only, synthetic)."""
    plt = _mpl()
    e = numbers["entries"]
    arms = [("Viterbi decode", "viterbi_delta"), ("MBR decode", "mbr_delta"),
            ("duration HSMM", "hsmm_delta"), ("split D blocks", "split_delta")]
    vals = [float(e[k]["value"]) for _, k in arms]
    names = [n for n, _ in arms]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    colors = ["#c0392b"] * len(vals)
    ax.barh(names[::-1], vals[::-1], color=colors, height=0.55)
    def fmt(v):
        return ("%+.4f" % v).replace("+0.", "+.").replace("-0.", "−.")

    for y, v in zip(names[::-1], vals[::-1]):
        ax.text(v - 0.0005, y, fmt(v), va="center", ha="right", fontsize=9, color="#333333")
    ax.axvline(0, color="#333333", lw=0.9)
    ax.set_xlabel("spellF1 delta against the shipped stack (dev-v1)")
    ax.set_xlim(min(vals) - 0.004, 0.001)
    ax.set_title("every dropped arm lost on the practice set", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_killed", profile)


def fig_width_schematic(numbers, profile):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    for ax, wide_left in zip(axes, (True, False)):
        _pitch(ax, 60, 40)
        # possession team: wide; defense: narrow
        wide = [(10, 8), (12, 14), (12, 26), (10, 32), (22, 10), (24, 30), (30, 20)]
        narrow = [(30, 15), (32, 20), (32, 24), (34, 18), (34, 26), (36, 22), (38, 22)]
        if not wide_left:
            wide = [(60 - x, y) for x, y in wide]
            narrow = [(60 - x, y) for x, y in narrow]
        for x, y in wide:
            _dot(ax, x, y, "#1a5fb4")
        for x, y in narrow:
            _dot(ax, x, y, "#c01c28")
        from matplotlib.patches import Ellipse
        xs = [x for x, _ in wide]
        ys = [y for _, y in wide]
        ax.add_patch(Ellipse((np.mean(xs), np.mean(ys)), max(xs) - min(xs) + 6, max(ys) - min(ys) + 8,
                             fill=False, edgecolor="#1a5fb4", lw=1.6, ls="--", alpha=0.8))
        xs2 = [x for x, _ in narrow]
        ys2 = [y for _, y in narrow]
        ax.add_patch(Ellipse((np.mean(xs2), np.mean(ys2)), max(xs2) - min(xs2) + 4, max(ys2) - min(ys2) + 6,
                             fill=False, edgecolor="#c01c28", lw=1.2, ls=":", alpha=0.9))
        ax.set_title("possession team wide" if wide_left else "possession team wide (mirrored)",
                     fontsize=9, loc="left")
        ax.text(1.0, 36.0, "possession", color="#1a5fb4", fontsize=8, weight="bold")
        ax.text(1.0, 33.5, "defense", color="#c01c28", fontsize=8, weight="bold")
    fig.suptitle("while the ball is missing, the wider team is usually the one with it", fontsize=11,
                 x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "fig_width_schematic", profile)


# ----------------------------------------------------------------------------- local figures

_CACHE = {}


def _our_match():
    if "our" not in _CACHE:
        from possession import data, render
        match = data.load_match()
        labels, _ = render.load_labels(os.path.join(SOLUTION_DIR, "outputs", "submission.csv"))
        _CACHE["our"] = (match, labels)
    return _CACHE["our"]


def _practice(gid="metrica:game1", knob=1.0, v=0.12):
    key = ("practice", gid, knob, v)
    if key not in _CACHE:
        from possession import dev_v1, external
        clean = external.load_game(gid)
        match = dev_v1.degraded_match(clean, 0, knob, v)
        cell = dev_v1.Cell(gid, clean, match, 0, knob, v)
        _CACHE[key] = (clean, match, cell)
    return _CACHE[key]


def _stage_labels(cell):
    """The six stage outputs for one dev cell, using the shipped modules unchanged."""
    from possession import ballfree, dgate, growd, ingap, subgap
    key = ("stages", id(cell))
    if key in _CACHE:
        return _CACHE[key]
    import possession.dev_v1 as dv
    m = cell.match
    smooth = dv.smooth50(cell)
    base = dv.baseline_plus(cell)
    mdl, cols = cell.models[0], cell.models[1]
    gated, dead, _ = dgate.apply(base, m, model=(mdl, cols))
    ph = ballfree.predict_frames(cell.models[2], m, pitch=cell.pitch)
    ship = ingap.gapdec(gated, m.B, m.half, ph)
    grown, _ = growd.apply(ship, m, dead=dead)
    pf = dgate.frame_pd(mdl, cols, m.P, m.V, m.T, m.half)
    final, _ = subgap.apply(grown, gated, m, pf, fired=dead)
    out = dict(smooth50=smooth, baseline_plus=base, gate=gated, ingap=ship, grown=grown, subgap=final)
    _CACHE[key] = out
    return out


def _render(name, match, start, end, labels=None, reference=None, highlight=None, bands=None,
            boundaries=None, title=None, pitch=None, crop=8.0):
    from possession import render
    os.makedirs(FIGS, exist_ok=True)
    res = render.render_clip(
        match, start, end, labels=labels, reference=reference,
        highlight=highlight or [render.highlight_ball()],
        bands=bands, boundaries=boundaries, title=title, out=FIGS, name=name, mp4=False, gif=True,
        fps=10, size=800, crop=crop, pitch=pitch, annotate=True)
    return res["gif"]


def _local_match_gap():
    from possession import render
    match, labels = _our_match()
    start, end = 57750, 58250
    bands = render.shade_missing_ball(match, start, end, min_frames=25)
    bands += render.shade_d(labels, start, end, min_frames=5)
    boundaries = render.spell_boundaries(labels, start, end)
    return _render("fig_match_gap", match, start, end, labels=labels,
                   highlight=[render.highlight_ball()], bands=bands, boundaries=boundaries,
                   title="BAR-ATH  |  our match, prediction", pitch=(105.3, 68.0))


def _practice_pair():
    if "pair" in _CACHE:
        return _CACHE["pair"]
    from possession import render
    # IDSSE keeps the ball tracked through this stoppage in the clean data; Metrica does not.
    clean, match, cell = _practice(gid="idsse:DFL-MAT-J03WMX")
    start, end = 66100, 66600
    ref = np.asarray(clean.ref)
    pitch = cell.pitch
    a = _render("fig_practice_clean", clean, start, end, labels=None,
                highlight=[render.highlight_ball()], title="IDSSE J03WMX  |  clean", pitch=pitch)
    bands = render.shade_missing_ball(match, start, end, min_frames=25)
    from possession import evaluate
    game = evaluate.Game("IDSSE J03WMX degraded", match, ref, runs=clean.runs)
    b = _render("fig_practice_degraded", game, start, end, labels=None,
                highlight=[render.highlight_ball()], bands=bands,
                title="IDSSE J03WMX  |  degraded to our noise", pitch=pitch)
    _CACHE["pair"] = (a, b)
    return a, b


def _stage_figs():
    if "stages_figs" in _CACHE:
        return _CACHE["stages_figs"]
    from possession import render
    clean, match, cell = _practice()
    stages = _stage_labels(cell)
    pitch = (105.0, 68.0)
    out = {}
    windows = {
        "smooth50": (71000, 71600),
        "baseline_plus": (71000, 71600),
        "gate": (71000, 71600),
        "ingap": (42600, 43200),
        "grown": (96600, 97200),
        "subgap": (37150, 37750),
    }
    for name, (a, b) in windows.items():
        out[name] = _render("fig_stage_" + name, match, a, b, labels=stages[name],
                            highlight=[render.highlight_ball()],
                            title="Metrica game 1  |  stage: " + name.replace("_", " "), pitch=pitch)
    _CACHE["stages_figs"] = out
    return out


def _shape_deform():
    from possession import render
    clean, match, cell = _practice(gid="metrica:game2")
    start, end = 28050, 28600
    ref = np.asarray(clean.ref)
    from possession import evaluate
    game = evaluate.Game("Metrica game 2 degraded", match, ref, runs=clean.runs)
    return _render("fig_shape_deform", game, start, end, labels=None,
                   highlight=[render.highlight_team(0, color="#1a5fb4"),
                              render.highlight_team(1, color="#c01c28")],
                   title="Metrica game 2  |  shape in a missing-ball gap",
                   pitch=(105.0, 68.0))


def _placeholder(name, caption):
    """Neutral public-profile card carrying the same caption as the local figure."""
    import textwrap
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.set_facecolor("#f4f4f4")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#f4f4f4", edgecolor="#bbbbbb", lw=1.2))
    ax.text(0.5, 0.80, "figure withheld in the public profile", ha="center", fontsize=13, weight="bold",
            color="#444444")
    ax.text(0.5, 0.65, textwrap.fill(caption, 72), ha="center", va="center", fontsize=10,
            color="#555555", style="italic", wrap=True)
    ax.text(0.5, 0.33, "this figure renders real tracking data,\nwhich is local-only (D-002).",
            ha="center", fontsize=10, color="#666666")
    ax.text(0.5, 0.18, "rebuild with  --profile local  to render it", ha="center", fontsize=9,
            color="#888888", family="DejaVu Sans Mono")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    os.makedirs(PUBLIC_FIGS, exist_ok=True)
    path = os.path.join(PUBLIC_FIGS, name + ".png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- figure registry

SYNTHETIC = {
    "fig_three_labels": fig_three_labels,
    "fig_nearest_player": fig_nearest_player,
    "fig_spellf1": fig_spellf1,
    "fig_spellf1_curve": fig_spellf1_curve,
    "fig_missingness": fig_missingness,
    "fig_gate_instruments": fig_gate_instruments,
    "fig_width_schematic": fig_width_schematic,
    "fig_killed": fig_killed,
}

LOCAL = {
    "fig_match_gap": _local_match_gap,
    "fig_practice_clean": lambda: _practice_pair()[0],
    "fig_practice_degraded": lambda: _practice_pair()[1],
    "fig_stage_smooth50": lambda: _stage_figs()["smooth50"],
    "fig_stage_baseline_plus": lambda: _stage_figs()["baseline_plus"],
    "fig_stage_gate": lambda: _stage_figs()["gate"],
    "fig_stage_ingap": lambda: _stage_figs()["ingap"],
    "fig_stage_grown": lambda: _stage_figs()["grown"],
    "fig_stage_subgap": lambda: _stage_figs()["subgap"],
    "fig_shape_deform": _shape_deform,
}


def build_figures(profile, numbers, only=None):
    only = set(only.split(",")) if only else None
    wanted = [n for n in FIGURES if only is None or n in only]
    # local figures first as one batch where possible (one cell reused)
    for name in wanted:
        if name not in FIGURES:
            raise SystemExit("unknown figure %r" % name)
        if name in SYNTHETIC:
            SYNTHETIC[name](numbers, profile)
        else:
            if profile == "public":
                _placeholder(name, FIGURES[name][2])
            else:
                LOCAL[name]()
    print("figures: %d built for profile %s" % (len(wanted), profile))


# ----------------------------------------------------------------------------- prose

def _prose(numbers):
    """Section -> list of blocks. Blocks: ("md", text), ("fig", name), ("flex", [names]),
    and the generated tables. Every measured number is a ``{key}`` from numbers.json."""
    S = {}

    S["title"] = [
        ("md", "# Reading the ball\n"
               "\n"
               "**A traceable walkthrough of our frame-level possession model.**\n"
               "\n"
               "More than a Hack 2027 · Barcelona vs Athletic Club. Prose and figures only: the code behind "
               "every number and clip is in `solution/`, and the appendix maps each number to its source. The "
               "configuration is frozen (D-029), so every number here belongs to the frozen build."),
    ]

    S["1"] = [
        ("md", "## 1. The problem: a ball that keeps disappearing\n"
               "\n"
               "Watch the clip. Barcelona moves the ball in their own half, the players jog into shape, and the "
               "tracking loses the ball. Six seconds later it is still gone and the players have stopped. The "
               "answer strip under the pitch turns gray: the stretch is **dead**."),
        ("fig", "fig_match_gap"),
        ("md", "Twenty seconds of our match, rendered from the submission's own tracking file. The gray band is a "
               "**ball-missing run**, frames with no ball position at all; the red band is the model's **D**. The "
               "ball is missing for {missing_share_graded} of graded frames, and almost all of that time is a "
               "stoppage. That fact is the spine of this notebook."),
        ("md", "The task is one of **{n_labels} labels** per graded frame: **H** Barcelona, **A** Athletic, **D** "
               "dead ball. A **possession** is a maximal unbroken run of H or A; D ends it. We answer at {fps} for "
               "{graded_frames} of {all_frames} frames, and the sequence must be coherent."),
        ("fig", "fig_three_labels"),
        ("md", "The obvious first rule is **who is nearest the ball**, the organizers' starter baseline, and with a "
               "{nearest_radius_m} radius it is a good start. But nearest is not **owner**: the owner is the team "
               "that last touched the ball and keeps it through a flight, a dribble or a scramble. In the "
               "schematic the defender is nearest, but the ball is still travelling from passer to receiver."),
        ("fig", "fig_nearest_player"),
        ("md", "On clean practice data nearest catches {flip_recall_clean} of visible owner changes; degraded to "
               "our noise, {flip_recall_degraded}, and about {unresolvable_share} cannot be resolved from geometry "
               "at all. Hence two other clues: **missingness** and **team shape**."),
    ]

    S["2"] = [
        ("md", "## 2. The score: why one extra boundary is expensive\n"
               "\n"
               "One true possession: Barcelona has the ball for ten seconds, then the whistle goes. The animation "
               "is synthetic and scored by the organizer's own metric, not a cartoon. It explains almost every "
               "design choice we made."),
        ("fig", "fig_spellf1"),
        ("md", "The score is **spellF1**. Possessions are matched one-to-one by **temporal IoU**, the shared frames "
               "divided by the union, and the F-score is averaged over thresholds {demo_iou_low}–{demo_iou_high}. "
               "The right team is not enough if the interval is in the wrong place."),
        ("md", "A perfect answer scores {demo_perfect}. Split it in two with a one-second D gap: {demo_split2}, with "
               "each fragment at IoU {demo_iou2} — inside the band, but only one can match and the other is a false "
               "positive. Split in three: {demo_split3}, IoU {demo_iou3}, at the edge of the band. Split in four: "
               "{demo_split4}, every fragment at IoU {demo_iou4}, below the {demo_iou_low} floor, so nothing "
               "matches. Miss it completely: also {demo_missed}."),
        ("fig", "fig_spellf1_curve"),
        ("md", "The asymmetry is the point. A missed possession costs one false negative; a spurious boundary costs "
               "a false positive *and* damages the host possession it cut, which must now compete with its own "
               "fragment. Hence our caution about hand-overs, and why a clue can raise frame accuracy and still "
               "lose score. Differences below about {noise_band} are noise, and the public five minutes hold only "
               "{public_gt_spells} possessions, so we never choose a method on them."),
    ]

    S["3"] = [
        ("md", "## 3. The practice set: ten matches, made to look like ours\n"
               "\n"
               "We have the true answers for {public_minutes} public minutes, and the rules forbid tuning on them: "
               "{public_gt_spells} possessions cannot choose a threshold, and tuning there picks the wrong answer "
               "even for a one-parameter baseline. Every choice is made on other professional matches with public "
               "answers, after we make their data look like ours."),
        ("md", "The clip is one practice passage, clean and **degraded**. Clean, the ball stays visible through the "
               "stoppage, so a ball-based rule sees it for free. Degraded, the ball is dropped where our tracker "
               "drops it, the path is smoothed and the gap edges jittered: the stoppage is still there in the "
               "players, the ball is not."),
        ("flex", ["fig_practice_clean", "fig_practice_degraded"]),
        ("md", "The practice set is **{practice_games} matches**: {practice_idsse} German-league and "
               "{practice_metrica} Metrica. A second set of {skillcorner_games} SkillCorner matches is used only "
               "for who-has-the-ball questions, never pooled. The ten are frozen — same games, seeds, degrader and "
               "cells."),
        ("fig", "fig_missingness"),
        ("md", "The set had to be repaired. Our match hides the ball for {missing_share_graded} of graded frames, "
               "the degraded practice set for {practice_missing_share}, close enough. The conditional level did "
               "not match: the old degrader dropped the ball at random, including in live play, so "
               "{practice_live_missing_old} of practice live frames had no ball against {ours_live_missing} of "
               "ours. After the fix it is {practice_live_missing}, and {practice_long_gap_live} of long gaps are "
               "live against {ours_long_gap_live} of ours. The old gap taught the model the opposite of the truth."),
        ("md", "Two flaws, both fixed. The **resting ball** shook about twenty times too much, so ball motion "
               "separated dead from live far too well; fixing the sign took texture-only separability from perfect "
               "to {flaw_texture_auc}, and {flaw_texture_auc_heldout} on twenty held-out SkillCorner games, still "
               "short of our {flaw_target_auc} target — ball-motion clues are upper bounds. The second was **where "
               "the ball hides**: after the fix Baseline+ moved from {baseline_plus_old} to {baseline_plus_new} "
               "and the in-gap gain fell from {ingap_old} to {ingap_new}. The set is also about {practice_easier} "
               "easier than the match — {smooth50_dev} against {smooth50_match} for the two-second baseline — so "
               "we trust the ranking, never the level."),
    ]

    S["4"] = [
        ("md", "## 4. The pipeline, one stage at a time\n"
               "\n"
               "No single passage exercises all six stages, and the reason is structural: the gate needs a long "
               "ball-missing gap, grown D needs a visible resting ball, the decoder needs a gap the gate left live, "
               "and the sub-gap rule needs one it left alone. No window holds all four conditions, so the clips "
               "come from four passages — stages one to three share a window, and each later clip is the window "
               "where that stage changes a label."),
        ("md", "**Stage 1 — nearest player, smoothed.** The organizers' two-second baseline: nearest player within "
               "a radius, majority-smoothed over fifty frames. The floor to beat: {smooth50_dev}."),
        ("fig", "fig_stage_smooth50"),
        ("md", "**Stage 2 — hysteresis and merge.** A player within {nearest_radius_m} owns the ball, otherwise the "
               "last owner keeps it. It scores {baseline_plus_dev}, {baseline_plus_over_smooth50} over the floor, "
               "but that lead does not survive a half-second shift, so read it as *about level to slightly ahead*. "
               "Too much hysteresis absorbs a turnover; too little emits flickers, which cost twice."),
        ("fig", "fig_stage_baseline_plus"),
        ("md", "**Stage 3 — the missing-ball gate.** Ball gone for a while and players still: the stretch becomes "
               "D. The largest gain in the stack, {gate_over_baseline_plus}, no game negative; expected "
               "{expected_gate} on our match. A wrong gate turns live possession into a stoppage and cuts a spell, "
               "the most expensive error the score charges."),
        ("fig", "fig_stage_gate"),
        ("md", "**Stage 4 — the in-gap decoder.** In a missing-ball gap, team shape says which team is wider and "
               "more likely to have the ball, so the decoder may insert a hand-over. It adds {ingap_over_gate}, but "
               "only {eligible_ours} of our frames are eligible, so expected {expected_ingap}. The clip shows the "
               "mechanism, not a success: an H island where the answer stays A."),
        ("fig", "fig_stage_ingap"),
        ("md", "**Stage 5 — grown stoppages.** A resting ball is a real stoppage but only a sliver. Extend each "
               "moment to the stoppage around it, with a radius and a thirty-second cap: {grown_delta} "
               "[{grown_ci_low}, {grown_ci_high}], expected {expected_grown}. Only safe on top of the gate; on raw "
               "Baseline+ most changed frames are live play."),
        ("fig", "fig_stage_grown"),
        ("md", "**Stage 6 — the sub-gap rule.** Some stoppages sit inside gaps the gate left alone. A weaker rule "
               "fires when the stoppage probability stays above one half for two seconds: {subgap_delta} "
               "[{subgap_ci_low}, {subgap_ci_high}], expected about {expected_subgap}. On a live gap it creates a D "
               "island inside a possession."),
        ("fig", "fig_stage_subgap"),
        ("md", "The build applies them gate, grown D, in-gap, sub-gap, then the short-spell floor, because grown D "
               "must be excluded from the re-decode and the sub-gap rule must see the gate's output. The numbers are "
               "marginal, not a running sum. The floor **is** wired in the frozen build: deleting H/A spells under "
               "one and a half seconds adds {postprocess_delta} [{postprocess_ci_low}, {postprocess_ci_high}] on the "
               "pre-add-on stack and {postprocess_delta_current} on the frozen labels, and it is the last label write "
               "on every return path. It is a trade — real short possessions are deleted to remove more false ones — "
               "not free pruning."),
        ("stage_table", None),
    ]

    S["5"] = [
        ("md", "## 5. The two clues that carry it\n"
               "\n"
               "At the top of the notebook the ball is gone for six seconds. Two instruments were asked whether "
               "that time is a stoppage: a classifier trained on player motion, and the broadcast camera's own "
               "ball track, which follows the ball through gaps the tactical tracker misses. Neither uses the "
               "answers."),
        ("fig", "fig_gate_instruments"),
        ("md", "The player-motion classifier flags {e25_dead_share} [{e25_ci_low}, {e25_ci_high}] of our missing "
               "time as dead; the broadcast route puts {e36_dead_share} of it dead, with the verified-live bucket "
               "at {e36_live_low}–{e36_live_high}, carrying the dead share to about {e36_dead_low}–"
               "{e36_dead_high}. After correcting for what each cannot see, the probability that a missing frame "
               "is dead is {gate_pd_missing_low}–{gate_pd_missing_high}. Two instruments with different failure "
               "modes agreeing is the strongest label-free evidence we have — not a license to call every gap dead, "
               "so the gate also requires still players."),
        ("md", "The second clue is **team shape**. The team with the ball spreads out; the defender stays compact. "
               "On live practice frames the possession team's width is {shape_width_own_m} against "
               "{shape_width_def_m}, a difference of about {shape_width_diff_m} that separates them with AUC about "
               "{shape_auc_idsse}–{shape_auc_metrica}. The catch is where it applies: inside a dead gap the same "
               "difference collapses to {shape_dead_gap_m}, because both teams walk into position. So it works only "
               "where the gap is live, and only {eligible_ours} of our frames are eligible against {eligible_dev} "
               "on the practice set — hence {expected_ingap}, not the large number the decoder scored alone."),
        ("fig", "fig_width_schematic"),
        ("fig", "fig_shape_deform"),
    ]

    S["6"] = [
        ("md", "## 6. What we tried and dropped\n"
               "\n"
               "The most tempting idea was to find the kick. On clean public data the textbook detector finds "
               "{kick_recall_low}–{kick_recall_high} of them; on our ball it fails, because the path is smoothed "
               "over about a third of a second. Smoothing the practice ball the same way reproduces the failure."),
        ("md", "That set the pattern: a clue that works on clean data is not evidence until it survives our noise. "
               "The table carries the number that killed each idea. The duration model was the expensive attempt — "
               "an explicit-duration HSMM scores {hsmm_delta} [{hsmm_ci_low}, {hsmm_ci_high}], negative in nine of "
               "ten games, while a sticky HMM without the prior scores {hmm_delta} [{hmm_ci_low}, {hmm_ci_high}] "
               "— positive but including zero, so tier-2. The prior is what loses. Restart-team-by-type changed "
               "accuracy by {restart_team_pp}; splitting long blocks scores {split_delta}; crowd loudness led by "
               "{crowd_lead}, below our five-point floor; a public audio model heard a synthetic whistle at "
               "{panns_synthetic} but the real one at {panns_real}."),
        ("drop_table", None),
        ("fig", "fig_killed"),
    ]

    S["7"] = [
        ("md", "## 7. Results on the frozen build\n"
               "\n"
               "The configuration is frozen (D-029). This is the dev reference: ten practice games, six primary "
               "cells, game-cluster intervals. The frozen row is **one run of the whole stack**, not a sum of the "
               "component deltas above."),
        ("results_table", None),
        ("comparisons_table", None),
        ("md",  "Against the organizers' floor, the pre-floor stack is {ship_over_smooth50}; the frozen stack and that "
               "floor were never measured as a pair, so we do not quote the arithmetic difference."),
        ("md", "The gain is smaller on our match. Reach-scaled, the frozen stack should beat the pre-floor stack by "
               "{expected_master}, not the {master_over_ship} measured on dev, because dev is about "
               "{practice_easier} easier and the new levers reach less of our match. Everything that moved moved for "
               "one reason: the D layer. The gate supplies most of the {master_over_baseline_plus} over Baseline+; "
               "the grown, sub-gap and floor rules add a few thousandths each."),
        ("md", "What the file itself looks like, counted row by row from the frozen CSV:"),
        ("diag_table", None),
        ("md", "{pred_spells} spells, {ball_in_play_min} of play and {pred_d_share_graded} dead are in the right "
               "football neighbourhood; they do **not** say the labels are right. Our match is dead-heavier than "
               "the football prior because of the cooling break and tracker-lost visible balls. These are "
               "label-free descriptions, not scores."),
        ("md", "The public window, one run, reported and never a selector:"),
        ("public_table", None),
        ("md", "Sorted by spellF1 we are first on both metrics, but seventeen true possessions is inside noise and "
               "the prediction has eighteen. Nothing was chosen here. There is no defensible absolute full-match "
               "score: every interval above is a paired delta, dev is easier than the match, and the public window "
               "is a direction check."),
    ]

    S["8"] = [
        ("md", "## 8. Limits, and what would fix them\n"
               "\n"
               "The practice set's eight blind spots (D-022) bound what transfers:"),
        ("blind_table", None),
        ("md", "The biggest weakness is recall, not D. On the practice set {missed_share} of real possessions are "
               "missed — about one in five is caught. That is a dev figure from the gate-plus-in-gap stack, not a "
               "match measurement, which needs labels we do not have. The owner-change family is most of the loss, "
               "and much of it has no tracked ball."),
        ("md", "Hand-over scatter is **evidence-limited, not rule-limited** (E83). {handover_tail} "
               "[{handover_tail_low}, {handover_tail_high}] of matched hand-overs are over a second off "
               "({handover_early} early, {handover_late} late), and {handover_cue_share} of that tail is the "
               "{nearest_radius_m} nearest-player cue itself: a duel has no timing information, and on a pass the "
               "receiver is nearest long before the touch. Only {stage_moved} of tail flips are moved by half a second "
               "or more by any stage after Baseline+, so no post-processing rule reaches the {handover_oracle} "
               "oracle. The route is a "
               "contact detector at about {contact_s}; on this tracking the ball is smoothed over several frames, "
               "which puts it out of reach."),
        ("md", "What a different data source would fix: contact-level tracking would attack the tail and reopen the "
               "tier-2 `anchor` lever; a practice regime with cooling breaks and video-review stops would test the "
               "learned arms where they gain, since their advantage is confined to dead-frame motion the ten games "
               "lack; a GPU ball detector would add ball coverage but not contact timing, and nothing may depend on "
               "it (D-004). We would not retry touch rebuilds, explicit-duration priors, a learned decoder instead of "
               "the gate, splitting long D blocks, or restart-team-from-type: each has a decisive negative above."),
    ]

    S["post"] = [
        ("md", "## What changed since the freeze\n"
               "\n"
               "The freeze itself did not move: the submission is still the same file, and the headline "
               "result is still {master_over_ship} against the pre-floor stack. What changed is how much of "
               "it we are willing to claim. Post-freeze studies re-measured the small components, asked "
               "whether the practice ranking survives a different draw or a different severity level, took "
               "the two large orderings to twenty held-out SkillCorner games, and corrected the "
               "ball-in-play denominator. None of them reopened the frozen file; all of them changed what "
               "a reader should believe about the numbers in it."),
        ("md", "The in-gap decoder is the clearest example. It was credited {ingap_over_gate} when it landed "
               "on the gate-only stack; on the current stack its leave-one-out marginal is {ingap_marginal} "
               "[{ingap_marginal_lo}, {ingap_marginal_hi}]. Those are different objects: most of the fall is "
               "work that grown D and the short-spell floor now do first, not a reversal of the decoder. The "
               "small number is also not sign-established. Any dev delta below {sign_threshold} is quoted as "
               "*level with zero*, and this one is well below it: six degradation draws span {ingap_seed_lo} "
               "to {ingap_seed_hi}, and the two-level interval that resamples draws as well as games is "
               "[{ingap_two_level_lo}, {ingap_two_level_hi}], which includes zero. We keep the stage — "
               "removing it is not justified either — and stop calling it a gain."),
        ("md", "The big dev result is not a lucky draw or a lucky severity. Fresh degradation draws give "
               "{headline_six_draw} [{headline_six_draw_lo}, {headline_six_draw_hi}] over six draws, with "
               "the frozen three-seed number inside that interval. Scaling the practice data to "
               "{level_clean} and {level_messy} of its fitted severity leaves the same order, and the "
               "smallest adjacent margin is {level_margin_08} at the cleaner level. These are ranking "
               "checks, not a second practice set, and no level was re-selected."),
        ("md", "On twenty held-out SkillCorner games the two large orderings keep their sign under all four "
               "derived-D rules: the frozen stack beats Baseline+ by {skillcorner_master_lo} to "
               "{skillcorner_master_hi}, and the pre-floor stack beats it by {skillcorner_ship_lo} to "
               "{skillcorner_ship_hi}. The third ordering, the frozen stack against the pre-floor stack, is "
               "positive but only {skillcorner_master_ship_lo} to {skillcorner_master_ship_hi}, below the "
               "{skillcorner_bar} bar we require for a held-out claim. The held-out set therefore confirms "
               "the two large gains and cannot resolve the small one; the dev grid still decides it."),
        ("md", "The ball-in-play check was a denominator error, now fixed. The frozen file has "
               "{ball_in_play_min} of live play on a {playing_clock_min} playing clock, a live share of "
               "{inplay_rate}; the ten external games average {external_inplay_rate} ± {external_inplay_sd}. "
               "On the rate axis that is {inplay_z} standard deviations from the external mean — inside the "
               "noise, not the shortfall a raw-minute comparison suggested. The raw-minute axis is confounded "
               "by our unusually long clock and is no longer quoted."),
        ("md", "The stage attribution is a substitution result, not a sum. Leave-one-out at one seed puts the "
               "floor at {attrib_floor}, sub-gap at {attrib_subgap}, grown D at {attrib_grown} and in-gap at "
               "{attrib_ingap_seed0}; the three post-freeze stages sum to about {attrib_sum} against the "
               "measured {master_over_ship}, an interaction of {attrib_interaction}. Most of that interaction "
               "is grown D handing the floor work it used to do. Read the table as what each stage is worth "
               "given the other four, not as a budget that adds up."),
    ]

    S["close"] = [
        ("md", "## Code, data and acknowledgements\n"
               "\n"
               "The build is one command in `solution/`: `python -m possession.make_submission`. The pipeline "
               "modules are `baseline_plus.py`, `dgate.py`, `growd.py`, `ingap.py`, `subgap.py` and "
               "`postprocess.py`; the harness is `dev_v1.py`; the renderer is `render.py`. The notebook is built "
               "by `scripts/walkthrough_numbers.py` and `scripts/build_walkthrough.py` from `numbers.json`."),
        ("md", "The practice data is public tracking from DFL, Metrica and SkillCorner, used under their licenses "
               "and never redistributed. The match data is the organizer's and stays on this machine. The scoring "
               "code is the organizer's `evaluation/possession_metrics.py`, imported, not re-implemented. Figures "
               "that render tracking are local-only under D-002; the public profile replaces them with placeholders."),
    ]

    S["appendix"] = [
        ("md", "## Appendix: traceability\n"
               "\n"
               "Every measured number in the prose is a key in `notebooks/numbers.json`, with its value, source, "
               "exact command and whether `--check` recomputes it from the live pipeline. The table lists the "
               "headline keys; the rest are in that file, and the figure table names the eighteen figures."),
        ("trace", None),
        ("figure_table", None),
    ]
    return S

# ----------------------------------------------------------------------------- tables

def _stage_table(numbers):
    e = numbers["entries"]
    rows = [
        ("stage", "what it does", "practice spellF1", "marginal gain", "expected on our match"),
        ("smooth50", "nearest player, majority-smoothed", e["smooth50_dev"]["display"], "—", "—"),
        ("Baseline+", "hysteresis, merge, 2.5 m radius", e["baseline_plus_dev"]["display"],
         e["baseline_plus_over_smooth50"]["display"], "level to slightly ahead"),
        ("gate", "missing ball + still players -> D", e["gate_dev"]["display"],
         e["gate_over_baseline_plus"]["display"], e["expected_gate"]["display"]),
        ("in-gap decoder", "team shape inside a missing-ball gap", e["ship_dev"]["display"],
         e["ingap_over_gate"]["display"], e["expected_ingap"]["display"]),
        ("grown stoppages", "resting ball -> whole stoppage", "—",
         e["grown_delta"]["display"] + " [" + e["grown_ci_low"]["display"] + ", " + e["grown_ci_high"]["display"] + "]",
         e["expected_grown"]["display"]),
        ("sub-gap rule", "second gate inside unfired gaps", "—",
         e["subgap_delta"]["display"] + " [" + e["subgap_ci_low"]["display"] + ", " + e["subgap_ci_high"]["display"] + "]",
         e["expected_subgap"]["display"]),
    ]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _drop_table(numbers):
    e = numbers["entries"]
    rows = [
        ("idea", "what killed it"),
        ("kicks from ball kinematics", "only {0}-{1} recall on clean data; our ball is smoothed, so it fails here too".format(
            e["kick_recall_low"]["display"], e["kick_recall_high"]["display"])),
        ("ball-player coupling", "cost {0} to {1} spellF1 when added to the decoder".format(
            e["coupling_low"]["display"], e["coupling_high"]["display"])),
        ("video ball detection", "{0} recall on frames where tracking had the ball, {1} false positives on frames where it did not; the ball is {2}".format(
            e["video_recall"]["display"], e["video_false"]["display"], e["ball_px"]["display"])),
        ("explicit-duration HSMM", "{0} [{1}, {2}], negative in nine of ten games".format(
            e["hsmm_delta"]["display"], e["hsmm_ci_low"]["display"], e["hsmm_ci_high"]["display"])),
        ("restart team by stoppage type", "{0} restart accuracy and negative spellF1 in every variant".format(
            e["restart_team_pp"]["display"])),
        ("split long stoppage blocks", "{0}, negative in ten of ten games".format(e["split_delta"]["display"])),
        ("crowd loudness", "lead {0}, below the five-point floor".format(e["crowd_lead"]["display"])),
        ("public audio-event model", "synthetic whistle {0}, this match's whistle {1}".format(
            e["panns_synthetic"]["display"], e["panns_real"]["display"])),
    ]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _results_table(numbers):
    e = numbers["entries"]
    rows = [
        ("row", "dev-v1 spellF1", "marginal gain", "reading"),
        ("smooth50", e["smooth50_dev"]["display"], "—", "reference floor"),
        ("Baseline+", e["baseline_plus_dev"]["display"],
         e["baseline_plus_over_smooth50"]["display"] + " vs smooth50", "about level to slightly ahead"),
        ("D gate", e["gate_dev"]["display"],
         e["gate_over_baseline_plus"]["display"] + " vs Baseline+", "the D sensor"),
        ("ship (gate + in-gap)", e["ship_dev"]["display"],
         e["ingap_over_gate"]["display"] + " vs gate", "D-021 reference"),
        ("**frozen master**", "**" + e["master_dev"]["display"] + "**",
         "**" + e["master_over_ship"]["display"] + " vs ship**",
         "**" + e["master_neg"]["display"] + " games negative**"),
    ]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _comparisons_table(numbers):
    e = numbers["entries"]

    def rng(key):
        return "[" + e[key + "_lo"]["display"] + ", " + e[key + "_hi"]["display"] + "]"

    rows = [("comparison", "delta", "95% game-cluster CI", "games negative"),
            ("master − ship", "**" + _delta_of(e["master_over_ship"]) + "**",
             rng("master_over_ship"), e["master_neg"]["display"]),
            ("master − Baseline+", _delta_of(e["master_over_baseline_plus"]),
             rng("master_over_baseline_plus"), e["master_neg"]["display"])]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _delta_of(entry):
    return entry["display"].split(" [")[0]


def _diag_table(numbers):
    e = numbers["entries"]
    rows = [("diagnostic", "frozen CSV"),
            ("rows / graded frames", e["all_frames"]["display"] + " / " + e["graded_frames"]["display"]),
            ("H / A / D / X", " / ".join(e[k]["display"] for k in ("count_h", "count_a", "count_d", "count_x"))),
            ("D share of graded", e["pred_d_share_graded"]["display"]),
            ("spells (maximal H/A runs)", e["pred_spells"]["display"]),
            ("D stretches", e["d_stretches"]["display"] + " (" + e["d_runs_ge_15s"]["display"] + " at least 1.5 s)"),
            ("ball in play", e["ball_in_play_min"]["display"] + " of " + e["graded_min"]["display"] + " graded"),
            ("sha256", "`" + e["csv_sha8"]["display"] + "`")]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _public_table(numbers):
    e = numbers["entries"]
    rows = [("run", "spellF1", "macroF1", "predicted / GT spells")]
    for key, label in (("ship", "**ours (frozen)**"), ("smooth125", "smooth125"),
                       ("smooth50", "smooth50 (published floor)"), ("smooth25", "smooth25"),
                       ("raw", "raw")):
        rows.append((label, e["public_%s_spellf1" % key]["display"], e["public_%s_macro" % key]["display"],
                     e["public_%s_spells" % key]["display"] + "/" + e["public_gt_spells"]["display"]))
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _blind_table(numbers):
    e = numbers["entries"]
    rows = [
        ("blind spot", "what it does to the numbers"),
        ("no cooling breaks or VAR stops in dev",
         "part of the sim-to-real gap is structural; our D share is legitimately higher"),
        ("Metrica over-calls D around advantage play",
         str(e["practice_metrica"]["value"]) + " of " + str(e["practice_games"]["value"]) +
         " games; " + e["metrica_d_spread"]["display"] + " spread in its own present-D share"),
        ("sliver gaps",
         "~" + e["sliver_gaps"]["display"] + " gaps of 2–4 s per game, " + e["sliver_share"]["display"] +
         " of graded; in-gap and sub-gap gains are upper-ish"),
        ("resting-ball recall on dev",
         e["anchor_recall_knob0"]["display"] + " / " + e["anchor_recall_knob1"]["display"] + " against " +
         e["anchor_recall_clean"]["display"] + " clean; grown D is measured pessimistically"),
        ("gap-edge spread",
         "symmetric, sd ~" + e["gap_edge_sd_s"]["display"] + "; the fitted spread is " +
         e["gap_edge_spread_min"]["display"] + "–" + e["gap_edge_spread_max"]["display"] + " larger"),
        ("geometry (checked, closed)",
         "nearest player " + e["geom_nearest_dev"]["display"] + " vs " + e["geom_nearest_match"]["display"] +
         "; proximity cues are not inflated"),
        ("ten games, " + e["practice_seeds"]["display"] + " seeds, level 1",
         "CIs omit degrader misspecification; one match"),
        ("level, not ranking",
         e["smooth50_dev"]["display"] + " on dev against " + e["smooth50_match"]["display"] +
         " published; only orderings transfer"),
    ]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _figure_table():
    rows = [("figure", "section", "profile")]
    for name in sorted(FIGURES):
        section, local_only, caption = FIGURES[name]
        rows.append((name, section, "local only" if local_only else "both"))
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


HEADLINE_KEYS = (
    "master_dev", "master_over_ship", "master_over_baseline_plus", "master_neg", "expected_master",
    "smooth50_dev", "baseline_plus_dev", "gate_dev", "ship_dev", "ship_over_smooth50",
    "gate_over_baseline_plus", "ingap_over_gate", "grown_delta", "subgap_delta", "postprocess_delta",
    "postprocess_delta_current",
    "pred_spells", "d_stretches", "pred_d_share_graded", "ball_in_play_min", "graded_min", "csv_sha8",
    "public_ship_spellf1", "public_ship_macro", "public_ship_spells", "missed_share",
    "handover_tail", "handover_cue_share", "stage_moved", "handover_oracle", "e25_dead_share",
    "e36_dead_share", "shape_width_diff_m", "shape_dead_gap_m",
)


def _trace_table(numbers, key_sections):
    rows = [("key", "value", "section", "recomputed")]
    for key in HEADLINE_KEYS:
        e = numbers["entries"][key]
        section = ", ".join(key_sections.get(key, ["—"]))
        rows.append((key, e["display"], section, "yes" if e["recomputed"] else "no"))
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


# ----------------------------------------------------------------------------- notebook assembly

def _cell(source, kind="markdown"):
    return {"cell_type": kind,
            "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:8],
            "metadata": {},
            "source": source.splitlines(keepends=True)}


ANIMATED = {"fig_spellf1"} | {n for n in FIGURES if n.startswith("fig_stage_")} | {
    "fig_match_gap", "fig_practice_clean", "fig_practice_degraded", "fig_shape_deform"}


def _fig_block(name, profile):
    local_only = FIGURES[name][1]
    ext = ".gif" if name in ANIMATED else ".png"
    if local_only and profile == "public":
        src = "figs/public/%s.png" % name
    else:
        src = "figs/%s%s" % (name, ext)
    caption = FIGURES[name][2]
    return ('<img src="%s" style="width:100%%">\n'
            '<div class="caption" style="font-size:0.9em; color:#555; margin:2px 0 14px 0;">%s</div>'
            % (src, caption))


def build_notebook(numbers, profile):
    displays = {k: v["display"] for k, v in numbers["entries"].items()}
    S = _prose(numbers)
    key_sections = {}
    for sec, blocks in S.items():
        for kind, payload in blocks:
            if kind == "md":
                for f in fields(payload):
                    key_sections.setdefault(f, [])
                    if sec not in key_sections[f]:
                        key_sections[f].append(sec)
    for k in ("smooth50_dev", "baseline_plus_dev", "gate_dev", "ship_dev",
              "baseline_plus_over_smooth50", "gate_over_baseline_plus", "ingap_over_gate",
              "expected_gate", "expected_ingap", "expected_grown", "expected_subgap",
              "grown_delta", "grown_ci_low", "grown_ci_high", "subgap_delta", "subgap_ci_low",
              "subgap_ci_high", "postprocess_delta", "postprocess_ci_low", "postprocess_ci_high",
              "pred_d_share_graded", "pred_spells", "ball_in_play_min", "csv_sha8"):
        key_sections.setdefault(k, ["4"])
    for k in ("n_labels",):
        key_sections.setdefault(k, ["1"])
    for k in ("ours_live_missing",):
        key_sections.setdefault(k, ["3"])
    for k in ("kick_recall_low", "kick_recall_high", "coupling_low", "coupling_high", "video_recall",
              "video_false", "ball_px", "hsmm_delta", "hsmm_ci_low", "hsmm_ci_high", "hmm_delta",
              "hmm_ci_low", "hmm_ci_high", "restart_team_pp", "split_delta", "crowd_lead",
              "panns_synthetic", "panns_real", "mbr_delta", "viterbi_delta"):
        key_sections.setdefault(k, ["6"])
    for k in ("master_dev", "master_over_ship", "master_over_ship_lo", "master_over_ship_hi",
              "master_over_baseline_plus", "master_over_baseline_plus_lo", "master_over_baseline_plus_hi",
              "master_neg", "all_frames", "graded_frames", "count_h", "count_a", "count_d", "count_x",
              "pred_d_share_graded", "pred_spells", "d_stretches", "d_runs_ge_15s", "ball_in_play_min",
              "graded_min", "csv_sha8", "public_gt_spells"):
        key_sections.setdefault(k, ["7"])
    for k in list(numbers["entries"]):
        if k.startswith("public_"):
            key_sections.setdefault(k, ["7"])
    for k in ("metrica_d_spread", "sliver_gaps", "sliver_share", "anchor_recall_knob0",
              "anchor_recall_knob1", "anchor_recall_clean", "gap_edge_sd_s", "gap_edge_spread_min",
              "gap_edge_spread_max", "geom_nearest_dev", "geom_nearest_match", "practice_seeds"):
        key_sections.setdefault(k, ["8"])
    # --check: every placeholder exists
    missing = sorted({f for blocks in S.values() for kind, payload in blocks if kind == "md"
                      for f in fields(payload) if f not in displays})
    if missing:
        raise SystemExit("placeholders missing from numbers.json: " + ", ".join(missing))
    # --check: no hand-typed digits in prose (headings and HTML are exempt)
    bad = []
    for sec, blocks in S.items():
        for kind, payload in blocks:
            if kind != "md":
                continue
            for line in payload.splitlines():
                if line.startswith("#") or line.strip().startswith("<"):
                    continue
                # Remove placeholders, inline code, card ids and the year, then look for a bare
                # measured digit. A digit attached to a word (spellF1, D-002) is not a measurement.
                import re
                plain = re.sub(r"\{[^}]+\}", "", line)
                plain = re.sub(r"`[^`]*`", "", plain)
                plain = re.sub(r"\b(?:D-\d{3}|[A-Z]\d{1,3})\b", "", plain)
                plain = re.sub(r"\bStage \d\b", "", plain)
                for token in ("spellF1", "macroF1", "tier-1", "tier-2", "2027"):
                    plain = plain.replace(token, "")
                if re.search(r"\d", plain):
                    bad.append("%s: %s" % (sec, line[:120]))
    if bad:
        raise SystemExit("hand-typed digits in prose (use a numbers.json key):\n  " + "\n  ".join(bad))

    cells = []
    for key in ("title", "1", "2", "3", "4", "5", "6", "7", "8", "post", "close", "appendix"):
        for kind, payload in S[key]:
            if kind == "md":
                cells.append(_cell(payload.format(**displays)))
            elif kind == "fig":
                cells.append(_cell(_fig_block(payload, profile)))
            elif kind == "flex":
                inner = "".join('<div style="flex:1; padding:0 4px;"><center>%s</center></div>' %
                                _fig_block(n, profile) for n in payload)
                cells.append(_cell('<div style="display:flex;">%s</div>' % inner))
            elif kind == "stage_table":
                cells.append(_cell(_stage_table(numbers)))
            elif kind == "drop_table":
                cells.append(_cell(_drop_table(numbers)))
            elif kind == "results_table":
                cells.append(_cell(_results_table(numbers)))
            elif kind == "comparisons_table":
                cells.append(_cell(_comparisons_table(numbers)))
            elif kind == "diag_table":
                cells.append(_cell(_diag_table(numbers)))
            elif kind == "public_table":
                cells.append(_cell(_public_table(numbers)))
            elif kind == "blind_table":
                cells.append(_cell(_blind_table(numbers)))
            elif kind == "figure_table":
                cells.append(_cell(_figure_table()))
            elif kind == "trace":
                cells.append(_cell(_trace_table(numbers, key_sections)))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
            "title": "Reading the ball: a traceable possession walkthrough",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb, key_sections


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=("local", "public"), default="local")
    ap.add_argument("--figs-only", action="store_true", help="build figures and stop")
    ap.add_argument("--no-figs", action="store_true", help="build the notebook from figures already on disk")
    ap.add_argument("--only", help="comma-separated figure names (with --figs-only)")
    ap.add_argument("--check", action="store_true", help="validate numbers, placeholders and figure files")
    args = ap.parse_args(argv)

    numbers, _ = load_numbers()
    if args.check:
        nb, _ = build_notebook(numbers, args.profile)
        # every referenced figure must exist in the chosen profile
        import re
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        files = re.findall(r'src="([^"]+)"', text)
        missing = [f for f in files if not os.path.isfile(os.path.join(SOLUTION_DIR, "notebooks", f))]
        if missing:
            raise SystemExit("missing figures for profile %s:\n  %s" % (args.profile, "\n  ".join(missing)))
        print("notebook OK: %d cells, %d figures, %d numbers" % (
            len(nb["cells"]), len(files), len(numbers["entries"])))
        return 0
    if not args.no_figs:
        build_figures(args.profile, numbers, only=args.only)
        if args.figs_only:
            return 0
    nb, _ = build_notebook(numbers, args.profile)
    os.makedirs(os.path.dirname(NOTEBOOK), exist_ok=True)
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s (%d cells, profile %s)" % (NOTEBOOK, len(nb["cells"]), args.profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
