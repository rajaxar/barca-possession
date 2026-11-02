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
    axt.tick_params(labelsize=8)
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
    axp.tick_params(labelsize=8)
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
    for i, v in enumerate(scene["ious"]):
        axi.scatter([v], [0.5], s=110, color=colors[i % len(colors)], edgecolors="white", linewidths=1.2, zorder=4)
        axi.text(v, 0.22 - 0.10 * (i % 2), "%.2f" % v, ha="center", fontsize=9, color="#333333")
    if not scene["ious"]:
        axi.text(0.5, 0.5, "no predicted H spell", ha="center", va="center", fontsize=9, color="#888888")
    axi.set_yticks([])
    axi.set_xlabel("IoU of one predicted fragment against the true ten-second spell", fontsize=8.5)
    axi.tick_params(labelsize=8)
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
    axf.set_yticks([])
    axf.tick_params(labelsize=8)
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
    names = ["player-motion classifier\n(E25)", "broadcast ball track\n(E36)"]
    ax.bar(names, vals, color=["#1a5fb4", "#2e7d32"], width=0.5)
    lo = e["e25_ci_low"]["value"]
    hi = e["e25_ci_high"]["value"]
    ax.errorbar([0], [vals[0]], yerr=[[vals[0] - lo], [hi - vals[0]]], fmt="none",
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
    """Neutral public-profile card; carries the caption in the notebook, not the image."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_facecolor("#f4f4f4")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="#f4f4f4", edgecolor="#bbbbbb", lw=1.2))
    ax.text(0.5, 0.62, "figure withheld in the public profile", ha="center", fontsize=13, weight="bold",
            color="#444444")
    ax.text(0.5, 0.48, "this figure renders real tracking data,\nwhich is local-only (D-002).",
            ha="center", fontsize=10, color="#666666")
    ax.text(0.5, 0.30, "rebuild with  --profile local  to render it", ha="center", fontsize=9,
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
    ("stage_table", None), ("trace", None)."""
    S = {}

    S["title"] = [
        ("md", "# Reading the ball\n"
               "\n"
               "**A traceable walkthrough of our frame-level possession model.**\n"
               "\n"
               "More than a Hack 2027 · Barcelona vs Athletic Club. This notebook is prose and figures only: "
               "the code that produced every number and every clip is in `solution/`, and the appendix at the "
               "bottom goes from a sentence to the exact command that recomputes it."),
    ]

    S["1"] = [
        ("md", "## 1. The problem: a ball that keeps disappearing\n"
               "\n"
               "Watch the clip below. Barcelona is moving the ball in their own half. The players are jogging "
               "into shape, the ball is visible, and then the tracking loses it. Six seconds later the ball is "
               "still gone and the players have stopped. The answer strip under the pitch turns gray: the "
               "stretch is **dead**."),
        ("fig", "fig_match_gap"),
        ("md", "The clip is twenty seconds of our match, rendered from the same tracking file the submission "
               "uses. The gray band is a **ball-missing run**: a run of frames where the tracker has no ball "
               "position at all. The red band is the model's **D** label. On our match the ball is missing for "
               "{missing_share_graded} of the graded frames, and almost all of that missing time is a stoppage. "
               "That single fact is the spine of this notebook."),
        ("md", "The task is to give one of **{n_labels} labels** to every graded frame. **H** means Barcelona has "
               "the ball, **A** means Athletic has it, and **D** means the ball is dead: a throw-in, a corner, a "
               "foul, a goal, an injury, a cooling break. A **possession** is a maximal unbroken run of H or A; "
               "D ends it. We answer at {fps}, for {graded_frames} graded frames out of "
               "{all_frames} in the match, and the answer has to be a coherent sequence, not a per-frame guess."),
        ("fig", "fig_three_labels"),
        ("md", "The obvious first rule is **who is nearest the ball**. It is the organizers' own starter "
               "baseline, and it is a good place to start: with a {nearest_radius_m} radius, a player near the "
               "ball is usually the player who has it. But nearest is not **owner**. The owner is the team that "
               "last touched the ball; it keeps the ball through a pass in the air, a dribble, or a scramble, "
               "until the other team touches it. In the schematic below the defender is the nearest player to "
               "the ball, but the ball is still traveling from the passer to the receiver, so the owner has "
               "not changed."),
        ("fig", "fig_nearest_player"),
        ("md", "This is not a corner case. On clean practice data the nearest-player rule catches {flip_recall_clean} "
               "of the visible changes of owner; on data degraded to our noise it catches {flip_recall_degraded}. "
               "About {unresolvable_share} of owner changes cannot be resolved from geometry at all: the ball "
               "is hidden, the players are in a duel, or the tracking disagrees with the label. A model that "
               "only looks at the nearest player is therefore blind to a large part of the problem, and that "
               "is why the rest of this notebook is about two other clues: **ball missingness** and **team "
               "shape**."),
    ]

    S["2"] = [
        ("md", "## 2. The score: why one extra boundary is expensive\n"
               "\n"
               "Imagine one true possession: Barcelona has the ball for ten seconds, then the whistle goes. "
               "The animation below is synthetic — no match data — and it is scored by the organizer's own "
               "metric, not by a cartoon. It is the single most important figure in this notebook, because "
               "the scoring rule explains almost every design choice we made."),
        ("fig", "fig_spellf1"),
        ("md", "The score is **spellF1**. Possessions are maximal runs of H or A. Each predicted possession is "
               "matched one-to-one to a true possession by **temporal IoU**, which is the length of the shared "
               "frames divided by the length of the union: one means the two intervals are identical, zero "
               "means they do not overlap. The F-score of that matching is averaged over IoU thresholds from "
               "{demo_iou_low} to {demo_iou_high}. A prediction is not credited for finding the right team if "
               "its interval is in the wrong place."),
        ("md", "The animation walks through the same true spell five times. A perfect answer scores "
               "{demo_perfect}. Split the answer in two with a one-second D gap and the score falls to "
               "{demo_split2}: each fragment has an IoU of {demo_iou2} against the true spell, so it is still "
               "inside the averaging band, but only one fragment can be matched and the other is a false "
               "positive. Split it in three and the score is {demo_split3}; each fragment's IoU is "
               "{demo_iou3}, at the edge of the band. Split it in four and the score is {demo_split4}: every "
               "fragment's IoU is {demo_iou4}, below the {demo_iou_low} floor, so nothing matches. Miss the "
               "spell completely and the score is also {demo_missed}."),
        ("fig", "fig_spellf1_curve"),
        ("md", "The asymmetry is the point. A missed possession costs one false negative. A spurious boundary "
               "costs a false positive *and* damages the host possession it cut, because the host now has to "
               "compete with its own fragment for the one-to-one match. This is why we are conservative about "
               "inserting hand-overs, and why a clue that improves frame accuracy can still lose score. On a "
               "full match, differences below about {noise_band} are noise. The public five minutes hold only "
               "{public_gt_spells} true possessions, which is why we never choose a method on them."),
    ]

    S["3"] = [
        ("md", "## 3. The practice set: ten matches, made to look like ours\n"
               "\n"
               "We have the true answers for {public_minutes} public minutes of our match, and the rules forbid "
               "tuning on them. {public_gt_spells} possessions is not enough to choose a threshold, and the "
               "organizers show that tuning on that sample picks the wrong answer even for a one-parameter "
               "baseline. So every choice is made on other professional matches with public answers, after we "
               "make their data look like ours."),
        ("md", "The clip below is one practice passage, before and after **degrading**. On the left, the clean "
               "match: the ball stays visible while play is stopped, so a ball-based rule would see the "
               "stoppage for free. On the right, the same passage after the ball is dropped where our tracker "
               "drops it, the ball path is smoothed, and the gap edges are jittered. The stoppage is still "
               "there in the players; the ball is not."),
        ("flex", ["fig_practice_clean", "fig_practice_degraded"]),
        ("md", "The practice set is **{practice_games} matches**: {practice_idsse} from the German league and "
               "{practice_metrica} from Metrica. A second set of {skillcorner_games} SkillCorner matches is used "
               "only for who-has-the-ball questions, never pooled with the ten. The ten are frozen: same games, "
               "same seeds, same degrader, same cells. Any method we compare sees exactly the same data."),
        ("fig", "fig_missingness"),
        ("md", "The histogram is the reason the practice set had to be repaired. Our match hides the ball for "
               "{missing_share_graded} of graded frames; the degraded practice set hides it for "
               "{practice_missing_share}, close enough. The conditional level did not match at first: the "
               "original degrader dropped the ball at random, including during live play, so "
               "{practice_live_missing_old} of practice live frames had no ball against {ours_live_missing} of "
               "ours. After the fix the practice set is at {practice_live_missing}, and "
               "{practice_long_gap_live} of long gaps are live against {ours_long_gap_live} of ours. The old "
               "gap taught the model that a missing ball is compatible with live play, which is exactly wrong "
               "for our data."),
        ("md", "We found two flaws and fixed both. The first was the **resting ball**: in the practice data a "
               "ball lying still for a restart shook about twenty times more than in a real match, so ball "
               "motion separated dead from live far too well. Fixing the sign dropped the texture-only "
               "separability from a perfect score to {flaw_texture_auc} on the ten games and "
               "{flaw_texture_auc_heldout} on twenty held-out SkillCorner games, still worse than our "
               "{flaw_target_auc} target; ball-motion clues are therefore quoted as upper bounds. The second "
               "was **where the ball hides**. After the fix, Baseline+ moved from {baseline_plus_old} to "
               "{baseline_plus_new}, and the in-gap decoder's gain fell from {ingap_old} to {ingap_new}. Every "
               "number measured before that fix was wrong in the same direction."),
        ("md", "One property cannot be measured on our match: how often a stoppage keeps the ball visible. "
               "Label-free instruments put it between {visible_share_lo} and {visible_share_hi} of stoppages, "
               "so every method is tested across that range. The practice set is also about {practice_easier} "
               "easier than the real match — the organizers' two-second baseline scores {smooth50_dev} there "
               "and {smooth50_match} on the real match — so we trust the ranking of methods, never the level."),
    ]

    S["4"] = [
        ("md", "## 4. The pipeline, one stage at a time\n"
               "\n"
               "Here is a practice passage through the shipped stack. The first three stages use one passage; "
               "the last three use a passage where that stage actually fires, because no single passage "
               "exercises all six. Each stage has one job, one number on the frozen practice set, and one way "
               "to hurt us if it is wrong. The numbers are marginal: each stage is measured on the stack it "
               "actually ships on, not as an additive decomposition."),
        ("md", "**Stage 1 — nearest player, smoothed.** The organizers' two-second baseline: the nearest player "
               "within a radius votes, and the vote is majority-smoothed over fifty frames. It is the floor we "
               "have to beat: {smooth50_dev} spellF1 on the frozen practice set."),
        ("fig", "fig_stage_smooth50"),
        ("md", "**Stage 2 — hysteresis and merge (Baseline+).** A player within {nearest_radius_m} owns the "
               "ball; otherwise the last owner keeps it. Short flickers are held back and short spells are "
               "merged. On the practice set it scores {baseline_plus_dev}, a gain of "
               "{baseline_plus_over_smooth50} over the two-second baseline. That lead does not survive a "
               "half-second shift of the answers, so the honest reading is *about level to slightly ahead*. "
               "If hysteresis is too strong, we absorb a real short turnover; if it is too weak, we emit "
               "flickers, which the score punishes twice."),
        ("fig", "fig_stage_baseline_plus"),
        ("md", "**Stage 3 — the missing-ball gate.** When the ball has been gone for a while and the players "
               "are standing still, the stretch becomes D. This is the largest single gain in the stack: "
               "{gate_over_baseline_plus} on the practice set, with no game negative. On our match the expected "
               "gain is {expected_gate} because our gaps are longer and cleaner than the practice set's. If "
               "the gate is wrong, it turns live possession into a stoppage and cuts a real spell in two, which "
               "is the most expensive error the score can charge."),
        ("fig", "fig_stage_gate"),
        ("md", "**Stage 4 — the in-gap decoder.** Inside a missing-ball gap, team shape says which team is "
               "wider and therefore more likely to have the ball. The decoder is allowed to insert a hand-over "
               "there. On the practice set it adds {ingap_over_gate}; on our match only {eligible_ours} of "
               "graded frames are eligible, so the expected gain is {expected_ingap}. The clip below shows the "
               "mechanism, not a success story: on this passage the decoder inserts an H island where the "
               "answer stays A. That is exactly why its average gain is small and why it runs only after the "
               "gate."),
        ("fig", "fig_stage_ingap"),
        ("md", "**Stage 5 — grown stoppages.** A resting ball is a real stoppage, but it is only a sliver of "
               "one. We extend each resting-ball moment out to the stoppage around it, with a radius and a "
               "thirty-second cap. On the practice set this adds {grown_delta} "
               "[{grown_ci_low}, {grown_ci_high}], expected {expected_grown} on our match. It is only safe on "
               "top of the gate: on raw Baseline+ most of the changed frames land in live play. If the radius "
               "is too wide, we eat the restart."),
        ("fig", "fig_stage_grown"),
        ("md", "**Stage 6 — the sub-gap rule.** Some stoppages sit inside missing-ball gaps that the main gate "
               "left alone. A second, weaker rule fires when the stoppage probability stays above one half for "
               "two seconds inside such a gap. It adds {subgap_delta} [{subgap_ci_low}, {subgap_ci_high}] on "
               "the practice set, expected about {expected_subgap} on our match. If it fires on a live gap, it "
               "creates a D island inside a possession."),
        ("fig", "fig_stage_subgap"),
        ("md", "The stack is not quite the order above: the build applies the gate, then grown stoppages, then "
               "the in-gap decoder, then the sub-gap rule. The marginal numbers are the ones measured for each "
               "module on its shipping base in the build cards, so read them as evidence for each stage, not as a "
               "running sum. The next tier-1 add-on is not wired yet: deleting every H/A spell shorter than "
               "one and a half seconds and absorbing it into D adds {postprocess_delta} "
               "[{postprocess_ci_low}, {postprocess_ci_high}] on the practice set, with an expected "
               "{expected_postprocess} on our match. It is waiting for the build card."),
        ("stage_table", None),
        ("md", "The gate-plus-decoder stack is {ship_over_smooth50} over the two-second baseline on the practice "
               "set. On our "
               "match the frozen shipped file calls {pred_d_share_graded} of graded frames dead, finds "
               "{pred_spells} possessions across {pred_d_runs} D runs, and leaves {ball_in_play_min} of ball "
               "in play. The build is the clean-clone Q26 reproduction, sha `{csv_sha8}`. Those are "
               "label-free descriptions, not scores."),
    ]

    S["5"] = [
        ("md", "## 5. The two clues that carry it\n"
               "\n"
               "In the clip at the top of the notebook the ball is gone for six seconds. Two completely "
               "different instruments were asked the same question — is this missing time a stoppage? — and "
               "they agree. One is a classifier trained on player motion. The other is the broadcast camera's "
               "own ball track, which follows the ball through gaps the tactical tracker misses. Neither uses "
               "the answers."),
        ("fig", "fig_gate_instruments"),
        ("md", "The player-motion classifier flags {e25_dead_share} [{e25_ci_low}, {e25_ci_high}] of our "
               "missing time as dead; the broadcast route puts {e36_dead_share} of it dead. After correcting "
               "for the frames each instrument cannot see, the probability that a missing frame is dead is "
               "between {gate_pd_missing_low} and {gate_pd_missing_high}. Two instruments with different "
               "failure modes landing on the same number is the strongest label-free evidence we have. It is "
               "still not a license to call every gap dead: a long live dropout would cut a possession, so the "
               "gate also requires the players to be still."),
        ("md", "The second clue is **team shape**. The team with the ball spreads out to create space; the "
               "defending team stays compact. The schematic makes the idea; the real clip after it shows it "
               "inside a missing-ball gap."),
        ("fig", "fig_width_schematic"),
        ("fig", "fig_shape_deform"),
        ("md", "A model trained only on team shape picks the team with the ball correctly {shape_correct} of "
               "the time on unseen matches, without seeing the ball. That is why the in-gap decoder can recover "
               "a hand-over the ball never showed. The catch is coverage: only {eligible_ours} of our graded "
               "frames are eligible for the decoder, against {eligible_dev} on the practice set, so the "
               "expected gain is {expected_ingap}. The clue is real; its reach on this match is small. It also "
               "shares a blind spot with the gate — both are weakest at the edge of a stoppage, where the "
               "players are just starting to move — so we do not treat their agreement as independent "
               "confirmation of a boundary."),
    ]

    S["6"] = [
        ("md", "## 6. What we tried and dropped\n"
               "\n"
               "The most tempting idea was to find the kick. A kick is a sudden change in the ball's velocity; "
               "on clean public data the textbook detector finds {kick_recall_low} to {kick_recall_high} of "
               "them. On our ball it fails, because our ball path is smoothed over about a third of a second, "
               "which spreads the kick out. We confirmed the cause by smoothing the practice ball the same way: "
               "the method fails there too."),
        ("md", "That result set the pattern for the rest of the notebook: a clue that works on clean data is "
               "not evidence until it survives our noise. The table below is the honest list, with the number "
               "that killed each idea."),
        ("drop_table", None),
        ("md", "The duration model deserves a sentence of its own because it was the most expensive attempt. "
               "An explicit-duration HSMM over learned emissions scores {hsmm_delta} "
               "[{hsmm_ci_low}, {hsmm_ci_high}] against the shipped stack, negative in nine of ten games. A "
               "learned sticky HMM without the duration prior scores {hmm_delta} "
               "[{hmm_ci_low}, {hmm_ci_high}], which is positive but includes zero and is negative in two "
               "games: a tier-2 lever, not a replacement. The duration prior is what loses; the emission "
               "calibration is the open question."),
        ("md", "Restart-team-by-stoppage-type was falsified as pre-registered. Adding the stoppage type changed "
               "restart-team accuracy by {restart_team_pp} and made spellF1 worse in every variant, because "
               "most long predicted stoppages bridge several real stoppages and some live play: the team "
               "before the block is not the team that last touched the ball. Splitting long blocks looked good "
               "on the old practice data, but after the degrader fix it scores {split_delta}, negative in ten "
               "of ten games. Crowd loudness tracked microphone placement, not the match: its lead was "
               "{crowd_lead}, below our five-point floor. A public audio-event model heard a synthetic whistle "
               "at {panns_synthetic} but this match's real whistle at about {panns_real}; our small hand-built "
               "detector stays."),
    ]

    S["limits"] = [
        ("md", "## Limitations\n"
               "\n"
               "The public sample is {public_minutes} minutes and {public_gt_spells} possessions. We report "
               "the shipped entry's score there as {public_ship_spellf1} spellF1 and {public_ship_macro} "
               "macroF1, against {public_baseline2} and {public_baseline5} for the organizers' two- and "
               "five-second baselines, but it is a direction, not a result: the noise band on a full match is "
               "about {noise_band}, and {public_gt_spells} possessions cannot resolve a small difference. "
               "Nothing in this notebook was selected on that sample."),
        ("md", "The practice set is about {practice_easier} easier than the real match, so its levels are not "
               "our levels. It has no cooling breaks and no video-review stops; Metrica over-calls dead ball "
               "around advantage play; it contains short sliver gaps our match does not; and its resting-ball "
               "recall is much lower than clean data's. The share of stoppages that keep the ball visible on "
               "our match is only pinned to {visible_share_lo}-{visible_share_hi}. The in-gap decoder is "
               "eligible on {eligible_ours} of our graded frames, so its {expected_ingap} expected gain is near "
               "the noise floor."),
        ("md", "The biggest weakness is unchanged: we catch only about one in five real possessions well "
               "enough to count. About {unresolvable_share} of owner changes cannot be resolved from geometry "
               "at all, and our hand-overs are slightly early, with a scatter that costs far more than the "
               "bias. The score is dominated by the possessions we miss, not by the frames we label. The "
               "shipped file calls {pred_d_share_graded} of graded frames dead, which is high against a "
               "football prior; part of that is a real cooling break and part is the residue of present-ball "
               "stoppages we still miss. The short-spell floor is measured but not wired, so the current CSV "
               "does not yet contain its gain."),
        ("md", "Finally, every figure that renders tracking is local-only under D-002. The public profile of "
               "this notebook replaces those figures with placeholders; the prose and the synthetic schematics "
               "are the publishable part. The full evidence, including the numbers this notebook does not "
               "quote, is in `research/` and `solution/docs/WHAT_WE_TRIED.md`."),
    ]

    S["close"] = [
        ("md", "## Code, data and acknowledgements\n"
               "\n"
               "The shipped build is one command in `solution/`: `python -m possession.make_submission`. The "
               "pipeline modules are `baseline_plus.py`, `dgate.py`, `ingap.py`, `growd.py`, `subgap.py` and "
               "`postprocess.py`; the scoring harness is `dev_v1.py`; the clip renderer is `render.py`. This "
               "notebook is built by `scripts/walkthrough_numbers.py` and `scripts/build_walkthrough.py`, and "
               "the numbers manifest is `notebooks/numbers.json`."),
        ("md", "The practice data is public tracking from the German league (DFL), Metrica and SkillCorner, "
               "used under their licenses and never redistributed here. The match data is the organizer's and "
               "stays on this machine. The scoring code is the organizer's `evaluation/possession_metrics.py`, "
               "imported rather than re-implemented."),
        ("md", "Thanks to the organizers for a problem where the honest answer is usually a negative result, "
               "and to the four reference notebooks that showed how to write one down."),
    ]

    S["appendix"] = [
        ("md", "## Appendix: traceability\n"
               "\n"
               "Every measured number in the prose is a key in `notebooks/numbers.json`. The table is generated "
               "from that file, so it cannot drift from the notebook: the *source* is where the value lives, "
               "the *command* is how to see it again, and *recomputed* says whether `--check` rebuilds it from "
               "the live pipeline or reads it from a digest. The section column is the part of the notebook "
               "that uses the key."),
        ("figure_table", None),
        ("trace", None),
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


def _figure_table():
    rows = [("figure", "section", "profile", "built by", "what it shows")]
    builders = {
        "fig_three_labels": "fig_three_labels", "fig_nearest_player": "fig_nearest_player",
        "fig_spellf1": "fig_spellf1", "fig_spellf1_curve": "fig_spellf1_curve",
        "fig_missingness": "fig_missingness", "fig_gate_instruments": "fig_gate_instruments",
        "fig_width_schematic": "fig_width_schematic", "fig_match_gap": "_local_match_gap",
        "fig_practice_clean": "_practice_pair", "fig_practice_degraded": "_practice_pair",
        "fig_shape_deform": "_shape_deform",
    }
    for name in sorted(FIGURES):
        section, local_only, caption = FIGURES[name]
        builder = builders.get(name, "_stage_figs")
        profile = "local only" if local_only else "both"
        command = "`build_walkthrough.py --figs-only --only %s`" % name
        rows.append((name, section, profile, "`%s` / %s" % (builder, command), caption))
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join(["---"] * len(rows[0])) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _trace_table(numbers, key_sections):
    rows = [("key", "value", "section", "source", "command", "recomputed")]
    for key in sorted(numbers["entries"]):
        e = numbers["entries"][key]
        section = ", ".join(key_sections.get(key, [])) or "—"
        rows.append((key, e["display"], section, e["source"], "`" + e["command"].replace("|", "\\|") + "`",
                     "yes" if e["recomputed"] else "no"))
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
    return '<img src="%s" style="width:100%%">' % src


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
              "panns_synthetic", "panns_real"):
        key_sections.setdefault(k, ["6"])
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
    for key in ("title", "1", "2", "3", "4", "5", "6", "limits", "close", "appendix"):
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
