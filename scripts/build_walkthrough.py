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
    ("0", "Start here"),
    ("0b", "What is an in-gap decoder?"),
    ("1", "The problem: a ball that keeps disappearing"),
    ("2", "The score: why one extra boundary is expensive"),
    ("3", "The practice set: ten matches, made to look like ours"),
    ("4", "The pipeline, one stage at a time"),
    ("5", "The two clues that carry it"),
    ("6", "What we tried and dropped"),
]

# figure name -> (section number, local_only, caption)
FIGURES = {
    "fig_gap_ambiguity": ("ingap", False, "One passage: the ball is visible, missing for four seconds, visible again. Calling the whole gap dead splits one possession; carrying the last owner hides a hand-over. Illustration, not match data."),
    "fig_ingap_steps": ("ingap", False, "The decoder in four steps: it works only inside a gap the gate left live, reads the players' shape, scores every frame for the home team, then smooths the sequence with a switching penalty. Illustration, not match data."),
    "fig_ingap_before_after": ("ingap", False, "The same passage under the fixed rule (call the gap dead: two short spells) and with the decoder (one spell). Each split half overlaps the true possession at IoU .33; the continuous spell at 1.00."),
    "fig_ingap_result": ("ingap", False, "The decoder's measured marginal on the practice set: +.0017, inside the .004 band, so level with zero; the gate-only +.0052 is what it was credited when it landed. Chart from measured numbers."),
    "fig_ingap_locator": ("ingap", False, "Where the in-gap decoder sits: after the gate and grown D, before the sub-gap rule and the floor."),
    "fig_pipeline_flow": ("appendix", False, "The pipeline left to right: two jobs (who owns the ball, when it is dead) and a clean-up. Each arrow label is the score lost when that stage is removed from the full build (spellF1, practice set); the labels do not add up to the total."),
    "fig_gain_waterfall": ("appendix", False, "Panel A: the practice-set ladder, with each stage's leave-one-out marginal and the .004 line below which a practice-set difference has no established sign. Panel B: the public five minutes, report only."),
    "fig_spellf1_explainer": ("2", False, "One synthetic timeline, three answers, scored by the organizer's metric: exact, a one-frame gap near the middle of a possession, and a missed short possession. The gap costs more than the miss."),
    "fig_three_labels": ("1", False, "The three labels, and the 25 answers per second they ask for. Illustration, not match data."),
    "fig_match_gap": ("1", True, "Our match: live play, then the tracking loses the ball and the label turns D. Real tracking; local copy only."),
    "fig_nearest_player": ("1", False, "Why the nearest player is not the owner: the ball is in flight, and the nearest player has not touched it. Illustration, not match data."),
    "fig_spellf1": ("2", False, "The centerpiece: a true ten-second possession, a perfect answer, the same answer split, and a missed spell, scored by the organizer's metric. Illustration, not match data."),
    "fig_spellf1_curve": ("2", False, "The same experiment as a curve: score against the number of fragments. Illustration, not match data."),
    "fig_practice_clean": ("3", True, "The practice clip before degrading: the ball stays visible through the stoppage. Real tracking; local copy only."),
    "fig_practice_degraded": ("3", True, "The same clip after degrading: the ball is missing where our tracker loses it. Real tracking; local copy only."),
    "fig_missingness": ("3", False, "Where the ball is missing: our match against the degraded practice set. Aggregates, no match frames."),
    "fig_stage_smooth50": ("4", True, "Stage 1, the organizers' 2-second baseline: nearest player, majority-smoothed. Real tracking; local copy only."),
    "fig_stage_baseline_plus": ("4", True, "Stage 2, Baseline+: hysteresis, merge and a 2.5 m radius. Real tracking; local copy only."),
    "fig_stage_gate": ("4", True, "Stage 3, the missing-ball gate: the ball is gone and the players are still, so the stretch becomes D. Real tracking; local copy only."),
    "fig_stage_ingap": ("4", True, "Stage 4, the in-gap decoder: team shape inserts a hand-over inside a missing-ball gap. Real tracking; local copy only."),
    "fig_stage_grown": ("4", True, "Stage 5, grown stoppages: each resting-ball moment is extended to the stoppage around it. Real tracking; local copy only."),
    "fig_stage_subgap": ("4", True, "Stage 6, the sub-gap rule: a stoppage inside a gap the main gate left alone. Real tracking; local copy only."),
    "fig_gate_instruments": ("5", False, "Two independent instruments agree on how much of the missing time is dead. Aggregates, no match frames."),
    "fig_killed": ("6", False, "Each dropped decoding idea's spellF1 change against the shipped build, on the practice set. Every bar points left: each idea lost. Drawn from measured numbers."),
    "fig_width_schematic": ("5", False, "The team with the ball is wider; the defending team is narrow. Illustration, not match data."),
    "fig_shape_deform": ("5", True, "The shape clue in a real missing-ball gap: the wide team changes as the owner changes. Real tracking; local copy only."),
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
    _legible(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


READING_WIDTH_IN = 8.1   # the 780 px text column at 96 px per inch
MIN_FIG_PT = 8.5         # smallest figure text, in points at reading width


def _legible(fig):
    """Raise any text that would print under MIN_FIG_PT at the page's reading width, by at most 1.15x."""
    from matplotlib.text import Text
    floor = MIN_FIG_PT * fig.get_figwidth() / READING_WIDTH_IN
    for t in fig.findobj(Text):
        size = t.get_fontsize()
        if t.get_text() and size < floor:
            t.set_fontsize(min(size * 1.15, floor))


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
    ax.text(35.0, 27.0, "4.2 m to receiver", color="#1a5fb4", fontsize=9, ha="center")
    ax.annotate("nearest, but has not touched it", xy=(33, 21), xytext=(22, 33),
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
    ax.plot(xs, ys, "o-", color="#c0392b", lw=2.0, ms=8)
    for x, y in zip(xs, ys):
        ax.annotate("%.2f" % y, (x, y), textcoords="offset points", xytext=(10, 8), ha="left", fontsize=11)
    ax.set_xlabel("number of fragments the true spell is split into (IoU of each fragment below)", fontsize=11)
    ax.set_ylabel("spellF1 (0 to 1)", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d\nIoU %.2f" % (x, i) if i == 1.0 else "%d\nIoU %s" % (x, ("%.2f" % i).lstrip("0"))
                        for x, i in zip(xs, ious)], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xlim(0.7, 4.4)
    ax.set_ylim(-0.04, 1.12)
    ax.set_title("score falls fast as one possession is cut into pieces", fontsize=12, loc="left")
    ax.grid(alpha=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
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
    axes[0].yaxis.set_major_formatter(lambda v, _: "%.0f%%" % (100 * v))
    axes[0].set_ylabel("share of graded frames (%)")
    axes[0].set_title("ball missing at all", fontsize=10, loc="left")
    vals2 = [e["ours_live_missing"]["value"], e["practice_live_missing"]["value"]]  # same order as labels
    axes[1].bar(labels, vals2, color=["#1a5fb4", "#7f8c8d"], width=0.55)
    for i, v in enumerate(vals2):
        axes[1].text(i, v + 0.004, "%.1f%%" % (100 * v), ha="center", fontsize=10)
    axes[1].set_ylim(0, 0.07)
    axes[1].yaxis.set_major_formatter(lambda v, _: "%.0f%%" % (100 * v))
    axes[1].set_ylabel("share of live-play frames (%)")
    axes[1].set_title("ball missing during live play", fontsize=10, loc="left")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("the practice set hides the ball about as often as our match does", fontsize=11, x=0.04, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "fig_missingness", profile)


def fig_gate_instruments(numbers, profile):
    plt = _mpl()
    e = numbers["entries"]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    rows = [("player-motion classifier\n(95% interval)", e["e25_dead_share"]["value"],
             e["e25_ci_low"]["value"], e["e25_ci_high"]["value"], "#1a5fb4"),
            ("broadcast ball track\n(range after live-bucket correction)", e["e36_dead_share"]["value"],
             e["e36_dead_low"]["value"], e["e36_dead_high"]["value"], "#2e7d32")]
    for y, (name, v, lo, hi, color) in zip((1, 0), rows):
        ax.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", ms=9, color=color,
                    ecolor="#555555", elinewidth=1.6, capsize=5, zorder=3)
        ax.text(hi + 0.012, y, "%.1f%%  [%.0f%%, %.0f%%]" % (100 * v, 100 * lo, 100 * hi), va="center",
                fontsize=11, color=color, weight="bold")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0.6, 1.0)
    ax.xaxis.set_major_formatter(lambda v, _: "%.0f%%" % (100 * v))
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlabel("share of our ball-missing time judged dead (axis starts at 60%)", fontsize=11)
    ax.set_title("two instruments with different blind spots land in the same place", fontsize=12, loc="left")
    ax.grid(axis="x", alpha=0.2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
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
    ax.set_xlabel("spellF1 change against the shipped stack (practice set)")
    ax.set_xlim(min(vals) - 0.004, 0.001)
    ax.set_title("every dropped idea lost on the practice set", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, "fig_killed", profile)


def _point(entries, key):
    """The point estimate of a range entry (display '+.0208 [+.0169, +.0250]' -> '+.0208')."""
    return entries[key]["display"].split(" [")[0]


def fig_pipeline_flow(numbers, profile):
    """One horizontal flow: the six stages, the decision each makes, and its dev marginal."""
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch
    e = numbers["entries"]
    fig = plt.figure(figsize=(11.2, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 123.5)
    ax.set_ylim(0, 40)
    ax.axis("off")
    fig.text(0.035, 0.940, "One pipeline: decide the owner, decide dead ball, clean the seams",
             fontsize=11.5, ha="left", va="top", color="#1c1f21")

    nodes = [
        ("Tracking\n+ structure", "25 labels/s;\npublished halves", "INPUT", "#5a6368", "#f1f2f2"),
        ("Baseline+", "last owner\npersists; merge", "OWNER", "#1a5fb4", "#e8f0f8"),
        ("D gate", "ball gone,\nplayers still", "DEAD BALL", "#2e7d32", "#eaf3ea"),
        ("Grown D", "resting ball →\nits stoppage", "DEAD BALL", "#2e7d32", "#eaf3ea"),
        ("In-gap\ndecoder", "re-ask the owner\ninside a live gap", "OWNER", "#1a5fb4", "#e8f0f8"),
        ("Sub-gap\nrule", "second gate in\nan unfired gap", "DEAD BALL", "#2e7d32", "#eaf3ea"),
        ("Short-spell\nfloor", "H/A spells too\nshort to count", "CLEAN-UP", "#b26a00", "#fbf0df"),
        ("CSV", "one label\nper frame:\nH / A / D / X", "OUTPUT", "#5a6368", "#f1f2f2"),
    ]
    w, gap, y0, h = 10.6, 5.2, 12.0, 20.0
    xs = [1.4 + i * (w + gap) for i in range(len(nodes))]
    for x, (name, decision, job, color, tint) in zip(xs, nodes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.22,rounding_size=0.55",
                                    facecolor=tint, edgecolor=color, lw=1.4, zorder=2))
        ax.text(x + w / 2, y0 + h + 1.6, job, fontsize=6.6, ha="center", va="bottom",
                color=color, weight="bold", zorder=3)
        ax.text(x + w / 2, y0 + h - 4.2, name, fontsize=8.3, ha="center", va="center",
                color="#1c1f21", weight="bold", linespacing=1.2, zorder=3)
        ax.text(x + w / 2, y0 + h / 2 - 3.6, decision, fontsize=7.2, ha="center", va="center",
                color="#333333", linespacing=1.3, zorder=3)

    arrow_y = y0 + h / 2 - 0.5
    for i in range(1, len(nodes)):
        ax.annotate("", xy=(xs[i], arrow_y), xytext=(xs[i - 1] + w, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.25,
                                    shrinkA=0, shrinkB=0), zorder=1)
    labels = {2: (_point(e, "gate_over_baseline_plus"), "#2e7d32"),
              3: (_point(e, "attrib_grown"), "#2e7d32"),
              4: (_point(e, "ingap_marginal"), "#1a5fb4"),
              5: (_point(e, "attrib_subgap"), "#2e7d32"),
              6: (_point(e, "attrib_floor"), "#b26a00")}
    for i, (text, color) in labels.items():
        cx = xs[i - 1] + w + gap / 2
        ax.plot([cx, cx], [y0 - 0.4, y0 - 2.2], color=color, lw=0.9, zorder=3)
        ax.text(cx, y0 - 2.6, text, fontsize=7.6, ha="center", va="top", color=color, weight="bold",
                zorder=4)

    fig.text(0.035, 0.115,
             "Arrow labels: practice-set score lost when that stage is removed (spellF1); the gate is its difference from "
             "Baseline+ (%s)." % _point(e, "gate_over_baseline_plus"),
             fontsize=8.0, ha="left", va="top", color="#444444")
    fig.text(0.035, 0.078,
             "They are substitutes, not a sum: the three post-ship rules total %s against the measured %s, "
             "and the whole stack is %s over Baseline+."
             % (_point(e, "attrib_sum"), _point(e, "master_over_ship"),
                _point(e, "master_over_baseline_plus")),
             fontsize=8.0, ha="left", va="top", color="#444444")
    fig.text(0.035, 0.041,
             "The in-gap decoder is level with zero on the full stack (%s); the gate is the architecture, "
             "everything after it is repair." % _point(e, "ingap_marginal"),
             fontsize=8.0, ha="left", va="top", color="#444444")
    return _save(fig, "fig_pipeline_flow", profile)


def fig_gain_waterfall(numbers, profile):
    """Panel A: the dev ladder and the nine marginals; Panel B: the public window, report only."""
    plt = _mpl()
    e = numbers["entries"]
    fig = plt.figure(figsize=(10.9, 8.2))
    axA = fig.add_axes([0.075, 0.585, 0.515, 0.33])
    axS = fig.add_axes([0.665, 0.585, 0.315, 0.33])
    axB = fig.add_axes([0.115, 0.125, 0.855, 0.30])

    # ---- Panel A: the ladder, as dots (a level read off a truncated axis is not drawn as a bar)
    levels = [e["baseline_plus_dev"]["value"], e["gate_dev"]["value"],
              e["ship_dev"]["value"], e["master_dev"]["value"]]
    names = ["Baseline+", "D gate", "ship", "frozen"]
    colors = ["#8c9a96", "#1a5fb4", "#6b9bd8", "#123f78"]
    axA.set_ylim(0.31, 0.44)
    axA.set_xlim(-0.5, 3.6)
    axA.plot(range(4), levels, "-", color="#999999", lw=1.2, zorder=2)
    for i in range(4):
        axA.plot([i], [levels[i]], "o", ms=11, color=colors[i], zorder=3)
    axA.text(0, levels[0] - 0.009, e["baseline_plus_dev"]["display"], ha="center", va="top", fontsize=10,
             color="#333333")
    steps = [_point(e, "gate_over_baseline_plus"), _point(e, "ingap_over_gate"),
             _point(e, "master_over_ship")]
    for i, text in enumerate(steps, start=1):
        axA.text(i - 0.5, (levels[i - 1] + levels[i]) / 2 + (0.004 if i == 1 else 0.007), text,
                 ha="center", va="bottom", fontsize=10.5, weight="bold", color="#1d4f8f",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    axA.set_xticks(range(4))
    axA.set_xticklabels(["%s\n%s" % (n, e[k]["display"])
                         for n, k in zip(names, ("baseline_plus_dev", "gate_dev", "ship_dev", "master_dev"))],
                        fontsize=10)
    axA.set_ylabel("practice-set spellF1 (axis starts at .31)", fontsize=10)
    axA.tick_params(axis="y", labelsize=10)
    axA.set_title("Panel A: the practice set, where every choice was made", fontsize=11.5, loc="left", pad=8)
    axA.text(1.3, 0.316, "the ship step is the in-gap\ndecoder on the gate-only stack;\non the full stack it is "
             "%s,\nlevel with zero" % _point(e, "ingap_marginal"),
             fontsize=8.6, color="#555555", ha="left", va="bottom")
    axA.grid(axis="y", alpha=0.18)
    for s_ in ("top", "right"):
        axA.spines[s_].set_visible(False)

    # ---- Panel A inset: leave-one-out marginals and the .004 rule
    rows = [("floor", "attrib_floor", "attrib_floor_lo", "attrib_floor_hi", "#b26a00"),
            ("sub-gap", "attrib_subgap", "attrib_subgap_lo", "attrib_subgap_hi", "#2e7d32"),
            ("grown D", "attrib_grown", "attrib_grown_lo", "attrib_grown_hi", "#2e7d32"),
            ("in-gap", "ingap_marginal", "ingap_marginal_lo", "ingap_marginal_hi", "#1a5fb4")]
    axS.set_xlim(-0.0055, 0.0195)
    axS.set_ylim(-0.6, 3.7)
    band = e["sign_threshold"]["value"]
    axS.axvspan(-band, band, color="#e9e9e9", zorder=0)
    axS.axvline(0, color="#333333", lw=0.9, zorder=1)
    axS.text(band - 0.0002, 3.52, "level with zero: |Δ| < %s" % e["sign_threshold"]["display"],
             fontsize=7.0, color="#6b6b6b", ha="right", va="top")
    for y, (label, key, lo_k, hi_k, color) in zip((3, 2, 1, 0), rows):
        v, lo, hi = e[key]["value"], e[lo_k]["value"], e[hi_k]["value"]
        axS.errorbar([v], [y], xerr=[[v - lo], [hi - v]], fmt="o", ms=6.5, color=color,
                     ecolor="#555555", elinewidth=1.2, capsize=3.5, zorder=3)
        axS.text(hi + 0.0005, y, e[key]["display"], fontsize=7.6, va="center", color=color,
                 weight="bold")
    axS.set_yticks(range(4))
    axS.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.4)
    axS.set_xlabel("spellF1 delta vs the frozen stack", fontsize=7.8)
    axS.set_title("leave-one-out marginals, 95% interval", fontsize=8.8, loc="left",
                  color="#555555", pad=5)
    axS.grid(axis="x", alpha=0.18)
    for s in ("top", "right", "left"):
        axS.spines[s].set_visible(False)

    # ---- Panel B: the public five minutes (report only)
    runs = [("raw", "public_raw_spellf1", "#b8c2c0"),
            ("smooth25", "public_smooth25_spellf1", "#b8c2c0"),
            ("smooth50 (published floor)", "public_smooth50_spellf1", "#8c9a96"),
            ("smooth125", "public_smooth125_spellf1", "#b8c2c0"),
            ("ours (frozen)", "public_ship_spellf1", "#1a5fb4")]
    ys = list(range(len(runs)))
    for y, (label, key, color) in zip(ys, runs):
        v = e[key]["value"]
        axB.barh(y, v, height=0.58, color=color, zorder=3)
        axB.text(v + 0.007, y, e[key]["display"], va="center", fontsize=9,
                 weight="bold" if "ours" in label else "normal", color="#1c1f21")
    axB.set_yticks(ys)
    axB.set_yticklabels([r[0] for r in runs], fontsize=8.8)
    axB.set_xlim(0, 0.56)
    axB.set_xlabel("spellF1 on the five public minutes", fontsize=8.6)
    axB.axvline(e["public_smooth50_spellf1"]["value"], color="#7f8c8d", lw=0.9, ls="--", zorder=1)
    axB.set_title("Panel B: the public five minutes: report only, %s true spells, never used to select"
                  % e["public_gt_spells"]["display"],
                  fontsize=10.2, loc="left", pad=8, color="#7a3b00")
    axB.text(0.555, 0.68, "macroF1 %s\n%s predicted vs %s true possessions"
             % (e["public_ship_macro"]["display"], e["public_ship_spells"]["display"],
                e["public_gt_spells"]["display"]),
             fontsize=8.0, ha="right", va="top", color="#555555")
    axB.grid(axis="x", alpha=0.18)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.text(0.035, 0.040,
             "Panel A is the selection measurement. Panel B is a different measurement on %s true "
             "possessions — a direction check, not a second selection set."
             % e["public_gt_spells"]["display"],
             fontsize=7.6, ha="left", va="bottom", color="#444444")
    fig.text(0.035, 0.014,
             "The parts are substitutes: post-ship marginals total %s against %s measured, and the floor "
             "deletes %s real short possessions to remove about %s false ones."
             % (_point(e, "attrib_sum"), _point(e, "master_over_ship"),
                e["floor_real_short"]["display"], e["floor_false_short"]["display"]),
             fontsize=7.6, ha="left", va="bottom", color="#444444")
    return _save(fig, "fig_gain_waterfall", profile)


def fig_spellf1_explainer(numbers, profile):
    """Three synthetic answers to one timeline, the two definitions, and the IoU ladder."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    e = numbers["entries"]
    fig = plt.figure(figsize=(10.9, 8.6))
    n = 700
    host = (100, 300)
    short = (340, 365)
    third = (380, 480)
    team = {"H": "#1a5fb4", "A": "#c01c28", "D": "#f2f2f2"}

    rows = [
        ("(a) the exact answer", "every possession matched at IoU 1.00",
         [("H", 100, 300), ("A", 340, 365), ("H", 380, 480)],
         [("H", 100, 300), ("A", 340, 365), ("H", 380, 480)], "explain_clean", "#2e7d32"),
        ("(b) one frame of D near the middle", "the host splits; one half is a false positive",
         [("H", 100, 300), ("A", 340, 365), ("H", 380, 480)],
         [("H", 100, 199), ("D", 199, 200), ("H", 200, 300), ("A", 340, 365), ("H", 380, 480)],
         "explain_flicker", "#c0392b"),
        ("(c) the short possession missed", "predicted D; no extra spell, one false negative",
         [("H", 100, 300), ("A", 340, 365), ("H", 380, 480)],
         [("H", 100, 300), ("H", 380, 480)], "explain_missed", "#7a3b00"),
    ]
    bases = (0.775, 0.605, 0.435)
    row_axes = []
    for (title, note, truth, answer, key, color), y in zip(rows, bases):
        ax = fig.add_axes([0.115, y, 0.72, 0.135])
        row_axes.append(ax)
        ax.set_xlim(-95, 700)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(-92, 0.73, "truth", fontsize=7.4, ha="left", va="center", color="#777777")
        ax.text(-92, 0.25, "answer", fontsize=7.4, ha="left", va="center", color="#777777")
        for lab, a, b in truth:
            ax.add_patch(Rectangle((a, 0.56), b - a, 0.30, facecolor="#e4e4e4",
                                   edgecolor="#9a9a9a", lw=0.7, zorder=2))
            ax.text((a + b) / 2, 0.71, lab, fontsize=7.2, ha="center", va="center", color="#444444")
        for lab, a, b in answer:
            ax.add_patch(Rectangle((a, 0.09), b - a, 0.30, facecolor=team[lab],
                                   edgecolor="#888888" if lab != "D" else "none", lw=0.6, zorder=2))
            if lab != "D" and b - a > 30:
                ax.text((a + b) / 2, 0.24, lab, fontsize=7.2, ha="center", va="center",
                        color="white", weight="bold")
        fig.text(0.015, y + 0.135, title, fontsize=9.2, ha="left", va="bottom", weight="bold",
                 color="#1c1f21")
        fig.text(0.015, y + 0.113, note, fontsize=7.3, ha="left", va="bottom", color="#555555")
        fig.text(0.885, y + 0.062, "spellF1", fontsize=7.4, ha="center", va="bottom", color="#777777")
        fig.text(0.885, y + 0.018, e[key]["display"], fontsize=15, ha="center", va="bottom",
                 weight="bold", color=color)

        if key == "explain_flicker":
            ax.annotate("one frame of D", xy=(199, 0.40), xytext=(228, 0.47), fontsize=7.2,
                        color="#c0392b", ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=0.9))
            ax.text(500, 0.71, "the two halves of the host:\nIoU %s and %s"
                    % (e["explain_flicker_iou_lo"]["display"], e["explain_flicker_iou_hi"]["display"]),
                    fontsize=7.2, color="#c0392b", ha="left", va="center")
        if key == "explain_missed":
            ax.annotate("missed", xy=((short[0] + short[1]) / 2, 0.40), xytext=(400, 0.47),
                        fontsize=7.2, color="#7a3b00", ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="-|>", color="#7a3b00", lw=0.9))

    # time axis belongs to the bottom row, so no fourth axes can cover it
    axb = row_axes[-1]
    axb.set_axis_on()
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)
    axb.spines["left"].set_visible(False)
    axb.spines["bottom"].set_visible(True)
    axb.set_yticks([])
    axb.set_xticks([0, 125, 250, 375, 500, 625])
    axb.set_xticklabels(["0", "5", "10", "15", "20", "25"], fontsize=7.6)
    axb.tick_params(axis="x", length=3, pad=2)
    axb.set_xlabel("seconds", fontsize=8.0, labelpad=2)

    # ---- the two definitions, as vector math rendered at build time
    fig.text(0.05, 0.335, "How a pair is scored", fontsize=9.4, weight="bold", color="#1c1f21")
    fig.text(0.05, 0.295,
             r"$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}$" + "\n\n"
             r"$F_1=\frac{2\,\mathrm{TP}}{2\,\mathrm{TP}+\mathrm{FP}+\mathrm{FN}}$" + "\n\n"
             r"$\mathrm{spellF1}=\frac{1}{|T|}\sum_{\tau\in T}F_1(\tau)$",
             fontsize=13, ha="left", va="top", color="#1c1f21", linespacing=1.5)

    # ---- the IoU ladder: why the mean falls below the missed case
    axc = fig.add_axes([0.615, 0.060, 0.355, 0.225])
    th = [0.3, 0.4, 0.5, 0.6, 0.7]
    fl = [e["explain_flicker_f1_%02d" % int(round(t * 10))]["value"] for t in th]
    axc.plot(th, fl, "o-", color="#c0392b", lw=1.6, ms=5, zorder=3)
    axc.axhline(e["explain_missed"]["value"], color="#7a3b00", lw=1.3, ls="--", zorder=2)
    axc.axhline(e["explain_clean"]["value"], color="#2e7d32", lw=1.3, ls="--", zorder=2)
    axc.set_ylim(0.45, 1.08)
    axc.set_xlim(0.26, 0.74)
    axc.set_xticks(th)
    axc.set_xticklabels(["%.1f" % t for t in th], fontsize=7.6)
    axc.set_xlabel("IoU threshold", fontsize=8.0)
    axc.set_ylabel("F1", fontsize=8.0)
    axc.set_title("the same three answers, by threshold", fontsize=8.8, loc="left", pad=5)
    axc.text(0.725, e["explain_clean"]["value"] + 0.012, "exact", fontsize=7.2, ha="right", va="bottom",
             color="#2e7d32")
    axc.text(0.725, e["explain_missed"]["value"] + 0.012, "missed", fontsize=7.2, ha="right", va="bottom",
             color="#7a3b00")
    axc.text(0.660, fl[-1] - 0.02, "flicker", fontsize=7.2, ha="right", va="top", color="#c0392b")
    axc.grid(alpha=0.18)
    for s in ("top", "right"):
        axc.spines[s].set_visible(False)
    return _save(fig, "fig_spellf1_explainer", profile)


def _toy_frame(ax, shade=False):
    """Twenty-two synthetic players on a small pitch: home spread, away compact."""
    _pitch(ax, 60, 40)
    ax.set_anchor("N")
    home = [(6, 10), (10, 28), (14, 16), (18, 32), (22, 8), (26, 22),
            (34, 30), (40, 12), (46, 26), (52, 18), (56, 34)]
    away = [(27, 18), (29, 17), (31, 19), (33, 18), (28, 22), (32, 22),
            (30, 24), (27, 14), (33, 14), (30, 26), (30, 11)]
    if shade:
        from matplotlib.patches import Circle
        ax.add_patch(Circle((30, 20), 8.5, facecolor="#f2f2f2", edgecolor="#bbbbbb",
                            lw=1.0, ls="--", zorder=1, alpha=0.8))
    for x, y in home:
        ax.scatter([x], [y], s=95, c="#1a5fb4", edgecolors="white", linewidths=1.0, zorder=4)
    for x, y in away:
        ax.scatter([x], [y], s=95, c="#c01c28", edgecolors="white", linewidths=1.0, zorder=4)
    ax.scatter([30], [20], s=110, facecolors="none", edgecolors="#222222", linewidths=1.5, zorder=6)
    ax.annotate("ball last seen", xy=(30.8, 20.2), xytext=(36.0, 22.0), fontsize=6.8,
                ha="left", va="center", color="#222222", zorder=7,
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7))
    ax.text(2, 37, "home", fontsize=7.6, color="#1a5fb4", weight="bold")
    ax.text(50, 37, "away", fontsize=7.6, color="#c01c28", weight="bold")


def fig_gap_ambiguity(numbers, profile):
    """One passage: ball visible, missing, visible. Two fixed rules, both wrong sometimes."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    e = numbers["entries"]
    fig = plt.figure(figsize=(10.8, 7.4))
    fig.text(0.02, 0.965, "One passage: the ball disappears for four seconds", fontsize=11.5,
             ha="left", va="top", color="#1c1f21")
    ax = fig.add_axes([0.055, 0.10, 0.54, 0.78])
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(0, 10.2)
    ax.axvspan(4, 8, color="#f2f2f2", zorder=0)
    ax.text(6, 9.75, "the gap: four seconds", ha="center", fontsize=8.2, color="#555555")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("one passage, seconds", fontsize=8.6)

    # (a) the ball
    y = 8.5
    for a, b, face, hatch in ((0, 4, "#cfe0f3", None), (4, 8, "#ececec", "//"), (8, 12, "#cfe0f3", None)):
        ax.add_patch(Rectangle((a, y - 0.28), b - a, 0.56, facecolor=face,
                               edgecolor="#8a9a8e" if hatch else "#1a5fb4", lw=1.0,
                               hatch=hatch, zorder=2))
    for x, text in ((2, "ball visible"), (6, "ball missing"), (10, "ball visible")):
        ax.text(x, y, text, ha="center", va="center", fontsize=7.4,
                color="#333333" if text == "ball missing" else "#1a5fb4", zorder=3)
    ax.text(0, 9.35, "ball", fontsize=8.4, weight="bold", color="#1c1f21")

    # (c1) the whole gap called dead
    y = 5.7
    for a, b, label, face, hatch, color in (
            (0, 4, "H", "#cfe0f3", None, "#1a5fb4"),
            (4, 8, "D", "#e8e8e8", "//", "#666666"),
            (8, 12, "H", "#cfe0f3", None, "#1a5fb4")):
        ax.add_patch(Rectangle((a, y - 0.30), b - a, 0.60, facecolor=face,
                               edgecolor=color, lw=1.0, hatch=hatch, zorder=2))
        ax.text((a + b) / 2, y, label, ha="center", va="center", fontsize=9,
                color="#1a5fb4" if label == "H" else "#444444", weight="bold", zorder=3)
    for a, b, label in ((0, 4, "spell one"), (8, 12, "spell two")):
        ax.plot([a + 0.1, b - 0.1], [y - 0.55, y - 0.55], color="#777777", lw=0.9)
        ax.plot([a + 0.1, a + 0.1], [y - 0.55, y - 0.68], color="#777777", lw=0.9)
        ax.plot([b - 0.1, b - 0.1], [y - 0.55, y - 0.68], color="#777777", lw=0.9)
        ax.text((a + b) / 2, y - 0.95, label, ha="center", va="top", fontsize=7.2, color="#555555")
    ax.text(6, 6.55, "if the whole gap is called dead", fontsize=8.4, weight="bold", color="#1c1f21")
    ax.text(6, 4.55, "one possession becomes two", ha="center", fontsize=7.8, color="#a23b2a")

    # (c2) the last owner carried
    y = 2.9
    ax.add_patch(Rectangle((0, y - 0.30), 12, 0.60, facecolor="#cfe0f3", edgecolor="#1a5fb4", lw=1.0, zorder=2))
    ax.text(6, y, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold", zorder=3)
    ax.text(6, 2.05, "one spell, but blind to a hand-over inside the gap", ha="center",
            fontsize=7.8, color="#a26a00")
    ax.text(0, 3.75, "if the last owner is carried through", fontsize=8.4, weight="bold", color="#1c1f21")

    # (b) the players
    axp = fig.add_axes([0.63, 0.55, 0.33, 0.34])
    _toy_frame(axp, shade=False)
    axp.set_title("at six seconds: the ball is gone, the shape is not", fontsize=9.2, loc="left")
    fig.text(0.63, 0.50,
             "Home is spread across the pitch; the away team is compact around the\n"
             "ball's last position. The shape says who is likely in possession even\n"
             "when the ball itself is not visible.",
             fontsize=8.4, ha="left", va="top", color="#444444", linespacing=1.5)
    fig.text(0.63, 0.24,
             "Neither fixed rule can tell a stoppage from a\ntracker dropout. That is the ambiguity the\ndecoder is built to resolve.",
             fontsize=8.8, ha="left", va="top", color="#1c1f21", linespacing=1.5)
    return _save(fig, "fig_gap_ambiguity", profile)


def fig_ingap_steps(numbers, profile):
    """Four steps: only inside a gap, look at the players, score each frame, smooth and write."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    e = numbers["entries"]
    fig = plt.figure(figsize=(11.0, 5.8))
    fig.text(0.02, 0.965, "The in-gap decoder in four steps", fontsize=11.5,
             ha="left", va="top", color="#1c1f21")

    # 1 — only inside a gap
    ax1 = fig.add_axes([0.025, 0.13, 0.215, 0.70])
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 3)
    ax1.axis("off")
    for a, b, face, hatch, edge in ((0, 4, "#cfe0f3", None, "#1a5fb4"),
                                    (4, 8, "#ececec", "//", "#999999"),
                                    (8, 12, "#cfe0f3", None, "#1a5fb4")):
        ax1.add_patch(Rectangle((a, 1.45), b - a, 0.6, facecolor=face, edgecolor=edge,
                                lw=1.0, hatch=hatch))
    ax1.text(2, 1.75, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold")
    ax1.text(6, 1.75, "no ball", ha="center", va="center", fontsize=7.4, color="#444444")
    ax1.text(10, 1.75, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold")
    ax1.annotate("", xy=(4, 1.15), xytext=(8, 1.15),
                 arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0))
    ax1.text(6, 0.85, "the decoder's\nonly scope", ha="center", va="top", fontsize=7.4,
             color="#333333", linespacing=1.3)
    ax1.text(6, 0.25, "the gate still decides D", ha="center", va="top", fontsize=7.4, color="#555555")
    ax1.set_title("1 · only inside a gap", fontsize=9.2, loc="left", pad=8)

    # 2 — look at the players
    ax2 = fig.add_axes([0.265, 0.13, 0.215, 0.70])
    _toy_frame(ax2, shade=True)
    ax2.set_title("2 · look at the players", fontsize=9.2, loc="left", pad=8)
    fig.text(0.265, 0.30, "the spread team is the one\nmost likely in possession",
             fontsize=7.6, ha="left", va="top", color="#333333", linespacing=1.4)

    # 3 — score each frame
    ax3 = fig.add_axes([0.505, 0.13, 0.215, 0.70])
    x = np.linspace(0, 4, 17)
    y = np.array([.78, .82, .74, .85, .88, .79, .62, .71, .84, .86, .81, .77, .83, .80, .76, .85, .79])
    ax3.plot(x, y, "o-", color="#1a5fb4", ms=3.2, lw=1.1)
    ax3.set_ylim(0, 1.0)
    ax3.set_xlim(0, 4)
    ax3.set_xticks([0, 1, 2, 3, 4])
    ax3.set_yticks([0, .5, 1.0])
    ax3.set_xlabel("seconds into the gap", fontsize=7.4)
    ax3.set_ylabel("P(home has the ball)", fontsize=7.4)
    ax3.tick_params(labelsize=7)
    ax3.grid(alpha=0.2)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.text(2.0, 0.90, "P(H) ≈ %s" % e["demo_ingap_ph"]["display"], ha="center",
             fontsize=8.0, color="#1a5fb4", weight="bold")
    ax3.text(2.0, 0.08, "one dip does not move the label", ha="center", fontsize=7.0, color="#555555")
    ax3.set_title("3 · score each frame", fontsize=9.2, loc="left", pad=8)

    # 4 — smooth, then write
    ax4 = fig.add_axes([0.745, 0.13, 0.235, 0.70])
    ax4.set_xlim(0, 4)
    ax4.set_ylim(0, 3)
    ax4.axis("off")
    for i in range(20):
        color = "#c01c28" if i in (6, 7, 12) else "#1a5fb4"
        ax4.add_patch(Rectangle((i * 0.2, 1.9), 0.2, 0.45, facecolor=color, edgecolor="white", lw=0.5))
    ax4.text(2, 2.60, "the model's view, frame by frame", ha="center", fontsize=7.4, color="#444444")
    ax4.add_patch(Rectangle((0, 1.05), 4, 0.45, facecolor="#1a5fb4", edgecolor="#1a5fb4"))
    ax4.text(2, 1.27, "H", ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax4.text(2, 0.78, "after smoothing: one spell", ha="center", fontsize=7.4, color="#333333")
    ax4.annotate("", xy=(3.7, 1.5), xytext=(3.7, 1.9),
                 arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.0))
    ax4.text(2, 0.17, "a change of team costs;\none noisy frame cannot split the spell",
             ha="center", va="top", fontsize=7.2, color="#555555", linespacing=1.4)
    ax4.set_title("4 · smooth, then write", fontsize=9.2, loc="left", pad=8)
    return _save(fig, "fig_ingap_steps", profile)


def fig_ingap_before_after(numbers, profile):
    """The same passage with the gap called dead and with the decoder; the IoU consequence."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    e = numbers["entries"]
    fig = plt.figure(figsize=(10.8, 6.4))
    fig.text(0.02, 0.965, "The same passage, with and without the decoder", fontsize=11.5,
             ha="left", va="top", color="#1c1f21")
    ax = fig.add_axes([0.07, 0.46, 0.89, 0.44])
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    # without
    y = 2.45
    for a, b, label, face, hatch, edge in (
            (0, 4, "H", "#cfe0f3", None, "#1a5fb4"),
            (4, 8, "D", "#e8e8e8", "//", "#666666"),
            (8, 12, "H", "#cfe0f3", None, "#1a5fb4")):
        ax.add_patch(Rectangle((a, y), b - a, 0.62, facecolor=face, edgecolor=edge,
                               lw=1.0, hatch=hatch))
        ax.text((a + b) / 2, y + 0.31, label, ha="center", va="center", fontsize=9,
                color="#1a5fb4" if label == "H" else "#444444", weight="bold")
    ax.text(0, 3.18, "without the decoder: the fixed rule calls the gap dead",
            fontsize=8.4, weight="bold", color="#1c1f21")
    for a, b, label in ((0, 4, "spell one"), (8, 12, "spell two")):
        ax.plot([a + 0.1, b - 0.1], [y - 0.12, y - 0.12], color="#777777", lw=0.9)
        ax.text((a + b) / 2, y - 0.28, label, ha="center", va="top", fontsize=7.2, color="#555555")
    ax.text(6, 3.44, "one extra boundary → two short spells", ha="center", fontsize=8.0,
            color="#a23b2a")

    # with
    y = 0.75
    ax.add_patch(Rectangle((0, y), 12, 0.62, facecolor="#cfe0f3", edgecolor="#1a5fb4", lw=1.0))
    ax.add_patch(Rectangle((4, y), 4, 0.62, facecolor="#8fb7e6", edgecolor="#1a5fb4", lw=1.0))
    ax.text(2, y + 0.31, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold")
    ax.text(6, y + 0.31, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold")
    ax.text(10, y + 0.31, "H", ha="center", va="center", fontsize=9, color="#1a5fb4", weight="bold")
    ax.text(6, y + 0.78, "changed: D → H", ha="center", fontsize=7.8, color="#1a5fb4")
    ax.text(0, y - 0.22, "with the decoder: decide inside the gap", fontsize=8.6, weight="bold",
            color="#1c1f21", va="top")

    # boundaries
    for x0 in (4, 8):
        ax.plot([x0, x0], [0.45, 3.3], color="#bbbbbb", lw=0.8, ls=":")
    ax.text(6, 1.75, "the only change is inside the gap", ha="center", fontsize=7.4, color="#555555")

    # IoU panel
    axi = fig.add_axes([0.07, 0.075, 0.42, 0.31])
    axi.set_xlim(-0.3, 12.3)
    axi.set_ylim(0, 3.2)
    axi.axis("off")
    axi.add_patch(Rectangle((0, 2.25), 12, 0.45, facecolor="#e4e4e4", edgecolor="#9a9a9a", lw=0.8))
    axi.text(6, 2.47, "true possession (12 s)", ha="center", va="center", fontsize=7.4, color="#444444")
    for a, b in ((0, 4), (8, 12)):
        axi.add_patch(Rectangle((a, 1.25), b - a, 0.45, facecolor="#8fb7e6", edgecolor="#1a5fb4", lw=0.8))
    axi.text(2, 0.98, "IoU %s" % e["demo_split_iou"]["display"], ha="center", fontsize=7.4, color="#1a5fb4")
    axi.text(10, 0.98, "IoU %s" % e["demo_split_iou"]["display"], ha="center", fontsize=7.4, color="#1a5fb4")
    axi.text(-0.2, 1.47, "split", ha="right", va="center", fontsize=7.4, color="#555555")
    axi.add_patch(Rectangle((0, 0.25), 12, 0.45, facecolor="#cfe0f3", edgecolor="#1a5fb4", lw=0.8))
    axi.text(6, 0.47, "IoU %s" % e["demo_continuous_iou"]["display"], ha="center", va="center",
             fontsize=7.4, color="#1a5fb4")
    axi.text(-0.2, 0.47, "one", ha="right", va="center", fontsize=7.4, color="#555555")
    axi.set_title("against the true possession", fontsize=8.4, loc="left", color="#555555", pad=4)

    fig.text(0.54, 0.30,
             "A split does two things at once: it creates a\n"
             "false positive, and it cuts the host possession so\n"
             "neither fragment overlaps enough to match at the\n"
             "higher thresholds. Frames are cheap; spells are not.",
             fontsize=8.4, ha="left", va="top", color="#333333", linespacing=1.5)
    return _save(fig, "fig_ingap_before_after", profile)


def fig_ingap_result(numbers, profile):
    """The honest marginal: +.0017, inside the .004 band, with the gate-only reference."""
    plt = _mpl()
    e = numbers["entries"]
    fig = plt.figure(figsize=(9.6, 4.4))
    ax = fig.add_axes([0.095, 0.40, 0.86, 0.46])
    band = e["sign_threshold"]["value"]
    ax.set_xlim(-0.0048, 0.0078)
    ax.set_ylim(-0.2, 1.75)
    ax.axvspan(-band, band, color="#eeeeee", zorder=0)
    ax.axvline(0, color="#333333", lw=0.9)

    v = e["ingap_marginal"]["value"]
    lo = e["ingap_marginal_lo"]["value"]
    hi = e["ingap_marginal_hi"]["value"]
    ax.errorbar([v], [0.55], xerr=[[v - lo], [hi - v]], fmt="o", ms=8.5, color="#1a5fb4",
                ecolor="#333333", elinewidth=1.4, capsize=4, zorder=3)
    ax.text(v, 0.78, "on the full stack: %s [%s, %s]" % (e["ingap_marginal"]["display"],
                                                         e["ingap_marginal_lo"]["display"],
                                                         e["ingap_marginal_hi"]["display"]),
            ha="center", va="bottom", fontsize=8.4, color="#1a5fb4", weight="bold")

    v2 = e["ingap_over_gate"]["value"]
    lo2 = e["ingap_over_gate_lo"]["value"]
    hi2 = e["ingap_over_gate_hi"]["value"]
    ax.errorbar([v2], [1.25], xerr=[[v2 - lo2], [hi2 - v2]], fmt="o", ms=7.5, color="#8a9a96",
                ecolor="#8a9a96", elinewidth=1.2, capsize=4, zorder=3)
    ax.text(v2, 1.48, "when it landed: %s" % e["ingap_over_gate"]["display"],
            ha="center", va="bottom", fontsize=8.0, color="#5a6368")

    ax.set_yticks([0.55, 1.25])
    ax.set_yticklabels(["full stack", "gate-only"], fontsize=8.2)
    ax.set_xlabel("spellF1 change from removing the decoder", fontsize=8.4)
    ax.text(-band + 0.0001, 1.68, "level with zero: |Δ| < %s" % e["sign_threshold"]["display"],
            fontsize=7.6, color="#6b6b6b", ha="left", va="top")
    ax.grid(axis="x", alpha=0.18)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("the in-gap decoder's marginal", fontsize=9.6, loc="left", pad=6)

    fig.text(0.095, 0.29,
             "adds %s spells of a second and a half or more, removes %s short ones; flicker %s → %s; D share unchanged"
             % (e["ingap_added_spells"]["display"], e["ingap_removed_short"]["display"].replace("−", ""),
                e["ingap_flicker_before"]["display"], e["ingap_flicker_after"]["display"]),
             fontsize=8.2, ha="left", va="top", color="#333333")
    fig.text(0.095, 0.21,
             "%s games negative, %s primary cells non-negative; %s of the drop from the gate-only number "
             "is grown D and the floor."
             % (e["ingap_neg_games"]["display"], e["ingap_cells_pos"]["display"],
                e["ingap_drop_two_thirds"]["display"]),
             fontsize=8.2, ha="left", va="top", color="#333333")
    fig.text(0.095, 0.12,
             "Small; the sign is stable, the size is not established. The concept is worth understanding; "
             "the gain is not the headline.",
             fontsize=8.4, ha="left", va="top", color="#1c1f21", weight="bold")
    return _save(fig, "fig_ingap_result", profile)


def fig_ingap_locator(numbers, profile):
    """A thin locator: the six stages, with the in-gap decoder highlighted."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(10.8, 2.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 10)
    ax.axis("off")
    stages = ["tracking", "Baseline+", "D gate", "grown D", "in-gap\ndecoder", "sub-gap", "floor", "CSV"]
    w, gap, x0 = 10.6, 1.9, 1.2
    for i, name in enumerate(stages):
        x = x0 + i * (w + gap)
        hot = i == 4
        ax.add_patch(Rectangle((x, 3.2), w, 3.4, facecolor="#1a5fb4" if hot else "#f2f4f3",
                               edgecolor="#1a5fb4" if hot else "#bbbbbb", lw=1.3))
        ax.text(x + w / 2, 4.9, name, ha="center", va="center", fontsize=7.4,
                color="white" if hot else "#333333", weight="bold" if hot else "normal")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + w + gap, 4.9), xytext=(x + w, 4.9),
                        arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.9))
    xh = x0 + 4 * (w + gap) + w / 2
    ax.annotate("this section", xy=(xh, 6.7), xytext=(xh, 8.6), ha="center", fontsize=8.2,
                color="#1a5fb4", weight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#1a5fb4", lw=1.0))
    ax.text(50, 1.3, "the decoder sits after the gate and grown D, before the sub-gap rule and the floor",
            ha="center", fontsize=8.0, color="#555555")
    return _save(fig, "fig_ingap_locator", profile)


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
    "fig_gap_ambiguity": fig_gap_ambiguity,
    "fig_ingap_steps": fig_ingap_steps,
    "fig_ingap_before_after": fig_ingap_before_after,
    "fig_ingap_result": fig_ingap_result,
    "fig_ingap_locator": fig_ingap_locator,
    "fig_pipeline_flow": fig_pipeline_flow,
    "fig_gain_waterfall": fig_gain_waterfall,
    "fig_spellf1_explainer": fig_spellf1_explainer,
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
               "More than a Hack 2027 · Barcelona vs Athletic Club. This page explains, in plain football terms, how "
               "we label every frame of one match with who has the ball. It needs no machine-learning background. "
               "The code behind every number is in `solution/`, and the appendix maps each number to its source. "
               "Our method is frozen. A late section re-measures some small parts of it and says how far to trust them.\n"
               "\n"
               "**How to read it.** The box below is the five-minute version. Then comes the full story, from "
               "the problem to the limits. The glossary and the traceability appendix are at the end. Some figures are "
               "clips that play only in our local copy; this public copy shows a labelled placeholder instead."),
    ]

    S["start"] = [
        ("md", "| **Start here** | the whole story, in five lines |\n"
               "|---|---|\n"
               "| **The task** | One label per frame: **H** Barcelona, **A** Athletic, **D** dead ball (a "
               "stoppage), **X** not graded. |\n"
               "| **The score** | **spellF1**, the ranking score, matches each true possession to a predicted one "
               "by time overlap (**IoU**: shared time over total time). It averages the F-score over five "
               "overlap levels, {demo_iou_low} to {demo_iou_high}. A boundary that should not exist costs twice: "
               "a false positive, and a true possession cut in two. |\n"
               "| **What we built** | **Baseline+**, our first stage: the nearest player has the ball, or else the "
               "last owner. A **gate** that calls a long stretch with no tracked ball and still players a "
               "stoppage. An **in-gap decoder** that reads team shape to pick the owner in a stretch the gate "
               "leaves live. Three later rules clean up (see the pipeline section). |\n"
               "| **How much better** | On ten practice matches the gate adds {gate_over_baseline_plus} "
               "spellF1 over Baseline+, and the full stack {master_over_baseline_plus}. Baseline+ is our own "
               "first stage, not the organizers' reference (`smooth50`): the gate-plus-decoder stack beats "
               "that reference by {ship_over_smooth50}, and the final build was never paired with it. The decoder "
               "alone adds {ingap_marginal}, **level with zero** under our {sign_threshold} rule. |\n"
               "| **The main limit** | The practice matches are easier than ours. The decoder can rewrite only "
               "{eligible_ours} of our graded frames. The five public minutes with answers hold {public_gt_spells} "
               "true possessions and are **report-only**: reported, never used to choose anything. |"),
    ]

    S["ingap"] = [
        ("md", "## 4. What is an in-gap decoder?\n"
               "\n"
               "A **possession** is a run of frames in which one team has the ball; a dead ball ends it. "
               "The tracking loses the ball on {missing_share_graded} of our graded frames, usually in "
               "blocks of a few seconds. That is the central difficulty: for those frames the labels "
               "must say either *play continues, this team has it* or *the game is stopped*, and the "
               "ball itself cannot tell us which."),
        ("md", "Take one invented passage, twelve seconds long. The ball is visible for four "
               "seconds, the tracker loses it for four, and it comes back. The players never stop "
               "moving. The gap could be a stoppage — a throw-in, an injury, a goal kick — or a tracking "
               "dropout while play continues. Two fixed rules both fail: call the whole gap dead and "
               "one possession becomes two; carry the last owner and the labels are wrong for the rest "
               "of the gap whenever the ball changed hands."),
        ("fig", "fig_gap_ambiguity"),
        ("md", "The first error is the expensive one. A spurious dead stretch creates a boundary, so the "
               "score sees two short possessions where the truth has one. The second error costs team "
               "accuracy inside the gap but leaves the spell count alone. The gate is built for the first "
               "problem; the decoder is built for the second. The gate decides whether the gap is dead; "
               "the decoder decides who has the ball when it is not."),
        ("md", "The **in-gap decoder** resolves the gap from the players, not the ball. It runs only "
               "inside a ball-missing stretch of at least two seconds, and only on frames the gate left "
               "live — the gate still owns the dead-ball call. Inside the gap it asks which team looks "
               "like it has the ball: the team that is spread out, with players ahead of the ball's last "
               "position, is usually the one in possession; the defending team is compact around the "
               "ball. It scores every frame for the home team, then smooths the sequence with a penalty "
               "for changing team, so one noisy frame cannot create a boundary."),
        ("fig", "fig_ingap_steps"),
        ("md", "In the toy frame the home team is spread across the pitch and the away team is compact "
               "around the ball's last position, so the ball-free probability of home possession, "
               "**P(H)**, sits near {demo_ingap_ph} across the gap. The switching penalty is what turns "
               "that noisy curve into one H label: without it, a single dip would write A and cut the "
               "spell."),
        ("md", "The same passage under the fixed rule and under the decoder. If the gap is called dead, the "
               "two four-second spells each cover only a third of the true twelve-second possession: IoU "
               "{demo_split_iou}, so at the lowest threshold one matches and the other is a false "
               "positive, and at the middle threshold and above neither matches. With the decoder the "
               "gap is H, one spell covers the possession at IoU {demo_continuous_iou}."),
        ("fig", "fig_ingap_before_after"),
        ("md", "That is the argument for the stage: a handful of frames can decide whether a "
               "twelve-second possession is found or cut in two. It is also why the measured gain is "
               "small — the decoder can help only where a gap is both live and long enough, and every "
               "boundary it writes is a chance to be wrong."),
        ("md", "**Where it sits.** The pipeline does two jobs and a clean-up. It decides the owner while the ball "
               "is visible, and dead ball while it is not. Then it removes the seams. The decoder re-opens "
               "the owner question inside a live gap. The gate, grown D (extend a visible resting ball to "
               "its stoppage) and the sub-gap rule (a second, weaker gate) decide dead ball. The "
               "short-spell floor (delete spells under a second and a half) cleans up."),
        ("fig", "fig_ingap_locator"),
        ("md", "**What it buys.** On the ten practice games, removing the decoder from the final "
               "stack changes spellF1 by {ingap_marginal} [{ingap_marginal_lo}, {ingap_marginal_hi}], "
               "with {ingap_neg_games} games negative and {ingap_cells_pos} primary cells non-negative. "
               "Under our {sign_threshold} rule that is **level with zero**: the sign is stable, the size "
               "is not established. It was credited {ingap_over_gate} when it landed on the gate-only "
               "stack (the pipeline before the later rules). On the full stack, grown D and the "
               "short-spell floor now do much of that work — {ingap_drop_two_thirds} of the drop is those "
               "two stages, the rest is the sub-gap rule. Mechanically it adds {ingap_added_spells} spells "
               "of a second and a half or more and removes {ingap_removed_short} shorter ones, so flicker "
               "falls from {ingap_flicker_before} to {ingap_flicker_after}, with the dead share unchanged."),
        ("fig", "fig_ingap_result"),
        ("md", "In short, the idea is worth understanding, but the gain is not our headline. "
               "The decoder is kept because its sign is positive in every primary cell and the ball-free "
               "evidence is what makes hidden-ball stretches legible at all; on our match only "
               "{eligible_ours} of graded frames are eligible, and with the gate in front it rewrites "
               "only a small fraction of them."),
    ]

    S["1"] = [
        ("md", "## 1. The problem: a ball that keeps disappearing\n"
               "\n"
               "Picture a passage from our match. Barcelona moves the ball in their own half, the players jog into shape, and the "
               "tracking loses the ball. Six seconds later it is still gone and the players have stopped. The "
               "answer strip under the pitch turns gray: the stretch is **dead**. The figure below is that passage as a clip."),
        ("fig", "fig_match_gap"),
        ("md", "The clip covers twenty seconds of our match, rendered from the submission's own tracking file. The gray band is a "
               "**ball-missing run**, frames with no ball position at all; the red band is the model's **D**. The "
               "ball is missing for {missing_share_graded} of graded frames, and almost all of that time is a "
               "stoppage. Everything else on this page follows from that fact."),
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
               "divided by the union. The F-score is averaged over five overlap levels, {demo_iou_low} to {demo_iou_high}. "
               "The right team is not enough if the interval is in the wrong place."),
        ("md", "A perfect answer scores {demo_perfect}. Split it in two with a one-second D gap: {demo_split2}, with "
               "each fragment at IoU {demo_iou2}. That is inside the range, but only one can match and the other is a false "
               "positive. Split in three: {demo_split3}, IoU {demo_iou3}, at the edge of the range. Split in four: "
               "{demo_split4}, every fragment at IoU {demo_iou4}, below the lowest level, {demo_iou_low}, so nothing "
               "matches. Miss it completely: also {demo_missed}."),
        ("fig", "fig_spellf1_curve"),
        ("md", "A second synthetic timeline makes the asymmetry concrete: the exact answer, a "
               "one-frame gap near the middle of a possession, and a short possession predicted dead. "
               "The gap scores below the miss, because it pays twice."),
        ("fig", "fig_spellf1_explainer"),
        ("md", "The asymmetry is the point. A missed possession costs one false negative. A boundary that should not "
               "exist costs a false positive *and* cuts a true possession in two, so that possession must now compete "
               "with its own fragment. Hence our caution about hand-overs, and why a clue can raise frame accuracy and still "
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
        ("md", "The next figures show one practice passage, clean and **degraded**. Clean, the ball stays visible through the "
               "stoppage, so a ball-based rule sees it for free. Degraded, the ball is dropped where our tracker "
               "drops it, the path is smoothed and the gap edges jittered: the stoppage is still there in the "
               "players, the ball is not."),
        ("flex", ["fig_practice_clean", "fig_practice_degraded"]),
        ("md", "The practice set is **{practice_games} matches** from two public providers: {practice_idsse} "
               "German-league (DFL) and {practice_metrica} Metrica. A third provider, SkillCorner, gives "
               "{skillcorner_games} more matches. They are used only for who-has-the-ball questions and never pooled. "
               "The ten are frozen: same games, same random seeds, same degrader, same six configurations."),
        ("fig", "fig_missingness"),
        ("md", "The set had to be repaired. Our match hides the ball for {missing_share_graded} of graded frames, "
               "the degraded practice set for {practice_missing_share}, close enough. The share inside live play did "
               "not match: the old degrader dropped the ball at random, including in live play, so "
               "{practice_live_missing_old} of practice live frames had no ball against {ours_live_missing} of "
               "ours. After the fix it is {practice_live_missing}, and {practice_long_gap_live} of long gaps are "
               "live against {ours_long_gap_live} of ours. The old gap taught the model the opposite of the truth."),
        ("md", "Two flaws, both fixed. The first: the **resting ball** shook about twenty times too much, so ball motion "
               "separated dead from live far too well. We measure that separation as an **AUC**, where lower is harder "
               "and a perfect split is one. Fixing the sign took ball-motion-only separation from perfect "
               "to {flaw_texture_auc}, and {flaw_texture_auc_heldout} on twenty held-out SkillCorner games. Both are "
               "still above our {flaw_target_auc} target, so ball-motion clues remain optimistic upper bounds. "
               "The second flaw was **where the ball hides**: after the fix Baseline+ moved from {baseline_plus_old} "
               "to {baseline_plus_new} and the in-gap gain fell from {ingap_old} to {ingap_new}. The set is also "
               "about {practice_easier} easier than the match: {smooth50_dev} against {smooth50_match} for the "
               "two-second baseline. So we trust the ranking, never the level."),
    ]

    S["4"] = [
        ("md", "## 5. The pipeline, one stage at a time\n"
               "\n"
               "The section above explains the in-gap decoder. This one takes the six stages in "
               "the order we met them and shows a passage for each. No single passage shows all six, "
               "because they need different conditions. The gate needs a long gap. Grown D needs a visible "
               "resting ball. The decoder needs a gap the gate left live. The sub-gap rule needs one the gate "
               "left alone. So the clips come from four passages. Stages one to three share one; each later clip "
               "shows where that stage changes a label."),
        ("md", "**Stage 1 — nearest player, smoothed.** The organizers' two-second baseline: nearest player within "
               "a radius, majority-smoothed over fifty frames. The reference to beat: {smooth50_dev}."),
        ("fig", "fig_stage_smooth50"),
        ("md", "**Stage 2 — hysteresis and merge.** *Hysteresis* means a player must clearly take the ball before the owner changes. A player within {nearest_radius_m} owns the ball, otherwise the last owner keeps it; short spells are *merged* into their neighbors. It scores {baseline_plus_dev}, {baseline_plus_over_smooth50} over that reference, "
               "but that lead does not survive a half-second shift, so read it as *about level to slightly ahead*. "
               "Too much hysteresis absorbs a real hand-over; too little emits flickers, which cost twice."),
        ("fig", "fig_stage_baseline_plus"),
        ("md", "**Stage 3 — the missing-ball gate.** Ball gone for a while and players still: the stretch becomes "
               "D. The largest gain in the stack, {gate_over_baseline_plus} over Baseline+, no game negative; expected "
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
        ("md", "The code runs them in a different order from the list above: gate, grown D, in-gap, sub-gap, then the "
               "short-spell floor. Grown D must be excluded from the re-decode, and the sub-gap rule must see the "
               "gate's output. The floor is not a seventh stage but a last clean-up, and it **is** in the final build. "
               "Deleting H/A spells under one and a half seconds adds {postprocess_delta} "
               "[{postprocess_ci_low}, {postprocess_ci_high}] on the gate-plus-decoder stack and "
               "{postprocess_delta_current} on the final labels. It is a trade: real short possessions are deleted "
               "to remove more false ones.\n"
               "\n"
               "The gains in the table are measured one stage at a time, so they are not a running sum. "
               "The practice column is blank for grown D and the sub-gap rule because neither has a standalone "
               "score; each was measured only as a paired difference."),
        ("stage_table", None),
    ]

    S["5"] = [
        ("md", "## 6. The two clues that carry it\n"
               "\n"
               "In the clip from the first section the ball is gone for six seconds. Two independent checks were asked whether "
               "that time is a stoppage: a classifier trained on player motion, and the broadcast camera's own "
               "ball track, which follows the ball through gaps our tracking misses. Neither uses the "
               "answers."),
        ("fig", "fig_gate_instruments"),
        ("md", "The player-motion classifier flags {e25_dead_share} [{e25_ci_low}, {e25_ci_high}] of our missing "
               "time as dead. The broadcast camera's track puts {e36_dead_share} of it dead. A further slice, "
               "{e36_live_low}–{e36_live_high} of the time, is confirmed live, which carries the dead share to "
               "about {e36_dead_low}–{e36_dead_high}. After correcting for what each check cannot see, the "
               "probability that a missing frame is dead is {gate_pd_missing_low}–{gate_pd_missing_high}. Two checks "
               "with different failure modes agreeing is the strongest label-free evidence we have. It is not a "
               "license to call every gap dead, so the gate also requires still players."),
        ("md", "The second clue is **team shape**. The team with the ball spreads out; the defender stays compact. "
               "On live practice frames the possession team's width is {shape_width_own_m} against "
               "{shape_width_def_m}, a difference of about {shape_width_diff_m} that separates them with an **AUC** (a score where a coin flip is one half and a perfect separator is one) of about "
               "{shape_auc_idsse}–{shape_auc_metrica}. The catch is where it applies. Inside a dead gap the same "
               "difference collapses to {shape_dead_gap_m}, because both teams walk into position. So it works only "
               "where the gap is live. Only {eligible_ours} of our frames are eligible, against {eligible_dev} "
               "on the practice set. Hence {expected_ingap}, not the large number the decoder scored alone."),
        ("fig", "fig_width_schematic"),
        ("fig", "fig_shape_deform"),
    ]

    S["6"] = [
        ("md", "## 7. What we tried and dropped\n"
               "\n"
               "The most tempting idea was to find the kick from how the ball moves. On clean public data the textbook "
               "detector finds {kick_recall_low}–{kick_recall_high} of them. On our ball it fails, because the path is "
               "smoothed over about a third of a second. Smoothing the practice ball the same way reproduces the failure."),
        ("md", "That set the pattern: a clue that works on clean data is not evidence until it survives our noise. "
               "The table gives the number that killed each idea. The chart compares four decoding ideas with our "
               "full build. Viterbi and MBR are two standard ways to turn per-frame probabilities into one label "
               "sequence (see the glossary)."),
        ("md", "The duration model was the expensive attempt. An explicit-duration **HSMM** (a sequence model "
               "that also learns how long possessions last) scores {hsmm_delta} [{hsmm_ci_low}, {hsmm_ci_high}], "
               "negative in nine of ten games. A sticky **HMM** (the same model without the duration prior) scores "
               "{hmm_delta} [{hmm_ci_low}, {hmm_ci_high}]. That is positive but includes zero, so not yet "
               "trustworthy. The duration prior is what loses."),
        ("md", "Four smaller ideas also failed. Guessing the restarting team from the type of stoppage changed "
               "accuracy by {restart_team_pp}. Splitting long dead blocks scores {split_delta}. Crowd loudness led by "
               "{crowd_lead}, below our five-point minimum. A public sound-event model heard a synthetic whistle at "
               "{panns_synthetic} but the real one at {panns_real}."),
        ("drop_table", None),
        ("fig", "fig_killed"),
    ]

    S["7"] = [
        ("md", "## 8. Results on the final build\n"
               "\n"
               "The method is frozen. These are the practice-set results: ten practice games, six primary "
               "cells, game-cluster intervals. The final-build row is **one run of the whole stack**, not a sum of the "
               "component deltas above."),
        ("results_table", None),
        ("comparisons_table", None),
        ("md",  "Against the organizers' reference, the gate-plus-decoder stack is {ship_over_smooth50}. The final build and that "
               "reference were never measured as a pair, so we do not quote the arithmetic difference."),
        ("md", "The gain is smaller on our match. We scale each rule by the share of our match it can touch. On that "
               "basis the final build should beat the gate-plus-decoder stack by {expected_master}, not the "
               "{master_over_ship} measured on the practice set. The practice set is about {practice_easier} easier, "
               "and the newer rules reach less of our match. Everything that moved moved for one reason: the D "
               "layer. The gate supplies most of the {master_over_baseline_plus} over Baseline+. The grown, sub-gap "
               "and floor rules add a few thousandths each."),
        ("md", "What the file itself looks like, counted row by row from the frozen CSV:"),
        ("diag_table", None),
        ("md", "{pred_spells} spells, {ball_in_play_min} of play and {pred_d_share_graded} dead are in the right "
               "football neighborhood; they do **not** say the labels are right. Our match is dead-heavier than "
               "the football prior because of the cooling break and tracker-lost visible balls. These are "
               "label-free descriptions, not scores."),
        ("md", "The public window, one run, reported and never a selector:"),
        ("public_table", None),
        ("md", "Sorted by spellF1 we are first on both metrics, but seventeen true possessions is inside noise and "
               "the prediction has eighteen. Nothing was chosen here. Do not compare this table with the practice-set "
               "numbers above: different matches, different difficulty, different base.\n"
               "\n"
               "There is no defensible absolute full-match score. Every interval above is a paired difference, the "
               "practice set is easier than the match, and the public window is only a direction check."),
    ]

    S["8"] = [
        ("md", "## 9. Limits, and what would fix them\n"
               "\n"
               "The practice set has eight blind spots, and they bound what carries over to our match. Read the right-hand "
               "column as the effect on the numbers:"),
        ("blind_table", None),
        ("md", "After the table, the biggest weakness is recall, not D. On the practice set {missed_share} of real possessions are "
               "missed, so about one in five is caught. That figure is from the gate-plus-decoder stack on practice data, "
               "not a match measurement, which would need labels we do not have. Hand-overs are most of the loss, "
               "and much of it has no tracked ball."),
        ("md", "Hand-over scatter is **evidence-limited, not rule-limited**. {handover_tail} "
               "[{handover_tail_low}, {handover_tail_high}] of matched hand-overs are over a second off "
               "({handover_early} early, {handover_late} late), and {handover_cue_share} of that tail is the "
               "{nearest_radius_m} nearest-player cue itself. A duel has no timing information, and on a pass the "
               "receiver is nearest long before the touch. Only {stage_moved} of tail flips are moved by half a second "
               "or more by any stage after Baseline+. So no rule applied after the fact comes near the {handover_oracle} "
               "that perfect hand-over timing would add (the **oracle**). The way forward is a detector of the moment "
               "a player touches the ball, accurate to about {contact_s}. On this tracking the ball is smoothed over "
               "several frames, which puts that out of reach."),
        ("md", "A different data source would help in three ways. Tracking that records each touch would attack the "
               "hand-over tail and reopen the shelved idea of anchoring hand-overs to detected touches. Practice "
               "matches with drinks breaks and video-review stops would test the learned methods where they gain, "
               "since their advantage is confined to dead-frame motion that the ten games lack. A GPU ball detector "
               "would add ball coverage but not touch timing, and nothing may depend on it, since we assume no "
               "graphics card.\n"
               "\n"
               "We would not retry rebuilding touches from ball motion, explicit-duration priors, a learned decoder "
               "in place of the gate, splitting long D blocks, or guessing the restarting team from the type of "
               "stoppage. Each has a decisive negative above."),
    ]

    S["post"] = [
        ("md", "## What changed since the freeze\n"
               "\n"
               "The freeze itself did not move: the submission is still the same file, and the headline "
               "result is still {master_over_ship} against the gate-plus-decoder stack. What changed is how much of "
               "it we are willing to claim. Post-freeze studies re-measured the small components, asked "
               "whether the practice ranking survives a different draw or a different severity level (how much ball "
               "the degrader drops), took the two large orderings to twenty held-out SkillCorner games, and "
               "corrected how we measured the live share of the match clock. None of them reopened the frozen file; all of them changed what "
               "a reader should believe about the numbers in it."),
        ("md", "The in-gap decoder is the clearest example. It was credited {ingap_over_gate} when it landed "
               "on the gate-only stack; on the current stack its leave-one-out marginal is {ingap_marginal} "
               "[{ingap_marginal_lo}, {ingap_marginal_hi}]. Those are different objects: most of the fall is "
               "work that grown D and the short-spell floor now do first, not a reversal of the decoder. The "
               "small number is also not sign-established. Any practice-set difference below {sign_threshold} is quoted as "
               "*level with zero*, and this one is well below it. Six degradation draws span {ingap_seed_lo} "
               "to {ingap_seed_hi}. The two-level interval, which resamples draws as well as games, is "
               "[{ingap_two_level_lo}, {ingap_two_level_hi}] and includes zero. We keep the stage, since "
               "removing it is not justified either, and stop calling it a gain."),
        ("md", "The big practice-set result is not a lucky draw or a lucky severity. Fresh degradation draws give "
               "{headline_six_draw} [{headline_six_draw_lo}, {headline_six_draw_hi}] over six draws, with "
               "the three-seed number from the results section inside that interval. Scaling the practice data to "
               "{level_clean} and {level_messy} of its fitted severity leaves the same order, and the "
               "smallest adjacent margin is {level_margin_08} at the cleaner level. These are ranking "
               "checks, not a second practice set, and no level was re-selected."),
        ("md", "On twenty held-out SkillCorner games the two large orderings keep their sign under all four "
               "rules we use to derive D labels there: the final build beats Baseline+ by {skillcorner_master_lo} to "
               "{skillcorner_master_hi}, and the gate-plus-decoder stack beats it by {skillcorner_ship_lo} to "
               "{skillcorner_ship_hi}. The third ordering, the final build against the gate-plus-decoder stack, is "
               "positive but only {skillcorner_master_ship_lo} to {skillcorner_master_ship_hi}, below the "
               "{skillcorner_bar} bar we require for a held-out claim. The held-out set therefore confirms "
               "the two large gains and cannot resolve the small one; the practice-set grid still decides it."),
        ("md", "The ball-in-play check had the wrong clock in its ratio, now fixed. The final file has "
               "{ball_in_play_min} of live play on a {playing_clock_min} playing clock (the length of the two halves), a live share of "
               "{inplay_rate}; the ten external games average {external_inplay_rate} ± {external_inplay_sd}. "
               "On the rate axis that is {inplay_z} standard deviations from the external mean — inside the "
               "noise, not the shortfall a raw-minute comparison suggested. The raw-minute axis is confounded "
               "by our unusually long clock and is no longer quoted."),
        ("md", "The stage attribution is a substitution result, not a sum. Removing one stage at a time, with one degradation draw, puts the "
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
               "`postprocess.py`; the harness is `dev_v1.py`; the renderer is `render.py`. This page is built by `scripts/walkthrough_numbers.py` and `scripts/build_walkthrough.py` from `numbers.json`."),
        ("md", "The practice data is public tracking from DFL, Metrica and SkillCorner, used under their licenses "
               "and never redistributed. The match data is the organizer's and stays on this machine. The scoring "
               "code is the organizer's `evaluation/possession_metrics.py`, imported, not re-implemented. Figures "
               "that render tracking are local-only under D-002; the public profile replaces them with placeholders."),
    ]

    S["glossary"] = [
        ("md", "## Glossary\n"
               "\n"
               "Every technical term used on this page, in plain words. Listed alphabetically; each entry gives the football meaning first, then our name for it.\n"
               "\n"
               "| term | plain meaning |\n"
               "|---|---|\n"
               "| **H / A / D / X** | The four labels: Barcelona, Athletic, dead ball (a stoppage), and not "
               "graded. A frame with no grade is X. |\n"
               "| **possession (spell)** | A run of frames in which one team has the ball. A dead ball ends "
               "it; the score counts these runs, not frames. |\n"
               "| **stoppage / dead ball** | Play is stopped by the referee: a foul, throw-in, goal kick, "
               "injury or break. The ball can still be on the pitch. |\n"
               "| **ball-missing run (gap)** | A stretch with no tracked ball. It can be a stoppage or a "
               "tracker dropout. |\n"
               "| **tracker dropout** | The tracking loses the ball while play continues. |\n"
               "| **in-gap decoder** | The stage that decides H or A inside a gap the gate left live, from "
               "the players' shape. |\n"
               "| **D gate** | The rule that turns a long ball-missing stretch with still players into D. |\n"
               "| **Baseline+** | The first stage: nearest player within a radius, otherwise the last owner, "
               "with a merge. |\n"
               "| **`smooth50`** | The organizers' reference: nearest player, majority-smoothed over two "
               "seconds. |\n"
               "| **grown D** | A rule that extends a visible resting ball to the stoppage around it. |\n"
               "| **sub-gap rule** | A second, weaker gate that finds a stoppage inside a gap the main gate "
               "left alone. |\n"
               "| **short-spell floor** | The rule that deletes H/A spells shorter than a second and a half. |\n"
               "| **flicker** | A very short spurious spell, usually one frame. |\n"
               "| **hand-over** | The ball changes team. |\n"
               "| **ball-free model** | A classifier that estimates who has the ball from the players, "
               "without the ball. |\n"
               "| **emission** | A per-frame score from the ball-free model: how likely each team is to have "
               "the ball. |\n"
               "| **HMM (hidden Markov model)** | A sequence model with hidden states (here H/A) and a cost "
               "for switching. |\n"
               "| **Viterbi** | The algorithm that finds the most likely state sequence under the switching "
               "cost. |\n"
               "| **switching penalty** | A cost added whenever the model changes team; it stops flicker. |\n"
               "| **P(H)** | The ball-free probability that Barcelona has the ball on a frame. |\n"
               "| **IoU (intersection over union)** | Shared time divided by the time either spell covers. "
               "One means identical. |\n"
               "| **spellF1** | The ranking score (spell-level F-score): match predicted possessions to true "
               "ones one-to-one by IoU, then average the F-score over five thresholds. |\n"
               "| **macroF1** | Per-frame F-score averaged over H, A and D; a tie-break, not the ranking "
               "score. |\n"
               "| **precision / recall** | Of the spells we predict, how many are right. Of the true "
               "spells, how many we find. |\n"
               "| **false positive / false negative** | A predicted spell with no true match / a true spell "
               "we miss. |\n"
               "| **marginal (leave-one-out)** | The score change from removing one stage and re-running the "
               "whole build. |\n"
               "| **game-cluster interval** | A confidence interval from resampling the ten practice games, "
               "not the frames. |\n"
               "| **`dev-v1` (practice set)** | Our internal name for the ten public matches with answers, degraded to look like our tracking. This page says *practice set*. |\n"
               "| **degradation draw (seed)** | One random version of the degradation; the seeds are "
               "separate draws. |\n"
               "| **graded frames** | The frames inside the two halves that are scored; the rest are X. |\n"
               "| **primary cell** | One of the six practice-set configurations: three ball-loss severities "
               "times two thresholds for how still the players must be. The headline averages them. |\n"
               "| **public window** | The five public minutes with answers in our match; report-only, never "
               "used to choose. |\n"
               "| **report-only** | A number we report but never use to choose a method or a threshold. |\n"
                              "| **level with zero** | Our rule: a practice-set delta below {sign_threshold} has no "
               "established sign. |\n"
               "| **eligible frames** | Frames inside a gap the gate left live, which the decoder may "
               "rewrite. |\n"
               "| **substitutes / interaction** | Stage gains overlap: removing two stages does not add "
               "their separate gains. |\n"
               "| **gate-only stack** | The gate on top of Baseline+, before the in-gap decoder, grown D, the sub-gap rule and "
               "the floor. |\n"
               "| **CSV** | The submission file: one row per frame with the label. |\n"
               "| **HSMM** | A hidden Markov model with an explicit duration model; a variant we tried and dropped. |\n"
               "| **MBR** | Minimum Bayes risk decoding: pick the label sequence with the best expected score. We tried it and dropped it. |\n"
               "| **Viterbi decode** | Pick the single most likely label sequence. We tried it in place of our rules and dropped it. |\n"
               "| **hysteresis** | A rule that changes the owner only when the evidence is clear, so one noisy frame cannot flip it. |\n"
               "| **merge** | Joining a very short spell into the longer ones around it. |\n"
                              "| **owner** | The team that last touched the ball and still controls it. |\n"
               "| **resting ball** | A ball lying still, for example before a free kick. |\n"
               "| **Metrica / SkillCorner** | Two more providers of public match tracking data. |\n"
               "| **tracker / tracking** | The system that follows players and ball through the match footage. It sometimes loses the ball. |\n"
               "| **Hz** | Times per second. Our labels come once per frame, at the frame rate. |\n"
               "| **severity (level)** | How much ball the degrader drops. The practice set uses the severity fitted to our match. |\n"
               "| **oracle** | A score with perfect knowledge of the answer: an upper bound no real rule reaches. |\n"
               "| **CI (confidence interval)** | The range a number could plausibly take. Narrow means well measured. |\n"
               "| **playing clock** | The length of the two halves on the match clock, stoppages included. |\n"
               "| **gate-plus-decoder stack (ship)** | The build with the gate and the in-gap decoder only. The final build adds grown D, the sub-gap rule and the floor. |\n"
               "| **final build (frozen)** | The stack that produced the submission file, fixed before this page was written. |\n"
               "| **DFL** | The German football league's tracking data, one of the practice-set providers. |\n"
               "| **AUC** | Area under the ROC curve: how well a score separates two classes; half is "
               "chance, one is perfect. |\n"
               "| **classifier** | A model that maps frames to a probability, here of the home team having "
               "the ball. |\n"
               "| **degradation** | The process that makes clean practice tracking look like ours: drop the "
               "ball, smooth it, jitter the edges. |\n"),
    ]

    gtext = S["glossary"][0][1]
    glines = gtext.split("\n")
    k = glines.index("|---|---|") + 1
    rows = sorted((l for l in glines[k:] if l.startswith("|")),
                  key=lambda l: l.split("|")[1].replace("*", "").replace("`", "").strip().lower())
    S["glossary"] = [("md", "\n".join(glines[:k] + rows) + "\n")]

    S["appendix"] = [
        ("md", "## Appendix: traceability\n"
               "\n"
               "Every measured number in the prose is a key in `notebooks/numbers.json`, with its value, source, "
               "exact command and whether `--check` recomputes it from the live pipeline. The table lists the "
               "headline keys; the rest are in that file, and the figure table names every figure."),
        ("md", "### The process, stage by stage\n"
               "\n"
               "The full diagram for a reader who wants the process: each stage's decision, and the measured "
               "marginal on each arrow. The notes under the figure say why the arrows are substitutes and "
               "not a sum."),
        ("fig", "fig_pipeline_flow"),
        ("md", "### The gain ladder\n"
               "\n"
               "The supporting picture for the results section: Panel A is the practice-set selection and "
               "Panel B the public five minutes. The notes under the figure carry the substitution caveat "
               "and the {sign_threshold} rule."),
        ("fig", "fig_gain_waterfall"),
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
        ("row", "practice-set spellF1", "marginal gain", "reading"),
        ("smooth50", e["smooth50_dev"]["display"], "—", "reference floor"),
        ("Baseline+", e["baseline_plus_dev"]["display"],
         e["baseline_plus_over_smooth50"]["display"] + " vs smooth50", "about level to slightly ahead"),
        ("D gate", e["gate_dev"]["display"],
         e["gate_over_baseline_plus"]["display"] + " vs Baseline+", "the stoppage detector"),
        ("ship (gate + in-gap)", e["ship_dev"]["display"],
         e["ingap_over_gate"]["display"] + " vs gate", "the decoder's original credit"),
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
        ("no cooling breaks or video-review stops in the practice set",
         "part of the practice-to-match gap is structural; our D share is legitimately higher"),
        ("Metrica over-calls D when play continues after a foul (advantage)",
         str(e["practice_metrica"]["value"]) + " of " + str(e["practice_games"]["value"]) +
         " games; " + e["metrica_d_spread"]["display"] + " spread in its own present-D share"),
        ("short gaps (2–4 s)",
         "~" + e["sliver_gaps"]["display"] + " per game, " + e["sliver_share"]["display"] +
         " of graded; in-gap and sub-gap gains are probably a little high"),
        ("resting balls we still see in the practice set",
         e["anchor_recall_knob0"]["display"] + " / " + e["anchor_recall_knob1"]["display"] + " of them, against " +
         e["anchor_recall_clean"]["display"] + " in clean data; so grown D is measured pessimistically"),
        ("how exactly a gap starts and ends",
         "we jitter gap edges by about " + e["gap_edge_sd_s"]["display"] + " s (standard deviation); the spread we fitted to our match is " +
         e["gap_edge_spread_min"]["display"] + "–" + e["gap_edge_spread_max"]["display"] + " larger, so the practice set is too tidy here"),
        ("geometry (checked, closed)",
         "nearest player " + e["geom_nearest_dev"]["display"] + " vs " + e["geom_nearest_match"]["display"] +
         "; the nearest-player cue is not inflated"),
        ("ten games, " + e["practice_seeds"]["display"] + " seeds, one severity",
         "the intervals ignore that our degrader may be wrong, and we have one match"),
        ("level, not ranking",
         e["smooth50_dev"]["display"] + " on the practice set against " + e["smooth50_match"]["display"] +
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
    "gate_over_baseline_plus", "ingap_over_gate", "grown_delta", "subgap_delta",
    "postprocess_delta", "postprocess_delta_current", "attrib_floor", "attrib_subgap", "attrib_grown",
    "ingap_marginal", "sign_threshold", "floor_real_short", "floor_short_total", "floor_false_short",
    "explain_clean", "explain_flicker", "explain_missed",
    "ingap_added_spells", "ingap_removed_short", "ingap_flicker_before", "ingap_flicker_after",
    "ingap_neg_games", "ingap_cells_pos", "demo_ingap_ph", "demo_split_iou", "demo_continuous_iou",
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
    if local_only and profile == "public":
        caption = caption.replace("Real tracking; local copy only.", "Withheld in this public copy because it renders real tracking.")
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
    for key in ("title", "start", "1", "2", "3", "ingap", "4", "5", "6", "7", "8", "post", "close",
                "glossary", "appendix"):
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
