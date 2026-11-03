"""Label-free ball-in-play rate diagnostics with an explicit playing clock (E105).

Why this module exists
----------------------
The project quotes a written La Liga prior of ~53-55 ball-in-play minutes, and that figure is an
**absolute** number for a ~98-minute clock. Our graded span is longer than that clock, so comparing
our live-minute count to 53-55 directly produces a spurious "we are ~3 minutes over" reading. Three
cards (B04, E66, E84) and one draft of the report made that comparison before E84 corrected it.

The cure is not a better band, it is a denominator and an axis:

* **playing clock** = graded frames minus the published pre-kickoff block of the second half
  (:data:`possession.structure.KICKOFFS`, D-007; those frames are all ``D`` by rule, not evidence).
  The frozen CSV's playing clock is **103.95 min**, not the 105.90 min of its graded span and not the
  ~98 min the absolute band assumes.
* **live share of the playing clock** = (``H`` or ``A``) frames / playing-clock frames. This is the
  honest comparison to another match, because it divides out clock length. A raw live-minute count
  does not: a longer clock mechanically carries more live minutes at the same rate.
* external games' graded spans are kickoff-to-whistle (their loaders already drop Metrica's
  ``dead_pre`` and mark SkillCorner pre-kickoff frames ``X``; none has a published multi-minute
  forced block), so their graded clock **is** their playing clock. To compare like with like our
  denominator must be the playing clock too, not our graded span.

The 10 external games (``dev_v1.GAMES``, D-022) supply the reference distribution: live-minute
mean/sd and live-share mean/sd. We report z on both axes and label the rate axis as the honest one.

This is a plausibility read, not a selector: no threshold, no pass/fail gate on our labels, no
build wiring, and nothing here is imported by :mod:`possession.pipeline` or ``make_submission``.
It reads the submission CSV, the published structure constants and provider labels on *other*
games. It never reads the public GT.

A caveat the report prints rather than hides: the pooled 10-game sd is mostly between-provider
spread. Against the 7 IDSSE games alone our rate z is ~0 while the raw-minute z is > +1 -- the same
denominator lesson in a second form. ``sensitivity()`` returns the provider split and the
leave-one-game-out range.

Use
---
    from possession import diagnostics
    rep = diagnostics.diagnose()          # frozen CSV + the 10 external games
    print(diagnostics.format_report(rep))

    python -m possession.diagnostics      # same report; --csv PATH for another candidate file

``check_e84_reference(rep)`` reproduces E84's published external reference (55.6 +/- 6.4 live min,
58.2 +/- 6.4% share) as the acceptance test.
"""
import csv
import os

import numpy as np

from . import config, structure

FPS = 25.0
FRAMES_PER_MIN = 60.0 * FPS
CANONICAL_CSV = os.path.join(config.SOLUTION_DIR, "outputs", "submission.csv")

# E84 sec. 2 / Z01 sec. 6: the external-game reference, rounded as published. Used only by
# check_e84_reference(); it is a reproduction fixture, not a target for our labels.
E84_REFERENCE = dict(live_min_mean=55.56, live_min_sd=6.37,
                     live_share_mean=0.582, live_share_sd=0.064)


def read_submission(path=None):
    """Labels of a submission CSV as one ``<U1`` array indexed by video frame."""
    path = CANONICAL_CSV if path is None else path
    frames, labels = [], []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["frame", "label"]:
            raise ValueError("unexpected header in {}: {}".format(path, header))
        for row in reader:
            frames.append(int(row[0]))
            labels.append(row[1])
    frames = np.asarray(frames, np.int64)
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("frames in {} are not 0..N-1 in order".format(path))
    out = np.asarray(labels, dtype="<U1")
    bad = np.setdiff1d(np.unique(out), np.array(list("HADX"), dtype="<U1"))
    if len(bad):
        raise ValueError("unexpected labels in {}: {}".format(path, bad.tolist()))
    return out


def forced_dead_mask(n_frames):
    """Graded frames published as dead before their half's kickoff (D-007; 2H only here).

    The single source is :data:`possession.structure.HALF_BLOCKS` / ``KICKOFFS``; no frame constant
    is introduced.
    """
    mask = np.zeros(n_frames, bool)
    for h, a, _z in structure.HALF_BLOCKS:
        b = min(structure.KICKOFFS[h], n_frames)
        if a < b:
            mask[a:b] = True
    return mask


def inplay(labels, forced=None):
    """Ball-in-play summary of one label array, on both denominators.

    ``clock`` is graded and not forced-dead. ``live_share_clock`` is the number to compare across
    matches; ``live_share_graded`` is kept only to make the old (denominator-mixed) figure explicit.
    The default forced mask is this match's published structure, so ``labels`` is expected to be a
    full-match 25 Hz array (``H/A/D/X`` per video frame). Pass ``forced`` to override it.
    """
    labels = np.asarray(labels)
    n = len(labels)
    graded = labels != "X"
    forced = forced_dead_mask(n) if forced is None else np.asarray(forced, bool)
    if forced.shape != graded.shape:
        raise ValueError("forced mask does not match the label array")
    clock = graded & ~forced
    live = np.isin(labels, ("H", "A")) & clock
    dead = (labels == "D") & clock
    live_graded = np.isin(labels, ("H", "A")) & graded
    dead_graded = (labels == "D") & graded
    live_off_clock = np.isin(labels, ("H", "A")) & forced
    frames = max(1, int(clock.sum()))
    gframes = max(1, int(graded.sum()))
    return dict(
        frames=int(n),
        graded_frames=int(graded.sum()),
        ungraded_frames=int((~graded).sum()),
        forced_dead_frames=int(forced.sum()),
        forced_dead_all_d=bool((labels[forced] == "D").all()) if forced.any() else True,
        playing_clock_frames=int(clock.sum()),
        live_frames=int(live.sum()),
        d_frames=int(dead.sum()),
        live_frames_graded=int(live_graded.sum()),
        d_frames_graded=int(dead_graded.sum()),
        live_label_in_forced_block=int(live_off_clock.sum()),
        graded_min=float(graded.sum() / FRAMES_PER_MIN),
        forced_dead_min=float(forced.sum() / FRAMES_PER_MIN),
        playing_clock_min=float(clock.sum() / FRAMES_PER_MIN),
        live_min=float(live.sum() / FRAMES_PER_MIN),
        d_min=float(dead.sum() / FRAMES_PER_MIN),
        live_share_clock=float(live.sum() / frames),
        live_share_graded=float(live_graded.sum() / gframes),
        d_share_clock=float(dead.sum() / frames),
        d_share_graded=float(dead_graded.sum() / gframes),
    )


def external_inplay(game_ids=None):
    """Per-game live minutes and live share for the external dev set (provider labels, other games).

    The denominator is each game's graded span, which is kickoff-to-whistle for these loaders, i.e.
    its playing clock. Returns one dict per game, in ``dev_v1.GAMES`` order.
    """
    from . import dev_v1, external

    ids = dev_v1.GAMES if game_ids is None else tuple(game_ids)
    rows = []
    for game_id in ids:
        game = external.load_game(game_id)
        ref = np.asarray(game.ref)
        graded = np.zeros(len(ref), bool)
        for a, b in (game.runs if game.runs is not None else [(0, len(ref))]):
            graded[a:b] = True
        live = np.isin(ref, ("H", "A")) & graded
        dead = (ref == "D") & graded
        frames = max(1, int(graded.sum()))
        rows.append(dict(
            game=game_id,
            playing_clock_min=float(graded.sum() / FRAMES_PER_MIN),
            live_min=float(live.sum() / FRAMES_PER_MIN),
            live_share=float(live.sum() / frames),
            d_share=float(dead.sum() / frames),
        ))
    return rows


def reference_stats(rows):
    """Mean/sd (ddof=1, as E84) and the spread of the external reference distribution."""
    live = np.array([r["live_min"] for r in rows], float)
    share = np.array([r["live_share"] for r in rows], float)
    clock = np.array([r["playing_clock_min"] for r in rows], float)
    return dict(
        n=len(rows),
        live_min_mean=float(live.mean()),
        live_min_sd=float(live.std(ddof=1)),
        live_min_median=float(np.median(live)),
        live_min_min=float(live.min()),
        live_min_max=float(live.max()),
        live_share_mean=float(share.mean()),
        live_share_sd=float(share.std(ddof=1)),
        live_share_median=float(np.median(share)),
        live_share_min=float(share.min()),
        live_share_max=float(share.max()),
        clock_min_mean=float(clock.mean()),
        clock_min_sd=float(clock.std(ddof=1)),
        clock_min_min=float(clock.min()),
        clock_min_max=float(clock.max()),
        corr_clock_share=float(np.corrcoef(clock, share)[0, 1]) if len(rows) > 2 else float("nan"),
    )


def zscores(ours, ref):
    """z on the raw-minute axis, the rate axis (honest), and two explicit cross-checks.

    * ``z_minutes``: ours vs the external live-minute distribution. Confounded by clock length.
    * ``z_rate``: ours (live / playing clock) vs the external live-share distribution. Honest.
    * ``z_rate_graded``: the same with our graded denominator; this is the convention E84/Z01
      quoted before the denominator was corrected in E105, and it is not like-for-like, because
      our graded span carries the 1.96-min published forced block while the external clocks do not.
    * ``z_minutes_at_our_clock``: external mean share applied to our playing clock, then z on the
      minutes spread *at a fixed clock* (share sd x our clock). This is algebraically identical to
      ``z_rate``; it is printed to show the same prior in minute units, and it is what makes the
      raw-minute axis's +0.30 above-average reading visibly an artifact of clock length.
    """
    expected_min = ref["live_share_mean"] * ours["playing_clock_min"]
    sd_min_at_our_clock = ref["live_share_sd"] * ours["playing_clock_min"]
    return dict(
        z_minutes=float((ours["live_min"] - ref["live_min_mean"]) / ref["live_min_sd"]),
        z_rate=float((ours["live_share_clock"] - ref["live_share_mean"]) / ref["live_share_sd"]),
        z_rate_graded=float((ours["live_share_graded"] - ref["live_share_mean"]) / ref["live_share_sd"]),
        expected_live_min_at_our_clock=float(expected_min),
        z_minutes_at_our_clock=float((ours["live_min"] - expected_min) / sd_min_at_our_clock),
    )


def sensitivity(ours, rows, ref=None):
    """Provider split and leave-one-game-out range of the two z-scores.

    The pooled 10-game sd is mostly between-provider spread, so the sign of the rate z can depend on
    the mix. This reports that dependence; it does not select or redefine anything. Subgroups keep the
    headline pool intact and are shown only because a 6.4-point sd across 10 games is not a stable
    reference on its own.
    """
    ref = reference_stats(rows) if ref is None else ref
    by_provider = {}
    for provider in sorted({r["game"].split(":", 1)[0] for r in rows}):
        sub = [r for r in rows if r["game"].split(":", 1)[0] == provider]
        if len(sub) >= 3:
            sr = reference_stats(sub)
            by_provider[provider] = dict(n=len(sub), live_share_mean=sr["live_share_mean"],
                                         live_share_sd=sr["live_share_sd"],
                                         z_rate=zscores(ours, sr)["z_rate"],
                                         z_minutes=zscores(ours, sr)["z_minutes"])
    loo = {}
    if len(rows) > 3:
        zs = [zscores(ours, reference_stats(rows[:i] + rows[i + 1:])) for i in range(len(rows))]
        loo = dict(z_rate_min=min(z["z_rate"] for z in zs),
                   z_rate_max=max(z["z_rate"] for z in zs),
                   z_minutes_min=min(z["z_minutes"] for z in zs),
                   z_minutes_max=max(z["z_minutes"] for z in zs))
    return dict(all=zscores(ours, ref), by_provider=by_provider, loo=loo)


def check_e84_reference(ref, atol=0.01):
    """Acceptance test: does the computed external reference match E84's published figures?

    Published as 55.6 +/- 6.4 live min and 58.2 +/- 6.4% share; ``atol`` is in minutes / share
    units after the published rounding (0.01 is tight enough to catch a different game set).
    """
    checks = {
        "live_min_mean": (ref["live_min_mean"], E84_REFERENCE["live_min_mean"]),
        "live_min_sd": (ref["live_min_sd"], E84_REFERENCE["live_min_sd"]),
        "live_share_mean": (ref["live_share_mean"], E84_REFERENCE["live_share_mean"]),
        "live_share_sd": (ref["live_share_sd"], E84_REFERENCE["live_share_sd"]),
    }
    return {k: dict(ours=v[0], e84=v[1], diff=abs(v[0] - v[1]), ok=bool(abs(v[0] - v[1]) <= atol))
            for k, v in checks.items()}


def diagnose(csv_path=None, game_ids=None):
    """The full read: frozen/candidate CSV summary, external reference, z on both axes, sensitivity."""
    frozen = inplay(read_submission(csv_path))
    games = external_inplay(game_ids)
    ref = reference_stats(games)
    return dict(frozen=frozen, external=games, reference=ref, z=zscores(frozen, ref),
                sensitivity=sensitivity(frozen, games, ref),
                e84=check_e84_reference(ref))


def format_report(rep):
    """One-page table. The denominator is printed next to every share, on purpose."""
    f, r, z = rep["frozen"], rep["reference"], rep["z"]
    lines = [
        "E105 ball-in-play diagnostic (label-free; plausibility read, not a selector)",
        "playing clock = graded frames minus the published 2H pre-kickoff block (D-007)",
        "",
        "frozen CSV",
        "  rows %d | graded %d (%.2f min) | forced-dead %d (%.2f min) | playing clock %d (%.2f min)"
        % (f["frames"], f["graded_frames"], f["graded_min"], f["forced_dead_frames"],
           f["forced_dead_min"], f["playing_clock_frames"], f["playing_clock_min"]),
        "  live %d fr = %.2f min = %.2f%% of the playing clock = %.2f%% of graded"
        % (f["live_frames"], f["live_min"], 100 * f["live_share_clock"], 100 * f["live_share_graded"]),
        "  D %d fr = %.2f min = %.2f%% of the playing clock = %.2f%% of graded"
        % (f["d_frames"], f["d_min"], 100 * f["d_share_clock"], 100 * f["d_share_graded"]),
        "  forced block all D: %s | live labels in the forced block: %d"
        % (f["forced_dead_all_d"], f["live_label_in_forced_block"]),
        "",
        "external reference (%d games, provider labels; graded span = kickoff-to-whistle)" % r["n"],
        "  live minutes   %6.2f +/- %.2f  (median %.2f, range %.2f-%.2f)"
        % (r["live_min_mean"], r["live_min_sd"], r["live_min_median"], r["live_min_min"], r["live_min_max"]),
        "  live share     %6.2f%% +/- %.2f%% (median %.2f%%, range %.2f-%.2f%%)"
        % (100 * r["live_share_mean"], 100 * r["live_share_sd"], 100 * r["live_share_median"],
           100 * r["live_share_min"], 100 * r["live_share_max"]),
        "  playing clock  %6.2f +/- %.2f min (range %.2f-%.2f); corr(clock, share) %.2f"
        % (r["clock_min_mean"], r["clock_min_sd"], r["clock_min_min"], r["clock_min_max"],
           r["corr_clock_share"]),
        "",
        "z-scores (ours vs the external distribution)",
        "  live minutes, raw                                  %+.2f   confounded by clock length" % z["z_minutes"],
        "  live share of the playing clock                    %+.2f   HONEST comparison" % z["z_rate"],
        "  live share of graded (E84/Z01's 54.3%%)            %+.2f   not like-for-like: includes the forced block"
        % z["z_rate_graded"],
        "  minutes at our clock (%.2f%% x %.2f = %.2f min)  %+.2f   same prior, minute units"
        % (100 * r["live_share_mean"], f["playing_clock_min"], z["expected_live_min_at_our_clock"],
           z["z_minutes_at_our_clock"]),
        "",
        "sensitivity of the reference pool (not a selector; the pooled sd is between-provider spread)",
    ]
    sens = rep.get("sensitivity") or {}
    for provider, s in sens.get("by_provider", {}).items():
        lines.append("  %-10s n=%-2d share %5.2f%% +/- %.2f%%   z_rate %+.2f   z_minutes %+.2f"
                     % (provider, s["n"], 100 * s["live_share_mean"], 100 * s["live_share_sd"],
                        s["z_rate"], s["z_minutes"]))
    loo = sens.get("loo") or {}
    if loo:
        lines.append("  leave-one-game-out      z_rate %+.2f..%+.2f   z_minutes %+.2f..%+.2f"
                     % (loo["z_rate_min"], loo["z_rate_max"],
                        loo["z_minutes_min"], loo["z_minutes_max"]))
    lines += [
        "",
        "E84 reference reproduction (55.6 +/- 6.4 min, 58.2 +/- 6.4%%): %s"
        % ("PASS" if all(c["ok"] for c in rep["e84"].values()) else "FAIL"),
    ]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=None, help="submission CSV (default: solution/outputs/submission.csv)")
    args = parser.parse_args(argv)
    print(format_report(diagnose(args.csv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
