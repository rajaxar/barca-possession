"""Thin CLI over :mod:`possession.render`: render a frame range as MP4 (+ GIF).

    cd solution && .venv/bin/python scripts/clip.py --start 23750 --end 24250 --gif
    .venv/bin/python scripts/clip.py --game metrica:game1 --start 30000 --end 30500 --gif
    .venv/bin/python scripts/clip.py --start 23750 --end 24250 --highlight player:23 \
        --trail 23 --gif --crop 10

Defaults: our match (``data.load_match()``), labels from ``outputs/submission.csv``, touches
detected on a 60 s padded window around the clip, D/ball-missing bands and spell boundaries
on, MP4 at 10 fps. Every output path is under ``scratch/viz/`` unless ``--out`` says otherwise;
nothing here is publishable (D-002).
"""
import argparse
import os
import sys

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION_DIR)
sys.path.insert(0, SOLUTION_DIR)

from possession import data, external, render, touches  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "scratch", "viz")
TOUCH_PAD = 1500  # 60 s of context, so the detector can calibrate its threshold


def _highlight(text):
    """'player:23' | 'team:1' | 'ball' -> annotation spec."""
    if text == "ball":
        return render.highlight_ball()
    head, _, tail = text.partition(":")
    if head == "player":
        return render.highlight_player(int(tail))
    if head == "team":
        return render.highlight_team(int(tail))
    raise argparse.ArgumentTypeError("expected player:SLOT, team:N or ball, got {!r}".format(text))


def _trail(text):
    """'23' | '23:75' | 'ball' | 'ball:75' -> trail spec (number = source frames)."""
    head, _, tail = text.partition(":")
    target = "ball" if head == "ball" else int(head)
    return render.trail(target, length=int(tail) if tail else 25)


def _touches(match, start, end):
    """Touch DataFrame in absolute frames, or None when the detector cannot calibrate."""
    a, b = max(0, start - TOUCH_PAD), min(len(match.B), end + TOUCH_PAD)
    try:
        df = touches.detect_touches(match.P[a:b], match.T, match.B[a:b], match.half[a:b])
    except ValueError as exc:
        print("clip: touch detection skipped ({}), pass --no-touches to silence".format(exc))
        return None
    df = df.copy()
    df["frame"] = df["frame"].astype(int) + a
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, help="first frame, inclusive")
    ap.add_argument("--end", type=int, help="last frame, exclusive")
    ap.add_argument("--pred", help="labels CSV (frame,label[,conf]); default outputs/submission.csv")
    ap.add_argument("--game", help="external game id, e.g. metrica:game1 or idsse:DFL-MAT-J03WMX")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory (default scratch/viz)")
    ap.add_argument("--name", help="output file stem")
    ap.add_argument("--gif", action="store_true", help="also write a GIF (<= 1.5 MB by default)")
    ap.add_argument("--no-mp4", action="store_true", help="skip the MP4")
    ap.add_argument("--fps", type=int, default=10, help="output frames per second (8-12 for GIF)")
    ap.add_argument("--size", type=int, default=800, help="output width in pixels (default 800)")
    ap.add_argument("--crop", type=float, default=8.0, help="margin around the action, meters")
    ap.add_argument("--highlight", action="append", type=_highlight, default=[],
                    metavar="player:SLOT|team:N|ball")
    ap.add_argument("--trail", action="append", type=_trail, default=[], metavar="SLOT[:LEN]|ball[:LEN]")
    ap.add_argument("--no-touches", action="store_true", help="do not run the touch detector")
    ap.add_argument("--no-annotate", action="store_true",
                    help="no automatic D/ball-missing bands or spell boundaries")
    ap.add_argument("--title", help="banner title (default: game id / frame range)")
    ap.add_argument("--list-games", action="store_true", help="print external game ids and exit")
    args = ap.parse_args(argv)

    if args.list_games:
        for game_id in external.list_games():
            print(game_id)
        return 0
    if args.start is None or args.end is None:
        ap.error("--start and --end are required")

    if args.game:
        game = external.load_game(args.game)
        match, game_ref = game.match, game.ref
        title = args.title or args.game
        name = args.name or "{}_{}_{}".format(args.game.replace(":", "_"), args.start, args.end)
        pitch = external.pitch(args.game)
    else:
        match, game_ref = data.load_match(), None
        title = args.title or "BAR-ATH tactical tracking"
        name = args.name or "bar_ath_{}_{}".format(args.start, args.end)
        pitch = (data.PITCH_L, data.PITCH_W)

    labels = confidence = reference = None
    if args.pred is None and not args.game:
        args.pred = os.path.join(SOLUTION_DIR, "outputs", "submission.csv")
    if args.pred:
        labels, confidence = render.load_labels(args.pred)
        reference = game_ref
        title += "  |  " + os.path.basename(args.pred)
    elif game_ref is not None:
        # leave labels None: render_clip falls back to the game's local reference and labels
        # the banner LABEL (not PRED), so a practice game is never mistaken for a prediction
        title += "  |  label = local reference"

    target = game if (args.game and args.pred is None) else match
    touch_df = None if args.no_touches else _touches(match, args.start, args.end)
    res = render.render_clip(
        target, args.start, args.end, labels=labels, confidence=confidence, reference=reference,
        highlight=args.highlight, trail=args.trail, touches=touch_df, title=title,
        out=args.out, name=name, mp4=not args.no_mp4, gif=args.gif, fps=args.fps,
        size=args.size, crop=args.crop, pitch=pitch, annotate=not args.no_annotate)

    print("clip: {} frames -> {}x{} px in {:.1f}s".format(
        res["frames"], res["pixels"][0], res["pixels"][1], res["seconds"]))
    if res["mp4"]:
        print("  mp4 " + os.path.abspath(res["mp4"]))
    if res["gif"]:
        print("  gif {} ({:.2f} MB)".format(os.path.abspath(res["gif"]),
                                            (res["gif_bytes"] or 0) / 1e6))
    if touch_df is not None:
        inside = touch_df[(touch_df.frame >= args.start) & (touch_df.frame < args.end)]
        print("  touches in clip: {}".format(len(inside)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
