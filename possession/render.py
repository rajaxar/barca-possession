"""Top-down pitch clip renderer: tracking + ball + predicted labels, as MP4 and GIF.

Developer tooling, not part of the shipped CSV path (nothing in ``make_submission`` imports it).
It lives in ``possession/`` rather than ``scripts/`` so the walkthrough notebook (P02) can import
``from possession.render import render_clip``: ``scripts/`` is not a package and would need a
``sys.path`` hack. Matplotlib and Pillow are already pinned runtime dependencies.

**Local only (D-002 / GUARDRAILS 1.3).** Every frame here is derived from match tracking;
never publish a rendered clip, a frame or a thumbnail, and never send one off-machine.

Everything is indexed by absolute video frame, the same row as the submission CSV. ``start``
is inclusive, ``end`` exclusive. The API is deliberately plain: annotation helpers return
dicts, ``render_clip`` consumes lists of them, so a notebook can compose a figure without
touching matplotlib internals::

    from possession import data, render
    m = data.load_match()
    labels, conf = render.load_labels("outputs/submission.csv")
    ann = render.default_annotations(m, labels, 23750, 24250)
    render.render_clip(m, 23750, 24250, labels=labels, confidence=conf,
                       highlight=render.highlight_player(23),
                       trail=render.trail(23, length=50),
                       bands=ann["bands"], boundaries=ann["boundaries"],
                       out="../scratch/viz", gif=True, fps=10)

Encoding uses the system ``ffmpeg`` when present (it is not in the venv); without it the GIF
falls back to Pillow and the MP4 is skipped. Rendered frames go to a throwaway directory
under ``out`` and are deleted afterward, so a run leaves only the video files.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

FPS = 25                     # source tracking rate
TOUCH_FADE = 8               # frames a touch label stays on screen after the event (0.3 s)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(_ROOT, "scratch", "viz")  # local-only (D-002)
TEAM_COLORS = ("#1a5fb4", "#c01c28")
LABEL_COLORS = {"H": TEAM_COLORS[0], "A": TEAM_COLORS[1], "D": "#5c5c5c", "X": "#b0b0b0"}
BAND_COLORS = {"D": "#c0392b", "missing": "#34558b", "custom": "#8e44ad"}
PITCH_BG = "#e9f2e4"
PITCH_LINE = "#9db39d"
HIGHLIGHT = "#f5b301"
TRAIL_COLOR = "#e07b00"
_CODE_TO_LABEL = np.array(["H", "A", "D", "X"])
_VALID = ("H", "A", "D", "X")


# ------------------------------------------------------------------ labels

def load_labels(path):
    """Read a labels CSV -> (labels, confidence).

    ``labels`` is a ``<U1`` array indexed by absolute frame, ``""`` where the file has no row.
    ``confidence`` is ``None``, a per-frame array, or an (N, 4) array over H/A/D/X, depending on
    what the file carries. Extra columns are optional: ``conf``/``confidence``/``p``, or one
    probability column per code (``H``, ``p_H``, ``prob_H``, ...). Numeric labels 0..3 are read
    as H/A/D/X (the organizer's code order).
    """
    import pandas as pd

    df = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in df.columns}
    fcol = lower.get("frame", lower.get("frame_id"))
    lcol = lower.get("label", lower.get("pred", lower.get("prediction")))
    if fcol is None or lcol is None:
        raise ValueError("labels CSV needs 'frame' and 'label' columns: " + path)
    frames = df[fcol].to_numpy(np.int64)
    if len(frames) and frames.min() < 0:
        raise ValueError("negative frame in " + path)
    n = int(frames.max()) + 1 if len(frames) else 0

    raw = df[lcol].astype(str).str.strip().str.upper()
    raw = raw.replace({"0": "H", "1": "A", "2": "D", "3": "X"})
    bad = ~raw.isin(list(_VALID))
    if bad.any():
        raise ValueError("unknown label {!r} in {}".format(raw[bad].iloc[0], path))
    labels = np.full(n, "", dtype="<U1")
    labels[frames] = raw.to_numpy(dtype="<U1")

    confidence = None
    conf_col = next((lower[k] for k in ("conf", "confidence", "p") if k in lower), None)
    prob_cols = {}
    for code in _VALID:
        for stem in (code.lower(), "p_" + code.lower(), "prob_" + code.lower()):
            if stem in lower:
                prob_cols[code] = lower[stem]
                break
    if conf_col is not None:
        confidence = np.full(n, np.nan, np.float32)
        confidence[frames] = pd.to_numeric(df[conf_col], errors="coerce").to_numpy(np.float32)
    elif len(prob_cols) == 4:
        confidence = np.full((n, 4), np.nan, np.float32)
        for j, code in enumerate(_VALID):
            confidence[frames, j] = pd.to_numeric(
                df[prob_cols[code]], errors="coerce").to_numpy(np.float32)
    return labels, confidence


def as_labels(values):
    """Strings/codes/Series -> ``<U1`` array. ``None`` stays ``None``."""
    if values is None:
        return None
    if isinstance(values, str):
        return load_labels(values)[0]
    a = np.asarray(values)
    if a.dtype == np.dtype("<U1"):
        valid = ((a == "") | (a == "H") | (a == "A") | (a == "D") | (a == "X"))
        if valid.all():
            return a  # already normalized by load_labels; avoid a 200k-element Python pass
    if a.dtype.kind in "iu":
        if a.size and (a.min() < 0 or a.max() > 3):
            raise ValueError("numeric labels outside 0..3")
        return _CODE_TO_LABEL[a.astype(np.int8)]
    out = np.array([str(x).strip().upper()[:1] for x in a.ravel()], dtype="<U1")
    return out.reshape(a.shape)


def _label_at(labels, frame):
    if labels is None or frame < 0 or frame >= len(labels):
        return ""
    return str(labels[frame])


def _confidence_at(confidence, frame, code):
    """Scalar confidence for ``code`` at ``frame``, or None when the CSV has none."""
    if confidence is None or frame < 0 or frame >= len(confidence):
        return None
    if confidence.ndim == 1:
        value = float(confidence[frame])
    elif code in _VALID:
        value = float(confidence[frame, int(np.flatnonzero(_CODE_TO_LABEL == code)[0])])
    else:
        return None
    return value if np.isfinite(value) else None


# ------------------------------------------------------------------ annotation specs

def highlight_player(slot, color=HIGHLIGHT, size=230.0, label=None):
    """Ring one player slot (column of ``match.P``) for the whole clip."""
    return {"kind": "highlight", "target": ("player", int(slot)), "color": color,
            "size": float(size), "label": label}


def highlight_team(team, color=HIGHLIGHT, size=150.0, label=None):
    """Ring every tracked player of ``team`` (0 home, 1 away)."""
    return {"kind": "highlight", "target": ("team", int(team)), "color": color,
            "size": float(size), "label": label}


def highlight_ball(color=HIGHLIGHT, size=230.0, label=None):
    """Ring the ball (or its last known position when it is missing)."""
    return {"kind": "highlight", "target": ("ball",), "color": color,
            "size": float(size), "label": label}


def trail(target, length=25, color=TRAIL_COLOR, width=1.8, label=None):
    """Draw the last ``length`` source frames of a player ``slot`` or ``"ball"``."""
    target = ("ball",) if target == "ball" else ("player", int(target))
    return {"kind": "trail", "target": target, "length": max(2, int(length)),
            "color": color, "width": float(width), "label": label}


def shade_band(start, end, color=BAND_COLORS["custom"], alpha=0.12, label=None):
    """Tint the pitch and mark the timeline while ``start <= frame < end``."""
    if end <= start:
        raise ValueError("empty band {}..{}".format(start, end))
    return {"kind": "band", "start": int(start), "end": int(end), "color": color,
            "alpha": float(alpha), "label": label}


def draw_boundary(frame, label=None, color="#333333", style="--", width=1.0):
    """Mark one instant on the timeline (a spell edge, a hand-over, a marker)."""
    return {"kind": "boundary", "frame": int(frame), "label": label, "color": color,
            "style": style, "width": float(width)}


def _runs(mask):
    """Half-open [start, stop) runs of a boolean array."""
    x = np.asarray(mask, np.int8)
    edges = np.flatnonzero(np.diff(np.r_[0, x, 0]) != 0)
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def _ball_array(match):
    return np.asarray(getattr(match, "B", match), dtype=np.float64)


def shade_missing_ball(match, start, end, min_frames=25, color=BAND_COLORS["missing"],
                       alpha=0.10, label="ball missing"):
    """Bands over ball-free runs (``B`` is NaN) at least ``min_frames`` long, clipped to the clip."""
    ball = _ball_array(match)
    start, end = max(0, int(start)), min(len(ball), int(end))
    missing = ~np.isfinite(ball[start:end, 0])
    out = []
    for a, b in _runs(missing):
        if b - a >= min_frames:
            out.append(shade_band(start + a, start + b, color=color, alpha=alpha, label=label))
    return out


def shade_d(labels, start, end, min_frames=5, color=BAND_COLORS["D"], alpha=0.07, label="D"):
    """Bands over runs of predicted ``D``, clipped to the clip."""
    labels = as_labels(labels)
    if labels is None:
        return []
    start, end = max(0, int(start)), min(len(labels), int(end))
    out = []
    for a, b in _runs(labels[start:end] == "D"):
        if b - a >= min_frames:
            out.append(shade_band(start + a, start + b, color=color, alpha=alpha, label=label))
    return out


def spell_boundaries(labels, start, end, include_d=True, include_x=False):
    """Boundaries where the label changes inside the clip, as :func:`draw_boundary` specs.

    Spells are maximal H/A runs, so a D edge is a spell edge too; ``include_d=False`` keeps only
    H <-> A hand-overs. ``include_x=False`` ignores unknown (``""``) and X edges.
    """
    labels = as_labels(labels)
    if labels is None:
        return []
    start, end = max(0, int(start)), min(len(labels), int(end))
    out = []
    for frame in range(start + 1, end):
        old, new = _label_at(labels, frame - 1), _label_at(labels, frame)
        if old == new or not new:
            continue
        if not include_x and (old in ("", "X") or new == "X"):
            continue
        if not include_d and (old == "D" or new == "D"):
            continue
        out.append(draw_boundary(frame, label=new, color=LABEL_COLORS.get(new, "#333333")))
    return out


def d_events(labels, start, end):
    """D onsets/ends inside the clip: [{"frame", "kind"}] with kind 'onset' or 'end'."""
    labels = as_labels(labels)
    if labels is None:
        return []
    start, end = max(0, int(start)), min(len(labels), int(end))
    out = []
    for frame in range(start + 1, end):
        old, new = _label_at(labels, frame - 1), _label_at(labels, frame)
        if old == new:
            continue
        if new == "D":
            out.append({"frame": frame, "kind": "onset"})
        elif old == "D":
            out.append({"frame": frame, "kind": "end"})
    return out


def default_annotations(match, labels=None, start=0, end=None, min_missing_frames=25):
    """Bands (D runs + ball-missing runs) and spell boundaries for a clip.

    Returns ``{"bands": [...], "boundaries": [...]}``, ready to pass straight to
    :func:`render_clip`. ``render_clip`` builds the same set when ``annotate=True``.
    """
    end = len(match.B) if end is None else int(end)
    bands = shade_d(labels, start, end) + shade_missing_ball(
        match, start, end, min_frames=min_missing_frames)
    return {"bands": bands, "boundaries": spell_boundaries(labels, start, end)}


# ------------------------------------------------------------------ tiny data adapters

def _split_match(obj):
    """(Match, name, local reference or None) from a Match or an ``evaluate.Game``."""
    if hasattr(obj, "match") and hasattr(obj, "ref"):
        return obj.match, getattr(obj, "name", None), np.asarray(obj.ref)
    return obj, getattr(obj, "name", None), None


def _half_spans(half):
    """{half: (first, last+1)} for half values > 0."""
    spans = {}
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        spans[int(h)] = (int(idx[0]), int(idx[-1]) + 1)
    return spans


def attack_directions(match, pitch):
    """{(half, team): +1/-1}: +1 = attacks toward +x, from the goalkeeper's own end.

    Generic version of ``structure.attack_direction`` for external games with their own half
    blocks: the keeper stands near the goal it defends, so a keeper at low x attacks +x.
    """
    out = {}
    for h, (a, b) in _half_spans(np.asarray(match.half)).items():
        for team in (0, 1):
            gk = int(np.asarray(match.gk)[h - 1, team])
            x = np.nanmean(np.asarray(match.P[a:b, gk, 0], dtype=np.float64))
            out[(h, team)] = 1 if (np.isfinite(x) and x < pitch[0] / 2.0) else -1
    return out


def _normalise_specs(value, int_factory, ball_factory):
    """None / dict / int / "ball" / list of those -> list of annotation dicts."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        if all(isinstance(v, dict) for v in value):
            return list(value)
        out = []
        for v in value:
            out.extend(_normalise_specs(v, int_factory, ball_factory))
        return out
    if value == "ball":
        return [ball_factory()]
    if isinstance(value, (int, np.integer)):
        return [int_factory(int(value))]
    raise TypeError("cannot read annotation {!r}".format(value))


def _normalise_bands(value):
    if value is None:
        return []
    out = []
    for b in value:
        if isinstance(b, dict):
            out.append(b)
        else:
            out.append(shade_band(*b))
    return out


def _normalise_boundaries(value):
    if value is None:
        return []
    out = []
    for b in value:
        if isinstance(b, dict):
            out.append(b)
        elif isinstance(b, (tuple, list)):
            out.append(draw_boundary(*b))
        else:
            out.append(draw_boundary(b))
    return out


# ------------------------------------------------------------------ layout helpers

def _fit_span(lo, hi, target, lo_limit, hi_limit):
    """Grow the span [lo, hi] to ``target`` around its center, slid inside the limits."""
    target = min(float(target), float(hi_limit - lo_limit))
    center = (lo + hi) / 2.0
    lo, hi = center - target / 2.0, center + target / 2.0
    if lo < lo_limit:
        lo, hi = lo_limit, lo_limit + target
    if hi > hi_limit:
        lo, hi = hi_limit - target, hi_limit
    return float(max(lo_limit, lo)), float(min(hi_limit, hi))


def _crop_box(match, start, end, margin, pitch, min_size=(45.0, 30.0),
              aspect_range=(1.30, 1.85), pad=3.0):
    """(x0, x1, y0, y1) meters around the tracked action, margin = meters of padding.

    The window is kept landscape (aspect in ``aspect_range``) so a clip reads as a pitch view
    rather than a portrait of a goalmouth scramble, and it may run ``pad`` meters past the
    touchlines so a player on the line is not cut in half by the crop.
    """
    x_lo, x_hi = -pad, pitch[0] + pad
    y_lo, y_hi = -pad, pitch[1] + pad
    P = np.asarray(match.P[start:end], dtype=np.float64)
    B = _ball_array(match)[start:end]
    pts = np.concatenate([P.reshape(-1, 2), B.reshape(-1, 2)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return x_lo, x_hi, y_lo, y_hi
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin
    # keep a usable window even when everyone is bunched (dead ball, cut to a bench)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0, x1 = min(x0, cx - min_size[0] / 2), max(x1, cx + min_size[0] / 2)
    y0, y1 = min(y0, cy - min_size[1] / 2), max(y1, cy + min_size[1] / 2)
    x0, x1 = max(x_lo, x0), min(x_hi, x1)
    y0, y1 = max(y_lo, y0), min(y_hi, y1)
    if x1 - x0 < min_size[0]:
        x0, x1 = _fit_span(x0, x1, min_size[0], x_lo, x_hi)
    if y1 - y0 < min_size[1]:
        y0, y1 = _fit_span(y0, y1, min_size[1], y_lo, y_hi)
    aspect = (x1 - x0) / max(y1 - y0, 1e-6)
    if aspect < aspect_range[0]:
        x0, x1 = _fit_span(x0, x1, aspect_range[0] * (y1 - y0), x_lo, x_hi)
    elif aspect > aspect_range[1]:
        y0, y1 = _fit_span(y0, y1, (x1 - x0) / aspect_range[1], y_lo, y_hi)
    return float(x0), float(x1), float(y0), float(y1)


def _draw_pitch(ax, pitch):
    """Pitch furniture in meters; the caller owns the axes limits (the crop box)."""
    L, W = pitch
    line = dict(color=PITCH_LINE, lw=0.9, zorder=1)
    patch = dict(line, fill=False)
    ax.add_patch(_rect(0, 0, L, W, facecolor=PITCH_BG, edgecolor=PITCH_LINE, lw=1.2,
                       zorder=0, fill=True))
    ax.plot([L / 2, L / 2], [0, W], **line)
    ax.add_patch(_circle(L / 2, W / 2, 9.15, **patch))
    ax.plot([L / 2], [W / 2], marker="o", ms=2.4, color=PITCH_LINE, zorder=1)
    for side in (0, 1):
        x = 0.0 if side == 0 else L
        sgn = 1.0 if side == 0 else -1.0
        ax.add_patch(_rect(min(x, x + sgn * 16.5), W / 2 - 20.16, 16.5, 40.32, **patch))
        ax.add_patch(_rect(min(x, x + sgn * 5.5), W / 2 - 9.16, 5.5, 18.32, **patch))
        ax.plot([x + sgn * 11.0], [W / 2], marker="o", ms=2.4, color=PITCH_LINE, zorder=1)
        ax.add_patch(_rect(min(x, x + sgn * 2.0), W / 2 - 3.66, 2.0, 7.32, **patch))


def _rect(x, y, w, h, fill=False, **kw):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fill=fill, **kw)


def _chip(alpha=0.72):
    """Small white label background for on-pitch text."""
    return dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=alpha)


def _clamp_point(point, x0, x1, y0, y1, inset=2.0):
    """Clamp a point inside the crop, keeping an inset so the marker is not cut by the edge."""
    q = np.asarray(point, dtype=float).copy()
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        step = min(inset, max((hi - lo) / 4.0, 0.0))
        q[axis] = min(max(q[axis], lo + step), hi - step)
    return q


def _circle(x, y, r, fill=False, **kw):
    from matplotlib.patches import Circle
    return Circle((x, y), r, fill=fill, **kw)


def _frame_indices(start, end, fps):
    """Source frames sampled for an output at ``fps`` (deterministic, last frame < end)."""
    n = max(1, int(np.floor((end - start) * float(fps) / FPS + 1e-9)))
    offsets = np.minimum(np.rint(np.arange(n) * FPS / float(fps)).astype(int), end - start - 1)
    return start + offsets


# ------------------------------------------------------------------ encoding

def _ffmpeg():
    return shutil.which("ffmpeg")


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError("{} failed ({}):\n{}".format(
            os.path.basename(cmd[0]), proc.returncode, proc.stderr.strip()))


def _encode_mp4(frames_dir, fps, path):
    _run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps),
          "-i", os.path.join(frames_dir, "f_%05d.png"),
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "1",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", path])


def _encode_gif(frames_dir, fps, path, width, max_bytes, colors):
    """Palette GIF, shrinking color count (never below 700 px) until it fits ``max_bytes``."""
    source = os.path.join(frames_dir, "f_%05d.png")
    attempts = [(colors, "bayer:bayer_scale=5"), (max(32, colors // 2), "bayer:bayer_scale=5"),
                (32, "none"), (16, "none")]
    size = None
    for n_colors, dither in attempts:
        vf = ("scale={w}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors={c}:stats_mode=diff[p];"
              "[b][p]paletteuse=dither={d}:diff_mode=rectangle").format(
                  w=width, c=n_colors, d=dither)
        _run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps),
              "-i", source, "-vf", vf, "-loop", "0", path])
        size = os.path.getsize(path)
        if max_bytes is None or size <= max_bytes:
            break
    return size


def _encode_gif_pillow(frames_dir, fps, path):
    """Fallback GIF without ffmpeg (Pillow is a pinned dependency)."""
    from PIL import Image
    names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))
    images = [Image.open(os.path.join(frames_dir, n)).convert("P", palette=Image.ADAPTIVE)
              for n in names]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(round(1000.0 / fps)), loop=0, optimize=True)
    return os.path.getsize(path)


def _write_png(fig, path):
    """Current Agg renderer buffer -> PNG (no second full draw, unlike ``fig.savefig``)."""
    from PIL import Image
    buffer = np.asarray(fig.canvas.buffer_rgba())
    Image.fromarray(buffer[:, :, :3]).save(path, compress_level=1)


# ------------------------------------------------------------------ renderer

def render_clip(match, start, end, labels=None, confidence=None, reference=None,
                highlight=None, trail=None, bands=None, boundaries=None, touches=None,
                title=None, out=None, name=None, mp4=True, gif=False, fps=10,
                size=800, crop=8.0, pitch=None, annotate=True, min_missing_frames=25,
                gif_max_bytes=1_500_000, gif_colors=96):
    """Render frames ``start..end-1`` of ``match`` to ``out/name.mp4`` and/or ``.gif``.

    Parameters
    ----------
    match : possession.data.Match or possession.evaluate.Game
        A ``Game`` is unwrapped; its ``ref`` becomes ``reference`` when ``labels`` is None.
    start, end : int
        Absolute video frames, ``start`` inclusive, ``end`` exclusive.
    labels, reference : path, array of H/A/D/X or 0..3 codes, or None
        Predicted and (local-only) reference labels. A path is read by :func:`load_labels`.
    confidence : array or None
        Per-frame confidence, or an (N, 4) H/A/D/X probability array.
    highlight, trail : spec / spec list
        From :func:`highlight_player`, :func:`highlight_team`, :func:`highlight_ball`,
        :func:`trail`; an int slot or ``"ball"`` is accepted as shorthand.
    bands, boundaries : spec list or None
        From :func:`shade_band`, :func:`shade_missing_ball`, :func:`shade_d`,
        :func:`draw_boundary`, :func:`spell_boundaries`. ``None`` + ``annotate=True`` adds the
        default D bands, ball-missing bands and spell boundaries.
    touches : pandas.DataFrame or None
        Rows from ``possession.touches.detect_touches`` (absolute ``frame`` column).
    out, name : str
        Output directory (created) and file stem (default ``clip_<start>_<end>``).
    mp4, gif, fps, size : bool, bool, int, int
        ``size`` is the output width in pixels; height follows the crop (700-900 px suits P02).
    crop : float
        Margin around the tracked action, in meters.
    pitch : (length, width) or None
        Defaults to the arrays' extent, which is the 105.3 x 68 match pitch for our data.

    Returns
    -------
    dict with ``mp4``, ``gif`` (paths or None), ``frames``, ``seconds``, ``pixels``,
    ``gif_bytes`` and ``labels`` (the label array used, if any).
    """
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_start = time.time()
    match, game_name, game_ref = _split_match(match)
    n_total = len(match.B)
    start, end = int(start), int(end)
    if not (0 <= start < end <= n_total):
        raise ValueError("need 0 <= start < end <= {} (got {}..{})".format(n_total, start, end))
    if not (0 < fps <= FPS):
        raise ValueError("fps must be in (0, {}]".format(FPS))

    labels = as_labels(labels)
    auto_reference = False
    if labels is None and game_ref is not None:
        labels = as_labels(game_ref)
        auto_reference = True
        if title is None and game_name:
            title = "{} (label = local reference)".format(game_name)
    reference = as_labels(reference)
    label_name = "LABEL" if auto_reference else "PRED"

    if pitch is None:
        finite = np.isfinite(np.asarray(match.P[start:end, :, 0], dtype=np.float64))
        x_max = np.nanmax(np.asarray(match.P[start:end, :, 0], dtype=np.float64)) if finite.any() else 105.3
        y_max = np.nanmax(np.asarray(match.P[start:end, :, 1], dtype=np.float64)) if finite.any() else 68.0
        pitch = (105.3 if x_max <= 106.0 else float(np.ceil(x_max)), 68.0 if y_max <= 69.0 else float(np.ceil(y_max)))
    pitch = (float(pitch[0]), float(pitch[1]))

    highlight = _normalise_specs(highlight, highlight_player, highlight_ball)
    trails = _normalise_specs(trail, trail, lambda: trail("ball"))
    if annotate:
        defaults = default_annotations(match, labels, start, end, min_missing_frames)
        bands = defaults["bands"] if bands is None else bands
        boundaries = defaults["boundaries"] if boundaries is None else boundaries
    bands = _normalise_bands(bands)
    boundaries = _normalise_boundaries(boundaries)

    # -- figure geometry: axes aspect equals the crop aspect exactly ---------------------
    x0, x1, y0, y1 = _crop_box(match, start, end, float(crop), pitch)
    aspect = (x1 - x0) / (y1 - y0)
    dpi = 100
    width_px = int(size) // 2 * 2  # yuv420p needs even dimensions
    height_px = int(round(0.94 * width_px / (0.80 * aspect))) // 2 * 2
    base = width_px / 800.0
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.80])
    ax.set_facecolor("#d8e8d1")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    _draw_pitch(ax, pitch)
    overlays = []
    for band in bands:
        patch = _rect(x0, y0, x1 - x0, y1 - y0, facecolor=band["color"], edgecolor="none",
                      alpha=band["alpha"], zorder=2, fill=True)
        patch.set_visible(False)
        ax.add_patch(patch)
        overlays.append(patch)

    # -- banner -------------------------------------------------------------------------
    fs = max(5.0, 9.5 * base)
    title = title or game_name or "clip {}-{}".format(start, end)
    head = fig.text(0.03, 0.985, title, ha="left", va="top", fontsize=fs * 1.05,
                    family="DejaVu Sans Mono", color="#222222")
    line1 = fig.text(0.03, 0.958, "", ha="left", va="top", fontsize=fs,
                     family="DejaVu Sans Mono", color="#222222")
    line2 = fig.text(0.03, 0.932, "", ha="left", va="top", fontsize=fs,
                     family="DejaVu Sans Mono", color="#222222")
    fig.text(0.97, 0.985, "local render - not for publication (D-002)", ha="right", va="top",
             fontsize=fs * 0.75, color="#888888")

    # -- timeline -----------------------------------------------------------------------
    tl = fig.add_axes([0.03, 0.045, 0.94, 0.045])
    tl.set_xlim(start, end)
    tl.set_ylim(0, 1)
    tl.set_yticks([])
    tl.set_xticks([])
    for spine in tl.spines.values():
        spine.set_color("#cccccc")
    tl.set_facecolor("#fafafa")
    for band in bands:
        a, b = max(start, band["start"]), min(end, band["end"])
        if b > a:
            tl.add_patch(_rect(a, 0.10, b - a, 0.80, facecolor=band["color"], edgecolor="none",
                               alpha=0.60, lw=0, fill=True))
    for bnd in boundaries:
        if start < bnd["frame"] < end:
            tl.plot([bnd["frame"], bnd["frame"]], [0.05, 0.95], color=bnd["color"],
                    lw=bnd.get("width", 1.0), ls=bnd.get("style", "--"), alpha=0.9)
    cursor = tl.plot([start, start], [0, 1], color="#111111", lw=1.6)[0]
    fig.text(0.03, 0.012, "clip timeline  {}..{}  ({:.1f}s)".format(start, end, (end - start) / FPS),
             fontsize=fs * 0.75, color="#666666", va="bottom")

    # -- static artists -----------------------------------------------------------------
    home = np.flatnonzero(np.asarray(match.T) == 0)
    away = np.flatnonzero(np.asarray(match.T) == 1)
    scat_home = ax.scatter([], [], s=46 * base, c=TEAM_COLORS[0], edgecolors="white",
                           linewidths=0.5, zorder=5)
    scat_away = ax.scatter([], [], s=46 * base, c=TEAM_COLORS[1], edgecolors="white",
                           linewidths=0.5, zorder=5)
    gks = np.asarray(match.gk)[int(np.clip(match.half[start], 1, 2)) - 1]
    scat_gk = ax.scatter([], [], s=78 * base, c="#333333", edgecolors=HIGHLIGHT,
                         linewidths=1.4, zorder=6)
    ball_dot = ax.scatter([], [], s=36 * base, c="white", edgecolors="#222222",
                          linewidths=0.9, zorder=8)
    ball_off = ax.scatter([], [], s=90 * base, facecolors="none", edgecolors="#666666",
                          linewidths=1.1, linestyle="--", zorder=7)
    ball_note = ax.text(0, 0, "", fontsize=fs * 0.8, color="#444444", zorder=9,
                        bbox=_chip())

    rings = [ax.scatter([], [], s=spec["size"] * base ** 2, facecolors="none",
                        edgecolors=spec["color"], linewidths=1.6, zorder=7)
             for spec in highlight]
    lines = [ax.plot([], [], color=spec["color"], lw=spec["width"], alpha=0.85, zorder=4)[0]
             for spec in trails]

    touch_pos = np.full((0, 2), np.nan)
    touch_team = np.zeros(0, np.int8)
    touch_frames = np.zeros(0, np.int64)
    touch_kind = []
    if touches is not None and len(touches):
        df = touches
        sel = df[(df["frame"] >= start) & (df["frame"] < end)]
        if len(sel):
            frames = sel["frame"].to_numpy(np.int64)
            pos = _ball_array(match)[frames]
            ok = np.isfinite(pos).all(axis=1)
            touch_pos, touch_team = pos[ok], sel["team"].to_numpy(np.int8)[ok]
            touch_frames = frames[ok]
            touch_kind = list(sel["kind"].astype(str).to_numpy()[ok])
    touch_marks = ax.scatter([], [], s=26 * base, marker="v", c="#6a3d9a",
                             edgecolors="white", linewidths=0.4, zorder=6)
    touch_note = ax.text(0, 0, "", fontsize=fs * 0.8, color="#5b2c83", zorder=9,
                         bbox=_chip())
    events = d_events(labels, start, end)

    # big translucent label in the corner, legible in a small GIF
    label_big = ax.text(0.985, 0.02, "", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=32 * base, color="#333333", alpha=0.45, zorder=3)

    band_label = ax.text(0.012, 0.985, "", transform=ax.transAxes, ha="left", va="top",
                         fontsize=fs * 0.8, color="#333333", alpha=0.85, zorder=9,
                         bbox=_chip())

    # -- static background: pitch, banner title, timeline, touch triangles -----------------
    # Everything that never changes is drawn once and cached; per frame we restore it and
    # redraw only the dynamic artists (Agg blitting, ~3x faster than a full savefig).
    if len(touch_frames):
        touch_marks.set_offsets(touch_pos)
        touch_marks.set_facecolors(np.where(touch_team == 0, TEAM_COLORS[0], TEAM_COLORS[1]))
    cursor.set_data([], [])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    ax_dynamic = (overlays + lines + [label_big, scat_home, scat_away, scat_gk,
                                      ball_dot, ball_off] + rings +
                  [ball_note, touch_note, band_label])

    # -- one pass over frames ------------------------------------------------------------
    out_dir = os.path.abspath(DEFAULT_OUT if out is None else out)
    os.makedirs(out_dir, exist_ok=True)
    stem = name or "clip_{}_{}".format(start, end)
    mp4_path = os.path.join(out_dir, stem + ".mp4")
    gif_path = os.path.join(out_dir, stem + ".gif")
    indices = _frame_indices(start, end, fps)
    P = np.asarray(match.P, dtype=np.float64)
    ball = _ball_array(match)
    half = np.asarray(match.half)
    dirs = attack_directions(match, pitch)
    half_spans = _half_spans(half)

    with tempfile.TemporaryDirectory(prefix=".frames-", dir=out_dir) as frames_dir:
        for k, frame in enumerate(indices):
            ph = P[frame]
            ok = np.isfinite(ph).all(axis=1)
            for scat, slots in ((scat_home, home), (scat_away, away)):
                m = ok[slots]
                scat.set_offsets(ph[slots][m] if m.any() else np.empty((0, 2)))
            h = int(half[frame]) if 0 <= frame < len(half) else 0
            if h in (1, 2):
                gk_slots = np.asarray(match.gk)[h - 1]
            else:
                gk_slots = gks
            gk_ok = ok[gk_slots]
            scat_gk.set_offsets(ph[gk_slots][gk_ok] if gk_ok.any() else np.empty((0, 2)))

            # ball: filled when tracked inside the crop, hollow ring at the last known
            # (or clamped) spot when it is missing or off-frame, with an honest note
            bp = ball[frame]
            if np.isfinite(bp).all():
                if x0 <= bp[0] <= x1 and y0 <= bp[1] <= y1:
                    ball_shown = bp
                    ball_dot.set_offsets([bp])
                    ball_dot.set_visible(True)
                    ball_off.set_visible(False)
                    ball_note.set_text("")
                else:
                    ball_shown = _clamp_point(bp, x0, x1, y0, y1)
                    ball_dot.set_visible(False)
                    ball_off.set_offsets([ball_shown])
                    ball_off.set_visible(True)
                    ball_note.set_position((ball_shown[0] + 1.2, ball_shown[1] + 1.2))
                    ball_note.set_text("ball off-frame")
            else:
                ball_dot.set_visible(False)
                known = np.flatnonzero(np.isfinite(ball[:frame + 1, 0]))
                if not len(known):
                    known = np.flatnonzero(np.isfinite(ball[frame:, 0]))
                    nxt = frame + int(known[0]) if len(known) else None
                else:
                    nxt = int(known[-1])
                if nxt is not None:
                    ball_shown = _clamp_point(ball[nxt], x0, x1, y0, y1)
                    ball_off.set_offsets([ball_shown])
                    ball_off.set_visible(True)
                    ball_note.set_position((ball_shown[0] + 1.2, ball_shown[1] + 1.2))
                    ball_note.set_text("ball n/a")
                else:
                    ball_shown = None
                    ball_off.set_visible(False)
                    ball_note.set_text("")
            ball_note.set_visible(bool(ball_note.get_text()))

            for ring, spec in zip(rings, highlight):
                kind, *rest = spec["target"]
                if kind == "ball":
                    if np.isfinite(ball[frame]).all():
                        pos = ball_shown
                    else:
                        off = np.asarray(ball_off.get_offsets(), dtype=float)
                        pos = off.reshape(-1, 2)[0] if len(off) else None
                elif kind == "team":
                    slots = np.flatnonzero(np.asarray(match.T) == rest[0])
                    slots = slots[ok[slots]]
                    ring.set_offsets(ph[slots] if len(slots) else np.empty((0, 2)))
                    continue
                else:
                    slot = rest[0]
                    pos = ph[slot] if 0 <= slot < len(ph) and ok[slot] else None
                if pos is None:
                    ring.set_offsets(np.empty((0, 2)))
                else:
                    ring.set_offsets([np.asarray(pos, dtype=float).reshape(2)])

            for line, spec in zip(lines, trails):
                length = spec["length"]
                lo = max(start, frame - length)
                kind, *rest = spec["target"]
                if kind == "ball":
                    seg = ball[lo:frame + 1]
                    line.set_data(seg[:, 0], seg[:, 1])
                else:
                    slot = rest[0]
                    if 0 <= slot < P.shape[1]:
                        seg = P[lo:frame + 1, slot]
                        line.set_data(seg[:, 0], seg[:, 1])
                    else:
                        line.set_data([], [])

            # touches: persistent triangles for the clip, a label for the nearest one
            if len(touch_frames):
                j = int(np.argmin(np.abs(touch_frames - frame)))
                if abs(int(touch_frames[j]) - frame) <= TOUCH_FADE:
                    pos = touch_pos[j]
                    touch_note.set_position((pos[0] + 1.6, pos[1] - 3.0))
                    touch_note.set_text("touch {} {}".format(
                        "H" if touch_team[j] == 0 else "A", touch_kind[j]))
                    touch_note.set_visible(True)
                else:
                    touch_note.set_visible(False)
            else:
                touch_marks.set_offsets(np.empty((0, 2)))
                touch_note.set_visible(False)

            # banner
            t_in_half = (frame - half_spans[h][0]) / FPS if h in half_spans else float("nan")
            attack = ""
            if h in (1, 2):
                attack = "  H{} A{}".format(
                    "+x" if dirs.get((h, 0)) == 1 else "-x",
                    "+x" if dirs.get((h, 1)) == 1 else "-x")
            line1.set_text("frame {:>6}   t={:+.1f}s   {}H{}".format(
                frame, t_in_half if np.isfinite(t_in_half) else float("nan"),
                h if h else "-", attack))

            pred = _label_at(labels, frame)
            conf = _confidence_at(confidence, frame, pred)
            ref = _label_at(reference, frame)
            bits = []
            if pred:
                text = "{} {}".format(label_name, pred)
                if conf is not None:
                    text += " {:.2f}".format(conf)
                bits.append(text)
            else:
                bits.append(label_name + " -")
            if ref:
                bits.append("ref {}".format(ref))
            events_now = [e for e in events if abs(e["frame"] - frame) <= 12]
            if events_now:
                e = min(events_now, key=lambda x: abs(x["frame"] - frame))
                bits.append("D {}".format(e["kind"]))
            if len(touch_frames) and abs(int(touch_frames[np.argmin(np.abs(touch_frames - frame))]) - frame) <= TOUCH_FADE:
                j = int(np.argmin(np.abs(touch_frames - frame)))
                bits.append("touch {}".format(touch_kind[j]))
            line2.set_text("   |   ".join(bits))
            label_big.set_text(pred if pred else "-")
            label_big.set_color(LABEL_COLORS.get(pred, "#333333"))

            active = [b["label"] for b in bands if b["start"] <= frame < b["end"] and b.get("label")]
            band_label.set_text("  ".join(dict.fromkeys(active)))
            for patch, band in zip(overlays, bands):
                patch.set_visible(band["start"] <= frame < band["end"])
            cursor.set_data([frame, frame], [0, 1])

            fig.canvas.restore_region(background)
            for artist in ax_dynamic:
                ax.draw_artist(artist)
            for text in (line1, line2):
                fig.draw_artist(text)
            tl.draw_artist(cursor)
            _write_png(fig, os.path.join(frames_dir, "f_{:05d}.png".format(k)))
        plt.close(fig)

        frames_n = len(indices)
        result = {"mp4": None, "gif": None, "frames": frames_n,
                  "pixels": (width_px, height_px), "seconds": 0.0,
                  "gif_bytes": None, "labels": labels}
        if mp4:
            if _ffmpeg():
                _encode_mp4(frames_dir, fps, mp4_path)
                result["mp4"] = mp4_path
            else:
                print("render: ffmpeg not on PATH, skipping MP4")
        if gif:
            if _ffmpeg():
                result["gif_bytes"] = _encode_gif(frames_dir, fps, gif_path, width_px,
                                                  gif_max_bytes, gif_colors)
            else:
                result["gif_bytes"] = _encode_gif_pillow(frames_dir, fps, gif_path)
            result["gif"] = gif_path
        result["seconds"] = time.time() - t_start
    return result
