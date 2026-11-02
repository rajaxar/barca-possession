"""Paths and file-name resolution. The only place that knows where the match data lives."""
import os
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(HERE)

DEFAULT_DATA_DIR = os.path.join(SOLUTION_DIR, "..", "more-than-a-hack-2027", "data")
DEFAULT_REPO_DIR = os.path.join(SOLUTION_DIR, "..", "more-than-a-hack-2027")

N_VIDEO_FRAMES = 199319  # published by the organizers (games.py); submission length

# Candidate file names per asset, in preference order. The tactical video ships as
# "BAR-ATH_tactical-001.mp4" on some machines; the organizer's games.py expects
# "BAR-ATH_tactical.mp4". Both are accepted; files are never renamed.
ASSET_CANDIDATES = {
    "tactical": {
        "video": ["BAR-ATH_tactical.mp4", "BAR-ATH_tactical-001.mp4"],
        "atd_meta": ["BAR-ATH_tactical_atd.xml"],
        "atd_raw": ["BAR-ATH_tactical_atd.txt"],
        "homography": ["BAR-ATH_tactical_homography.csv"],
        "smart_tagging": ["BAR-ATH_tactical_smart_tagging.xml"],
    },
    "broadcast": {
        "video": ["BAR-ATH_broadcast.mp4"],
        "atd_meta": ["BAR-ATH_broadcast_atd.xml"],
        "atd_raw": ["BAR-ATH_broadcast_atd.txt"],
        "homography": ["BAR-ATH_broadcast_homography.csv"],
    },
}


def data_dir():
    return os.path.abspath(os.environ.get("MTAH_DATA_DIR", DEFAULT_DATA_DIR))


def repo_dir():
    """Organizer repository (validator, scorer). Override with MTAH_REPO_DIR."""
    return os.path.abspath(os.environ.get("MTAH_REPO_DIR", DEFAULT_REPO_DIR))


def asset_path(asset, source="tactical"):
    """First existing candidate for the asset; if none exists, the preferred name (for error messages)."""
    names = ASSET_CANDIDATES[source][asset]
    folder = os.path.join(data_dir(), source)
    for name in names:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return os.path.join(folder, names[0])


def cache_dir():
    """Where derived arrays are cached. Gitignored (.cache/) and never inside the data bundle."""
    path = os.environ.get("MTAH_CACHE_DIR") or os.path.join(SOLUTION_DIR, ".cache")
    os.makedirs(path, exist_ok=True)
    return path


def tmp_path(target):
    """Unique sibling of `target` (same extension) to write to before os.replace, so concurrent
    writers of one cache never share a temporary file."""
    stem, ext = os.path.splitext(target)
    return "{}.tmp{}-{}{}".format(stem, os.getpid(), uuid.uuid4().hex[:8], ext)
