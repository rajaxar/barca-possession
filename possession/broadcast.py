"""The broadcast feed as a second ball sensor and as a director-intent diagnostic.

The broadcast recording is a second camera with its own tracker. It is not synchronised to the
tactical timeline; :func:`possession.data.load_broadcast` re-indexes its players and ball with
code-derived per-half offsets, and this module builds the two things the feed is actually good for:

* a **live mask** -- a per-tactical-frame flag saying the broadcast is looking at the same play as
  the tactical tracker right now (player clouds agree to within a metre). Cutaways, replays,
  close-ups and sparse views are all "not live": their coordinates are stale or partial and must
  not be fused.
* **ball fusion** -- fill the tactical ball only where the tactical tracker has lost it *and* the
  broadcast is live and its ball projects onto the pitch. On BAR-ATH this adds 5,919 graded frames
  (3.80 pp of coverage; 9.2% of the missing frames); the live gate matters, because a naive
  ball-present fusion would also import 937 frames of stale replay ball (E06).

The third function, :func:`director_fov`, recovers the visible pitch width from the broadcast
homography, i.e. the director's zoom. On BAR-ATH it is **not** a stoppage cue: conditional on a
valid homography, dead frames are slightly *wider* than live ones (median 40.2 m vs 38.8 m) and the
narrow-zoom AUC is 0.435. Homography validity is also not a live check -- during replays it can
stay "valid" while the scale is nonsense (E06).

What this module deliberately does **not** do: turn a cutaway into D. On BAR-ATH 78.9% of cutaway
time carries D in the shipped labels, but 12,880 frames (8.3% of graded) that the shipped labels
call live are also cutaways, and those frames are fast (mean player speed 1.91 m/s, 14% still):
the director cuts away during live play too, so the cue is a coverage hole, not a dead-ball signal
(E06; E106's premise). The mask is safe as a fusion gate; it is not safe as a label rule.

Report-only: nothing on the ``make_submission`` path imports this module (D-029), and no function
here writes a label.
"""
import numpy as np

from . import config
from . import structure
from .structure import PITCH_L, PITCH_W

# --- the live check (01_data_audit.md section 6, reproduced to 10 frames: 94,879 vs 94,869).
BC_LIVE_M = 1.0          # median nearest-neighbour player distance, metres
BC_MIN_PLAYERS = 5       # the audit's lower bound: sparse views fail by construction
TAC_MIN_PLAYERS = 8

# --- published video geometry (DATA.md / OVERVIEW.md: 1920 x 1080, 25 fps).
IMAGE_W = 1920.0
IMAGE_H = 1080.0

# --- a cutaway/replay stretch is at least this long; the audit's "unmatched stretches" are >= 800
# broadcast frames. Two seconds is the shortest stretch that can hold a stoppage cue.
CUTAWAY_MIN_FRAMES = 50

# --- a "full scene" broadcast view: this many tracked players means the camera is wide but the
# player cloud does not match the tactical view (replay / other angle), not a close-up.
FULL_SCENE_PLAYERS = 15


def _frame_cloud(cloud):
    cloud = np.asarray(cloud)
    return cloud[~np.isnan(cloud[:, 0])]


def cloud_score(tac_cloud, bc_cloud, tac_min=TAC_MIN_PLAYERS, bc_min=BC_MIN_PLAYERS):
    """Median nearest-neighbour distance (m) from the broadcast players to the tactical players.

    The audit's direction (broadcast -> tactical): how far each broadcast player is from the
    nearest tactical player. NaN when either cloud is too sparse to judge.
    """
    a = _frame_cloud(bc_cloud)
    c = _frame_cloud(tac_cloud)
    if len(a) < bc_min or len(c) < tac_min:
        return np.nan
    d = np.linalg.norm(a[:, None, :] - c[None, :, :], axis=2)
    return float(np.median(d.min(1)))


def live_mask(match, broadcast, metres=BC_LIVE_M, tac_min=TAC_MIN_PLAYERS,
              bc_min=BC_MIN_PLAYERS):
    """Per-tactical-frame broadcast-live mask.

    ``match`` is a :class:`possession.data.Match`; ``broadcast`` is the dict returned by
    :func:`possession.data.load_broadcast`. Returns ``dict(live, score, covered)``: ``live`` is
    True where a broadcast record exists, both clouds are dense enough, and their median distance
    is below ``metres``. ``score`` is NaN wherever the test cannot run.

    This is a *lower bound* on broadcast coverage: a sparse but genuine view fails the density
    test by construction (the audit's caveat), which is the right behaviour for fusion -- a view
    with fewer than five tracked players cannot place a ball reliably either.
    """
    P, bcP = np.asarray(match.P), np.asarray(broadcast["P"])
    covered = np.asarray(broadcast["covered"], bool)
    n = len(P)
    score = np.full(n, np.nan, np.float32)
    for f in np.where(covered)[0]:
        score[f] = cloud_score(P[f], bcP[f], tac_min=tac_min, bc_min=bc_min)
    live = covered & np.isfinite(score) & (score < metres)
    return dict(live=live, score=score, covered=covered.copy())


def cached_live():
    """The live mask, cached on disk (``bc-live-v1``) through the loader's cache machinery.

    The 160k-frame cloud scan takes ~12 s cold and ~0 s warm. The cache key is the hash of the four
    ATD files; the version in the name must be bumped if the alignment or the cloud test changes.
    Lives here rather than in :mod:`possession.data` so that the loader keeps no import of this
    report-only module (the constant-provenance graph requires it).
    """
    from . import data as _data

    def build():
        return dict(live=live_mask(_data.load_match(), _data.load_broadcast())["live"])

    paths = [_data.config.asset_path(k, s) for s in ("tactical", "broadcast")
             for k in ("atd_meta", "atd_raw")]
    return _data._cached("bc-live-v1", paths, build)["live"].astype(bool)


def cutaway_runs(live, min_frames=CUTAWAY_MIN_FRAMES):
    """Maximal runs of ``~live`` as an (k, 2) array of ``[start, stop)``, optionally length-filtered."""
    m = ~np.asarray(live, bool)
    if not m.any():
        return np.zeros((0, 2), np.int64)
    pad = np.concatenate(([False], m, [False]))
    d = np.diff(pad.astype(np.int8))
    runs = np.c_[np.where(d == 1)[0], np.where(d == -1)[0]]
    if min_frames:
        runs = runs[(runs[:, 1] - runs[:, 0]) >= min_frames]
    return runs.astype(np.int64)


def inside_pitch(points, length=PITCH_L, width=PITCH_W, margin=0.0):
    """Boolean mask of finite points inside the pitch rectangle (with an optional margin)."""
    p = np.asarray(points)
    return (np.isfinite(p[:, 0]) & (p[:, 0] >= -margin) & (p[:, 0] <= length + margin)
            & (p[:, 1] >= -margin) & (p[:, 1] <= width + margin))


def fuse_ball(match, broadcast, live, inside=True, margin=0.0):
    """Tactical ball with broadcast fills where the tactical tracker has no ball.

    A fill needs all three: the tactical ball is missing, the broadcast is live at that frame
    (``live_mask``), and the broadcast ball exists. With ``inside`` (default) the fill must also
    project inside the pitch: 16% of the otherwise-eligible broadcast balls land outside it, and
    the tactical tracker's out-of-pitch detections drive Baseline+'s out-of-bounds D, so importing
    a homography error as an out-of-bounds ball would be worse than leaving the gap.

    Returns ``dict(B, source, added, n_added)``. ``source`` is 0 for tactical, 1 for broadcast and
    -1 where neither has a ball. ``B`` is a copy: the caller's match is not modified.
    """
    B = np.array(match.B, dtype=np.float32, copy=True)
    bcB = np.asarray(broadcast["B"])
    live = np.asarray(live, bool)
    have = np.isfinite(bcB[:, 0])
    add = live & have & ~np.isfinite(B[:, 0])
    if inside:
        add &= inside_pitch(bcB, margin=margin)
    B[add] = bcB[add]
    source = np.full(len(B), -1, np.int8)
    source[np.isfinite(B[:, 0])] = 0
    source[add] = 1
    return dict(B=B, source=source, added=add, n_added=int(add.sum()))


def load_broadcast_homography(path=None):
    """Broadcast homography as ``dict(frame, H (n,3,3), valid, error)``.

    H maps normalised pitch coordinates ``(x / PITCH_L, y / PITCH_W, 1)`` to image pixels, the
    same convention as the tactical CSV (see :func:`possession.features.camera_look_at_x`).
    """
    import pandas as pd

    path = path or config.asset_path("homography", "broadcast")
    table = pd.read_csv(path)
    cols = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
    H = table[cols].to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    valid = table["valid"].to_numpy().astype(bool) & np.isfinite(H).all(axis=(1, 2))
    return dict(frame=table["frame"].to_numpy(), H=H, valid=valid,
                error=table["reprojection_error"].to_numpy(dtype=np.float32))


def _to_tactical(values, offsets, n_frames):
    """Re-index a broadcast-frame array onto the tactical timeline (same rule as the loader)."""
    out = np.full(n_frames, np.nan, np.float32)
    values = np.asarray(values, np.float32)
    for info in offsets.values():
        k = int(info["offset"])
        lo, hi = max(int(info["bc_range"][0]), -k), min(int(info["bc_range"][1]), n_frames - 1 - k)
        out[lo + k:hi + 1 + k] = values[lo:hi + 1]
    return out


def director_fov(homography=None, offsets=None, n_frames=None):
    """Visible pitch width (m) at the image centre row, on the tactical timeline.

    The director's zoom is the distance between the projected left and right image edges. Uses the
    published 1920 x 1080 frame (``IMAGE_W``). NaN where the homography is invalid. This is a
    camera-behaviour diagnostic, not a label input: on BAR-ATH it is essentially flat across D/live and can be
    nonsense on a replay whose homography is still marked valid.

    ``offsets`` is :func:`possession.data.broadcast_offsets`; when omitted it is imported lazily so
    this module needs no match data to be imported or unit-tested.
    """
    h = load_broadcast_homography() if homography is None else homography
    if offsets is None:
        from . import data
        offsets = data.broadcast_offsets()
    n_frames = config.N_VIDEO_FRAMES if n_frames is None else n_frames
    H, valid = h["H"], h["valid"]
    width = np.full(len(H), np.nan, np.float32)
    idx = np.where(valid)[0]
    if len(idx):
        Hi = np.linalg.inv(H[idx])
        edge = np.array([[0.0, IMAGE_H / 2, 1.0], [IMAGE_W - 1.0, IMAGE_H / 2, 1.0]])
        ph = np.einsum("nij,kj->nki", Hi, edge)
        xy = ph[..., :2] / ph[..., 2:3]
        width[idx] = np.linalg.norm((xy[:, 0] - xy[:, 1]) * np.array([PITCH_L, PITCH_W]), axis=1)
    return _to_tactical(width, offsets, n_frames)


def stats(match, broadcast, live):
    """Label-free summary used by the E06 write-up (no labels, no match GT).

    Returns the frame counts the card quotes: live/cutaway frames, the fusion gain, and the
    two-population split of the cutaway time by how many broadcast players it holds.
    """
    cov = np.asarray(broadcast["covered"], bool)
    bcP = np.asarray(broadcast["P"])
    tac_ball = np.isfinite(np.asarray(match.B)[:, 0])
    bc_ball = np.isfinite(np.asarray(broadcast["B"])[:, 0])
    n_bc = np.isfinite(bcP[:, :, 0]).sum(1)
    fused = fuse_ball(match, broadcast, live)
    fused_all = fuse_ball(match, broadcast, live, inside=False)
    graded = np.asarray(match.graded, bool) & ~structure.structure(match).forced_dead_mask
    return dict(
        covered=int((cov & graded).sum()),
        live=int((live & graded).sum()),
        cutaway=int((cov & ~live & graded).sum()),
        cutaway_full_scene=int((cov & ~live & (n_bc >= FULL_SCENE_PLAYERS) & graded).sum()),
        cutaway_sparse=int((cov & ~live & (n_bc < BC_MIN_PLAYERS) & graded).sum()),
        tactical_ball=int((tac_ball & graded).sum()),
        broadcast_ball_live=int((bc_ball & live & graded).sum()),
        added=int((fused["added"] & graded).sum()),
        added_raw=int((fused_all["added"] & graded).sum()),
        added_share_of_graded=float((fused["added"] & graded).sum()) / max(1, int(graded.sum())),
        added_share_of_missing=float((fused["added"] & graded).sum())
        / max(1, int(np.sum(~tac_ball & graded))),
    )
