"""Every tracker wrapper recovers a known sub-pixel motion, in OpenCV pixel coordinates.

The classic bugs in wrapping point trackers are (x, y) vs (y, x), and the
half-pixel offset between tapnet's +0.5 pixel-centre convention and OpenCV's
integer one. Both show up as a constant offset or a mirrored track on a
synthetic textured patch moving at a known velocity over a flat background.

Needs the `lone-track` env (scripts/setup_track_env.sh), a GPU and the
checkpoints in outputs/track/ckpts; each tracker skips cleanly otherwise, so
`pytest tests/` still passes in the plain `cyberbrick` env.
"""

import os

import numpy as np
import pytest

H, W, T = 360, 640, 24
VEL = np.array([1.3, -0.7])      # px/frame
START = np.array([250.0, 190.0])  # top-left of the 120x120 patch at t=0
PATCH = 120
QUERY_OFFSETS = np.array([[30.0, 30.0], [85.0, 40.0], [60.0, 95.0]])  # inside the patch
# SAM2 is prompted with the points' bounding box (+12 px), so its points must span
# the object, as body + both fingertips span the gripper; a box around part of a
# flat patch is a featureless sub-region that SAM2 cannot follow.
SPAN_OFFSETS = np.array([[8.0, 8.0], [112.0, 20.0], [60.0, 112.0]])


def synthetic_video(solid=False):
    """A 120x120 patch drifting at VEL over grey. `solid` makes it one flat colour:
    an unambiguous object for SAM2 (on random texture SAM2 segments some sub-blob
    and its centroid wanders), but nothing a point tracker can lock onto."""
    import cv2

    rng = np.random.default_rng(0)
    tex = rng.integers(0, 256, (PATCH // 4, PATCH // 4, 3)).astype(np.uint8)
    tex = cv2.resize(tex, (PATCH, PATCH), interpolation=cv2.INTER_CUBIC)
    if solid:
        tex[:] = (200, 40, 40)
    canvas = np.full((H, W, 3), 128, np.uint8)
    frames = []
    for t in range(T):
        x0, y0 = START + VEL * t
        M = np.float32([[1, 0, x0], [0, 1, y0]])
        img = cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(np.full((PATCH, PATCH), 255, np.uint8), M, (W, H)) > 127
        frame = canvas.copy()
        frame[mask] = img[mask]
        frames.append(frame)
    offsets = SPAN_OFFSETS if solid else QUERY_OFFSETS
    queries = np.column_stack([np.zeros(len(offsets)), START + offsets])
    truth = START + offsets[None] + VEL[None, None] * np.arange(T)[:, None, None]
    return np.stack(frames), queries.astype(np.float32), truth


# (name, tolerance in px). SAM2 tracks the region's centroid (on the solid
# patch), so under pure translation it is exact up to mask edge noise; the
# others track the points.
# The tapnet models run at 256x256, a 2.5x downscale of the 640-px width: their
# sub-pixel error at native resolution (~0.3-0.7 model px, sign varying with the
# scene -- noise, not a convention offset) becomes ~1-2 display px here.
CASES = [
    ("lk", 0.5),
    ("cotracker3_offline", 1.0),
    ("cotracker3_online", 1.0),
    ("bootstapir", 1.5),
    ("bootstapir_online", 2.5),
    ("tapnextpp", 1.5),
    ("sam2_small_online", 1.0),
    ("sam2_large", 1.0),
]


@pytest.mark.parametrize("name,tol", CASES)
def test_tracker_recovers_known_motion(name, tol):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    if name != "lk":
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        pkg = {"cotracker": "cotracker", "bootstapir": "tapnet", "tapnextpp": "tapnet",
               "sam2": "sam2"}[next(k for k in ("cotracker", "bootstapir", "tapnextpp", "sam2")
                                     if k in name)]
        pytest.importorskip(pkg)
    from lone_track import trackers

    if not os.path.isdir(trackers.CKPT_DIR) and name != "lk":
        pytest.skip("no checkpoints (run scripts/setup_track_env.sh)")
    tracker = trackers.build(name)
    frames, queries, truth = synthetic_video(solid=name.startswith("sam2"))
    res = tracker.track(frames, queries)

    assert res.xy.shape == (T, len(queries), 2)
    err = np.linalg.norm(res.xy - truth, axis=-1)
    assert np.nanmax(err[1:]) < tol, f"{name}: max error {np.nanmax(err):.2f}px\n{err.round(2)}"
    # A mirrored (y, x) track would also fail the error bound; this names it.
    assert np.all(np.sign(np.diff(res.xy[:, 0, 0])) >= 0), "x should increase every frame"
    if tracker.causal:
        assert res.lag_frames is not None and np.all(res.lag_frames >= 0)
