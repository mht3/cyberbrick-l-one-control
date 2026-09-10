"""Choosing where camera frames come from -- a local device or a remote sender.

collect_data.py and deploy_policy.py both need the same behavior: default to a
local camera, offer the network receiver as an alternative, survive a camera that
will not open, and say so in the video pane. That behavior lives here rather than
in either GUI, because two copies of "which camera am I using" is exactly the kind
of thing that drifts until the two tools disagree.

There are up to two camera slots -- Cam1 (front) and Cam2 (side) -- and each one
picks its own source independently. Slots are numbered, so a remote sender for
slot i connects to `remote_port(base, i)`: one stream_camera.py process per
camera, one port each, and the one-camera case still uses the base port exactly
as it did before Cam2 existed.

Everything here is Tk-free. `placeholder_frame` returns an RGB array and the caller
wraps it, which is also what makes it testable on a machine with no display.
"""

import contextlib
import os
import sys

import cv2
import numpy as np

from lone_data.camera import CameraStream
from lone_data.camera_stream import CameraStreamReceiver

# Sentinel for the source dropdown; any other value is a local cv2 index.
REMOTE_SOURCE = "remote"
PROBE_MAX = 6
# Indices are contiguous, so this many misses in a row means there are no more.
MAX_MISSES = 2

_BACKGROUND = (32, 32, 32)
_TITLE_COLOR = (220, 220, 220)
_HINT_COLOR = (150, 150, 150)


@contextlib.contextmanager
def _quiet_stderr():
    """Silence the probe's console noise for the duration of the block.

    Asking cv2 for a camera index that does not exist is how you find out it does
    not exist, and every backend complains on the way -- V4L2 prints "can't open
    camera by index", AVFoundation prints "out device of bound (0-1): 2" and
    "camera failed to properly initialize!". Those come from OpenCV's C++ layer
    writing to fd 2 directly, so redirecting sys.stderr in Python does nothing;
    the file descriptor itself has to be pointed elsewhere.

    Only wraps probing. A camera that fails to open for real still reports through
    the GUI log, which is where the operator is actually looking.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def remote_port(base_port, slot=0):
    """The port slot `slot` receives its remote stream on.

    Slot 0 keeps the base port, so a single-camera setup is wired exactly as
    before; each further slot takes the next port up.
    """
    return base_port + slot


def source_label(source, port):
    if source == REMOTE_SOURCE:
        return f"Remote (:{port})"
    return f"Camera {source}"


def parse_label(label):
    """Inverse of source_label. Returns None if the text is not a known source."""
    if label.startswith("Remote"):
        return REMOTE_SOURCE
    try:
        return int(label.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


def available_sources(open_sources=(), include_remote=True):
    """Local indices that actually deliver a frame, plus optionally the receiver.

    Probed on demand rather than at startup, and quietly: see _quiet_stderr.

    Probing stops after MAX_MISSES consecutive failures *once at least one camera
    has been found*, rather than always walking to PROBE_MAX -- on a machine with
    one camera at index 0 that is three probes rather than six, while a machine
    whose cameras start higher up is still found.

    `open_sources` are the indices the caller already holds open -- one per camera
    slot -- which would fail to open a second time for the probe and must still be
    listed. `include_remote` is False when the app was not launched to receive a
    stream -- offering a receiver nothing will ever connect to is just a dead menu
    entry.
    """
    held = {s for s in open_sources if s != REMOTE_SOURCE and s is not None}
    sources = []
    misses = 0
    with _quiet_stderr():
        for i in range(PROBE_MAX):
            # The early exit only applies once something has been found. Stopping
            # before that assumes cameras start at index 0, and a machine whose
            # only cameras sit at 2 and 3 (a disabled internal camera, a hub that
            # enumerates late) would report having none at all.
            if misses >= MAX_MISSES and sources:
                break
            if i in held:
                sources.append(i)
                misses = 0
                continue
            cap = cv2.VideoCapture(i)
            # isOpened() alone is not enough: a UVC camera exposes a metadata node
            # beside its capture node, and some backends open a device that then
            # never delivers. Require an actual frame.
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                sources.append(i)
                misses = 0
            else:
                misses += 1
    if include_remote:
        sources.append(REMOTE_SOURCE)
    return sources


def pick_initial_sources(preferred, slots, include_remote=True):
    """One source per slot at startup: distinct local cameras, else the receiver.

    `preferred` is the per-slot wish list from the command line, padded or
    truncated to `slots`. A slot whose preference is unavailable takes the next
    local camera nothing else claimed, and falls back to the remote receiver when
    there is none -- which is also the whole of the "two cameras plugged in means
    two views" default, since probing returns both and each slot takes one.

    Defaulting blindly to index 0 means a machine with no camera opens with an
    error even when a perfectly good camera sits at index 2, or when the operator
    was always going to use the network stream. Probing is cheap here: an index
    with no device node fails immediately rather than timing out.
    """
    local = available_sources(include_remote=False)
    chosen, taken = [], set()
    for slot in range(slots):
        want = preferred[slot] if slot < len(preferred) else None
        if want == REMOTE_SOURCE:
            chosen.append(REMOTE_SOURCE)
            continue
        if want in local and want not in taken:
            taken.add(want)
            chosen.append(want)
            continue
        spare = next((s for s in local if s not in taken), None)
        if spare is not None:
            taken.add(spare)
            chosen.append(spare)
        elif include_remote:
            chosen.append(REMOTE_SOURCE)
        else:
            # Never None: the dropdown shows this value and _open_camera reports
            # it, so "index None" would reach both. An index that does not exist
            # is at least a truthful thing to fail on.
            chosen.append(want if want is not None else slot)
    return chosen


def detect_camera_count(maximum, prefer_remote=False):
    """How many camera views to open at startup: as many as there are, capped.

    Two views are the default because that is what the arm is set up with, but a
    machine with one camera must still open with one rather than a dead second
    pane. With --remote-camera the local probe says nothing about how many senders
    will dial in, so the cap is taken at face value.
    """
    if prefer_remote:
        return maximum
    found = len(available_sources(include_remote=False))
    return max(1, min(found, maximum))


def open_source(source, port, width=1280, height=720, fps=30):
    """Construct and start a camera for `source`. Raises if it cannot be opened."""
    if source == REMOTE_SOURCE:
        camera = CameraStreamReceiver(port=port)
    else:
        # Opened on the caller's thread so macOS's camera-permission prompt gets a run loop.
        camera = CameraStream(source, width, height, fps)
    camera.start()
    return camera


def placeholder_frame(image_size, camera, source, port, display_width=640, name=None):
    """RGB image for the video pane when no frames are arriving.

    Both GUIs deliberately start without a working camera so a source can be picked
    from the toolbar, which means the pane has to say why it is empty rather than
    sitting blank. `name` is the slot it stands in for ("Cam2 (side)"), so with two
    panes on screen an empty one still says which camera is missing.
    """
    store_h, store_w = image_size
    w = display_width
    h = max(1, int(store_h * (display_width / store_w)))

    if camera is None:
        title, hint = "No camera", "Choose a source from the Camera dropdown"
    elif source == REMOTE_SOURCE:
        title, hint = "Waiting for sender", f"listening on port {port}"
    else:
        title, hint = "No frames yet", source_label(source, port)
    if name:
        title = f"{name} -- {title}"

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = _BACKGROUND
    _put_fitted(canvas, title, 24, h // 2 - 8, 0.9, _TITLE_COLOR, 2)
    _put_fitted(canvas, hint, 24, h // 2 + 24, 0.5, _HINT_COLOR, 1)
    return canvas


def _put_fitted(canvas, text, x, y, scale, color, thickness):
    """Draw `text` shrunk until it fits the canvas, so nothing runs off the edge.

    The pane is as narrow as 440px with two views on screen, and the text grew a
    slot-name prefix -- at a fixed scale "Cam1 (front) -- Waiting for sender" lost
    its last word, which is the half that says what is being waited for.
    """
    room = canvas.shape[1] - x - 12
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if tw <= room:
            break
        scale -= 0.05
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)
