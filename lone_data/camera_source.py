"""Choosing where camera frames come from -- a local device or a remote sender.

collect_data.py and deploy_policy.py both need the same behavior: default to a
local camera, offer the network receiver as an alternative, survive a camera that
will not open, and say so in the video pane. That behavior lives here rather than
in either GUI, because two copies of "which camera am I using" is exactly the kind
of thing that drifts until the two tools disagree.

Everything here is Tk-free. `placeholder_frame` returns an RGB array and the caller
wraps it, which is also what makes it testable on a machine with no display.
"""

import cv2
import numpy as np

from lone_data.camera import CameraStream
from lone_data.camera_stream import CameraStreamReceiver

# Sentinel for the source dropdown; any other value is a local cv2 index.
REMOTE_SOURCE = "remote"
PROBE_MAX = 6

_BACKGROUND = (32, 32, 32)
_TITLE_COLOR = (220, 220, 220)
_HINT_COLOR = (150, 150, 150)


def source_label(source, remote_port):
    if source == REMOTE_SOURCE:
        return f"Remote (:{remote_port})"
    return f"Camera {source}"


def parse_label(label):
    """Inverse of source_label. Returns None if the text is not a known source."""
    if label.startswith("Remote"):
        return REMOTE_SOURCE
    try:
        return int(label.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return None


def available_sources(active_source=None, active_open=False):
    """Local indices that actually deliver a frame, plus the remote receiver.

    Probed on demand rather than at startup: a missing index takes cv2 about a
    second to give up on, and a dropdown is opened far less often than an app is
    launched.

    `active_source`/`active_open` describe a camera the caller already holds open,
    which would fail to open a second time for the probe and must still be listed.
    """
    sources = []
    for i in range(PROBE_MAX):
        if active_open and i == active_source:
            sources.append(i)
            continue
        cap = cv2.VideoCapture(i)
        # isOpened() alone is not enough: a UVC camera exposes a metadata node
        # beside its capture node, and some backends open a device that then never
        # delivers. Require an actual frame.
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            sources.append(i)
    sources.append(REMOTE_SOURCE)
    return sources


def pick_initial_source(preferred, remote_port=None):
    """The source to open at startup: the preferred local camera if it works, else
    any other local camera, else the remote receiver.

    Defaulting blindly to index 0 means a machine with no camera opens with an error
    even when a perfectly good camera sits at index 2, or when the operator was
    always going to use the network stream. Probing is cheap here: an index with no
    device node fails immediately rather than timing out.
    """
    if preferred == REMOTE_SOURCE:
        return REMOTE_SOURCE
    local = [s for s in available_sources() if s != REMOTE_SOURCE]
    if preferred in local:
        return preferred
    return local[0] if local else REMOTE_SOURCE


def open_source(source, remote_port, width=1280, height=720, fps=30):
    """Construct and start a camera for `source`. Raises if it cannot be opened."""
    if source == REMOTE_SOURCE:
        camera = CameraStreamReceiver(port=remote_port)
    else:
        # Opened on the caller's thread so macOS's camera-permission prompt gets a run loop.
        camera = CameraStream(source, width, height, fps)
    camera.start()
    return camera


def placeholder_frame(image_size, camera, source, remote_port, display_width=640):
    """RGB image for the video pane when no frames are arriving.

    Both GUIs deliberately start without a working camera so a source can be picked
    from the toolbar, which means the pane has to say why it is empty rather than
    sitting blank.
    """
    store_h, store_w = image_size
    w = display_width
    h = max(1, int(store_h * (display_width / store_w)))

    if camera is None:
        title, hint = "No camera", "Choose a source from the Camera dropdown"
    elif source == REMOTE_SOURCE:
        title, hint = "Waiting for sender", f"listening on port {remote_port}"
    else:
        title, hint = "No frames yet", source_label(source, remote_port)

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = _BACKGROUND
    cv2.putText(canvas, title, (24, h // 2 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, _TITLE_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, hint, (24, h // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, _HINT_COLOR, 1, cv2.LINE_AA)
    return canvas
