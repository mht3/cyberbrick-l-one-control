"""Background-thread webcam capture for the L-ONE data collector.

The capture device is opened on the caller's thread (macOS AVFoundation needs a
real run loop pumping for the camera-permission prompt, which a bare background
thread doesn't provide) -- only the continuous `.read()` pumping happens on the
background thread.

Frames carry a sequence number so the recorder can tell a fresh frame from a
repeat: if the camera can't sustain the dataset's fps, recording anyway would
silently write duplicate frames under distinct timestamps.
"""

import threading
import time

import cv2

_FPS_WINDOW = 60  # frames used for the rolling rate estimate


class CameraStream:
    def __init__(self, index, width=1280, height=720, fps=30):
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        # Without this, most UVC webcams default to raw YUYV, which at 1280x720 exceeds
        # USB2 bandwidth and silently caps the driver at ~10fps regardless of the fps
        # requested below. MJPG uses the camera's onboard hardware compression instead.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Actual negotiated size -- the camera may not honor the request exactly.
        self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._lock = threading.Lock()
        self._latest = None  # (frame, monotonic_ts, seq)
        self._running = False
        self._thread = None

        self._seq = 0
        self._recent = []  # monotonic timestamps of the last _FPS_WINDOW frames

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = time.monotonic()
            with self._lock:
                self._seq += 1
                self._latest = (frame, ts, self._seq)
                self._recent.append(ts)
                if len(self._recent) > _FPS_WINDOW:
                    del self._recent[: len(self._recent) - _FPS_WINDOW]

    def get_latest(self):
        """Returns (frame, monotonic_ts, seq), or None if nothing captured yet."""
        with self._lock:
            return self._latest

    @property
    def measured_fps(self):
        """Rolling capture rate over the last _FPS_WINDOW frames."""
        with self._lock:
            recent = self._recent
            if len(recent) < 2:
                return 0.0
            span = recent[-1] - recent[0]
            return (len(recent) - 1) / span if span > 0 else 0.0

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._cap.release()
