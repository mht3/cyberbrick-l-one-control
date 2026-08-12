"""Background-thread webcam capture for the L-ONE data collector.

The capture device is opened on the caller's thread (macOS AVFoundation
needs a real run loop pumping for the camera-permission prompt, which a
bare background thread doesn't provide) -- only the continuous `.read()`
pumping happens on the background thread.
"""

import threading
import time

import cv2


class CameraStream:
    def __init__(self, index, width=1280, height=720, fps=30):
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps

        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        # Actual negotiated size -- the camera may not honor the request exactly.
        self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._lock = threading.Lock()
        self._latest = None  # (frame, monotonic_ts)
        self._running = False
        self._thread = None

        self._frame_count = 0
        self._measure_start = None

    def start(self):
        self._running = True
        self._measure_start = time.monotonic()
        self._frame_count = 0
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
                self._latest = (frame, ts)
                self._frame_count += 1

    def get_latest(self):
        """Returns (frame, monotonic_ts) for the latest frame, or None if nothing captured yet."""
        with self._lock:
            return self._latest

    @property
    def measured_fps(self):
        elapsed = time.monotonic() - self._measure_start if self._measure_start else 0
        if elapsed <= 0:
            return 0.0
        with self._lock:
            count = self._frame_count
        return count / elapsed

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._cap.release()
