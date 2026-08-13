"""Read one episode's frames back from its LeRobot video file.

LeRobot's own reader can't be used mid-session: episode metadata is buffered
and the parquet writers stay open until finalize(), so both meta.episodes and
dataset[i] fail until the app exits. The .mp4 is complete after every
save_episode(), though, and several episodes share one file -- each occupying
the frame range starting at from_timestamp * fps.

Decoding is sequential with a one-frame cursor: AV1 seeks are unreliable
through OpenCV, so stepping forward reads the next frame and jumping backwards
reopens and skips. Playback and scrubbing are both forward-biased, so this is
cheap in practice.
"""

import cv2


class EpisodeVideo:
    def __init__(self, path, from_timestamp, length, fps):
        self.path = path
        self.length = length
        self.start_frame = int(round(from_timestamp * fps))
        self._cap = None
        self._pos = None  # index of the frame the cursor will return next

    def _open(self):
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open {self.path}")
        for _ in range(self.start_frame):
            if not self._cap.grab():
                raise RuntimeError(f"{self.path} ended before episode start")
        self._pos = 0

    def frame(self, index):
        """RGB uint8 frame `index` within this episode."""
        if not 0 <= index < self.length:
            raise IndexError(f"frame {index} out of range (0..{self.length - 1})")
        if self._cap is None or self._pos is None or index < self._pos:
            self.close()
            self._open()
        while self._pos < index:
            if not self._cap.grab():
                raise RuntimeError(f"{self.path} ended early at frame {self._pos}")
            self._pos += 1
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError(f"{self.path} could not decode frame {index}")
        self._pos += 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._pos = None
