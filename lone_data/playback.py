"""Read one episode's frames back from its LeRobot video file.

LeRobot's own reader can't be used mid-session: episode metadata is buffered
and the parquet writers stay open until finalize(), so both meta.episodes and
dataset[i] fail until the app exits. The .mp4 is complete after every
save_episode(), though, and several episodes share one file -- each occupying
the frame range starting at from_timestamp * fps.

Decoded with PyAV (libdav1d), not cv2.VideoCapture: opencv-python's bundled
FFmpeg tries hardware AV1 decoding and fails outright with "Failed to get
pixel format" on platforms without it, rather than falling back to software
decode. PyAV is already a lerobot dependency and decodes these files fine.

Decoding is sequential with a one-frame cursor: AV1 has no cheap frame-skip,
so stepping forward decodes and discards, and jumping backwards reopens and
skips. Playback and scrubbing are both forward-biased, so this is cheap in
practice.
"""

import av


class EpisodeVideo:
    def __init__(self, path, from_timestamp, length, fps):
        self.path = path
        self.length = length
        self.start_frame = int(round(from_timestamp * fps))
        self._container = None
        self._decoder = None
        self._pos = None  # index of the frame the cursor will return next

    def _open(self):
        self._container = av.open(self.path)
        self._decoder = self._container.decode(video=0)
        for _ in range(self.start_frame):
            if next(self._decoder, None) is None:
                raise RuntimeError(f"{self.path} ended before episode start")
        self._pos = 0

    def frame(self, index):
        """RGB uint8 frame `index` within this episode."""
        if not 0 <= index < self.length:
            raise IndexError(f"frame {index} out of range (0..{self.length - 1})")
        if self._decoder is None or self._pos is None or index < self._pos:
            self.close()
            self._open()
        while self._pos < index:
            if next(self._decoder, None) is None:
                raise RuntimeError(f"{self.path} ended early at frame {self._pos}")
            self._pos += 1
        av_frame = next(self._decoder, None)
        if av_frame is None:
            raise RuntimeError(f"{self.path} could not decode frame {index}")
        self._pos += 1
        return av_frame.to_ndarray(format="rgb24")

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = None
        self._decoder = None
        self._pos = None
