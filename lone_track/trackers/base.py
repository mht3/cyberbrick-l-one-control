"""One interface over every tracker the benchmark compares.

    tracker.track(frames, queries) -> TrackResult

frames   (T,H,W,3) uint8 RGB, one view of one episode
queries  (N,3) float [t, x, y] -- the frame each point is given on, and where

Coordinates everywhere in lone_track are OpenCV pixels: the centre of the
top-left pixel is (0, 0). Each wrapper converts to and from its model's own
convention (tapnet's models put pixel centres at +0.5, CoTracker's resize uses
align_corners=True); tests/test_track_wrappers.py checks that a known shift
comes back as that shift, which is how a convention slip shows up.

Offline trackers see the whole episode and track both ways from each query.
Causal trackers (`causal = True`) start at the queries' frame and only ever see
frames up to the one they are estimating; frames before the start are NaN.
They also report per-frame compute time and `lag_frames` -- how many frames
after frame t arrives its estimate exists (0 for per-frame trackers, up to 7
for CoTracker3-online's 8-frame step).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackResult:
    xy: np.ndarray                  # (T,N,2) float, NaN where not estimated
    vis: np.ndarray                 # (T,N) bool, the model's own visibility call
    conf: np.ndarray                # (T,N) float in [0,1]: higher = trust it more
    latency_ms: np.ndarray = None   # (T,) compute per frame, causal trackers only
    lag_frames: np.ndarray = None   # (T,) causal trackers only
    extra: dict = field(default_factory=dict)  # e.g. SAM2 mask stats

    def save(self, path, **meta):
        arrays = {"xy": self.xy, "vis": self.vis, "conf": self.conf}
        if self.latency_ms is not None:
            arrays["latency_ms"] = self.latency_ms
        if self.lag_frames is not None:
            arrays["lag_frames"] = self.lag_frames
        for k, v in self.extra.items():
            arrays[f"extra_{k}"] = np.asarray(v)
        for k, v in meta.items():
            arrays[f"meta_{k}"] = np.asarray(v)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        extra = {k[len("extra_"):]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(xy=z["xy"], vis=z["vis"], conf=z["conf"],
                   latency_ms=z["latency_ms"] if "latency_ms" in z.files else None,
                   lag_frames=z["lag_frames"] if "lag_frames" in z.files else None,
                   extra=extra)


class Tracker:
    name = "base"
    causal = False

    def track(self, frames, queries):
        raise NotImplementedError

    @staticmethod
    def check_inputs(frames, queries):
        frames = np.asarray(frames)
        queries = np.asarray(queries, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(f"frames must be (T,H,W,3) uint8, got {frames.shape} {frames.dtype}")
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError(f"queries must be (N,3) [t,x,y], got {queries.shape}")
        if np.any(queries[:, 0] < 0) or np.any(queries[:, 0] >= len(frames)):
            raise ValueError("query frame index out of range")
        return frames, queries

    @staticmethod
    def causal_start(queries):
        """Causal trackers start every point on one frame: the queries must agree."""
        t0 = np.unique(queries[:, 0].astype(int))
        if len(t0) != 1:
            raise ValueError(f"causal tracker needs all queries on one frame, got {t0}")
        return int(t0[0])

    @staticmethod
    def empty(T, N):
        return (np.full((T, N, 2), np.nan, np.float32), np.zeros((T, N), bool),
                np.zeros((T, N), np.float32))


def cuda_timer():
    """A context-manager-free stopwatch that syncs CUDA, in ms."""
    import time

    import torch

    def now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() * 1e3

    return now
