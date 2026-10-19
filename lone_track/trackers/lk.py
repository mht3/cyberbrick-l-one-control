"""Pyramidal Lucas-Kanade, frame to frame: the no-deep-learning baseline.

Causal and cheap (well under a millisecond per frame). Each step is checked
forward-backward; a point whose round trip misses by more than FB_MAX px is
marked not visible and held at its last position, which is how LK fails on a
textureless white gripper -- the benchmark should show how quickly.
"""

import cv2
import numpy as np

from lone_track.trackers.base import TrackResult, Tracker, cuda_timer

FB_MAX = 2.0
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class LucasKanade(Tracker):
    name = "lk"
    causal = True

    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        t0 = self.causal_start(queries)
        T = len(frames)
        N = len(queries)
        xy, vis, conf = self.empty(T, N)
        latency = np.full(T, np.nan, np.float32)
        lag = np.full(T, np.nan, np.float32)
        now = cuda_timer()

        pts = queries[:, 1:].astype(np.float32).reshape(-1, 1, 2)
        prev = cv2.cvtColor(frames[t0], cv2.COLOR_RGB2GRAY)
        xy[t0], vis[t0], conf[t0], lag[t0], latency[t0] = queries[:, 1:], True, 1.0, 0, 0
        for t in range(t0 + 1, T):
            t_start = now()
            cur = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK_PARAMS)
            back, st_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK_PARAMS)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            ok = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < FB_MAX)
            pts = np.where(ok[:, None, None], nxt, pts)
            latency[t] = now() - t_start
            xy[t] = pts.reshape(-1, 2)
            vis[t] = ok
            conf[t] = np.exp(-fb / FB_MAX) * ok
            lag[t] = 0
            prev = cur
        return TrackResult(xy=xy, vis=vis, conf=conf, latency_ms=latency, lag_frames=lag)
