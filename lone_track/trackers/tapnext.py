"""TAPNext++ (causal, per frame) from google-deepmind/tapnet.

Uses the inner TAPNext module the way TAPNextPP.track_frame does
(tapnet/tapnextpp/votsp2026/model.py) -- 256x256 squashed input in [-1, 1],
fp16 autocast, recurrent state carried frame to frame -- but converts
coordinates with tapnet's +0.5 pixel-centre convention, like the TAPIR wrapper.
The model's tracks come out (y, x).

TAPNext++ was fine-tuned on 1024-frame sequences, so unlike the original TAPNext
it should not degrade after ~150 frames -- the drift curve will show it.
"""

import numpy as np
import torch
import torch.nn.functional as F

from lone_track.trackers.base import TrackResult, Tracker, cuda_timer
from lone_track.trackers.tapir import _to_model

MODEL_SIZE = 256


class TAPNextPP(Tracker):
    name = "tapnextpp"
    causal = True

    def __init__(self, checkpoint, device="cuda"):
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP as Wrapper

        self.device = torch.device(device)
        self.model = Wrapper.from_checkpoint(checkpoint, device=self.device)._model

    def _frame(self, frame):
        v = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float()
        v = F.interpolate(v, (MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
        return (v / 127.5 - 1.0).permute(0, 2, 3, 1)[None]  # (1, 1, S, S, 3)

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        t0 = self.causal_start(queries)
        T, H, W, _ = frames.shape
        N = len(queries)
        xy, vis, conf = self.empty(T, N)
        latency = np.full(T, np.nan, np.float32)
        lag = np.full(T, np.nan, np.float32)

        q = np.column_stack([np.zeros(N), _to_model(queries[:, 1:], H, W, (MODEL_SIZE, MODEL_SIZE))])
        q = torch.from_numpy(q.astype(np.float32)).to(self.device)[None]
        now = cuda_timer()
        state = None
        for t in range(t0, T):
            t_start = now()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                if state is None:
                    tracks, _, vis_logits, state = self.model(video=self._frame(frames[t]),
                                                              query_points=q)
                else:
                    tracks, _, vis_logits, state = self.model(video=self._frame(frames[t]),
                                                              state=state)
            yx = tracks[0, 0].float().cpu().numpy()
            p = torch.sigmoid(vis_logits[0, 0, :, 0].float()).cpu().numpy()
            latency[t] = now() - t_start
            xy[t, :, 0] = yx[:, 1] * W / MODEL_SIZE - 0.5
            xy[t, :, 1] = yx[:, 0] * H / MODEL_SIZE - 0.5
            conf[t] = p
            vis[t] = p > 0.5
            lag[t] = 0
        xy[t0] = queries[:, 1:]
        vis[t0] = True
        conf[t0] = 1.0
        return TrackResult(xy=xy, vis=vis, conf=conf, latency_ms=latency, lag_frames=lag)
