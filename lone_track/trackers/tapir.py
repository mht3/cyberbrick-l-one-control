"""BootsTAPIR (offline, bidirectional) and online BootsTAPIR (causal, per frame).

PyTorch ports from google-deepmind/tapnet (tapnet/torch/tapir_model.py), used as
in colabs/torch_tapir_demo.ipynb and colabs/torch_causal_tapir_demo.ipynb.

Frames go in resized to `size` (default 256x256 -- the 640x360 frame is
squashed, as the reference demos do) and scaled to [-1, 1]. tapnet puts pixel
centres at +0.5 (its coordinates scale edge-to-edge), so points are shifted by
half a pixel on the way in and back on the way out; queries are (t, y, x).
Confidence is the demo's own: (1 - p_occluded) * (1 - p_far), visible when > 0.5.
"""

import numpy as np
import torch
import torch.nn.functional as F

from lone_track.trackers.base import TrackResult, Tracker, cuda_timer


def _to_model(xy, H, W, size):
    """OpenCV (x, y) -> tapnet (y, x) in the resized frame."""
    sh, sw = size
    return np.stack([(xy[..., 1] + 0.5) * sh / H, (xy[..., 0] + 0.5) * sw / W], -1)


def _from_model(xy_model, H, W, size):
    """tapnet (x, y) in the resized frame -> OpenCV (x, y). (Outputs are x-first.)"""
    sh, sw = size
    return np.stack([xy_model[..., 0] * W / sw - 0.5, xy_model[..., 1] * H / sh - 0.5], -1)


def _prep(frames, size, device):
    v = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float()
    v = F.interpolate(v, size, mode="bilinear", align_corners=False, antialias=True)
    return (v / 255 * 2 - 1).permute(0, 2, 3, 1)  # (T, h, w, 3)


def _confidence(occ, dist):
    return (1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(dist))


class BootsTAPIR(Tracker):
    name = "bootstapir"
    causal = False

    def __init__(self, checkpoint, device="cuda", size=(256, 256)):
        from tapnet.torch import tapir_model

        self.device = torch.device(device)
        self.size = tuple(size)
        self.model = tapir_model.TAPIR(pyramid_level=1)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.device).eval()
        if self.size != (256, 256):
            self.name = f"{self.name}_{self.size[0]}x{self.size[1]}"

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        T, H, W, _ = frames.shape
        video = _prep(frames, self.size, self.device)[None]
        q = np.column_stack([queries[:, 0], _to_model(queries[:, 1:], H, W, self.size)])
        q = torch.from_numpy(q.astype(np.float32)).to(self.device)[None]
        out = self.model(video, q, query_chunk_size=64)
        tracks = out["tracks"][0].cpu().numpy()          # (N, T, 2) x-first
        conf = _confidence(out["occlusion"][0], out["expected_dist"][0]).cpu().numpy()
        xy = _from_model(tracks, H, W, self.size).transpose(1, 0, 2)
        conf = conf.T
        for i, (t, x, y) in enumerate(queries):
            xy[int(t), i] = (x, y)
            conf[int(t), i] = 1.0
        return TrackResult(xy=xy.astype(np.float32), vis=conf > 0.5, conf=conf.astype(np.float32))


class OnlineBootsTAPIR(Tracker):
    name = "bootstapir_online"
    causal = True

    def __init__(self, checkpoint, device="cuda", size=(256, 256)):
        from tapnet.torch import tapir_model

        self.device = torch.device(device)
        self.size = tuple(size)
        self.model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(self.device).eval()
        if self.size != (256, 256):
            self.name = f"{self.name}_{self.size[0]}x{self.size[1]}"

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        t0 = self.causal_start(queries)
        T, H, W, _ = frames.shape
        N = len(queries)
        xy, vis, conf = self.empty(T, N)
        latency = np.full(T, np.nan, np.float32)
        lag = np.full(T, np.nan, np.float32)

        q = np.column_stack([np.zeros(N), _to_model(queries[:, 1:], H, W, self.size)])
        q = torch.from_numpy(q.astype(np.float32)).to(self.device)[None]
        now = cuda_timer()
        first = _prep(frames[t0:t0 + 1], self.size, self.device)[None]
        grids = self.model.get_feature_grids(first, is_training=False)
        qfeat = self.model.get_query_features(first, is_training=False, query_points=q,
                                              feature_grids=grids)
        state = self.model.construct_initial_causal_state(N, len(qfeat.resolutions) - 1)
        state = [{k: v.to(self.device) for k, v in s.items()} for s in state]

        for t in range(t0, T):
            t_start = now()
            frame = _prep(frames[t:t + 1], self.size, self.device)[None]
            grids = self.model.get_feature_grids(frame, is_training=False)
            traj = self.model.estimate_trajectories(
                frame.shape[-3:-1], is_training=False, feature_grids=grids,
                query_features=qfeat, query_points_in_video=None, query_chunk_size=64,
                causal_context=state, get_causal_context=True,
            )
            state = traj["causal_context"]
            c = _confidence(traj["occlusion"][-1], traj["expected_dist"][-1])[0, :, 0]
            tr = traj["tracks"][-1][0, :, 0].cpu().numpy()
            latency[t] = now() - t_start
            xy[t] = _from_model(tr, H, W, self.size)
            conf[t] = c.cpu().numpy()
            vis[t] = conf[t] > 0.5
            lag[t] = 0
        xy[t0] = queries[:, 1:]
        vis[t0] = True
        conf[t0] = 1.0
        return TrackResult(xy=xy, vis=vis, conf=conf, latency_ms=latency, lag_frames=lag)
