"""CoTracker3, offline (bidirectional, 60-frame windows) and online (16/8 windows).

The predictor classes in cotracker/predictor.py threshold visibility to a bool;
the benchmark needs the raw probabilities (confidence AUROC), so the sparse path
is reproduced here around the same `model.forward` call.

CoTracker resizes frames to 384x512 with align_corners=True and scales points by
(S-1)/(W-1) -- i.e. it uses the same pixel-centre convention as OpenCV, so no
half-pixel shift is needed on the way in or out.

Online mode is genuinely windowed: the model only advances in steps of 8 frames,
each call seeing the last 16. Frame t's first estimate therefore exists only
once the step containing it completes, up to 7 frames later. That first
estimate is what a controller would have had, so it is what is recorded --
later windows refine earlier frames, but too late to act on.
"""

import numpy as np
import torch
import torch.nn.functional as F

from lone_track.trackers.base import TrackResult, Tracker, cuda_timer

SUPPORT_GRID = 6  # the predictor's default: a 6x6 grid over the frame for context


def _grid(n, shape, device):
    from cotracker.models.core.model_utils import get_points_on_a_grid
    return get_points_on_a_grid(n, shape, device=device)


def _local_grid(queries, k, radius):
    """k x k points within +-radius px of each query, on the query's frame."""
    if k <= 0:
        return np.zeros((0, 3), np.float32)
    offs = np.linspace(-radius, radius, k)
    dx, dy = np.meshgrid(offs, offs)
    d = np.stack([dx.ravel(), dy.ravel()], 1)
    out = []
    for t, x, y in queries:
        pts = np.column_stack([np.full(len(d), t), x + d[:, 0], y + d[:, 1]])
        out.append(pts)
    return np.concatenate(out).astype(np.float32)


class CoTracker3Offline(Tracker):
    name = "cotracker3_offline"
    causal = False

    def __init__(self, checkpoint, device="cuda", local_grid=0, local_radius=8.0,
                 vis_threshold=0.9):
        from cotracker.models.build_cotracker import build_cotracker

        self.device = torch.device(device)
        self.model = build_cotracker(checkpoint, offline=True, window_len=60).to(self.device).eval()
        self.shape = tuple(self.model.model_resolution)  # (H, W)
        self.local_grid = local_grid
        self.local_radius = local_radius
        self.vis_threshold = vis_threshold
        if local_grid:
            self.name = f"{self.name}_lg{local_grid}"

    def _scale(self, H, W):
        return np.array([(self.shape[1] - 1) / (W - 1), (self.shape[0] - 1) / (H - 1)], np.float32)

    @torch.no_grad()
    def _run(self, video, queries):
        """video (1,T,3,h,w) resized; queries (1,N,3) in resized px -> tracks, vis, conf."""
        tracks, vis, conf, _ = self.model.forward(video=video, queries=queries, iters=6)
        return tracks, vis, conf

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        T, H, W, _ = frames.shape
        N = len(queries)
        q_all = np.concatenate([queries, _local_grid(queries, self.local_grid, self.local_radius)])
        scale = self._scale(H, W)

        video = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float()
        video = F.interpolate(video, self.shape, mode="bilinear", align_corners=True)[None]
        q = torch.from_numpy(q_all).to(self.device)[None].clone()
        q[..., 1:] *= torch.from_numpy(scale).to(self.device)
        grid = _grid(SUPPORT_GRID, self.shape, self.device)
        q_fwd = torch.cat([q, torch.cat([torch.zeros_like(grid[..., :1]), grid], 2)], 1)

        tracks, vis, conf = self._run(video, q_fwd)
        # Backward pass over the time-reversed video for frames before each query.
        q_bwd = q_fwd.clone()
        q_bwd[..., 0] = T - 1 - q_bwd[..., 0]
        tb, vb, cb = self._run(video.flip(1), q_bwd)
        tb, vb, cb = tb.flip(1), vb.flip(1), cb.flip(1)
        before = (torch.arange(T, device=self.device)[:, None] < q_fwd[0, :, 0][None])[None]
        tracks = torch.where(before[..., None], tb, tracks)
        vis = torch.where(before, vb, vis)
        conf = torch.where(before, cb, conf)

        n = len(q_all)
        xy = (tracks[0, :, :n] / torch.from_numpy(scale).to(self.device)).cpu().numpy()
        p_vis = vis[0, :, :n].cpu().numpy()
        score = (vis[0, :, :n] * conf[0, :, :n]).cpu().numpy()
        # The query frame itself is given, not estimated.
        for i, (t, x, y) in enumerate(q_all):
            xy[int(t), i] = (x, y)
            p_vis[int(t), i] = 1.0
            score[int(t), i] = 1.0
        return TrackResult(
            xy=xy[:, :N].astype(np.float32),
            vis=p_vis[:, :N] > self.vis_threshold,
            conf=score[:, :N].astype(np.float32),
            extra={"local_xy": xy[:, N:].astype(np.float32)} if self.local_grid else {},
        )


class CoTracker3Online(Tracker):
    name = "cotracker3_online"
    causal = True

    def __init__(self, checkpoint, device="cuda", vis_threshold=0.6):
        from cotracker.models.build_cotracker import build_cotracker

        self.device = torch.device(device)
        self.model = build_cotracker(checkpoint, offline=False, window_len=16).to(self.device).eval()
        self.shape = tuple(self.model.model_resolution)
        self.step = self.model.window_len // 2
        self.vis_threshold = vis_threshold

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        t0 = self.causal_start(queries)
        T, H, W, _ = frames.shape
        N = len(queries)
        xy, vis, conf = self.empty(T, N)
        latency = np.full(T, np.nan, np.float32)
        lag = np.full(T, np.nan, np.float32)
        scale = torch.tensor([(self.shape[1] - 1) / (W - 1), (self.shape[0] - 1) / (H - 1)],
                             device=self.device)

        q = torch.from_numpy(queries).to(self.device)[None].clone()
        q[..., 0] = 0  # relative to the stream start
        q[..., 1:] *= scale
        grid = _grid(SUPPORT_GRID, self.shape, self.device)
        q = torch.cat([q, torch.cat([torch.zeros_like(grid[..., :1]), grid], 2)], 1)
        self.model.init_video_online_processing()

        # The model's own protocol (cotracker3_online.forward, online_demo.py): call k
        # processes stream frames [k*step, k*step + window) and stores them at those
        # absolute indices, so it can only run once the window's last frame exists.
        # The first call therefore waits for 16 frames; after that each call adds 8
        # new frames. A short final window is padded with its last frame, as the
        # model itself pads, and the padding's outputs are ignored.
        stream = frames[t0:]
        L, S = len(stream), self.model.window_len
        now = cuda_timer()
        done, k = 0, 0
        while done < L:
            start = k * self.step
            end = min(start + S, L)
            window = stream[start:end]
            if len(window) < S:
                window = np.concatenate([window, np.repeat(window[-1:], S - len(window), 0)])
            t_start = now()
            chunk = torch.from_numpy(window).to(self.device).permute(0, 3, 1, 2).float()
            chunk = F.interpolate(chunk, self.shape, mode="bilinear", align_corners=True)[None]
            tracks, v, c, _ = self.model(video=chunk, queries=q, iters=6, is_online=True)
            elapsed = now() - t_start
            new = range(done, end)
            for s in new:
                t = t0 + s
                xy[t] = (tracks[0, s, :N] / scale).cpu().numpy()
                score = (v[0, s, :N] * c[0, s, :N]).cpu().numpy()
                conf[t] = score
                vis[t] = score > self.vis_threshold
                lag[t] = (end - 1) - s
                latency[t] = elapsed / max(1, len(new))
            done, k = end, k + 1
        xy[t0] = queries[:, 1:]
        vis[t0] = True
        conf[t0] = 1.0
        return TrackResult(xy=xy, vis=vis, conf=conf, latency_ms=latency, lag_frames=lag)
