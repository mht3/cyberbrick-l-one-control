"""SAM2.1 video segmentation, used as a gripper *region* tracker.

Prompted on the query frame with the labelled points as positive clicks plus
their bounding box (padded by BOX_MARGIN px) -- points alone on a white gripper
attached to a white arm tend to grab the whole arm. The mask is propagated with
SAM2's memory, forward only (`causal=True`) or both ways.

SAM2 gives a mask, not points, so each query's estimate is the mask centroid
plus that point's offset from the centroid on the prompt frame. That is crude
under rotation, and is reported as such; SAM2's other uses here are the mask
itself (sampling a point cluster on the gripper for the point trackers) and as
a sanity check (is a tracked point still inside the gripper?).

SAM2's loader reads JPEG folders; the frames are already decoded, so the
loader is bypassed with an equivalent tensor (resize to the model's square
input, /255, ImageNet mean/std) instead of re-encoding to JPEG.
"""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from lone_track.trackers.base import TrackResult, Tracker, cuda_timer

BOX_MARGIN = 12.0
CONFIGS = {
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
}
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


class SAM2Video(Tracker):
    def __init__(self, checkpoint, size="large", device="cuda", causal=False):
        from sam2.build_sam import build_sam2_video_predictor

        self.device = torch.device(device)
        self.predictor = build_sam2_video_predictor(CONFIGS[size], checkpoint, device=self.device)
        self.causal = causal
        self.name = f"sam2_{size}" + ("_online" if causal else "")

    def _images(self, frames):
        S = self.predictor.image_size
        mean = torch.tensor(IMG_MEAN, device=self.device)[:, None, None]
        std = torch.tensor(IMG_STD, device=self.device)[:, None, None]
        out = []
        for chunk in np.array_split(frames, max(1, len(frames) // 64)):
            v = torch.from_numpy(chunk).to(self.device).permute(0, 3, 1, 2).float() / 255
            v = F.interpolate(v, (S, S), mode="bilinear", align_corners=False, antialias=True)
            out.append((v - mean) / std)
        return torch.cat(out)

    @contextlib.contextmanager
    def _video(self, frames):
        import sam2.sam2_video_predictor as svp

        images = self._images(frames)
        H, W = frames.shape[1:3]
        original = svp.load_video_frames
        svp.load_video_frames = lambda **_: (images, H, W)
        try:
            yield self.predictor.init_state(video_path="<in-memory>")
        finally:
            svp.load_video_frames = original

    @torch.no_grad()
    def track(self, frames, queries):
        frames, queries = self.check_inputs(frames, queries)
        t0 = self.causal_start(queries)
        T, H, W, _ = frames.shape
        N = len(queries)
        pts = queries[:, 1:]
        box = np.concatenate([pts.min(0) - BOX_MARGIN, pts.max(0) + BOX_MARGIN])
        box = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1]).astype(np.float32)

        centroid = np.full((T, 2), np.nan, np.float32)
        area = np.zeros(T, np.float32)
        score = np.zeros(T, np.float32)
        latency = np.full(T, np.nan, np.float32) if self.causal else None
        mask_t0 = None
        now = cuda_timer()
        with torch.autocast("cuda", dtype=torch.bfloat16), self._video(frames) as state:
            self.predictor.add_new_points_or_box(
                state, frame_idx=t0, obj_id=1, points=pts, labels=np.ones(N, np.int32), box=box,
            )
            directions = [False] if self.causal else [False, True]
            for reverse in directions:
                last = now()
                for f, _, logits in self.predictor.propagate_in_video(
                        state, start_frame_idx=t0, reverse=reverse):
                    lg = logits[0, 0].float()
                    m = (lg > 0).cpu().numpy()
                    if latency is not None:
                        t_now = now()
                        latency[f] = t_now - last
                        last = t_now
                    area[f] = m.sum()
                    score[f] = torch.sigmoid(lg.max()).item()
                    if m.any():
                        ys, xs = np.nonzero(m)
                        centroid[f] = (xs.mean(), ys.mean())
                    if f == t0:
                        mask_t0 = m
            self.predictor.reset_state(state)

        if np.isnan(centroid[t0]).any():
            raise RuntimeError("SAM2 produced an empty mask on the prompt frame")
        offsets = pts - centroid[t0]
        xy = centroid[:, None, :] + offsets[None]
        xy[t0] = pts
        vis = np.repeat((area > 0)[:, None], N, 1)
        conf = np.repeat(score[:, None], N, 1)
        conf[t0] = 1.0
        lag = None
        if self.causal:
            xy[:t0] = np.nan
            vis[:t0] = False
            conf[:t0] = 0
            lag = np.where(np.arange(T) >= t0, 0.0, np.nan).astype(np.float32)
        return TrackResult(
            xy=xy.astype(np.float32), vis=vis, conf=conf.astype(np.float32),
            latency_ms=latency, lag_frames=lag,
            extra={"centroid": centroid, "area": area, "mask_t0": mask_t0, "box": box},
        )
