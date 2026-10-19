"""The per-episode, per-view tracker inputs every benchmark run shares.

For fairness every tracker gets the same queries, built once from the labels
and saved:

  t_q      the first labelled frame where all keypoints are *visible* in this
           view (usually frame 0). Causal trackers start here; nothing at or
           before t_q is scored, for any tracker.
  kp_xy    the labelled keypoints on t_q, (3,2) in KEYPOINTS order.
  mask     SAM2's gripper mask on t_q, prompted with those points + their box.
  cluster  K points sampled inside the (eroded) mask with a fixed seed -- for
           the "cluster" strategy: track many surface points on the rigid
           gripper and carry the keypoints by a robust similarity fit, instead
           of trusting a single tracked point.
"""

import os

import numpy as np

from lone_track.labels import KEYPOINTS

N_CLUSTER = 24
ERODE_PX = 2


def first_full_frame(labels, episode, view, keypoints=KEYPOINTS):
    for f in labels.labelled_frames(episode):
        entries = [labels.get(episode, f, view, kp) for kp in keypoints]
        if all(e is not None and e["vis"] == "visible" for e in entries):
            return f, np.array([e["xy"] for e in entries], np.float32)
    return None, None


def sample_cluster(mask, n=N_CLUSTER, erode=ERODE_PX, seed=0):
    """n well-spread points inside the eroded mask (farthest-point sampling)."""
    import cv2

    m = mask.astype(np.uint8)
    if erode:
        eroded = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
        if eroded.sum() >= n:
            m = eroded
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    rng = np.random.default_rng(seed)
    chosen = [rng.integers(len(pts))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(min(n, len(pts)) - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(pts - pts[chosen[-1]], axis=1))
    return pts[chosen]


class MaskPrompter:
    """SAM2.1 image predictor, prompted exactly like the SAM2 video tracker."""

    def __init__(self, checkpoint, size="large", device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        from lone_track.trackers.sam2 import CONFIGS

        self.predictor = SAM2ImagePredictor(build_sam2(CONFIGS[size], checkpoint, device=device))

    def __call__(self, image, points):
        import torch

        from lone_track.trackers.sam2 import BOX_MARGIN

        H, W = image.shape[:2]
        box = np.concatenate([points.min(0) - BOX_MARGIN, points.max(0) + BOX_MARGIN])
        box = np.clip(box, 0, [W - 1, H - 1, W - 1, H - 1])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=points, point_labels=np.ones(len(points)), box=box,
                multimask_output=False,
            )
        return masks[0] > 0, float(scores[0])


def query_path(out_dir, episode, view):
    return os.path.join(out_dir, f"ep{episode:03d}_{view}.npz")


def build_queries(labels, index, out_dir, prompter, episodes=None):
    """Write one query file per labelled (episode, view); returns what was skipped."""
    os.makedirs(out_dir, exist_ok=True)
    skipped = []
    for ep in episodes if episodes is not None else labels.episodes():
        for view in index.views:
            t_q, kp = first_full_frame(labels, ep, view)
            if t_q is None:
                skipped.append((ep, view, "no labelled frame with all keypoints visible"))
                continue
            image = index.video(ep, view).frame(t_q)
            mask, score = prompter(image, kp)
            cluster = sample_cluster(mask)
            np.savez_compressed(query_path(out_dir, ep, view), t_q=t_q, kp_xy=kp, mask=mask,
                                mask_score=score, cluster=cluster)
    return skipped


def load_queries(out_dir, episode, view):
    path = query_path(out_dir, episode, view)
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}
