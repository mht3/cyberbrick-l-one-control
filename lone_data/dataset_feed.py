"""One recorded episode, served frame by frame in place of a camera.

deploy_policy_training_data_replay.py drives the arm from a dataset instead of a
webcam, which is only a meaningful test if the frames it hands the policy are the
frames training saw. So this wraps LeRobotDataset rather than decoding the .mp4
itself: same decoder, same dtype, same scaling, and no second implementation to
drift out of agreement with the one the training run used.

That is also why this is not lone_data/playback.py, which reads the video with
OpenCV. playback.py exists to scrub episodes *during* a collection session, when
LeRobot's own reader cannot be used at all -- the parquet writers are still open
and dataset[i] raises. Here the dataset is finished on disk, so the real reader
works and is the more faithful choice.

Frames come back from LeRobot as float32 CHW in [0,1]. They were stored as uint8
and divided by 255 on read, so multiplying by 255 and rounding recovers the
original bytes exactly -- which is what lets the replay push them through
deploy_policy.py's unmodified uint8 camera path without changing what the policy
sees. verify_roundtrip() checks that rather than trusting it.
"""

import numpy as np
import torch

from lone_data.features import CAMERA_KEY

# The preview redraws the cursor's frame ~20x/s and the control tick reads one or
# two more, so a handful of entries removes essentially all repeat decoding. Full
# episodes are not cached: 406 frames of 360x640 float32 is 1.1 GB.
_CACHE_SIZE = 8


class DatasetFeed:
    """Read-only access to episode `episode` of a LeRobot dataset, by frame index.

    Indices are episode-local: 0 .. length-1, whichever episode was selected.
    """

    def __init__(self, repo_id, root, episode=0):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.root = root
        self.episode = episode
        # episodes=[episode] re-bases indices to 0, so the cursor is a frame number
        # within the episode and not an offset into the whole dataset.
        self.dataset = LeRobotDataset(repo_id, root=root, episodes=[episode])
        self.length = len(self.dataset)
        if not self.length:
            raise ValueError(f"episode {episode} of {repo_id} has no frames")
        self.fps = self.dataset.meta.fps

        h, w = self.dataset.meta.features[CAMERA_KEY]["shape"][:2]
        # Named to match CameraStream, so the GUI's camera readouts work unchanged.
        self.actual_height, self.actual_width = int(h), int(w)

        self.task = self.dataset[0].get("task", "")
        self._cache = {}

    @property
    def total_episodes(self):
        return self.dataset.meta.total_episodes

    def _check(self, index):
        if not 0 <= index < self.length:
            raise IndexError(f"frame {index} out of range (0..{self.length - 1})")
        return index

    def tensor(self, index):
        """The image exactly as LeRobot delivers it: float32 CHW in [0,1]."""
        self._check(index)
        cached = self._cache.get(index)
        if cached is None:
            cached = self.dataset[index][CAMERA_KEY]
            if len(self._cache) >= _CACHE_SIZE:
                # Plain FIFO. Access here is a forward-moving cursor plus its
                # lookahead, so recency and insertion order coincide.
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = cached
        return cached

    def rgb(self, index):
        """uint8 HWC RGB -- the form the deploy path carries frames in."""
        chw = self.tensor(index).permute(1, 2, 0).numpy()
        return np.clip(np.round(chw * 255.0), 0, 255).astype(np.uint8)

    def bgr(self, index):
        """uint8 HWC BGR, for the OpenCV-shaped callers (preview, resize_keep_aspect)."""
        return self.rgb(index)[:, :, ::-1].copy()

    def action(self, index):
        """The action recorded at this frame, in raw command units."""
        self._check(index)
        return np.asarray(self.dataset[index]["action"], dtype=np.float32).reshape(-1)

    def verify_roundtrip(self, index=0):
        """Max abs difference between the deploy path's tensor and LeRobot's own.

        The replay is only worth anything if feeding a dataset frame through
        deploy_policy.py's camera path reconstructs the training input. Expected
        to be exactly 0.0; anything else is a silent preprocessing mismatch, and
        finding those is what this script is for.
        """
        from lone_data.features import resize_keep_aspect

        rgb = resize_keep_aspect(self.bgr(index), (self.actual_height, self.actual_width))
        through_deploy = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return float((through_deploy - self.tensor(index)).abs().max())

    def close(self):
        self._cache.clear()
