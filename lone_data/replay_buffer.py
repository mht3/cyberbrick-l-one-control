"""Zarr-backed, model-agnostic replay buffer for L-ONE demonstrations.

Schema (see README / plan for rationale):

    lone_dataset.zarr/
        data/img         uint8    (N,H,W,3)
        data/action       float32  (N,4)
        data/timestamp    float64  (N,)
        meta/episode_ends int64    (E,)

All episodes are concatenated along axis 0. Writes happen once per episode
(a single resize + bulk slice-assignment) rather than once per frame, since
per-row appends into a chunked array force a read-decompress-recompress of
the in-progress chunk on every write.
"""

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

ACTION_NAMES = [
    "base_motor_speed",
    "upper_arm_servo_speed",
    "lower_arm_servo_speed",
    "gripper_angle",
]
ACTION_DIM = len(ACTION_NAMES)

_IMG_CHUNK_LEN = 32
_VEC_CHUNK_LEN = 1024

_IMG_COMPRESSOR = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)


class LoneReplayBuffer:
    """Owns the on-disk Zarr dataset. One instance per collection session."""

    def __init__(self, zarr_path):
        self.zarr_path = zarr_path
        self._root = zarr.open_group(store=zarr_path, mode="a")
        self._data = self._root.require_group("data")
        self._meta = self._root.require_group("meta")

        if "episode_ends" in self._meta:
            self._episode_ends = self._meta["episode_ends"]
        else:
            self._episode_ends = self._meta.create_array(
                "episode_ends", shape=(0,), dtype="int64", chunks=(_VEC_CHUNK_LEN,)
            )

        self._img = self._data.get("img")
        self._action = self._data.get("action")
        self._timestamp = self._data.get("timestamp")

        # In-RAM buffer for the episode currently being recorded.
        self._ep_imgs = None
        self._ep_actions = None
        self._ep_timestamps = None
        self._ep_start_idx = None

    # -- shape info -------------------------------------------------

    @property
    def n_steps(self):
        return int(self._timestamp.shape[0]) if self._timestamp is not None else 0

    @property
    def n_episodes(self):
        return int(self._episode_ends.shape[0])

    @property
    def img_shape(self):
        return tuple(self._img.shape[1:]) if self._img is not None else None

    def _ensure_arrays(self, img_shape):
        """Lazily create data/img|action|timestamp once the camera's frame
        shape is known (first call to start_episode with a live frame)."""
        if self._img is not None:
            return
        h, w, c = img_shape
        self._img = self._data.create_array(
            "img",
            shape=(0, h, w, c),
            dtype="uint8",
            chunks=(_IMG_CHUNK_LEN, h, w, c),
            compressors=_IMG_COMPRESSOR,
        )
        self._action = self._data.create_array(
            "action", shape=(0, ACTION_DIM), dtype="float32", chunks=(_VEC_CHUNK_LEN, ACTION_DIM)
        )
        self._timestamp = self._data.create_array(
            "timestamp", shape=(0,), dtype="float64", chunks=(_VEC_CHUNK_LEN,)
        )

    # -- episode lifecycle --------------------------------------------

    def start_episode(self, img_shape):
        if self._ep_imgs is not None:
            raise RuntimeError("An episode is already being recorded")
        self._ensure_arrays(img_shape)
        if img_shape != self.img_shape:
            raise ValueError(
                f"Frame shape {img_shape} doesn't match dataset's existing shape {self.img_shape}"
            )
        self._ep_imgs = []
        self._ep_actions = []
        self._ep_timestamps = []
        self._ep_start_idx = self.n_steps
        return self._ep_start_idx

    def add_step(self, img, action, timestamp):
        """Buffers one (image, action, timestamp) triple in RAM, between start_episode() and end/discard."""
        if self._ep_imgs is None:
            raise RuntimeError("No episode is being recorded")
        if img.shape != self.img_shape:
            raise ValueError(f"Frame shape {img.shape} != dataset shape {self.img_shape}")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action shape {action.shape} != ({ACTION_DIM},)")
        self._ep_imgs.append(img)
        self._ep_actions.append(action)
        self._ep_timestamps.append(float(timestamp))

    @property
    def current_episode_len(self):
        return len(self._ep_timestamps) if self._ep_timestamps is not None else 0

    def end_episode(self):
        """Bulk-writes the buffered episode to disk and appends to meta/episode_ends; returns its length."""
        if self._ep_imgs is None:
            raise RuntimeError("No episode is being recorded")
        ep_len = len(self._ep_timestamps)
        if ep_len == 0:
            self._ep_imgs = self._ep_actions = self._ep_timestamps = None
            self._ep_start_idx = None
            return 0

        start = self._ep_start_idx
        end = start + ep_len
        self._img.resize((end,) + self.img_shape)
        self._action.resize((end, ACTION_DIM))
        self._timestamp.resize((end,))

        self._img[start:end] = np.stack(self._ep_imgs, axis=0)
        self._action[start:end] = np.stack(self._ep_actions, axis=0)
        self._timestamp[start:end] = np.asarray(self._ep_timestamps, dtype=np.float64)

        self._episode_ends.resize((self.n_episodes + 1,))
        self._episode_ends[-1] = end

        self._ep_imgs = self._ep_actions = self._ep_timestamps = None
        self._ep_start_idx = None
        return ep_len

    def discard_episode(self):
        """Drops the in-RAM buffer for the episode in progress; nothing was ever written to disk."""
        if self._ep_imgs is None:
            raise RuntimeError("No episode is being recorded")
        ep_len = len(self._ep_timestamps)
        self._ep_imgs = self._ep_actions = self._ep_timestamps = None
        self._ep_start_idx = None
        return ep_len

    @property
    def episode_in_progress(self):
        return self._ep_imgs is not None

    def delete_episode(self, episode_idx):
        """Removes one episode's rows, shifts later rows down to close the gap, and re-indexes
        meta/episode_ends. Returns the number of steps removed; MP4/metadata.json re-indexing is
        the caller's job (see lone_data/curation.py)."""
        if self.episode_in_progress:
            raise RuntimeError("Cannot delete an episode while one is being recorded")
        start, end = self.episode_bounds(episode_idx)
        removed = end - start
        n = self.n_steps
        tail_len = n - end
        if tail_len > 0:
            self._img[start:start + tail_len] = self._img[end:n]
            self._action[start:start + tail_len] = self._action[end:n]
            self._timestamp[start:start + tail_len] = self._timestamp[end:n]

        new_n = n - removed
        self._img.resize((new_n,) + self.img_shape)
        self._action.resize((new_n, ACTION_DIM))
        self._timestamp.resize((new_n,))

        ends = self._episode_ends[:]
        new_ends = np.delete(ends, episode_idx)
        new_ends[episode_idx:] -= removed
        self._episode_ends.resize((len(new_ends),))
        if len(new_ends):
            self._episode_ends[:] = new_ends
        return removed

    # -- read-only access for inspect/visualize scripts ----------------

    def episode_bounds(self, episode_idx):
        ends = self._episode_ends[:]
        if episode_idx < 0 or episode_idx >= len(ends):
            raise IndexError(f"episode {episode_idx} out of range (0..{len(ends) - 1})")
        start = 0 if episode_idx == 0 else int(ends[episode_idx - 1])
        end = int(ends[episode_idx])
        return start, end

    @property
    def episode_ends(self):
        return self._episode_ends[:] if self.n_episodes else np.array([], dtype=np.int64)

    @property
    def img(self):
        return self._img

    @property
    def action(self):
        return self._action

    @property
    def timestamp(self):
        return self._timestamp
