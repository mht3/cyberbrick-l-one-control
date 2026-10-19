"""Read-only access to a finished LeRobot dataset, one whole episode at a time.

Built on LeRobotDatasetMetadata, never on LoneRecorder: the recorder resumes an
existing dataset for *writing* and finalizes it on close, which is the wrong
thing to do to a dataset a benchmark is only reading.

Frames are decoded with lone_data.playback.EpisodeVideo (PyAV/libdav1d) -- the
files are AV1 and cv2.VideoCapture cannot read them. Trackers want a whole
episode at once, so `frames()` decodes sequentially into one array: ~400 frames
of 360x640x3 uint8 is ~280 MB per view, fine to hold.
"""

import os
from dataclasses import dataclass

import numpy as np

import lone_data  # noqa: F401  (HF offline flags before lerobot)
from lone_data.features import ACTION, CAMERA_VIEWS, dataset_camera_keys
from lone_data.playback import EpisodeVideo

# A view is "still" when every actuator command has been zero for this many
# ticks. 3 ticks (120 ms) covers the ~50 ms command latency plus the motors
# spinning down; the analysis in servo_model measures the actual coast.
STILL_SETTLE_TICKS = 3


@dataclass(frozen=True)
class EpisodeRecord:
    index: int
    length: int
    video_paths: dict      # view name -> absolute .mp4 path
    from_timestamps: dict  # view name -> seconds into that file
    data_path: str         # parquet holding this episode's rows


class EpisodeIndex:
    """Every episode of one dataset: where its video and rows live, and readers."""

    def __init__(self, repo_id, root):
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        self.repo_id = repo_id
        self.root = str(root)
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.fps = int(meta.fps)
        keys = dataset_camera_keys(meta.features)
        # "observation.images.front" -> "front": the short names are what labels,
        # filenames and plots use.
        self.views = [k.rsplit(".", 1)[-1] for k in keys]
        self._keys = dict(zip(self.views, keys))
        h, w = meta.features[keys[0]]["shape"][:2]
        self.height, self.width = int(h), int(w)

        rows = meta.episodes
        self.episodes = []
        for i in range(len(rows["length"])):
            video_paths, from_ts = {}, {}
            for view, key in self._keys.items():
                rel = meta.video_path.format(
                    video_key=key,
                    chunk_index=int(rows[f"videos/{key}/chunk_index"][i]),
                    file_index=int(rows[f"videos/{key}/file_index"][i]),
                )
                video_paths[view] = os.path.join(self.root, rel)
                from_ts[view] = float(rows[f"videos/{key}/from_timestamp"][i])
            data_rel = meta.data_path.format(
                chunk_index=int(rows["data/chunk_index"][i]),
                file_index=int(rows["data/file_index"][i]),
            )
            self.episodes.append(
                EpisodeRecord(
                    index=int(rows["episode_index"][i]),
                    length=int(rows["length"][i]),
                    video_paths=video_paths,
                    from_timestamps=from_ts,
                    data_path=os.path.join(self.root, data_rel),
                )
            )
        self._by_index = {e.index: e for e in self.episodes}

    def __len__(self):
        return len(self.episodes)

    def episode(self, index):
        return self._by_index[index]

    def video(self, index, view):
        """A seekable (forward-biased) reader for one view of one episode."""
        ep = self.episode(index)
        return EpisodeVideo(ep.video_paths[view], ep.from_timestamps[view],
                            ep.length, self.fps)

    def frames(self, index, view):
        """(T,H,W,3) uint8 RGB, the whole episode, decoded once in order."""
        reader = self.video(index, view)
        try:
            return np.stack([reader.frame(i) for i in range(reader.length)])
        finally:
            reader.close()

    def actions(self, index):
        """(T,4) float32 commands, read straight from the parquet."""
        import pyarrow.parquet as pq

        ep = self.episode(index)
        table = pq.read_table(ep.data_path, columns=["episode_index", "frame_index", ACTION])
        mask = np.asarray(table["episode_index"]) == index
        order = np.argsort(np.asarray(table["frame_index"])[mask])
        actions = np.stack(
            [np.asarray(a, dtype=np.float32) for a in table[ACTION].to_pylist()]
        )[mask][order]
        if len(actions) != ep.length:
            raise ValueError(
                f"episode {index}: {len(actions)} action rows, metadata says {ep.length}"
            )
        return actions


def still_mask(actions, settle=STILL_SETTLE_TICKS):
    """True where the arm should be stationary: all of dims 0-2 zero for `settle`+1 ticks.

    The gripper (dim 3) is a position command and is ignored -- the arm can be
    still while the jaws open or close. Frame t counts as still only if t and the
    `settle` ticks before it all commanded zero motion, so the spin-down after a
    stop is excluded.
    """
    moving = np.any(np.asarray(actions)[:, :3] != 0, axis=1)
    still = ~moving
    out = still.copy()
    for k in range(1, settle + 1):
        out[k:] &= still[:-k]
        out[:k] = False
    return out


def duplicate_frames(frames, threshold=0.05):
    """True where a frame is an exact re-record of the previous one.

    The recorder takes each camera's latest frame on every 25 Hz tick. When a
    tick runs late and the next fires to catch up, both get the same frames: in
    the multiview eval set 21% (front) / 16% (side) of ticks are exact copies,
    15% in both views on the same tick -- far more coincidence than independent
    camera drop-outs would give, so it is the tick loop. A duplicate is a stale
    sample (its image is from an earlier tick), and its zero motion would corrupt
    any velocity estimate. `threshold` is the mean absolute difference in grey
    levels: copies are ~0, distinct frames of even a static scene are ~1.
    """
    f = np.asarray(frames, dtype=np.int16)
    diff = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    return np.concatenate([[False], diff < threshold])


__all__ = ["CAMERA_VIEWS", "EpisodeIndex", "EpisodeRecord", "still_mask",
           "duplicate_frames", "STILL_SETTLE_TICKS"]
