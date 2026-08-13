"""LeRobotDataset writer for L-ONE demonstrations.

A plain LeRobotDataset (CODEBASE_VERSION v3.0) -- LeRobot owns episode
indexing, timestamps, video encoding and all metadata. Nothing custom is
written alongside it.

Timestamps are LeRobot's own frame_index/fps grid: add_frame() computes them
and the recorder paces itself to make that grid true, rather than recording
whenever it can and back-filling wall-clock times. Real capture timing goes to
the session log instead, so the dataset stays standard.
"""

import json
import os
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.features import (
    ACTION_DIM,
    CAMERA_KEY,
    DEFAULT_IMAGE_SIZE,
    ROBOT_TYPE,
    STATE_DIM,
    lone_features,
)




def _read_info(root):
    """The dataset's meta/info.json, or None if this isn't a dataset directory."""
    path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _first_task(tasks):
    """LeRobot stores an episode's tasks as a list (one episode may carry
    several); this recorder writes exactly one per episode."""
    if isinstance(tasks, (list, tuple)):
        return tasks[0] if tasks else None
    return tasks


def has_saved_episodes(root):
    """Whether `root` holds a dataset with at least one saved episode.

    Lets a caller ask "is there anything to review?" without constructing a
    LoneRecorder -- which opens the dataset for writing, starts an image writer,
    and deletes the directory outright when it holds metadata but no episodes.
    """
    info = _read_info(root)
    return info is not None and info.get("total_episodes", 0) > 0


def _open_local(repo_id, root):
    """Opens an existing dataset from disk. Anything missing locally is a local
    problem -- with the Hub disabled (see lone_data/__init__.py) LeRobot's
    download fallback surfaces as a confusing network error, so say so plainly."""
    try:
        return LeRobotDataset(repo_id, root=root)
    except Exception as e:
        raise RuntimeError(
            f"Could not open the dataset at {root}: {e}\n"
            "It looks incomplete -- an interrupted session can leave metadata without data. "
            "Delete that directory or pass a different --repo-id to start a new one."
        ) from e


class LoneRecorder:
    """Owns the dataset for one collection session. Create once, then
    start_episode/add_frame/finish_episode per demonstration, close() at exit."""

    def __init__(self, repo_id, root, fps, image_size=DEFAULT_IMAGE_SIZE, image_writer_threads=8):
        self.repo_id = repo_id
        self.root = root
        self.fps = fps
        self.image_size = tuple(image_size)
        self._episode_open = False
        self._closed = False
        self.episodes = []          # one record per saved episode, for playback
        self.episode_actions = {}   # episode_index -> (T,4) float32, this session only

        features = lone_features(self.image_size)
        info = _read_info(root)

        # create() writes meta/info.json before the first episode is saved, so a
        # session that exited early (no camera, no link, immediate quit) leaves a
        # directory with metadata but zero episodes. Resuming into that makes
        # LeRobotDataset treat the missing data as "not downloaded yet" and reach
        # for the Hub. Only a dataset with at least one saved episode is resumable.
        if info is not None and info.get("total_episodes", 0) > 0:
            # Same resume path lerobot_record.py uses: plain construction gives
            # back a writable dataset, the image writer just has to be restarted.
            self.dataset = _open_local(repo_id, root)
            self.dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)
            self._check_compatible(features)
            self.resumed = True
            self.episodes = self._load_episode_index()
        else:
            if info is not None:
                shutil.rmtree(root)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=root,
                robot_type=ROBOT_TYPE,
                use_videos=True,
                image_writer_threads=image_writer_threads,
            )
            self.resumed = False
            self.episodes = []

    def _check_compatible(self, features):
        existing = self.dataset.meta.features
        for key, spec in features.items():
            if key not in existing:
                raise ValueError(f"Existing dataset at {self.root} has no feature {key!r}")
            if tuple(existing[key]["shape"]) != tuple(spec["shape"]):
                raise ValueError(
                    f"Existing dataset's {key} shape {tuple(existing[key]['shape'])} "
                    f"!= {tuple(spec['shape'])} -- use a different --repo-id"
                )
        if self.dataset.meta.fps != self.fps:
            raise ValueError(
                f"Existing dataset was recorded at {self.dataset.meta.fps} fps, not {self.fps} "
                "-- use a different --repo-id"
            )

    @property
    def num_episodes(self):
        return self.dataset.meta.total_episodes

    @property
    def num_frames(self):
        return self.dataset.meta.total_frames

    # -- episode index for playback ---------------------------------------
    # LeRobot buffers episode metadata and leaves the parquet writers open
    # until finalize(), so mid-session neither meta.episodes nor dataset[i]
    # is readable -- not even from a second read-only handle, since the
    # parquet files have no footer yet. The video files, though, are complete
    # after each save_episode(). Recording where each episode landed as it is
    # saved is what makes playback possible before quitting.

    def _load_episode_index(self):
        """Episode records for a dataset being resumed (its metadata is on disk)."""
        episodes = self.dataset.meta.episodes
        if episodes is None:
            return []
        out = []
        for i in range(len(episodes["length"])):
            out.append(
                {
                    "index": int(episodes["episode_index"][i]),
                    "length": int(episodes["length"][i]),
                    "task": _first_task(episodes["tasks"][i]),
                    "video_path": self._video_path(
                        int(episodes[f"videos/{CAMERA_KEY}/chunk_index"][i]),
                        int(episodes[f"videos/{CAMERA_KEY}/file_index"][i]),
                    ),
                    "from_timestamp": float(episodes[f"videos/{CAMERA_KEY}/from_timestamp"][i]),
                }
            )
        return out

    def _video_path(self, chunk_index, file_index):
        rel = self.dataset.meta.video_path.format(
            video_key=CAMERA_KEY, chunk_index=chunk_index, file_index=file_index
        )
        return os.path.join(self.dataset.root, rel)

    def _record_saved_episode(self, task, actions):
        """Captures where the episode just written landed, from latest_episode."""
        ep = self.dataset.meta.latest_episode

        def first(key):
            v = ep[key]
            return v[0] if isinstance(v, (list, tuple)) else v

        self.episodes.append(
            {
                "index": int(first("episode_index")),
                "length": int(first("length")),
                "task": task,
                "video_path": self._video_path(
                    int(first(f"videos/{CAMERA_KEY}/chunk_index")),
                    int(first(f"videos/{CAMERA_KEY}/file_index")),
                ),
                "from_timestamp": float(first(f"videos/{CAMERA_KEY}/from_timestamp")),
            }
        )
        # Actions are tiny (4 floats/frame) and the parquet is unreadable until
        # finalize(), so keep this session's in RAM for the playback overlay.
        self.episode_actions[int(first("episode_index"))] = np.stack(actions)

    # -- episode lifecycle -------------------------------------------------

    def start_episode(self):
        if self._episode_open:
            raise RuntimeError("An episode is already being recorded")
        self._episode_open = True
        self._frames = 0
        self._pending_actions = []
        self._pending_task = None

    def add_frame(self, rgb, action, task):
        """rgb: (H,W,3) uint8 already resized to image_size. action: (4,) float32."""
        if not self._episode_open:
            raise RuntimeError("No episode is being recorded")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Action shape {action.shape} != ({ACTION_DIM},)")
        if rgb.shape != (*self.image_size, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Frame {rgb.shape}/{rgb.dtype} != {(*self.image_size, 3)}/uint8")

        # Always zeros -- L-ONE measures nothing. See STATE_NAMES in features.py.
        state = np.zeros(STATE_DIM, dtype=np.float32)
        self.dataset.add_frame({CAMERA_KEY: rgb, OBS_STATE: state, ACTION: action, "task": task})
        self._pending_actions.append(action)
        self._pending_task = task
        self._frames += 1

    @property
    def current_episode_len(self):
        return self._frames if self._episode_open else 0

    def finish_episode(self):
        """Writes the buffered episode. Returns its length; 0 saves nothing."""
        if not self._episode_open:
            raise RuntimeError("No episode is being recorded")
        self._episode_open = False
        if self._frames == 0:
            self.discard_episode_buffer()
            return 0
        self.dataset.save_episode()
        self._record_saved_episode(self._pending_task, self._pending_actions)
        return self._frames

    def discard_episode(self):
        if not self._episode_open:
            raise RuntimeError("No episode is being recorded")
        self._episode_open = False
        dropped = self._frames
        self.discard_episode_buffer()
        return dropped

    def discard_episode_buffer(self):
        self._frames = 0
        try:
            self.dataset.clear_episode_buffer()
        except Exception:
            pass  # nothing buffered yet, or already cleared

    # -- teardown ----------------------------------------------------------

    def close(self):
        """Idempotent. finalize() is the important part: without it the parquet
        footers are never written and the dataset can't be loaded back."""
        if self._closed:
            return
        self._closed = True
        if self._episode_open:
            self._episode_open = False
            self.discard_episode_buffer()
        try:
            self.dataset.stop_image_writer()
        except Exception:
            pass
        self.dataset.finalize()
