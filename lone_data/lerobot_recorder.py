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


def _first(value):
    """Unwrap a latest_episode field, which holds one-element lists per column."""
    return value[0] if isinstance(value, (list, tuple)) else value


def has_saved_episodes(root):
    """Whether `root` holds a dataset with at least one saved episode.

    Lets a caller ask "is there anything to review?" without constructing a
    LoneRecorder -- which opens the dataset for writing, starts an image writer,
    and deletes the directory outright when it holds metadata but no episodes.
    """
    info = _read_info(root)
    return info is not None and info.get("total_episodes", 0) > 0


def _open_local(repo_id, root, image_writer_threads):
    """Reopens an existing dataset for appending.

    lerobot 0.6 replaced "construct, then start_image_writer()" with a resume()
    classmethod that builds the DatasetWriter itself; plain construction now gives
    a read-only view and start_image_writer no longer exists on the dataset.

    Anything missing locally is a local problem -- with the Hub disabled (see
    lone_data/__init__.py) LeRobot's download fallback surfaces as a confusing
    network error, so say so plainly."""
    try:
        return LeRobotDataset.resume(
            repo_id, root=root, image_writer_threads=image_writer_threads
        )
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
        self.last_warning = None    # set by finish_episode; the GUI surfaces it

        features = lone_features(self.image_size)
        info = _read_info(root)

        # create() writes meta/info.json before the first episode is saved, so a
        # session that exited early (no camera, no link, immediate quit) leaves a
        # directory with metadata but zero episodes. Resuming into that makes
        # LeRobotDataset treat the missing data as "not downloaded yet" and reach
        # for the Hub. Only a dataset with at least one saved episode is resumable.
        if info is not None and info.get("total_episodes", 0) > 0:
            # LeRobotDataset.resume() is 0.6's writable-reopen path.
            self.dataset = _open_local(repo_id, root, image_writer_threads)
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
        self.last_warning = self._check_saved_video_span()
        self._record_saved_episode(self._pending_task, self._pending_actions)
        return self._frames

    def _check_saved_video_span(self):
        """Verify the episode just saved occupies exactly length/fps of video.

        lerobot *defines* to_timestamp as from_timestamp + length/fps when it
        writes episode metadata, and asserts it later in delete_episodes(). Frames
        that outlive a discarded take break that: they are encoded ahead of the
        next episode's own frames, so its segment runs longer than its frame count
        and every following from_timestamp shifts with it.

        Nothing reads the video back during collection, so the damage is silent --
        it surfaces much later as an assertion inside an unrelated operation, long
        after the session that caused it. Checking here names the episode while the
        operator can still act on it. Returns a message, or None when consistent.
        """
        ep = self.dataset.meta.latest_episode
        try:
            span = float(_first(ep[f"videos/{CAMERA_KEY}/to_timestamp"])) - float(
                _first(ep[f"videos/{CAMERA_KEY}/from_timestamp"])
            )
            length = int(_first(ep["length"]))
            index = int(_first(ep["episode_index"]))
        except (KeyError, TypeError, IndexError):
            return None  # metadata shape changed; not worth failing a save over

        expected = length / self.fps
        if abs(span - expected) <= 1.0 / self.fps:
            return None
        return (
            f"episode {index:06d} occupies {span:.2f}s of video but holds {length} "
            f"frames ({expected:.2f}s) -- {span - expected:+.2f}s of stray footage was "
            "encoded into it, most likely frames left over from a discarded take. "
            "The dataset is still readable, but delete_episodes() will refuse it."
        )

    def discard_episode(self):
        if not self._episode_open:
            raise RuntimeError("No episode is being recorded")
        self._episode_open = False
        dropped = self._frames
        self.discard_episode_buffer()
        return dropped

    def discard_episode_buffer(self, best_effort=False):
        """Drop the open episode's buffer, its already-written frames included.

        Failing to clear is not cosmetic, so it is raised rather than ignored: the
        frames stay queued for the video file and are encoded into whichever
        episode is saved next, corrupting it in the way _check_saved_video_span
        describes. Silently continuing is what lets that reach the dataset.

        best_effort is for teardown only, where finalize() still has to run.
        """
        self._frames = 0
        try:
            self.dataset.clear_episode_buffer()
        except Exception as e:
            if not best_effort:
                raise RuntimeError(
                    f"Could not discard the episode buffer ({e}). Its frames may still "
                    "be encoded into the next episode you save -- restart the session "
                    "before recording again."
                ) from e

    # -- teardown ----------------------------------------------------------

    def close(self):
        """Idempotent. finalize() is the important part: without it the parquet
        footers are never written and the dataset can't be loaded back."""
        if self._closed:
            return
        self._closed = True
        if self._episode_open:
            self._episode_open = False
            # Teardown: finalize() below still has to run, so a failed clear must
            # not abort it. Anything left behind is caught on the next save.
            self.discard_episode_buffer(best_effort=True)
        try:
            self.dataset.writer.stop_image_writer()
        except Exception:
            pass
        self.dataset.finalize()
