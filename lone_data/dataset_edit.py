"""Relabel or delete an episode in a recorded LeRobotDataset, in place.

Both operations are lerobot's own (`modify_tasks`, `delete_episodes`); this only
adapts them to how the collector uses a dataset: one fixed directory that must
still be there, under the same name, when the GUI reopens it.

That adaptation is the whole reason this module exists. `modify_tasks` already
edits in place, but `delete_episodes` *writes a new dataset* -- it has to, because
removing an episode renumbers every later one and re-encodes any video file that
mixed kept and deleted episodes. Getting the result back under the original path
means the move-aside-and-swap dance below, which is what lerobot's own CLI does.

Call these only with the recorder closed. LeRobotDataset buffers episode metadata
and holds parquet writers open until finalize(), so an edit underneath a live
recorder is simply overwritten by it.
"""

import os
import shutil

from lerobot.datasets.dataset_tools import delete_episodes, modify_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def set_episode_task(root, repo_id, episode_index, new_task):
    """Point `episode_index` at `new_task`. Returns a description, or None if unchanged.

    Task-related files only -- frames and video are untouched, so this is quick
    and safe to do repeatedly while reviewing.
    """
    new_task = (new_task or "").strip()
    if not new_task:
        raise ValueError("Task text cannot be empty")

    dataset = LeRobotDataset(repo_id, root=root)
    if episode_index >= dataset.meta.total_episodes:
        raise ValueError(f"Episode {episode_index} does not exist")
    current = dataset.meta.episodes["tasks"][episode_index]
    if isinstance(current, (list, tuple)):
        current = current[0] if current else None
    if current == new_task:
        return None

    modify_tasks(dataset, episode_tasks={episode_index: new_task})
    return f"episode {episode_index}: {current!r} -> {new_task!r}"


def delete_episode(root, repo_id, episode_index, keep_backup=True):
    """Remove `episode_index`, leaving the dataset at `root` under `repo_id`.

    The previous dataset is moved to `<root>_old` first and the rebuilt one is
    written back to `root`. That backup is the only copy of what was deleted, so
    it is kept by default -- deleting an episode is not something to discover was
    a mistake with nothing to restore from.

    Returns (episodes_remaining, backup_path or None).
    """
    dataset = LeRobotDataset(repo_id, root=root)
    total = dataset.meta.total_episodes
    if episode_index >= total:
        raise ValueError(f"Episode {episode_index} does not exist")
    if total <= 1:
        # delete_episodes refuses this, and an empty dataset directory is worse
        # than none: LoneRecorder deletes metadata-without-data on next open.
        raise ValueError(
            "Cannot delete the only episode -- delete the dataset directory instead"
        )

    root = str(root)
    backup = root.rstrip("/") + "_old"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    shutil.move(root, backup)

    # Read from the backup, write to the original path, so the dataset keeps its
    # name and the GUI's --root stays valid.
    try:
        source = LeRobotDataset(repo_id, root=backup)
        new_dataset = delete_episodes(
            source, episode_indices=[episode_index], output_dir=root, repo_id=repo_id
        )
    except Exception:
        # Put it back rather than leave the operator with only a _old directory.
        if os.path.exists(root):
            shutil.rmtree(root)
        shutil.move(backup, root)
        raise

    remaining = new_dataset.meta.total_episodes
    if not keep_backup:
        shutil.rmtree(backup)
        return remaining, None
    return remaining, backup
