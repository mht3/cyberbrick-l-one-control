"""What a checkpoint was trained on, read back off disk.

A checkpoint records its training dataset in train_config.json, and both
evaluation and deployment need to follow that pointer: eval to score against the
right episodes, deployment to recover the task string pi0.5 was conditioned on.

This exists because getting it wrong is silent. `LeRobotDataset(repo_id, root=...)`
loads from `root` and treats `repo_id` as a label when the data is local, so a
command that names one dataset and points at another reads the second without
complaint -- and scores a checkpoint against episodes it has never seen while
looking like it worked.
"""

import json
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "data/lerobot"


def default_root(repo_id):
    """Where a locally collected dataset with this repo_id lives."""
    return os.path.join(DATA_DIR, repo_id)


def training_dataset(checkpoint):
    """(repo_id, root) the checkpoint trained on, or (None, None).

    `root` is absolute if train_config.json recorded a relative path, since the
    caller is not necessarily running from the repo root.
    """
    path = os.path.join(checkpoint or "", "train_config.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            dataset = json.load(f).get("dataset", {})
    except (OSError, json.JSONDecodeError):
        return None, None
    repo_id, root = dataset.get("repo_id"), dataset.get("root")
    if root and not os.path.isabs(root):
        root = os.path.join(REPO_DIR, root)
    return repo_id, root


def same_dataset(a, b):
    """Whether two dataset directories are the same place on disk."""
    if not a or not b:
        return False
    return os.path.realpath(a) == os.path.realpath(b)


def dataset_tasks(repo_id, root):
    """The task strings in a dataset's metadata, or [] if it is not on this machine."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        return list(LeRobotDatasetMetadata(repo_id, root=root).tasks.index)
    except Exception:
        return []
