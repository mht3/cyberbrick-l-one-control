"""Relabel one episode's task string in a finalized LeRobotDataset.

A task typed in a hurry, or left over from the previous episode, is worth fixing:
pi0.5 conditions on that text, so a wrong prompt is wrong supervision for every
frame of the episode.

The string lives in three places and all three have to agree:

    meta/tasks.parquet          task_index -> task string
    meta/episodes/*.parquet     `tasks`, a per-episode list of strings
    data/*.parquet              `task_index`, one per frame

So relabelling is not a string replace. If other episodes still use the old text,
the edit must *split* the task -- mint a new index, repoint only this episode's
frames -- rather than rename it out from under them. If no episode is left using
the old string it is dropped, keeping indices dense.

Only ever call this with the dataset closed. LeRobotDataset buffers episode
metadata and holds parquet writers open until finalize(), so editing underneath a
live recorder would be overwritten by it.
"""

import json
import os
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TASKS_REL = os.path.join("meta", "tasks.parquet")


def _parquet_files(root, subdir):
    base = os.path.join(root, subdir)
    return sorted(
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(base)
        for name in names
        if name.endswith(".parquet")
    )


def read_tasks(root):
    """{task_index: task string} for the dataset at `root`."""
    df = pd.read_parquet(os.path.join(root, TASKS_REL))
    return dict(zip(df["task_index"].tolist(), df.index.tolist()))


def _write_tasks(root, tasks, backup):
    """Round-trip through pandas, because the task string is the frame's *index*.

    Writing a plain arrow table with a `__index_level_0__` column loses the pandas
    metadata that says which column is the index, and LeRobot then reads the task
    back as an integer -- ds[i]["task"] returns 0 instead of the sentence.
    """
    path = os.path.join(root, TASKS_REL)
    df = pd.DataFrame(
        {"task_index": list(tasks.keys())},
        index=pd.Index(list(tasks.values()), name=pd.read_parquet(path).index.name),
    )
    if backup:
        shutil.copy2(path, path + ".bak")
    df.to_parquet(path)


def set_episode_task(root, episode_index, new_task, backup=True):
    """Point `episode_index` at `new_task`, splitting or dropping tasks as needed.

    Returns a short description of what changed, or None if the task already read
    that way. Raises ValueError if the episode does not exist or the text is empty.
    """
    new_task = (new_task or "").strip()
    if not new_task:
        raise ValueError("Task text cannot be empty")

    tasks = read_tasks(root)
    by_string = {v: k for k, v in tasks.items()}

    # -- which index does this episode use now? ---------------------------
    data_files = _parquet_files(root, "data")
    current_index = None
    for path in data_files:
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_pylist()
        for row, ep in enumerate(episodes):
            if ep == episode_index:
                current_index = table.column("task_index").to_pylist()[row]
                break
        if current_index is not None:
            break
    if current_index is None:
        raise ValueError(f"Episode {episode_index} not found under {root}")

    if tasks.get(current_index) == new_task:
        return None

    # -- does any *other* episode still use the old string? ---------------
    others = set()
    for path in data_files:
        table = pq.read_table(path)
        eps = table.column("episode_index").to_pylist()
        idxs = table.column("task_index").to_pylist()
        others.update(i for e, i in zip(eps, idxs) if e != episode_index)
    shared = current_index in others

    if new_task in by_string:
        target_index = by_string[new_task]   # merge into an existing task
        if not shared:
            tasks.pop(current_index, None)   # nothing points at the old text now
    elif shared:
        # Other episodes still mean the old text, so split rather than rename.
        target_index = max(tasks) + 1 if tasks else 0
        tasks[target_index] = new_task
    else:
        # Sole user: rename in place, which keeps task indices dense.
        target_index = current_index
        tasks[target_index] = new_task

    # -- rewrite the frame-level task_index -------------------------------
    for path in data_files:
        table = pq.read_table(path)
        eps = table.column("episode_index").to_pylist()
        idxs = table.column("task_index").to_pylist()
        updated = [target_index if e == episode_index else i for e, i in zip(eps, idxs)]
        if updated == idxs:
            continue
        col = table.schema.get_field_index("task_index")
        table = table.set_column(col, "task_index", pa.array(updated, pa.int64()))
        if backup:
            shutil.copy2(path, path + ".bak")
        pq.write_table(table, path)

    # -- rewrite the per-episode `tasks` list ------------------------------
    for path in _parquet_files(root, os.path.join("meta", "episodes")):
        table = pq.read_table(path)
        if "tasks" not in table.column_names:
            continue
        eps = table.column("episode_index").to_pylist()
        col_tasks = table.column("tasks").to_pylist()
        updated = [[new_task] if e == episode_index else t for e, t in zip(eps, col_tasks)]
        if updated == col_tasks:
            continue
        col = table.schema.get_field_index("tasks")
        table = table.set_column(col, "tasks", pa.array(updated, table.column("tasks").type))
        if backup:
            shutil.copy2(path, path + ".bak")
        pq.write_table(table, path)

    _write_tasks(root, tasks, backup)
    _sync_info_total_tasks(root, len(tasks), backup)

    if new_task in by_string:
        return f"episode {episode_index} -> existing task {target_index}"
    return f"episode {episode_index} -> new task {target_index} ({'split' if shared else 'renamed'})"


def _sync_info_total_tasks(root, total, backup):
    path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        info = json.load(f)
    if info.get("total_tasks") == total:
        return
    info["total_tasks"] = total
    if backup:
        shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(info, f, indent=4)
