"""Self-consistency checks for an L-ONE LeRobotDataset.

Every check returns human-readable strings rather than raising, so a partly
broken dataset can still be inspected. An empty list means everything passed.
"""

import numpy as np
from lerobot.configs.types import FeatureType
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.features import ACTION_DIM, ACTION_NAMES, CAMERA_KEY

TIMESTAMP_TOLERANCE_S = 1e-4

# The repair for either task-table fault below: rewrite meta/tasks.parquet's
# task_index to 0..N-1 in the row order already there, then remap the task_index
# column in data/**.parquet through the same old->new mapping. Task strings, row
# order and episode metadata stay as they are -- only the integer used to reach a
# task changes -- so no episode ends up pointing at different text.
TASK_REMEDY = ("Renumber meta/tasks.parquet so task_index equals row position, "
               "remapping data/**.parquet task_index to match.")


def validate_dataset(ds):
    warnings = []
    features = ds.meta.features

    for key in (CAMERA_KEY, ACTION):
        if key not in features:
            warnings.append(f"missing canonical feature {key!r} (have: {sorted(features)})")
    if warnings:
        return warnings

    # dataset_to_policy_features matches "action" exactly and the "observation"
    # prefix, dropping anything else -- so this is what policies actually see.
    policy_features = dataset_to_policy_features(features)
    types = {ft.type for ft in policy_features.values()}
    if FeatureType.VISUAL not in types:
        warnings.append("no VISUAL feature reaches policies via dataset_to_policy_features")
    if FeatureType.ACTION not in types:
        warnings.append("no ACTION feature reaches policies via dataset_to_policy_features")

    action_ft = features[ACTION]
    if tuple(action_ft["shape"]) != (ACTION_DIM,):
        warnings.append(f"action shape {tuple(action_ft['shape'])} != ({ACTION_DIM},)")
    if action_ft["dtype"] != "float32":
        warnings.append(f"action dtype {action_ft['dtype']} != float32")
    if list(action_ft.get("names") or []) != ACTION_NAMES:
        warnings.append(f"action names {action_ft.get('names')} != {ACTION_NAMES}")

    cam_ft = features[CAMERA_KEY]
    if cam_ft["dtype"] not in ("video", "image"):
        warnings.append(f"{CAMERA_KEY} dtype {cam_ft['dtype']} is not video/image")
    if len(cam_ft["shape"]) != 3:
        warnings.append(f"{CAMERA_KEY} shape {cam_ft['shape']} is not 3-dimensional")

    warnings += _check_tasks(ds)

    # Reading any frame resolves item["task"] through the task table, so that has
    # to be sound before a frame is touched -- otherwise the sampling below raises
    # instead of reporting, and takes every remaining check down with it.
    if _tasks_resolvable(ds):
        warnings += _check_frames(ds)
    else:
        warnings.append("skipped frame checks -- every frame read raises while the "
                        "task table is unusable; rerun once it is repaired")

    warnings += _check_stats(ds)
    return warnings


def _check_stats(ds):
    """A zero std is invisible at training time -- the normalizer just divides by
    1e-8 -- so it has to be checked here. lerobot once overflowed uint8 images into
    a zero std; fixed as of 0.6.1, but cheap to keep watching for."""
    warnings = []
    stats = getattr(ds.meta, "stats", None)
    if not stats:
        return ["dataset has no meta/stats.json"]
    causes = {
        CAMERA_KEY: "lerobot's uint8 stats overflow is back; image normalization "
                    "would divide by ~1e-8",
        ACTION: "every action dimension is constant across the whole dataset, "
                "so there is nothing for a policy to learn from it",
    }
    for key in (CAMERA_KEY, ACTION):
        if key not in stats:
            warnings.append(f"no statistics recorded for {key!r}")
            continue
        std = np.asarray(stats[key]["std"])
        if not np.all(np.isfinite(std)):
            warnings.append(f"{key} std contains NaN/Inf")
        elif np.all(std == 0):
            warnings.append(
                f"{key} std is all zeros (normalization would divide by ~1e-8): {causes[key]}"
            )
    return warnings


def _episode_table(ds):
    """(episode_index, length) pairs from LeRobot's own episode metadata."""
    episodes = ds.meta.episodes
    return [
        (int(episodes["episode_index"][i]), int(episodes["length"][i]))
        for i in range(len(episodes["episode_index"]))
    ]


def _check_frames(ds):
    warnings = []
    fps = ds.meta.fps
    total = 0
    for ep_idx, length in _episode_table(ds):
        total += length
        if length <= 0:
            warnings.append(f"episode {ep_idx} has non-positive length {length}")

    if total != ds.meta.total_frames:
        warnings.append(f"episode lengths sum to {total} but total_frames is {ds.meta.total_frames}")

    # Sample rather than decode every frame -- videos are expensive to touch.
    n = min(len(ds), 64)
    if n == 0:
        return warnings
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    for i in idxs:
        item = ds[int(i)]
        action = np.asarray(item[ACTION])
        if not np.all(np.isfinite(action)):
            warnings.append(f"frame {i}: action contains NaN/Inf")
            break

    for i in idxs:
        item = ds[int(i)]
        ts, frame_index = float(item["timestamp"]), int(item["frame_index"])
        if abs(ts - frame_index / fps) > TIMESTAMP_TOLERANCE_S:
            warnings.append(
                f"frame {i}: timestamp {ts:.6f} != frame_index/fps {frame_index / fps:.6f}"
            )
            break

    img = ds[int(idxs[0])][CAMERA_KEY]
    if img.ndim != 3 or img.shape[0] != 3:
        warnings.append(f"{CAMERA_KEY} loads as {tuple(img.shape)}, expected channel-first (3,H,W)")
    if img.dtype != np.dtype("float32") and str(img.dtype) != "torch.float32":
        warnings.append(f"{CAMERA_KEY} loads as {img.dtype}, expected float32")

    return warnings


def _positional_task_gaps(ds):
    """Rows of meta/tasks.parquet whose task_index differs from their position.

    lerobot reads a frame's task string as `tasks.iloc[task_index].name` --
    positionally -- so the task_index column is only ever as good as its agreement
    with row order. Nothing in lerobot enforces that, and editing tasks can break
    it: drop a row without renumbering and every later index resolves one task
    early, or off the end of the table entirely. Returns [] when they agree.
    """
    tasks = getattr(ds.meta, "tasks", None)
    if tasks is None or not hasattr(tasks, "columns") or "task_index" not in tasks.columns:
        return []
    return [
        (position, int(recorded))
        for position, recorded in enumerate(tasks["task_index"])
        if int(recorded) != position
    ]


def _tasks_resolvable(ds):
    """Whether every frame's task_index lands on the task it names."""
    return not _positional_task_gaps(ds) and not _out_of_range_task_indices(ds)


def _out_of_range_task_indices(ds):
    """Frame task_index values with no row at that position -- these raise IndexError."""
    tasks = getattr(ds.meta, "tasks", None)
    if tasks is None:
        return []
    used = np.unique(np.asarray(ds.hf_dataset["task_index"], dtype=np.int64))
    return sorted(int(v) for v in used if v < 0 or v >= len(tasks))


def _check_tasks(ds):
    warnings = []
    tasks = ds.meta.tasks
    if tasks is None or len(tasks) == 0:
        warnings.append("dataset has no tasks recorded")
        return warnings
    names = list(tasks.index) if hasattr(tasks, "index") else list(tasks)
    for task in names:
        if not str(task).strip():
            warnings.append("dataset contains an empty task string")

    out_of_range = _out_of_range_task_indices(ds)
    if out_of_range:
        warnings.append(
            f"frames use task_index {out_of_range} but meta/tasks.parquet has only "
            f"{len(tasks)} rows -- reading those frames raises IndexError. {TASK_REMEDY}"
        )

    gaps = _positional_task_gaps(ds)
    if gaps:
        # Silent by construction: an in-range index still resolves, just to the
        # wrong string, so nothing downstream notices the episodes are mislabelled.
        shown = ", ".join(f"row {p} holds task_index {r}" for p, r in gaps[:3])
        warnings.append(
            f"{len(gaps)} task(s) sit at a row that does not match their task_index "
            f"({shown}{', ...' if len(gaps) > 3 else ''}). lerobot resolves tasks by "
            "row position, so affected frames silently train on another task's text. "
            f"{TASK_REMEDY}"
        )
    return warnings


def describe_state_policy_support(ds):
    """L-ONE measures nothing, so `observation.state` is present but all zeros --
    the shape every policy requires, carrying no information."""
    if OBS_STATE not in ds.meta.features:
        return (
            f"{OBS_STATE} absent. ACT and pi0/pi0.5 raise without a state tensor, and\n"
            "  Diffusion Policy and SmolVLA cannot even build. Run scripts/add_state_column.py."
        )
    # Verified against lerobot 0.6.1: all four consume the state, by different
    # routes. ACT projects it to a token for both the VAE and transformer encoders
    # (modeling_act.py:413,465); pi0.5 has no state_proj because it discretises the
    # state into 256 bins and writes it into the text prompt instead
    # (processor_pi05.py:68-74); Diffusion Policy passes it as global conditioning
    # (modeling_diffusion.py:273); SmolVLA embeds it via state_proj
    # (modeling_smolvla.py:508,619). So the column is not decorative -- it reaches
    # the network in every case. All zeros makes it a constant input, which is
    # uninformative but harmless: a constant is something a network can absorb.
    return (
        f"{OBS_STATE} present and all zeros -- L-ONE has no proprioceptive sensors.\n"
        "  Every policy both requires this schema and feeds the values to the network:\n"
        "  ACT as an encoder token, pi0.5 discretised into the prompt text, Diffusion\n"
        "  Policy as global conditioning, SmolVLA through state_proj. Being all zeros,\n"
        "  each receives a constant -- so they are vision-only in substance, not because\n"
        "  the state is ignored but because it never varies."
    )
