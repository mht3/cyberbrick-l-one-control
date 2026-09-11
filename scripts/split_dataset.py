#!/usr/bin/env python3
"""Split a recorded dataset into a training set and a held-out evaluation set.

    python scripts/split_dataset.py \
        --src-repo-id lone/l_one_manipulation \
        --train-repo-id lone/l_one_manipulation_train \
        --eval-repo-id lone/l_one_manipulation_eval10 \
        --eval-episodes 10

Whole episodes go to one side or the other and every frame is copied verbatim --
nothing is dropped, reordered or resampled. Both outputs are written frame by
frame through LoneRecorder, so they are ordinary LeRobotDatasets with LeRobot's
own timestamps, episode indexing and video encoding, and the schema in
lone_data/features.py. The source is only ever read.

Episodes are held out rather than frames, because a held-out set is only worth
anything if the policy has never seen anything from those demonstrations. Frames
sampled out of episodes that are otherwise trained on measure memorization: the
frame either side of a held-out one is nearly the same picture.

Frames are never dropped to even out the skewed teleop action distribution. Idle
frames are real supervision, and pi0.5 trains on chunks of consecutive actions:
cutting frames out of an episode leaves chunks that span stretched, discontinuous
time and teach the motion too fast.
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import lone_data  # noqa: F401  -- sets HF_HUB_OFFLINE before lerobot is imported

from lone_data.checkpoints import default_root
from lone_data.features import ACTION_LEVELS, ACTION_NAMES, dataset_camera_keys
from lone_data.lerobot_recorder import LoneRecorder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-repo-id", required=True, help="dataset to read")
    p.add_argument("--src-root", default=None, help="default: data/lerobot/<src-repo-id>")
    p.add_argument("--train-repo-id", required=True, help="training set to write")
    p.add_argument("--train-root", default=None)
    p.add_argument("--eval-repo-id", required=True, help="held-out eval set to write")
    p.add_argument("--eval-root", default=None)
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="how many source episodes to hold out (default 5)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete the output directories first if they exist")
    p.add_argument("--dry-run", action="store_true",
                   help="report the split and the distributions, write nothing")
    args = p.parse_args()
    args.src_root = args.src_root or default_root(args.src_repo_id)
    args.train_root = args.train_root or default_root(args.train_repo_id)
    args.eval_root = args.eval_root or default_root(args.eval_repo_id)
    if args.eval_episodes < 1:
        p.error("--eval-episodes must be at least 1")
    return args


def level_index(values, levels):
    return np.argmin(np.abs(np.asarray(values)[:, None] - np.array(levels)[None, :]), axis=1)


def describe(tag, actions):
    """Per-dimension level distribution, so the split's effect on it is visible."""
    print(f"\n  {tag}: {len(actions)} frames")
    if not len(actions):
        return
    for d, name in enumerate(ACTION_NAMES):
        counts = np.bincount(level_index(actions[:, d], ACTION_LEVELS[d]),
                             minlength=len(ACTION_LEVELS[d]))
        pct = 100 * counts / counts.sum()
        print(f"    [{d}] {name:24s} " + "  ".join(f"{v:5.1f}%" for v in pct))
    moving = np.zeros(len(actions), bool)
    for d in range(3):
        moving |= level_index(actions[:, d], ACTION_LEVELS[d]) != 1
    print(f"    no joint driven: {100 * (~moving).mean():.1f}%")


def episode_bounds(meta):
    rows = meta.episodes
    return [(int(rows["dataset_from_index"][i]), int(rows["dataset_to_index"][i]))
            for i in range(meta.total_episodes)]


def pick_eval_episodes(total, count):
    """Evenly spaced episodes, avoiding both ends.

    Demonstrations recorded back to back share lighting, start pose and whatever
    the operator was doing that afternoon, so five consecutive episodes are five
    samples of one moment. Spreading them across the collection makes the held-out
    set a fairer question; the ends are skipped as the least typical takes.
    """
    count = min(count, total)
    return sorted({int(round((i + 0.5) * total / count)) % total for i in range(count)})


def rgb_of(item, camera_keys):
    """uint8 HWC RGB per view, as LoneRecorder wants them.

    LeRobot hands back float32 CHW in [0,1] that was uint8 on disk, so scaling by
    255 and rounding recovers the original bytes exactly.
    """
    return [
        np.clip(np.round(item[key].permute(1, 2, 0).numpy() * 255.0), 0, 255).astype(np.uint8)
        for key in camera_keys
    ]


def prepare_output(root, overwrite):
    if not os.path.exists(root):
        return
    if not overwrite:
        raise SystemExit(f"{root} already exists -- pass --overwrite to replace it.")
    shutil.rmtree(root)


def build(source, bounds, episodes, camera_keys, repo_id, root, fps, image_size):
    """Copy `episodes` of `source` into a new dataset, frame for frame."""
    recorder = LoneRecorder(repo_id, root, fps=fps, image_size=image_size,
                            num_cameras=len(camera_keys))
    written, started = 0, time.time()
    try:
        for n, e in enumerate(episodes):
            lo, hi = bounds[e]
            recorder.start_episode()
            for i in range(lo, hi):
                item = source[i]
                recorder.add_frame(rgb_of(item, camera_keys),
                                   np.asarray(item["action"], dtype=np.float32),
                                   item["task"])
                written += 1
            recorder.finish_episode()
            if recorder.last_warning:
                print(f"    warning: {recorder.last_warning}")
            if (n + 1) % 10 == 0 or n + 1 == len(episodes):
                rate = written / max(1e-9, time.time() - started)
                remaining = (sum(bounds[x][1] - bounds[x][0] for x in episodes[n + 1:]) / max(rate, 1e-9))
                print(f"    {n + 1}/{len(episodes)} episodes, {written} frames "
                      f"({rate:.0f} fps, ~{remaining / 60:.1f} min left)", flush=True)
    finally:
        recorder.close()
    return written


def main():
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(args.src_repo_id, root=args.src_root)
    meta = source.meta
    camera_keys = dataset_camera_keys(meta.features)
    if not camera_keys:
        raise SystemExit(f"{args.src_root} has no L-ONE camera columns.")
    image_size = tuple(meta.features[camera_keys[0]]["shape"][:2])
    bounds = episode_bounds(meta)

    print(f"source: {args.src_root}")
    print(f"  {meta.total_episodes} episodes, {meta.total_frames} frames, {meta.fps} fps")
    print(f"  {len(camera_keys)} camera view(s): {', '.join(camera_keys)}")
    print(f"  frames stored at {image_size[1]}x{image_size[0]}")

    eval_eps = pick_eval_episodes(meta.total_episodes, args.eval_episodes)
    train_eps = [e for e in range(meta.total_episodes) if e not in set(eval_eps)]
    n_eval = sum(bounds[e][1] - bounds[e][0] for e in eval_eps)
    n_train = sum(bounds[e][1] - bounds[e][0] for e in train_eps)
    print(f"\nheld out for eval: episodes {eval_eps}  ({n_eval} frames)")
    print(f"training set:      {len(train_eps)} episodes  ({n_train} frames)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    prepare_output(args.train_root, args.overwrite)
    prepare_output(args.eval_root, args.overwrite)

    print(f"\nwriting eval set -> {args.eval_root}", flush=True)
    build(source, bounds, eval_eps, camera_keys, args.eval_repo_id, args.eval_root,
          meta.fps, image_size)

    print(f"\nwriting training set -> {args.train_root}", flush=True)
    build(source, bounds, train_eps, camera_keys, args.train_repo_id, args.train_root,
          meta.fps, image_size)

    # Read both back and report what actually landed, rather than what was planned.
    for tag, repo_id, root in (("training set", args.train_repo_id, args.train_root),
                               ("eval set", args.eval_repo_id, args.eval_root)):
        ds = LeRobotDataset(repo_id, root=root)
        actions = np.stack([np.asarray(ds[i]["action"], dtype=np.float32)
                            for i in range(ds.meta.total_frames)])
        describe(f"{tag} ({ds.meta.total_episodes} episodes)", actions)

    print("\nNext, for both datasets:")
    for repo_id, root in ((args.train_repo_id, args.train_root),
                          (args.eval_repo_id, args.eval_root)):
        print(f"  python scripts/fix_action_stats.py --root {root}")
        print(f"  python scripts/validate_dataset.py --repo-id {repo_id} --root {root}")


if __name__ == "__main__":
    main()
