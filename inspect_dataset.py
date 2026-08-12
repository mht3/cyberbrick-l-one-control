#!/usr/bin/env python3
"""Print a summary report + run integrity checks for a LoneReplayBuffer
Zarr dataset produced by collect_data.py.

    python inspect_dataset.py [--path lone_dataset.zarr]
"""

import argparse

import numpy as np

from lone_data.replay_buffer import ACTION_NAMES, LoneReplayBuffer
from lone_data.validation import validate_dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="datasets/lone_dataset.zarr", help="path to the Zarr dataset")
    return p.parse_args()


def _mean_fps_within_episodes(timestamp, episode_ends):
    """1/mean(diff(timestamp)), computed within each episode only -- a
    diff across an episode boundary is meaningless and is excluded."""
    diffs = []
    start = 0
    for end in episode_ends:
        ts = timestamp[start:end]
        if len(ts) > 1:
            diffs.append(np.diff(ts))
        start = end
    if not diffs:
        return 0.0
    all_diffs = np.concatenate(diffs)
    mean_dt = np.mean(all_diffs)
    return 1.0 / mean_dt if mean_dt > 0 else 0.0


def main():
    args = parse_args()
    rb = LoneReplayBuffer(args.path)

    n_episodes = rb.n_episodes
    n_steps = rb.n_steps

    print(f"Episodes: {n_episodes}")
    print(f"Total timesteps: {n_steps}")

    if n_steps == 0:
        print("(dataset is empty)")
        return

    print(f"Image shape: {rb.img_shape}")
    print(f"Image dtype: {rb.img.dtype}")
    print(f"Action shape: {rb.action.shape}")
    print(f"Action dtype: {rb.action.dtype}")
    print(f"Timestamp dtype: {rb.timestamp.dtype}")

    ends = rb.episode_ends
    lengths = np.diff(np.concatenate(([0], ends))).tolist()
    print(f"Episode lengths: {lengths}")

    mean_fps = _mean_fps_within_episodes(rb.timestamp[:], ends)
    print(f"Mean FPS: {mean_fps:.2f}")

    actions = rb.action[:]
    print("Action min/max per dimension:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  [{i}] {name}: min={actions[:, i].min():.2f}  max={actions[:, i].max():.2f}")

    warnings = validate_dataset(rb)
    if warnings:
        print("\nValidation warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nValidation: OK -- no issues found")


if __name__ == "__main__":
    main()
