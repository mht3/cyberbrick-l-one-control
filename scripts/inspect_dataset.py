#!/usr/bin/env python3
"""Summarize an L-ONE LeRobotDataset.

    python scripts/inspect_dataset.py [--root data/lerobot/lone/l_one]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- disables Hub access before lerobot is imported

import lerobot
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION

from lone_data.features import ACTION_NAMES, ACTION_SEMANTICS, CAMERA_KEY, SYNC_NOTE, ZERO_DISPATCH_CONVENTION
from lone_data.validation import describe_state_policy_support

DEFAULT_ROOT = "data/lerobot/lone/l_one"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--repo-id", default="lone/l_one", help="dataset repo id")
    p.add_argument("--frames", type=int, default=3, help="how many frames to load through __getitem__")
    return p.parse_args()


def main():
    args = parse_args()
    ds = LeRobotDataset(args.repo_id, root=args.root)
    meta = ds.meta

    print(f"path             {os.path.abspath(args.root)}")
    print(f"repo_id          {ds.repo_id}")
    print(f"lerobot          {lerobot.__version__}  (dataset codebase {CODEBASE_VERSION})")
    print(f"robot_type       {meta.robot_type}")
    print(f"episodes         {meta.total_episodes}")
    print(f"frames           {meta.total_frames}")
    print(f"fps              {meta.fps}")

    print("\nfeatures")
    for key, ft in meta.features.items():
        names = ft.get("names")
        suffix = f"  names={names}" if names else ""
        print(f"  {key:32s} {ft['dtype']:8s} {tuple(ft['shape'])}{suffix}")

    print("\npolicy features (what dataset_to_policy_features exposes)")
    for key, ft in dataset_to_policy_features(meta.features).items():
        print(f"  {key:32s} {ft.type.name:8s} {tuple(ft.shape)}")

    print(f"\nproprioception\n  {describe_state_policy_support(ds)}")

    lengths = [int(v) for v in meta.episodes["length"]]
    print(f"\nepisode lengths  {lengths}")
    print(f"  min/mean/max   {min(lengths)} / {sum(lengths) / len(lengths):.1f} / {max(lengths)}"
          if lengths else "  (none)")

    tasks = list(meta.tasks.index) if hasattr(meta.tasks, "index") else list(meta.tasks or [])
    print(f"\ntasks ({len(tasks)})")
    for t in tasks:
        print(f"  {t!r}")

    print("\naction range per dimension")
    step = max(1, len(ds) // 2000)
    actions = np.stack([np.asarray(ds[i][ACTION]) for i in range(0, len(ds), step)])
    if actions.ndim == 3:
        actions = actions[:, 0, :]
    for i, name in enumerate(ACTION_NAMES):
        col = actions[:, i]
        print(f"  [{i}] {name:24s} min={col.min():9.2f}  max={col.max():9.2f}  unique={len(np.unique(col))}")

    print("\naction semantics")
    for s in ACTION_SEMANTICS:
        print(f"  [{s['index']}] {s['name']} -- {s['channel']} via {s['call']}, "
              f"{s['unit']} in {s['range']}")
    print(f"\n  {ZERO_DISPATCH_CONVENTION}")
    print(f"\n  {SYNC_NOTE}")

    print(f"\nloading {args.frames} frame(s) through __getitem__")
    for i in np.linspace(0, len(ds) - 1, args.frames).astype(int):
        item = ds[int(i)]
        img = item[CAMERA_KEY]
        print(f"  frame {int(i):6d}  {CAMERA_KEY}={tuple(img.shape)} {img.dtype}  "
              f"action={tuple(np.asarray(item[ACTION]).shape)}  task={item['task']!r}")


if __name__ == "__main__":
    main()
