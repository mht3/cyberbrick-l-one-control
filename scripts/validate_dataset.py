#!/usr/bin/env python3
"""Self-consistency checks for an L-ONE LeRobotDataset.

    python scripts/validate_dataset.py [--root data/lerobot/lone/l_one_marker_pickup]

Exits non-zero if any check fails.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- disables Hub access before lerobot is imported

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from lone_data.validation import describe_state_policy_support, validate_dataset

DEFAULT_ROOT = "data/lerobot/lone/l_one_marker_pickup"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--repo-id", default="lone/l_one_marker_pickup", help="dataset repo id")
    return p.parse_args()


def main():
    args = parse_args()
    ds = LeRobotDataset(args.repo_id, root=args.root)
    meta = ds.meta

    print(
        f"episodes {meta.total_episodes}   frames {meta.total_frames}   "
        f"fps {meta.fps}   codebase {CODEBASE_VERSION}"
    )
    print(f"features {sorted(meta.features)}\n")

    warnings = validate_dataset(ds)

    # A chunked load is what training actually does, so prove it works here
    # rather than assuming the fixed-rate timestamp grid is queryable.
    try:
        horizon = min(30, meta.total_frames)
        chunked = LeRobotDataset(
            args.repo_id, root=args.root,
            delta_timestamps={"action": [t / meta.fps for t in range(horizon)]},
        )
        item = chunked[0]
        if tuple(item["action"].shape)[0] != horizon:
            warnings.append(f"chunked action load returned {tuple(item['action'].shape)}, expected ({horizon}, 4)")
        else:
            print(f"chunked load     PASS  delta_timestamps horizon={horizon} -> {tuple(item['action'].shape)}")
    except Exception as e:
        warnings.append(f"chunked load failed: {e}")

    print(f"\nproprioception\n  {describe_state_policy_support(ds)}")

    if warnings:
        print(f"\nFAILED ({len(warnings)} issue(s)):")
        for w in warnings:
            print(f"  - {w}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
