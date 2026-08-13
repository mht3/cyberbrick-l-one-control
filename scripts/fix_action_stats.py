#!/usr/bin/env python3
"""Rewrite a dataset's `action` normalization statistics to the known command limits.

    python scripts/fix_action_stats.py [--root data/lerobot/lone/l_one]

Rerun this after every collection session. `LeRobotDataset.save_episode()` calls
aggregate_stats() and write_stats() on each saved episode (lerobot_dataset.py:424),
so meta/stats.json is regenerated from the recorded data every time and any earlier
edit is overwritten.

WHY: teleop emits three discrete levels per channel, so estimating an action range
from recorded frames is unreliable at any dataset size and outright degenerate at
small ones. In the first recorded episode the base motor never turned positive, so
min/max came out (-900, 0) and q01/q99 (-900, -0.0019). Under those statistics:

    MIN_MAX      (Diffusion Policy):  0 -> +1.0    +900 -> +3.0
    QUANTILES    (pi0.5):             0 -> +1.0    +900 -> +3.0

i.e. the single most common action -- 0, meaning stopped -- lands on the extreme of
the normalized range, and any future command in the unseen direction lands well
outside the [-1, 1] the policy's output head expects.

The limits are known exactly from virtual_gripper.py, so they are declared in
lone_data.features.ACTION_COMMAND_LIMITS and written here rather than estimated.

Each normalization mode reads a different pair of statistics, so all four pairs are
written -- whichever mode a policy config selects then spans the true envelope:

    MEAN_STD     -> mean, std      (ACT)
    MIN_MAX      -> min, max       (Diffusion Policy)
    QUANTILES    -> q01, q99       (pi0.5)
    QUANTILE10   -> q10, q90

For MEAN_STD, mean/std are set to the envelope's midpoint and half-range so that a
limit maps to +-1, matching the other three modes. This makes them describe the
command envelope rather than the recorded sample; pass --keep-mean-std to leave the
measured values in place. Nothing outside the `action` entry is touched, and q50 is
left as measured since no normalization mode reads it.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lone_data.features import ACTION_COMMAND_LIMITS, ACTION_NAMES

DEFAULT_ROOT = "data/lerobot/lone/l_one"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--keep-mean-std", action="store_true",
                   help="leave mean/std as measured instead of setting them to the envelope")
    p.add_argument("--dry-run", action="store_true", help="print the change without writing")
    return p.parse_args()


def main():
    args = parse_args()
    path = os.path.join(args.root, "meta", "stats.json")
    if not os.path.exists(path):
        print(f"no stats at {path}", file=sys.stderr)
        return 1

    with open(path) as f:
        stats = json.load(f)

    if "action" not in stats:
        print(f"{path} has no 'action' entry", file=sys.stderr)
        return 1

    action = stats["action"]
    dim = len(ACTION_COMMAND_LIMITS)
    if len(action.get("min", [])) != dim:
        print(f"action stats have {len(action.get('min', []))} dims, "
              f"ACTION_COMMAND_LIMITS has {dim}", file=sys.stderr)
        return 1

    lows = [lo for lo, _ in ACTION_COMMAND_LIMITS]
    highs = [hi for _, hi in ACTION_COMMAND_LIMITS]

    new = dict(action)
    for name in ("min", "q01", "q10"):
        new[name] = list(lows)
    for name in ("max", "q99", "q90"):
        new[name] = list(highs)
    if not args.keep_mean_std:
        new["mean"] = [(lo + hi) / 2.0 for lo, hi in ACTION_COMMAND_LIMITS]
        new["std"] = [(hi - lo) / 2.0 for lo, hi in ACTION_COMMAND_LIMITS]

    changed = [k for k in new if new[k] != action.get(k)]
    width = max(len(n) for n in ACTION_NAMES)
    print(f"{path}\n")
    for key in ("min", "max", "mean", "std", "q01", "q99", "q10", "q90"):
        if key not in new:
            continue
        mark = "*" if key in changed else " "
        print(f" {mark} {key:5s}")
        for i, name in enumerate(ACTION_NAMES):
            before, after = action[key][i], new[key][i]
            arrow = "->" if before != after else "  "
            print(f"     {name:{width}s}  {before:12.4f} {arrow} {after:12.4f}")

    if not changed:
        print("\nalready at the command limits; nothing to write")
        return 0
    if args.dry_run:
        print(f"\ndry run -- would rewrite {', '.join(sorted(changed))}")
        return 0

    stats["action"] = new
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"\nrewrote {', '.join(sorted(changed))}")
    print("rerun after the next collection session -- save_episode() regenerates this file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
