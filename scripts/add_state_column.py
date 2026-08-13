#!/usr/bin/env python3
"""Backfill the all-zero `observation.state` into a dataset recorded without one.

    python scripts/add_state_column.py [--root data/lerobot/lone/l_one]

L-ONE measures nothing, so the column is zeros -- it exists to give the dataset
the shape every LeRobot policy expects, not to carry information. See STATE_NAMES
in lone_data/features.py.

Rewrites the data parquet and meta/info.json in place. Idempotent: exits without
changes if the column is already present. Rerun scripts/fix_action_stats.py
afterwards.
"""

import argparse
import json
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- disables Hub access before lerobot is imported

from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.features import STATE_DIM, STATE_NAMES

DEFAULT_ROOT = "data/lerobot/lone/l_one"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--no-backup", action="store_true", help="skip the .bak copies")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root
    info_path = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info_path):
        print(f"no dataset at {root}", file=sys.stderr)
        return 1

    with open(info_path) as f:
        info = json.load(f)

    if OBS_STATE in info["features"]:
        print(f"{OBS_STATE} already present; nothing to do")
        return 0

    data_dir = os.path.join(root, "data")
    parquets = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(data_dir)
        for name in sorted(names)
        if name.endswith(".parquet")
    ]
    if not parquets:
        print(f"no parquet files under {data_dir}", file=sys.stderr)
        return 1

    total = 0
    for path in sorted(parquets):
        table = pq.read_table(path)
        if OBS_STATE in table.column_names:
            print(f"  {path}: already has {OBS_STATE}, skipped")
            continue

        rows = table.num_rows
        # A list column of width STATE_DIM, matching how action is stored.
        column = pa.array([[0.0] * STATE_DIM] * rows, type=pa.list_(pa.float32()))
        action_at = table.column_names.index(ACTION)
        table = table.add_column(action_at, OBS_STATE, column)

        if not args.no_backup:
            shutil.copy2(path, path + ".bak")
        pq.write_table(table, path)
        total += rows
        print(f"  {path}: wrote {rows} rows of {STATE_DIM} zeros")

    # Insert the feature before `action`, mirroring the parquet column order.
    features, rebuilt = info["features"], {}
    for key, value in features.items():
        if key == ACTION:
            rebuilt[OBS_STATE] = {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES,
            }
        rebuilt[key] = value
    info["features"] = rebuilt

    if not args.no_backup:
        shutil.copy2(info_path, info_path + ".bak")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # What save_episode() would have computed for an all-zero column, so a
    # backfilled dataset and a freshly recorded one carry the same statistics.
    stats_path = os.path.join(root, "meta", "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        if OBS_STATE not in stats:
            zeros = [0.0] * STATE_DIM
            stats[OBS_STATE] = {
                name: list(zeros)
                for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
            }
            stats[OBS_STATE]["count"] = [total]
            if not args.no_backup:
                shutil.copy2(stats_path, stats_path + ".bak")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"wrote all-zero {OBS_STATE} statistics")

    print(f"\nadded {OBS_STATE} to {total} frames and to meta/info.json")
    print("now run: python scripts/fix_action_stats.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
