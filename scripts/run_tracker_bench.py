"""Run gripper trackers over a labelled dataset and save their tracks.

    python scripts/run_tracker_bench.py \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval \
        --trackers cotracker3_offline cotracker3_online sam2_large lk

Every tracker gets the same inputs (lone_track/queries.py): the labelled body
and fingertips on the first labelled frame where all three are visible, plus
24 points sampled inside SAM2's gripper mask there. Point trackers follow all
27 at once, so one run gives both strategies the scorer compares -- the three
keypoints tracked directly, and the keypoints carried along by a similarity fit
to the 24-point cluster. SAM2 trackers are prompted with the three keypoints.

--crop auto runs on a fixed per-view crop around everywhere the gripper is
labelled (+48 px), the way a deployed tracker would crop the workspace: the
models resize to 256-512 px, so a crop gives the 30-50 px gripper more pixels.

Tracks land in outputs/track/<dataset>/runs/<tracker>[_crop]/ep<NNN>_<view>.npz;
existing ones are skipped, so a run can be resumed. Then:
    python scripts/score_tracker_bench.py ...
"""

import argparse
import json
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")  # SAM2's per-propagation progress bars

import numpy as np  # noqa: E402

from lone_track import trackers  # noqa: E402
from lone_track.episodes import EpisodeIndex  # noqa: E402
from lone_track.labels import KEYPOINTS, LabelSet  # noqa: E402
from lone_track.queries import MaskPrompter, build_queries, load_queries, query_path  # noqa: E402

CROP_MARGIN = 48


def dataset_dir(repo_id):
    return os.path.join("outputs", "track", repo_id.split("/")[-1])


def auto_crops(labels, views, width, height, margin=CROP_MARGIN):
    """Per view, one (x0, y0, x1, y1) box around every labelled point, + margin."""
    crops = {}
    for view in views:
        pts = [e["xy"] for frame in labels.frames.values()
               for kp, e in frame.get(view, {}).items() if e["xy"] is not None]
        pts = np.asarray(pts)
        x0, y0 = np.floor(pts.min(0) - margin).astype(int)
        x1, y1 = np.ceil(pts.max(0) + margin).astype(int)
        crops[view] = (max(0, x0), max(0, y0), min(width, x1 + 1), min(height, y1 + 1))
    return crops


def run_one(tracker, frames, q, crop):
    """Track the query points in one episode-view; returns (TrackResult, seconds)."""
    pts = q["kp_xy"] if tracker.name.startswith("sam2") else np.concatenate([q["kp_xy"], q["cluster"]])
    queries = np.column_stack([np.full(len(pts), int(q["t_q"])), pts]).astype(np.float32)
    x0 = y0 = 0
    if crop is not None:
        x0, y0, x1, y1 = crop
        frames = np.ascontiguousarray(frames[:, y0:y1, x0:x1])
        queries[:, 1] -= x0
        queries[:, 2] -= y0
    t = time.perf_counter()
    res = tracker.track(frames, queries)
    elapsed = time.perf_counter() - t
    res.xy[..., 0] += x0
    res.xy[..., 1] += y0
    if "centroid" in res.extra:
        res.extra["centroid"] = res.extra["centroid"] + [x0, y0]
    return res, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--labels", default=None, help="default labels/gripper_gt_<dataset>.json")
    p.add_argument("--trackers", nargs="+", required=True, choices=trackers.NAMES)
    p.add_argument("--crop", choices=["none", "auto"], default="none")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="default: every episode with a query file (i.e. labelled)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", default=None,
                   help="output root (default outputs/track/<dataset>); queries and runs go under it")
    args = p.parse_args()

    name = args.repo_id.split("/")[-1]
    labels = LabelSet.load(args.labels or os.path.join("labels", f"gripper_gt_{name}.json"))
    index = EpisodeIndex(args.repo_id, args.root)
    base = args.out or dataset_dir(args.repo_id)
    qdir = os.path.join(base, "queries")

    episodes = args.episodes if args.episodes is not None else labels.episodes()
    missing = [ep for ep in episodes for v in index.views
               if not os.path.exists(query_path(qdir, ep, v))]
    if missing:
        print(f"building queries for episodes {sorted(set(missing))} ...")
        prompter = MaskPrompter(os.path.join(trackers.CKPT_DIR, "sam2.1_hiera_large.pt"))
        for ep, view, why in build_queries(labels, index, qdir, prompter, sorted(set(missing))):
            print(f"  skip ep {ep} {view}: {why}")
        del prompter
    crops = auto_crops(labels, index.views, index.width, index.height) if args.crop == "auto" else {}

    for tname in args.trackers:
        tracker = trackers.build(tname)
        run = tracker.name + ("_crop" if crops else "")
        out_dir = os.path.join(base, "runs", run)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"tracker": tname, "causal": tracker.causal, "crop": crops or None,
                       "keypoints": list(KEYPOINTS), "repo_id": args.repo_id}, f, indent=1)
        for ep in episodes:
            frames_by_view = {}
            for view in index.views:
                q = load_queries(qdir, ep, view)
                path = os.path.join(out_dir, f"ep{ep:03d}_{view}.npz")
                if q is None or (os.path.exists(path) and not args.overwrite):
                    continue
                if view not in frames_by_view:
                    frames_by_view[view] = index.frames(ep, view)
                res, secs = run_one(tracker, frames_by_view[view], q, crops.get(view))
                res.save(path, t_q=int(q["t_q"]), n_kp=len(KEYPOINTS), seconds=secs)
                print(f"{run:28s} ep {ep:3d} {view:5s} t_q={int(q['t_q']):3d} "
                      f"{len(res.xy)} frames in {secs:5.1f}s")
        del tracker
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
