"""Choose which frames get hand-labelled, before any tracker output exists.

Picking frames after looking at tracker results would bias the benchmark toward
(or away from) its failures, so the choice is made from the actions alone and
written to a tracked file. Per episode:

  - frame 0 and the last frame (initialisation, and the end of the longest drift)
  - every --every'th frame (unbiased error-vs-time curves)
  - around each gripper command change at t: t-1 and t+10. The arm is still and
    the jaws differ, so this pair measures how much the grasp point moves when
    the gripper opens or closes -- and the grasp moments are the ones control
    has to get right.

Frames closer than --min-gap to an already-chosen frame are dropped (the
grasp-moment frames win). Episodes are listed in a pilot-first order: the first
--pilot episodes are spread across the dataset, so labelling can stop after the
pilot and still be representative.

    python scripts/select_label_frames.py \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval
"""

import argparse
import json
import os

import numpy as np

from lone_track.episodes import EpisodeIndex

GRASP_PRE, GRASP_POST = 1, 10


def select_frames(actions, every=60, min_gap=6):
    """Sorted frame indices to label for one episode, from its (T,4) actions."""
    T = len(actions)
    grip = np.asarray(actions)[:, 3]
    changes = np.flatnonzero(grip[1:] != grip[:-1]) + 1
    priority = []
    for t in changes:
        priority += [t - GRASP_PRE, t + GRASP_POST]
    priority += [0, T - 1]
    regular = list(range(every, T - 1, every))

    chosen = []
    for f in priority + regular:
        if not 0 <= f < T:
            continue
        if all(abs(f - c) >= min_gap for c in chosen):
            chosen.append(int(f))
    return sorted(chosen)


def pilot_first_order(episode_ids, pilot):
    """Episode ids with `pilot` evenly spread ones first, then the rest in order."""
    ids = list(episode_ids)
    if pilot <= 0 or pilot >= len(ids):
        return ids
    picks = [ids[int(round(i))] for i in np.linspace(0, len(ids) - 1, pilot)]
    return picks + [e for e in ids if e not in picks]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--every", type=int, default=60)
    p.add_argument("--min-gap", type=int, default=6)
    p.add_argument("--pilot", type=int, default=5)
    p.add_argument("--out", default=None,
                   help="default: labels/label_frames_<dataset name>.json")
    args = p.parse_args()

    index = EpisodeIndex(args.repo_id, args.root)
    name = args.repo_id.split("/")[-1]
    out = args.out or os.path.join("labels", f"label_frames_{name}.json")

    frames = {}
    for ep in index.episodes:
        frames[str(ep.index)] = select_frames(index.actions(ep.index),
                                              every=args.every, min_gap=args.min_gap)
    order = pilot_first_order([ep.index for ep in index.episodes], args.pilot)
    doc = {
        "repo_id": args.repo_id,
        "every": args.every,
        "min_gap": args.min_gap,
        "pilot": order[:args.pilot],
        "order": order,
        "frames": frames,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    n = sum(len(v) for v in frames.values())
    n_pilot = sum(len(frames[str(e)]) for e in order[:args.pilot])
    print(f"{out}: {n} frames over {len(frames)} episodes "
          f"({n / len(frames):.1f}/episode); pilot episodes {order[:args.pilot]} = "
          f"{n_pilot} frames x {len(index.views)} views")


if __name__ == "__main__":
    main()
