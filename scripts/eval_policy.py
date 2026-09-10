#!/usr/bin/env python3
"""Score a trained checkpoint against recorded demonstrations.

    python scripts/eval_policy.py --checkpoint outputs/train/.../checkpoints/last/pretrained_model
    python scripts/eval_policy.py --checkpoint <ck> --episodes 0,1 --repeats 8 --json out.json

Loads the checkpoint exactly as deployment does, so the normalization statistics
frozen in at training time are the ones used -- never recomputed from the dataset.

WHAT THIS MEASURES, AND WHAT IT DOES NOT: this is teacher-forced open-loop
evaluation. Every prediction is made from a *recorded* frame, and the policy's own
actions are never fed back into the next observation. It measures how well the
policy fits the demonstrations. It cannot see compounding error, so a policy can
score well here and still fail on hardware.

Read the numbers in this order:

1. Balanced accuracy per dimension. Teleop emits only the discrete levels in
   features.ACTION_LEVELS, so predictions are snapped to those levels and scored
   as a classification. Balanced accuracy -- mean recall over the classes present
   in the ground truth -- is the verdict signal, because a policy that ignores the
   image and emits one constant scores exactly at chance (1/k) no matter how
   skewed the dimension is. Plain accuracy does not have that property: dim 0 is
   stopped in 90.6% of frames, so constant-zero scores 0.906 there.
2. The confusion matrix for any dimension whose balanced accuracy is near chance.
   It says immediately whether the policy is collapsing to one class, flipping
   direction, or getting direction right and timing wrong.
3. Prediction spread per dimension. A std near zero is mode collapse -- the policy
   emits one action regardless of input.
4. Same-input spread (--repeats). pi0.5 is a flow-matching model that draws fresh
   noise every call, so repeated predictions on an identical frame genuinely differ.
   That variance is the policy, not measurement error, and a large error alongside
   a large same-input spread means something different from a large error alone.
5. MSE, read last and without a verdict. It is reported because it is what the
   policy was trained on and it is comparable across runs, but on a bang-bang
   action space it is not a fitness test: squared error rewards hedging toward the
   mean on rare high-amplitude classes, so a do-nothing policy sits at the
   predict-the-mean baseline while a policy with the right direction and one frame
   of timing error scores far worse. Read balanced accuracy for the verdict.

Raw per-dimension numbers are in command units and are the readable ones. The
*average* is reported normalized by ACTION_COMMAND_LIMITS, because dim 0 spans
+/-900 while dim 3 spans 90 -- a raw average would be almost entirely dim 0.

Sample size matters more than it looks. Dimension 0 is non-zero in under 10% of
frames, so the default --max-frames is set high enough that the rare classes get
a usable number of examples; scoring 200 frames leaves ~16 base-motion events and
any verdict on dim 0 is then sampling noise.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- sets the offline flags before lerobot is imported

# The checkpoint may reference the Hub (pi0.5's tokenizer); this script is allowed
# to fetch it, like scripts/train.py.
if os.environ.get("LONE_EVAL_OFFLINE") != "1":
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.checkpoints import default_root, same_dataset, training_dataset
from lone_data.features import (
    ACTION_COMMAND_LIMITS,
    ACTION_LEVEL_LABELS,
    ACTION_LEVELS,
    ACTION_NAMES,
    dataset_camera_keys,
)
from lone_data.metrics import score_dimension, verdict

SPANS = np.array([hi - lo for lo, hi in ACTION_COMMAND_LIMITS], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="a checkpoint's pretrained_model/ directory")
    p.add_argument("--root", default=None,
                   help="dataset directory (default: data/lerobot/<repo-id>)")
    p.add_argument("--repo-id", required=True, help="dataset repo id") # e.g. lone/l_one_marker_pickup
    p.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    p.add_argument("--max-frames", type=int, default=2000,
                   help="cap on frames scored. The rare classes set this: dim 0 is "
                        "non-zero in under 10%% of frames, so 2000 frames buys ~190 "
                        "base-motion events and 200 would buy ~16. Lower it only for "
                        "a smoke test, and do not read a per-dimension verdict off one.")
    p.add_argument("--repeats", type=int, default=8,
                   help="re-predictions per frame for the same-input spread (0 disables)")
    p.add_argument("--repeat-frames", type=int, default=4, help="frames used for --repeats")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write results to this path")
    args = p.parse_args()
    # LeRobotDataset loads from `root` and treats `repo_id` as a label when the
    # data is local, so a --root left at some other dataset's default is read
    # without complaint and scores the checkpoint against episodes it never saw.
    # Deriving it from --repo-id removes the chance for the two to disagree.
    if args.root is None:
        args.root = default_root(args.repo_id)
    return args


def warn_if_not_training_dataset(checkpoint, root):
    """Say so when the dataset being scored is not the one the checkpoint trained on.

    Not an error -- scoring against held-out data is a legitimate thing to do --
    but it changes what the numbers mean completely, and the two cases look
    identical in the output otherwise.
    """
    trained_repo_id, trained_root = training_dataset(checkpoint)
    if trained_root is None:
        return
    if same_dataset(trained_root, root):
        return
    print(f"\nNOTE: this checkpoint trained on {trained_repo_id} ({trained_root}),")
    print(f"      but is being scored against {root}.")
    print("      These numbers measure generalization, not fit to the training data.")


def load_policy(checkpoint, device):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint)
    policy.eval().to(device)
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
    return cfg, policy, pre, post


def build_dataset(args, cfg, fps):
    """delta_timestamps mirrors what the training run used, read off the checkpoint."""
    delta = {ACTION: [i / fps for i in cfg.action_delta_indices]}
    if getattr(cfg, "observation_delta_indices", None):
        # The checkpoint's own image keys, so a two-camera policy gets a delta for
        # both views and a one-camera policy is unaffected.
        keys = [*dataset_camera_keys(getattr(cfg, "input_features", {}) or {}), OBS_STATE]
        for key in keys:
            delta[key] = [i / fps for i in cfg.observation_delta_indices]
    episodes = None
    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    return LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta, episodes=episodes)


def predict(policy, pre, post, sample):
    """One action chunk, in raw command units, for a single dataset sample."""
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sample.items()}
    with torch.no_grad():
        chunk = policy.predict_action_chunk(pre(batch))
    return post(chunk)[0].float().cpu().numpy()


def summarize(values):
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
    }


def report_classification(true_first, pred_first):
    """The headline block: snap both to the discrete levels, score the confusion.

    See lone_data/metrics.py for why this and not squared error.
    """
    print("\n" + "=" * 78)
    print("CLASSIFICATION (predictions snapped to the discrete levels teleop emits)")
    print("=" * 78)
    print(f"  {'dimension':24s} {'balanced':>9s} {'chance':>7s} "
          f"{'accuracy':>9s} {'majority':>9s}   verdict")

    per_dim = [score_dimension(true_first[:, j], pred_first[:, j], ACTION_LEVELS[j])
               for j in range(len(ACTION_NAMES))]
    for name, sc in zip(ACTION_NAMES, per_dim):
        print(f"  {name:24s} {sc['balanced_accuracy']:9.3f} {sc['chance']:7.3f} "
              f"{sc['accuracy']:9.3f} {sc['majority_accuracy']:9.3f}   {verdict(sc)}")
    mean_balanced = np.nanmean([sc["balanced_accuracy"] for sc in per_dim])
    print(f"  {'-- mean (balanced)':24s} {mean_balanced:9.3f}")
    print("\n  balanced = mean recall over the classes present; chance = 1/k, which is also")
    print("  what a policy emitting one constant scores. majority = that constant's plain")
    print("  accuracy, shown to make clear why plain accuracy cannot be the verdict either.")

    print("\n" + "-" * 78)
    print("CONFUSION (rows = ground truth, columns = prediction)")
    print("-" * 78)
    for j, name in enumerate(ACTION_NAMES):
        labels = ACTION_LEVEL_LABELS[j]
        cm = np.array(per_dim[j]["confusion"])
        recall = per_dim[j]["recall"]
        print(f"\n  {name}")
        print(" " * 12 + "".join(f"{l:>9s}" for l in labels) + f"{'recall':>9s}")
        for r, label in enumerate(labels):
            row = "".join(f"{v:9d}" for v in cm[r])
            rec = "      n/a" if np.isnan(recall[r]) else f"{recall[r]:9.3f}"
            print(f"    {label:>8s}{row}{rec}")
    return per_dim


def table(title, per_dim, average=None, width=24):
    print(f"\n{title}")
    for name, value in zip(ACTION_NAMES, per_dim):
        print(f"  {name:{width}s} {value:12.4f}")
    if average is not None:
        print(f"  {'-- average (normalized)':{width}s} {average:12.4f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg, policy, pre, post = load_policy(args.checkpoint, args.device)
    ds0 = LeRobotDataset(args.repo_id, root=args.root)
    ds = build_dataset(args, cfg, ds0.meta.fps)

    horizon = len(cfg.action_delta_indices)
    n = min(args.max_frames, len(ds))
    indices = np.linspace(0, len(ds) - 1, n).astype(int)

    print(f"checkpoint  {args.checkpoint}")
    print(f"policy      {cfg.type}   chunk {horizon}   device {args.device}")
    print(f"dataset     {args.repo_id}  {args.root}")
    print(f"            {ds0.meta.total_episodes} episodes, {len(ds)} frames, scoring {n}")
    print("\nTeacher-forced open-loop evaluation: fit to demonstrations, NOT task success.")
    warn_if_not_training_dataset(args.checkpoint, args.root)

    pred_first, true_first, chunk_se, kept = [], [], [], 0
    for i in indices:
        sample = ds[int(i)]
        true_chunk = np.asarray(sample[ACTION], dtype=np.float64)
        pred_chunk = predict(policy, pre, post, sample).astype(np.float64)
        pred_first.append(pred_chunk[0])
        true_first.append(true_chunk[0])
        # action_is_pad marks frames whose horizon runs past the episode end; scoring
        # those would compare against repeated final actions rather than real ones.
        pad = sample.get("action_is_pad")
        if pad is not None and bool(pad.any()):
            continue
        chunk_se.append(((pred_chunk - true_chunk) / SPANS) ** 2)
        kept += 1

    pred_first = np.stack(pred_first)
    true_first = np.stack(true_first)

    # The verdict block. Everything below it is a diagnostic for reading it.
    class_stats = report_classification(true_first, pred_first)

    one_step_raw = ((pred_first - true_first) ** 2).mean(0)
    one_step_norm = (((pred_first - true_first) / SPANS) ** 2).mean(0)
    # Baseline: the best a policy can do while ignoring the image entirely.
    mean_action = true_first.mean(0, keepdims=True)
    base_raw = ((mean_action - true_first) ** 2).mean(0)
    base_norm = (((mean_action - true_first) / SPANS) ** 2).mean(0)

    print("\n" + "=" * 78)
    print("ONE-STEP MSE (raw command units, per dimension) -- diagnostic, not a verdict")
    print("=" * 78)
    print(f"  {'dimension':24s} {'policy':>12s} {'baseline':>12s}")
    for j, name in enumerate(ACTION_NAMES):
        print(f"  {name:24s} {one_step_raw[j]:12.3f} {base_raw[j]:12.3f}")
    print(f"  {'-- average (normalized)':24s} {one_step_norm.mean():12.4f} {base_norm.mean():12.4f}")
    print("\n  No verdict column: on a bang-bang action space, beating the predict-the-mean")
    print("  baseline is neither necessary nor sufficient for having learned the dimension.")
    print("  Constant-zero sits at baseline on dim 0 while learning nothing, and a policy")
    print("  with the right direction and a frame of timing error scores far above it.")
    print("  Read the balanced accuracy above instead.")

    chunk_norm = np.stack(chunk_se).mean(axis=(0, 1)) if chunk_se else None
    if chunk_norm is not None:
        table(f"CHUNK MSE over {horizon} steps (normalized, {kept} unpadded frames)",
              chunk_norm, chunk_norm.mean())
    else:
        print(f"\nCHUNK MSE: skipped -- every sampled frame's {horizon}-step horizon was padded")

    print("\n" + "=" * 66)
    print("PREDICTION DISTRIBUTION vs GROUND TRUTH (raw units)")
    print("=" * 66)
    pred_stats, true_stats = summarize(pred_first), summarize(true_first)
    print(f"  {'dimension':24s} {'pred mean':>10s} {'pred std':>10s} {'true mean':>10s} {'true std':>10s}")
    for j, name in enumerate(ACTION_NAMES):
        print(f"  {name:24s} {pred_stats['mean'][j]:10.2f} {pred_stats['std'][j]:10.2f} "
              f"{true_stats['mean'][j]:10.2f} {true_stats['std'][j]:10.2f}")
    for j, name in enumerate(ACTION_NAMES):
        if pred_stats["std"][j] < 1e-6:
            print(f"  WARNING: {name} predictions are constant -- mode collapse")

    repeat_std = None
    if args.repeats > 1:
        frames = indices[np.linspace(0, len(indices) - 1, min(args.repeat_frames, len(indices))).astype(int)]
        spreads = []
        for i in frames:
            sample = ds[int(i)]
            draws = np.stack([predict(policy, pre, post, sample)[0] for _ in range(args.repeats)])
            spreads.append(draws.std(0))
        repeat_std = np.stack(spreads).mean(0)
        table(f"SAME-INPUT SPREAD (std over {args.repeats} draws, {len(frames)} frames, raw units)",
              repeat_std)
        if repeat_std.max() < 1e-6:
            print("  deterministic -- repeated predictions are identical")
        else:
            print("  non-zero spread is expected for pi0.5: flow matching resamples noise per call")

    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "policy_type": cfg.type,
            "frames_scored": int(n),
            "action_names": ACTION_NAMES,
            "action_levels": [list(l) for l in ACTION_LEVELS],
            "classification": class_stats,
            "one_step_mse_raw": one_step_raw.tolist(),
            "one_step_mse_normalized": one_step_norm.tolist(),
            "baseline_mse_raw": base_raw.tolist(),
            "baseline_mse_normalized": base_norm.tolist(),
            "chunk_mse_normalized": chunk_norm.tolist() if chunk_norm is not None else None,
            "prediction": pred_stats,
            "ground_truth": true_stats,
            "same_input_std": repeat_std.tolist() if repeat_std is not None else None,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
