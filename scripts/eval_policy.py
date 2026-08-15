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

1. MSE vs the predict-the-mean baseline. A policy at or above baseline has learned
   nothing from the image; that is the single most useful signal here.
2. Prediction spread per dimension. A std near zero is mode collapse -- the policy
   emits one action regardless of input.
3. Same-input spread (--repeats). pi0.5 is a flow-matching model that draws fresh
   noise every call, so repeated predictions on an identical frame genuinely differ.
   That variance is the policy, not measurement error, and a large MSE alongside a
   large same-input spread means something different from a large MSE alone.

Raw per-dimension numbers are in command units and are the readable ones. The
*average* is reported normalized by ACTION_COMMAND_LIMITS, because dim 0 spans
+/-900 while dim 3 spans 90 -- a raw average would be almost entirely dim 0.
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
from lerobot.utils.constants import ACTION

from lone_data.features import ACTION_COMMAND_LIMITS, ACTION_NAMES

DEFAULT_ROOT = "data/lerobot/lone/l_one_marker_pickup"
SPANS = np.array([hi - lo for lo, hi in ACTION_COMMAND_LIMITS], dtype=np.float64)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True, help="a checkpoint's pretrained_model/ directory")
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--repo-id", default="lone/l_one_marker_pickup", help="dataset repo id")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    p.add_argument("--max-frames", type=int, default=200, help="cap on frames scored")
    p.add_argument("--repeats", type=int, default=8,
                   help="re-predictions per frame for the same-input spread (0 disables)")
    p.add_argument("--repeat-frames", type=int, default=4, help="frames used for --repeats")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="also write results to this path")
    return p.parse_args()


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
        for key in ("observation.images.front", "observation.state"):
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
    print(f"dataset     {args.repo_id}  {len(ds)} frames, scoring {n}")
    print("\nTeacher-forced open-loop evaluation: fit to demonstrations, NOT task success.")

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

    one_step_raw = ((pred_first - true_first) ** 2).mean(0)
    one_step_norm = (((pred_first - true_first) / SPANS) ** 2).mean(0)
    # Baseline: the best a policy can do while ignoring the image entirely.
    mean_action = true_first.mean(0, keepdims=True)
    base_raw = ((mean_action - true_first) ** 2).mean(0)
    base_norm = (((mean_action - true_first) / SPANS) ** 2).mean(0)

    print("\n" + "=" * 66)
    print("ONE-STEP MSE (raw command units, per dimension)")
    print("=" * 66)
    print(f"  {'dimension':24s} {'policy':>12s} {'baseline':>12s}   verdict")
    for j, name in enumerate(ACTION_NAMES):
        if true_first[:, j].std() < 1e-9:
            # The mean predicts a constant perfectly, so the baseline is 0 and nothing
            # can beat it. That says the sample is uninformative for this dimension --
            # too few frames, or a joint unused in them -- not that the policy failed.
            verdict = "ground truth constant -- uninformative sample"
        elif one_step_raw[j] < base_raw[j]:
            verdict = "learned something"
        else:
            verdict = "no better than mean"
        print(f"  {name:24s} {one_step_raw[j]:12.3f} {base_raw[j]:12.3f}   {verdict}")
    print(f"  {'-- average (normalized)':24s} {one_step_norm.mean():12.4f} {base_norm.mean():12.4f}")

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
