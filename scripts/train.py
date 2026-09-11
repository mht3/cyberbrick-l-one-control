#!/usr/bin/env python3
"""lerobot-train with this repo's defaults and a held-out set from a separate folder.

    python scripts/train.py --policy.type=act --dataset.repo_id=lone/l_one_train ...

Takes exactly the arguments lerobot-train takes and forwards them untouched -- see
the README for the pi0.5 invocation. It only sets defaults that would otherwise
have to be repeated on every run, and adds one thing lerobot has no equivalent
for:

    --eval-dataset.repo-id / --eval-dataset.root

lerobot's own held-out split (`--dataset.eval_split`) carves episodes out of the
training dataset. That is not what we want here: demonstrations recorded in one
session share lighting, start pose and marker placement, so episodes held out of
the same folder measure memorization far more than generalization. Pointing eval
at a separately collected folder makes the eval loss mean what it should. The two
datasets must share a schema -- same fps, same camera views, same action space --
since the policy is scored with the preprocessor it trained with.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- sets HF_HUB_OFFLINE before lerobot is imported

# The PaliGemma tokenizer is built once in the main process, then --num_workers forks
# the dataloader, and HF's Rust tokenizer warns once per fork. Nothing tokenizes inside
# a worker, so the parallel path buys nothing. Must precede any tokenizers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# lone_data disables Hub access for every entry point in the repo. Training is the one
# that legitimately needs it, for pi05_base and the gated PaliGemma tokenizer, so undo
# it here. LONE_TRAIN_OFFLINE=1 keeps the offline default -- useful for ACT, which needs
# no Hub access at all.
if os.environ.get("LONE_TRAIN_OFFLINE") != "1":
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

# How often to score the held-out set when one is given and --eval_steps is not.
DEFAULT_EVAL_STEPS = 10000


def take_flag(name):
    """Pop `--name=value` (or `--name value`) out of sys.argv and return the value.

    Done by hand rather than with argparse because everything else on the command
    line belongs to draccus, which rejects any flag it does not recognise -- so
    these have to be gone before it ever sees argv.
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            del sys.argv[i + 1:i + 3]
            return args[i + 1]
        if arg.startswith(name + "="):
            del sys.argv[i + 1]
            return arg.split("=", 1)[1]
    return None


def has_flag(name):
    return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])


EVAL_REPO_ID = take_flag("--eval-dataset.repo-id")
EVAL_ROOT = take_flag("--eval-dataset.root")

from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402

# push_to_hub defaults to True, so TrainPipelineConfig.validate() refuses to start
# without a policy.repo_id to push the trained model to. An explicit
# --policy.push_to_hub=true still wins. This concerns the checkpoint, not the dataset.
if not has_flag("--policy.push_to_hub"):
    sys.argv.append("--policy.push_to_hub=false")


def _eval_dataset(cfg, train_dataset):
    """The held-out dataset, built the way lerobot builds its own eval split.

    Same delta_timestamps, same uint8 return and tolerance as the training set, and
    no image transforms -- an augmented eval set would score a different input
    distribution every time it ran. delta_timestamps come from the *training*
    metadata so both datasets are sliced identically even if the eval folder's
    metadata differs in some way it should not.
    """
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import IMAGENET_STATS

    dataset = LeRobotDataset(
        EVAL_REPO_ID or cfg.dataset.repo_id,
        root=EVAL_ROOT,
        delta_timestamps=resolve_delta_timestamps(cfg.trainable_config, train_dataset.meta),
        image_transforms=None,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return dataset


def _make_train_eval_datasets(cfg):
    """Replaces lerobot's splitter: train is the whole training folder, eval is ours."""
    from lerobot.datasets.factory import make_dataset

    train_dataset = make_dataset(cfg)
    eval_dataset = _eval_dataset(cfg, train_dataset)
    print(f"[train] held-out eval set: {EVAL_ROOT} "
          f"({eval_dataset.num_episodes} episode(s), {len(eval_dataset)} frames)")
    return train_dataset, eval_dataset


def _allow_eval_without_split():
    """Let eval_steps > 0 through with dataset.eval_split at 0.0.

    lerobot only knows about the in-dataset split and validate() refuses that
    combination, so eval_split is nudged non-zero for the duration of that call
    alone and restored afterwards -- the replaced splitter never reads it, and
    train_config.json keeps recording the 0.0 that is actually true.
    """
    original_validate = TrainPipelineConfig.validate

    def validate(self):
        saved = self.dataset.eval_split
        self.dataset.eval_split = saved or 1e-6
        try:
            original_validate(self)
        finally:
            self.dataset.eval_split = saved

    TrainPipelineConfig.validate = validate


if EVAL_ROOT or EVAL_REPO_ID:
    if not EVAL_ROOT:
        from lone_data.checkpoints import default_root

        EVAL_ROOT = default_root(EVAL_REPO_ID)
    lerobot_train.make_train_eval_datasets = _make_train_eval_datasets
    _allow_eval_without_split()
    if not has_flag("--eval_steps"):
        sys.argv.append(f"--eval_steps={DEFAULT_EVAL_STEPS}")

if __name__ == "__main__":
    lerobot_train.main()
