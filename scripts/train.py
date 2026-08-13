#!/usr/bin/env python3
"""lerobot-train with this repo's defaults applied.

    python scripts/train.py --policy.type=act --dataset.repo_id=lone/l_one ...

Takes exactly the arguments lerobot-train takes and forwards them untouched -- see
the README for the ACT and pi0.5 invocations. It only sets three defaults that
would otherwise have to be repeated on every run.
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

from lerobot.scripts.lerobot_train import main

# push_to_hub defaults to True, so TrainPipelineConfig.validate() refuses to start
# without a policy.repo_id to push the trained model to. An explicit
# --policy.push_to_hub=true still wins. This concerns the checkpoint, not the dataset.
if not any(a.split("=")[0] == "--policy.push_to_hub" for a in sys.argv[1:]):
    sys.argv.append("--policy.push_to_hub=false")

if __name__ == "__main__":
    main()
