#!/usr/bin/env python3
"""Prove the L-ONE dataset is a policy-agnostic LeRobotDataset.

    python scripts/test_policy_pipeline.py [--root data/lerobot/lone/l_one]

Runs on CPU with no weights and no network: it reproduces exactly what
make_policy() does when wiring dataset features into a policy config
(factory.py:457-471), then checks what each policy makes of an RGB-only
dataset. Exits non-zero if anything doesn't match expectations.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- disables Hub access before lerobot is imported

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.features import ACTION_DIM, CAMERA_KEY

DEFAULT_ROOT = "data/lerobot/lone/l_one"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--repo-id", default="lone/l_one", help="dataset repo id")
    return p.parse_args()


def wire(config, features):
    """Same wiring make_policy() performs before a policy is built."""
    config.output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    config.input_features = {k: ft for k, ft in features.items() if k not in config.output_features}
    config.validate_features()
    return config


def main():
    args = parse_args()
    failures = []

    def check(name, ok, detail=""):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    ds = LeRobotDataset(args.repo_id, root=args.root)
    features = dataset_to_policy_features(ds.meta.features)

    print("dataset features exposed to policies")
    for key, ft in features.items():
        print(f"  {key:32s} {ft.type.name:8s} {tuple(ft.shape)}")

    print("\nschema")
    types = {ft.type for ft in features.values()}
    check("VISUAL feature reaches policies", FeatureType.VISUAL in types)
    check("ACTION feature reaches policies", FeatureType.ACTION in types)
    check("no observation.state in dataset", OBS_STATE not in ds.meta.features)

    # -- pi0.5: the state slot is the policy's job, not the dataset's ------
    print("\npi05 (the intended training path)")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    pi05 = wire(PI05Config(empty_cameras=2), features)
    check(
        "PI05Config injects an observation.state feature",
        OBS_STATE in pi05.input_features
        and tuple(pi05.input_features[OBS_STATE].shape) == (pi05.max_state_dim,),
        f"shape={tuple(pi05.input_features[OBS_STATE].shape)}",
    )
    empty_cams = [k for k in pi05.input_features if "empty_camera" in k]
    check("PI05Config fills the 2 unused camera slots", len(empty_cams) == 2, str(empty_cams))
    check("PI05Config sees our camera", CAMERA_KEY in pi05.input_features)
    check("PI05Config outputs action", ACTION in pi05.output_features)

    # The config check above is necessary but NOT sufficient, and believing
    # otherwise is exactly the trap: validate_features() injects a state
    # *feature* (shapes, normalization wiring) while the runtime pipeline still
    # demands a state *tensor* in the batch. pi0.5 discretizes the state into
    # the text prompt, so it is structurally required. Assert the real behavior
    # here rather than discovering it partway into a training run.
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor.core import TransitionKey

    step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=pi05.max_state_dim)
    transition = {
        TransitionKey.OBSERVATION: {CAMERA_KEY: torch.zeros(1, 3, 224, 224)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Pick up the green marker."]},
    }
    try:
        step(transition)
        raised = None
    except ValueError as e:
        raised = str(e)
    check(
        "pi05 runtime requires a state tensor the dataset does not have",
        raised is not None and "State is required" in raised,
        f"ValueError: {raised}" if raised else "no error -- lerobot made state optional, "
        "so this dataset can now train pi05 unmodified; drop this check",
    )

    # -- ACT: no injection, but every state use is guarded ----------------
    print("\nact (vision-only baseline)")
    from lerobot.policies.act.configuration_act import ACTConfig

    act = wire(ACTConfig(), features)
    check("ACTConfig accepts a state-less dataset", act.robot_state_feature is None)
    check("ACTConfig sees our camera", CAMERA_KEY in act.input_features)

    # -- policies that genuinely need proprioception ----------------------
    print("\npolicies that require proprioception (expected to be unusable)")
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

    dp = wire(DiffusionConfig(), features)
    check("DiffusionConfig has no robot_state_feature", dp.robot_state_feature is None)
    try:
        DiffusionModel(dp)
        check("DiffusionModel rejects a state-less dataset", False, "it built without error")
    except AttributeError as e:
        check("DiffusionModel rejects a state-less dataset", True, f"AttributeError: {e}")

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    smolvla = wire(SmolVLAConfig(), features)
    check(
        "SmolVLAConfig has no robot_state_feature",
        smolvla.robot_state_feature is None,
        "(modeling_smolvla reads batch[OBS_STATE] unconditionally)",
    )

    # -- a real batch, the shape training consumes ------------------------
    print("\nbatch")
    horizon = min(30, ds.meta.total_frames)
    chunked = LeRobotDataset(
        args.repo_id, root=args.root,
        delta_timestamps={"action": [t / ds.meta.fps for t in range(horizon)]},
    )
    batch = torch.utils.data.default_collate([chunked[i] for i in range(min(4, len(chunked)))])
    img, action = batch[CAMERA_KEY], batch[ACTION]
    print(f"  {CAMERA_KEY:32s} {tuple(img.shape)} {img.dtype}")
    print(f"  {ACTION:32s} {tuple(action.shape)} {action.dtype}")
    print(f"  task                             {batch['task'][0]!r}")

    check("image is float32 CHW", img.dtype == torch.float32 and img.shape[1] == 3)
    check("image values in [0,1]", bool(img.min() >= 0 and img.max() <= 1),
          f"[{img.min():.3f}, {img.max():.3f}]")
    check("action chunk shape", tuple(action.shape[1:]) == (horizon, ACTION_DIM),
          str(tuple(action.shape)))
    check("no NaN/Inf", bool(torch.isfinite(img).all() and torch.isfinite(action).all()))
    check("task is a non-empty string", isinstance(batch["task"][0], str) and bool(batch["task"][0].strip()))

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        return 1
    print(
        "\nAll checks passed -- a standard LeRobotDataset, trainable as-is with ACT.\n"
        "pi0/pi0.5 additionally need an observation.state tensor supplied at train time;\n"
        "the dataset deliberately provides none (L-ONE has no proprioceptive sensors)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
