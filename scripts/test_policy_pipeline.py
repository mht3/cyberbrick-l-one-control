#!/usr/bin/env python3
"""Prove the L-ONE dataset is a policy-agnostic LeRobotDataset.

    python scripts/test_policy_pipeline.py [--root data/lerobot/lone/l_one_marker_pickup]

Runs on CPU with no weights and no network: it reproduces exactly what
make_policy() does when wiring dataset features into a policy config
then checks what each policy makes of an RGB-only
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
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.constants import ACTION, OBS_STATE

from lone_data.features import ACTION_DIM, CAMERA_KEY, STATE_DIM

DEFAULT_ROOT = "data/lerobot/lone/l_one_marker_pickup"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT, help="dataset directory")
    p.add_argument("--repo-id", default="lone/l_one_marker_pickup", help="dataset repo id")
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
    check("STATE feature reaches policies", FeatureType.STATE in types)

    # The state exists for its shape, not its contents. If it ever stops being
    # all zeros, something started writing commands into it -- see STATE_NAMES.
    state = np.stack([np.asarray(ds[i][OBS_STATE]) for i in range(min(64, len(ds)))])
    check(
        "observation.state is all zeros",
        bool((state == 0).all()),
        f"shape={tuple(state.shape)}",
    )

    # -- pi0.5 ------------------------------------------------------------
    print("\npi05 (the intended training path)")
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    pi05 = wire(PI05Config(empty_cameras=2), features)
    empty_cams = [k for k in pi05.input_features if "empty_camera" in k]
    check("PI05Config fills the 2 unused camera slots", len(empty_cams) == 2, str(empty_cams))
    check("PI05Config sees our camera", CAMERA_KEY in pi05.input_features)
    check("PI05Config sees our state", OBS_STATE in pi05.input_features)
    check("PI05Config outputs action", ACTION in pi05.output_features)

    # pi0.5 discretizes the state into the text prompt and raises without one, so
    # a config that merely declares the feature is not enough -- assert that the
    # dataset's own tensor satisfies the runtime, and that a constant state
    # contributes only a constant prefix.
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TransitionKey

    step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=pi05.max_state_dim)
    transition = {
        TransitionKey.OBSERVATION: {
            CAMERA_KEY: torch.zeros(1, 3, 224, 224),
            OBS_STATE: torch.zeros(1, STATE_DIM),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Pick up the green marker."]},
    }
    prompt = step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"][0]
    bins = set(prompt.split("State: ")[1].split(";")[0].split())
    check(
        "pi05 tokenizes the state into a constant prompt prefix",
        bins == {"128"},
        f"{len(bins)} distinct bin value(s): {sorted(bins)}",
    )

    # -- ACT --------------------------------------------------------------
    print("\nact (cheaper baseline)")
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # pretrained_backbone_weights would fetch resnet18; keep this script offline.
    act = wire(ACTConfig(pretrained_backbone_weights=None, chunk_size=8, n_action_steps=8,
                         device="cpu"), features)
    check("ACTConfig sees our camera", CAMERA_KEY in act.input_features)
    check(
        "ACTConfig sees our state",
        act.robot_state_feature is not None
        and tuple(act.robot_state_feature.shape) == (STATE_DIM,),
    )

    # A clean config is necessary but not sufficient -- run a real batch through.
    act_policy = ACTPolicy(act)
    act_batch = torch.utils.data.default_collate(
        [
            {CAMERA_KEY: torch.zeros(3, 224, 224), OBS_STATE: torch.zeros(STATE_DIM),
             ACTION: torch.zeros(act.chunk_size, ACTION_DIM),
             "action_is_pad": torch.zeros(act.chunk_size, dtype=torch.bool)}
            for _ in range(2)
        ]
    )
    act_policy.train()
    loss, _ = act_policy.forward(act_batch)
    check("ACT trains on this schema", torch.isfinite(loss).item(), f"loss={float(loss):.3f}")

    # -- policies that read the state into the model ----------------------
    # These build fine now that the column exists, but the column is zeros, so
    # what they gain is a shape rather than information.
    print("\npolicies that consume the state directly")
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel

    dp = wire(DiffusionConfig(), features)
    check(
        "DiffusionConfig sees our state",
        dp.robot_state_feature is not None
        and tuple(dp.robot_state_feature.shape) == (STATE_DIM,),
    )
    try:
        DiffusionModel(dp)
        check("DiffusionModel builds", True)
    except ImportError as e:
        # lerobot[diffusion] is not installed -- nothing here needs it.
        print(f"  SKIP  DiffusionModel builds  {e}")
    except Exception as e:
        check("DiffusionModel builds", False, f"{type(e).__name__}: {e}")

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    smolvla = wire(SmolVLAConfig(), features)
    check("SmolVLAConfig sees our state", smolvla.robot_state_feature is not None)

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
        "\nAll checks passed -- a standard LeRobotDataset every policy above accepts.\n"
        "observation.state is all zeros: L-ONE measures nothing, so the column carries the\n"
        "shape policies require and no information. Every L-ONE policy is vision-only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
