# cyberbrick-l-one-control
Custom control modules for the [CyberBrick L-One](https://us.store.bambulab.com/products/l-one-desktop-robotic-arm-cyberbrick-rc).

<p align="center">
  <img src="assets/l_one.png" alt="L-One Robot Arm" width="300">
</p>

## Setup

```sh
git clone git@github.com:mht3/cyberbrick-l-one-control.git
cd cyberbrick-l-one-control
conda create -n cyberbrick python=3.12 && conda activate cyberbrick
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128  # or /cpu
pip install -e ".[dev]"
conda install -c conda-forge "ffmpeg=7"
pytest tests/          # environment check; no hardware needed
```

## Cameras

Two views by default: **Cam1** (front) and **Cam2** (side), stored as
`observation.images.front` and `observation.images.side`. Every GUI shows Cam1
above Cam2, says how many views are selected, and lets you switch to 1 view for
a single-camera rig or a single-camera dataset. Both views are passed to the
policy; a checkpoint's own config decides how many it reads.

To run the cameras on a different machine from the GPU, stream them (Cam1 goes
to `--port`, Cam2 to the next port up) and add `--remote-camera` to any script
below:

```sh
python stream_camera.py --host <gpu-machine>          # sender: streams every camera it finds
```

## Data collection

```sh
python collect_data.py --repo-id lone/l_one_manipulation --fps 25 \
    --task "Move the green marker from the orange tape to the blue tape."
```

`Return` starts an episode, `F` finishes, `Backspace` discards, `R` reviews,
**Quit** exits. The dataset lands in `data/lerobot/<repo-id>/`; rerun with the
same `--repo-id` to append.

Split one recording into a training set and a held-out evaluation set — whole
episodes, evenly spaced, copied frame for frame:

```sh
python scripts/split_dataset.py \
    --src-repo-id lone/l_one_manipulation \
    --train-repo-id lone/l_one_manipulation_rebalanced_train \
    --eval-repo-id lone/l_one_manipulation_eval \
    --eval-episodes 5
```

Run after every session, before training — teleop is bang-bang, so statistics
estimated from recorded frames are degenerate:

```sh
ID=lone/l_one_manipulation_rebalanced_train; ROOT=data/lerobot/$ID
python scripts/fix_action_stats.py --root $ROOT
python scripts/validate_dataset.py --repo-id $ID --root $ROOT
python scripts/inspect_dataset.py  --repo-id $ID --root $ROOT
```

## Training

`scripts/train.py` is `lerobot-train` plus this repo's defaults, and adds two
things lerobot has no equivalent for: `--eval-dataset.root`, a held-out set from
a **separate folder** so the eval loss measures generalization rather than
memorization of the same session; and `--balance-actions`.

```sh
hf auth login   # pi0.5 only: google/paligemma-3b-pt-224 is gated

python scripts/train.py \
  --dataset.repo_id=lone/l_one_manipulation_rebalanced_train \
  --dataset.root=data/lerobot/lone/l_one_manipulation_rebalanced_train \
  --eval-dataset.repo-id=lone/l_one_manipulation_eval \
  --eval-dataset.root=data/lerobot/lone/l_one_manipulation_eval \
  --eval_steps=5000 \
  --balance-actions \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}' \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --policy.dtype=bfloat16 --policy.device=cuda \
  --batch_size=8 --num_workers=8 \
  --steps=250000 --policy.scheduler_decay_steps=250000 \
  --save_freq=50000 \
  --wandb.enable=true --wandb.project=lone --wandb.disable_artifact=true
```

- **`--balance-actions`** weights the training sampler so the five scenarios —
  base moving, upper arm moving, lower arm moving, gripper closed, no action —
  are drawn equally often. Teleop's own diet is badly skewed (10.7% / 16.2% /
  25.2% / 18.0% / 29.9% on the training split), and a policy can score well on
  that by learning the marginals instead of the task. It balances by *sampling*,
  never by deleting frames: idle frames are real supervision, and a chunked
  policy trains on runs of consecutive actions, so cutting frames out of an
  episode would leave chunks spanning stretched time and teach the motion too
  fast. It evens out the chunk *start* frames — the decision points — which
  lifts base motion over the executed 10-step window from 9.3% to 14.6% (1.57×);
  the full 50-step chunk shifts far less (1.13×), since a two-second chunk
  contains mostly common actions whichever frame it starts from. Chunks stay
  intact, so this works the same for pi0.5 and for ACT's L1/MSE regression.
- The two datasets must share a schema — same fps, same camera views, same
  action space.
- `--policy.scheduler_decay_steps` must track `--steps`; it defaults to 30000.
- `--policy.normalization_mapping` must be passed as the whole dict.
- `--policy.pretrained_path` loads weights only and resets the config, which is
  why `n_action_steps` and `dtype` are spelled out.

Checkpoints land in `outputs/train/<date>/<time>_pi05/checkpoints/`, with a
`last/` symlink. Resume with
`--resume=true --config_path=<checkpoint>/pretrained_model/train_config.json`.

## Evaluation

Scores a checkpoint against recorded episodes, no hardware. Point it at the
held-out folder:

```sh
python scripts/eval_policy.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_manipulation_eval \
  --root data/lerobot/lone/l_one_manipulation_eval
```

Read **balanced accuracy per dimension** first: teleop only ever emitted three
levels per channel, so a policy that ignores the image and emits one constant
scores exactly chance (1/k) on it — which neither plain accuracy nor MSE does.
Then the confusion matrix for any dimension near chance. `--json <path>` writes
the same figures machine-readably.

## Deployment

```sh
python deploy_policy.py --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model
```

Connect over Serial or WiFi, teleoperate the arm to a start pose, confirm the
task prompt, then hand over to the policy. **STOP ALL** and `<space>` stop the
arm at any time. Each run writes `results/deploy_<timestamp>/` with
`video_cam1.mp4` (and `video_cam2.mp4`), `actions.jsonl` and `run.json`.

- The task prompt is prefilled from the checkpoint — pi0.5 conditions on it, so
  a differently worded prompt is different conditioning.
- *Snap to demonstrated levels* (default) quantizes the policy's output back
  onto the three levels teleop used; *raw* clamps to the command limits.
- `--rtc` enables [real-time chunking](https://www.physicalintelligence.company/research/real_time_chunking)
  for flow-matching policies (pi0/pi0.5/SmolVLA).
- On a fresh machine, prime the Hub cache once:
  `HF_HUB_OFFLINE=0 python deploy_policy.py --checkpoint <checkpoint>`.

### Replaying training data through the policy

Same GUI with the cameras replaced by a recorded episode — reality on the left,
demonstration on the right, both views in each column:

```sh
python deploy_policy_training_data_replay.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_manipulation_rebalanced_train \
  --root data/lerobot/lone/l_one_manipulation_rebalanced_train
```

Run three configurations in order, each isolating a different failure:
`--source dataset` (recorded actions, no policy — needs no checkpoint),
`--source policy --lookahead 0` (what deployment does), then
`--source policy --lookahead 10` (inference lag dialled out). The arm must start
where the demonstration started, so scrub to frame 0 and teleoperate until the
panes agree before pressing Start.

## Layout

- [`collect_data.py`](collect_data.py) — teleop GUI that records into a LeRobotDataset.
- [`deploy_policy.py`](deploy_policy.py) — runs a checkpoint on the arm.
- [`deploy_policy_training_data_replay.py`](deploy_policy_training_data_replay.py) — the same GUI fed a recorded episode.
- [`virtual_gripper.py`](virtual_gripper.py) — teleop only, over USB or WiFi.
- [`stream_camera.py`](stream_camera.py) — serves webcams over the network.
- [`lone_data/`](lone_data/) — dataset schema, LeRobot writer, cameras, the shared GUI base, command bus, dispatch.
- [`scripts/`](scripts/) — `train.py`, `eval_policy.py`, `split_dataset.py`, `fix_action_stats.py`, `validate_dataset.py`, `inspect_dataset.py`, `add_state_column.py`, `test_policy_pipeline.py`, `test_rtc.py`.
- [`LOneGripper/`](LOneGripper/), [`LOneRC/`](LOneRC/) — receiver and transmitter firmware for the CyberBrick core board.

`data/`, `outputs/`, `results/` and `logs/` are gitignored.

### Dataset schema

| feature | dtype | shape | |
|---|---|---|---|
| `observation.images.front` | `video` | (360, 640, 3) | Cam1 — AV1, downscaled from 1280x720 |
| `observation.images.side` | `video` | (360, 640, 3) | Cam2 — omitted in one-camera datasets |
| `observation.state` | `float32` | (4,) | joint angles — always zeros; L-ONE has no encoders |
| `action` | `float32` | (4,) | the command sent to the board |

| i | name | channel | teleop emits | hardware range |
|---|------|---------|------|-------|
| 0 | `base_motor_speed` | MOTOR1 | -900 / 0 / 900 | -2048..2048 raw PWM |
| 1 | `upper_arm_servo_speed` | PWM1 | -100 / 0 / 100 | -100..100 percent |
| 2 | `lower_arm_servo_speed` | PWM2 | -100 / 0 / 100 | -100..100 percent |
| 3 | `gripper_angle` | PWM3 | 30 / 120 | 0..180 degrees |

`0` on dims 0-2 is dispatched as `stop_motor()`/`stop_servo()`, not
`set_speed(idx, 0)` — different hardware states, and any replay has to reproduce
that. See [`lone_data/features.py`](lone_data/features.py) for the rest.

## Firmware

1. Copy `LOneGripper/wifi_secrets.example.py` to `wifi_secrets.py` and fill in your credentials.
2. Upload `LOneGripper/` (receiver) or `LOneRC/` (transmitter) to the core board, e.g. with [Pymakr](https://marketplace.visualstudio.com/items?itemName=pycom.Pymakr) or `mpremote`.
3. Power on the transmitter first, then the receiver, so they pair over ESP-NOW. With no transmitter, the receiver falls back to WiFi after 15s so `virtual_gripper.py` can reach it.

Both firmware folders build on [CyberBrick's `CyberBrick_Controller_Core`](https://github.com/CyberBrick-Official/CyberBrick_Controller_Core); see [`LICENSE`](LICENSE).
