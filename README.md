# cyberbrick-l-one-control
Custom control modules for the [CyberBrick L-One](https://us.store.bambulab.com/products/l-one-desktop-robotic-arm-cyberbrick-rc).

<p align="center">
  <img src="assets/l_one.png" alt="L-One Robot Arm" width="300">
</p>


## Getting Started

Clone the environment and change directories. The following uses cloning via ssh:

```bash
git clone git@github.com:mht3/cyberbrick-l-one-control.git
cd cyberbrick-l-one-control
```

### Environment Setup

Create a new conda environment with Python 3.12.
```bash
conda create -n cyberbrick python=3.12
```

Activate the environment.
```sh
conda activate cyberbrick
```

Install torch

<details>
<summary>PyTorch on GPU</summary>
<br>
Install a CUDA enabled PyTorch that matches your system architecture.
  
```sh
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
</details>

<details>
<summary>PyTorch on CPU Only</summary>
<br>
Alternatively, install PyTorch on the CPU.
  
```sh
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```
</details>


Install the package and its dependencies in editable mode.

```sh
pip install -e ".[dev]"
```

Install FFmpeg with conda.

```sh
conda install -c conda-forge "ffmpeg=7"
```

Run the tests to make sure the codebase is setup properly. If all tests pass, you're good to go!

```sh
pytest tests/
```

## Project Layout

- [`LOneGripper/`](LOneGripper/) -- receiver firmware (the arm/gripper) for the CyberBrick core board.
- [`LOneRC/`](LOneRC/) -- transmitter firmware for the handheld remote.
- [`virtual_gripper.py`](virtual_gripper.py) -- host-side GUI for driving the arm directly, either over USB (raw REPL) or over WiFi (via `LOneGripper/wifi_bridge.py`).
- [`test_motors.py`](test_motors.py) -- step-through hardware test for each configured motor/servo channel; reuses the USB link from `virtual_gripper.py`.
- [`collect_data.py`](collect_data.py) -- teleop GUI that records demonstrations into a LeRobotDataset.
- [`lone_data/`](lone_data/) -- dataset schema ([`features.py`](lone_data/features.py)), the LeRobot writer ([`lerobot_recorder.py`](lone_data/lerobot_recorder.py)), camera capture, and the command bus that keeps robot I/O off the record loop.
- [`scripts/`](scripts/) -- `train.py`, `fix_action_stats.py`, `add_state_column.py`, `inspect_dataset.py`, `validate_dataset.py`, `test_policy_pipeline.py`.

Both `LOneGripper/` and `LOneRC/` share a `bbl/` module (buzzer, motors, servos, LEDs, sleep) and an `app/` folder (`control`, `devices`, `parser`, `rc_main`) that build on [CyberBrick's `CyberBrick_Controller_Core`](https://github.com/CyberBrick-Official/CyberBrick_Controller_Core) firmware -- `bbl/` is used as-is from core, while `boot.py`/`rc_main.py` extend core's versions with WiFi-fallback behavior. See [`LICENSE`](LICENSE) -- code derived from core carries CyberBrick's own license terms in addition to this repo's.

## Data Collection

`collect_data.py` teleoperates the arm exactly like `virtual_gripper.py` (same keybindings)
while recording into a standard [LeRobotDataset](https://github.com/huggingface/lerobot).

```sh
python collect_data.py --fps 25 --task "Move the green marker from the orange tape to the blue tape."
```

`Return` starts an episode, `F` finishes it, `Backspace` discards it, `R` reviews what's been
recorded, `Q` quits. The dataset lands in `data/lerobot/<repo-id>/` and grows across sessions --
rerun with the same `--repo-id` to append.

Everything stays on disk. `--repo-id` is just a folder name here, not a Hugging Face account:
`lone_data/__init__.py` sets `HF_HUB_OFFLINE` before LeRobot is imported, so nothing in this
repo downloads or uploads anything.

### Dataset format

A plain LeRobotDataset, `CODEBASE_VERSION` v3.0, written by LeRobot's own API. Nothing custom
sits alongside it, and it is not specific to any one policy.

| feature | dtype | shape | |
|---|---|---|---|
| `observation.images.front` | `video` | (360, 640, 3) | Logitech webcam, downscaled from 1280x720 at its native 16:9 |
| `observation.state` | `float32` | (4,) | joint angles, always zeros -- see below |
| `action` | `float32` | (4,) | see below |

plus LeRobot's own `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` and the
task strings. Names follow `lerobot.utils.constants` because `dataset_to_policy_features()`
matches `action` exactly and the `observation` prefix, and drops everything else.

**`observation.state` is all zeros.** It holds what an encodered arm would report -- one angle
per DOF, in the same order as the action vector, so index `i` is the same joint in both:

| i | `observation.state` | `action` |
|---|---|---|
| 0 | `base_angle` | `base_motor_speed` |
| 1 | `upper_arm_angle` | `upper_arm_servo_speed` |
| 2 | `lower_arm_angle` | `lower_arm_servo_speed` |
| 3 | `gripper_angle` | `gripper_angle` |

The state is angles because proprioception measures where a joint *is*, while dimensions 0-2 of
the action say how fast to drive it. L-ONE has no encoders and no read-back path to the host, so
every entry stays zero -- the column exists for its shape. Every LeRobot policy expects the key:
omitting it makes ACT and π0.5 raise (they demand a state tensor they never feed to the network)
and Diffusion Policy and SmolVLA unbuildable. Zeros say "this robot reports no joint angles" in
the format policies already understand, which leaves every L-ONE policy vision-only in
substance whatever its architecture allows.

Do not read meaning into that column, and do not fill it with commands -- a command is not a
measurement. `scripts/test_policy_pipeline.py` asserts it stays zero. Datasets recorded before
the column existed are migrated with [`scripts/add_state_column.py`](scripts/add_state_column.py).

The action vector is the exact command sent to the board -- L-ONE reports nothing back, so
every dimension is a command, never a measurement:

| i | name | channel | call | unit | range |
|---|------|---------|------|------|-------|
| 0 | `base_motor_speed` | MOTOR1 | `MotorsController.set_speed` | raw PWM | -2048..2048 |
| 1 | `upper_arm_servo_speed` | PWM1 | `ServosController.set_speed` | percent | -100..100 |
| 2 | `lower_arm_servo_speed` | PWM2 | `ServosController.set_speed` | percent | -100..100 |
| 3 | `gripper_angle` | PWM3 | `ServosController.set_angle` | degrees | 0..180 |

For dimensions 0-2 a value of `0` was dispatched as `stop_motor()`/`stop_servo()`, **not**
`set_speed(idx, 0)` -- a motor's `stop()` sets both H-bridge channels to duty 100 while
`set_speed(idx, 0)` sets them to duty 0, which are different hardware states. Anything
replaying these actions must reproduce that dispatch.

`timestamp` is LeRobot's nominal `frame_index / fps` grid; the recorder paces itself to make
that grid true rather than back-filling wall-clock times. Real per-episode timing (achieved
fps, missed deadlines, command latency) goes to `logs/collection_*.jsonl`, outside the dataset.

### Inspecting

```sh
python scripts/inspect_dataset.py       # schema, action ranges, tasks, episode lengths
python scripts/validate_dataset.py      # self-consistency checks; non-zero exit on failure
python scripts/test_policy_pipeline.py  # what each policy makes of the dataset
lerobot-dataset-viz --repo-id lone/l_one --root data/lerobot/lone/l_one
```

### Normalization statistics

Run this after every collection session, before training:

```sh
python scripts/fix_action_stats.py
```

Action statistics estimated from recorded frames are unreliable here. Teleop emits three
discrete levels per channel, so a session where a joint moved in only one direction produces a
degenerate range -- after the first episode `action` dim 0 had `min/max = (-900, 0)` and
`q01/q99 = (-900, -0.0019)`, which maps the single most common action (`0`, stopped) onto the
extreme of the normalized range and puts any future `+900` at `+3.0`, outside the `[-1, 1]` a
policy's output head expects.

The limits are known exactly, so `ACTION_COMMAND_LIMITS` in
[`features.py`](lone_data/features.py) declares them and the script writes them in. All four
statistic pairs are written, since each normalization mode reads a different one --
`mean`/`std` (ACT), `min`/`max` (Diffusion Policy), `q01`/`q99` (π0.5), `q10`/`q90`.
Rerunning is necessary because `LeRobotDataset.save_episode()` regenerates `meta/stats.json`
from the recorded data on every saved episode.

### Training

Both policies go through [`scripts/train.py`](scripts/train.py), which takes exactly the
arguments `lerobot-train` takes and only applies this repo's defaults:
`--policy.push_to_hub=false`, `TOKENIZERS_PARALLELISM=false`, and re-enabling the Hub access
that `lone_data` switches off everywhere else.

ACT is the cheaper baseline, needs no Hub access, and takes no `empty_cameras`:

```sh
python scripts/train.py \
  --dataset.repo_id=lone/l_one \
  --dataset.root=data/lerobot/lone/l_one \
  --policy.type=act \
  --policy.device=cuda \
  --batch_size=8 --steps=30000
```

Checkpoints land in `outputs/train/` and stay there. `PreTrainedConfig.push_to_hub` defaults to
`true`, which makes `TrainPipelineConfig.validate()` refuse to start without a `policy.repo_id`
to push the trained model to, so `scripts/train.py` defaults it to `false`; pass
`--policy.push_to_hub=true` explicitly to override. This concerns the checkpoint only -- the
dataset is read straight off disk via `--dataset.root`, and its `repo_id` is just a folder name.

**Logging.** LeRobot has wandb built in (`wandb` is already a dependency); `wandb login` once,
then add:

```sh
  --wandb.enable=true --wandb.project=lone --wandb.disable_artifact=true
```

`--wandb.disable_artifact=true` matters. With wandb enabled, `log_policy()` uploads every
checkpoint as a wandb artifact, which for π0.5 means a multi-GB upload at each `--save_freq`.
`--wandb.mode=offline` logs to disk for a later `wandb sync` instead. Without wandb, metrics
still go to the console and to `outputs/train/`.

For π0.5, L-ONE has one camera and the policy takes three image slots, so two are declared
empty. Unlike ACT it needs Hub access, for the gated PaliGemma tokenizer and the base weights:

```sh
hf auth login   # google/paligemma-3b-pt-224 is gated; accept its licence first

python scripts/train.py \
  --dataset.repo_id=lone/l_one \
  --dataset.root=data/lerobot/lone/l_one \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.empty_cameras=2 \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=8 --num_workers=8 \
  --steps=30000 --save_freq=5000 \
  --wandb.enable=true --wandb.project=lone --wandb.disable_artifact=true
```

**Checkpoints.** `--save_freq=5000` writes a full checkpoint every 5000 steps to
`outputs/train/<date>/<time>_pi05/checkpoints/`, plus a `last/` symlink:

```
checkpoints/000005000/pretrained_model/model.safetensors   the policy
checkpoints/000005000/pretrained_model/config.json         + its processor pipelines
checkpoints/000005000/training_state/                      optimizer, scheduler, RNG, step
```

`outputs/` is gitignored. `--wandb.disable_artifact=true` only suppresses the *upload* of those
checkpoints to wandb -- they are still written locally, which is the copy that matters. Resume
from one with `--resume=true --config_path=<checkpoint>/pretrained_model/train_config.json`.

`--policy.pretrained_path` loads weights only and resets the checkpoint's config to defaults,
which is why `n_action_steps`, `dtype` and `empty_cameras` are spelled out; `--policy.path`
would load weights *and* the checkpoint's `config.json`. The two are not interchangeable.

A correct load prints `All keys loaded successfully!`. Anything else means the checkpoint and
the installed lerobot disagree about key names, and the weights are partly random.

**What is actually trained.** `--policy.train_expert_only=true` freezes all of PaliGemma and
trains the action expert. Measured on this configuration:

| module | params | trainable |
|---|---:|---:|
| VLM: Gemma-2B language model | 3037.5M | 0.0M |
| action expert: `gemma_300m` transformer | 427.9M | 427.9M |
| VLM: SigLIP vision tower | 412.4M | 0.0M |
| action expert: `lm_head` | 263.3M | 263.3M |
| action projections | 2.2M | 2.2M |
| **total** | **4.14B** | **693.4M (16.7%)** |

So this is the action-specific *part of the model*, not an output head: the head proper is
`action_out_proj`, a single `Linear(1024, 32)` -- 32,768 weights -- and no flag trains it alone.
Since the action space transfers nothing from pretraining (below), the expert is where a new one
gets learned, and in `pi05_base` it arrives pretrained rather than cold.

The `lm_head` row is an artifact of the expert being a `GemmaForCausalLM`: π0.5 decodes actions
by flow matching through `action_out_proj`, and the forward pass calls
`gemma_expert.model.forward`, which bypasses `lm_head` entirely. It is counted as trainable and
stored in checkpoints, but never enters the graph, so the effective trained set is closer to
430M.

Upstream targets a single 80 GB GPU at `--batch_size=64` for a full fine-tune. Freezing the VLM
is what makes this fit a 32 GB card -- the command above reports `mem_gb:12.85` at
`--batch_size=2`. Raise it until it stops fitting.

**On the action space:** π0.5 was pretrained on joint angles in radians with a gripper in
[0,1] at 20/50 Hz. L-ONE emits open-loop velocity/PWM commands at three discrete levels per
channel. There is no mapping between the two, so fine-tuning learns a new action space on top
of the vision/language backbone rather than transferring pretrained action semantics. Never
reuse a pretrained stats asset; use `scripts/fix_action_stats.py` above. Being RGB-only as well
means the visual prior does all the transfer work, so expect to need more episodes than a
proprioceptive setup would.

This is also why zero-shot action prediction is not worth attempting. `predict_action_chunk()`
slices the model's 32-dim output down to this dataset's 4 and the postprocessor unnormalizes it
into L-ONE's command ranges, so `pi05_base` will emit plausible-looking numbers -- but they
reinterpret four pretrained position dimensions as PWM and servo speeds, which is arbitrary.
lerobot's π0.5 port has no text-generation path either, so the hierarchical subtask prediction
that would be genuinely useful zero-shot is not available through it.

### Versions

`pyproject.toml` pins `lerobot[pi,dataset,training]==0.6.1`. The extras are not optional in
practice: `pi` pulls the `transformers` π0.5 needs, `dataset` the `torchcodec`/`av` stack for
reading video datasets, and `training` the `accelerate` that `lerobot-train` imports. Without
the last one training aborts at startup with `ImportError: 'accelerate' is required`.

`torchcodec==0.3.0` is pinned separately because it must match the installed torch. lerobot's
`dataset` extra allows `>=0.3.0,<0.12.0` and pip takes the newest, which is built against a
newer torch and fails to load with `Symbol not found: c10::MessageLogger`. 0.3.0 is the build
matching torch 2.7.

lerobot ≥0.5 requires **Python ≥3.12** -- 0.4.4 was the last release supporting 3.11. Staying
on 0.4.4 costs two extra workarounds that 0.6.1 does not need: openpi's patched `transformers`
fork (0.4.4's π0.5 asserts `transformers.models.siglip.check`, a marker module carried only on
a branch, and rejects any stock build with `ValueError: An incorrect transformer version is
used`), and a pinned `lerobot/pi05_base` revision (0.4.4's registry has no
`relative_actions_processor`, which the Hub checkpoint gained on 2026-06-03). 0.6.1 uses stock
`transformers` and loads `pi05_base` at `main` cleanly, so neither pin exists here.

## Firmware Setup (on-device)

1. Copy `LOneGripper/wifi_secrets.example.py` to `LOneGripper/wifi_secrets.py` and fill in your own `AP_SSID`/`AP_PASSWORD`/`STA_SSID`/`STA_PASSWORD`.
2. Upload the contents of `LOneGripper/` (for the receiver board) or `LOneRC/` (for the transmitter board) to the CyberBrick core board's filesystem, e.g. with [Pymakr](https://marketplace.visualstudio.com/items?itemName=pycom.Pymakr) (each folder has a `pymakr.conf`) or `mpremote`/Thonny.
3. Power on the transmitter first, then the receiver, so they pair over ESP-NOW. If the receiver doesn't see a transmitter within `NO_PAIRING_FALLBACK_TIMEOUT` seconds of boot, it falls back to WiFi mode automatically (see `LOneGripper/boot.py`) so `virtual_gripper.py` can still reach it.