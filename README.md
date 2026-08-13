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

Create a new conda environment with Python 3.11.
```bash
conda create -n cyberbrick python=3.11
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

Install FFmpeg. LeRobot stores camera streams as video, and `torchcodec` links against
system FFmpeg libraries rather than bundling them -- without this, reading a dataset back
fails with `Library not loaded: @rpath/libavutil.*.dylib`.

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
- [`scripts/`](scripts/) -- `inspect_dataset.py`, `validate_dataset.py`, `test_policy_pipeline.py`.

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
repo downloads or uploads anything. (LeRobot normally treats `repo_id` as a Hub coordinate and
tries to download a dataset whose local files look incomplete, which shows up as a 401
credentials prompt.) No Hugging Face account is needed to collect or inspect data -- only
`lerobot-train --policy.pretrained_path=lerobot/pi05_base` needs the network, once, to fetch
the base weights.

### Dataset format

A plain LeRobotDataset, `CODEBASE_VERSION` v3.0, written by LeRobot's own API. Nothing custom
sits alongside it, and it is not specific to any one policy.

| feature | dtype | shape | |
|---|---|---|---|
| `observation.images.front` | `video` | (360, 640, 3) | Logitech webcam, downscaled from 1280x720 at its native 16:9 |
| `action` | `float32` | (4,) | see below |

plus LeRobot's own `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` and the
task strings. Names follow `lerobot.utils.constants` because `dataset_to_policy_features()`
matches `action` exactly and the `observation` prefix, and drops everything else.

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

### Image format: native aspect, no letterboxing

Frames are stored at the camera's own aspect ratio and **never padded**. This matches every
published LeRobot dataset -- SO-101 `480x640`, ALOHA `480x640`, Berkeley UR5 `480x640`, DROID
`180x320` -- none of which pad to square. DROID is the direct precedent for a 16:9 webcam: it
downscales 1280x720 to 320x180 and lets the policy pad.

Padding here would be actively harmful, because π0.5 already pads. `resize_with_pad_torch()` in
`modeling_pi05.py` uses the identical ratio formula, so letterboxing at record time is applied a
second time at 224x224 and produces byte-for-byte the same model input while storing 6.2x fewer
real pixels:

```
store 256x256 letterboxed  ->  pi0.5 224x224, live content 224x126   (43.8% of the frame black)
store 640x360 native       ->  pi0.5 224x224, live content 224x126   (0% black)
```

640x360 leaves ~4x the linear resolution π0.5 consumes, so a later run can train at a higher
input size or re-crop the workspace. `--image-width`/`--image-height` change it; the aspect
ratio must match the camera's, and `collect_data.py` refuses at startup if it doesn't, since
frames are scaled rather than cropped.

### No proprioception

L-ONE has a camera and motors, and no encoders. The dataset therefore has **no
`observation.state`** -- inventing one would mean writing fabricated sensor readings into a
dataset other people may train on.

| policy | works | why |
|---|---|---|
| ACT | yes | every state use is guarded; the docstring marks it optional |
| Pi0 / Pi0.5 / Pi0-FAST / GR00T | needs a state supplied at train time | see below |
| Diffusion Policy | **no** | `modeling_diffusion.py` reads `robot_state_feature.shape[0]` unguarded |
| SmolVLA | **no** | `modeling_smolvla.py` reads `batch[OBS_STATE]` unconditionally |

`PI05Config.validate_features()` injects a padded `observation.state` *feature* and fills unused
camera slots via `empty_cameras` -- but a feature only carries shapes and normalization wiring,
not data. At runtime `Pi05PrepareStateTokenizerProcessorStep` raises
`ValueError("State is required for PI05")` on a batch without a state tensor, because π0.5
discretizes the state into the text prompt (`"Task: …, State: <256 bins>; Action: "`). No config
flag disables this in any current lerobot release (checked against 0.4.4 and `main`), so
`lerobot-train --policy.type=pi05` on this dataset fails at the first batch until a state is
supplied from outside the dataset.

`scripts/test_policy_pipeline.py` asserts all of the above -- including running a real batch
through π0.5's processor step, not just its config -- so this table is checked rather than
claimed. It needs no GPU, weights or network.

### Inspecting

```sh
python scripts/inspect_dataset.py       # schema, action ranges, tasks, episode lengths
python scripts/validate_dataset.py      # self-consistency checks; non-zero exit on failure
python scripts/test_policy_pipeline.py  # what each policy makes of the dataset
lerobot-dataset-viz --repo-id lone/l_one --root data/lerobot/lone/l_one
```

### Training π0.5

L-ONE has one camera and π0.5 takes three image slots, so two are declared empty.

```sh
hf auth login   # google/paligemma-3b-pt-224 is gated; accept its licence first

lerobot-train \
  --dataset.repo_id=lone/l_one \
  --dataset.root=data/lerobot/lone/l_one \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.empty_cameras=2 \
  --policy.device=cuda \
  --batch_size=32 --steps=30000
```

**This does not run yet.** As above, π0.5 requires an `observation.state` tensor that this
dataset deliberately does not contain, so the run stops at the first batch with
`ValueError: State is required for PI05`. Resolving that is a training-side decision (supply a
constant state, or the previous action, via a processor step or a schema addition) and is
tracked separately; the dataset itself needs no change either way.

`--policy.type=act` trains today with no such caveat and gives a cheaper vision-only baseline;
it needs no `empty_cameras`.

**On the action space:** π0.5 was pretrained on joint angles in radians with a gripper in
[0,1] at 20/50 Hz. L-ONE emits open-loop velocity/PWM commands at three discrete levels per
channel. There is no mapping between the two, so fine-tuning learns a new action space on top
of the vision/language backbone rather than transferring pretrained action semantics. Compute
normalization statistics from L-ONE data; do not reuse a pretrained stats asset. Being
RGB-only as well means the visual prior does all the transfer work, so expect to need more
episodes than a proprioceptive setup would.

### Versions

Pinned in `pyproject.toml`: `lerobot==0.4.4` (the newest release supporting Python 3.11 --
0.5+ requires ≥3.12 -- and already writing dataset `CODEBASE_VERSION` v3.0 with the `pi05`
policy) and `torchcodec==0.3.0` (the build matching torch 2.7; newer torchcodec is compiled
against a newer torch and fails to load with `Symbol not found: c10::MessageLogger`).

## Firmware Setup (on-device)

1. Copy `LOneGripper/wifi_secrets.example.py` to `LOneGripper/wifi_secrets.py` and fill in your own `AP_SSID`/`AP_PASSWORD`/`STA_SSID`/`STA_PASSWORD`.
2. Upload the contents of `LOneGripper/` (for the receiver board) or `LOneRC/` (for the transmitter board) to the CyberBrick core board's filesystem, e.g. with [Pymakr](https://marketplace.visualstudio.com/items?itemName=pycom.Pymakr) (each folder has a `pymakr.conf`) or `mpremote`/Thonny.
3. Power on the transmitter first, then the receiver, so they pair over ESP-NOW. If the receiver doesn't see a transmitter within `NO_PAIRING_FALLBACK_TIMEOUT` seconds of boot, it falls back to WiFi mode automatically (see `LOneGripper/boot.py`) so `virtual_gripper.py` can still reach it.