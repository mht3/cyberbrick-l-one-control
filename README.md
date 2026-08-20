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

Run the tests to make sure the environment is set up properly. If all tests pass, you're good to go!

```sh
pytest tests/
```

### Versions

`pyproject.toml` pins `lerobot[pi,dataset,training]==0.6.1`. The extras are not optional: `pi`
pulls the `transformers` π0.5 needs, `dataset` the `torchcodec`/`av` stack for video datasets, and
`training` the `accelerate` `lerobot-train` imports (without it, training aborts at startup with
`ImportError: 'accelerate' is required`).

`torchcodec==0.3.0` is pinned separately because it must match the installed torch -- the newest
build fails to load with `Symbol not found: c10::MessageLogger`. lerobot ≥0.5 requires Python ≥3.12.

## Project Layout

- [`collect_data.py`](collect_data.py) -- teleop GUI that records demonstrations into a LeRobotDataset.
- [`deploy_policy.py`](deploy_policy.py) -- GUI that runs a trained checkpoint on the arm, logging video and actions to `results/`.
- [`deploy_policy_training_data_replay.py`](deploy_policy_training_data_replay.py) -- the same GUI with the camera replaced by a recorded episode.
- [`virtual_gripper.py`](virtual_gripper.py) -- teleop GUI only, over USB (raw REPL) or WiFi (via `LOneGripper/wifi_bridge.py`).
- [`stream_camera.py`](stream_camera.py) -- serves a webcam over the network as JPEG frames, for `--remote-camera`.
- [`lone_data/`](lone_data/) -- dataset schema ([`features.py`](lone_data/features.py)), the LeRobot writer ([`lerobot_recorder.py`](lone_data/lerobot_recorder.py)), camera capture, the shared GUI base ([`robot_gui.py`](lone_data/robot_gui.py)), the command bus that keeps robot I/O off the record loop, and action dispatch ([`dispatch.py`](lone_data/dispatch.py)).
- [`scripts/`](scripts/) -- `train.py`, `eval_policy.py`, `fix_action_stats.py`, `add_state_column.py`, `inspect_dataset.py`, `validate_dataset.py`, `test_policy_pipeline.py`, `test_rtc.py`.
- [`tests/`](tests/) -- environment checks only; no hardware, dataset or checkpoint.
- [`LOneGripper/`](LOneGripper/) -- receiver firmware (the arm/gripper) for the CyberBrick core board.
- [`LOneRC/`](LOneRC/) -- transmitter firmware for the handheld remote.

`data/`, `outputs/`, `results/` and `logs/` are local working directories and are gitignored.

Both firmware folders share a `bbl/` module (buzzer, motors, servos, LEDs, sleep) and an `app/`
folder (`control`, `devices`, `parser`, `rc_main`) that build on
[CyberBrick's `CyberBrick_Controller_Core`](https://github.com/CyberBrick-Official/CyberBrick_Controller_Core):
`bbl/` is used as-is, while `boot.py`/`rc_main.py` extend core's versions with WiFi fallback. See
[`LICENSE`](LICENSE) -- code derived from core carries CyberBrick's own license terms too.

## Data Collection

`collect_data.py` teleoperates the arm exactly like `virtual_gripper.py` (same keybindings) while
recording into a standard [LeRobotDataset](https://github.com/huggingface/lerobot).

```sh
python collect_data.py --fps 25 --task "Move the green marker from the orange tape to the blue tape."
```

`Return` starts an episode, `F` finishes it, `Backspace` discards it, `R` reviews what's been
recorded, `Q` quits. The dataset lands in `data/lerobot/<repo-id>/` and grows across sessions --
rerun with the same `--repo-id` to append. `--repo-id` is a folder name here, not a Hugging Face
account: `lone_data/__init__.py` sets `HF_HUB_OFFLINE` before LeRobot is imported, so nothing in
this repo uploads or downloads.

### Dataset format

A plain LeRobotDataset, `CODEBASE_VERSION` v3.0, written by LeRobot's own API, plus LeRobot's own
`timestamp`, `frame_index`, `episode_index`, `index`, `task_index` and task strings.

| feature | dtype | shape | |
|---|---|---|---|
| `observation.images.front` | `video` | (360, 640, 3) | AV1, downscaled from the webcam's 1280x720 |
| `observation.state` | `float32` | (4,) | joint angles -- always zeros |
| `action` | `float32` | (4,) | the command sent to the board |

Actions are commands, never measurements -- L-ONE reports nothing back:

| i | name | channel | call | teleop emits | hardware range |
|---|------|---------|------|------|-------|
| 0 | `base_motor_speed` | MOTOR1 | `MotorsController.set_speed` | -900 / 0 / 900 | -2048..2048 raw PWM |
| 1 | `upper_arm_servo_speed` | PWM1 | `ServosController.set_speed` | -100 / 0 / 100 | -100..100 percent |
| 2 | `lower_arm_servo_speed` | PWM2 | `ServosController.set_speed` | -100 / 0 / 100 | -100..100 percent |
| 3 | `gripper_angle` | PWM3 | `ServosController.set_angle` | 30 / 120 | 0..180 degrees |

- **`0` on dims 0-2 was dispatched as `stop_motor()`/`stop_servo()`, not `set_speed(idx, 0)`** --
  `stop()` sets both H-bridge channels to duty 100 while `set_speed(idx, 0)` sets them to duty 0.
  Those are different hardware states, and any replay has to reproduce the dispatch.
- **`observation.state` is all zeros.** L-ONE has no encoders and no read-back path, but every
  LeRobot policy expects the key -- ACT and π0.5 raise without it, Diffusion Policy and SmolVLA
  will not build. Every L-ONE policy is vision-only in substance. Don't fill it with commands;
  `scripts/add_state_column.py` migrates datasets recorded before the column existed.
- The teleop envelope, not the hardware range, is `ACTION_COMMAND_LIMITS` in
  [`features.py`](lone_data/features.py) -- what everything downstream normalizes, clamps and
  snaps against.
- Feature names come from `lerobot.utils.constants`: `dataset_to_policy_features()` keeps `action`
  and the `observation` prefix and silently drops every other key.
- `timestamp` is the nominal `frame_index / fps` grid. Real timing (achieved fps, missed deadlines,
  command latency) goes to `logs/collection_*.jsonl`, outside the dataset.

### Normalization statistics

Run after every collection session, before training:

```sh
python scripts/fix_action_stats.py --root data/lerobot/lone/l_one_green_marker
```

Teleop is bang-bang -- three levels per channel -- so statistics estimated from recorded frames are
degenerate: after the first episode, `action` dim 0 had `min/max = (-900, 0)`, which maps the most
common action (stopped) onto an extreme of the normalized range. The limits are known exactly, so
`ACTION_COMMAND_LIMITS` declares them and this script writes all four statistic pairs into
`meta/stats.json`. Rerun it because `LeRobotDataset.save_episode()` regenerates that file on every
saved episode.

### Inspecting

```sh
ID=lone/l_one_green_marker; ROOT=data/lerobot/$ID   # scripts otherwise default to lone/l_one_marker_pickup

python scripts/inspect_dataset.py --repo-id $ID --root $ROOT       # schema, action ranges, tasks, episode lengths
python scripts/validate_dataset.py --repo-id $ID --root $ROOT      # self-consistency; non-zero exit on failure
python scripts/test_policy_pipeline.py --repo-id $ID --root $ROOT  # what each policy makes of the dataset
lerobot-dataset-viz --repo-id $ID --root $ROOT
```

## VLA Finetuning

Both policies go through [`scripts/train.py`](scripts/train.py), which takes exactly the arguments
`lerobot-train` takes and only adds this repo's defaults: `--policy.push_to_hub=false`,
`TOKENIZERS_PARALLELISM=false`, and re-enabling the Hub access `lone_data` switches off elsewhere.
Checkpoints land in `outputs/train/` and stay there; the dataset is read off disk via
`--dataset.root`.

**ACT** is the cheaper baseline and needs no Hub access:

```sh
python scripts/train.py \
  --dataset.repo_id=lone/l_one_green_marker \
  --dataset.root=data/lerobot/lone/l_one_green_marker \
  --policy.type=act \
  --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
  --policy.device=cuda \
  --batch_size=8 --steps=30000
```

**π0.5** needs Hub access for the gated PaliGemma tokenizer and the base weights:

```sh
hf auth login   # google/paligemma-3b-pt-224 is gated; accept its licence first

python scripts/train.py \
  --dataset.repo_id=lone/l_one_green_marker \
  --dataset.root=data/lerobot/lone/l_one_green_marker \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.normalization_mapping='{"VISUAL":"IDENTITY","STATE":"QUANTILES","ACTION":"MIN_MAX"}' \
  --policy.empty_cameras=0 \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=false \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=8 --num_workers=8 \
  --steps=150000 --policy.scheduler_decay_steps=150000 \
  --save_freq=5000 \
  --wandb.enable=true --wandb.project=lone --wandb.disable_artifact=true
```

The flags that matter:

- **`--policy.normalization_mapping`** must be passed as the whole dict; draccus rejects the dotted
  `--policy.normalization_mapping.ACTION=MIN_MAX` form. `VISUAL` and `STATE` are π0.5's own
  defaults, so `ACTION` is the only entry that changes.
- **`--policy.scheduler_decay_steps` must track `--steps`.** It defaults to 30000, so a longer run
  silently spends its tail at the floor LR.
- **`--policy.empty_cameras=0`** (already the default). `empty_cameras=N` *adds* N synthetic
  cameras rather than declaring unused slots: with one real camera, `=2` fabricates two all-`-1`
  images, runs SigLIP on each and appends 512 tokens that are then masked out of every attention
  involving a real token. Dropping them is 3.8x faster and moves the network output by 5.8e-4,
  under the 7.8e-4 run-to-run noise floor. (π0.5 declares three slots because openpi is JAX and
  needs a statically shaped observation pytree; PyTorch has no such constraint.)
- **`--policy.gradient_checkpointing=false`.** It is a memory trade, not a speed one -- upstream
  enables it for `--batch_size=64` on an 80 GB GPU; at batch 8 on 32 GB it costs ~1.6x for nothing.
- **`--policy.pretrained_path` loads weights only** and resets config to defaults, which is why
  `n_action_steps` and `dtype` are spelled out. `--policy.path` loads weights *and* the
  checkpoint's `config.json`. A correct load prints `All keys loaded successfully!`; anything else
  means the weights are partly random.
- **`--wandb.disable_artifact=true`.** Without it, `log_policy()` uploads every checkpoint as a
  wandb artifact -- a multi-GB push at each `--save_freq`. `--wandb.mode=offline` logs to disk for
  a later `wandb sync`.
- **`--policy.tokenizer_max_length` is a no-op** when `--policy.pretrained_path` is set: the
  tokenizer step is deserialized from the pretrained repo's `policy_preprocessor.json`
  (`max_length: 200`) and the flag is never read.

Throughput, RTX 5090 (32 GB), one camera, `l_one_green_marker`:

| config | s/step | samples/s | peak VRAM |
|---|---:|---:|---:|
| `empty_cameras=2`, `gradient_checkpointing=true`, bs 8 | 0.913 | 8.8 | 16.7 GB |
| `empty_cameras=0`, `gradient_checkpointing=false`, bs 8 | **0.238** | **33.6** | 24.2 GB |

Batch size buys no throughput: 8 → 16 → 32 holds at ~7 samples/s, and above bs 12
(`empty_cameras=0`) or bs 5 (`empty_cameras=2`) it OOMs. Raise it only for less noisy steps.

`--save_freq=5000` writes a full checkpoint to `outputs/train/<date>/<time>_pi05/checkpoints/`
plus a `last/` symlink:

```
checkpoints/000005000/pretrained_model/model.safetensors   the policy
checkpoints/000005000/pretrained_model/config.json         + its processor pipelines
checkpoints/000005000/training_state/                      optimizer, scheduler, RNG, step
```

Resume with `--resume=true --config_path=<checkpoint>/pretrained_model/train_config.json`.

**What `--policy.train_expert_only=true` trains** -- 693.4M of 4.14B params (16.7%):

| module | params | trainable |
|---|---:|---:|
| VLM: Gemma-2B language model | 3037.5M | 0.0M |
| action expert: `gemma_300m` transformer | 427.9M | 427.9M |
| VLM: SigLIP vision tower | 412.4M | 0.0M |
| action expert: `lm_head` | 263.3M | 263.3M |
| action projections | 2.2M | 2.2M |

`lm_head` is an artifact of the expert being a `GemmaForCausalLM`: π0.5 decodes through
`action_out_proj`, and the forward pass bypasses `lm_head` entirely, so the effective trained set
is ~430M. Freezing the VLM is what fits this on a 32 GB card -- upstream full fine-tunes on an
80 GB GPU at `--batch_size=64`.

**The action space transfers nothing from pretraining.** π0.5 was pretrained on joint angles in
radians with a gripper in [0,1] at 20/50 Hz; L-ONE emits open-loop PWM/velocity commands at three
levels. Fine-tuning learns a new action space on top of the vision/language backbone, so never
reuse a pretrained stats asset, and expect to need more episodes than a proprioceptive setup
would. Zero-shot is not worth attempting: `pi05_base` emits plausible numbers that reinterpret
four pretrained position dimensions as PWM and servo speeds.

### Evaluating a checkpoint

```sh
python scripts/eval_policy.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_green_marker
```

Scores a checkpoint against the recorded demonstrations without touching hardware, loading it the
way deployment does so the normalization statistics frozen in at training time are the ones used.
It is teacher-forced and open-loop -- every prediction comes from a recorded frame and the policy's
own actions are never fed back -- so it measures fit to the demonstrations, not task success, and
cannot see compounding error.

Read the output in this order:

1. **MSE against the predict-the-mean baseline.** At or above baseline means nothing was learned
   from the image. Where a dimension is constant across sampled frames the baseline is 0 and
   unbeatable, and the script says so.
2. **Prediction spread per dimension**, beside the ground truth's. Near zero is mode collapse.
3. **Same-input spread** (`--repeats`, default 8). π0.5 draws fresh noise every call, so repeated
   predictions on one frame genuinely differ -- roughly `[240, 27, 15, 13]` per dimension, where
   ACT reports exactly `0.0000`.

Per-dimension numbers are raw command units; the average is normalized by `ACTION_COMMAND_LIMITS`,
since dim 0 spans ±900 and dim 3 spans 90. `--json <path>` writes the same figures machine-readably.

`--root` defaults to `data/lerobot/<repo-id>` and is worth watching: `LeRobotDataset` loads from
`root` and treats `repo_id` as a label, so a stale `--root` scores the checkpoint against episodes
it never saw while the output looks completely normal.

## Deployment

```sh
python deploy_policy.py --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model
```

A GUI in the same shape as `collect_data.py`: connect over Serial or WiFi, confirm the task
prompt, and run. **STOP ALL** and `<space>` stop the arm at any time; a stop also fires on Stop, on
window close, on link death, and from a `finally` around the main loop, each preceded by
`cancel_pending()` so a queued speed cannot land after it.

- **The task prompt is prefilled from the checkpoint's `train_config.json`.** π0.5 conditions on
  that text, so a differently worded prompt is different conditioning. A multi-task dataset
  prefills nothing -- type the one you want, verbatim.
- **Commands are sent on change, not every tick.** The board holds a command until superseded, and
  a round trip costs 73/378/1116 ms min/avg/max over WiFi. Dispatching all four dims every tick is
  100 commands/s at 25 Hz, which the link cannot drain: droppable speeds go stale and are
  discarded while the non-droppable stops still land, so the arm gets the stops and almost none of
  the motion while the log shows a perfect trajectory. Held speeds are re-sent by
  `RobotAppBase._wifi_heartbeat_tick`, which also feeds `wifi_bridge.py`'s 2.0 s
  `COMMAND_DEADMAN_TIMEOUT`.
- **Action mode.** *Snap to demonstrated levels* (default) quantizes the policy's continuous output
  back onto the three levels teleop used, keeping the arm in distribution; *raw* clamps to
  `ACTION_COMMAND_LIMITS`. Both the raw prediction and the dispatched action are logged.
- **The checkpoint loads once, at startup**, on a worker thread -- π0.5 takes ~70 s to reach the
  GPU. Start Policy stays disabled until the log says `Checkpoint ready`, and the weights outlive
  every rollout.
- **Inference runs on a worker thread** (~280 ms on an RTX 5090, most of a control period) and
  keeps the action queue filled; the control tick only pops and dispatches. One inference covers
  `n_action_steps / fps` = 0.40 s at 25 Hz. On an underrun the last action is held for 3 ticks and
  then the arm stops. The first tick is exempt: the status reads *Policy priming* until the first
  chunk lands, so any underrun message after that is real.
- **Action normalization is checked at load** -- the mode from `policy_preprocessor.json` and the
  statistic pair it decodes against, versus MIN_MAX/`ACTION_COMMAND_LIMITS`. A mismatch has no
  other symptom (the mapping stays invertible, the arm just drives a little too far), so it is
  logged with the offending dimensions; the check never blocks a load.
- **Plots are on a wall-clock axis** -- last 8 s, drawn as steps, carried to the present edge --
  because actions do not arrive on a regular grid and a command holds until superseded.

**Real-time chunking** (`--rtc`, plus a checkbox that stays live mid-run) replaces the action queue
instead of appending to it. Default behaviour queues a fresh chunk behind the old plan, so the arm
works through actions drawn from a frame up to ~19 ticks ago and the join between chunks is a jerk.
RTC is [Physical Intelligence's method](https://www.physicalintelligence.company/research/real_time_chunking):
a guidance term on the flow-matching denoiser holds the new chunk's head against the unexecuted
tail of the running plan, decaying across the overlap, and the first `ceil(latency x fps)` actions
are dropped as already spent. It needs a flow-matching policy (π0, π0.5, SmolVLA); the checkbox
greys out otherwise. Knobs: `--rtc-execution-horizon` (20; keep it well above the ~4-tick inference
delay or the guidance becomes a hard clamp), `--rtc-max-guidance-weight` (10.0 suits 10 denoising
steps), `--rtc-schedule`, `--rtc-queue-threshold` (30, the open-loop horizon). Snapping hides most
of the effect -- expect fewer commands and fewer one-tick reversals in `raw`/`rtc_delay` rather than
visibly smoother motion. `python scripts/test_rtc.py --checkpoint <dir>` checks the whole path.

Every run writes `results/deploy_<timestamp>/`:

```
video.mp4      frames exactly as they were given to the policy
actions.jsonl  per tick: raw prediction, dispatched action, inference latency, underrun flag,
               `rtc`/`rtc_delay`/`rtc_prefix`, and `link` -- the CommandBus's backlog, dropped
               count and mean latency, so the log says whether a command reached the board
run.json       checkpoint, task, fps, action mode, RTC settings, device, git SHA
```

### Replaying training data through the policy

```sh
python deploy_policy_training_data_replay.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_green_marker --root data/lerobot/lone/l_one_green_marker --remote-camera
```

`deploy_policy.py` with the camera swapped for a recorded episode. It subclasses `DeployApp`, so
the checkpoint loading, action queue, snapping, `dispatch_action()` and results logging are the
ones that run on the arm for real. Fed back the exact frames it trained on, does the arm reproduce
that trajectory and pick the thing up? `scripts/eval_policy.py` answers the offline half of that;
this answers the half with a motor in it.

Run the three configurations in order -- each isolates a different failure:

1. **`--source dataset`** dispatches the *recorded* actions with no policy in the loop. If the arm
   cannot pick the marker up from the demonstration's own actions, the fault is start pose, timing
   or hardware; stop here. Needs no checkpoint.
2. **`--source policy --lookahead 0`** -- what deployment actually does.
3. **`--source policy --lookahead 10`** -- the same run with the inference lag dialled out.

- **The arm must start where the demonstration started.** Scrub to frame 0 and teleoperate until
  the live camera pane and the demo frame agree before pressing Start. `--no-camera` drops the
  pane; `--loop`, `--start-frame` and `--episode` do what they say.
- **Lag is measured, not assumed.** Every action carries the index of the frame it was predicted
  from; expect about `n_action_steps`, or 400 ms at 25 Hz. `--lookahead K` submits frame `cursor+K`
  while the panes and plots still follow the cursor.
- **Ground truth is on screen**: a dashed demonstration trace beside the commanded one, plus a
  running match rate. That and the mean/max lag are appended to `actions.jsonl` as a `run_summary`
  line when the run ends.
- **Feed perturbation** (`--brightness`, `--contrast`, `--noise`, `--jpeg-quality`, and live
  sliders) degrades the dataset frame before the deploy path, with the demonstration's own actions
  underneath as the control. This is how you find out why a live camera does worse: training and
  replay see AV1-decoded pixels, a live camera does not, `--remote-camera` adds JPEG q80, and the
  room's lighting has moved. Against the single-episode π0.5 checkpoint a **0.3% mean pixel change
  flipped the commanded action** -- 42x further than the policy's own sampling noise. Turn the dial
  up until the arm stops doing the task; how far you got is the number worth having.
- **The observation path is checked at startup** against the tensor LeRobot itself returns -- `0.0`
  when the stored frame size equals `--image-width/height`, which rules out silent preprocessing
  mismatch.
- **`--repo-id` is required and `--root` follows it**, for the reason under *Evaluating a
  checkpoint*: two independent defaults let you name one dataset and replay another in silence.

## Firmware Setup (on-device)

1. Copy `LOneGripper/wifi_secrets.example.py` to `LOneGripper/wifi_secrets.py` and fill in your own `AP_SSID`/`AP_PASSWORD`/`STA_SSID`/`STA_PASSWORD`.
2. Upload the contents of `LOneGripper/` (receiver board) or `LOneRC/` (transmitter board) to the CyberBrick core board's filesystem, e.g. with [Pymakr](https://marketplace.visualstudio.com/items?itemName=pycom.Pymakr) (each folder has a `pymakr.conf`) or `mpremote`/Thonny.
3. Power on the transmitter first, then the receiver, so they pair over ESP-NOW. If the receiver sees no transmitter within `NO_PAIRING_FALLBACK_TIMEOUT` seconds of boot (15, in `LOneGripper/app/rc_main.py`), it falls back to WiFi mode automatically so `virtual_gripper.py` can still reach it.
