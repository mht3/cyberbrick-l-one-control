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
- [`lone_data/`](lone_data/) -- dataset schema ([`features.py`](lone_data/features.py)), the LeRobot writer ([`lerobot_recorder.py`](lone_data/lerobot_recorder.py)), camera capture, the command bus that keeps robot I/O off the record loop, and action dispatch ([`dispatch.py`](lone_data/dispatch.py)).
- [`deploy_policy.py`](deploy_policy.py) -- GUI that runs a trained checkpoint on the arm and logs video and actions to `results/`.

- [`deploy_policy_training_data_replay.py`](deploy_policy_training_data_replay.py) -- the same GUI with the camera replaced by a recorded episode, for checking whether a checkpoint reproduces its own demonstration on the real arm.
- [`scripts/`](scripts/) -- `train.py`, `eval_policy.py`, `fix_action_stats.py`, `add_state_column.py`, `inspect_dataset.py`, `validate_dataset.py`, `test_policy_pipeline.py`.

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
lerobot-dataset-viz --repo-id lone/l_one_green_marker --root data/lerobot/lone/l_one_green_marker
```

### Normalization statistics

Run this after every collection session, before training:

```sh
python scripts/fix_action_stats.py --root data/lerobot/lone/l_one_green_marker
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
`mean`/`std` (ACT), `min`/`max` (Diffusion Policy and this repo's π0.5 default), `q01`/`q99`,
`q10`/`q90`. Rerunning is necessary because `LeRobotDataset.save_episode()` regenerates
`meta/stats.json` from the recorded data on every saved episode.

**Actions are normalized MIN_MAX, not by quantile**, which is why every training command below
passes `--policy.normalization_mapping`. π0.5 defaults to QUANTILES, and quantiles are the wrong
statistic for three-level commands twice over:

- A level appearing in under 1% of frames sits inside the tail the quantile discards, so that
  whole direction of the joint's range normalizes away. Levels this rare are normal here -- one
  joint may be nudged once an episode and driven hard in the other direction throughout.
- `aggregate_feature_stats` combines per-episode quantiles with a count-weighted mean, which is
  not a valid operation on a quantile. The result is neither the pooled quantile nor a value any
  action takes. `min`/`max` in the same function aggregate exactly -- `np.min` of mins, `np.max`
  of maxes -- so they are immune.

Commands are hard-clamped to known limits, so there are no outliers for a quantile to reject, and
once the statistics are written from the envelope both modes compute identical numbers. MIN_MAX is
simply the one that survives aggregation. It still collapses if a whole session never drives a
joint in one direction, so `fix_action_stats.py` remains the thing that makes it correct by
construction.

### Training

Both policies go through [`scripts/train.py`](scripts/train.py), which takes exactly the
arguments `lerobot-train` takes and only applies this repo's defaults:
`--policy.push_to_hub=false`, `TOKENIZERS_PARALLELISM=false`, and re-enabling the Hub access
that `lone_data` switches off everywhere else.

ACT is the cheaper baseline, needs no Hub access, and takes no `empty_cameras`:

```sh
python scripts/train.py \
  --dataset.repo_id=lone/l_one_green_marker \
  --dataset.root=data/lerobot/lone/l_one_green_marker \
  --policy.type=act \
  --policy.normalization_mapping='{"VISUAL":"MEAN_STD","STATE":"MEAN_STD","ACTION":"MIN_MAX"}' \
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

π0.5 declares three image slots because openpi, where it was pretrained, is JAX: the
observation pytree must be statically shaped across jitted calls, so every robot emits all
three keys and toggles `image_masks` to say which are real. L-ONE has one camera, and in
PyTorch that constraint does not exist -- so the extra slots are left off
(`--policy.empty_cameras=0`). See *Why `empty_cameras=0`* below. Unlike ACT it needs Hub
access, for the gated PaliGemma tokenizer and the base weights:

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

`--policy.normalization_mapping` must be passed as the whole dict; draccus rejects the dotted
`--policy.normalization_mapping.ACTION=MIN_MAX` form. `VISUAL` and `STATE` keep π0.5's own
defaults -- images are already `[0, 1]` and `observation.state` is all zeros -- so `ACTION` is the
only entry that changes.

**Throughput.** Measured on an RTX 5090 (32 GB), one camera, `l_one_green_marker`:

| config | s/step | samples/s | peak VRAM |
|---|---:|---:|---:|
| `empty_cameras=2`, `gradient_checkpointing=true`, bs 8 | 0.913 | 8.8 | 16.7 GB |
| `empty_cameras=0`, `gradient_checkpointing=false`, bs 8 | **0.238** | **33.6** | 24.2 GB |

3.8x, from dropping work the model discards anyway. Two things to know:

* **Gradient checkpointing is a memory trade, not a speed one.** It frees activations and
  recomputes them in the backward pass -- roughly one extra forward. The π0.5 doc turns it on
  because those commands are `--batch_size=64` and "sized for a single 80 GB GPU". At batch 8
  on 32 GB it costs ~1.6x for nothing. It is numerically exact (there is no dropout anywhere in
  the π0.5 forward, so `preserve_rng_state=False` is safe).
* **Batch size buys no throughput here.** 8 -> 16 -> 32 holds at ~7 samples/s; the GPU is
  already saturated at batch 8. Raise it only if you want fewer, less noisy optimizer steps.
  Above bs 12 (`empty_cameras=0`) or bs 5 (`empty_cameras=2`) it OOMs.

`--policy.tokenizer_max_length` looks like a third lever and is not: whenever
`--policy.pretrained_path` is set, `make_pre_post_processors()` deserializes the tokenizer step
from the pretrained repo's `policy_preprocessor.json` (`max_length: 200`), and the
`preprocessor_overrides` dict in `lerobot_train.py` has no `tokenizer_processor` entry. The flag
lands in `train_config.json` and is never read. Setting it changes neither step time nor memory.

If you change `--steps`, change `--policy.scheduler_decay_steps` with it. It defaults to 30000,
so a longer run silently spends its tail at the floor LR.

**Why `empty_cameras=0`.** `empty_cameras=N` *adds* N synthetic cameras -- it does not declare
how many of three slots are unused. With one real camera, `=2` fabricates two all-`-1` images,
runs SigLIP on each, and appends 512 tokens to the prefix (968 vs 456). All 512 are then masked
out of every attention computation involving a real token. Verified as mask arithmetic, with no
model involved:

| | 3 slots vs 1 slot |
|---|---|
| real-to-real attention submatrix identical | true |
| position ids of real tokens identical | true |
| any real query attends a masked key | false |

`pad_2d_masks = pad_masks[:,None,:] * pad_masks[:,:,None]` drops masked positions as both
queries and keys, and `position_ids = cumsum(pad_masks) - 1` does not advance across them, so
real tokens land on identical RoPE positions either way. On the pretrained `pi05_base` weights
in fp32, the resulting change in network output is 5.8e-4 against a 7.8e-4 run-to-run noise
floor -- i.e. below the noise of running the same config twice.

This does not contradict pretraining. SigLIP is never handed three images at once; it is called
in a Python loop, once per image, with no cross-image interaction. What has a three-slot
structure is the token sequence fed to Gemma, and robots with fewer cameras were pretrained with
the spare slots masked -- which is the same computation as omitting them. Corroboration: openpi
pads with `np.zeros_like(base_image)` while lerobot pads with `-1`. Two different fill values,
both correct, which is only possible because the pixels are never read.

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
which is why `n_action_steps` and `dtype` are spelled out (`empty_cameras=0` is already the
default and is passed only to make the choice explicit); `--policy.path`
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

### Evaluating a checkpoint

```sh
python scripts/eval_policy.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_green_marker
```

Scores a checkpoint against the recorded demonstrations without touching hardware. It loads the
checkpoint the same way deployment does, so the normalization statistics frozen in at training
time are the ones used -- never recomputed from the dataset.

**`--root` defaults to `data/lerobot/<repo-id>`, and is worth knowing about**, because
`LeRobotDataset` loads from `root` and treats `repo_id` as a label once the data is local. A
`--root` left pointing at some other dataset is read without complaint, and the checkpoint gets
scored against episodes it has never seen while the output looks completely normal -- a fit that
beats baseline sixfold reads as *worse than predicting the mean*. The script now derives `root`
from `--repo-id` unless you override it, and says so when the dataset being scored is not the
one `train_config.json` says the checkpoint trained on.

**This is teacher-forced open-loop evaluation.** Every prediction is made from a *recorded*
frame and the policy's own actions are never fed back, so it measures fit to the
demonstrations, not task success. It cannot see compounding error.

Read the output in this order:

1. **MSE against the predict-the-mean baseline.** A policy at or above baseline has learned
   nothing from the image. This is the single most useful number. Where a dimension is constant
   across the sampled frames the baseline is 0 and unbeatable, so the script says so rather than
   printing a meaningless verdict.
2. **Prediction spread per dimension**, beside the ground truth's. A std near zero is mode
   collapse -- the policy emitting one action regardless of input.
3. **Same-input spread** (`--repeats`, default 8). π0.5 is a flow-matching model that draws
   fresh noise every call, so repeated predictions on an identical frame genuinely differ --
   measured at roughly `[240, 27, 15, 13]` per dimension. That variance *is* the policy, not
   measurement error, and a large MSE next to a large spread means something different from a
   large MSE alone. ACT reports exactly `0.0000` here, which is the honest contrast.

Per-dimension numbers are in raw command units; the average is normalized by
`ACTION_COMMAND_LIMITS`, because dim 0 spans ±900 while dim 3 spans 90 and a raw average would
be almost entirely dim 0. `--json <path>` writes the same figures machine-readably.

### Deployment

```sh
python deploy_policy.py --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model
```

A GUI in the same shape as `collect_data.py`: connect over Serial or WiFi, confirm the task
prompt, and run. The prompt is prefilled by following the checkpoint's `train_config.json` to
its dataset and reading the task back out of the metadata -- π0.5 conditions on that text, so a
differently worded prompt is a different conditioning. A multi-task dataset prefills nothing,
since no single one of its prompts is the right default; type the one you want, verbatim.

**The checkpoint loads once, at startup.** π0.5 takes ~70 s to reach the GPU, so loading it on
the Start Policy click would freeze the window at the moment the arm was about to move, and pay
that cost again on every run. It loads on a worker thread while the GUI comes up instead --
Start Policy stays disabled until the log says `Checkpoint ready` -- and the loaded weights then
outlive every rollout. Only the task prompt and the action queue are per-run, so retyping the
prompt and starting again is immediate. Stopping clears the queue and the last frame, so a run
never begins on the previous one's actions.

**Commands are sent on change, not every tick.** The board holds a command until it is
superseded, so re-sending an identical one buys nothing and costs a full round trip -- measured
at 73/378/1116 ms min/avg/max on the routed campus path. Dispatching all four dimensions every
tick is 100 commands/s at 25 Hz, which no WiFi path here can drain: the queue then sits
permanently past `_MAX_COMMAND_AGE`, every droppable *speed* is discarded for being stale, and
the non-droppable stops still go through. The arm receives the stops and almost none of the
motion, while the log shows a perfect trajectory, because the log records what was submitted.
On the one recorded episode in `data/lerobot/lone/l_one` this is the difference between 1624
commands and 22 -- and 22 is also what teleop sent while recording it, since a key press emits
one command and a hold emits none. `RobotAppBase._wifi_heartbeat_tick` re-sends whatever speed
is still held, which is what feeds `wifi_bridge.py`'s single global 2.0 s `COMMAND_DEADMAN_TIMEOUT`
and what makes speeds safe to mark droppable in the first place.

**Action mode.** The policy emits continuous values, but the demonstrations only ever contained
three discrete levels per channel. *Snap to demonstrated levels* (default) quantizes back onto
them, keeping the arm inside the distribution it was trained on; *raw* clamps to
`ACTION_COMMAND_LIMITS` instead. Both the raw prediction and the dispatched action are logged
either way, so the choice stays visible after the fact.

**Real-time chunking** (`--rtc`, and a checkbox that stays live mid-run) changes how one chunk
gives way to the next. By default a fresh chunk is appended behind whatever is still queued, so
the arm works through an old plan -- drawn from a frame up to ~19 ticks ago -- before it sees the
new one, and where the two meet nothing smoothed the join. That seam is the jerk. RTC is
[Physical Intelligence's method](https://www.physicalintelligence.company/research/real_time_chunking):
it treats the new chunk as an inpainting problem, adding a guidance term to the flow-matching
denoiser that holds the chunk's head against the unexecuted tail of the plan already running,
under a weight that decays across the overlap. The queue is then *replaced* rather than appended,
with the first `ceil(latency x fps)` actions dropped as already spent -- so the arm also reaches
the newest plan far sooner. None of it is implemented here; it is `lerobot.policies.rtc` wired to
`PolicyRunner`, and it needs a flow-matching policy (π0, π0.5, SmolVLA) -- the checkbox greys out
for anything else.

The knobs are `--rtc-execution-horizon` (default 20; keep it well above the inference delay in
ticks, ~4 here, or the guidance degenerates into a hard clamp with no ramp to blend over),
`--rtc-max-guidance-weight` (10.0 suits 10 denoising steps), `--rtc-schedule`, and
`--rtc-queue-threshold`, which is the open-loop horizon: the worker replans once the queue falls
to that many actions.

Expect to *measure* the improvement rather than see it. Snapping quantizes each channel to three
levels, and RTC's smoothing is mostly sub-quantum -- what survives the quantizer is the timing of
level changes, so the effect shows up as fewer commands and fewer one-tick reversals, not as
visibly smoother motion. `raw` in `actions.jsonl` is the pre-snap value and is where the smoothing
is actually legible; `rtc_delay` is rewritten exactly when a chunk was replaced, which is how the
seams are found. Run in *raw* mode if you want to watch it on the plot panel, which draws the
dispatched action. `python scripts/test_rtc.py --checkpoint <dir>` checks the whole path -- weight
schedules, that the guidance measurably pulls the chunk head toward the prefix while leaving the
tail free, and that the queue bookkeeping survives a live toggle in both directions.

**The checkpoint's action normalization is checked at load.** Deployment expects actions
normalized MIN_MAX against `ACTION_COMMAND_LIMITS` (see *Normalization statistics* above for why).
Both halves are verified when the checkpoint loads: the mode is read out of
`policy_preprocessor.json`, and the statistic pair that mode actually decodes against is compared
to the declared limits. A mismatch is logged with the offending dimensions and where *stopped*
decodes to. Neither failure has any other symptom -- the mapping stays invertible, so the arm just
drives a little too far on the affected joints. The check is advisory and never blocks a load; the
fix is to correct the dataset statistics, retrain, or both.

**Why inference runs on a worker thread.** π0.5 takes ~280 ms per inference on an RTX 5090,
which is most of a control period. It runs off the Tk thread and keeps a queue of upcoming
actions filled; the control tick only pops and dispatches. One inference covers
`n_action_steps / fps` seconds of motion — 0.40 s at 25 Hz with the default 10 — and the live
readout shows measured latency against that budget so the margin is visible rather than
inferred. If the queue empties, the last action is held for up to 3 ticks to ride out jitter and
then the arm is stopped; it never keeps driving on stale commands.

A run's first tick is exempt. The frame that triggers the first inference is submitted on that
tick, so the queue is necessarily empty when it is first read — every run used to open by
reporting an underrun and stopping an already-stopped arm, which trained you to ignore the one
message that matters. Until the first chunk lands the status reads *Policy priming*, nothing is
dispatched, and the arm stays where teleop left it; if that takes more than 2 s the log says so.
After that, an underrun message is always real.

**The action plots are on a wall-clock axis** — the last 8 seconds, 0 at the right edge —
because actions do not arrive on a regular grid. Manual control emits one per keypress and
nothing while the arm is still, and the policy skips ticks on an underrun, so plotting against
sample index drew a stalled minute and a busy second at the same width. The traces are drawn as
steps and carry the last command to the present edge, which is what the arm is actually doing:
a command holds until the next one supersedes it, and nothing interpolates between them.

Every run writes `results/deploy_<timestamp>/` (gitignored):

```
video.mp4      frames exactly as they were given to the policy
actions.jsonl  per tick: raw prediction, dispatched action, inference latency, underrun flag,
               `rtc`/`rtc_delay`/`rtc_prefix` (per tick, because the checkbox is live),
               and `link` -- the CommandBus's backlog, cumulative dropped count, and mean
               latency, so the log says whether a command reached the board and not just
               that it was submitted
run.json       checkpoint, task, fps, action mode, RTC settings, device, git SHA
```

**STOP ALL** and `<space>` stop the arm at any time. A stop also fires on Stop, on window close,
on link death, and from a `finally` around the main loop, each preceded by `cancel_pending()` so
a queued speed cannot land after it.

### Replaying training data through the policy

```sh
python deploy_policy_training_data_replay.py \
  --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
  --repo-id lone/l_one_green_marker --root data/lerobot/lone/l_one_green_marker --remote-camera
```

`deploy_policy.py` with the camera swapped for a recorded episode. It subclasses `DeployApp`
rather than copying it, so the checkpoint loading, action queue, snapping, `dispatch_action()`
and results logging it exercises are the ones that run on the arm for real.

It asks the easiest question that can be asked of a checkpoint: fed back the exact frames it
trained on, does the arm reproduce that trajectory and pick the thing up? A policy that cannot
imitate its own demonstration under its own training inputs has a problem in the inference and
dispatch path, and no amount of watching a live camera separates that from a generalization
failure. `scripts/eval_policy.py` answers the offline half of this; this answers the half with a
motor in it.

**Run it in this order.** The three runs isolate different failures, and taking them out of
order wastes the information:

1. **`Source = Dataset`.** Dispatches the *recorded* actions, with no policy in the loop, at the
   same tick rate through the same bus. If the arm cannot pick the marker up from the
   demonstration's own actions, the fault is start pose, timing or hardware -- stop here,
   because nothing about the checkpoint is being measured yet. Needs no checkpoint at all.
2. **`Source = Policy`, `--lookahead 0`.** What deployment actually does.
3. **`Source = Policy`, `--lookahead 10`.** The same run with the inference lag dialled out.

**Lag is measured, not assumed.** Every action carries the index of the frame it was predicted
from, so the log says exactly how stale each command was. Expect roughly `n_action_steps`: the
worker refills whenever the queue drops below a full chunk, so the queue settles near full and
an action reaches the arm about a chunk after the frame that produced it -- 400 ms at 25 Hz.
On a canned feed that makes the arm execute a *lagged* copy of the demonstration, which is its
own kind of failure and worth being able to see separately. `--lookahead K` submits frame
`cursor+K` while the panes, plots and recorded video still follow the cursor.

**Ground truth is on screen and in the numbers.** The action plots carry a dashed
*demonstration* trace beside the commanded one, and the status line shows a running match rate
-- the fraction of dimension-ticks where the dispatched action equalled the recorded one.
Under *Snap to demonstrated levels* both sides are on the same discrete grid, so exact agreement
is meaningful; in raw mode nothing matches exactly and the normalized MSE beside it is the
number to read. Both, plus mean and max lag, are appended to `actions.jsonl` as a
`run_summary` line when a run ends.

**The observation path is checked at startup.** Dataset frames reach the policy through
`deploy_policy.py`'s unmodified camera path -- uint8 BGR, `resize_keep_aspect`, `/255`, CHW --
and the log reports the max absolute difference against the tensor LeRobot itself returns.
It is `0.0` when the stored frame size equals `--image-width/height`, which is the case for
these datasets. Ruling out that class of silent preprocessing mismatch is most of the point.

**Feed perturbation is how you find out why a live camera does worse.** Contrast, brightness,
Gaussian noise and JPEG quality (75–100 in steps of 5, behind a checkbox — 75 is a real quality,
not "off"), adjustable while the arm is running, applied to the dataset frame before the deploy
path — so the pane, the recorded video and the policy all see the degraded image, with the
demonstration's own actions still underneath as the control. The readout gives the mean absolute
pixel change in 0–255 units, which is what puts the input change on a comparable scale to the
action change it produces.

The panel is collapsed by default, since a clean feed is the normal case, but **the header always
carries the current setting** — `▶ Feed perturbation · noise σ=5.0 · JPEG q80`. A run quietly
poisoned by noise left on from the previous one is exactly what a hidden panel would otherwise
cause.

These are not decorations. Training and the replay both see **AV1**-decoded pixels
(`meta/info.json`); a live camera skips that round trip entirely, `--remote-camera` adds JPEG q80
on top ([stream_camera.py](stream_camera.py)), and the room's lighting has moved since the
episode was recorded. Measured against the single-episode π0.5 checkpoint, a **0.3% mean pixel
change flipped the commanded action** — moving the prediction 42× further than the policy's own
flow-matching noise, from *drive the lower arm* to *stop*. Turn the dial up until the arm stops
doing the task; how far you got is the number worth having. `--brightness`, `--contrast`,
`--noise` and `--jpeg-quality` set the same knobs from the command line, and the settings land in
`run.json` and in every tick of `actions.jsonl`.

**`--repo-id` is required and `--root` follows it** (`data/lerobot/<repo-id>` unless you override
it), for the reason described under *Evaluating a checkpoint*: two independent defaults let you
name one dataset and replay another in silence. The log also says so when the episode you are
replaying is not the one `train_config.json` says the checkpoint trained on.

**The live camera sits directly above the demo frame**, same scene at the same scale, so the
difference between them is the only thing that draws the eye. None of this means anything unless
the arm starts where the demonstration started: scrub to frame 0 and teleoperate the real arm
until the two panes agree before pressing Start. `--no-camera` drops the pane. `--loop`,
`--start-frame` and `--episode` do what they say; the episode dropdown and frame slider are the
same controls while stopped.

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