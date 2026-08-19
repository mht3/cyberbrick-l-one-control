#!/usr/bin/env python3
"""Deployment GUI: drive the CyberBrick L-ONE arm from a trained checkpoint.

A peer of collect_data.py -- same connection bar, same joint controls and
keybindings, same camera dropdown -- with a policy mode on top.

The working shape of a deployment is: connect, teleoperate the arm to a sensible
starting position, check the task prompt, hand control to the policy, watch what
it commands, and take control back. So manual control is the default and always
the fallback; the policy is something you switch into and out of.

    python deploy_policy.py --checkpoint outputs/train/.../checkpoints/last/pretrained_model

The checkpoint is a command-line argument only. Choosing a policy is not something
to do by accident mid-session with an arm powered up.

Two details shape the control loop:

Inference is slow relative to the control rate -- pi0.5 takes ~280ms on an RTX
5090, most of a control period -- so it runs on a worker thread that keeps a queue
of upcoming actions filled, and the Tk tick only pops and dispatches. Same reason
CommandBus exists: anything blocking the main thread stalls control.

The policy emits continuous values, but the demonstrations only ever contained
three discrete levels per channel. "Snap" quantizes back onto them (default, and
closest to how the arm was actually driven); "raw" clamps to the command limits.
Both are logged either way, so the choice stays visible after the fact.

Real-time chunking (--rtc, and the checkbox) changes how one chunk gives way to
the next. By default a fresh chunk is appended behind whatever is still queued, so
the arm finishes an old plan drawn from a frame up to ~19 ticks ago before it sees
the new one, and the two plans meet at a seam nothing smoothed. With RTC the
denoiser is guided so the head of the new chunk agrees with the unexecuted tail of
the old one, and the queue is *replaced* rather than appended -- both a smoother
join and a much fresher plan. Note that snapping quantizes away most of the
smoothing: what survives is the timing of level changes, so the effect shows up as
fewer commands and fewer one-tick reversals rather than as visibly smoother
motion. The pre-snap values are logged as `raw` on every tick, which is where to
look for it.
"""

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from virtual_gripper import (
    BASE_MOTOR,
    GRIPPER_SERVO,
    LOWER_ARM_SERVO,
    PALETTE,
    UPPER_ARM_SERVO,
    LinkDesynced,
)

from lone_data import camera_source
from lone_data.checkpoints import dataset_tasks, training_dataset
from lone_data.command_bus import CommandBus
from lone_data.dispatch import clamp_to_limits, dispatch_action, snap_to_levels
from lone_data.features import (
    ACTION_COMMAND_LIMITS,
    ACTION_DIM,
    ACTION_NAMES,
    DEFAULT_IMAGE_SIZE,
    resize_keep_aspect,
)
from lone_data.robot_gui import RobotAppBase, WIFI_HEARTBEAT_MS

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNELS = (BASE_MOTOR, UPPER_ARM_SERVO, LOWER_ARM_SERVO, GRIPPER_SERVO)

DISPLAY_MAX_WIDTH = 640
FEED_REFRESH_MS = 50    # ~20Hz preview, deliberately below the control rate
PLOT_REFRESH_MS = 100   # ~10Hz, slower still so drawing never competes with control
PLOT_WINDOW_S = 8.0     # seconds of wall-clock history on screen
PLOT_MAX_SAMPLES = 4000  # hard bound on the buffer; the window is what trims it
MAX_HELD_TICKS = 3      # ticks to reuse the last action before stopping
PRIME_WARN_S = 2.0      # waiting this long for a run's first chunk is worth saying


def build_parser(description=__doc__):
    """The deployment argument set, as a parser a peer script can add to.

    deploy_policy_training_data_replay.py subclasses DeployApp, which reaches
    through RobotAppBase into args.remote_camera_port, args.width/height and
    args.camera_fps -- so it needs these arguments present, not a fresh parser
    with a subset of them.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--checkpoint", default=None,
                   help="a checkpoint's pretrained_model/ directory (required to run a policy)")
    p.add_argument("--task", default="", help="task prompt (defaults to the checkpoint's)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--remote-camera", action="store_true",
                   help="receive camera frames over the network (see stream_camera.py)")
    p.add_argument("--remote-camera-port", type=int, default=8267)
    p.add_argument("--fps", type=int, default=25, help="control rate")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override how many actions are executed per inference")
    p.add_argument("--device", default=None)
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    p.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])

    # Real-time chunking. Implemented in lerobot (lerobot.policies.rtc) and
    # supported by pi0.5; these only choose whether and how to switch it on. The
    # schedule stays a string here because build_parser() runs at import time and
    # every lerobot import in this file is deferred into PolicyRunner -- it
    # becomes an RTCAttentionSchedule there. Constrain it with choices=: an
    # unrecognized name raises UnboundLocalError from inside the denoiser.
    rtc = p.add_argument_group("real-time chunking")
    rtc.add_argument("--rtc", action="store_true",
                     help="guide each chunk to agree with the tail of the one it replaces "
                          "(flow-matching policies only -- pi0/pi0.5/SmolVLA)")
    rtc.add_argument("--rtc-execution-horizon", type=int, default=20,
                     help="ticks of the previous chunk the new one is guided against. Must "
                          "stay above the inference delay in ticks (~7 at 25fps) or the "
                          "guidance degenerates into a hard clamp with no ramp")
    rtc.add_argument("--rtc-max-guidance-weight", type=float, default=10.0,
                     help="how hard to enforce agreement; 10.0 suits 10 denoising steps")
    rtc.add_argument("--rtc-schedule", choices=("ZEROS", "ONES", "LINEAR", "EXP"),
                     default="LINEAR", help="how the agreement weight decays across the overlap")
    rtc.add_argument("--rtc-queue-threshold", type=int, default=30,
                     help="re-infer once the queue falls to this many actions. This is the "
                          "open-loop horizon: at or above chunk_size the worker replans "
                          "continuously")
    return p


def parse_args():
    return build_parser().parse_args()


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def checkpoint_task(checkpoint):
    """The task string the checkpoint trained on, so deployment matches training.

    pi0.5 conditions on this text, so a differently worded prompt is a different
    conditioning. It is not stored in the checkpoint -- train_config.json records only
    which dataset was used -- so follow that pointer and read the task out of the
    dataset's metadata. Returns "" if the dataset is not on this machine.
    """
    repo_id, root = training_dataset(checkpoint)
    if not repo_id or not root:
        return ""
    tasks = dataset_tasks(repo_id, root)
    # Several tasks means the run was multi-task and no single prompt is right.
    return tasks[0] if len(tasks) == 1 else ""


def rtc_settings(args):
    """The real-time chunking arguments as a plain dict.

    PolicyRunner takes this rather than the argparse namespace so it stays
    constructible from a test harness, and so the lerobot types the settings turn
    into stay behind the deferred import.
    """
    return {
        "enabled": bool(getattr(args, "rtc", False)),
        "execution_horizon": int(getattr(args, "rtc_execution_horizon", 20)),
        "max_guidance_weight": float(getattr(args, "rtc_max_guidance_weight", 10.0)),
        "schedule": str(getattr(args, "rtc_schedule", "LINEAR")),
        "queue_threshold": int(getattr(args, "rtc_queue_threshold", 30)),
    }


# The statistic pair each normalization mode decodes against, per
# lerobot/processor/normalize_processor.py. MEAN_STD is absent on purpose:
# mean/std make no claim about an envelope, so there is nothing to check.
NORM_STAT_PAIR = {
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}
EXPECTED_NORM_MODE = "MIN_MAX"


def action_stats_warning(checkpoint):
    """Why this checkpoint's action decoding is not what deployment expects, or None.

    Actions are normalized MIN_MAX against ACTION_COMMAND_LIMITS -- see the
    Normalization statistics section of the README for why quantiles are the wrong
    statistic for three-level commands. Both halves of that are checked here,
    because both fail silently: the mapping stays invertible either way, and the
    only symptom is an arm that drives a little too far on the affected joints.

    Advisory only. Never blocks a load.
    """
    try:
        import glob

        from safetensors.torch import load_file

        matches = glob.glob(os.path.join(checkpoint, "*normalizer_processor.safetensors"))
        if not matches:
            return None
        # Read the mode off the saved pipeline rather than assuming it. Checking
        # the wrong pair is worse than not checking: a MIN_MAX checkpoint can
        # carry skewed quantiles and still be correct, because nothing reads them.
        mode = None
        with open(os.path.join(checkpoint, "policy_preprocessor.json")) as f:
            for step in json.load(f).get("steps", []):
                norm_map = step.get("config", {}).get("norm_map") or {}
                if "ACTION" in norm_map:
                    mode = norm_map["ACTION"]
        if mode is None:
            return None
        stats = load_file(matches[0])
    except Exception:
        return None

    problems = []
    if mode != EXPECTED_NORM_MODE:
        problems.append(
            f"actions are normalized {mode}, not {EXPECTED_NORM_MODE} -- retrain with "
            f"--policy.normalization_mapping='{{...,\"ACTION\":\"{EXPECTED_NORM_MODE}\"}}'"
        )

    if mode in NORM_STAT_PAIR:
        lo_key, hi_key = NORM_STAT_PAIR[mode]
        try:
            low = stats[f"action.{lo_key}"].flatten().tolist()
            high = stats[f"action.{hi_key}"].flatten().tolist()
        except KeyError:
            return "; ".join(problems) or None
        skewed = []
        for i, name in enumerate(ACTION_NAMES):
            lo, hi = ACTION_COMMAND_LIMITS[i]
            # Both ends within a percent of the declared envelope is the intended state.
            if abs(low[i] - lo) > abs(hi - lo) * 0.01 or abs(high[i] - hi) > abs(hi - lo) * 0.01:
                span = high[i] - low[i]
                at_zero = 2 * (0.0 - low[i]) / span - 1 if span else 0.0
                skewed.append(f"{name} {lo_key}/{hi_key} {low[i]:.0f}/{high[i]:.0f} vs limits "
                              f"{lo:.0f}/{hi:.0f} (stopped decodes at {at_zero:+.2f}, not 0)")
        if skewed:
            problems.append(
                f"{lo_key}/{hi_key} does not span the command limits on {len(skewed)} of "
                f"{len(ACTION_NAMES)} dimensions, so those joints drive off-center: "
                + "; ".join(skewed)
                + " -- rerun scripts/fix_action_stats.py on the training dataset"
            )

    if not problems:
        return None
    return "Action normalization: " + ". ".join(problems) + "."


def validate_checkpoint(path):
    """Reason `path` is not a usable checkpoint directory, or None."""
    if not path:
        return "No --checkpoint given; manual control only."
    if not os.path.isdir(path):
        return f"{path} is not a directory."
    for name in ("config.json", "model.safetensors"):
        if not os.path.exists(os.path.join(path, name)):
            return f"{path} has no {name} -- point at a checkpoint's pretrained_model/ folder."
    return None


class PolicyRunner:
    """Loads a checkpoint and serves action chunks from a worker thread.

    Inference is far too slow to run inline with a 25-30Hz control tick, so the
    thread keeps a queue filled and the caller only ever pops.

    Constructing this loads the weights, which for pi0.5 is seconds of disk and
    host-to-device copying; the task prompt arrives per run at `start()` instead,
    so one instance serves every rollout of a session and a checkpoint is never
    loaded twice.
    """

    def __init__(self, checkpoint, device=None, n_action_steps=None, fps=25, rtc=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.checkpoint = checkpoint
        self.task = ""
        self.fps = int(fps) or 25

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if device:
            cfg.device = device
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_type = cfg.type
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint)
        self.policy.eval().to(self.device)
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
        self.n_action_steps = int(n_action_steps or getattr(cfg, "n_action_steps", 1) or 1)
        # Read here rather than on the Tk thread; surfaced by the caller once the
        # load lands, so it appears in the log next to "Checkpoint ready".
        self.stats_warning = action_stats_warning(checkpoint)
        self._configure_rtc(rtc)

        self._pending = deque()
        # The chunk _pending was derived from, before the postprocessor and still
        # on the policy's device -- the space RTC's prefix has to be expressed in.
        # _consumed indexes into it, so the invariant every merge must preserve is
        #     len(_pending) == len(_chunk_raw) - _consumed
        # and _pending[i] is _chunk_raw[_consumed + i], post-processed. Track the
        # index explicitly rather than deriving it from len(_pending): the two
        # diverge as soon as the queue is appended to rather than replaced, and a
        # wrong offset is a silent slice off the wrong end, not an error.
        self._chunk_raw = None
        self._consumed = 0
        self._lock = threading.Lock()
        self._frame = None
        # Caller-supplied label for the frame currently staged for inference. Every
        # action the worker queues carries the label of the frame it was predicted
        # from, which is the only way to know afterwards how stale a dispatched
        # action was -- queue depth alone cannot recover it.
        self._frame_meta = None
        self._running = False
        self._thread = None
        # Bumped per run so a worker that outlived its stop() -- join() times out
        # while an inference is still in flight -- cannot push actions into the
        # next run's queue or report its error against it.
        self._generation = 0
        self.last_latency = None
        self.error = None
        # Last merge's shape, for the status line and the per-tick log.
        self.last_rtc_delay = None
        self.last_rtc_prefix = None

    def _configure_rtc(self, rtc):
        """Attach an RTCConfig to the loaded policy, if it can use one.

        lerobot builds the processor from policy.config.rtc_config, and reads
        .enabled fresh on every inference -- so this runs once at load and the
        checkbox only has to flip a bool afterwards. It must be policy.config and
        not the cfg above: from_pretrained() loads a config object of its own, and
        assigning to the wrong one is a silent no-op.
        """
        supports = getattr(self.policy, "supports_rtc", None)
        self.supports_rtc = bool(callable(supports) and supports())
        self.rtc_config = None
        self.rtc_enabled = False
        self.rtc_execution_horizon = int((rtc or {}).get("execution_horizon", 20))
        self.rtc_queue_threshold = int((rtc or {}).get("queue_threshold", 30))
        if not self.supports_rtc or rtc is None:
            return
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc import RTCConfig

        self.rtc_config = RTCConfig(
            enabled=bool(rtc["enabled"]),
            prefix_attention_schedule=RTCAttentionSchedule(rtc["schedule"]),
            max_guidance_weight=float(rtc["max_guidance_weight"]),
            execution_horizon=self.rtc_execution_horizon,
        )
        self.policy.config.rtc_config = self.rtc_config
        self.policy.init_rtc_processor()
        self.rtc_enabled = bool(rtc["enabled"])

    def start(self, task):
        """Serve actions for `task`, from a clean queue. Reusable across runs."""
        self.stop()
        self.task = task
        self.error = None
        with self._lock:
            self._generation += 1
            generation = self._generation
            self.last_latency = None
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(generation,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        with self._lock:
            # Actions predicted from the old run's frames must not survive into
            # the next one, and a stale frame must not seed its first inference.
            # The chunk mirror goes with them: guiding a new run's first chunk
            # toward the previous run's plan is exactly what it must not do.
            self._pending.clear()
            self._chunk_raw = None
            self._consumed = 0
            self._frame = None
            self._frame_meta = None
            self.last_rtc_delay = None
            self.last_rtc_prefix = None

    def submit_frame(self, rgb, meta=None):
        with self._lock:
            self._frame = rgb
            self._frame_meta = meta

    def pop(self):
        """(action, meta) for the next action, or None if the queue is empty.

        `meta` is whatever submit_frame() was given for the frame this action was
        predicted from -- None in live deployment, the dataset frame index in the
        training-data replay.
        """
        with self._lock:
            if not self._pending:
                return None
            # The one place an action leaves the queue, so the one place the RTC
            # prefix's start index may move.
            self._consumed += 1
            return self._pending.popleft()

    @property
    def queued(self):
        with self._lock:
            return len(self._pending)

    @property
    def is_serving(self):
        """Whether a worker is actually producing actions right now.

        False for a loaded-but-idle runner, which is not the same as a queue that
        happens to be empty: the training-data replay can drive the arm from the
        recording with the policy loaded and deliberately out of the loop.
        """
        return self._running

    def _observation(self, rgb):
        # Must match what the dataset delivered at training time: CHW float in [0,1].
        image = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return {
            "observation.images.front": image,
            "observation.state": self.torch.zeros(ACTION_DIM),
            "task": self.task,
        }

    def _run(self, generation):
        from lerobot.policies.rtc import LatencyTracker

        # Per-run, so one run's hiccup cannot inflate the next run's delay estimate.
        tracker = LatencyTracker(maxlen=50)
        while self._running and generation == self._generation:
            # Snapshot once per cycle. The worker is the only writer of
            # rtc_config.enabled, and lerobot reads it once per inference, so a
            # click mid-cycle cannot leave the guidance and the merge disagreeing
            # about which mode this chunk was drawn in.
            rtc_on = bool(self.rtc_config is not None and self.rtc_enabled)
            with self._lock:
                need = (len(self._pending) <= self.rtc_queue_threshold if rtc_on
                        else len(self._pending) < self.n_action_steps)
                frame, meta = self._frame, self._frame_meta
            if not need or frame is None:
                time.sleep(0.005)
                continue
            try:
                # Timed from here, not from the model call: what the delay has to
                # measure is how much of the world moved on between this frame and
                # these actions reaching the queue, and tokenizing a 200-token
                # prompt is part of that.
                started = time.perf_counter()
                batch = {
                    k: (v.unsqueeze(0) if self.torch.is_tensor(v) else [v])
                    for k, v in self._observation(frame).items()
                }
                prepared = self.pre(batch)

                # Read after preprocessing, not before: tokenizing is milliseconds,
                # but a prefix read ahead of it can have had its first row
                # dispatched in the meantime, which shifts the whole alignment.
                kwargs, prefix = {}, None
                with self._lock:
                    if self.rtc_config is not None:
                        self.rtc_config.enabled = rtc_on
                    if rtc_on and self._chunk_raw is not None:
                        prefix = self._chunk_raw[
                            self._consumed:self._consumed + self.rtc_execution_horizon
                        ].clone()
                    if prefix is not None and len(prefix):
                        # p95, not max(): LatencyTracker.max_latency is a lifetime
                        # high-water mark that ignores the window, so one slow
                        # inference would pin the delay forever -- and a delay at
                        # or past the horizon flattens the schedule into a hard
                        # clamp with no ramp left to blend over.
                        delay = min(int(math.ceil(tracker.p95() * self.fps)),
                                    self.rtc_execution_horizon)
                        kwargs = {"inference_delay": delay, "prev_chunk_left_over": prefix}
                    else:
                        prefix = None

                # no_grad, never inference_mode: RTC re-enables grad inside the
                # denoiser and needs to set requires_grad on the latent, which
                # inference_mode forbids.
                with self.torch.no_grad():
                    chunk = self.policy.predict_action_chunk(prepared, **kwargs)
                original = chunk[0].detach().clone()  # pre-postprocessor, on device
                actions = self.post(chunk)[0].float().cpu().numpy()
                latency = time.perf_counter() - started
                tracker.add(latency)

                with self._lock:
                    if generation != self._generation:
                        return
                    if rtc_on:
                        # Chunk index i is the action for observation time + i
                        # ticks, so what has gone stale is wall clock, not pops --
                        # pops undercount through held ticks and underruns, and
                        # dispatching on that count replays moments already past.
                        dropped = max(0, min(int(math.ceil(latency * self.fps)), len(actions)))
                        self._pending.clear()
                        for action in actions[dropped:]:
                            self._pending.append((np.asarray(action, dtype=np.float32), meta))
                        self._chunk_raw = original[dropped:]
                        self.last_rtc_delay = dropped
                        self.last_rtc_prefix = 0 if prefix is None else int(prefix.shape[0])
                    else:
                        # Append this chunk's first n_action_steps behind whatever
                        # is still queued. The mirror is kept up to date here too,
                        # so that switching RTC on mid-run has a valid prefix to
                        # guide its first chunk against.
                        kept = (self._chunk_raw[self._consumed:]
                                if self._chunk_raw is not None else original[:0])
                        self._chunk_raw = self.torch.cat([kept, original[: self.n_action_steps]])
                        for action in actions[: self.n_action_steps]:
                            self._pending.append((np.asarray(action, dtype=np.float32), meta))
                        self.last_rtc_delay = self.last_rtc_prefix = None
                    self._consumed = 0
                    self.last_latency = latency
            except Exception as e:  # surfaced on the Tk thread by the caller
                if generation != self._generation:
                    return
                self.error = e
                self._running = False
                return


class DeployApp(RobotAppBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("CyberBrick L-ONE Policy Deployment")
        self.configure(background=PALETTE["bg"])
        self.minsize(1100, 720)
        self._setup_style()

        # -- robot link state --------------------------------------------
        self.link = None
        self.ap_ip = None
        self.wifi_kind = None
        self.gripper_open = True
        self._pressed_keys = set()
        self._key_release_after = {}
        self._wifi_connect_generation = 0
        self._shutting_down = False
        self._link_up_mono = None
        self._failed_reconnects = 0

        self.bus = CommandBus(
            on_error=lambda m: self.after(0, self._log, f"Command failed: {m}", "error"),
            on_link_dead=lambda m: self.after(0, self._on_link_dead, m),
            fatal_errors=(LinkDesynced,),
        )
        self.bus.start()

        # -- camera -------------------------------------------------------
        self.camera = None
        # Remote is only offered when the app was launched to receive a stream:
        # on a machine nothing streams to, it is a dead menu entry.
        self._camera_source = camera_source.pick_initial_source(
            camera_source.REMOTE_SOURCE if args.remote_camera else args.camera_index,
            include_remote=args.remote_camera,
        )
        self._camera_connected = False
        self.image_size = (args.image_height, args.image_width)
        # Instance-level so a subclass showing two video panes can narrow both.
        self._display_width = DISPLAY_MAX_WIDTH

        # -- action state --------------------------------------------------
        # _current_action is the single record of what was last commanded, whoever
        # commanded it. The plots and the recorder read only this, so they behave
        # identically in manual and policy mode and cannot disagree about what the
        # arm was told to do.
        self._current_action = np.zeros(ACTION_DIM, dtype=np.float32)
        # (wall-clock time, action) pairs. Timestamped rather than counted because
        # actions do not arrive on a regular grid -- manual control emits one per
        # keypress and none while the arm is still -- so an index axis would draw
        # a stalled minute and a busy second at the same width.
        self._history = deque(maxlen=PLOT_MAX_SAMPLES)
        self._period = 1.0 / args.fps
        self._feed_ticks = 0

        # -- policy state --------------------------------------------------
        # The runner is built once, at startup, and outlives every rollout: it
        # holds the loaded weights, and only the task and the action queue are
        # per-run. `None` means no checkpoint, or one that failed to load.
        self.runner = None
        self._policy_loading = False
        self._policy_running = False
        self._next_deadline = None
        self._held = None
        self._held_ticks = 0
        self._underrun_reported = False
        self._priming = False
        self._prime_started = None
        self._prime_warned = False
        self._tick_index = 0
        self._last_raw = None
        # Label of the frame the action being dispatched was predicted from; see
        # PolicyRunner.submit_frame. Always None in live deployment.
        self._action_meta = None
        # What the board was last told. dispatch_action() sends only the difference
        # against it; None means "re-send everything", which is what a run's first
        # tick and anything after a stop_all need.
        self._last_dispatched = None

        # -- recording state (results/ only; never a LeRobotDataset) --------
        self._recording = False
        self._writer = None
        self._log_file = None
        self._result_dir = None

        # -- UI ---------------------------------------------------------
        self._build_log_panel(self)
        self._build_plot_panel(self)

        main_row = ttk.Frame(self)
        main_row.pack(fill="both", expand=True)

        camera_frame = ttk.Frame(main_row, padding=(16, 16, 8, 16))
        camera_frame.pack(side="left", fill="both", expand=True)

        video_group = ttk.Frame(camera_frame)
        video_group.pack(anchor="n")

        source_row = ttk.Frame(video_group)
        source_row.pack(fill="x", pady=(0, 6))
        ttk.Label(source_row, text="Camera").pack(side="left")
        self.camera_source_var = tk.StringVar(value=self._camera_source_label(self._camera_source))
        self.camera_combo = ttk.Combobox(
            source_row, textvariable=self.camera_source_var, width=22, state="readonly",
            postcommand=self._refresh_camera_sources,
        )
        self.camera_combo.pack(side="left", padx=(8, 0))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_source)

        self.camera_label = ttk.Label(video_group)
        self.camera_label.pack()

        # Below the camera pane, in the same column: a subclass with a second video
        # source stacks it under the live one rather than competing for width.
        self._build_extra_video_panel(camera_frame)

        right = ttk.Frame(main_row, padding=(8, 16, 16, 16))
        right.pack(side="left", fill="y")

        self._build_connection_bar(right)
        self.joint_controls_frame = self._build_controls(right)
        self._build_policy_controls(right)
        self._build_extra_controls(right)

        self._setup_keybindings()
        self.bind_all("<Button-1>", self._maybe_reclaim_focus, add="+")
        self.protocol("WM_DELETE_WINDOW", self._shutdown)
        signal.signal(signal.SIGINT, lambda signum, frame: self._shutdown())
        self._signal_pump()

        self.mode_var.set("wifi")
        self._on_mode_select()
        self._set_policy_buttons_state()
        self._update_status_label()
        self.focus_set()

        self._begin_policy_load()

        if not self._open_camera(self._camera_source):
            self._log("Pick another source from the Camera dropdown, or run "
                      "stream_camera.py elsewhere and choose Remote.", level="warn")

        self.after(0, self._update_camera_feed)
        self.after(WIFI_HEARTBEAT_MS, self._wifi_heartbeat_tick)
        self.after(PLOT_REFRESH_MS, self._plot_tick)

    # -- RobotAppBase hooks ------------------------------------------------

    def _controls_enabled(self):
        """Manual input is ignored while the policy drives -- the buttons are
        visibly disabled, and a keypress landing mid-rollout would fight it."""
        return not self._policy_running

    def _controls_busy(self):
        return "Stop the policy" if self._policy_running else None

    def _mode_status_text(self):
        return "Mode: Policy" if self._policy_running else "Mode: Manual"

    def _on_link_state_changed(self):
        self._set_policy_buttons_state()

    def _on_link_lost(self):
        if self._policy_running:
            self._log("Link lost while the policy was running -- returning to manual", level="warn")
            self._stop_policy(reason="link lost")

    def _on_emergency_stop(self):
        if self._policy_running:
            self._stop_policy(reason="STOP ALL pressed")

    def _on_action_changed(self):
        """Manual control just moved a joint: record it like a policy action would be."""
        if not self._policy_running:
            # Manual control talks to the bus directly rather than through
            # dispatch_action(), so what the board holds no longer matches the cache.
            self._last_dispatched = None
            self._note_action(raw=None, dispatched=self._current_action, underrun=False)

    def _held_speeds(self):
        """While the policy drives, the held speeds are its, not JointControl's."""
        if self._policy_running:
            return tuple(float(v) for v in self._current_action[:3])
        return super()._held_speeds()

    # -- hooks: defaults are the live-camera deployment --------------------
    #
    # deploy_policy_training_data_replay.py is this app with the camera swapped for
    # a recorded episode. It subclasses rather than copies, so the checkpoint, the
    # action queue, snapping, dispatch and logging it exercises are the ones that
    # actually run on the arm -- these are the seams where the two differ.

    def _build_extra_video_panel(self, parent):
        """Another video pane, stacked under the camera one."""

    def _build_extra_controls(self, parent):
        """More controls, below the policy section."""

    def _policy_observation(self):
        """(rgb, meta) for this tick, or (None, None) if there is no frame yet.

        `rgb` is what the policy sees, at self.image_size; `meta` labels where it
        came from and rides along with every action predicted from it.
        """
        latest = self.camera.get_latest() if self.camera is not None else None
        if latest is None:
            return None, None
        return resize_keep_aspect(latest[0], self.image_size), None

    def _observation_ready(self):
        """Why the policy cannot be started yet, or None.

        Starting blind would drive the arm from whatever frame arrives first.
        """
        if self.camera is None or self.camera.get_latest() is None:
            return "No camera frame yet -- wait for the feed before starting."
        return None

    def _extra_log_fields(self):
        """Additional keys for this tick's actions.jsonl entry."""
        return {}

    def _run_metadata(self):
        """The dict written to a recording's run.json."""
        return {
            "checkpoint": self.args.checkpoint,
            "task": self.task_var.get().strip(),
            "fps": self.args.fps,
            "action_mode": "snap" if self._snap_var.get() else "raw",
            # The settings, not the state: the checkbox is live, so whether any
            # given tick was guided is recorded per tick in actions.jsonl.
            "rtc": {
                **rtc_settings(self.args),
                "enabled": bool(self._rtc_var.get()),
                "supported": bool(self.runner and self.runner.supports_rtc),
            },
            "policy_type": self.runner.policy_type if self.runner else None,
            "device": self.runner.device if self.runner else None,
            "image_size": list(self.image_size),
            "git_sha": git_sha(),
        }

    def _draw_extra_plots(self, now):
        """Extra traces on self.axes, before the canvas is redrawn."""

    # -- policy controls ---------------------------------------------------

    def _build_policy_controls(self, parent):
        frame = ttk.Frame(parent, padding=(0, 8, 0, 0))
        frame.pack(fill="x")

        ttk.Label(frame, text="Policy", style="SectionHeading.TLabel").pack(anchor="w", pady=(0, 4))

        task_row = ttk.Frame(frame)
        task_row.pack(fill="x", pady=(0, 8))
        ttk.Label(task_row, text="Task", width=12, style="JointName.TLabel").pack(side="left")
        self.task_var = tk.StringVar(value=self.args.task or checkpoint_task(self.args.checkpoint))
        self.task_entry = ttk.Entry(task_row, textvariable=self.task_var, width=30)
        self.task_entry.pack(side="left", fill="x", expand=True)

        # The policy emits continuous values; the demonstrations only ever contained
        # three levels per channel, so snapping keeps the arm inside the distribution
        # it was trained on. Raw is there to see what the policy actually meant.
        self._snap_var = tk.BooleanVar(value=True)
        mode_row = ttk.Frame(frame)
        mode_row.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(mode_row, text="Snap to levels", variable=self._snap_var,
                        value=True).pack(side="left")
        ttk.Radiobutton(mode_row, text="Raw (clamped)", variable=self._snap_var,
                        value=False).pack(side="left", padx=(10, 0))

        # Unlike the snap radios, which _control_tick reads live, this has to
        # reach a plain attribute the worker thread polls -- hence a command
        # rather than a bare variable. It stays enabled while the policy runs on
        # purpose: comparing the two within one rollout is the point of it.
        self._rtc_var = tk.BooleanVar(value=bool(getattr(self.args, "rtc", False)))
        rtc_row = ttk.Frame(frame)
        rtc_row.pack(fill="x", pady=(0, 8))
        self.rtc_check = ttk.Checkbutton(rtc_row, text="Real-time chunking",
                                         variable=self._rtc_var, command=self._on_rtc_toggle)
        self.rtc_check.pack(side="left")

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill="x")
        self.start_policy_btn = ttk.Button(
            btn_row, text="Start Policy", style="Accent.TButton", command=self._start_policy, width=14
        )
        self.start_policy_btn.pack(side="left")
        self.stop_policy_btn = ttk.Button(
            btn_row, text="Stop Policy", command=lambda: self._stop_policy(), width=13
        )
        self.stop_policy_btn.pack(side="left", padx=(6, 0))
        self.record_btn = ttk.Button(btn_row, text="Record", command=self._toggle_record, width=10)
        self.record_btn.pack(side="left", padx=(6, 0))

        self.policy_status_var = tk.StringVar(value="Manual control.")
        ttk.Label(frame, textvariable=self.policy_status_var, style="Status.TLabel").pack(
            anchor="w", pady=(8, 0)
        )
        self.policy_detail_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.policy_detail_var, style="Status.TLabel").pack(anchor="w")
        return frame

    def _set_policy_buttons_state(self):
        can_start = (
            not self._policy_running
            and not self._policy_loading
            and self.link is not None
            and self.runner is not None
        )
        self.start_policy_btn.state(["!disabled"] if can_start else ["disabled"])
        self.stop_policy_btn.state(["!disabled"] if self._policy_running else ["disabled"])
        for child in self.joint_controls_frame.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.state(["disabled"] if self._policy_running else ["!disabled"])
        self.record_btn.configure(text="Stop Record" if self._recording else "Record")
        # Deliberately outside the loop above: this one stays usable mid-run.
        # runner is None until the load worker lands, and this runs during __init__.
        usable = self.runner is not None and getattr(self.runner, "supports_rtc", False)
        self.rtc_check.state(["!disabled"] if usable else ["disabled"])
        if not usable:
            self._rtc_var.set(False)
        self.app_mode_status_var.set(self._mode_status_text())

    def _on_rtc_toggle(self):
        if self.runner is None:
            return
        enabled = bool(self._rtc_var.get())
        self.runner.rtc_enabled = enabled
        if enabled:
            self._log("Real-time chunking on -- new chunks replace the queue, guided "
                      "to meet the plan already running.")
        else:
            self._log("Real-time chunking off -- the queue drains before the next "
                      "replan, so it may be a second before this takes effect.", level="warn")
        self._update_status_label()

    def _update_status_label(self):
        if self._policy_running and self.runner is not None:
            latency = self.runner.last_latency
            if getattr(self.runner, "rtc_enabled", False):
                # No budget to quote: the worker refills on a queue threshold
                # rather than racing the drain of one chunk's n_action_steps.
                detail = f"queued {self.runner.queued}  ·  rtc"
                if latency:
                    detail += f"  ·  inference {latency * 1000:.0f}ms"
                if self.runner.last_rtc_delay is not None:
                    detail += f"  ·  delay {self.runner.last_rtc_delay} ticks"
            else:
                # n_action_steps - 1: the worker refills as soon as the queue drops
                # below a full chunk, which is the first pop, so what it has to beat
                # is the drain of what is left after that pop -- not the whole chunk.
                budget = max(1, self.runner.n_action_steps - 1) * self._period
                detail = f"queued {self.runner.queued}"
                if latency:
                    detail += f"  ·  inference {latency * 1000:.0f}ms / {budget * 1000:.0f}ms budget"
            self.policy_status_var.set(
                "Policy priming -- waiting for the first actions"
                if self._priming else f"Policy running -- tick {self._tick_index}"
            )
            self.policy_detail_var.set(detail)
        elif self._policy_loading:
            self.policy_status_var.set("Loading checkpoint...")
            self.policy_detail_var.set(
                f"camera {self._camera_fps():.1f} fps  ·  {self._link_readout()}"
            )
        else:
            self.policy_status_var.set(
                "Recording (manual control)." if self._recording else "Manual control."
            )
            self.policy_detail_var.set(
                f"camera {self._camera_fps():.1f} fps  ·  {self._link_readout()}"
            )

    # -- policy lifecycle --------------------------------------------------

    def _begin_policy_load(self):
        """Load the checkpoint once, at startup, off the Tk thread.

        pi0.5 is ~4B parameters and takes tens of seconds to reach the GPU. Doing
        that on the first Start Policy click froze the window at the moment the
        arm was about to move, and paid the cost again on every later run; doing
        it inline here would just move the freeze to launch. So it happens on a
        worker while the GUI comes up, and Start stays disabled until it lands.
        """
        problem = validate_checkpoint(self.args.checkpoint)
        if problem:
            self._log(problem, level="warn")
            return
        self._log(f"Checkpoint: {self.args.checkpoint}")
        self._log("Loading checkpoint...")
        self._policy_loading = True
        self._load_started = time.monotonic()
        self._set_policy_buttons_state()
        self._update_status_label()
        threading.Thread(target=self._load_policy_worker, daemon=True).start()

    def _load_policy_worker(self):
        try:
            runner = PolicyRunner(
                self.args.checkpoint,
                device=self.args.device,
                n_action_steps=self.args.n_action_steps,
                fps=self.args.fps,
                rtc=rtc_settings(self.args),
            )
        except Exception as e:
            runner, error = None, e
        else:
            error = None
        try:
            self.after(0, self._on_policy_loaded, runner, error)
        except Exception:
            pass  # window already gone

    def _on_policy_loaded(self, runner, error):
        self._policy_loading = False
        if self._shutting_down:
            return
        if error is not None:
            self._log(f"Could not load checkpoint: {error}", level="error")
        else:
            self.runner = runner
            self._log(f"Checkpoint ready in {time.monotonic() - self._load_started:.0f}s: "
                      f"{runner.policy_type} on {runner.device}, "
                      f"{runner.n_action_steps} actions per inference"
                      f"{', RTC available' if runner.supports_rtc else ', no RTC'}",
                      level="connected")
            if runner.stats_warning:
                self._log(runner.stats_warning, level="warn")
            if self.args.rtc and not runner.supports_rtc:
                self._log(f"--rtc ignored: {runner.policy_type} is not a flow-matching "
                          "policy and has no real-time chunking path.", level="warn")
            self._rtc_var.set(bool(runner.rtc_enabled))
        self._set_policy_buttons_state()
        self._update_status_label()

    def _start_policy(self):
        if self._policy_running:
            return
        if self._policy_loading:
            self._log("Checkpoint is still loading.", level="warn")
            return
        if self.runner is None:
            self._log(validate_checkpoint(self.args.checkpoint)
                      or "Checkpoint failed to load; restart to retry.", level="warn")
            return
        if self.link is None:
            self._log("Connect to the board before starting the policy.", level="warn")
            return
        not_ready = self._observation_ready()
        if not_ready:
            self._log(not_ready, level="warn")
            return
        task = self.task_var.get().strip()
        if not task:
            self._log("Enter the task prompt the policy was trained with.", level="warn")
            self.task_entry.focus_set()
            return

        self.runner.start(task)
        # Hand over from a known state: manual control may have left a joint
        # driving, and the policy's first action is a whole inference away.
        self.bus.cancel_pending()
        self.bus.send_now("stop_all")
        self._current_action[0:3] = 0.0
        self._policy_running = True
        self._held = None
        self._held_ticks = 0
        self._action_meta = None
        # stop_all bypassed the bus, so nothing the board holds can be assumed.
        self._last_dispatched = None
        self._underrun_reported = False
        self._tick_index = 0
        # Priming: no chunk has arrived yet. An empty queue means the first
        # inference is still running, which is not the same failure as the queue
        # running dry mid-rollout, and must not be reported as one.
        self._priming = True
        self._prime_started = time.monotonic()
        self._prime_warned = False
        self._note_action(raw=None, dispatched=self._current_action, underrun=False)
        self._set_policy_buttons_state()
        self._log(f"Policy running: {task!r}", level="connected")
        self._next_deadline = time.monotonic()
        self._control_tick()

    def _stop_policy(self, reason="stopped"):
        if not self._policy_running:
            return
        self._policy_running = False
        self._last_dispatched = None
        try:
            self.bus.cancel_pending()
            self.bus.send_now("stop_all")
        finally:
            if self.runner is not None:
                self.runner.stop()  # the loaded weights stay, for the next run
        # The policy left the arm wherever it was; manual state must agree.
        self._reset_action_state(reopen_gripper=False)
        self._set_policy_buttons_state()
        self._update_status_label()
        self._log(f"Policy {reason} -- back to manual control.", level="warn")

    def _control_tick(self):
        if not self._policy_running:
            return
        self._next_deadline += self._period
        delay = max(1, int((self._next_deadline - time.monotonic()) * 1000))
        self.after(delay, self._control_tick)

        if self.runner is not None and self.runner.error is not None:
            self._log(f"Inference failed: {self.runner.error}", level="error")
            self._stop_policy(reason="inference error")
            return

        rgb, meta = self._policy_observation()
        if rgb is None:
            return
        self.runner.submit_frame(rgb, meta)

        popped = self.runner.pop()
        underrun = popped is None
        if underrun and self._priming:
            # The frame that triggers the first inference was submitted three
            # lines above, on this very tick, so the queue cannot possibly have
            # anything in it yet -- every run began by announcing an underrun and
            # stopping an arm that was already stopped. Nothing is dispatched
            # until the first chunk lands; the arm stays where teleop left it.
            waited = time.monotonic() - self._prime_started
            if waited > PRIME_WARN_S and not self._prime_warned:
                self._log(f"No actions yet, {waited:.1f}s after start -- the first "
                          "inference is slower than usual.", level="warn")
                self._prime_warned = True
            return
        if underrun:
            # Hold briefly through jitter, then stop rather than drive on stale commands.
            self._held_ticks += 1
            if self._held is None or self._held_ticks > MAX_HELD_TICKS:
                if not self._underrun_reported:
                    self._log("Inference underrun -- stopping the arm", level="warn")
                    self._underrun_reported = True
                self.bus.cancel_pending()
                self.bus.send_now("stop_all")
                self._held = None
                self._last_dispatched = None
                # A stop is a command like any other, and on a time axis that
                # holds the last value to the present, leaving it unrecorded
                # draws the arm as still driving at whatever it drove at last.
                # stop_all leaves the gripper where it is, so only 0-2 zero.
                self._current_action[0:3] = 0.0
                self._note_action(raw=None, dispatched=self._current_action,
                                  underrun=True, frame=rgb)
                return
            # A held action is still the one predicted from its original frame, so
            # its label carries over rather than going blank for the held ticks.
            action, raw = self._held, None
        else:
            self._priming = False
            self._held_ticks = 0
            self._underrun_reported = False
            raw, self._action_meta = popped
            action = snap_to_levels(raw) if self._snap_var.get() else clamp_to_limits(raw)
            self._held = action

        dispatch_action(self.bus, action, CHANNELS, last=self._last_dispatched)
        self._last_dispatched = np.asarray(action, dtype=np.float32).copy()
        self._current_action[:] = action
        self._note_action(raw=raw, dispatched=action, underrun=underrun, frame=rgb)
        self._tick_index += 1
        if self._tick_index % max(1, self.args.fps // 5) == 0:
            self._update_status_label()

    # -- action history, plots, recording ----------------------------------

    def _note_action(self, raw, dispatched, underrun, frame=None):
        """One place where an action becomes history, a plot point and a log line."""
        self._history.append((time.time(), np.asarray(dispatched, dtype=np.float32).copy()))
        if not self._recording:
            return
        if frame is not None and self._writer is not None:
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if self._log_file is None:
            return
        entry = {
            "tick": self._tick_index,
            "t": round(time.time(), 4),
            "mode": "policy" if self._policy_running else "manual",
            "raw": None if raw is None else [round(float(v), 3) for v in raw],
            "dispatched": [round(float(v), 3) for v in dispatched],
            "underrun": bool(underrun),
        }
        # Only while the policy drives: the runner is loaded in manual mode too,
        # and an idle queue depth logged against a hand-driven tick reads as if a
        # policy had been involved in it. Same for a tick driven from a recording
        # with the policy deliberately out of the loop.
        if self._policy_running and self.runner is not None and self.runner.is_serving:
            entry["queued"] = self.runner.queued
            entry["inference_s"] = (round(self.runner.last_latency, 4)
                                    if self.runner.last_latency else None)
            # Per tick because the checkbox is live. rtc_delay is also how the
            # seams are found offline: it is rewritten exactly when a chunk was
            # replaced, so a change in it marks the tick a new plan took over.
            entry["rtc"] = bool(self.runner.rtc_enabled)
            if self.runner.rtc_enabled:
                entry["rtc_delay"] = self.runner.last_rtc_delay
                entry["rtc_prefix"] = self.runner.last_rtc_prefix
        # What the link did with it. Without these the log records what was
        # *submitted* and says nothing about what reached the board -- a run whose
        # speed commands were all discarded for staleness reads identically to one
        # the arm executed perfectly. `dropped` is cumulative; diff it across ticks.
        s = self.bus.stats()
        entry["link"] = {
            "backlog": s["backlog"],
            "dropped": s["dropped"],
            "sent": s["sent"],
            "latency_ms": round(s["latency_mean"] * 1000, 1),
        }
        entry.update(self._extra_log_fields())
        self._log_file.write(json.dumps(entry) + "\n")

    def _toggle_record(self):
        if self._recording:
            self._close_results()
            self._log(f"Recording stopped -- {os.path.relpath(self._result_dir, REPO_DIR)}")
        else:
            self._open_results()
        self._set_policy_buttons_state()
        self._update_status_label()

    def _open_results(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Claimed rather than assumed: the name has one-second resolution, so two
        # sessions recording within the same second used to land in one directory
        # and the second's open(..., "w") truncated the first's actions.jsonl --
        # losing a run that was still being written, silently. exist_ok=False makes
        # the collision a failure to claim, and the suffix resolves it.
        base = os.path.join(REPO_DIR, "results", f"deploy_{stamp}")
        self._result_dir = base
        for attempt in range(2, 100):
            try:
                os.makedirs(self._result_dir, exist_ok=False)
                break
            except FileExistsError:
                self._result_dir = f"{base}_{attempt}"
        else:
            raise RuntimeError(f"could not claim a results directory beside {base}")
        h, w = self.image_size
        self._writer = cv2.VideoWriter(
            os.path.join(self._result_dir, "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps, (w, h),
        )
        self._log_file = open(os.path.join(self._result_dir, "actions.jsonl"), "w")
        with open(os.path.join(self._result_dir, "run.json"), "w") as f:
            json.dump({**self._run_metadata(), "started": stamp}, f, indent=2)
        self._recording = True
        self._log(f"Recording to {os.path.relpath(self._result_dir, REPO_DIR)}", level="connected")

    def _close_results(self):
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # -- plot panel --------------------------------------------------------

    def _build_plot_panel(self, parent):
        """Collapsible, matching the log panel. Four stacked subplots sharing x:
        one shared y-axis would be useless, since base_motor spans +/-900 while
        gripper spans 30-120 and the small channels would read as flat lines."""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.plot_frame = ttk.Frame(parent, padding=(16, 4, 16, 4))
        self.plot_frame.pack(side="bottom", fill="x")

        self._plot_expanded = False
        self.plot_header_var = tk.StringVar()
        header = ttk.Label(
            self.plot_frame, textvariable=self.plot_header_var,
            style="SectionHeading.TLabel", cursor="hand2",
        )
        header.pack(anchor="w", pady=(0, 4))
        header.bind("<Button-1>", lambda e: self._toggle_plot_panel())

        self.plot_body = ttk.Frame(self.plot_frame)
        self.figure = Figure(figsize=(9, 3.6), dpi=90, facecolor=PALETTE["bg"])
        self.axes = self.figure.subplots(ACTION_DIM, 1, sharex=True)
        self._lines = []
        # "base_motor_speed" wrapped onto three lines is noise; the joint is the
        # part worth reading, and the unit is on the y-ticks anyway.
        labels = ["base", "upper arm", "lower arm", "gripper"]
        for ax, name, (lo, hi) in zip(self.axes, labels, ACTION_COMMAND_LIMITS):
            ax.set_facecolor(PALETTE["bg"])
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center", labelpad=8)
            ax.tick_params(labelsize=6, colors=PALETTE["muted"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
            ax.grid(True, alpha=0.15)
            # A command holds until the next one supersedes it -- the arm is not
            # interpolating between them -- so steps are what actually happened
            # and a straight line between two samples would be an invention.
            self._lines.append(ax.plot([], [], lw=1.2, drawstyle="steps-post")[0])
        self.axes[-1].set_xlabel("seconds (0 = now)", fontsize=7, color=PALETTE["muted"])
        # Fixed: the window is a duration, and the trace scrolls through it.
        self.axes[-1].set_xlim(-PLOT_WINDOW_S, 0.0)
        self.figure.tight_layout(pad=0.6)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._update_plot_header()

    def _toggle_plot_panel(self):
        self._plot_expanded = not self._plot_expanded
        if self._plot_expanded:
            self.plot_body.pack(fill="both", expand=True)
        else:
            self.plot_body.pack_forget()
        self._update_plot_header()

    def _update_plot_header(self):
        arrow = "▼" if self._plot_expanded else "▶"
        self.plot_header_var.set(
            f"{arrow} Action plots (click to {'collapse' if self._plot_expanded else 'expand'})"
        )

    def _plot_tick(self):
        self.after(PLOT_REFRESH_MS, self._plot_tick)
        # Drawing a collapsed figure is pure waste on the thread that runs control.
        if not self._plot_expanded or not self._history:
            return
        now = time.time()
        # Trimming belongs here, not only on append: with the arm holding still
        # nothing is appended, and the window has to keep scrolling anyway. One
        # sample older than the left edge is kept, since it is the value the
        # trace carries into the window.
        while len(self._history) > 1 and self._history[1][0] < now - PLOT_WINDOW_S:
            self._history.popleft()
        x = np.fromiter((t - now for t, _ in self._history), dtype=np.float64,
                        count=len(self._history))
        data = np.stack([a for _, a in self._history])
        # Carry the last command to the right edge. It is still in force -- nothing
        # has superseded it -- so a trace that stopped short would read as if the
        # arm had stopped being commanded at all.
        x = np.append(x, 0.0)
        data = np.vstack([data, data[-1]])
        for j, line in enumerate(self._lines):
            line.set_data(x, data[:, j])
        self._draw_extra_plots(now)
        self.canvas.draw_idle()

    # -- camera feed -------------------------------------------------------

    def _update_camera_feed(self):
        self.after(FEED_REFRESH_MS, self._update_camera_feed)
        self._feed_ticks += 1
        if not self._policy_running and self._feed_ticks % max(1, 1000 // FEED_REFRESH_MS) == 0:
            self._update_status_label()
        latest = self.camera.get_latest() if self.camera is not None else None
        if latest is None:
            self._show_camera_placeholder()
            return
        self._note_camera_connected()
        frame = latest[0]
        h, w = frame.shape[:2]
        width = self._display_width
        small = cv2.resize(frame, (width, int(h * (width / w))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        label = "policy" if self._policy_running else "manual"
        cv2.putText(rgb, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def _show_camera_placeholder(self):
        canvas = camera_source.placeholder_frame(
            self.image_size, self.camera, self._camera_source,
            self.args.remote_camera_port, self._display_width,
        )
        photo = ImageTk.PhotoImage(Image.fromarray(canvas))
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    # -- teardown ----------------------------------------------------------

    def _shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self._policy_running:
                self._stop_policy(reason="shutting down")
        except Exception:
            pass
        # Stop the robot first, synchronously, before anything that can fail.
        try:
            self.bus.cancel_pending()
            self.bus.send_now("stop_all")
            self.bus.close()
        except Exception:
            pass
        self._close_results()
        if self.link is not None:
            link, self.link = self.link, None
            self._close_link_quietly(link, reset_board=True)
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        self.destroy()


def main():
    args = parse_args()
    if args.list_cameras:
        for source in camera_source.available_sources():
            print(" ", camera_source.source_label(source, args.remote_camera_port))
        return
    app = DeployApp(args)
    try:
        app.mainloop()
    finally:
        # Whatever happened, the arm must not be left driving.
        app._shutdown()


if __name__ == "__main__":
    main()
