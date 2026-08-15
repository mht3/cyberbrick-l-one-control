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
"""

import argparse
import datetime
import json
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
PLOT_WINDOW = 200       # ticks kept on screen
MAX_HELD_TICKS = 3      # ticks to reuse the last action before stopping


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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
    return p.parse_args()


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
    path = os.path.join(checkpoint or "", "train_config.json")
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            dataset = json.load(f).get("dataset", {})
    except (OSError, json.JSONDecodeError):
        return ""
    root, repo_id = dataset.get("root"), dataset.get("repo_id")
    if not root or not repo_id:
        return ""
    if not os.path.isabs(root):
        root = os.path.join(REPO_DIR, root)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        tasks = list(LeRobotDatasetMetadata(repo_id, root=root).tasks.index)
    except Exception:
        return ""
    # Several tasks means the run was multi-task and no single prompt is right.
    return tasks[0] if len(tasks) == 1 else ""


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
    """

    def __init__(self, checkpoint, task, device=None, n_action_steps=None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.checkpoint = checkpoint
        self.task = task

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if device:
            cfg.device = device
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_type = cfg.type
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint)
        self.policy.eval().to(self.device)
        self.pre, self.post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
        self.n_action_steps = int(n_action_steps or getattr(cfg, "n_action_steps", 1) or 1)

        self._pending = deque()
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self.last_latency = None
        self.error = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def submit_frame(self, rgb):
        with self._lock:
            self._frame = rgb

    def pop(self):
        with self._lock:
            return self._pending.popleft() if self._pending else None

    @property
    def queued(self):
        with self._lock:
            return len(self._pending)

    def _observation(self, rgb):
        # Must match what the dataset delivered at training time: CHW float in [0,1].
        image = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return {
            "observation.images.front": image,
            "observation.state": self.torch.zeros(ACTION_DIM),
            "task": self.task,
        }

    def _run(self):
        while self._running:
            with self._lock:
                need = len(self._pending) < self.n_action_steps
                frame = self._frame
            if not need or frame is None:
                time.sleep(0.005)
                continue
            try:
                started = time.perf_counter()
                batch = {
                    k: (v.unsqueeze(0) if self.torch.is_tensor(v) else [v])
                    for k, v in self._observation(frame).items()
                }
                with self.torch.no_grad():
                    chunk = self.policy.predict_action_chunk(self.pre(batch))
                actions = self.post(chunk)[0].float().cpu().numpy()
                latency = time.perf_counter() - started
                with self._lock:
                    for action in actions[: self.n_action_steps]:
                        self._pending.append(np.asarray(action, dtype=np.float32))
                    self.last_latency = latency
            except Exception as e:  # surfaced on the Tk thread by the caller
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

        # -- action state --------------------------------------------------
        # _current_action is the single record of what was last commanded, whoever
        # commanded it. The plots and the recorder read only this, so they behave
        # identically in manual and policy mode and cannot disagree about what the
        # arm was told to do.
        self._current_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._history = deque(maxlen=PLOT_WINDOW)
        self._period = 1.0 / args.fps
        self._feed_ticks = 0

        # -- policy state --------------------------------------------------
        self.runner = None
        self._policy_running = False
        self._next_deadline = None
        self._held = None
        self._held_ticks = 0
        self._underrun_reported = False
        self._tick_index = 0
        self._last_raw = None

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

        right = ttk.Frame(main_row, padding=(8, 16, 16, 16))
        right.pack(side="left", fill="y")

        self._build_connection_bar(right)
        self.joint_controls_frame = self._build_controls(right)
        self._build_policy_controls(right)

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

        problem = validate_checkpoint(args.checkpoint)
        if problem:
            self._log(problem, level="warn")
        else:
            self._log(f"Checkpoint: {args.checkpoint}")

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
            self._note_action(raw=None, dispatched=self._current_action, underrun=False)

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
            and self.link is not None
            and validate_checkpoint(self.args.checkpoint) is None
        )
        self.start_policy_btn.state(["!disabled"] if can_start else ["disabled"])
        self.stop_policy_btn.state(["!disabled"] if self._policy_running else ["disabled"])
        for child in self.joint_controls_frame.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Button):
                    widget.state(["disabled"] if self._policy_running else ["!disabled"])
        self.record_btn.configure(text="Stop Record" if self._recording else "Record")
        self.app_mode_status_var.set(self._mode_status_text())

    def _update_status_label(self):
        if self._policy_running and self.runner is not None:
            latency = self.runner.last_latency
            budget = self.runner.n_action_steps * self._period
            detail = f"queued {self.runner.queued}"
            if latency:
                detail += f"  ·  inference {latency * 1000:.0f}ms / {budget * 1000:.0f}ms budget"
            self.policy_status_var.set(f"Policy running -- tick {self._tick_index}")
            self.policy_detail_var.set(detail)
        else:
            self.policy_status_var.set(
                "Recording (manual control)." if self._recording else "Manual control."
            )
            self.policy_detail_var.set(
                f"camera {self._camera_fps():.1f} fps  ·  {self._link_readout()}"
            )

    # -- policy lifecycle --------------------------------------------------

    def _start_policy(self):
        if self._policy_running:
            return
        problem = validate_checkpoint(self.args.checkpoint)
        if problem:
            self._log(problem, level="warn")
            return
        if self.link is None:
            self._log("Connect to the board before starting the policy.", level="warn")
            return
        if self.camera is None or self.camera.get_latest() is None:
            # Starting blind would drive the arm from whatever frame arrives first.
            self._log("No camera frame yet -- wait for the feed before starting.", level="warn")
            return
        task = self.task_var.get().strip()
        if not task:
            self._log("Enter the task prompt the policy was trained with.", level="warn")
            self.task_entry.focus_set()
            return

        self._log("Loading checkpoint...")
        self.update_idletasks()
        try:
            self.runner = PolicyRunner(
                self.args.checkpoint, task,
                device=self.args.device, n_action_steps=self.args.n_action_steps,
            )
        except Exception as e:
            self._log(f"Could not load checkpoint: {e}", level="error")
            self.runner = None
            return

        self.runner.start()
        self._policy_running = True
        self._held = None
        self._held_ticks = 0
        self._underrun_reported = False
        self._tick_index = 0
        self._set_policy_buttons_state()
        self._log(f"Policy running: {self.runner.policy_type} on {self.runner.device}, "
                  f"{self.runner.n_action_steps} actions per inference", level="connected")
        self._next_deadline = time.monotonic()
        self._control_tick()

    def _stop_policy(self, reason="stopped"):
        if not self._policy_running:
            return
        self._policy_running = False
        try:
            self.bus.cancel_pending()
            self.bus.send_now("stop_all")
        finally:
            if self.runner is not None:
                self.runner.stop()
                self.runner = None
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

        latest = self.camera.get_latest() if self.camera is not None else None
        if latest is None:
            return
        rgb = resize_keep_aspect(latest[0], self.image_size)
        self.runner.submit_frame(rgb)

        raw = self.runner.pop()
        underrun = raw is None
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
                return
            action = self._held
        else:
            self._held_ticks = 0
            self._underrun_reported = False
            action = snap_to_levels(raw) if self._snap_var.get() else clamp_to_limits(raw)
            self._held = action

        dispatch_action(self.bus, action, CHANNELS)
        self._current_action[:] = action
        self._note_action(raw=raw, dispatched=action, underrun=underrun, frame=rgb)
        self._tick_index += 1
        if self._tick_index % max(1, self.args.fps // 5) == 0:
            self._update_status_label()

    # -- action history, plots, recording ----------------------------------

    def _note_action(self, raw, dispatched, underrun, frame=None):
        """One place where an action becomes history, a plot point and a log line."""
        self._history.append(np.asarray(dispatched, dtype=np.float32).copy())
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
        if self.runner is not None:
            entry["queued"] = self.runner.queued
            entry["inference_s"] = (round(self.runner.last_latency, 4)
                                    if self.runner.last_latency else None)
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
        self._result_dir = os.path.join(REPO_DIR, "results", f"deploy_{stamp}")
        os.makedirs(self._result_dir, exist_ok=True)
        h, w = self.image_size
        self._writer = cv2.VideoWriter(
            os.path.join(self._result_dir, "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps, (w, h),
        )
        self._log_file = open(os.path.join(self._result_dir, "actions.jsonl"), "w")
        with open(os.path.join(self._result_dir, "run.json"), "w") as f:
            json.dump({
                "checkpoint": self.args.checkpoint,
                "task": self.task_var.get().strip(),
                "fps": self.args.fps,
                "action_mode": "snap" if self._snap_var.get() else "raw",
                "policy_type": self.runner.policy_type if self.runner else None,
                "device": self.runner.device if self.runner else None,
                "image_size": list(self.image_size),
                "git_sha": git_sha(),
                "started": stamp,
            }, f, indent=2)
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
            self._lines.append(ax.plot([], [], lw=1.2)[0])
        self.axes[-1].set_xlabel("tick", fontsize=7, color=PALETTE["muted"])
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
        data = np.stack(self._history)
        x = np.arange(len(data))
        for j, line in enumerate(self._lines):
            line.set_data(x, data[:, j])
        self.axes[-1].set_xlim(0, max(PLOT_WINDOW, len(data)))
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
        scale = DISPLAY_MAX_WIDTH / w
        small = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
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
            self.args.remote_camera_port, DISPLAY_MAX_WIDTH,
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
