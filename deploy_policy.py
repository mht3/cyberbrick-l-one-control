#!/usr/bin/env python3
"""Deployment GUI: run a trained checkpoint on the CyberBrick L-ONE arm.

Connects over serial or WiFi exactly like collect_data.py, loads a checkpoint
folder, and drives the arm from camera frames at a fixed rate while logging
everything to results/.

Three things shape the control loop:

Inference is slow relative to the control rate. pi0.5 takes ~280ms on an RTX 5090,
which is most of a control period, so it runs on a worker thread that keeps a queue
of upcoming actions filled. The Tk tick only pops and dispatches. This is the same
reason CommandBus exists -- anything that blocks the main thread stalls control.

The policy predicts a chunk. n_action_steps actions come back per inference and are
executed in order, so one inference covers n_action_steps/fps seconds of motion.

The policy emits continuous values, but the demonstrations only ever contained
three discrete levels per channel. "Snap" quantizes back onto them (default, and
closest to what the arm was actually driven at); "raw" clamps to the command limits
instead. Both are logged either way, so the choice is visible after the fact.
"""

import argparse
import datetime
import json
import os
import subprocess
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from virtual_gripper import (
    AP_FIXED_IP,
    BASE_MOTOR,
    GRIPPER_SERVO,
    LOWER_ARM_SERVO,
    PALETTE,
    STA_HOSTNAME,
    UPPER_ARM_SERVO,
    WIFI_PORT,
    CyberBrickLink,
    CyberBrickWifiLink,
    LinkDesynced,
    find_default_port,
    list_usb_ports,
)

from lone_data.camera import CameraStream
from lone_data.camera_stream import CameraStreamReceiver
from lone_data.command_bus import CommandBus
from lone_data.dispatch import clamp_to_limits, dispatch_action, snap_to_levels
from lone_data.features import ACTION_DIM, ACTION_NAMES, DEFAULT_IMAGE_SIZE, resize_keep_aspect

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNELS = (BASE_MOTOR, UPPER_ARM_SERVO, LOWER_ARM_SERVO, GRIPPER_SERVO)

WIFI_HEARTBEAT_MS = 200   # under wifi_bridge.py's COMMAND_DEADMAN_TIMEOUT
PLOT_REFRESH_MS = 100     # ~10Hz, deliberately slower than the control tick
FEED_REFRESH_MS = 50
PLOT_WINDOW = 150         # ticks kept on screen
MAX_HELD_TICKS = 3        # ticks to reuse the last action before stopping


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--remote-camera", action="store_true",
                   help="receive camera frames over the network (see stream_camera.py) instead of "
                        "opening a local camera")
    p.add_argument("--remote-camera-port", type=int, default=8267,
                   help="port to listen on for --remote-camera")
    p.add_argument("--fps", type=int, default=25, help="control rate")
    p.add_argument("--checkpoint", default=None, help="pretrained_model/ dir to preload")
    p.add_argument("--task", default="", help="task prompt (defaults to the checkpoint's)")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="override how many actions are executed per inference")
    p.add_argument("--device", default=None)
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1])
    p.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0])
    return p.parse_args()


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def checkpoint_task(checkpoint):
    """The task string the checkpoint was trained on, so deployment matches training.

    pi0.5 conditions on this text, so a differently worded prompt is a different
    conditioning. It is not stored in the checkpoint -- train_config.json only records
    which dataset was used -- so follow that pointer and read the task back out of the
    dataset's metadata. Returns "" if the dataset is not on this machine, in which case
    the operator types it.
    """
    path = os.path.join(checkpoint, "train_config.json")
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


class PolicyRunner:
    """Loads a checkpoint and serves action chunks from a worker thread.

    Inference is far too slow to run inline with a 25-30Hz control tick, so the
    thread refills a queue and the caller only ever pops.
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
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint
        )
        self.n_action_steps = int(n_action_steps or getattr(cfg, "n_action_steps", 1) or 1)

        self._pending = deque()
        self._lock = threading.Lock()
        self._frame = None
        self._busy = False
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
        """Hand the newest frame to the worker. Cheap; called every tick."""
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
                observation = self._observation(frame)
                batch = {
                    k: (v.unsqueeze(0) if self.torch.is_tensor(v) else [v])
                    for k, v in observation.items()
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


class DeployApp(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("CyberBrick L-ONE Policy Deployment")
        self.configure(background=PALETTE["bg"])
        self.minsize(1100, 760)

        self.link = None
        self.runner = None
        self._running = False
        self._shutting_down = False
        self._held = None
        self._held_ticks = 0
        self._underrun_reported = False
        self._tick_index = 0
        self._writer = None
        self._log_file = None
        self._result_dir = None
        self._period = 1.0 / args.fps
        self._next_deadline = None
        self._history = deque(maxlen=PLOT_WINDOW)
        self.image_size = (args.image_height, args.image_width)

        self.bus = CommandBus(
            on_error=lambda m: self.after(0, self._log, f"Command failed: {m}", "error"),
            on_link_dead=lambda m: self.after(0, self._on_link_dead, m),
            fatal_errors=(LinkDesynced,),
        )
        self.bus.start()

        if args.remote_camera:
            self.camera = CameraStreamReceiver(port=args.remote_camera_port)
        else:
            self.camera = CameraStream(args.camera_index, args.width, args.height, args.camera_fps)
        self.camera.start()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._shutdown)
        self.bind("<space>", lambda e: self._stop_all())
        self.bind("q", lambda e: self._shutdown())

        if args.checkpoint:
            self._set_checkpoint(args.checkpoint)

        self._wifi_heartbeat_tick()
        self._plot_tick()

    # -- UI -----------------------------------------------------------------

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(0, 10))
        self.mode_var = tk.StringVar(value="")
        ttk.Radiobutton(bar, text="Serial", variable=self.mode_var, value="serial",
                        command=self._on_mode).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(bar, text="WiFi", variable=self.mode_var, value="wifi",
                        command=self._on_mode).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.port_var = tk.StringVar(value=find_default_port())
        self.port_combo = ttk.Combobox(bar, textvariable=self.port_var, width=22,
                                       values=list_usb_ports(),
                                       postcommand=lambda: self.port_combo.config(values=list_usb_ports()))
        self.port_combo.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.host_var = tk.StringVar(value=STA_HOSTNAME)
        self.host_entry = ttk.Entry(bar, textvariable=self.host_var, width=26)
        self.host_entry.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(6, 0))
        self.ap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="AP Mode", variable=self.ap_var,
                        command=self._on_ap_toggle).grid(row=1, column=3, padx=(10, 0), pady=(6, 0))
        ttk.Button(bar, text="Connect", command=self._connect).grid(row=1, column=4, padx=(10, 0), pady=(6, 0))
        ttk.Button(bar, text="STOP ALL (space)", command=self._stop_all).grid(row=0, column=5, sticky="e", padx=(10, 0))
        ttk.Button(bar, text="Quit (Q)", command=self._shutdown).grid(row=0, column=4, sticky="e", padx=(10, 0))
        self._update_transport()

        setup = ttk.LabelFrame(root, text="Policy", padding=10)
        setup.pack(fill="x", pady=(0, 10))
        self.ck_var = tk.StringVar(value=self.args.checkpoint or "")
        ttk.Entry(setup, textvariable=self.ck_var, width=70).grid(row=0, column=0, sticky="we")
        ttk.Button(setup, text="Browse...", command=self._browse).grid(row=0, column=1, padx=(8, 0))
        ttk.Label(setup, text="Task").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.task_var = tk.StringVar(value=self.args.task)
        ttk.Entry(setup, textvariable=self.task_var, width=70).grid(row=2, column=0, sticky="we")
        self.mode_action = tk.StringVar(value="snap")
        modes = ttk.Frame(setup)
        modes.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Radiobutton(modes, text="Snap to demonstrated levels", variable=self.mode_action,
                        value="snap").pack(side="left")
        ttk.Radiobutton(modes, text="Raw (clamped)", variable=self.mode_action,
                        value="raw").pack(side="left", padx=(12, 0))
        self.start_btn = ttk.Button(setup, text="Start", command=self._start)
        self.start_btn.grid(row=3, column=1, sticky="e", pady=(8, 0))
        self.stop_btn = ttk.Button(setup, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=3, column=2, sticky="e", padx=(8, 0), pady=(8, 0))
        setup.columnconfigure(0, weight=1)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True)
        self.feed = ttk.Label(body)
        self.feed.pack(side="left", padx=(0, 10))
        self._build_plot(body)

        self.status_var = tk.StringVar(value="Idle.")
        ttk.Label(root, textvariable=self.status_var).pack(fill="x", pady=(8, 0))
        self.log = tk.Text(root, height=8, background=PALETTE["log_bg"], foreground=PALETTE["log_fg"])
        self.log.pack(fill="both", expand=False, pady=(8, 0))

        self._feed_tick()

    def _build_plot(self, parent):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(6, 4), dpi=90)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title("Dispatched action")
        self.axes.set_xlabel("tick")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _log(self, message, level="info"):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{stamp}] {message}\n")
        self.log.see("end")

    # -- connection ----------------------------------------------------------

    def _on_mode(self):
        self._update_transport()
        if self.mode_var.get() == "serial":
            self._connect()

    def _on_ap_toggle(self):
        self.host_var.set(AP_FIXED_IP if self.ap_var.get() else STA_HOSTNAME)

    def _update_transport(self):
        wifi = self.mode_var.get() == "wifi"
        self.host_entry.configure(state="normal" if wifi else "disabled")
        self.port_combo.configure(state="disabled" if wifi else "normal")

    def _connect(self):
        mode = self.mode_var.get()
        if not mode:
            self._log("Choose Serial or WiFi first.", "warn")
            return
        self._drop_link()
        try:
            if mode == "serial":
                port = self.port_var.get()
                if not port:
                    self._log("Select the board's USB port.", "warn")
                    return
                self.link = CyberBrickLink(port)
                self._log(f"Connected via Serial ({port})")
            else:
                host = self.host_var.get().strip()
                if not host:
                    self._log("Enter the board's IP address or hostname.", "warn")
                    return
                self.link = CyberBrickWifiLink(host, WIFI_PORT)
                self._log(f"Connected via WiFi ({host}:{WIFI_PORT})")
            self.bus.set_link(self.link)
        except Exception as e:
            self._log(f"Connection failed: {e}", "error")
            self.link = None

    def _drop_link(self):
        if self.link is None:
            return
        link, self.link = self.link, None
        self.bus.set_link(None)
        try:
            link.close()
        except Exception:
            pass

    def _on_link_dead(self, message):
        self._log(f"Link lost: {message}", "error")
        if self._running:
            self._stop(reason="link lost")
        self._drop_link()

    def _wifi_heartbeat_tick(self):
        self.after(WIFI_HEARTBEAT_MS, self._wifi_heartbeat_tick)
        # Held speeds must be re-sent or wifi_bridge.py's deadman fires mid-motion.
        if self._running and isinstance(self.link, CyberBrickWifiLink) and self._held is not None:
            dispatch_action(self.bus, self._held, CHANNELS)

    # -- policy --------------------------------------------------------------

    def _browse(self):
        path = filedialog.askdirectory(title="Select a checkpoint's pretrained_model folder")
        if path:
            self._set_checkpoint(path)

    def _set_checkpoint(self, path):
        self.ck_var.set(path)
        if not self.task_var.get():
            task = checkpoint_task(path)
            if task:
                self.task_var.set(task)
                self._log(f"Task from checkpoint: {task}")

    def _validate_checkpoint(self, path):
        if not path or not os.path.isdir(path):
            return "Select a checkpoint folder."
        for name in ("config.json", "model.safetensors"):
            if not os.path.exists(os.path.join(path, name)):
                return f"{path} has no {name} -- point at a checkpoint's pretrained_model/ folder."
        return None

    def _start(self):
        if self._running:
            return
        problem = self._validate_checkpoint(self.ck_var.get())
        if problem:
            self._log(problem, "warn")
            return
        if self.link is None:
            self._log("Connect to the board before starting.", "warn")
            return
        if not self.task_var.get().strip():
            self._log("Enter the task prompt the policy was trained with.", "warn")
            return

        self._log("Loading checkpoint...")
        self.update_idletasks()
        try:
            self.runner = PolicyRunner(
                self.ck_var.get(), self.task_var.get().strip(),
                device=self.args.device, n_action_steps=self.args.n_action_steps,
            )
        except Exception as e:
            self._log(f"Could not load checkpoint: {e}", "error")
            self.runner = None
            return

        self._open_results()
        self.runner.start()
        self._running = True
        self._tick_index = 0
        self._held = None
        self._held_ticks = 0
        self._underrun_reported = False
        self._history.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._log(f"Running {self.runner.policy_type} on {self.runner.device}, "
                  f"{self.runner.n_action_steps} actions per inference")
        self._next_deadline = time.monotonic()
        self._control_tick()

    def _stop(self, reason="stopped"):
        if not self._running:
            return
        self._running = False
        try:
            self.bus.cancel_pending()
            self.bus.send_now("stop_all")
        finally:
            if self.runner is not None:
                self.runner.stop()
            self._close_results()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._log(f"Deployment {reason}; arm stopped.")

    def _stop_all(self):
        self.bus.cancel_pending()
        self.bus.send_now("stop_all")
        if self._running:
            self._stop(reason="STOP ALL pressed")
        else:
            self._log("Stopped all joints.")

    # -- results -------------------------------------------------------------

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
                "checkpoint": self.ck_var.get(),
                "policy_type": self.runner.policy_type,
                "task": self.runner.task,
                "fps": self.args.fps,
                "n_action_steps": self.runner.n_action_steps,
                "action_mode": self.mode_action.get(),
                "device": self.runner.device,
                "image_size": list(self.image_size),
                "git_sha": git_sha(),
                "started": stamp,
            }, f, indent=2)
        self._log(f"Recording to {os.path.relpath(self._result_dir, REPO_DIR)}")

    def _close_results(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # -- control loop --------------------------------------------------------

    def _control_tick(self):
        if not self._running:
            return
        self._next_deadline += self._period
        delay = max(0, int((self._next_deadline - time.monotonic()) * 1000))
        self.after(delay, self._control_tick)

        if self.runner is not None and self.runner.error is not None:
            self._log(f"Inference failed: {self.runner.error}", "error")
            self._stop(reason="inference error")
            return

        latest = self.camera.get_latest()
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
                    # Once per episode -- a sustained underrun fires every tick and the
                    # repeats would bury everything else in the log.
                    self._log("Inference underrun -- stopping the arm", "warn")
                    self._underrun_reported = True
                self.bus.cancel_pending()
                self.bus.send_now("stop_all")
                self._held = None
                return
            action = self._held
        else:
            self._held_ticks = 0
            self._underrun_reported = False
            action = snap_to_levels(raw) if self.mode_action.get() == "snap" else clamp_to_limits(raw)
            self._held = action

        dispatch_action(self.bus, action, CHANNELS)
        self._history.append(action.copy())
        self._write_tick(rgb, raw, action, underrun)
        self._tick_index += 1

    def _write_tick(self, rgb, raw, action, underrun):
        if self._writer is not None:
            self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self._log_file is None:
            return
        entry = {
            "tick": self._tick_index,
            "t": round(time.time(), 4),
            "raw": None if raw is None else [round(float(v), 3) for v in raw],
            "dispatched": [round(float(v), 3) for v in action],
            "underrun": bool(underrun),
            "queued": self.runner.queued,
            "inference_s": (round(self.runner.last_latency, 4)
                            if self.runner.last_latency else None),
        }
        self._log_file.write(json.dumps(entry) + "\n")

    # -- periodic UI ---------------------------------------------------------

    def _feed_tick(self):
        self.after(FEED_REFRESH_MS, self._feed_tick)
        latest = self.camera.get_latest()
        if latest is None:
            return
        rgb = resize_keep_aspect(latest[0], self.image_size)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize((480, 270)))
        self.feed.configure(image=photo)
        self.feed.image = photo

    def _plot_tick(self):
        self.after(PLOT_REFRESH_MS, self._plot_tick)
        if not self._history:
            return
        data = np.stack(self._history)
        self.axes.clear()
        self.axes.set_title("Dispatched action")
        self.axes.set_xlabel("tick")
        for j, name in enumerate(ACTION_NAMES):
            self.axes.plot(data[:, j], label=name)
        self.axes.legend(loc="upper left", fontsize=7)
        self.canvas.draw_idle()
        if self.runner is not None:
            status = f"tick {self._tick_index}  queued {self.runner.queued}"
            latency = self.runner.last_latency
            if latency:
                # One inference has to cover n_action_steps of motion; over budget
                # means underruns, which is why this is on screen rather than inferred.
                budget = self.runner.n_action_steps * self._period
                status += f"  inference {latency * 1000:.0f}ms / {budget * 1000:.0f}ms budget"
            self.status_var.set(status)

    # -- teardown ------------------------------------------------------------

    def _shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        try:
            if self._running:
                self._stop(reason="shutting down")
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
        self._drop_link()
        try:
            self.camera.stop()
        except Exception:
            pass
        self.destroy()


def main():
    args = parse_args()
    app = DeployApp(args)  # if this raises, no link was opened and nothing is driving
    try:
        app.mainloop()
    finally:
        # Whatever happened, the arm must not be left driving.
        app._shutdown()


if __name__ == "__main__":
    main()
