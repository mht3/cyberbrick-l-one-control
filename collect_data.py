#!/usr/bin/env python3
"""Data-collection GUI for the CyberBrick L-ONE arm.

Teleoperates the robot exactly like virtual_gripper.py (same link classes, same
joint keybindings) while recording demonstrations straight into a standard
LeRobotDataset. See lone_data/features.py for the schema.

Robot commands go through a CommandBus worker thread rather than being sent
from the Tk main thread, so a slow board can't stall frame capture -- that is
what held the previous recorder to ~14.8 Hz.
"""

import argparse
import datetime
import json
import os
import queue
import signal
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from virtual_gripper import (
    AP_FIXED_IP,
    AP_PASSWORD,
    AP_SSID,
    BASE_MOTOR,
    GRIPPER_CLOSED_ANGLE,
    GRIPPER_OPEN_ANGLE,
    GRIPPER_SERVO,
    JOINT_SPEED,
    KEY_JOINTS,
    LOWER_ARM_SERVO,
    MOTOR_SPEED,
    PALETTE,
    STA_HOSTNAME,
    UPPER_ARM_SERVO,
    WIFI_PORT,
    CyberBrickLink,
    CyberBrickWifiLink,
    JointControl,
    LinkDesynced,
    find_default_port,
    list_usb_ports,
    pick_default_port,
)

from lone_data import camera_source
from lone_data.robot_gui import RobotAppBase, WIFI_HEARTBEAT_MS
from lone_data.command_bus import CommandBus
from lone_data.features import (
    ACTION_DIM,
    ACTION_NAMES,
    CAMERA_KEY,
    DEFAULT_IMAGE_SIZE,
    resize_keep_aspect,
)
from lone_data.lerobot_recorder import LoneRecorder, has_saved_episodes
from lone_data.playback import EpisodeVideo
from lone_data.task_edit import set_episode_task

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

DISPLAY_MAX_WIDTH = 640
FEED_REFRESH_MS = 50  # ~20Hz preview; deliberately below the record rate so the
                      # preview never competes with the record tick for the main thread


def list_cameras(max_index=6):
    print(f"Probing camera indices 0..{max_index - 1}:")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index {i}: opened, reports {w}x{h}")
        else:
            print(f"  index {i}: not available")
        cap.release()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index (see --list-cameras)")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    p.add_argument("--width", type=int, default=1280, help="requested camera capture width")
    p.add_argument("--height", type=int, default=720, help="requested camera capture height")
    p.add_argument("--camera-fps", type=int, default=30, help="requested camera capture fps")
    p.add_argument("--remote-camera", action="store_true",
                   help="receive camera frames over the network (see stream_camera.py) instead of "
                        "opening a local camera")
    p.add_argument("--remote-camera-port", type=int, default=8267,
                   help="port to listen on for --remote-camera")
    p.add_argument("--fps", type=int, default=25, help="dataset recording rate")
    p.add_argument("--repo-id", default="lone/l_one", help="LeRobot dataset repo id")
    p.add_argument("--root", default=None, help="dataset directory (default: data/lerobot/<repo-id>)")
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1],
                   help="frame width stored in the dataset (must match the camera's aspect ratio)")
    p.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0],
                   help="frame height stored in the dataset (must match the camera's aspect ratio)")
    p.add_argument("--task", default="", help="default task/instruction string")
    return p.parse_args()


class CollectDataApp(RobotAppBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.title("CyberBrick L-ONE Data Collector")
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
        self._link_up_mono = None      # when the current link connected
        self._failed_reconnects = 0    # consecutive auto-reconnects that died young

        # Robot I/O runs here, off the Tk main thread (see module docstring).
        # Both callbacks fire on the bus worker thread, so they hop back to the
        # main thread via after() before touching any widget.
        self.bus = CommandBus(
            on_error=lambda m: self.after(0, self._log, f"Command failed: {m}", "error"),
            on_link_dead=lambda m: self.after(0, self._on_link_dead, m),
            fatal_errors=(LinkDesynced,),
        )
        self.bus.start()

        # -- camera -------------------------------------------------------
        # Selected from the toolbar and swappable at runtime, so a machine with no
        # local camera still opens -- pick the remote source and carry on. A failure
        # here is logged rather than fatal for the same reason.
        self.camera = None
        # Remote is only offered when the app was launched to receive a stream:
        # on a machine nothing streams to, it is a dead menu entry.
        self._camera_source = camera_source.pick_initial_source(
            camera_source.REMOTE_SOURCE if args.remote_camera else args.camera_index,
            include_remote=args.remote_camera,
        )
        self._camera_connected = False

        # -- dataset --------------------------------------------------------
        # Not created here. LeRobot writes meta/info.json the moment a dataset
        # exists, so creating one up front leaves an empty dataset behind every
        # time the app is opened and closed without recording. Deferred to the
        # first episode -- see _ensure_recorder().
        self.image_size = (args.image_height, args.image_width)
        self._aspect_error = None  # set by _check_storage_aspect once a camera is up
        root = args.root or os.path.join(REPO_DIR, "data", "lerobot", *args.repo_id.split("/"))
        self.dataset_root = os.path.abspath(root)
        self.recorder = None

        self.log_dir = os.path.join(REPO_DIR, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_log_path = os.path.join(
            self.log_dir, f"collection_{datetime.datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )

        # -- recording state ------------------------------------------------
        self._current_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._recording = False
        self._period = 1.0 / args.fps
        self._next_deadline = None
        self._episode_step_count = 0
        self._episode_missed = 0
        self._episode_repeats = 0
        self._episode_start_mono = None
        self._episode_start_wall = None
        self._episode_task = ""
        self._last_frame_seq = None
        self._feed_ticks = 0

        # -- review state --------------------------------------------------
        self._review_mode = False
        self._review_playing = False
        self._review_episode_idx = None
        self._review_video = None
        self._review_len = 0
        self._review_frame_idx = 0
        self._review_frame_interval_ms = max(1, int(1000 * self._period))
        self._review_slider_syncing = False

        # -- UI ---------------------------------------------------------
        # Packed before main_row so it spans the full window width at the bottom (pack order matters here).
        self._build_log_panel(self)

        main_row = ttk.Frame(self)
        main_row.pack(fill="both", expand=True)

        camera_frame = ttk.Frame(main_row, padding=(16, 16, 8, 16))
        camera_frame.pack(side="left", fill="both", expand=True)

        # video_group keeps the image and scrub bar grouped as one unit so
        # the scrub bar stays flush under the video instead of drifting away.
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

        # Review-only; shown/hidden by _apply_mode_visibility.
        self.scrub_row = ttk.Frame(video_group)
        self.review_play_btn = ttk.Button(
            self.scrub_row, text="▶ Play (P)", command=self._toggle_review_play, width=12
        )
        self.review_play_btn.pack(side="left")
        self.review_slider_var = tk.DoubleVar(value=0)
        self.review_slider = ttk.Scale(
            self.scrub_row, from_=0, to=0, orient="horizontal", length=420,
            variable=self.review_slider_var, command=self._on_review_slider_drag,
        )
        self.review_slider.pack(side="left", padx=(8, 8))
        self.review_time_var = tk.StringVar(value="0.0s / 0.0s")
        ttk.Label(self.scrub_row, textvariable=self.review_time_var, style="KeyHint.TLabel", width=12).pack(
            side="left"
        )

        right = ttk.Frame(main_row, padding=(8, 16, 16, 16))
        right.pack(side="left", fill="y")

        self._build_connection_bar(right)
        self.joint_controls_frame = self._build_controls(right)
        self.episode_controls_frame = self._build_episode_controls(right)
        self._build_review_controls(right)

        self._setup_keybindings()
        self._setup_episode_keybindings()
        self.bind_all("<Button-1>", self._maybe_reclaim_focus, add="+")
        self.protocol("WM_DELETE_WINDOW", self._shutdown)
        signal.signal(signal.SIGINT, lambda signum, frame: self._shutdown())
        self._signal_pump()

        self.mode_var.set("wifi")
        self._on_mode_select()
        self._apply_mode_visibility()
        self._set_episode_buttons_state()
        self._update_status_label()
        self.focus_set()

        # "@ N fps" is the *recording* rate, not the camera's -- they are different
        # numbers and confusing them is easy. Name both so the mismatch is visible:
        # sampling a 30fps camera at 25 means ~1 frame in 5 is a duplicate of the last.
        self._log(
            f"Recording at {args.fps} fps into {self.dataset_root} "
            f"(camera requested at {args.camera_fps} fps)", level="info",
        )
        if args.camera_fps != args.fps:
            self._log(
                f"Recording rate {args.fps} != camera {args.camera_fps} fps -- frames the camera "
                f"has not refreshed are recorded again. Use --fps {args.camera_fps} to match.",
                level="warn",
            )

        # After the log panel exists, so a camera that will not open reports itself
        # in the GUI instead of taking the process down before there is a GUI.
        if self._open_camera(self._camera_source):
            self._check_storage_aspect()
        elif self._camera_source != camera_source.REMOTE_SOURCE:
            self._log("Pick another source from the Camera dropdown, or run "
                      "stream_camera.py elsewhere and choose Remote.", level="warn")

        self.after(0, self._update_camera_feed)
        self.after(WIFI_HEARTBEAT_MS, self._wifi_heartbeat_tick)

    # -- misc setup -----------------------------------------------------

    # -- camera source -------------------------------------------------------

    # -- RobotAppBase hooks ------------------------------------------------

    def _controls_enabled(self):
        """Joint controls are visibly disabled during review, so ignore input then."""
        return not self._review_mode

    def _controls_busy(self):
        return "Finish or discard the episode" if self._recording else None

    def _mode_status_text(self):
        return "Mode: Review" if self._review_mode else "Mode: Collect"

    def _on_link_state_changed(self):
        self._set_episode_buttons_state()

    def _on_link_lost(self):
        if self._recording:
            self._log("Link lost while recording -- discarding in-progress episode", level="warn")
            self._discard_episode()

    def _on_emergency_stop(self):
        if self._recording:
            self._log("STOP ALL during recording -- discarding in-progress episode", level="warn")
            self._discard_episode()

    def _after_camera_opened(self):
        self._check_storage_aspect()

    def _check_storage_aspect(self):
        """Frames are downscaled, never cropped or padded, so a stored size
        whose aspect ratio doesn't match the camera's silently stretches every
        frame in the dataset. Refuse up front instead.

        Once the camera is switchable this can no longer exit the process -- the
        operator is mid-session and can simply pick another source -- so a mismatch
        is reported and recording is blocked by _start_episode instead."""
        self._aspect_error = None
        if self.camera is None:
            return
        store_h, store_w = self.image_size
        cam_w, cam_h = self.camera.actual_width, self.camera.actual_height
        if not cam_w or not cam_h:
            return
        if abs((store_w / store_h) - (cam_w / cam_h)) > 0.01:
            self._aspect_error = (
                f"--image-width/--image-height {store_w}x{store_h} "
                f"({store_w / store_h:.3f}:1) does not match the camera's "
                f"{cam_w}x{cam_h} ({cam_w / cam_h:.3f}:1). "
                "Frames are scaled, not cropped or padded, so this would distort every frame. "
                f"Use a size with the camera's aspect ratio (e.g. {cam_w // 2}x{cam_h // 2})."
            )
            self._log(self._aspect_error, level="error")

    # -- dataset (created lazily, on the first episode) ------------------

    @property
    def _n_episodes(self):
        return self.recorder.num_episodes if self.recorder is not None else 0

    @property
    def _n_frames(self):
        return self.recorder.num_frames if self.recorder is not None else 0

    def _ensure_recorder(self):
        """Opens (or creates) the dataset on first use. Returns False and logs
        if it can't, so a dataset problem never leaves a half-started episode."""
        if self.recorder is not None:
            return True
        try:
            os.makedirs(os.path.dirname(self.dataset_root), exist_ok=True)
            self.recorder = LoneRecorder(
                self.args.repo_id, self.dataset_root, fps=self.args.fps, image_size=self.image_size
            )
        except Exception as e:
            self._log(f"Could not open dataset: {e}", level="error")
            return False
        if self.recorder.resumed:
            self._log(
                f"Appending to existing dataset ({self.recorder.num_episodes} episode(s), "
                f"{self.recorder.num_frames} frames) at {self.dataset_root}",
                level="info",
            )
        else:
            self._log(f"Created dataset at {self.dataset_root} @ {self.args.fps} fps", level="connected")
        return True

    # -- connection bar (reimplemented so virtual_gripper.py stays untouched) --

    # -- joint controls (identical keybindings/behavior to virtual_gripper) --

    # -- episode controls (new keybindings on keys virtual_gripper doesn't use) --

    def _build_episode_controls(self, parent):
        frame = ttk.Frame(parent, padding=(0, 0, 0, 12))
        frame.pack(fill="x")
        ttk.Label(frame, text="Episode", style="SectionHeading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Task").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.task_var = tk.StringVar(value=self.args.task)
        self.task_entry = ttk.Entry(frame, textvariable=self.task_var, width=30)
        self.task_entry.grid(row=1, column=1, sticky="w", pady=(6, 0), padx=(6, 0))

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.start_btn = ttk.Button(btn_row, text="Start (Return)", style="Accent.TButton", command=self._start_episode)
        self.finish_btn = ttk.Button(btn_row, text="Finish (F)", command=self._finish_episode)
        self.discard_btn = ttk.Button(btn_row, text="Discard (⌫)", style="Danger.TButton", command=self._discard_episode)
        self.start_btn.pack(side="left")
        self.finish_btn.pack(side="left", padx=(6, 0))
        self.discard_btn.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frame, textvariable=self.status_var, style="KeyHint.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.rate_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.rate_var, style="KeyHint.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        return frame

    def _setup_episode_keybindings(self):
        def guarded(handler):
            def wrapped(event):
                if self._focused_widget_wants_text():
                    return
                handler()
            return wrapped

        self.bind("<Return>", guarded(self._start_episode))
        self.bind("<KeyPress-f>", guarded(self._finish_episode))
        self.bind("<KeyPress-F>", guarded(self._finish_episode))
        self.bind("<BackSpace>", guarded(self._discard_episode))
        self.bind("<KeyPress-q>", guarded(self._quit))
        self.bind("<KeyPress-Q>", guarded(self._quit))
        self.bind("<KeyPress-r>", guarded(self._enter_review_mode))
        self.bind("<KeyPress-R>", guarded(self._enter_review_mode))
        self.bind("<KeyPress-p>", guarded(self._toggle_review_play))
        self.bind("<KeyPress-P>", guarded(self._toggle_review_play))
        self.bind("<KeyPress-n>", guarded(self._next_review_episode))
        self.bind("<KeyPress-N>", guarded(self._next_review_episode))
        # add="+" layers review nav onto Left/Right without replacing the
        # joint binding; both are mode-guarded internally, so it's safe.
        self.bind("<KeyPress-Left>", guarded(self._previous_review_episode), add="+")
        self.bind("<KeyPress-Right>", guarded(self._next_review_episode), add="+")
        self.bind("<Escape>", guarded(self._exit_review_mode))

    def _set_episode_buttons_state(self):
        connected = self.link is not None
        can_start = connected and not self._recording and not self._review_mode
        self.start_btn.config(state="normal" if can_start else "disabled")
        self.finish_btn.config(state="normal" if self._recording else "disabled")
        self.discard_btn.config(state="normal" if self._recording else "disabled")

    def _update_status_label(self):
        if self._recording:
            elapsed = time.monotonic() - self._episode_start_mono
            self.status_var.set(
                f"Recording episode {self._n_episodes:06d} -- "
                f"{self._episode_step_count} steps, {elapsed:0.1f}s"
            )
            achieved = self._episode_step_count / elapsed if elapsed > 0 else 0.0
            self.rate_var.set(
                f"{achieved:.1f}/{self.args.fps} fps  ·  camera {self._camera_fps():.1f} fps  ·  "
                f"missed {self._episode_missed}  ·  stale frames {self._episode_repeats}  ·  "
                f"{self._link_readout()}"
            )
        else:
            self.status_var.set(
                f"Idle -- {self._n_episodes} episode(s), {self._n_frames} frames"
            )
            self.rate_var.set(
                f"camera {self._camera_fps():.1f} fps  ·  {self._link_readout()}"
            )

    # -- review mode: play back recorded episodes ------------------------
    # lerobot closes the parquet writer and reopens for reading on first
    # __getitem__, then starts a new file on the next save_episode, so reading
    # back mid-session is supported.

    def _build_review_controls(self, parent):
        # review_active_frame only shows while reviewing -- _apply_mode_visibility.
        self.review_section_frame = ttk.Frame(parent, padding=(0, 0, 0, 12))
        self.review_section_frame.pack(fill="x")
        frame = self.review_section_frame
        ttk.Label(frame, text="Review", style="SectionHeading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")

        self.review_status_var = tk.StringVar(value="Not reviewing")
        ttk.Label(frame, textvariable=self.review_status_var, style="KeyHint.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self.review_enter_row = ttk.Frame(frame)
        self.review_enter_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.review_enter_btn = ttk.Button(
            self.review_enter_row, text="Review Episodes (R)", command=self._enter_review_mode
        )
        self.review_enter_btn.pack(side="left")

        # Not gridded here -- only shown while reviewing (_apply_mode_visibility).
        self.review_active_frame = ttk.Frame(frame)

        nav_row = ttk.Frame(self.review_active_frame)
        nav_row.pack(fill="x")
        self.review_prev_btn = ttk.Button(nav_row, text="◀ Previous (←)", command=self._previous_review_episode)
        self.review_next_btn = ttk.Button(nav_row, text="Next ▶ (N / →)", command=self._next_review_episode)
        self.review_exit_btn = ttk.Button(nav_row, text="Exit (Esc)", command=self._exit_review_mode)
        self.review_prev_btn.pack(side="left")
        self.review_next_btn.pack(side="left", padx=(6, 0))
        self.review_exit_btn.pack(side="left", padx=(6, 0))

        # Editable, not a label: a task typed in a hurry or left over from the
        # previous episode is wrong supervision for every frame of this one, and
        # the moment you notice is while watching it back.
        task_row = ttk.Frame(self.review_active_frame)
        task_row.pack(fill="x", pady=(6, 0))
        ttk.Label(task_row, text="Task", style="KeyHint.TLabel").pack(side="left")
        self.review_task_var = tk.StringVar(value="")
        self.review_task_entry = ttk.Entry(task_row, textvariable=self.review_task_var, width=30)
        self.review_task_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.review_task_entry.bind("<Return>", lambda e: self._save_review_task())
        self.review_task_save_btn = ttk.Button(
            task_row, text="Save", width=6, command=self._save_review_task
        )
        self.review_task_save_btn.pack(side="left")

        self._set_review_buttons_state()

    def _set_review_buttons_state(self):
        self.review_enter_btn.config(state="disabled" if (self._review_mode or self._recording) else "normal")
        state = "normal" if self._review_mode else "disabled"
        self.review_prev_btn.config(state=state)
        self.review_next_btn.config(state=state)
        self.review_exit_btn.config(state=state)
        self.review_task_entry.config(state="normal" if self._review_mode else "disabled")
        self.review_task_save_btn.config(state=state)
        self.review_play_btn.config(state=state)
        self.review_slider.config(state=state)

    def _update_review_status_label(self):
        if self._review_mode and self._review_episode_idx is not None:
            n = self._n_episodes
            self.review_status_var.set(
                f"Episode {self._review_episode_idx:06d} / {n - 1:06d}  ({self._review_len} steps)"
            )
        else:
            self.review_status_var.set("Not reviewing")

    def _enter_review_mode(self):
        if self._recording:
            self._log("Finish or discard the current episode before reviewing.", level="warn")
            return
        if self._review_mode:
            return
        # The recorder is created lazily on the first episode, so on a fresh
        # launch it is None and _n_episodes reads 0 -- which used to report "no
        # episodes" over a dataset directory full of them. Open it here, but
        # only once there is something to open: LoneRecorder deletes a dataset
        # directory that holds metadata and no episodes, which must not be what
        # entering review does.
        if self.recorder is None:
            if not has_saved_episodes(self.dataset_root):
                self._log("No episodes recorded yet.", level="warn")
                return
            self._log(f"Opening {self.dataset_root} for review...", level="info")
            self.update_idletasks()
            if not self._ensure_recorder():
                return
        if self._n_episodes == 0:
            self._log("No episodes recorded yet.", level="warn")
            return
        self._review_mode = True
        self._apply_mode_visibility()
        self._load_review_episode(0)
        self._set_episode_buttons_state()
        self._set_review_buttons_state()
        self.after(0, self._review_tick)

    def _save_review_task(self):
        """Write the edited task for the episode being reviewed.

        The dataset has to be closed first: LeRobotDataset buffers episode metadata
        and holds its parquet writers open until finalize(), so an edit underneath a
        live recorder is simply overwritten by it. Reopening afterwards is what makes
        the change visible to review and to the next recording session alike.
        """
        if not self._review_mode or self._review_episode_idx is None:
            return
        new_task = self.review_task_var.get().strip()
        if not new_task:
            self._log("Task text cannot be empty.", level="warn")
            return
        episode_idx = self._review_episode_idx
        if self.recorder is not None and self.recorder.episodes[episode_idx]["task"] == new_task:
            return  # unchanged; nothing to rewrite

        self._close_review_video()
        try:
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
            changed = set_episode_task(self.dataset_root, episode_idx, new_task)
        except Exception as e:
            self._log(f"Could not save task: {e}", level="error")
            self._exit_review_mode()
            return

        self._log(f"Task saved -- {changed}" if changed else "Task unchanged.", level="connected")
        if not self._ensure_recorder():
            self._exit_review_mode()
            return
        self._load_review_episode(episode_idx)

    def _exit_review_mode(self):
        if not self._review_mode:
            return
        self._review_mode = False
        self._set_review_playing(False)
        self._review_episode_idx = None
        self._review_video = None
        self._review_len = 0
        self.review_task_var.set("")
        self.review_slider.config(to=0)
        self.review_slider_var.set(0)
        self.review_time_var.set("0.0s / 0.0s")
        self._apply_mode_visibility()
        self._set_episode_buttons_state()
        self._set_review_buttons_state()
        self._update_review_status_label()
        self._log("Exited review mode.", level="info")

    def _apply_mode_visibility(self):
        """Collect and Review are separate top-level modes; entering one hides the other's controls."""
        self.app_mode_status_var.set("Mode: Review" if self._review_mode else "Mode: Collect")
        if self._review_mode:
            self.joint_controls_frame.pack_forget()
            self.episode_controls_frame.pack_forget()
            self.review_enter_row.grid_forget()
            self.review_active_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
            self.scrub_row.pack(anchor="n", pady=(8, 0))
        else:
            self.review_active_frame.grid_forget()
            self.scrub_row.pack_forget()
            self.review_enter_row.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
            # before= restores their original position; a plain pack() would append after Review.
            self.joint_controls_frame.pack(fill="x", before=self.review_section_frame)
            self.episode_controls_frame.pack(fill="x", before=self.review_section_frame)

    def _load_review_episode(self, episode_idx):
        n = self._n_episodes
        if n == 0:
            self._exit_review_mode()
            return
        episode_idx = max(0, min(episode_idx, n - 1))
        self._close_review_video()
        try:
            ep = self.recorder.episodes[episode_idx]
            self._review_video = EpisodeVideo(
                ep["video_path"], ep["from_timestamp"], ep["length"], self.args.fps
            )
        except Exception as e:
            self._log(f"Could not open episode {episode_idx} video: {e}", level="error")
            self._exit_review_mode()
            return
        self._review_episode_idx = episode_idx
        self._review_len = ep["length"]
        self._review_frame_idx = 0
        self.review_task_var.set(ep["task"] or "")
        self._review_frame_interval_ms = max(1, int(1000 * self._period))
        self.review_slider.config(to=max(0, self._review_len - 1))
        self._update_review_status_label()
        self._log(f"Reviewing episode {episode_idx:06d}/{n - 1:06d} ({self._review_len} steps)", level="info")
        self._set_review_playing(True)
        self._render_review_frame()

    def _close_review_video(self):
        if self._review_video is not None:
            try:
                self._review_video.close()
            except Exception:
                pass
            self._review_video = None

    def _set_review_playing(self, playing):
        self._review_playing = playing
        self.review_play_btn.config(text="⏸ Pause (P)" if playing else "▶ Play (P)")

    def _toggle_review_play(self):
        if not self._review_mode:
            return
        # Replaying from the start after it ran off the end reads better
        # than "playing" a paused freeze-frame at the last position.
        if not self._review_playing and self._review_frame_idx >= self._review_len - 1:
            self._review_frame_idx = 0
        self._set_review_playing(not self._review_playing)

    def _render_review_frame(self):
        if self._review_len == 0 or self._review_video is None:
            return
        try:
            display = self._review_video.frame(self._review_frame_idx).copy()
        except Exception as e:
            self._log(f"Review read failed: {e}", level="error")
            self._set_review_playing(False)
            return

        lines = [f"REVIEW episode {self._review_episode_idx:06d}"]
        actions = self.recorder.episode_actions.get(self._review_episode_idx)
        if actions is not None and self._review_frame_idx < len(actions):
            action = actions[self._review_frame_idx]
            lines += [f"{name}: {action[i]:.1f}" for i, name in enumerate(ACTION_NAMES)]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (6, 16 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1, cv2.LINE_AA)

        h, w = display.shape[:2]
        scale = DISPLAY_MAX_WIDTH / w
        display = cv2.resize(display, (DISPLAY_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

        self._sync_review_slider()
        self._update_review_time_label()

    def _sync_review_slider(self):
        # Guard so _on_review_slider_drag doesn't mistake this for a user drag.
        self._review_slider_syncing = True
        try:
            self.review_slider_var.set(self._review_frame_idx)
        finally:
            self._review_slider_syncing = False

    def _update_review_time_label(self):
        cur = self._review_frame_idx * self._period
        total = max(0, self._review_len - 1) * self._period
        self.review_time_var.set(f"{cur:.1f}s / {total:.1f}s")

    def _on_review_slider_drag(self, value):
        if self._review_slider_syncing or not self._review_mode or self._review_len == 0:
            return
        idx = max(0, min(int(round(float(value))), self._review_len - 1))
        if idx == self._review_frame_idx:
            return
        self._review_frame_idx = idx
        self._render_review_frame()

    def _review_tick(self):
        if not self._review_mode:
            return
        try:
            if not self._review_playing or self._review_len == 0:
                self.after(self._review_frame_interval_ms, self._review_tick)
                return
            self._render_review_frame()
            if self._review_frame_idx >= self._review_len - 1:
                self._set_review_playing(False)  # reached the end -- pause on the last frame, don't loop
                self.after(self._review_frame_interval_ms, self._review_tick)
                return
            self._review_frame_idx += 1
            self.after(self._review_frame_interval_ms, self._review_tick)
        except Exception as e:
            self._log(f"Review playback tick failed: {e}", level="error")
            self.after(self._review_frame_interval_ms, self._review_tick)

    def _step_review_episode(self, delta):
        """Move `delta` episodes, wrapping at both ends.

        Reviewing is a loop -- you are comparing episodes against each other, not
        reading a list front to back -- so stopping dead at the last one just means
        clicking all the way back. Wrapping is a no-op with a single episode.
        """
        if not self._review_mode or self._review_episode_idx is None:
            return
        n = self._n_episodes
        if n == 0:
            return
        self._load_review_episode((self._review_episode_idx + delta) % n)

    def _next_review_episode(self):
        self._step_review_episode(1)

    def _previous_review_episode(self):
        self._step_review_episode(-1)

    # -- log panel --------------------------------------------------------

    # -- camera feed ------------------------------------------------------

    def _show_camera_placeholder(self):
        """Stand-in for the video pane when no frames are arriving."""
        canvas = camera_source.placeholder_frame(
            self.image_size, self.camera, self._camera_source,
            self.args.remote_camera_port, DISPLAY_MAX_WIDTH,
        )
        photo = ImageTk.PhotoImage(Image.fromarray(canvas))
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def _update_camera_feed(self):
        self.after(FEED_REFRESH_MS, self._update_camera_feed)
        if self._review_mode:
            return  # _review_tick owns camera_label while reviewing
        # The idle readout is otherwise only written at startup, when the camera
        # hasn't produced a frame yet and its rate still reads 0.
        self._feed_ticks += 1
        if not self._recording and self._feed_ticks % max(1, 1000 // FEED_REFRESH_MS) == 0:
            self._update_status_label()
        latest = self.camera.get_latest() if self.camera is not None else None
        if latest is None:
            self._show_camera_placeholder()
            return
        self._note_camera_connected()
        frame, _ts, _seq = latest
        # Downscale first, then annotate -- drawing on the full-res frame costs a
        # 2.7MB copy per refresh on the same thread the record tick runs on.
        h, w = frame.shape[:2]
        scale = DISPLAY_MAX_WIDTH / w
        display = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_NEAREST)
        if self._recording:
            label, color = f"REC ep{self._n_episodes:06d}  step {self._episode_step_count}", (0, 0, 255)
        else:
            label, color = "idle", (200, 200, 200)
        cv2.putText(display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo  # keep a reference, Tk won't otherwise

    # -- robot command dispatch (also updates self._current_action, which the recorder reads) --

    # -- episode recording --------------------------------------------------

    def _start_episode(self):
        if self._recording:
            self._log("Already recording an episode.", level="warn")
            return
        if self._review_mode:
            self._log("Exit review mode before starting a new episode.", level="warn")
            return
        if not self._require_link():
            return
        if self.camera is None:
            self._log("No camera selected -- choose a source from the Camera dropdown.", level="warn")
            return
        if self.camera.get_latest() is None:
            self._log("No camera frame available yet -- wait a moment and try again.", level="warn")
            return
        if self._aspect_error:
            # Was a SystemExit at startup; with a switchable camera it blocks
            # recording instead, so the operator can just pick a matching source.
            self._log(f"Refusing to record: {self._aspect_error}", level="error")
            return
        task = self.task_var.get().strip()
        if not task:
            self._log("Enter a task/instruction before recording -- every episode needs one.", level="warn")
            self.task_entry.focus_set()
            return

        cam_fps = self._camera_fps()
        if cam_fps and cam_fps < self.args.fps * 0.9:
            self._log(
                f"Camera is only sustaining {cam_fps:.1f} fps but the dataset declares "
                f"{self.args.fps} -- frames will be duplicated. Restart with --fps {int(cam_fps)}.",
                level="warn",
            )

        if not self._ensure_recorder():
            return
        try:
            self.recorder.start_episode()
        except Exception as e:
            self._log(f"Could not start episode: {e}", level="error")
            return

        self._recording = True
        self._episode_step_count = 0
        self._episode_missed = 0
        self._episode_repeats = 0
        self._last_frame_seq = None
        self._episode_start_mono = time.monotonic()
        self._episode_start_wall = datetime.datetime.now().astimezone().isoformat()
        self._episode_task = task
        self._next_deadline = time.monotonic()
        self._set_episode_buttons_state()
        self._set_review_buttons_state()
        self._update_status_label()
        self._log(f"Started episode {self._n_episodes:06d} -- task: {task!r}", level="connected")
        self.after(0, self._record_tick)

    def _record_tick(self):
        if not self._recording:
            return
        # Absolute-deadline scheduling: drift is corrected against the episode
        # start rather than accumulating one Tk timer rounding error per tick.
        now = time.monotonic()
        self._next_deadline += self._period
        if self._next_deadline <= now:
            behind = now - self._next_deadline
            self._episode_missed += int(behind / self._period) + 1
            self._next_deadline = now + self._period
        self.after(max(1, int((self._next_deadline - now) * 1000)), self._record_tick)

        try:
            latest = self.camera.get_latest() if self.camera is not None else None
            if latest is None:
                self._log("No camera frame available -- skipped a step", level="warn")
                return
            frame, _cam_ts, seq = latest
            if seq == self._last_frame_seq:
                # Camera hasn't produced a new frame within one period. Recording
                # it keeps the fixed-rate grid honest; the count is reported.
                self._episode_repeats += 1
            self._last_frame_seq = seq

            action = self._current_action.copy()
            self.recorder.add_frame(
                resize_keep_aspect(frame, self.image_size), action, self._episode_task
            )
            self._episode_step_count += 1
            # Refreshing Tk vars is pointless at 30Hz and this is the one thread
            # that must not fall behind, so throttle the readout to ~5Hz.
            if self._episode_step_count % max(1, self.args.fps // 5) == 0:
                self._update_status_label()
        except Exception as e:
            self._log(f"Recording tick failed: {e} -- discarding episode", level="error")
            self._discard_episode()

    def _finish_episode(self):
        if not self._recording:
            self._log("Not currently recording.", level="warn")
            return
        self._recording = False
        elapsed = time.monotonic() - self._episode_start_mono
        episode_index = self._n_episodes

        self._log("Encoding episode video...", level="info")
        self.update_idletasks()
        try:
            ep_len = self.recorder.finish_episode()
        except Exception as e:
            self._log(f"Saving episode failed: {e}", level="error")
            self._set_episode_buttons_state()
            self._set_review_buttons_state()
            self._update_status_label()
            return

        if ep_len == 0:
            self._log("Episode had 0 steps -- nothing saved", level="warn")
        else:
            achieved = ep_len / elapsed if elapsed > 0 else 0.0
            level = "warn" if achieved < self.args.fps * 0.9 else "connected"
            self._log(
                f"Finished episode {episode_index:06d} -- {ep_len} steps, {achieved:.1f} fps "
                f"(target {self.args.fps}), {self._episode_missed} missed, "
                f"{self._episode_repeats} stale frames",
                level=level,
            )
            self._write_session_log(episode_index, ep_len, elapsed, achieved)

        self._set_episode_buttons_state()
        self._set_review_buttons_state()
        self._update_status_label()

    def _discard_episode(self):
        if not self._recording:
            self._log("Not currently recording.", level="warn")
            return
        self._recording = False
        try:
            dropped = self.recorder.discard_episode()
        except Exception as e:
            self._log(f"Discard failed: {e}", level="error")
            dropped = 0
        self._log(f"Discarded in-progress episode ({dropped} steps)", level="warn")
        self._set_episode_buttons_state()
        self._set_review_buttons_state()
        self._update_status_label()

    def _write_session_log(self, episode_index, ep_len, elapsed, achieved):
        """Real timing lives here, not in the dataset -- LeRobot's `timestamp`
        column is the nominal frame_index/fps grid by design."""
        entry = {
            "episode_index": episode_index,
            "task": self._episode_task,
            "frames": ep_len,
            "wall_seconds": round(elapsed, 3),
            "target_fps": self.args.fps,
            "achieved_fps": round(achieved, 3),
            "missed_deadlines": self._episode_missed,
            "stale_frames": self._episode_repeats,
            "camera_fps": round(self._camera_fps(), 3),
            "start_time": self._episode_start_wall,
            "command_bus": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in self.bus.stats().items()},
        }
        try:
            with open(self.session_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            self._log(f"Could not write session log: {e}", level="warn")

    # -- teardown ------------------------------------------------------------

    def _quit(self):
        self._shutdown()

    def _shutdown(self):
        """Single exit path for the window close button, Q, SIGINT and any
        unhandled error. Order matters: stop the robot before anything that
        can block, and always finalize the dataset."""
        if self._shutting_down:
            return
        self._shutting_down = True

        if self._recording:
            self._recording = False
            if self.recorder is not None:
                try:
                    self.recorder.discard_episode()
                except Exception:
                    pass
            print("Shutting down mid-episode -- in-progress episode discarded", file=sys.stderr)

        # Stop the robot first, synchronously, before anything else can fail.
        try:
            self.bus.cancel_pending()
            self.bus.send_now("stop_all")
        except Exception:
            pass
        try:
            self.bus.close()
        except Exception:
            pass

        if self.link is not None:
            link, self.link = self.link, None
            try:
                link.close()
            except Exception:
                pass
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        self._close_review_video()
        if self.recorder is None:
            print("No dataset was created (nothing recorded).", file=sys.stderr)
        else:
            try:
                self.recorder.close()  # finalize(): without it the parquet has no footer
                print(
                    f"Dataset finalized: {self._n_episodes} episode(s), "
                    f"{self._n_frames} frames at {self.dataset_root}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"Dataset finalize failed: {e}", file=sys.stderr)

        try:
            self.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.list_cameras:
        list_cameras()
        raise SystemExit(0)

    try:
        app = CollectDataApp(cli_args)
    except Exception as e:
        print(f"Failed to start collect_data.py: {e}", file=sys.stderr)
        raise SystemExit(1)

    try:
        app.mainloop()
    finally:
        app._shutdown()
