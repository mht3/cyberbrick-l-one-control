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

from lone_data.camera import CameraStream
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

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Comfortably under wifi_bridge.py's COMMAND_DEADMAN_TIMEOUT (a single global timer, not per-joint).
WIFI_HEARTBEAT_MS = 200

# Auto-reconnect budget. A link that survives HEALTHY_LINK_SECONDS counts as
# having genuinely worked, so losing it later starts the budget over; links that
# die sooner are failing for a persistent reason and burn through it instead.
HEALTHY_LINK_SECONDS = 10.0
MAX_AUTO_RECONNECTS = 3

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


def _serial_error_hint(port, exc, ports=()):
    """The raw-REPL handshake failing with an empty buffer means nothing came
    back at all. The underlying TimeoutError text is accurate but unreadable,
    and it doesn't say which of several quite different causes applies."""
    if isinstance(exc, TimeoutError) and "got b''" in str(exc):
        seen = ", ".join(ports) if ports else "none"
        return (
            f"{port} opened but didn't answer the raw-REPL handshake within 3s.\n"
            f"  ports seen: {seen}\n"
            "  - A board already in WiFi mode may not answer over USB -- use WiFi instead.\n"
            "  - Otherwise power-cycle the board with the cable attached, then retry.\n"
            "  - If several ports are listed, the board may be a different one."
        )
    return f"connection to {port} failed: {exc}"


def _wifi_error_hint(host, exc):
    """wifi_bridge.py tries STA first and only starts the AP if STA fails, so an
    unreachable STA hostname usually means it fell back to the AP."""
    return (
        f"couldn't reach {host}:{WIFI_PORT} ({exc}).\n"
        f"  - The board only enters WiFi mode ~15s after boot with no remote paired, then reboots "
        f"-- allow ~25s total.\n"
        f"  - wifi_bridge.py joins the STA network from wifi_secrets.py first and starts its own AP "
        f"only if that fails. An unreachable hostname usually means STA didn't work and it is on "
        f"the AP: tick 'AP Mode', join '{AP_SSID}' in macOS WiFi settings, then Connect."
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index (see --list-cameras)")
    p.add_argument("--list-cameras", action="store_true", help="probe camera indices and exit")
    p.add_argument("--width", type=int, default=1280, help="requested camera capture width")
    p.add_argument("--height", type=int, default=720, help="requested camera capture height")
    p.add_argument("--camera-fps", type=int, default=30, help="requested camera capture fps")
    p.add_argument("--fps", type=int, default=25, help="dataset recording rate")
    p.add_argument("--repo-id", default="lone/l_one", help="LeRobot dataset repo id")
    p.add_argument("--root", default=None, help="dataset directory (default: data/lerobot/<repo-id>)")
    p.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_SIZE[1],
                   help="frame width stored in the dataset (must match the camera's aspect ratio)")
    p.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_SIZE[0],
                   help="frame height stored in the dataset (must match the camera's aspect ratio)")
    p.add_argument("--task", default="", help="default task/instruction string")
    return p.parse_args()


class CollectDataApp(tk.Tk):
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
        # Opened on the main thread so macOS's camera-permission prompt gets a run loop.
        self.camera = CameraStream(args.camera_index, args.width, args.height, args.camera_fps)
        self.camera.start()

        # -- dataset --------------------------------------------------------
        # Not created here. LeRobot writes meta/info.json the moment a dataset
        # exists, so creating one up front leaves an empty dataset behind every
        # time the app is opened and closed without recording. Deferred to the
        # first episode -- see _ensure_recorder().
        self.image_size = (args.image_height, args.image_width)
        self._check_storage_aspect()
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

        self._log(f"Dataset will be written to {self.dataset_root} @ {args.fps} fps", level="info")

        self.after(0, self._update_camera_feed)
        self.after(WIFI_HEARTBEAT_MS, self._wifi_heartbeat_tick)

    # -- misc setup -----------------------------------------------------

    def _signal_pump(self):
        # Lets Ctrl+C get delivered promptly -- mainloop() doesn't check for signals on its own.
        self.after(200, self._signal_pump)

    def _check_storage_aspect(self):
        """Frames are downscaled, never cropped or padded, so a stored size
        whose aspect ratio doesn't match the camera's silently stretches every
        frame in the dataset. Refuse up front instead."""
        store_h, store_w = self.image_size
        cam_w, cam_h = self.camera.actual_width, self.camera.actual_height
        if not cam_w or not cam_h:
            return
        if abs((store_w / store_h) - (cam_w / cam_h)) > 0.01:
            raise SystemExit(
                f"--image-width/--image-height {store_w}x{store_h} "
                f"({store_w / store_h:.3f}:1) does not match the camera's "
                f"{cam_w}x{cam_h} ({cam_w / cam_h:.3f}:1).\n"
                "Frames are scaled, not cropped or padded, so this would distort every frame. "
                f"Use a size with the camera's aspect ratio (e.g. {cam_w // 2}x{cam_h // 2})."
            )

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

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=PALETTE["bg"], foreground=PALETTE["text"], font=("Helvetica", 11))
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("TLabel", background=PALETTE["bg"])
        style.configure("TRadiobutton", background=PALETTE["bg"])
        # Wider insertion cursor -- default 1px caret is easy to lose track of.
        style.configure(
            "TEntry", fieldbackground="white", foreground=PALETTE["text"],
            insertcolor=PALETTE["text"], insertwidth=2,
        )
        style.configure("TButton", padding=6)
        style.configure("Joint.TButton", font=("Helvetica", 13), padding=(10, 6))
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground="white")
        style.map("Accent.TButton",
                  background=[("active", PALETTE["accent_active"]), ("disabled", PALETTE["border"])])
        style.configure("Danger.TButton", background=PALETTE["danger"], foreground="white")
        style.map("Danger.TButton", background=[("active", PALETTE["danger_active"])])
        style.configure("JointName.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("KeyHint.TLabel", foreground=PALETTE["muted"], font=("Helvetica", 9))
        style.configure("SectionHeading.TLabel", foreground=PALETTE["muted"], font=("Helvetica", 10, "bold"))
        style.configure("ModeStatus.TLabel", foreground=PALETTE["accent"], font=("Helvetica", 12, "bold"))

    # -- connection bar (reimplemented so virtual_gripper.py stays untouched) --

    def _build_connection_bar(self, parent):
        bar = ttk.Frame(parent, padding=(0, 0, 0, 12))
        bar.pack(fill="x")

        self.mode_var = tk.StringVar(value="")
        self.serial_radio = ttk.Radiobutton(
            bar, text="Serial", variable=self.mode_var, value="serial", command=self._on_mode_select
        )
        self.wifi_radio = ttk.Radiobutton(
            bar, text="WiFi", variable=self.mode_var, value="wifi", command=self._on_mode_select
        )
        self.serial_radio.grid(row=0, column=0, sticky="w")
        self.wifi_radio.grid(row=0, column=1, sticky="w", padx=(10, 0))

        # Kept in sync by _apply_mode_visibility.
        self.app_mode_status_var = tk.StringVar(
            value="Mode: Review" if self._review_mode else "Mode: Collect"
        )
        ttk.Label(bar, textvariable=self.app_mode_status_var, style="ModeStatus.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

        self.port_var = tk.StringVar(value=find_default_port())
        self.port_combo = ttk.Combobox(
            bar, textvariable=self.port_var, width=22, values=list_usb_ports(), postcommand=self._refresh_ports
        )
        self.port_combo.bind("<<ComboboxSelected>>", self._on_port_change)

        self.host_var = tk.StringVar(value=STA_HOSTNAME)
        self.host_entry = ttk.Entry(bar, textvariable=self.host_var, width=26)
        self.host_entry.bind("<Return>", lambda e: self._connect_wifi(self.host_var.get()))
        self.connect_btn = ttk.Button(
            bar, text="Connect", style="Accent.TButton", command=lambda: self._connect_wifi(self.host_var.get())
        )

        self.ap_mode_var = tk.BooleanVar(value=False)
        self.ap_mode_check = ttk.Checkbutton(
            bar, text="AP Mode", variable=self.ap_mode_var, command=self._on_ap_mode_toggle
        )
        self.sta_btn = ttk.Button(bar, text="Join UCSD (STA)", command=self._join_sta)

        self._transport_row = ttk.Frame(bar)
        self._transport_row.grid(row=2, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # Mode-independent -- stay reachable in both Collect and Review.
        ttk.Button(bar, text="Quit (Q)", command=self._quit).grid(row=0, column=2, sticky="e", padx=(10, 0))
        ttk.Button(bar, text="STOP ALL", style="Danger.TButton", command=self._stop_all).grid(
            row=0, column=3, sticky="e", padx=(10, 0)
        )

    def _refresh_ports(self):
        self.port_combo["values"] = list_usb_ports()

    def _resolve_port(self):
        ports = list_usb_ports()
        self.port_combo["values"] = ports
        port = self.port_var.get()
        if port not in ports:
            port = pick_default_port(ports)
            self.port_var.set(port)
        return port

    def _update_transport_controls_visibility(self):
        for w in self._transport_row.winfo_children():
            w.pack_forget()
        mode = self.mode_var.get()
        if mode == "serial":
            self.port_combo.pack(in_=self._transport_row, side="left")
        elif mode == "wifi":
            self.host_entry.pack(in_=self._transport_row, side="left")
            self.connect_btn.pack(in_=self._transport_row, side="left", padx=(6, 0))
            self.ap_mode_check.pack(in_=self._transport_row, side="left", padx=(8, 0))

    def _update_sta_button_visibility(self):
        if self.wifi_kind == "ap" and isinstance(self.link, CyberBrickWifiLink):
            self.sta_btn.pack(in_=self._transport_row, side="left", padx=(8, 0))
        else:
            self.sta_btn.pack_forget()

    def _on_ap_mode_toggle(self):
        if self.ap_mode_var.get():
            self.host_var.set(AP_FIXED_IP)
            self._log(
                f"AP mode on -- join '{AP_SSID}' (password: {AP_PASSWORD}) in your "
                f"computer's WiFi settings, then this connects to {AP_FIXED_IP}.",
                level="info",
            )
        else:
            self.host_var.set(STA_HOSTNAME)
            self._log("AP mode off -- back to the STA hostname.", level="info")
        self._connect_wifi(self.host_var.get(), silent=True)

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.serial_radio.config(state=state)
        self.wifi_radio.config(state=state)
        self.port_combo.config(state=state)
        self.host_entry.config(state=state)
        self.connect_btn.config(state=state)
        self.ap_mode_check.config(state=state)
        self.sta_btn.config(state=state)

    def _drop_link(self, reset_board=True):
        if self.link is not None:
            link = self.link
            self.link = None
            self.bus.set_link(None)
            # close() talks to the board (RESET or STOP), which is a full round
            # trip on a link that may already be dead. Doing that inline froze
            # the Tk main thread for the socket timeout every time a connection
            # was dropped -- including the drop at the start of every reconnect.
            threading.Thread(
                target=self._close_link_quietly, args=(link, reset_board), daemon=True
            ).start()
        self.wifi_kind = None
        self.ap_ip = None
        self._link_up_mono = None
        self._set_episode_buttons_state()

    def _close_link_quietly(self, link, reset_board):
        try:
            if isinstance(link, CyberBrickWifiLink):
                link.close(reset_board=reset_board)
            else:
                link.close()
        except Exception as e:
            self.after(0, self._log, f"Link close failed: {e}", "warn")

    def _on_link_dead(self, message):
        """The link stopped answering. Until this existed the link stayed
        'connected' forever while every command failed, so the only way out was
        to notice the red log lines and reconnect by hand.

        Reconnecting is only worth doing for a link that was actually working:
        a connection that dies on its first command is failing for a reason
        retrying won't fix, and retrying it in a tight loop just buries the real
        error under reconnect spam."""
        if self._shutting_down or self.link is None:
            return
        self._log(f"Link lost: {message}", level="error")
        was_wifi = isinstance(self.link, CyberBrickWifiLink)
        host = self.host_var.get().strip() if was_wifi else None
        # Read before _drop_link() clears it.
        link_up_mono = self._link_up_mono
        if self._recording:
            self._log("Link lost while recording -- discarding in-progress episode", level="warn")
            self._discard_episode()
        # reset_board=False: the board is already unreachable, and a RESET would
        # drop it out of WiFi mode entirely just as we try to reconnect.
        self._drop_link(reset_board=False)
        self._update_transport_controls_visibility()
        self._update_sta_button_visibility()

        if not (was_wifi and host):
            self._log("Reconnect from the controls above.", level="info")
            return

        uptime = time.monotonic() - (link_up_mono or 0)
        if link_up_mono is None or uptime < HEALTHY_LINK_SECONDS:
            self._failed_reconnects += 1
        else:
            self._failed_reconnects = 1  # a real session ended; this one is fresh
        if self._failed_reconnects > MAX_AUTO_RECONNECTS:
            self._log(
                f"Gave up after {MAX_AUTO_RECONNECTS} reconnects that each died within "
                f"{HEALTHY_LINK_SECONDS:.0f}s. The board is reachable but drops the connection, "
                "so this is the firmware or the network, not a blip -- check the board's serial "
                "output, or tick 'AP Mode' to bypass the campus network. Connect to retry.",
                level="error",
            )
            self._failed_reconnects = 0
            return
        delay = min(0.5 * 2 ** (self._failed_reconnects - 1), 5.0)
        self._log(
            f"Reconnecting to {host} in {delay:.1f}s "
            f"({self._failed_reconnects}/{MAX_AUTO_RECONNECTS})...",
            level="connecting",
        )
        self.after(int(delay * 1000), self._connect_wifi, host, True)

    def _fall_back_to_idle(self):
        """Drops the link but keeps the selected transport's controls on screen.
        Clearing mode_var here would hide the port dropdown and host field --
        exactly the controls needed to recover from a failed connection."""
        self._drop_link()
        self._update_transport_controls_visibility()
        self._update_sta_button_visibility()
        self._log("Disconnected -- adjust the port/host above and reconnect.", level="info")

    def _on_mode_select(self):
        mode = self.mode_var.get()
        self._update_transport_controls_visibility()
        if mode == "serial":
            self._connect_serial(self._resolve_port())
        elif mode == "wifi":
            self._connect_wifi(self.host_var.get(), silent=True)
        self._update_sta_button_visibility()

    def _on_port_change(self, event=None):
        if self.mode_var.get() == "serial":
            self._connect_serial(self.port_var.get())

    def _connect_serial(self, port):
        if not port:
            self._log("Select the board's USB port.", level="warn")
            self._fall_back_to_idle()
            return
        self._drop_link(reset_board=False)
        self._set_busy(True)
        self._log(f"Connecting to {port}...", level="connecting")
        self.update_idletasks()
        try:
            self.link = CyberBrickLink(port)
            self.bus.set_link(self.link)
            self._link_up_mono = time.monotonic()
            self._log(f"Connected via Serial ({port}, remote inactive)", level="connected")
            self._sync_gripper_state()
        except Exception as e:
            self._log(f"Serial: {_serial_error_hint(port, e, list_usb_ports())}", level="error")
            self._fall_back_to_idle()
        finally:
            self._set_busy(False)
            self._set_episode_buttons_state()

    def _connect_wifi(self, host, silent=False):
        host = host.strip()
        if not host:
            if not silent:
                self._log("Enter the board's IP address or hostname.", level="warn")
            self._fall_back_to_idle()
            return
        if not silent:
            self._failed_reconnects = 0  # explicit user action -- fresh budget
        self._drop_link(reset_board=False)
        self._wifi_connect_generation += 1
        generation = self._wifi_connect_generation
        self._log(f"Connecting to {host}:{WIFI_PORT}...", level="connecting")
        result_q = queue.Queue()

        def worker():
            last_err = None
            for attempt in range(3):
                try:
                    result_q.put(("ok", CyberBrickWifiLink(host, WIFI_PORT)))
                    return
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(1.5)
            result_q.put(("err", last_err))

        threading.Thread(target=worker, daemon=True).start()
        self.after(200, self._poll_wifi_connect, result_q, host, silent, generation)

    def _poll_wifi_connect(self, result_q, host, silent, generation):
        try:
            status, payload = result_q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_wifi_connect, result_q, host, silent, generation)
            return

        if generation != self._wifi_connect_generation:
            if status == "ok":
                payload.close(reset_board=False)
            return

        if status == "err":
            if silent:
                self._log(f"Couldn't reach {host}:{WIFI_PORT} -- edit Host and Connect", level="warn")
                self._update_sta_button_visibility()
            else:
                self._fall_back_to_idle()
                self._log(f"WiFi: {_wifi_error_hint(host, payload)}", level="error")
            return

        self.link = payload
        self.bus.set_link(self.link)
        self._link_up_mono = time.monotonic()
        self.wifi_kind = "ap" if host == AP_FIXED_IP else "sta"
        self.ap_ip = host if self.wifi_kind == "ap" else None
        label = f"AP {host}" if self.wifi_kind == "ap" else f"STA at {host}"
        self._log(f"Connected via WiFi ({label}, remote inactive)", level="connected")
        self._sync_gripper_state()
        self._update_sta_button_visibility()
        self._set_episode_buttons_state()

    def _join_sta(self):
        if not isinstance(self.link, CyberBrickWifiLink):
            return
        link = self.link
        self._set_busy(True)
        self._log("Joining UCSD network (STA)...", level="connecting")
        result_q = queue.Queue()

        def worker():
            try:
                result_q.put(("ok", link.connect_sta()))
            except Exception as e:
                result_q.put(("err", e))

        threading.Thread(target=worker, daemon=True).start()
        self.after(200, self._poll_sta_join, result_q)

    def _poll_sta_join(self, result_q):
        try:
            status, payload = result_q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll_sta_join, result_q)
            return
        self._set_busy(False)
        if status == "err":
            self._log(f"STA connect failed: {payload} (still on WiFi at {self.ap_ip})", level="error")
            return
        ip = payload
        self._log(
            f"Connected via WiFi (AP {self.ap_ip} + UCSD STA {ip}) -- also reachable at "
            f"{ip}:{WIFI_PORT}, no need to stay in AP range",
            level="connected",
        )

    # -- joint controls (identical keybindings/behavior to virtual_gripper) --

    def _build_controls(self, parent):
        body = ttk.Frame(parent, padding=(0, 4, 0, 12))
        body.pack(fill="x")

        self.base_control = JointControl(lambda s: self._motor(BASE_MOTOR, s), MOTOR_SPEED)
        self.upper_control = JointControl(lambda s: self._servo_speed(UPPER_ARM_SERVO, s, 1), JOINT_SPEED)
        self.lower_control = JointControl(lambda s: self._servo_speed(LOWER_ARM_SERVO, s, 2), JOINT_SPEED)

        self._build_joint_row(body, "Base", self.base_control, row=0, key_hint=", / .")
        self._build_joint_row(body, "Upper Arm", self.upper_control, row=1, key_hint="W / S  ·  ↑ / ↓")
        self._build_joint_row(body, "Lower Arm", self.lower_control, row=2, key_hint="A / D  ·  ← / →")

        gripper_row = ttk.Frame(body)
        gripper_row.grid(row=3, column=0, pady=(18, 0), sticky="w")
        ttk.Label(gripper_row, text="Gripper", width=12, style="JointName.TLabel").pack(side="left")
        self.gripper_btn = ttk.Button(gripper_row, text="Close Clamp", command=self._toggle_gripper, width=16)
        self.gripper_btn.pack(side="left", padx=4)
        ttk.Label(gripper_row, text="Space", style="KeyHint.TLabel").pack(side="left", padx=(10, 0))
        return body

    def _build_joint_row(self, parent, label, control, row, key_hint):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, pady=6, sticky="w")
        ttk.Label(frame, text=label, width=12, style="JointName.TLabel").pack(side="left")

        neg_btn = ttk.Button(frame, text="◀", width=4, style="Joint.TButton")
        pos_btn = ttk.Button(frame, text="▶", width=4, style="Joint.TButton")

        neg_btn.bind("<ButtonPress-1>", lambda e: control.press(-1, "mouse"))
        neg_btn.bind("<ButtonRelease-1>", lambda e: control.release(-1, "mouse"))
        pos_btn.bind("<ButtonPress-1>", lambda e: control.press(1, "mouse"))
        pos_btn.bind("<ButtonRelease-1>", lambda e: control.release(1, "mouse"))

        neg_btn.pack(side="left", padx=4)
        pos_btn.pack(side="left", padx=4)
        ttk.Label(frame, text=key_hint, style="KeyHint.TLabel").pack(side="left", padx=(10, 0))

    def _setup_keybindings(self):
        joint_controls = {"base": self.base_control, "upper": self.upper_control, "lower": self.lower_control}
        for keysym, (joint, direction) in KEY_JOINTS.items():
            control = joint_controls[joint]
            source = f"key:{keysym}"
            self._bind_hold_key(
                keysym,
                lambda c=control, d=direction, s=source: c.press(d, s),
                lambda c=control, d=direction, s=source: c.release(d, s),
            )
        self._bind_hold_key("space", self._toggle_gripper, lambda: None)

    def _bind_hold_key(self, keysym, on_press, on_release):
        def handle_press(event):
            if self._focused_widget_wants_text() or self._review_mode:
                return
            pending = self._key_release_after.pop(keysym, None)
            if pending is not None:
                self.after_cancel(pending)
            if keysym in self._pressed_keys:
                return
            self._pressed_keys.add(keysym)
            on_press()

        def handle_release(event):
            def actually_release():
                self._key_release_after.pop(keysym, None)
                self._pressed_keys.discard(keysym)
                on_release()
            self._key_release_after[keysym] = self.after(40, actually_release)

        self.bind(f"<KeyPress-{keysym}>", handle_press)
        self.bind(f"<KeyRelease-{keysym}>", handle_release)

    def _focused_widget_wants_text(self):
        return isinstance(self.focus_get(), (ttk.Entry, ttk.Combobox))

    def _maybe_reclaim_focus(self, event):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            return
        self.after_idle(self.focus_set)

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

    def _link_readout(self):
        """Command latency, so a degrading link is visible before an episode is
        recorded over it rather than only in logs/collection_*.jsonl afterwards."""
        if self.link is None:
            return "link none"
        s = self.bus.stats()
        if not s["sent"]:
            return "link idle"
        text = f"link {s['latency_mean'] * 1000:.0f}/{s['latency_max'] * 1000:.0f} ms avg/max"
        if s["dropped"]:
            text += f"  ·  {s['dropped']} stale cmds"
        return text

    def _update_status_label(self):
        if self._recording:
            elapsed = time.monotonic() - self._episode_start_mono
            self.status_var.set(
                f"Recording episode {self._n_episodes:06d} -- "
                f"{self._episode_step_count} steps, {elapsed:0.1f}s"
            )
            achieved = self._episode_step_count / elapsed if elapsed > 0 else 0.0
            self.rate_var.set(
                f"{achieved:.1f}/{self.args.fps} fps  ·  camera {self.camera.measured_fps:.1f} fps  ·  "
                f"missed {self._episode_missed}  ·  stale frames {self._episode_repeats}  ·  "
                f"{self._link_readout()}"
            )
        else:
            self.status_var.set(
                f"Idle -- {self._n_episodes} episode(s), {self._n_frames} frames"
            )
            self.rate_var.set(
                f"camera {self.camera.measured_fps:.1f} fps  ·  {self._link_readout()}"
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

        self.review_task_var = tk.StringVar(value="")
        ttk.Label(self.review_active_frame, textvariable=self.review_task_var, style="KeyHint.TLabel").pack(
            fill="x", pady=(6, 0)
        )

        self._set_review_buttons_state()

    def _set_review_buttons_state(self):
        self.review_enter_btn.config(state="disabled" if (self._review_mode or self._recording) else "normal")
        state = "normal" if self._review_mode else "disabled"
        self.review_prev_btn.config(state=state)
        self.review_next_btn.config(state=state)
        self.review_exit_btn.config(state=state)
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
        self.review_task_var.set(f"Task: {ep['task'] or '(recorded in an earlier session)'}")
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

    def _next_review_episode(self):
        if not self._review_mode or self._review_episode_idx is None:
            return
        if self._review_episode_idx + 1 >= self._n_episodes:
            self._log("Reached the last episode.", level="info")
            return
        self._load_review_episode(self._review_episode_idx + 1)

    def _previous_review_episode(self):
        if not self._review_mode or self._review_episode_idx is None:
            return
        if self._review_episode_idx == 0:
            self._log("Already at the first episode.", level="info")
            return
        self._load_review_episode(self._review_episode_idx - 1)

    # -- log panel --------------------------------------------------------

    def _build_log_panel(self, parent):
        # Starts collapsed -- _log() still writes to log_text either way, so nothing is lost.
        self.log_frame = ttk.Frame(parent, padding=(16, 8, 16, 4))
        self.log_frame.pack(side="bottom", fill="x")

        self._log_expanded = False
        self.log_header_var = tk.StringVar()
        header = ttk.Label(
            self.log_frame, textvariable=self.log_header_var, style="SectionHeading.TLabel", cursor="hand2"
        )
        header.pack(anchor="w", pady=(0, 4))
        header.bind("<Button-1>", lambda e: self._toggle_log_panel())

        self.log_text_frame = ttk.Frame(self.log_frame)

        mono_font = ("Menlo" if sys.platform == "darwin" else "Consolas", 10)
        self.log_text = tk.Text(
            self.log_text_frame, height=10, width=48, wrap="word", state="disabled", font=mono_font,
            background=PALETTE["log_bg"], foreground=PALETTE["log_fg"], insertbackground=PALETTE["log_fg"],
            relief="flat", highlightthickness=1, highlightbackground=PALETTE["border"], padx=8, pady=6,
        )
        scrollbar = ttk.Scrollbar(self.log_text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for level, color in {
            "info": "#9aa4b2", "connecting": "#e0a640", "connected": "#7ee08a",
            "warn": "#e0a640", "error": "#ff7b72",
        }.items():
            self.log_text.tag_config(level, foreground=color)

        self._update_log_header()

    def _toggle_log_panel(self):
        # log_frame stays fill="x" only -- it grows via log_text_frame's own
        # height, and main_row (expand=True) shrinks to make room.
        self._log_expanded = not self._log_expanded
        if self._log_expanded:
            self.log_text_frame.pack(fill="both", expand=True)
        else:
            self.log_text_frame.pack_forget()
        self._update_log_header()

    def _update_log_header(self):
        arrow = "▼" if self._log_expanded else "▶"
        self.log_header_var.set(f"{arrow} Log (click to {'collapse' if self._log_expanded else 'expand'})")

    def _log(self, message, level="info"):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n", level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # -- camera feed ------------------------------------------------------

    def _update_camera_feed(self):
        self.after(FEED_REFRESH_MS, self._update_camera_feed)
        if self._review_mode:
            return  # _review_tick owns camera_label while reviewing
        # The idle readout is otherwise only written at startup, when the camera
        # hasn't produced a frame yet and its rate still reads 0.
        self._feed_ticks += 1
        if not self._recording and self._feed_ticks % max(1, 1000 // FEED_REFRESH_MS) == 0:
            self._update_status_label()
        latest = self.camera.get_latest()
        if latest is None:
            return
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

    def _require_link(self):
        if self.link is None:
            self._log("Connect to the board first.", level="warn")
            return False
        if self._review_mode:
            return False  # silently ignored -- joint controls are visibly disabled during review
        return True

    def _motor(self, idx, speed):
        if not self._require_link():
            return
        # Value 0 is dispatched as stop(), not set_speed(idx, 0) -- different
        # hardware states. See ZERO_DISPATCH_CONVENTION in lone_data/features.py.
        # Stops are not droppable: nothing re-sends them, so discarding one
        # leaves the joint driving until the board's deadman timer fires.
        if speed == 0:
            self.bus.submit(f"motor:{idx}", "stop_motor", idx)
        else:
            self.bus.submit(f"motor:{idx}", "set_motor_speed", idx, speed, droppable=True)
        self._current_action[0] = float(speed)

    def _servo_speed(self, idx, speed, action_index):
        if not self._require_link():
            return
        if speed == 0:
            self.bus.submit(f"servo:{idx}", "stop_servo", idx)
        else:
            self.bus.submit(f"servo:{idx}", "set_servo_speed", idx, speed, droppable=True)
        self._current_action[action_index] = float(speed)

    def _sync_gripper_state(self):
        self.bus.submit(f"servo_angle:{GRIPPER_SERVO}", "set_servo_angle", GRIPPER_SERVO, GRIPPER_OPEN_ANGLE)
        self.gripper_open = True
        self.gripper_btn.config(text="Close Clamp")
        self._reset_action_state(reopen_gripper=True)

    def _toggle_gripper(self):
        if not self._require_link():
            return
        angle = GRIPPER_CLOSED_ANGLE if self.gripper_open else GRIPPER_OPEN_ANGLE
        self.bus.submit(f"servo_angle:{GRIPPER_SERVO}", "set_servo_angle", GRIPPER_SERVO, angle)
        self.gripper_open = not self.gripper_open
        self._current_action[3] = float(angle)
        self.gripper_btn.config(text="Open Clamp" if not self.gripper_open else "Close Clamp")

    def _reset_action_state(self, reopen_gripper):
        # Reaches into JointControl's internals -- no public reset() exists upstream.
        for jc in (self.base_control, self.upper_control, self.lower_control):
            jc._pos_sources.clear()
            jc._neg_sources.clear()
            jc._current = 0
        self._current_action[0:3] = 0.0
        if reopen_gripper:
            self._current_action[3] = float(GRIPPER_OPEN_ANGLE)

    def _stop_all(self):
        if self.link is None:
            return
        # Drop anything queued first so a stale speed command can't land after the stop.
        self.bus.cancel_pending()
        self.bus.send_now("stop_all")
        self._log("Stopped all joints.", level="info")
        # stop_all() bypasses JointControl, so a still-held key could desync afterward.
        if self._recording:
            self._log("STOP ALL during recording -- discarding in-progress episode", level="warn")
            self._discard_episode()
        self._reset_action_state(reopen_gripper=False)

    def _wifi_heartbeat_tick(self):
        self.after(WIFI_HEARTBEAT_MS, self._wifi_heartbeat_tick)
        if not isinstance(self.link, CyberBrickWifiLink):
            return
        # Queued, not sent inline -- this used to block the main thread every 200ms.
        # These are the only commands that get re-sent on a timer, which is what
        # makes them safe to drop when stale (see CommandBus.submit).
        if self.base_control._current != 0:
            self.bus.submit(f"motor:{BASE_MOTOR}", "set_motor_speed", BASE_MOTOR,
                            self.base_control._current, droppable=True)
        if self.upper_control._current != 0:
            self.bus.submit(f"servo:{UPPER_ARM_SERVO}", "set_servo_speed", UPPER_ARM_SERVO,
                            self.upper_control._current, droppable=True)
        if self.lower_control._current != 0:
            self.bus.submit(f"servo:{LOWER_ARM_SERVO}", "set_servo_speed", LOWER_ARM_SERVO,
                            self.lower_control._current, droppable=True)

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
        if self.camera.get_latest() is None:
            self._log("No camera frame available yet -- wait a moment and try again.", level="warn")
            return
        task = self.task_var.get().strip()
        if not task:
            self._log("Enter a task/instruction before recording -- every episode needs one.", level="warn")
            self.task_entry.focus_set()
            return

        cam_fps = self.camera.measured_fps
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
            latest = self.camera.get_latest()
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
            "camera_fps": round(self.camera.measured_fps, 3),
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
