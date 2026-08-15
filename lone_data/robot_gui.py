"""Shared Tk application base for the L-ONE GUIs.

collect_data.py and deploy_policy.py are the same machine wearing two hats: the
same board, link, camera, joint controls and keybindings, differing only in what
they do with the frames and who decides the next action. This holds everything
they share, so the parts where drift would be dangerous -- link lifecycle, the
stop paths, the zero-dispatch convention -- exist once.

Everything here was moved verbatim out of collect_data.py, with the handful of
collector-specific reaches replaced by hooks a subclass overrides:

    _controls_enabled()      manual joint input accepted right now?
    _controls_busy()         reason to refuse a camera switch, or None
    _mode_status_text()      caption under the transport radio buttons
    _on_link_state_changed() link connected/dropped; re-enable buttons
    _on_link_lost()          the link died unexpectedly
    _on_emergency_stop()     STOP ALL was pressed
    _after_camera_opened()   a new camera source is live
    _update_status_label()   the app's own status line
    _on_action_changed()     _current_action was just written to

A subclass must, before calling _build_*: set self.args, self.camera = None,
self._camera_source, self._camera_connected, self.link = None, self.bus, and
self._current_action.
"""

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

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
    find_default_port,
    list_usb_ports,
    pick_default_port,
)

from lone_data import camera_source
from lone_data.features import ACTION_DIM

# Comfortably under wifi_bridge.py's COMMAND_DEADMAN_TIMEOUT (a single global timer).
WIFI_HEARTBEAT_MS = 200

# Auto-reconnect budget. A link that survives HEALTHY_LINK_SECONDS counts as having
# genuinely worked, so losing it later starts the budget over; links that die sooner
# are failing for a persistent reason and burn through it instead.
HEALTHY_LINK_SECONDS = 10.0
MAX_AUTO_RECONNECTS = 3


def _serial_error_hint(port, exc, ports=()):
    """The raw-REPL handshake failing with an empty buffer means nothing came
    back at all. The underlying TimeoutError text is accurate but unreadable,
    and it doesn't say which of several quite different causes applies."""
    if isinstance(exc, TimeoutError) and "got b''" in str(exc):
        others = [p for p in ports if p != port]
        hint = (
            f"No response from {port}. The board is plugged in but its REPL isn't "
            "answering -- most often it's running the RC firmware, or another program "
            "(Thonny, screen, Pymakr) already holds the port."
        )
        if others:
            hint += f" Other ports seen: {', '.join(others)}."
        return hint
    return f"{type(exc).__name__}: {exc}"


def _wifi_error_hint(host, exc):
    if isinstance(exc, (TimeoutError, OSError)) and not isinstance(exc, ConnectionRefusedError):
        return (
            f"Couldn't reach {host}:{WIFI_PORT} -- edit Host and Connect. "
            "The board only enters WiFi mode ~15s after boot with no remote paired, then reboots "
            "into it; if the remote is on, turn it off and power-cycle the board."
        )
    return f"{type(exc).__name__}: {exc}"


class RobotAppBase(tk.Tk):
    """Connection bar, manual joint control, camera source and log panel."""

    # -- hooks: defaults make the base usable on its own -------------------

    def _controls_enabled(self):
        return True

    def _controls_busy(self):
        return None

    def _mode_status_text(self):
        return ""

    def _on_link_state_changed(self):
        pass

    def _on_link_lost(self):
        pass

    def _on_emergency_stop(self):
        pass

    def _after_camera_opened(self):
        pass

    def _update_status_label(self):
        pass

    def _on_action_changed(self):
        """Called whenever _current_action changes, whoever changed it."""
        pass

    def _quit(self):
        """The connection bar's Quit button. Subclasses own _shutdown()."""
        self._shutdown()

    def _signal_pump(self):
        # Lets Ctrl+C get delivered promptly -- mainloop() doesn't check for signals on its own.
        self.after(200, self._signal_pump)

    def _camera_fps(self):
        return self.camera.measured_fps if self.camera is not None else 0.0

    def _camera_source_label(self, source):
        return camera_source.source_label(source, self.args.remote_camera_port)

    def _refresh_camera_sources(self):
        sources = camera_source.available_sources(
            self._camera_source, self.camera is not None,
            include_remote=self.args.remote_camera or self._camera_source == camera_source.REMOTE_SOURCE,
        )
        values = [self._camera_source_label(s) for s in sources]
        current = self._camera_source_label(self._camera_source)
        if current not in values:
            values.insert(0, current)
        self.camera_combo["values"] = values

    def _open_camera(self, source):
        """Swap to `source`, returning True on success. Never raises: a camera that
        will not open must leave the app usable so another source can be chosen."""
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = None
        try:
            self.camera = camera_source.open_source(
                source, self.args.remote_camera_port,
                self.args.width, self.args.height, self.args.camera_fps,
            )
        except Exception as e:
            self._log(f"{self._camera_source_label(source)}: {e}", level="error")
            return False
        self._camera_source = source
        # Cleared so the next frame to arrive announces itself -- for the remote
        # receiver, binding the port says nothing about a sender being there.
        self._camera_connected = False
        if source == camera_source.REMOTE_SOURCE:
            self._log(f"Waiting for a sender on port {self.args.remote_camera_port} "
                      "(run stream_camera.py on the machine with the camera)")
        else:
            self._log(f"Opened {self._camera_source_label(source)}")
        return True

    def _note_camera_connected(self):
        """Log the first frame from the current source, once."""
        if self._camera_connected or self.camera is None:
            return
        self._camera_connected = True
        w, h = self.camera.actual_width, self.camera.actual_height
        self._log(f"Receiving frames from {self._camera_source_label(self._camera_source)}"
                  f" ({w}x{h})", level="connected")

    def _on_camera_source(self, event=None):
        source = camera_source.parse_label(self.camera_source_var.get())
        if source is None:
            return
        if source == self._camera_source and self.camera is not None:
            return
        busy = self._controls_busy()
        if busy:
            self._log(f"{busy} before switching camera.", level="warn")
            self.camera_source_var.set(self._camera_source_label(self._camera_source))
            return
        if self._open_camera(source):
            self._after_camera_opened()
        self.camera_source_var.set(self._camera_source_label(self._camera_source))

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
        self.app_mode_status_var = tk.StringVar(value=self._mode_status_text())
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
        self._on_link_state_changed()

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
        self._on_link_lost()
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
            self._on_link_state_changed()

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
        self._on_link_state_changed()

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
            if self._focused_widget_wants_text() or not self._controls_enabled():
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

    def _require_link(self):
        if self.link is None:
            self._log("Connect to the board first.", level="warn")
            return False
        if not self._controls_enabled():
            return False  # silently ignored -- controls are visibly disabled in this mode
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
        self._on_action_changed()

    def _servo_speed(self, idx, speed, action_index):
        if not self._require_link():
            return
        if speed == 0:
            self.bus.submit(f"servo:{idx}", "stop_servo", idx)
        else:
            self.bus.submit(f"servo:{idx}", "set_servo_speed", idx, speed, droppable=True)
        self._current_action[action_index] = float(speed)
        self._on_action_changed()

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
        self._on_action_changed()
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
        self._on_action_changed()

    def _stop_all(self):
        if self.link is None:
            return
        # Drop anything queued first so a stale speed command can't land after the stop.
        self.bus.cancel_pending()
        self.bus.send_now("stop_all")
        self._log("Stopped all joints.", level="info")
        # stop_all() bypasses JointControl, so a still-held key could desync afterward.
        self._on_emergency_stop()
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

