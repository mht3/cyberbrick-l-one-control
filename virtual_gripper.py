#!/usr/bin/env python3
"""Virtual controller GUI for the CyberBrick L-ONE arm.
"""

import glob
import importlib.util
import os
import queue
import socket
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import serial

DEFAULT_BAUDRATE = 115200
DEFAULT_PORT_HINT = "usbmodem101"  # prefer this port if several are present
WIFI_MODE_FLAG = "/wifi_mode.flag"  # must match boot.py / wifi_bridge.py

# Channel indices (1-indexed), from Robotic arm.json / control.py
BASE_MOTOR = 1        # MOTOR1 - continuous rotation
UPPER_ARM_SERVO = 1   # PWM1 - continuous "speed"-type servo
LOWER_ARM_SERVO = 2   # PWM2 - continuous "speed"-type servo
GRIPPER_SERVO = 3     # PWM3 - positional "angle"-type servo (0-180)

GRIPPER_OPEN_ANGLE = 30
GRIPPER_CLOSED_ANGLE = 120
JOINT_SPEED = 100  # ServosController.set_speed range: -100..100 (bbl/servos.py) -- full power
MOTOR_SPEED = 900  # MotorsController.set_speed range: -2048..2048 (bbl/motors.py)

# keysym -> (joint name, direction). "comma"/"period" are the Tk keysyms
# for ',' and '.'; Left/Right/Up/Down are the arrow keys (alternates for
# Lower Arm and Upper Arm respectively).
KEY_JOINTS = {
    "a": ("lower", -1),
    "d": ("lower", 1),
    "Left": ("lower", -1),
    "Right": ("lower", 1),
    "w": ("upper", -1),
    "s": ("upper", 1),
    "Up": ("upper", -1),
    "Down": ("upper", 1),
    "comma": ("base", -1),
    "period": ("base", 1),
}
GRIPPER_KEY = "space"

WIFI_PORT = 8266


def _load_wifi_secrets():
    """Loads LOneGripper/wifi_secrets.py by path (gitignored -- see wifi_secrets.example.py)."""
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(repo_dir, "LOneGripper", "wifi_secrets.py")
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} not found -- copy LOneGripper/wifi_secrets.example.py to "
            "LOneGripper/wifi_secrets.py and fill in your AP/STA hostname, IP, "
            "SSID, and password."
        )
    spec = importlib.util.spec_from_file_location("_lone_wifi_secrets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_wifi_secrets = _load_wifi_secrets()
STA_HOSTNAME = _wifi_secrets.STA_HOSTNAME
AP_FIXED_IP = _wifi_secrets.AP_FIXED_IP
AP_SSID = _wifi_secrets.AP_SSID
AP_PASSWORD = _wifi_secrets.AP_PASSWORD

PALETTE = {
    "bg": "#f4f5f7",
    "border": "#d7dbe0",
    "text": "#20232a",
    "muted": "#6b7280",
    "accent": "#2563eb",
    "accent_active": "#1d4fd1",
    "danger": "#dc2626",
    "danger_active": "#b91c1c",
    "log_bg": "#1e2128",
    "log_fg": "#d7dae0",
}


def list_usb_ports():
    return sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/tty.usbmodem*"))


def pick_default_port(ports):
    """Prefer a port matching DEFAULT_PORT_HINT; fall back to the first one."""
    for port in ports:
        if DEFAULT_PORT_HINT in port:
            return port
    return ports[0] if ports else ""


def find_default_port():
    return pick_default_port(list_usb_ports())


def _raw_repl_read_until(ser, buf, marker, timeout=3.0):
    """Read from `ser` (draining into `buf`) until `marker` appears.

    Returns (data_before_marker, remaining_buf_after_marker) -- bytes read
    past the marker are handed back for the next call, since serial reads
    don't align to protocol boundaries: one chunk can contain the marker
    plus data belonging to the next response.
    """
    deadline = time.time() + timeout
    while marker not in buf:
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {marker!r}, got {buf!r}")
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
        else:
            time.sleep(0.01)
    idx = buf.index(marker)
    return buf[:idx], buf[idx + len(marker):]


def _raw_repl_enter(ser):
    """Interrupt whatever's running and drop into the raw REPL. Returns the
    (empty, normally) buffer tail to keep feeding into later reads."""
    ser.write(b"\r\x03\x03")  # interrupt whatever is running
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"\r\x01")  # Ctrl-A -> raw REPL
    _, buf = _raw_repl_read_until(ser, b"", b"raw REPL; CTRL-B to exit\r\n>")
    return buf


def _raw_repl_exec(ser, buf, code):
    """Run `code` in an active raw-REPL session. Returns (stdout, new_buf)."""
    ser.write(code.encode("utf-8") + b"\x04")  # Ctrl-D runs it
    ok, buf = _raw_repl_read_until(ser, buf, b"OK")
    if ok != b"":
        raise RuntimeError(f"Unexpected data before OK: {ok!r}")
    out, buf = _raw_repl_read_until(ser, buf, b"\x04")
    err, buf = _raw_repl_read_until(ser, buf, b"\x04")
    _, buf = _raw_repl_read_until(ser, buf, b">")  # trailing raw-REPL prompt
    if err:
        raise RuntimeError(err.decode(errors="replace"))
    return out.decode(errors="replace"), buf


class CyberBrickLink:
    """Raw-REPL bridge to the board, scoped to one connect/disconnect cycle."""

    def __init__(self, port, baudrate=DEFAULT_BAUDRATE):
        self.ser = serial.Serial(port, baudrate, timeout=2)
        try:
            time.sleep(0.3)
            self._rx_buf = _raw_repl_enter(self.ser)
            # Serial mode always hard-resets the board on disconnect (see
            # close()). If /wifi_mode.flag is still set from a prior WiFi
            # session that got dropped with reset_board=False (switching
            # here from WiFi, or a reconnect that never re-landed on a live
            # WiFi link -- see ControllerApp._drop_link), that reset would
            # boot straight back into wifi_bridge.py instead of RC firmware
            # -- boot.py checks the flag before rc_main.py's watchdog ever
            # runs. Clear it unconditionally here so taking over via Serial
            # always leaves the board able to boot back into RC.
            self.exec(
                "import uos\n"
                "try:\n"
                f"    uos.remove({WIFI_MODE_FLAG!r})\n"
                "except OSError:\n"
                "    pass"
            )
            self.exec(
                "from control import ServosControllerExecMapper as _S, "
                "MotorsControllerExecMapper as _M\n"
                "_servos = _S()\n"
                "_motors = _M()"
            )
        except Exception:
            # Otherwise the port stays open with no owner -- the next
            # connect attempt fails "port busy" until the GUI restarts.
            self.ser.close()
            raise

    def exec(self, code):
        out, self._rx_buf = _raw_repl_exec(self.ser, self._rx_buf, code)
        return out

    def set_servo_speed(self, idx, speed):
        self.exec(f"_servos.set_speed({idx}, {speed})")

    def stop_servo(self, idx):
        self.exec(f"_servos.stop({idx})")

    def set_servo_angle(self, idx, angle):
        self.exec(f"_servos.set_angle({idx}, {angle})")

    def set_motor_speed(self, idx, speed):
        self.exec(f"_motors.set_speed({idx}, {speed})")

    def stop_motor(self, idx):
        self.exec(f"_motors.stop({idx})")

    def stop_all(self):
        self.exec(
            "_motors.stop(1); _motors.stop(2); _servos.stop(1); _servos.stop(2)"
        )

    def close(self):
        try:
            self.stop_all()
        except Exception:
            pass
        try:
            # Real hardware reset, not a soft Ctrl-D one -- entering the raw
            # REPL may have interrupted a live RC/ESP-NOW session, and a
            # soft reset alone doesn't reliably release that radio state.
            # machine.reset() never returns, so don't wait for a response.
            self.ser.write(b"import machine\nmachine.reset()" + b"\x04")
        finally:
            time.sleep(0.3)
            self.ser.close()


class CyberBrickWifiLink:
    """TCP bridge to wifi_bridge.py running on the board (see LOneGripper/).

    Unlike the USB link, this doesn't interrupt a running main.py -- the
    board is running wifi_bridge.py as its main program instead of the
    stock RC firmware, so the physical remote is inactive the whole time
    that firmware choice is active, not just while a client is connected.
    """

    def __init__(self, host, port=WIFI_PORT, timeout=5):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self._rx_buf = b""

    def _readline(self):
        while b"\n" not in self._rx_buf:
            chunk = self.sock.recv(256)
            if not chunk:
                raise ConnectionError("Connection closed by board")
            self._rx_buf += chunk
        line, self._rx_buf = self._rx_buf.split(b"\n", 1)
        return line.decode(errors="replace")

    def _send(self, line):
        self.sock.sendall((line + "\n").encode())
        reply = self._readline()
        if reply.startswith("ERR"):
            raise RuntimeError(reply)
        return reply

    def set_servo_speed(self, idx, speed):
        self._send(f"SERVO_SPEED {idx} {speed}")

    def stop_servo(self, idx):
        self._send(f"SERVO_SPEED {idx} 0")

    def set_servo_angle(self, idx, angle):
        self._send(f"SERVO_ANGLE {idx} {angle}")

    def set_motor_speed(self, idx, speed):
        self._send(f"MOTOR {idx} {speed}")

    def stop_motor(self, idx):
        self._send(f"MOTOR {idx} 0")

    def stop_all(self):
        self._send("STOP")

    def connect_sta(self):
        """Ask the board to also join wifi_secrets.py's STA_SSID alongside
        its AP (see wifi_bridge.py's STA_CONNECT command -- AP stays up, so
        this connection survives). Returns the IP it's assigned there."""
        reply = self._send("STA_CONNECT")  # "OK <ip>"
        return reply.split(" ", 1)[1]

    def close(self, reset_board=True):
        """By default, RESET also deletes /wifi_mode.flag on the board (see
        wifi_bridge.py), handing control back to the stock RC firmware --
        not just closing this TCP session. Pass reset_board=False when
        dropping this link only to immediately reconnect or switch to
        Serial (see ControllerApp._drop_link) -- otherwise the board reboots
        out of WiFi mode out from under the connection attempt that's about
        to replace this one."""
        if reset_board:
            try:
                self._send("RESET")
            except Exception:
                try:
                    self.stop_all()
                except Exception:
                    pass
        else:
            try:
                self.stop_all()
            except Exception:
                pass
        try:
            self.sock.close()
        except Exception:
            pass


class JointControl:
    """Merges mouse-hold and keyboard-hold inputs for one joint so either
    (or both at once) can drive it without stomping on each other: the
    joint only actually stops once every source currently holding it has
    released, and a command is only sent when the resulting target speed
    actually changes."""

    def __init__(self, speed_cb, speed):
        self._speed_cb = speed_cb
        self._speed = speed
        self._neg_sources = set()
        self._pos_sources = set()
        self._current = 0

    def press(self, direction, source):
        (self._neg_sources if direction < 0 else self._pos_sources).add(source)
        self._update()

    def release(self, direction, source):
        (self._neg_sources if direction < 0 else self._pos_sources).discard(source)
        self._update()

    def _update(self):
        if self._pos_sources and not self._neg_sources:
            target = self._speed
        elif self._neg_sources and not self._pos_sources:
            target = -self._speed
        else:
            target = 0
        if target != self._current:
            self._current = target
            self._speed_cb(target)


class ControllerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CyberBrick L-ONE Virtual Controller")
        self.configure(background=PALETTE["bg"])
        self.minsize(640, 480)
        self._setup_style()

        self.link = None
        self.ap_ip = None
        self.wifi_kind = None  # "ap" or "sta" -- which wifi_bridge.py brought up
        self.gripper_open = True
        self._pressed_keys = set()
        self._key_release_after = {}
        # Bumped on every _connect_wifi() call so a slow/retrying attempt
        # (e.g. the AP Mode toggle's ~20s worst case while you haven't
        # joined that network yet) can be superseded by a newer one instead
        # of leaving the connection bar disabled/unresponsive until it
        # finally times out -- see _connect_wifi/_poll_wifi_connect.
        self._wifi_connect_generation = 0

        self._build_connection_bar()
        self._build_controls()
        self._build_log_panel()
        self._setup_keybindings()
        self.bind_all("<Button-1>", self._maybe_reclaim_focus, add="+")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # The board may already be sitting in WiFi mode on its own (see
        # rc_main.py's boot-time watchdog) -- try the default host (its
        # stable UCSD-DEVICE hostname, STA_HOSTNAME) automatically on
        # launch rather than waiting for a click. This is only a guess (the
        # board may not be in WiFi mode yet, or may have fallen back to its
        # own AP instead) so a failure here is expected and silent. Serial
        # stays available as the manual option (e.g. if a transmitter is
        # actively paired and you don't want to disrupt it).
        self.mode_var.set("wifi")
        self._on_mode_select()
        self.focus_set()

    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")  # the platform default (e.g. macOS's
            # "aqua") ignores most of the color options below.
        except tk.TclError:
            pass
        style.configure(".", background=PALETTE["bg"], foreground=PALETTE["text"],
                         font=("Helvetica", 11))
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure("TLabel", background=PALETTE["bg"])
        style.configure("TRadiobutton", background=PALETTE["bg"])
        style.configure("TButton", padding=6)
        style.configure("Joint.TButton", font=("Helvetica", 13), padding=(10, 6))
        style.configure("Accent.TButton", background=PALETTE["accent"], foreground="white")
        style.map("Accent.TButton",
                  background=[("active", PALETTE["accent_active"]), ("disabled", PALETTE["border"])])
        style.configure("Danger.TButton", background=PALETTE["danger"], foreground="white")
        style.map("Danger.TButton",
                  background=[("active", PALETTE["danger_active"])])
        style.configure("JointName.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("KeyHint.TLabel", foreground=PALETTE["muted"], font=("Helvetica", 9))
        style.configure("SectionHeading.TLabel", foreground=PALETTE["muted"],
                         font=("Helvetica", 10, "bold"))

    def _build_connection_bar(self):
        bar = ttk.Frame(self, padding=(16, 12))
        bar.pack(fill="x")

        self.mode_var = tk.StringVar(value="")
        self.serial_radio = ttk.Radiobutton(
            bar, text="Serial", variable=self.mode_var, value="serial",
            command=self._on_mode_select,
        )
        self.wifi_radio = ttk.Radiobutton(
            bar, text="WiFi", variable=self.mode_var, value="wifi",
            command=self._on_mode_select,
        )
        self.serial_radio.pack(side="left")
        self.wifi_radio.pack(side="left", padx=(10, 12))

        # Only shown in Serial mode -- see _update_transport_controls_visibility.
        self.port_var = tk.StringVar(value=find_default_port())
        self.port_combo = ttk.Combobox(
            bar,
            textvariable=self.port_var,
            width=20,
            values=list_usb_ports(),
            postcommand=self._refresh_ports,
        )
        self.port_combo.bind("<<ComboboxSelected>>", self._on_port_change)

        # Only shown in WiFi mode -- defaults to the board's stable UCSD-DEVICE
        # hostname (STA_HOSTNAME) rather than its DHCP IP; edit to AP_FIXED_IP
        # if it fell back to hosting its own AP instead.
        self.host_var = tk.StringVar(value=STA_HOSTNAME)
        self.host_entry = ttk.Entry(bar, textvariable=self.host_var, width=26)
        self.host_entry.bind("<Return>", lambda e: self._connect_wifi(self.host_var.get()))
        self.connect_btn = ttk.Button(
            bar, text="Connect", style="Accent.TButton",
            command=lambda: self._connect_wifi(self.host_var.get()),
        )

        # Only shown in WiFi mode -- off by default (STA). Flips the Host
        # field between STA_HOSTNAME and the fixed AP_FIXED_IP and tries
        # connecting there; the GUI still never configures WiFi itself, so
        # the log just reminds you to join the board's own network yourself.
        self.ap_mode_var = tk.BooleanVar(value=False)
        self.ap_mode_check = ttk.Checkbutton(
            bar, text="AP Mode", variable=self.ap_mode_var, command=self._on_ap_mode_toggle
        )

        # Only shown once a WiFi (AP) link is up -- see _update_sta_button_visibility.
        self.sta_btn = ttk.Button(
            bar, text="Join UCSD (STA)", command=self._join_sta
        )

        ttk.Button(bar, text="STOP ALL", style="Danger.TButton", command=self._stop_all).pack(side="right")

    def _refresh_ports(self):
        self.port_combo["values"] = list_usb_ports()

    def _resolve_port(self):
        """Re-scan for USB ports and swap in a live one if the selected path
        no longer exists -- macOS assigns a new /dev path on every USB
        reconnect, so a port picked before a power-cycle can go stale."""
        ports = list_usb_ports()
        self.port_combo["values"] = ports
        port = self.port_var.get()
        if port not in ports:
            port = pick_default_port(ports)
            self.port_var.set(port)
        return port

    def _update_transport_controls_visibility(self):
        # Serial exposes the port picker; WiFi exposes the host field + its
        # own Connect button (typing a new host doesn't auto-connect) plus
        # the AP Mode toggle.
        mode = self.mode_var.get()
        if mode == "serial":
            self.host_entry.pack_forget()
            self.connect_btn.pack_forget()
            self.ap_mode_check.pack_forget()
            self.port_combo.pack(side="left", padx=4, after=self.wifi_radio)
        elif mode == "wifi":
            self.port_combo.pack_forget()
            self.host_entry.pack(side="left", padx=4, after=self.wifi_radio)
            self.connect_btn.pack(side="left", after=self.host_entry)
            self.ap_mode_check.pack(side="left", padx=(8, 0), after=self.connect_btn)
        else:
            self.port_combo.pack_forget()
            self.host_entry.pack_forget()
            self.connect_btn.pack_forget()
            self.ap_mode_check.pack_forget()

    def _update_sta_button_visibility(self):
        # Only useful when we're on the AP fallback -- if wifi_bridge.py
        # already came up on STA directly, there's nothing to "join".
        if self.wifi_kind == "ap" and isinstance(self.link, CyberBrickWifiLink):
            self.sta_btn.pack(side="left", padx=(8, 0), after=self.ap_mode_check)
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
        """Lock down the mode/port/host controls while a connect attempt
        (Serial, WiFi, or an STA join) is in flight."""
        state = "disabled" if busy else "normal"
        self.serial_radio.config(state=state)
        self.wifi_radio.config(state=state)
        self.port_combo.config(state=state)
        self.host_entry.config(state=state)
        self.connect_btn.config(state=state)
        self.ap_mode_check.config(state=state)
        self.sta_btn.config(state=state)

    def _drop_link(self, reset_board=True):
        """reset_board=False closes the current link without telling the
        board to reset out of WiFi mode -- for reconnect/mode-switch, where
        a mid-flight reset would race the new connection attempt (see
        CyberBrickWifiLink.close). Only a real disconnect (window close, or
        an empty host/port) should reset the board."""
        if self.link is not None:
            if isinstance(self.link, CyberBrickWifiLink):
                self.link.close(reset_board=reset_board)
            else:
                self.link.close()
            self.link = None
        self.wifi_kind = None
        self.ap_ip = None

    def _fall_back_to_idle(self):
        self._drop_link()
        self.mode_var.set("")
        self._update_transport_controls_visibility()
        self._update_sta_button_visibility()
        self._log("Disconnected (power-cycle board for RC mode)", level="info")

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
        # reset_board=False: if we're arriving here from a live WiFi link,
        # resetting it here would race the raw REPL entry below -- raw REPL
        # entry (Ctrl-C, Ctrl-A) already interrupts wifi_bridge.py's asyncio
        # loop on its own, so no reset is needed to take over the board.
        self._drop_link(reset_board=False)
        self._set_busy(True)
        self._log(f"Connecting to {port}...", level="connecting")
        self.update_idletasks()  # CyberBrickLink() below blocks -- paint first
        try:
            self.link = CyberBrickLink(port)
            self._log(f"Connected via Serial ({port}, remote inactive)", level="connected")
            self._sync_gripper_state()
        except Exception as e:
            self._log(f"Serial connection failed: {e}", level="error")
            self._fall_back_to_idle()
        finally:
            self._set_busy(False)

    def _connect_wifi(self, host, silent=False):
        """Pure TCP client of wifi_bridge.py -- no USB, no board-side
        configuration. `host` is whatever the board is already reachable
        at (looked up by the caller, e.g. off a router's DHCP client list
        for STA, or the fixed AP_FIXED_IP after joining the board's own
        AP network). `silent` suppresses the failure log entry's severity
        (kept at "warn" instead of "error"), for the opportunistic
        auto-attempt on launch/mode-select where the default host is just
        a guess -- the status label still reflects the outcome either way."""
        host = host.strip()
        if not host:
            if not silent:
                self._log("Enter the board's IP address or hostname.", level="warn")
            self._fall_back_to_idle()
            return
        # reset_board=False: if we're already connected via WiFi (e.g.
        # re-Connecting to a different host, or a stale link), tearing the
        # old socket down with a RESET would reboot the board out of WiFi
        # mode before the new connection attempt below even starts.
        self._drop_link(reset_board=False)
        # No _set_busy(True) here (unlike Serial): a WiFi attempt can take
        # up to ~20s to time out across its retries, and disabling the
        # connection bar for that whole window (e.g. after toggling AP Mode
        # before you've actually joined that network) reads as the GUI
        # having frozen, with no way to back out. Instead every call here
        # bumps the generation counter, and a stale attempt's result is
        # just discarded once a newer one has superseded it.
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
            # Superseded by a newer connect attempt (e.g. AP Mode toggled
            # again, or Connect clicked again, before this one resolved) --
            # drop the result instead of clobbering whatever the newer
            # attempt already set up.
            if status == "ok":
                payload.close(reset_board=False)
            return

        if status == "err":
            if silent:
                # Just an opportunistic guess (launch or mode-select) --
                # stay in WiFi mode with the Host field visible so retrying
                # only needs an edit + Connect, not re-clicking the radio.
                self._log(
                    f"Couldn't reach {host}:{WIFI_PORT} -- edit Host and Connect", level="warn"
                )
                self._update_sta_button_visibility()
            else:
                self._fall_back_to_idle()
                self._log(
                    f"Couldn't reach {host}:{WIFI_PORT} -- check the board is powered, "
                    f"already in WiFi mode, and reachable ({payload})",
                    level="error",
                )
            return

        self.link = payload
        # The board doesn't tell us which one came up -- guess from the
        # address: the fixed AP address means AP, anything else means STA.
        self.wifi_kind = "ap" if host == AP_FIXED_IP else "sta"
        self.ap_ip = host if self.wifi_kind == "ap" else None
        label = f"AP {host}" if self.wifi_kind == "ap" else f"STA at {host}"
        self._log(f"Connected via WiFi ({label}, remote inactive)", level="connected")
        self._sync_gripper_state()
        self._update_sta_button_visibility()

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
            self._log(
                f"STA connect failed: {payload} (still on WiFi at {self.ap_ip})", level="error"
            )
            return

        ip = payload
        self._log(
            f"Connected via WiFi (AP {self.ap_ip} + UCSD STA {ip}) -- also reachable at "
            f"{ip}:{WIFI_PORT}, no need to stay in AP range",
            level="connected",
        )

    def _on_close(self):
        if self.link:
            self.link.close()
        self.destroy()

    def _build_controls(self):
        body = ttk.Frame(self, padding=(16, 4, 16, 12))
        body.pack(fill="x")

        self.base_control = JointControl(lambda s: self._motor(BASE_MOTOR, s), MOTOR_SPEED)
        self.upper_control = JointControl(lambda s: self._servo_speed(UPPER_ARM_SERVO, s), JOINT_SPEED)
        self.lower_control = JointControl(lambda s: self._servo_speed(LOWER_ARM_SERVO, s), JOINT_SPEED)

        self._build_joint_row(body, "Base", self.base_control, row=0, key_hint=", / .")
        self._build_joint_row(body, "Upper Arm", self.upper_control, row=1, key_hint="W / S  ·  ↑ / ↓")
        self._build_joint_row(body, "Lower Arm", self.lower_control, row=2, key_hint="A / D  ·  ← / →")

        gripper_row = ttk.Frame(body)
        gripper_row.grid(row=3, column=0, pady=(18, 0), sticky="w")
        ttk.Label(gripper_row, text="Gripper", width=12, style="JointName.TLabel").pack(side="left")
        self.gripper_btn = ttk.Button(
            gripper_row,
            text="Close Clamp",
            command=self._toggle_gripper,
            width=16,
        )
        self.gripper_btn.pack(side="left", padx=4)
        ttk.Label(gripper_row, text="Space", style="KeyHint.TLabel").pack(side="left", padx=(10, 0))

    def _build_joint_row(self, parent, label, control, row, key_hint):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, pady=6, sticky="w")
        ttk.Label(frame, text=label, width=12, style="JointName.TLabel").pack(side="left")

        neg_btn = ttk.Button(frame, text="◀", width=4, style="Joint.TButton")  # hold to move one way
        pos_btn = ttk.Button(frame, text="▶", width=4, style="Joint.TButton")  # hold to move other way

        neg_btn.bind("<ButtonPress-1>", lambda e: control.press(-1, "mouse"))
        neg_btn.bind("<ButtonRelease-1>", lambda e: control.release(-1, "mouse"))
        pos_btn.bind("<ButtonPress-1>", lambda e: control.press(1, "mouse"))
        pos_btn.bind("<ButtonRelease-1>", lambda e: control.release(1, "mouse"))

        neg_btn.pack(side="left", padx=4)
        pos_btn.pack(side="left", padx=4)
        ttk.Label(frame, text=key_hint, style="KeyHint.TLabel").pack(side="left", padx=(10, 0))

    def _build_log_panel(self):
        frame = ttk.Frame(self, padding=(16, 0, 16, 12))
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Log", style="SectionHeading.TLabel").pack(anchor="w", pady=(0, 4))

        text_frame = ttk.Frame(frame)
        text_frame.pack(fill="both", expand=True)

        mono_font = ("Menlo" if sys.platform == "darwin" else "Consolas", 11)
        self.log_text = tk.Text(
            text_frame,
            height=7,
            wrap="word",
            state="disabled",
            font=mono_font,
            background=PALETTE["log_bg"],
            foreground=PALETTE["log_fg"],
            insertbackground=PALETTE["log_fg"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            padx=8,
            pady=6,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for level, color in {
            "info": "#9aa4b2",
            "connecting": "#e0a640",
            "connected": "#7ee08a",
            "warn": "#e0a640",
            "error": "#ff7b72",
        }.items():
            self.log_text.tag_config(level, foreground=color)

    def _log(self, message, level="info"):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {message}\n", level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    # -- Keyboard controls ---------------------------------------------

    def _setup_keybindings(self):
        joint_controls = {
            "base": self.base_control,
            "upper": self.upper_control,
            "lower": self.lower_control,
        }
        for keysym, (joint, direction) in KEY_JOINTS.items():
            control = joint_controls[joint]
            source = f"key:{keysym}"
            self._bind_hold_key(
                keysym,
                lambda c=control, d=direction, s=source: c.press(d, s),
                lambda c=control, d=direction, s=source: c.release(d, s),
            )
        self._bind_hold_key(GRIPPER_KEY, self._toggle_gripper, lambda: None)

    def _bind_hold_key(self, keysym, on_press, on_release):
        """Debounces OS key-autorepeat (which fires KeyRelease/KeyPress
        pairs while a key is held) into a single press + single release:
        a release is delayed slightly, and cancelled if the same key is
        pressed again before it fires."""

        def handle_press(event):
            if self._focused_widget_wants_text():
                return
            pending = self._key_release_after.pop(keysym, None)
            if pending is not None:
                self.after_cancel(pending)
            if keysym in self._pressed_keys:
                return  # already down -- this is autorepeat
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
        """Keeps keyboard focus on the window (so WASD keeps working) after
        clicking anything that isn't a text field -- ttk buttons can grab
        focus on click, which would otherwise silently swallow the next
        keypress instead of driving a joint."""
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox)):
            return
        self.after_idle(self.focus_set)

    # --------------------------------------------------------------------

    def _require_link(self):
        if self.link is None:
            self._log("Connect to the board first.", level="warn")
            return False
        return True

    def _motor(self, idx, speed):
        # No blocking dialog on failure here: these run from hold handlers
        # (mouse press/release or the keyboard equivalent), and a modal on
        # press could swallow the release, leaving the stop command unsent
        # and the joint moving. _log only logs -- it never blocks.
        if not self._require_link():
            return
        try:
            if speed == 0:
                self.link.stop_motor(idx)
            else:
                self.link.set_motor_speed(idx, speed)
        except Exception as e:
            self._log(f"Command failed: {e}", level="error")

    def _servo_speed(self, idx, speed):
        if not self._require_link():
            return
        try:
            if speed == 0:
                self.link.stop_servo(idx)
            else:
                self.link.set_servo_speed(idx, speed)
        except Exception as e:
            self._log(f"Command failed: {e}", level="error")

    def _sync_gripper_state(self):
        """Force the clamp open and reset tracked state to match, right
        after connecting -- self.gripper_open's initial guess has no idea
        what position the board actually booted/was left in, so without
        this the button's label can be wrong until clicked twice."""
        try:
            self.link.set_servo_angle(GRIPPER_SERVO, GRIPPER_OPEN_ANGLE)
        except Exception:
            return
        self.gripper_open = True
        self.gripper_btn.config(text="Close Clamp")

    def _toggle_gripper(self):
        if not self._require_link():
            return
        try:
            angle = GRIPPER_CLOSED_ANGLE if self.gripper_open else GRIPPER_OPEN_ANGLE
            self.link.set_servo_angle(GRIPPER_SERVO, angle)
            self.gripper_open = not self.gripper_open
            self.gripper_btn.config(
                text="Open Clamp" if not self.gripper_open else "Close Clamp"
            )
        except Exception as e:
            self._log(f"Command failed: {e}", level="error")

    def _stop_all(self):
        if self.link is None:
            return
        try:
            self.link.stop_all()
            self._log("Stopped all joints.", level="info")
        except Exception as e:
            self._log(f"Command failed: {e}", level="error")


if __name__ == "__main__":
    app = ControllerApp()
    app.mainloop()
