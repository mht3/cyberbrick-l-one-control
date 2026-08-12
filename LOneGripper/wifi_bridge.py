# -*-coding:utf-8-*-
"""Wireless command bridge for the L-ONE arm.

This is NOT part of the stock RC firmware and is not run automatically
at boot -- it replaces the normal main.py program for as long as you
run it. While it's running, the board is not listening for the RC
transmitter's ESP-NOW link, so the physical remote will not respond.

On start, this tries STA_SSID/STA_PASSWORD (wifi_secrets.py) first -- e.g.
UCSD-DEVICE -- so the board just joins your existing network and prints
whatever IP DHCP hands it (not static -- read it fresh each boot). Only
if that's unavailable/fails does it fall back to hosting its own AP
(AP_SSID/AP_PASSWORD, fixed IP 192.168.4.1) as a last resort -- the AP
does NOT broadcast by default, only when STA isn't reachable.

Setup:
  1. Fill in wifi_secrets.py (same folder) with STA_SSID/STA_PASSWORD
     (and AP_SSID/AP_PASSWORD for the fallback case).
  2. Upload both files to the board (Pymakr: right-click -> upload, or
     drag into the device file browser).
  3. Run it: in the Pymakr terminal (connected, at the >>> prompt):
       import wifi_bridge
       wifi_bridge.main()
     It will print whichever came up -- "[WIFI] STA up: ..." with its
     DHCP IP, or "[WIFI] AP up: ..." with 192.168.4.1 if STA failed. Use
     that IP (joining the AP's own network first, if that's the one that
     came up) with virtual_gripper.py's Port/host:port field.
  4. To go back to normal RC/remote operation, send RESET (the
     "Disconnect" button in virtual_gripper.py does this automatically)
     or power-cycle the board.

Protocol: newline-terminated ASCII commands, one reply line each.
  MOTOR <idx 1-2> <speed -2048..2048>
  SERVO_SPEED <idx 1-4> <speed -100..100>
  SERVO_ANGLE <idx 1-4> <angle 0..180>
  STOP
  STA_CONNECT
  RESET
Replies: "OK" or "ERR <message>", except STA_CONNECT which replies
"OK <ip>" on success.

STA_CONNECT joins STA_SSID/STA_PASSWORD (wifi_secrets.py) *alongside*
whatever's already up (used when the board fell back to AP and you want
to also add STA without dropping the AP link you're issuing the command
over) -- ESP32 supports AP+STA concurrently. Once joined, the server is
reachable at either address (it's already bound to 0.0.0.0).

RESET also deletes /wifi_mode.flag before resetting, so the board comes
back up as the normal RC firmware instead of straight back into this
bridge (see boot.py).
"""

import sys
import time

import machine
import network
import uasyncio as asyncio
import uos

sys.path.append('/bbl')  # leds.py lives here; boot.py only adds /app for us

from control import MotorsControllerExecMapper, ServosControllerExecMapper
from leds import LEDController
from wifi_secrets import AP_PASSWORD, AP_SSID, STA_PASSWORD, STA_SSID

PORT = 8266
WIFI_MODE_FLAG = '/wifi_mode.flag'  # must match boot.py
STATUS_LED_WIFI = 0x8000FF  # solid purple while running WiFi bridge mode
# Deadman: if no command arrives for this long (e.g. the WiFi link drops
# mid-move without a clean TCP close, so readline() would otherwise just
# block forever), zero all outputs rather than hold the last speed forever.
COMMAND_DEADMAN_TIMEOUT = 0.5

servos = ServosControllerExecMapper()
motors = MotorsControllerExecMapper()

# Set while a client is connected, to the ticks_ms() of its last command;
# None means no client is currently connected. Read/written by
# handle_client() and _deadman_watchdog() -- see the latter for why this
# is a polled timestamp rather than a per-read timeout.
_last_command_ms = None


def _set_status_led(rgb):
    for channel in ("LED1", "LED2"):
        led = LEDController(channel)
        led.set_led_effect(0, 0, 0xFF, 0x0F, rgb)
        led.timing_proc()


def start_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=AP_SSID, password=AP_PASSWORD, security=network.WLAN.SEC_WPA2)
    while not ap.active():
        time.sleep(0.1)
    ip = ap.ifconfig()[0]
    # Parsed by virtual_gripper.py's configure_wifi_bridge() -- keep the format in sync.
    print(f"[WIFI] AP up: ssid={AP_SSID} password={AP_PASSWORD} ip={ip}")
    return ap


def connect_sta(timeout=15):
    """Join STA_SSID. Safe to call whether or not the AP is already up --
    ESP32 handles AP+STA concurrently, sharing one radio, and this doesn't
    touch the AP interface either way."""
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    if not sta.isconnected():
        sta.connect(STA_SSID, STA_PASSWORD)
        deadline = time.time() + timeout
        while not sta.isconnected() and time.time() < deadline:
            time.sleep(0.5)
    if not sta.isconnected():
        raise RuntimeError("STA connect failed/timed out")
    ip = sta.ifconfig()[0]
    print(f"[WIFI] STA up: ssid={STA_SSID} ip={ip}")
    return ip


def handle_command(line):
    parts = line.split()
    if not parts:
        return "ERR empty command"
    cmd = parts[0].upper()
    try:
        if cmd == "MOTOR":
            motors.set_speed(int(parts[1]), int(parts[2]))
        elif cmd == "SERVO_SPEED":
            servos.set_speed(int(parts[1]), int(parts[2]))
        elif cmd == "SERVO_ANGLE":
            servos.set_angle(int(parts[1]), int(parts[2]))
        elif cmd == "STOP":
            motors.stop(1)
            motors.stop(2)
            servos.stop(1)
            servos.stop(2)
        elif cmd == "STA_CONNECT":
            try:
                ip = connect_sta()
            except Exception as e:
                return f"ERR STA connect failed: {e}"
            return f"OK {ip}"
        elif cmd == "RESET":
            try:
                uos.remove(WIFI_MODE_FLAG)
            except OSError:
                pass
            return "RESET"
        else:
            return f"ERR unknown command {cmd}"
    except (IndexError, ValueError) as e:
        return f"ERR bad args: {e}"
    return "OK"


async def handle_client(reader, writer):
    global _last_command_ms
    peer = writer.get_extra_info("peername")
    print("[WIFI] client connected:", peer)
    _last_command_ms = time.ticks_ms()
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            _last_command_ms = time.ticks_ms()
            reply = handle_command(line.decode().strip())
            if reply == "RESET":
                writer.write(b"OK\n")
                await writer.drain()
                await writer.wait_closed()
                time.sleep(0.2)
                machine.reset()
            writer.write((reply + "\n").encode())
            await writer.drain()
    except Exception as e:
        print("[WIFI] client error:", e)
    finally:
        _last_command_ms = None
        motors.stop(1)
        motors.stop(2)
        servos.stop(1)
        servos.stop(2)
        await writer.wait_closed()
        print("[WIFI] client disconnected, all stopped")


async def _deadman_watchdog():
    """Zeros all outputs if a connected client goes quiet for too long
    (e.g. the WiFi link drops mid-move without a clean TCP close).

    Deliberately doesn't touch reader.readline() itself -- an earlier
    version wrapped that read in asyncio.wait_for(), which meant cancelling
    it on every idle tick (the common case: a command only goes out when
    the commanded speed actually changes, so most of a session is idle).
    Repeatedly cancelling a uasyncio StreamReader.readline() isn't safe to
    do that often and was implicated in frequent command timeouts. This
    polls a plain timestamp instead, so the read itself is never disturbed.
    """
    global _last_command_ms
    while True:
        await asyncio.sleep(0.25)
        if _last_command_ms is None:
            continue
        if time.ticks_diff(time.ticks_ms(), _last_command_ms) > int(COMMAND_DEADMAN_TIMEOUT * 1000):
            motors.stop(1)
            motors.stop(2)
            servos.stop(1)
            servos.stop(2)
            _last_command_ms = None  # already stopped -- don't re-stop every tick


async def _main():
    _set_status_led(STATUS_LED_WIFI)
    try:
        connect_sta()
    except Exception as e:
        print(f"[WIFI] STA unavailable ({e}), falling back to AP.")
        start_ap()
    server = await asyncio.start_server(handle_client, "0.0.0.0", PORT)
    print(f"[WIFI] listening on port {PORT}")
    asyncio.create_task(_deadman_watchdog())
    async with server:
        await server.wait_closed()


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
