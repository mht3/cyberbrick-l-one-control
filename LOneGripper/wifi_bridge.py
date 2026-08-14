# -*-coding:utf-8-*-
"""Wireless command bridge for the L-ONE arm.
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
from wifi_secrets import AP_PASSWORD, AP_SSID, STA_HOSTNAME, STA_PASSWORD, STA_SSID

PORT = 8266
WIFI_MODE_FLAG = '/wifi_mode.flag'  # must match boot.py
STATUS_LED_WIFI = 0x8000FF  # solid purple while running WiFi bridge mode
# Deadman: if no command arrives for this long (e.g. the WiFi link drops
# mid-move without a clean TCP close, so readline() would otherwise just
# block forever), zero all outputs rather than hold the last speed forever.
#
# Sized against measured round-trip time, not guessed. On the routed campus
# path (host and board on different subnets) RTT was 73/378/1116 ms
# min/avg/max with 17% loss, so the old 0.5s fired constantly *during normal
# teleop* and zeroed the arm mid-move. 2.0s clears the observed p99 with
# margin. The cost is that a genuinely dead link now coasts for up to 2s;
# that is covered by the host sending an explicit STOP on clean disconnect
# and by handle_client()'s finally block zeroing everything on socket close.
COMMAND_DEADMAN_TIMEOUT = 2.0

# lwIP numeric fallbacks -- MicroPython builds vary in which of these the
# socket module actually exports.
_IPPROTO_TCP = 6
_TCP_NODELAY = 1

servos = ServosControllerExecMapper()
motors = MotorsControllerExecMapper()

# Set while a client is connected, to the ticks_ms() of its last command;
# None means no client is currently connected. Read/written by
# handle_client() and _deadman_watchdog() -- see the latter for why this
# is a polled timestamp rather than a per-read timeout.
_last_command_ms = None

# The writer of the currently-served client, so a new connection can evict a
# stale one. asyncio.start_server() spawns an unbounded handle_client() task
# per connection and they all share _last_command_ms; with packet loss on the
# campus path, half-open sessions from a previous run are common, and the old
# task's finally block stops every motor -- under the live session.
_active_writer = None


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


def _disable_power_save(sta):
    """Turn off WiFi modem power save.

    MicroPython's ESP32 port defaults the station to WIFI_PS_MIN_MODEM, which
    parks the radio between DTIM beacons and leaves downlink packets sitting in
    the AP's buffer. For a request/response teleop protocol that shows up as a
    latency *floor* of tens of milliseconds plus periodic multi-hundred-ms
    stalls -- measured here as 73ms best case, 378ms average, 1116ms worst.
    PM_NONE keeps the receiver awake; the board is USB/battery powered on a
    desk, so the extra draw doesn't matter.

    The core board runs a custom MicroPython build, so neither the constant nor
    the config key is guaranteed to exist -- report and carry on rather than
    failing to come up at all.
    """
    for pm in (getattr(network.WLAN, 'PM_NONE', None), 0):
        if pm is None:
            continue
        try:
            sta.config(pm=pm)
            print(f"[WIFI] power save disabled (pm={pm})")
            return True
        except (AttributeError, OSError, ValueError) as e:
            err = e
    print(f"[WIFI] could not disable power save ({err}) -- expect high latency")
    return False


def _dhcp_label():
    """The DHCP hostname to claim, derived from STA_HOSTNAME.

    STA_HOSTNAME is what the desktop connects to and may be a fully qualified name
    (board.dynamic.example.edu); DHCP wants only the first label. If it is an IP
    address there is no name to register, so return None.
    """
    name = (STA_HOSTNAME or '').strip()
    if not name:
        return None
    label = name.split('.')[0]
    if not label or label.isdigit():  # an IP was configured, not a name
        return None
    return label


def _set_hostname(sta):
    """Claim a stable DHCP hostname before requesting a lease.

    Without this the board sends whatever its build defaults to, and a network that
    registers DHCP clients in DNS (…dynamic.ucsd.edu) publishes *that* name -- which
    is why the address resolved sometimes and returned NXDOMAIN after a lease lapsed.
    The IP stays dynamic; the name does not.

    Called before active()/connect(): the hostname rides along in the DHCP request,
    and the stricter MicroPython versions require it to be set before the interface
    comes up.

    The API moved between versions, so try each spelling. The catch is deliberately
    broad -- unknown config keys raise different types across ports, and an
    unregistered name is worth far less than a board that will not join the network
    at all. Never let this stop connect_sta().
    """
    label = _dhcp_label()
    if label is None:
        return
    for attempt in (
        lambda: network.hostname(label),            # MicroPython >= 1.20
        lambda: sta.config(dhcp_hostname=label),    # older esp32 port
        lambda: sta.config(hostname=label),         # some 1.19-era builds
    ):
        try:
            attempt()
            print(f"[WIFI] DHCP hostname set to {label}")
            return
        except Exception:
            continue
    print(f"[WIFI] could not set DHCP hostname ({label}) -- DNS name may be unstable")


def connect_sta(timeout=15):
    """Join STA_SSID. Safe to call whether or not the AP is already up --
    ESP32 handles AP+STA concurrently, sharing one radio, and this doesn't
    touch the AP interface either way."""
    sta = network.WLAN(network.STA_IF)
    _set_hostname(sta)  # before active(): stricter builds latch the name at bring-up
    sta.active(True)
    _disable_power_save(sta)
    if not sta.isconnected():
        sta.connect(STA_SSID, STA_PASSWORD)
        deadline = time.time() + timeout
        while not sta.isconnected() and time.time() < deadline:
            time.sleep(0.5)
    if not sta.isconnected():
        raise RuntimeError("STA connect failed/timed out")
    ip = sta.ifconfig()[0]
    label = _dhcp_label()
    # The IP is whatever DHCP handed out and will change; the hostname is the part
    # worth typing into the host tools, so print both and say which is stable.
    print(f"[WIFI] STA up: ssid={STA_SSID} ip={ip}"
          + (f" hostname={label} (connect to {STA_HOSTNAME})" if label else ""))
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


def _set_nodelay(writer):
    """Disable Nagle on an accepted connection.

    Every command is a short line answered by a short line, which is the exact
    shape Nagle plus the peer's delayed ACK punishes: the board holds a small
    reply waiting for more data to coalesce while the host holds its next
    command waiting for that reply.

    Best effort, and deliberately catching everything: uasyncio's
    Stream.get_extra_info() is `return self.e[v]` over a dict that only carries
    "peername", so asking it for "socket" raises KeyError -- which is not an
    OSError, escaped an earlier version of this function, and killed the
    connection before the serve loop even started. The client saw every first
    command fail with "Connection closed by board". Nagle is an optimization;
    nothing here is worth dropping a connection over. The socket itself is
    reachable as Stream.s on MicroPython.
    """
    try:
        sock = getattr(writer, "s", None) or writer.get_extra_info("socket")
        sock.setsockopt(_IPPROTO_TCP, _TCP_NODELAY, 1)
        return True
    except Exception as e:
        print("[WIFI] TCP_NODELAY unavailable:", e)
        return False


async def handle_client(reader, writer):
    global _last_command_ms, _active_writer
    peer = writer.get_extra_info("peername")
    print("[WIFI] client connected:", peer)

    # Evict whatever was being served before. Doing this here rather than in
    # the old task's finally block means the newest client always wins, and the
    # evicted task's cleanup can't stop motors this one is already driving.
    previous, _active_writer = _active_writer, writer
    if previous is not None:
        print("[WIFI] evicting previous client")
        try:
            previous.close()
        except Exception as e:
            print("[WIFI] evict failed:", e)

    _last_command_ms = time.ticks_ms()
    try:
        _set_nodelay(writer)
        while True:
            line = await reader.readline()
            if not line:
                break
            if _active_writer is not writer:
                break  # superseded by a newer connection
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
        superseded = _active_writer is not writer
        if not superseded:
            _active_writer = None
            _last_command_ms = None
            motors.stop(1)
            motors.stop(2)
            servos.stop(1)
            servos.stop(2)
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print("[WIFI] client disconnected"
              + (", superseded" if superseded else ", all stopped"))


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
