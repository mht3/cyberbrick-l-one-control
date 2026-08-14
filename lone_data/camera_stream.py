"""Network bridge for CameraStream, splitting capture and consumption across machines.

CameraStreamSender runs alongside a local CameraStream and pushes its frames to a
CameraStreamReceiver elsewhere. CameraStreamReceiver exposes the same start()/stop()/
get_latest()/measured_fps/actual_width/actual_height interface as CameraStream (see
camera.py), so it's a drop-in swap wherever CameraStream is used today.

Wire format, one TCP connection: a 16-byte header --
    struct.Struct(">Idi")  # payload length, sender's monotonic timestamp, sender's seq
-- followed by that many bytes of JPEG data. seq/timestamp are passed through from the
sender's CameraStream unchanged, so the receiver's fresh-vs-repeat-frame semantics match
the sender's actual capture rate rather than the network's delivery rate.

The connection is always initiated by the sender (see CameraStreamSender) -- the
receiver only binds and listens. Only one sender is served at a time.
"""

import socket
import struct
import threading
import time

import cv2
import numpy as np

_HEADER = struct.Struct(">Idi")
_FPS_WINDOW = 60  # matches camera.py's rolling rate estimate window

_IPPROTO_TCP = socket.IPPROTO_TCP
_TCP_NODELAY = socket.TCP_NODELAY


class CameraStreamSender:
    """Streams a local CameraStream's frames to (host, port), reconnecting on drop.

    Never blocks the capture side: `camera` keeps running independently of connection
    state, and a dead/reconnecting link just means frames aren't sent for a while, not
    that capture stops. Only ever sends the newest available frame -- if a send falls
    behind, older frames are dropped rather than queued, so the link can't build up
    latency it then has to work off.
    """

    def __init__(self, camera, host, port, jpeg_quality=80, retry_interval=1.5, connect_timeout=5):
        self._camera = camera
        self._host = host
        self._port = port
        self._quality = jpeg_quality
        self._retry_interval = retry_interval
        self._connect_timeout = connect_timeout

        self._running = False
        self._thread = None
        self._sock = None
        self._last_sent_seq = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            try:
                sock = socket.create_connection((self._host, self._port), timeout=self._connect_timeout)
                sock.setsockopt(_IPPROTO_TCP, _TCP_NODELAY, 1)
                self._sock = sock
                print(f"[STREAM] connected to {self._host}:{self._port}")
                self._send_loop(sock)
            except OSError as e:
                # self._running False means stop() closed our socket out from under us --
                # that's a clean shutdown, not a real connect/link failure, so stay quiet.
                if self._running:
                    print(f"[STREAM] link error: {e} -- retrying in {self._retry_interval}s")
            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
            if self._running:
                time.sleep(self._retry_interval)

    def _send_loop(self, sock):
        while self._running:
            latest = self._camera.get_latest()
            if latest is None or latest[2] == self._last_sent_seq:
                time.sleep(0.005)
                continue
            frame, ts, seq = latest
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                continue
            payload = buf.tobytes()
            sock.sendall(_HEADER.pack(len(payload), ts, seq) + payload)
            self._last_sent_seq = seq

    def stop(self):
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class CameraStreamReceiver:
    """Listens for a CameraStreamSender and exposes CameraStream's interface.

    start() binds and spawns an accept loop -- it does not block waiting for a sender
    to connect, matching CameraStream.start()'s non-blocking contract. actual_width/
    actual_height stay 0 until the first frame decodes (see _check_storage_aspect in
    collect_data.py, which already tolerates that). If the sender disconnects, the
    accept loop just waits for the next connection; get_latest() keeps returning the
    last frame received in the meantime.
    """

    def __init__(self, port, bind_host="0.0.0.0"):
        self._port = port
        self._bind_host = bind_host

        self.actual_width = 0
        self.actual_height = 0

        self._lock = threading.Lock()
        self._latest = None  # (frame, sender_ts, sender_seq)
        self._recent = []  # monotonic receive timestamps of the last _FPS_WINDOW frames

        self._running = False
        self._server_sock = None
        self._accept_thread = None

    def start(self):
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._bind_host, self._port))
        self._server_sock.listen(1)
        self._server_sock.settimeout(1.0)  # lets stop() unblock accept()
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        print(f"[STREAM] listening on {self._bind_host}:{self._port}")

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.setsockopt(_IPPROTO_TCP, _TCP_NODELAY, 1)
            print(f"[STREAM] sender connected: {addr}")
            self._read_loop(conn)

    def _read_loop(self, conn):
        buf = b""
        try:
            while self._running:
                while len(buf) < _HEADER.size:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                length, ts, seq = _HEADER.unpack(buf[: _HEADER.size])
                buf = buf[_HEADER.size :]
                while len(buf) < length:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                payload, buf = buf[:length], buf[length:]
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self._lock:
                    self.actual_height, self.actual_width = frame.shape[:2]
                    self._latest = (frame, ts, seq)
                    now = time.monotonic()
                    self._recent.append(now)
                    if len(self._recent) > _FPS_WINDOW:
                        del self._recent[: len(self._recent) - _FPS_WINDOW]
        except OSError as e:
            print(f"[STREAM] sender disconnected: {e}")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def get_latest(self):
        """Returns (frame, sender_monotonic_ts, sender_seq), or None if nothing received yet."""
        with self._lock:
            return self._latest

    @property
    def measured_fps(self):
        """Rolling receive rate over the last _FPS_WINDOW frames."""
        with self._lock:
            recent = self._recent
            if len(recent) < 2:
                return 0.0
            span = recent[-1] - recent[0]
            return (len(recent) - 1) / span if span > 0 else 0.0

    def stop(self):
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None
