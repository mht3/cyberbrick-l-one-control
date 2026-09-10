#!/usr/bin/env python3
"""Streams this machine's webcams to a CameraStreamReceiver running elsewhere.

Pairs with `collect_data.py --remote-camera` (or any other CameraStreamReceiver
consumer) running on another machine, e.g. one with the GPU for model inference.
This process dials out to that machine and reconnects on drop -- it never listens
for inbound connections itself, so it works from behind a WiFi network that only
allows outbound connections.

One connection per camera view, one port each: Cam1 (front) goes to `--port` and
Cam2 (side) to the next port up, which is exactly where the receiving GUI listens
for each slot. Two views are the default when two cameras are attached; name the
indices with `--camera-index` to choose which is which, since the receiving end
takes the first as Cam1.
"""

import argparse
import time

from lone_data import camera_source
from lone_data.camera import CameraStream
from lone_data.camera_stream import CameraStreamSender
from lone_data.features import MAX_CAMERAS, camera_label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", type=int, nargs="+", default=None, metavar="INDEX",
                   help="cv2.VideoCapture index per view, Cam1 first "
                        f"(default: the first {MAX_CAMERAS} cameras that answer)")
    p.add_argument("--width", type=int, default=1280, help="requested camera capture width")
    p.add_argument("--height", type=int, default=720, help="requested camera capture height")
    p.add_argument("--camera-fps", type=int, default=30, help="requested camera capture fps")
    p.add_argument("--host", required=True, help="receiver hostname/IP, e.g. fusion.ucsd.edu")
    p.add_argument("--port", type=int, default=8267,
                   help="receiver's --remote-camera-port; Cam2 is sent to the next port up")
    p.add_argument("--jpeg-quality", type=int, default=80, help="cv2 JPEG quality, 0-100")
    return p.parse_args()


def resolve_indices(requested):
    """Which local cameras to stream, in slot order.

    Probing rather than defaulting to [0] is what makes two views the default: a
    rig with two cameras streams both without being asked, and a rig with one
    streams one. An explicit --camera-index always wins, and is the way to say
    which physical camera is the front view.
    """
    if requested:
        if len(requested) > MAX_CAMERAS:
            print(f"[STREAM] {len(requested)} indices given but there are only "
                  f"{MAX_CAMERAS} camera slots -- streaming {list(requested)[:MAX_CAMERAS]}")
        return list(requested)[:MAX_CAMERAS]
    found = [s for s in camera_source.available_sources(include_remote=False)]
    return found[:MAX_CAMERAS] or [0]


def main():
    args = parse_args()
    indices = resolve_indices(args.camera_index)

    cameras, senders = [], []
    try:
        for slot, index in enumerate(indices):
            port = camera_source.remote_port(args.port, slot)
            try:
                camera = CameraStream(index, args.width, args.height, args.camera_fps)
            except Exception as e:
                # Keep whatever did open. The receiver treats each slot separately,
                # so one missing camera costs that view and nothing else -- taking
                # down a working Cam1 because Cam2 is unplugged helps no one.
                print(f"[STREAM] {camera_label(slot)}: {e} -- skipping this view")
                continue
            camera.start()
            sender = CameraStreamSender(camera, args.host, port, jpeg_quality=args.jpeg_quality)
            sender.start()
            cameras.append(camera)
            senders.append(sender)
            print(f"[STREAM] {camera_label(slot)}: camera {index} -> {args.host}:{port}")
        if not cameras:
            print(f"[STREAM] no camera could be opened (tried {indices}) -- nothing to stream.")
            return 1
        print(f"[STREAM] streaming {len(cameras)} view(s) (Ctrl+C to stop)")
        while True:
            time.sleep(1)
            rates = "  ·  ".join(
                f"{camera_label(i)} {cam.measured_fps:.1f} fps"
                for i, cam in enumerate(cameras)
            )
            print(f"[STREAM] {rates}")
    except KeyboardInterrupt:
        pass
    finally:
        # Senders first: a sender reading from a stopped camera is a race that
        # only ever produces noise on the way out.
        for sender in senders:
            sender.stop()
        for camera in cameras:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
