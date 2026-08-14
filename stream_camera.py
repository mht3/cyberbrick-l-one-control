#!/usr/bin/env python3
"""Streams this machine's webcam to a CameraStreamReceiver running elsewhere.

Pairs with `collect_data.py --remote-camera` (or any other CameraStreamReceiver
consumer) running on another machine, e.g. one with the GPU for model inference.
This process dials out to that machine and reconnects on drop -- it never listens
for inbound connections itself, so it works from behind a WiFi network that only
allows outbound connections.
"""

import argparse
import time

from lone_data.camera import CameraStream
from lone_data.camera_stream import CameraStreamSender


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera-index", type=int, default=0, help="cv2.VideoCapture index")
    p.add_argument("--width", type=int, default=1280, help="requested camera capture width")
    p.add_argument("--height", type=int, default=720, help="requested camera capture height")
    p.add_argument("--camera-fps", type=int, default=30, help="requested camera capture fps")
    p.add_argument("--host", required=True, help="receiver hostname/IP, e.g. fusion.ucsd.edu")
    p.add_argument("--port", type=int, default=8267, help="receiver's --remote-camera-port")
    p.add_argument("--jpeg-quality", type=int, default=80, help="cv2 JPEG quality, 0-100")
    return p.parse_args()


def main():
    args = parse_args()
    camera = CameraStream(args.camera_index, args.width, args.height, args.camera_fps)
    camera.start()
    sender = CameraStreamSender(camera, args.host, args.port, jpeg_quality=args.jpeg_quality)
    sender.start()
    print(f"[STREAM] streaming camera {args.camera_index} to {args.host}:{args.port} (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
            print(f"[STREAM] camera capturing at {camera.measured_fps:.1f} fps")
    except KeyboardInterrupt:
        pass
    finally:
        sender.stop()
        camera.stop()


if __name__ == "__main__":
    main()
