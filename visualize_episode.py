#!/usr/bin/env python3
"""Play back one episode from a LoneReplayBuffer Zarr dataset for debugging.

    python visualize_episode.py --episode 0 [--path lone_dataset.zarr]

Controls: any key to step to the next frame, 'p' to toggle auto-play at
the episode's recorded rate, 'q' to quit.
"""

import argparse

import cv2
import numpy as np

from lone_data.replay_buffer import ACTION_NAMES, LoneReplayBuffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default="datasets/lone_dataset.zarr", help="path to the Zarr dataset")
    p.add_argument("--episode", type=int, required=True, help="episode index to play back")
    return p.parse_args()


def main():
    args = parse_args()
    rb = LoneReplayBuffer(args.path)
    start, end = rb.episode_bounds(args.episode)

    imgs = rb.img[start:end]
    actions = rb.action[start:end]
    timestamps = rb.timestamp[start:end]
    ep_len = end - start

    mean_dt = np.mean(np.diff(timestamps)) if ep_len > 1 else 1.0 / 15
    delay_ms = max(1, int(mean_dt * 1000))

    print(f"Episode {args.episode}: {ep_len} steps, ~{1.0 / mean_dt:.1f} fps recorded")
    print("Any key = next frame, 'p' = toggle autoplay, 'q' = quit")

    window = "L-ONE episode playback"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    t = 0
    autoplay = False
    while 0 <= t < ep_len:
        frame = imgs[t].copy()
        action = actions[t]
        overlay_lines = [
            f"episode {args.episode}  step {t}/{ep_len - 1}",
        ] + [f"{name}: {action[i]:.1f}" for i, name in enumerate(ACTION_NAMES)]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(delay_ms if autoplay else 0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            autoplay = not autoplay
            continue
        t += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
