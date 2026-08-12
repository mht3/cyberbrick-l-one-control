"""Human-readable metadata.json sidecar for a LoneReplayBuffer dataset.

Dropped as a plain file inside the .zarr directory, alongside data/ and
meta/ -- Zarr doesn't care about extra files there.
"""

import datetime
import json
import os
import subprocess

from lone_data.replay_buffer import ACTION_NAMES

ACTION_SEMANTICS = [
    {
        "index": 0,
        "name": "base_motor_speed",
        "channel": "MOTOR1",
        "call": "MotorsController.set_speed",
        "unit": "raw_pwm_command",
        "range": [-2048, 2048],
        "notes": (
            "Linearly mapped to 0-100% PWM duty on one of two H-bridge "
            "channels by sign; not a calibrated physical velocity."
        ),
    },
    {
        "index": 1,
        "name": "upper_arm_servo_speed",
        "channel": "PWM1",
        "call": "ServosController.set_speed",
        "unit": "percent",
        "range": [-100, 100],
        "notes": "Continuous-rotation servo speed command.",
    },
    {
        "index": 2,
        "name": "lower_arm_servo_speed",
        "channel": "PWM2",
        "call": "ServosController.set_speed",
        "unit": "percent",
        "range": [-100, 100],
        "notes": "Continuous-rotation servo speed command.",
    },
    {
        "index": 3,
        "name": "gripper_angle",
        "channel": "PWM3",
        "call": "ServosController.set_angle",
        "unit": "degrees",
        "range": [0, 180],
        "notes": "Positional servo angle command.",
    },
]

ZERO_DISPATCH_CONVENTION = (
    "For channels 0-2, an action value of 0 is dispatched via "
    "stop_motor()/stop_servo() rather than set_speed(idx, 0) -- this is "
    "the existing convention in virtual_gripper.py/control code (e.g. a "
    "motor's stop() sets both H-bridge PWM channels to duty 100, whereas "
    "set_speed(idx, 0) sets them to duty 0 -- two different hardware "
    "states). A consumer replaying this dataset on real hardware should "
    "reproduce the same dispatch, not call set_speed(idx, 0) literally."
)

SYNC_NOTE = (
    "Synchronization is zero-order-hold: each row's action is whatever "
    "command was last sent to the robot at the time its image was "
    "captured (time.monotonic() timestamps for both). Host-to-board "
    "command latency (one raw-REPL round trip over serial, or one TCP "
    "round trip over WiFi) is NOT measured or compensated for -- treat "
    "the (image, action) pairing as accurate to within that unmeasured "
    "latency, not as perfectly simultaneous."
)


def _git_info(repo_dir):
    def run(args):
        try:
            return subprocess.check_output(args, cwd=repo_dir, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None}
    dirty = run(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty)}


class DatasetMetadata:
    def __init__(self, zarr_path, repo_dir=None):
        self.path = os.path.join(zarr_path, "metadata.json")
        self.repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {
                "robot_name": "cyberbrick-l-one",
                "action_dim": len(ACTION_NAMES),
                "action_names": ACTION_NAMES,
                "action_semantics": ACTION_SEMANTICS,
                "zero_dispatch_convention": ZERO_DISPATCH_CONVENTION,
                "synchronization": SYNC_NOTE,
                "dataset_created": _now_iso(),
                "episodes": [],
            }

    def save(self):
        self.data["last_updated"] = _now_iso()
        self.data["git"] = _git_info(self.repo_dir)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, self.path)

    def set_camera_info(self, requested_w, requested_h, requested_fps, actual_w, actual_h, record_hz):
        self.data["camera"] = {
            "requested_resolution": [requested_w, requested_h],
            "actual_resolution": [actual_w, actual_h],
            "requested_fps": requested_fps,
            "record_target_hz": record_hz,
        }

    def set_default_task(self, task):
        self.data["task"] = task

    def add_episode(self, index, task, length, start_time_iso, mean_fps):
        self.data.setdefault("episodes", []).append(
            {
                "index": index,
                "task": task,
                "length": length,
                "start_time": start_time_iso,
                "mean_fps": mean_fps,
            }
        )

    def episode_task(self, index):
        for ep in self.data.get("episodes", []):
            if ep["index"] == index:
                return ep.get("task", "")
        return ""

    def set_episode_task(self, index, task):
        for ep in self.data.get("episodes", []):
            if ep["index"] == index:
                ep["task"] = task
                return

    def remove_episode(self, index):
        """Drops entry `index` and shifts later episodes' indices down, matching delete_episode()."""
        episodes = [e for e in self.data.get("episodes", []) if e["index"] != index]
        for e in episodes:
            if e["index"] > index:
                e["index"] -= 1
        self.data["episodes"] = episodes


def _now_iso():
    return datetime.datetime.now().astimezone().isoformat()
