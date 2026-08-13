"""Canonical LeRobot feature schema for L-ONE demonstrations.

L-ONE is RGB-only: one webcam, no encoders, no proprioception of any kind. The
dataset therefore has no `observation.state`, and inventing one would mean
writing fabricated sensor readings into a dataset other people may train on.

That is a real constraint at training time, not a solved one. ACT handles it --
every state use is guarded. pi0/pi0.5 do not: PI05Config.validate_features()
injects a padded OBS_STATE *feature* and fills unused camera slots, but the
feature only carries shapes and normalization wiring. At runtime
Pi05PrepareStateTokenizerProcessorStep raises ValueError("State is required for
PI05") on a batch that has no state tensor, because pi0.5 discretizes the state
into the text prompt ("Task: ..., State: <256 bins>; Action: "). No config flag
turns that off in any current lerobot release, so training pi0.5 on this data
means supplying a state from outside the dataset. Diffusion Policy and SmolVLA
read robot_state_feature unconditionally and can't train on this data at all.

Feature names come from lerobot.utils.constants because
dataset_to_policy_features() matches "action" exactly and anything starting
with "observation", then silently drops every other key -- a differently named
action column would be invisible to every LeRobot policy.
"""

import cv2

from lerobot.utils.constants import ACTION, OBS_IMAGES

CAMERA_KEY = f"{OBS_IMAGES}.front"
ROBOT_TYPE = "cyberbrick-l-one"

# (height, width). Native 16:9, matching the camera, downscaled from 1280x720.
# Frames are stored at the camera's own aspect ratio and never padded -- see
# resize_keep_aspect() for why, and note that every published LeRobot dataset
# does the same (SO-101 480x640, ALOHA 480x640, DROID 180x320).
#
# 640x360 leaves ~4x the linear resolution pi0.5 consumes at 224x224, so there
# is room to train at a higher resolution or re-crop later. DROID's 320x180 is
# the practical floor; native 1280x720 is defensible but noticeably slows the
# AV1 encode that runs at the end of every episode.
DEFAULT_IMAGE_SIZE = (360, 640)

ACTION_NAMES = [
    "base_motor_speed",
    "upper_arm_servo_speed",
    "lower_arm_servo_speed",
    "gripper_angle",
]
ACTION_DIM = len(ACTION_NAMES)

# What each action dimension physically is. Every value is a *command* sent to
# the board -- L-ONE reports nothing back, so none of these are measurements.
ACTION_SEMANTICS = [
    {
        "index": 0,
        "name": "base_motor_speed",
        "channel": "MOTOR1",
        "call": "MotorsController.set_speed",
        "unit": "raw_pwm_command",
        "range": [-2048, 2048],
        "notes": "Linearly mapped to 0-100% PWM duty on one of two H-bridge channels by sign; "
                 "not a calibrated physical velocity. Teleop emits -900/0/900.",
    },
    {
        "index": 1,
        "name": "upper_arm_servo_speed",
        "channel": "PWM1",
        "call": "ServosController.set_speed",
        "unit": "percent",
        "range": [-100, 100],
        "notes": "Continuous-rotation servo speed command. Teleop emits -100/0/100.",
    },
    {
        "index": 2,
        "name": "lower_arm_servo_speed",
        "channel": "PWM2",
        "call": "ServosController.set_speed",
        "unit": "percent",
        "range": [-100, 100],
        "notes": "Continuous-rotation servo speed command. Teleop emits -100/0/100.",
    },
    {
        "index": 3,
        "name": "gripper_angle",
        "channel": "PWM3",
        "call": "ServosController.set_angle",
        "unit": "degrees",
        "range": [0, 180],
        "notes": "Positional servo angle command. Teleop emits 30 (open) / 120 (closed).",
    },
]

ZERO_DISPATCH_CONVENTION = (
    "For dimensions 0-2, an action value of 0 was dispatched via stop_motor()/stop_servo(), "
    "NOT set_speed(idx, 0) -- a motor's stop() sets both H-bridge PWM channels to duty 100 "
    "whereas set_speed(idx, 0) sets them to duty 0, which are different hardware states. "
    "Anything replaying these actions on real hardware must reproduce that dispatch."
)

SYNC_NOTE = (
    "Zero-order hold: each frame's action is the command in effect when its image was "
    "captured. Host-to-board latency (one raw-REPL round trip over serial, or one TCP round "
    "trip over WiFi) is measured and logged per session but NOT compensated for, so treat the "
    "(image, action) pairing as accurate to within that latency rather than simultaneous."
)


def lone_features(image_size=DEFAULT_IMAGE_SIZE):
    """LeRobot feature dict. LeRobot adds timestamp/frame_index/episode_index/
    index/task_index itself, so only the robot-specific columns go here."""
    h, w = image_size
    return {
        CAMERA_KEY: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        ACTION: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ACTION_NAMES,
        },
    }


def resize_keep_aspect(frame_bgr, image_size=DEFAULT_IMAGE_SIZE):
    """Plain downscale, BGR in -> RGB uint8 out. No padding.

    This deliberately does not letterbox. pi0.5 runs resize_with_pad_torch()
    over whatever it is given, using the identical ratio formula, so padding to
    a square here is applied a second time at 224x224 and produces byte-for-byte
    the same model input -- while storing 6.2x fewer real pixels:

        store 256x256 letterboxed  -> pi0.5 224x224, live content 224x126
        store 640x360 native       -> pi0.5 224x224, live content 224x126

    43.8% of each letterboxed frame was black, and the resolution it cost is
    unrecoverable if a later run wants a higher input size or a tighter crop.
    Padding is the consuming policy's business, not the dataset's.

    Callers are expected to have matched image_size to the camera's aspect
    ratio (collect_data.py checks at startup); a mismatch distorts rather than
    crops, which is why it is checked there rather than silently handled here.
    """
    h, w = image_size
    resized = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
