"""Turn a policy's 4-vector into commands on the board.

Two problems sit between a predicted action and the arm moving, and both are easy
to get subtly wrong:

1. A value of 0 on dimensions 0-2 must be dispatched as stop_motor()/stop_servo(),
   not set_speed(idx, 0) -- see ZERO_DISPATCH_CONVENTION in features.py. The
   dataset was recorded under that convention, so replaying any other way drives
   different hardware states than the demonstrations did.

2. Teleop only ever emitted three discrete levels per channel, so anything between
   them is a speed the arm has never been driven at under supervision. A policy
   emits continuous values regardless. snap_to_levels() quantizes back onto the
   demonstrated set; clamp_to_limits() keeps the raw value but bounded.

collect_data.py implements (1) inline in _motor/_servo_speed rather than calling
here, because those are entangled with JointControl's key-hold state. The two
must stay in agreement -- change one, check the other.
"""

import numpy as np

from lone_data.features import ACTION_COMMAND_LIMITS, ACTION_DIM

# What teleop can actually emit per channel, from virtual_gripper.py's constants.
# A continuous prediction is snapped onto the nearest of these.
DEMONSTRATED_LEVELS = [
    (-900.0, 0.0, 900.0),   # base_motor_speed      -- +/-MOTOR_SPEED or stopped
    (-100.0, 0.0, 100.0),   # upper_arm_servo_speed -- +/-JOINT_SPEED or stopped
    (-100.0, 0.0, 100.0),   # lower_arm_servo_speed -- +/-JOINT_SPEED or stopped
    (30.0, 120.0),          # gripper_angle         -- open or closed, nothing between
]


def clamp_to_limits(action):
    """Bound each dimension to ACTION_COMMAND_LIMITS, leaving the value otherwise intact."""
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    lo = np.array([a for a, _ in ACTION_COMMAND_LIMITS], dtype=np.float32)
    hi = np.array([b for _, b in ACTION_COMMAND_LIMITS], dtype=np.float32)
    return np.clip(action, lo, hi)


def snap_to_levels(action):
    """Quantize each dimension onto the nearest level teleop ever demonstrated."""
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    out = np.empty(ACTION_DIM, dtype=np.float32)
    for i, levels in enumerate(DEMONSTRATED_LEVELS):
        candidates = np.asarray(levels, dtype=np.float32)
        out[i] = candidates[int(np.argmin(np.abs(candidates - action[i])))]
    return out


def dispatch_action(bus, action, channels, last=None):
    """Queue the parts of one action that differ from the last one.

    channels: (base_motor, upper_servo, lower_servo, gripper_servo) indices, passed
    in rather than imported so this package stays independent of virtual_gripper.

    `last` is the previously dispatched action; None re-sends everything, which is
    what a run's first tick and anything following a stop_all want.

    RESENDING AN UNCHANGED COMMAND IS NOT FREE, and this is the whole reason the
    parameter exists. Every command is a round trip the CommandBus worker has to
    complete -- measured at 73/378/1116 ms min/avg/max on the routed campus path.
    Dispatching all four dimensions every tick is 100 commands/s at 25 Hz, which no
    WiFi path here can drain, so the queue runs permanently behind _MAX_COMMAND_AGE
    and every droppable *speed* is discarded for being stale while the
    non-droppable stops still go through. The arm then receives the stops and
    almost none of the motion.

    A demonstration is mostly repeats: the recorded episode in data/lerobot/lone/
    l_one changes its action on 18 of 406 frames. Teleop sent those 18 commands,
    because a key press emits one command and a hold emits none. Sending 1624
    instead is not a more faithful replay of that data -- it is a different
    dispatch pattern than the one the data was recorded under, and it is the one
    that saturates the link.

    The board holds a command until it is superseded, so the skipped ones change
    nothing about what the arm is doing. What still has to be fed is
    wifi_bridge.py's COMMAND_DEADMAN_TIMEOUT, a single global 2.0 s timer that any
    command resets -- see RobotAppBase._wifi_heartbeat_tick, which re-sends held
    speeds and is what makes them safe to mark droppable here.

    Speeds are droppable, stops are not: nothing re-sends a stop, so discarding one
    leaves the joint driving until that deadman fires.
    """
    action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    base, upper, lower, gripper = channels
    if last is None:
        changed = [True] * ACTION_DIM
    else:
        last = np.asarray(last, dtype=np.float32).reshape(ACTION_DIM)
        changed = [bool(action[i] != last[i]) for i in range(ACTION_DIM)]

    if changed[0]:
        speed = float(action[0])
        if speed == 0.0:
            bus.submit(f"motor:{base}", "stop_motor", base)
        else:
            bus.submit(f"motor:{base}", "set_motor_speed", base, int(round(speed)), droppable=True)

    for i, idx in ((1, upper), (2, lower)):
        if not changed[i]:
            continue
        value = float(action[i])
        if value == 0.0:
            bus.submit(f"servo:{idx}", "stop_servo", idx)
        else:
            bus.submit(f"servo:{idx}", "set_servo_speed", idx, int(round(value)), droppable=True)

    if changed[3]:
        bus.submit(
            f"servo_angle:{gripper}", "set_servo_angle", gripper, int(round(float(action[3])))
        )
