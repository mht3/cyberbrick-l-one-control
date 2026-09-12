"""Stopping the policy leaves nothing behind to act on, asserted.

Two paths can carry a stale command past a stop: the policy runner's action
queue, and the command bus between the GUI and the board. No hardware or
checkpoint is touched -- the policy and the link are stand-ins that pause at the
exact moment the race needs.
"""

import os
import sys
import threading
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy_policy import PolicyRunner  # noqa: E402
from lone_data.command_bus import CommandBus  # noqa: E402
from lone_data.features import ACTION_DIM, CAMERA_KEY, DEFAULT_IMAGE_SIZE  # noqa: E402

CHUNK = 50


class BlockingPolicy:
    """An inference that holds until released, like one still running at Stop."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def predict_action_chunk(self, batch, **kwargs):
        self.entered.set()
        self.release.wait(5)
        return torch.ones(1, CHUNK, ACTION_DIM)


def runner_around(policy):
    """A PolicyRunner serving `policy`, skipping __init__'s checkpoint load."""
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.torch = torch
    runner.policy = policy
    runner.pre = lambda batch: batch
    runner.post = lambda chunk: chunk
    runner.fps = 25
    runner.n_action_steps = 10
    runner.image_keys = [CAMERA_KEY]
    runner.rtc_config = None
    runner.rtc_enabled = False
    runner.rtc_execution_horizon = 20
    runner.rtc_queue_threshold = 30
    runner.task = ""
    runner._pending = deque()
    runner._chunk_raw = None
    runner._consumed = 0
    runner._lock = threading.Lock()
    runner._frame = runner._frame_meta = None
    runner._running = False
    runner._thread = None
    runner._generation = 0
    runner.last_latency = runner.error = None
    runner.last_rtc_delay = runner.last_rtc_prefix = None
    return runner


def test_inference_outliving_stop_does_not_refill_the_queue():
    policy = BlockingPolicy()
    runner = runner_around(policy)
    runner.STOP_JOIN_TIMEOUT_S = 0.05
    runner.start("task")
    runner.submit_frame([np.zeros((*DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)])
    assert policy.entered.wait(5)

    worker = runner._thread
    runner.stop()  # the join times out: this inference is still running
    assert worker.is_alive()
    policy.release.set()
    worker.join(5)

    assert not worker.is_alive()
    assert runner.queued == 0
    assert runner._chunk_raw is None


class RecordingLink:
    """Records the order commands reach the board; can hold one mid-send."""

    def __init__(self, hold):
        self.calls = []
        self.hold = hold
        self.holding = threading.Event()
        self.resume = threading.Event()

    def __getattr__(self, method):
        def call(*args):
            if method == self.hold:
                self.holding.set()
                self.resume.wait(5)
            self.calls.append(method)
        return call


def test_stop_is_not_overtaken_by_a_command_already_in_flight():
    link = RecordingLink(hold="set_motor_speed")
    bus = CommandBus()
    bus.set_link(link)
    bus.start()
    try:
        bus.submit("motor:1", "set_motor_speed", 1, 900, droppable=True)
        assert link.holding.wait(5)  # off the queue and on its way to the board

        bus.cancel_pending()
        stopper = threading.Thread(target=bus.send_now, args=("stop_all",))
        stopper.start()
        time.sleep(0.05)  # time enough for an unordered stop to land first
        link.resume.set()
        stopper.join(5)

        assert link.calls == ["set_motor_speed", "stop_all"]
    finally:
        bus.close()
