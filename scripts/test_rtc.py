#!/usr/bin/env python3
"""Check the real-time chunking wiring in deploy_policy.py, without an arm.

    python scripts/test_rtc.py                 # math only, CPU, seconds
    python scripts/test_rtc.py --checkpoint outputs/train/.../pretrained_model

Two tiers. The first exercises lerobot's guidance directly and needs no weights:
it pins the prefix weight schedule, which is what --rtc-execution-horizon and
--rtc-schedule actually control, and confirms an absent prefix is an exact no-op.

The second loads the real checkpoint and drives PolicyRunner the way
_control_tick does -- submit a frame, pop an action, 25 times a second -- with no
board and no camera. That is the only way to cover the queue bookkeeping, which
is threaded and mode-dependent and where the mistakes are: the mirror of the
pre-postprocessor chunk that RTC's prefix is sliced out of has to stay aligned
with the dispatch queue across replaces, appends, underruns and a mid-run toggle.

Exits non-zero if anything doesn't match expectations.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- disables Hub access before lerobot is imported

from lone_data.features import ACTION_DIM, DEFAULT_IMAGE_SIZE

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None,
                   help="run the runner harness too, against this pretrained_model/ dir")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--ticks", type=int, default=200, help="harness ticks per phase")
    p.add_argument("--device", default=None)
    return p.parse_args()


# -- tier 0: the guidance math -------------------------------------------------

def test_prefix_weights():
    """The weight schedule, which is what the two horizon knobs really select.

    start=inference_delay, end=execution_horizon, total=chunk_size. Everything up
    to the delay is held at 1 (those steps are already committed), the ramp spans
    delay->horizon, and past the horizon the new chunk is unconstrained.
    """
    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc import RTCConfig, RTCProcessor

    print("prefix weights (chunk=50, horizon=10)")
    proc = RTCProcessor(RTCConfig(execution_horizon=10,
                                  prefix_attention_schedule=RTCAttentionSchedule.LINEAR))

    w = proc.get_prefix_weights(0, 10, 50)
    check("delay 0 ramps linearly over the whole horizon",
          w.shape[0] == 50 and abs(float(w[0]) - 0.9091) < 1e-3
          and abs(float(w[8]) - 0.1818) < 1e-3 and float(w[10:].abs().max()) == 0.0,
          f"w[:3]={[round(float(v), 3) for v in w[:3]]}")

    w = proc.get_prefix_weights(7, 10, 50)
    check("delay 7 holds seven steps then ramps over three",
          float(w[:7].min()) == 1.0
          and [round(float(v), 2) for v in w[7:10]] == [0.75, 0.5, 0.25]
          and float(w[10:].abs().max()) == 0.0)

    # The reason --rtc-execution-horizon defaults above the expected delay.
    w = proc.get_prefix_weights(12, 10, 50)
    check("delay past the horizon leaves no ramp at all -- a hard clamp",
          float(w[:10].min()) == 1.0 and float(w[10:].abs().max()) == 0.0)

    check("a zero horizon disables guidance entirely",
          float(proc.get_prefix_weights(0, 0, 50).abs().max()) == 0.0)

    print("guidance no-ops")
    lin = torch.nn.Linear(32, 32)
    x = torch.randn(1, 50, 32)
    plain = lin(x.clone().detach())
    check("no prefix returns the unguided velocity",
          torch.equal(proc.denoise_step(x_t=x, prev_chunk_left_over=None, inference_delay=0,
                                        time=0.6, original_denoise_step_partial=lin), plain))
    check("an empty prefix returns the unguided velocity",
          torch.equal(proc.denoise_step(x_t=x, prev_chunk_left_over=torch.zeros(0, ACTION_DIM),
                                        inference_delay=0, time=0.6,
                                        original_denoise_step_partial=lin), plain))

    # Why the runner passes the leftover raw instead of padding it to the horizon
    # the way lerobot's compile-oriented helper does: padding puts live weights on
    # rows that carry no plan, pulling those actions toward normalized zero.
    short = torch.randn(3, ACTION_DIM)
    padded = torch.zeros(10, ACTION_DIM)
    padded[:3] = short
    raw_out = proc.denoise_step(x_t=x, prev_chunk_left_over=short, inference_delay=0,
                                time=0.6, original_denoise_step_partial=lin)
    pad_out = proc.denoise_step(x_t=x, prev_chunk_left_over=padded, inference_delay=0,
                                time=0.6, original_denoise_step_partial=lin)
    check("zero-padding a short prefix would distort the guidance",
          float((raw_out[:, 3:10, :ACTION_DIM] - pad_out[:, 3:10, :ACTION_DIM]).abs().max()) > 0.1,
          "so the runner truncates and never pads")


# -- tier 1: the runner, driven like the control tick --------------------------

def invariant(runner):
    """len(_pending) == len(_chunk_raw) - _consumed, the queue/mirror alignment."""
    with runner._lock:
        queued = len(runner._pending)
        mirror = 0 if runner._chunk_raw is None else len(runner._chunk_raw)
        return queued, mirror - runner._consumed


def drive(runner, ticks, fps, frame, label):
    """Submit a frame and pop an action per tick, as _control_tick does."""
    period = 1.0 / fps
    deadline = time.monotonic()
    breaks, underruns, delays = [], 0, []
    for _ in range(ticks):
        deadline += period
        if runner.error is not None:
            raise runner.error
        runner.submit_frame(frame, None)
        queued, expected = invariant(runner)
        if queued != expected:
            breaks.append((queued, expected))
        if runner.pop() is None:
            underruns += 1
        if runner.last_rtc_delay is not None:
            delays.append(runner.last_rtc_delay)
        time.sleep(max(0.0, deadline - time.monotonic()))
    print(f"    {label}: {ticks} ticks, {underruns} underrun, "
          f"{len(breaks)} invariant break(s), "
          f"latency {runner.last_latency * 1000:.0f}ms, "
          f"delay {sorted(set(delays)) if delays else 'n/a'}")
    return breaks, underruns, delays


def test_guidance_bites(runner, frame, task):
    """The chunk must actually move toward the prefix, not merely not crash.

    Everything else here would pass just as happily if the kwargs were being
    dropped on the floor somewhere between the runner and the denoiser, so this
    is the check that says the wiring reaches the guidance at all. Same
    observation and same noise seed both times; the only difference is the
    prefix, and what it should buy is agreement over the first
    execution_horizon steps and nothing much after them.
    """
    print("\n  guidance actually bites")
    runner.task = task
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
             for k, v in runner._observation(frame).items()}
    prepared = runner.pre(batch)
    horizon = runner.rtc_execution_horizon

    def chunk_with(prefix, seed=0):
        torch.manual_seed(seed)
        with torch.no_grad():
            kwargs = {} if prefix is None else {"inference_delay": 4,
                                                "prev_chunk_left_over": prefix}
            return runner.policy.predict_action_chunk(prepared, **kwargs)[0].float().cpu()

    runner.rtc_config.enabled = True
    free = chunk_with(None)
    # A target the policy would not have chosen on its own, so any pull is visible.
    prefix = (free[:horizon] + 1.5).to(runner.device)
    target = prefix.cpu()

    def distance(chunk):
        return float((chunk[:horizon] - target).abs().mean())

    guided = chunk_with(prefix)
    before, after = distance(free), distance(guided)
    check("guidance pulls the chunk head toward the prefix", after < before / 2,
          f"mean |head - prefix| {before:.3f} -> {after:.3f}")
    tail = float((guided[horizon:] - free[horizon:]).abs().mean())
    head = float((guided[:horizon] - free[:horizon]).abs().mean())
    check("the tail past the horizon stays roughly free", tail < head / 10,
          f"head moved {head:.3f}, tail moved {tail:.3f}")

    # Asked as a direction rather than an equality: repeating one call is not
    # bit-identical on GPU, and how far it drifts varies by orders of magnitude
    # between runs, so "the output is unchanged" is not a testable claim. "The
    # head was not dragged toward the prefix" is, and it is the claim that matters.
    runner.rtc_config.enabled = False
    ignored = distance(chunk_with(prefix))
    check("with RTC disabled the prefix does not pull the chunk", ignored > before * 0.9,
          f"mean |head - prefix| {ignored:.3f} vs {before:.3f} unguided")
    runner.rtc_config.enabled = True


def test_runner(args):
    from deploy_policy import PolicyRunner, rtc_settings

    print(f"\nrunner harness ({args.checkpoint})")
    settings = rtc_settings(argparse.Namespace(rtc=True))
    runner = PolicyRunner(args.checkpoint, device=args.device, fps=args.fps, rtc=settings)
    check("pi0.5 reports RTC support", runner.supports_rtc)
    check("the policy received the config", runner.policy.config.rtc_config is runner.rtc_config)
    check("the processor was built", getattr(runner.policy, "rtc_processor", None) is not None)
    check("the model sees the same processor",
          runner.policy.model.rtc_processor is runner.policy.rtc_processor)

    frame = np.zeros((*DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)
    task = runner.task or "Pick up the green marker."
    test_guidance_bites(runner, frame, task)

    # RTC on, then off mid-run, then on again -- the toggle is live, so the
    # bookkeeping has to survive a merge policy that changes underneath it.
    runner.start(task)
    try:
        on_breaks, _, delays = drive(runner, args.ticks, args.fps, frame, "rtc on")
        check("queue and chunk mirror stay aligned with RTC on", not on_breaks,
              "" if not on_breaks else f"first break {on_breaks[0]}")
        check("RTC produced a non-empty guidance prefix",
              runner.last_rtc_prefix and runner.last_rtc_prefix > 0,
              f"prefix {runner.last_rtc_prefix} rows")
        expected = math.ceil(runner.last_latency * args.fps)
        check("the dropped-action count tracks measured latency",
              delays and abs(delays[-1] - expected) <= 1,
              f"delay {delays[-1] if delays else None} vs ceil(latency*fps)={expected}")

        runner.rtc_enabled = False
        off_breaks, _, _ = drive(runner, args.ticks, args.fps, frame, "toggled off")
        check("the mirror survives a live toggle off", not off_breaks,
              "" if not off_breaks else f"first break {off_breaks[0]}")

        runner.rtc_enabled = True
        back_breaks, _, _ = drive(runner, args.ticks, args.fps, frame, "toggled back on")
        check("the mirror survives a live toggle on", not back_breaks,
              "" if not back_breaks else f"first break {back_breaks[0]}")
        check("the first chunk after switching on was guided",
              runner.last_rtc_prefix and runner.last_rtc_prefix > 0,
              f"prefix {runner.last_rtc_prefix} rows")
    finally:
        runner.stop()

    check("stop() clears the mirror with the queue",
          runner._chunk_raw is None and runner._consumed == 0 and not runner._pending)

    # The baseline the checkbox is compared against must be untouched.
    print("\n  non-RTC runner (unchanged path)")
    plain = PolicyRunner(args.checkpoint, device=args.device, fps=args.fps,
                         rtc=rtc_settings(argparse.Namespace()))
    check("RTC stays off without --rtc",
          not plain.rtc_enabled and not plain.policy.config.rtc_config.enabled)
    plain.start(task)
    try:
        breaks, _, delays = drive(plain, args.ticks, plain.fps, frame, "rtc off")
        check("queue and chunk mirror stay aligned with RTC off", not breaks)
        check("no delay is reported when chunks are appended", not delays)
    finally:
        plain.stop()


def main():
    args = parse_args()
    test_prefix_weights()
    if args.checkpoint:
        test_runner(args)
    else:
        print("\nskipping the runner harness -- pass --checkpoint to run it")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
