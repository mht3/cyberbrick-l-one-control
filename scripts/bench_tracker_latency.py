"""Per-tick latency of the causal trackers, alone and with pi0.5 sharing the GPU.

    python scripts/bench_tracker_latency.py \
        --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval

At deployment the tracker runs every tick on both views while pi0.5 runs its
own inference on the same GPU, in the same process. A tracker that does 15 ms
on an idle GPU can do 40 ms next to a 3B-parameter VLA, so this measures both:

  idle     tracker alone, both views per tick
  shared   the same, with a background thread calling pi0.5's
           predict_action_chunk back to back (worse than deployment without
           RTC, close to it with RTC, which re-plans almost continuously)

Per-tick cost = front + side (they run one after the other, as a single
tracker thread would). Accuracy does not depend on this, so the queries are 27
fixed points near the gripper's usual area -- the count the benchmark tracks.
Also reported: pi0.5's own inference time alone and next to each tracker.
Writes outputs/track/latency.json.
"""

import argparse
import json
import os
import threading
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402

from lone_track import trackers  # noqa: E402
from lone_track.episodes import EpisodeIndex  # noqa: E402

CAUSAL = ("lk", "cotracker3_online", "bootstapir_online", "tapnextpp", "sam2_small_online",
          "sam2_large_online")


class PolicyLoop:
    """pi0.5 inference back to back in a thread, on a real two-view observation."""

    def __init__(self, checkpoint, rgbs, task):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint).eval().to("cuda")
        pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=checkpoint)
        keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
        batch = {k: torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
                 for k, rgb in zip(keys, rgbs)}
        batch["observation.state"] = torch.zeros(1, 4)
        batch["task"] = [task]
        self.prepared = pre(batch)
        self.times = []
        self._stop = threading.Event()
        self._thread = None

    def once(self):
        t = time.perf_counter()
        with self.torch.no_grad():
            self.policy.predict_action_chunk(self.prepared)
        self.torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1e3

    def _run(self):
        while not self._stop.is_set():
            self.times.append(self.once())

    def start(self):
        self.times = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return list(self.times)


def per_tick(tracker, frames_by_view, queries):
    lat = []
    for frames in frames_by_view:
        res = tracker.track(frames, queries)
        lat.append(res.latency_ms)
    total = np.sum(lat, axis=0)[2:]  # skip the init frame and one warm-up frame
    lag = res.lag_frames[2:]
    return {"median_ms": float(np.nanmedian(total)), "p95_ms": float(np.nanpercentile(total, 95)),
            "max_ms": float(np.nanmax(total)), "lag_frames_mean": float(np.nanmean(lag))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--trackers", nargs="+", default=list(CAUSAL), choices=CAUSAL)
    p.add_argument("--out", default=os.path.join("outputs", "track", "latency.json"))
    args = p.parse_args()

    index = EpisodeIndex(args.repo_id, args.root)
    frames = [index.frames(args.episode, v)[:args.frames] for v in index.views]
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(250, 400, 27), rng.uniform(60, 220, 27)])
    queries = np.column_stack([np.zeros(27), pts]).astype(np.float32)

    import torch

    loop = PolicyLoop(args.checkpoint, [f[0] for f in frames],
                      "Pick up the marker and place in the box.")
    for _ in range(3):
        loop.once()
    policy_alone = [loop.once() for _ in range(10)]
    result = {"gpu": torch.cuda.get_device_name(0), "frames": args.frames,
              "policy_alone_ms": {"median": float(np.median(policy_alone))}, "trackers": {}}
    print(f"pi0.5 alone: {np.median(policy_alone):.0f} ms/chunk")

    for name in args.trackers:
        tracker = trackers.build(name)
        per_tick(tracker, [f[:20] for f in frames], queries)  # warm-up
        idle = per_tick(tracker, frames, queries)
        loop.start()
        time.sleep(1.0)
        shared = per_tick(tracker, frames, queries)
        policy_times = loop.stop()
        result["trackers"][name] = {
            "idle": idle, "shared": shared,
            "policy_ms_while_tracking": float(np.median(policy_times)) if policy_times else None}
        print(f"{name:20s} idle {idle['median_ms']:6.1f}/{idle['p95_ms']:6.1f} ms  "
              f"shared {shared['median_ms']:6.1f}/{shared['p95_ms']:6.1f} ms (median/p95, both views)"
              f"  lag {idle['lag_frames_mean']:.1f} frames  pi0.5 -> "
              f"{result['trackers'][name]['policy_ms_while_tracking']:.0f} ms")
        del tracker
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
