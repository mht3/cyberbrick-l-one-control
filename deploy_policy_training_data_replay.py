#!/usr/bin/env python3
"""Deployment GUI, but the policy's camera is a recorded episode.

    python deploy_policy_training_data_replay.py \
        --checkpoint outputs/train/2026-08-13/16-34-56_pi05/checkpoints/last/pretrained_model

This is deploy_policy.py -- literally, it subclasses DeployApp -- with one
substitution: observations come from a LeRobotDataset episode instead of a webcam.
The question it answers is the easiest one that can be asked of a checkpoint. Feed
it back the exact frames it trained on, on the real arm, and see whether it
reproduces that trajectory and picks the thing up. A policy that cannot imitate its
own single demonstration under its own training inputs has a problem in the
inference and dispatch path -- chunking, quantization, latency -- and no amount of
looking at a live camera will separate that from a generalization failure.

Subclassing rather than copying is the point. The checkpoint loading, the action
queue, snap/raw, dispatch_action() and the results logging are the ones that run
on the arm for real, so a failure here is a failure of the real thing.

Three things this shows that the live GUI cannot:

The recorded action is known, so every tick can be compared against ground truth
as it happens -- an exact-match rate under snapping, and a normalized MSE. That is
scripts/eval_policy.py's measurement taken live, with the arm moving.

Inference lag is measurable rather than inferred. Each action carries the index of
the frame it was predicted from, so the log says exactly how stale each command
was. Expect around n_action_steps: the worker refills whenever the queue drops
below a full chunk, so the queue settles near full and an action is dispatched
roughly a chunk after the frame that produced it. `--lookahead K` submits frame
cursor+K while the panes, plots and video still follow the cursor, which dials that
lag back out -- run at 0 first, since that is what deployment does.

And Source = Dataset dispatches the recorded actions instead of the policy's. It is
the control experiment, and the first run to make: if the arm cannot pick the marker
up from the demonstration's own actions, the fault is start pose, timing or hardware,
and the checkpoint is not what is being measured.

The live camera pane stays, stacked directly above the demo frame -- reality on
top, demonstration below, the same scene at the same scale so the difference is
the only thing that draws the eye. None of this means anything unless the arm
starts where the demonstration started, so scrub to frame 0 and teleoperate the
real arm until the two agree before pressing Start. `--no-camera` drops the pane.
"""

import json
import time
import tkinter as tk
from collections import deque
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

import lone_data  # noqa: F401  -- sets the offline flags before lerobot is imported

from virtual_gripper import PALETTE

from lone_data.checkpoints import default_root, same_dataset, training_dataset
from lone_data.dispatch import clamp_to_limits, dispatch_action, snap_to_levels
from lone_data.perturb import FeedPerturbation
from lone_data.features import (
    ACTION_COMMAND_LIMITS,
    ACTION_DIM,
    ACTION_NAMES,
    resize_keep_aspect,
)

from deploy_policy import (
    CHANNELS,
    PLOT_WINDOW_S,
    DeployApp,
    build_parser,
    validate_checkpoint,
)

# Two video panes stacked, live camera above the demo, so each gets less height
# than deploy_policy.py's single 640-wide pane would take.
PANE_WIDTH = 560
# Normalizing the ground-truth error: dim 0 spans +/-900 while dim 3 spans 90, so a
# raw average would be almost entirely dim 0. Same convention as eval_policy.py.
SPANS = np.array([hi - lo for lo, hi in ACTION_COMMAND_LIMITS], dtype=np.float64)


def parse_args():
    p = build_parser(description=__doc__)
    g = p.add_argument_group("training-data replay")
    # Required, and --root derived from it unless given. Two independent defaults
    # is the trap scripts/eval_policy.py documents: LeRobotDataset loads from
    # `root` and treats `repo_id` as a label once the data is local, so naming one
    # dataset while pointing at another replays the second without a word.
    g.add_argument("--repo-id", required=True, help="dataset repo id")
    g.add_argument("--root", default=None,
                   help="dataset directory (default: data/lerobot/<repo-id>)")
    g.add_argument("--episode", type=int, default=0, help="episode to replay")
    g.add_argument("--start-frame", type=int, default=0,
                   help="frame the cursor resets to when the policy starts")
    g.add_argument("--lookahead", type=int, default=0,
                   help="submit the frame this many ticks ahead of the cursor, to "
                        "compensate for inference lag (0 = what deployment does)")
    g.add_argument("--loop", action="store_true",
                   help="restart at --start-frame instead of stopping at the end")
    g.add_argument("--source", choices=("policy", "dataset"), default="policy",
                   help="dispatch the policy's actions, or the recorded ones")
    g.add_argument("--no-camera", action="store_true",
                   help="do not open the alignment camera")

    f = p.add_argument_group("feed perturbation (what a live camera does to the pixels)")
    f.add_argument("--brightness", type=float, default=1.0, help="multiply pixels by this")
    f.add_argument("--contrast", type=float, default=1.0, help="scale contrast about mid grey")
    f.add_argument("--noise", type=float, default=0.0, help="gaussian sigma, in 0-255 units")
    f.add_argument("--jpeg-quality", type=int, default=0,
                   help="re-encode each frame at this JPEG quality (0 = off; "
                        "stream_camera.py uses 80)")

    args = p.parse_args()
    if args.root is None:
        args.root = default_root(args.repo_id)
    return args


class ReplayStats:
    """Running comparison of what was dispatched against what was recorded.

    Under snapping both sides live on the same discrete grid, so exact agreement is
    a meaningful and very readable number -- it is the fraction of ticks on which
    the arm was told to do what the demonstration did. MSE is kept alongside it
    because in raw mode nothing matches exactly and the match rate goes to zero
    without saying how close the miss was.
    """

    def __init__(self):
        self.ticks = 0
        self.matches = np.zeros(ACTION_DIM, dtype=np.int64)
        self.sq_error = np.zeros(ACTION_DIM, dtype=np.float64)
        self.lags = []

    def add(self, dispatched, truth, lag):
        self.ticks += 1
        d = np.asarray(dispatched, dtype=np.float64)
        t = np.asarray(truth, dtype=np.float64)
        self.matches += (d == t)
        self.sq_error += ((d - t) / SPANS) ** 2
        if lag is not None:
            self.lags.append(lag)

    @property
    def match_rate(self):
        """Fraction of dimension-ticks that agreed exactly."""
        if not self.ticks:
            return 0.0
        return float(self.matches.sum()) / (self.ticks * ACTION_DIM)

    def summary(self):
        if not self.ticks:
            return {"ticks": 0}
        return {
            "ticks": self.ticks,
            "action_names": ACTION_NAMES,
            "match_rate": round(self.match_rate, 4),
            "match_rate_per_dim": [round(float(m) / self.ticks, 4) for m in self.matches],
            "mse_normalized_per_dim": [round(float(e) / self.ticks, 6) for e in self.sq_error],
            "mse_normalized": round(float(self.sq_error.sum()) / (self.ticks * ACTION_DIM), 6),
            "lag_ticks_mean": round(float(np.mean(self.lags)), 2) if self.lags else None,
            "lag_ticks_max": int(np.max(self.lags)) if self.lags else None,
        }


class ReplayApp(DeployApp):
    def __init__(self, args):
        from lone_data.dataset_feed import DatasetFeed

        # Opened before super().__init__ because DeployApp's constructor runs the
        # whole startup -- checkpoint load, camera, first tick schedule -- and
        # _observation_ready() is consulted inside it.
        self.feed = DatasetFeed(args.repo_id, root=args.root, episode=args.episode)

        # -- replay state, all needed before the first hook fires ----------
        self._cursor = args.start_frame
        self._observed_frame = args.start_frame
        self._submitted_frame = args.start_frame
        self._gt_history = deque(maxlen=4000)
        self._stats = ReplayStats()
        self._is_replay_tick = False
        self._shown_frame = None
        self.perturb = FeedPerturbation(
            brightness=args.brightness, contrast=args.contrast,
            noise=args.noise, jpeg_quality=args.jpeg_quality,
        )

        super().__init__(args)

        self.title("CyberBrick L-ONE Policy Deployment -- training-data replay")
        # Taller than deploy_policy.py rather than wider: the two panes stack.
        self.minsize(1150, 1000)
        self._display_width = PANE_WIDTH

        # The prompt comes from the episode being replayed rather than from
        # checkpoint_task()'s walk to the training dataset. Same string when the
        # checkpoint matches the data, and visibly not when it does not.
        if not args.task and self.feed.task:
            self.task_var.set(self.feed.task)

        self._log(f"Replaying {args.repo_id} episode {args.episode}: "
                  f"{self.feed.length} frames at {self.feed.fps} fps "
                  f"({self.feed.actual_width}x{self.feed.actual_height})", level="connected")
        if self.feed.fps != args.fps:
            self._log(f"Dataset is {self.feed.fps} fps but the control rate is {args.fps} Hz "
                      f"-- the replay will run at {args.fps / self.feed.fps:.2f}x speed.",
                      level="warn")
        self._check_observation_fidelity()
        self._check_checkpoint_matches_dataset()
        if self.perturb.active:
            self._log(f"Feed perturbation: {self.perturb.describe()}", level="warn")
        if args.lookahead:
            self._log(f"Lookahead {args.lookahead} frames: the policy is fed "
                      f"{args.lookahead / args.fps * 1000:.0f}ms ahead of the pane.")

        self._show_dataset_frame(self._cursor)
        self.after(0, self._replay_preview_tick)

    # -- startup checks ----------------------------------------------------

    def _check_observation_fidelity(self):
        """Confirm the deploy path reconstructs the training tensor exactly.

        Dataset frames go to the policy through deploy_policy.py's unmodified
        camera path -- uint8 BGR, resize_keep_aspect, /255, CHW. That is only the
        same input training saw if the round trip is lossless, which it is when the
        stored size matches image_size. Ruling out this class of silent mismatch is
        most of why this script exists, so it is checked rather than assumed.
        """
        if (self.feed.actual_height, self.feed.actual_width) != tuple(self.image_size):
            self._log(f"Dataset frames are {self.feed.actual_width}x{self.feed.actual_height} "
                      f"but --image-width/height say {self.image_size[1]}x{self.image_size[0]}"
                      " -- frames will be rescaled and will NOT match training inputs.",
                      level="warn")
            return
        try:
            diff = self.feed.verify_roundtrip(self._cursor)
        except Exception as e:
            self._log(f"Could not verify observation fidelity: {e}", level="warn")
            return
        if diff == 0.0:
            self._log("Observation check: deploy path reproduces the training tensor exactly.",
                      level="connected")
        else:
            self._log(f"Observation check: deploy path differs from the training tensor by "
                      f"{diff:.6f} -- the policy is not seeing what it trained on.", level="error")

    def _check_checkpoint_matches_dataset(self):
        """Say so when the episode being replayed is not from the training run.

        The whole premise here is "feed it back what it trained on", and pointing
        --root at a different dataset does not fail -- LeRobotDataset loads from
        root and treats repo_id as a label, so the wrong episodes replay silently
        and a memorized policy looks like it learned nothing.
        """
        trained_repo_id, trained_root = training_dataset(self.args.checkpoint)
        if trained_root is None:
            return
        if same_dataset(trained_root, self.args.root):
            return
        self._log(f"This checkpoint trained on {trained_repo_id} ({trained_root}), not on "
                  f"{self.args.root} -- this is a generalization test, not a replay of its "
                  "own demonstration.", level="warn")

    # -- DeployApp hooks ---------------------------------------------------

    def _build_extra_video_panel(self, parent):
        """The demo pane, under the live camera: reality on top, demonstration below."""
        frame = ttk.Frame(parent, padding=(0, 10, 0, 0))
        frame.pack(anchor="n")

        group = ttk.Frame(frame)
        group.pack(anchor="n")

        row = ttk.Frame(group)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="Replay", style="SectionHeading.TLabel").pack(side="left",
                                                                          padx=(0, 12))
        ttk.Label(row, text="Episode").pack(side="left")
        self.episode_var = tk.StringVar(value=str(self.args.episode))
        self.episode_combo = ttk.Combobox(
            row, textvariable=self.episode_var, width=6, state="readonly",
            values=[str(i) for i in range(self.feed.total_episodes)],
        )
        self.episode_combo.pack(side="left", padx=(8, 0))
        self.episode_combo.bind("<<ComboboxSelected>>", self._on_episode)
        self.frame_readout_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.frame_readout_var, style="Status.TLabel").pack(
            side="left", padx=(12, 0)
        )

        self.dataset_label = ttk.Label(group)
        self.dataset_label.pack()

        # Scrubbing is how the arm gets lined up with the demonstration before a run;
        # during one the cursor belongs to the control loop, so the slider follows it
        # instead of driving it.
        self.frame_scale = ttk.Scale(
            group, from_=0, to=max(0, self.feed.length - 1), orient="horizontal",
            command=self._on_scrub,
        )
        self.frame_scale.set(self._cursor)
        self.frame_scale.pack(fill="x", pady=(6, 0))
        return frame

    def _build_extra_controls(self, parent):
        frame = ttk.Frame(parent, padding=(0, 12, 0, 0))
        frame.pack(fill="x")

        ttk.Label(frame, text="Replay", style="SectionHeading.TLabel").pack(anchor="w", pady=(0, 4))

        # The control experiment. Dataset mode dispatches the recorded actions
        # through the same bus at the same rate, so a failure in both modes is a
        # failure of the setup rather than of the checkpoint.
        self._source_var = tk.StringVar(value=self.args.source)
        source_row = ttk.Frame(frame)
        source_row.pack(fill="x", pady=(0, 8))
        ttk.Label(source_row, text="Source", width=12, style="JointName.TLabel").pack(side="left")
        # Held in a list so they can be disabled mid-run: _control_tick reads the
        # variable every tick, and flipping it under a running loop would swap the
        # arm's command source without stopping anything.
        self._source_radios = [
            ttk.Radiobutton(source_row, text="Policy", variable=self._source_var,
                            value="policy", command=self._set_policy_buttons_state),
            ttk.Radiobutton(source_row, text="Dataset (ground truth)", variable=self._source_var,
                            value="dataset", command=self._set_policy_buttons_state),
        ]
        self._source_radios[0].pack(side="left")
        self._source_radios[1].pack(side="left", padx=(10, 0))

        opts = ttk.Frame(frame)
        opts.pack(fill="x", pady=(0, 4))
        self._loop_var = tk.BooleanVar(value=self.args.loop)
        ttk.Checkbutton(opts, text="Loop", variable=self._loop_var).pack(side="left")
        ttk.Label(opts, text="Lookahead", style="KeyHint.TLabel").pack(side="left", padx=(14, 4))
        self._lookahead_var = tk.IntVar(value=self.args.lookahead)
        ttk.Spinbox(opts, from_=0, to=100, width=5, textvariable=self._lookahead_var).pack(
            side="left"
        )

        self._build_perturb_controls(frame)

        self.replay_status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.replay_status_var, style="Status.TLabel").pack(
            anchor="w", pady=(6, 0)
        )
        self.replay_match_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.replay_match_var, style="Status.TLabel").pack(anchor="w")
        return frame

    def _open_camera(self, source):
        """The alignment camera is optional; the replay never reads from it."""
        if self.args.no_camera:
            return False
        return super()._open_camera(source)

    def _set_policy_buttons_state(self):
        super()._set_policy_buttons_state()
        for radio in getattr(self, "_source_radios", ()):
            radio.state(["disabled"] if self._policy_running else ["!disabled"])
        # Dataset mode drives the arm from the recording alone, so it can run with
        # no checkpoint at all -- which is exactly the situation you want it in when
        # the checkpoint is the thing under suspicion.
        if (getattr(self, "_source_var", None) is not None
                and self._source_var.get() == "dataset"
                and not self._policy_running and self.link is not None):
            self.start_policy_btn.state(["!disabled"])

    def _build_perturb_controls(self, parent):
        """Live knobs for degrading the feed the way a real camera path does.

        Adjustable mid-run on purpose. The interesting measurement is not whether
        a policy survives a fixed amount of noise, it is how far you get to turn
        the dial before the arm stops doing the task -- and that is something you
        watch happen, with the recorded episode still playing underneath as the
        control.
        """
        frame = ttk.Frame(parent, padding=(0, 10, 0, 0))
        frame.pack(fill="x")

        # Collapsed by default, like the log and plot panels: a clean feed is the
        # normal case and four sliders of it is clutter. The header carries the
        # state, though -- a run silently poisoned by noise left on from the last
        # one is exactly what a hidden panel would cause.
        self._perturb_expanded = False
        self.perturb_header_var = tk.StringVar()
        header = ttk.Label(frame, textvariable=self.perturb_header_var,
                           style="SectionHeading.TLabel", cursor="hand2")
        header.pack(anchor="w")
        header.bind("<Button-1>", lambda e: self._toggle_perturb_panel())

        self.perturb_body = ttk.Frame(frame)
        self._perturb_vars = {}
        self._perturb_snapping = False
        for key, label, lo, hi, step in (
            ("contrast", "Contrast", 0.5, 1.5, 0.01),
            ("brightness", "Brightness", 0.5, 1.5, 0.01),
            ("noise", "Noise σ", 0.0, 15.0, 0.1),
            ("jpeg_quality", "JPEG q", 75, 100, 5),
        ):
            row = ttk.Frame(self.perturb_body)
            row.pack(fill="x", pady=1)
            if key == "jpeg_quality":
                # Gated rather than folded into the slider's range: 75 is a real
                # quality, not "off", and this panel exists to measure differences
                # of a third of a percent -- calling a lossy re-encode "off" would
                # put a lie in the one readout that has to be trusted.
                self._jpeg_on = tk.BooleanVar(value=self.perturb.jpeg_quality > 0)
                ttk.Checkbutton(row, text=label, width=10, variable=self._jpeg_on,
                                command=self._apply_perturb).pack(side="left")
            else:
                ttk.Label(row, text=label, width=11, style="KeyHint.TLabel").pack(side="left")
            start = getattr(self.perturb, key) or (100 if key == "jpeg_quality" else 0.0)
            var = tk.DoubleVar(value=start)
            self._perturb_vars[key] = var
            readout = tk.StringVar()
            ttk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
                      command=lambda _v, k=key, s=step: self._on_perturb(k, s)).pack(
                          side="left", fill="x", expand=True)
            ttk.Label(row, textvariable=readout, width=6, style="KeyHint.TLabel").pack(side="left")
            var.trace_add("write", lambda *_a, k=key, r=readout, s=step:
                          r.set(f"{self._perturb_vars[k].get():.0f}" if s >= 1
                                else f"{self._perturb_vars[k].get():.2f}"))
            readout.set(f"{start:.0f}" if step >= 1 else f"{start:.2f}")

        tail = ttk.Frame(self.perturb_body)
        tail.pack(fill="x", pady=(2, 0))
        self.perturb_status_var = tk.StringVar(value="")
        ttk.Label(tail, textvariable=self.perturb_status_var, style="Status.TLabel").pack(
            side="left")
        ttk.Button(tail, text="Clean", width=7, command=self._reset_perturb).pack(side="right")

        self._update_perturb_status()
        return frame

    def _toggle_perturb_panel(self):
        self._perturb_expanded = not self._perturb_expanded
        if self._perturb_expanded:
            self.perturb_body.pack(fill="x")
        else:
            self.perturb_body.pack_forget()
        self._update_perturb_status()

    def _on_perturb(self, key, step):
        if self._perturb_snapping:
            return
        if step >= 1:
            # ttk.Scale is continuous; quantize the JPEG slider onto its steps so
            # runs are comparable and the readout is a quality anyone can restate.
            var = self._perturb_vars[key]
            snapped = round(var.get() / step) * step
            if snapped != var.get():
                self._perturb_snapping = True
                try:
                    var.set(snapped)
                finally:
                    self._perturb_snapping = False
        self._apply_perturb()

    def _apply_perturb(self):
        self.perturb.contrast = float(self._perturb_vars["contrast"].get())
        self.perturb.brightness = float(self._perturb_vars["brightness"].get())
        self.perturb.noise = float(self._perturb_vars["noise"].get())
        quality = int(round(self._perturb_vars["jpeg_quality"].get()))
        self.perturb.jpeg_quality = quality if self._jpeg_on.get() else 0
        self._update_perturb_status()
        # Redraw the pane so the operator sees what the policy is being given, not
        # the clean frame -- the whole point is watching the input degrade.
        if not self._policy_running:
            self._show_dataset_frame(self._observed_frame, force=True)

    def _reset_perturb(self):
        self.perturb.reset()
        self._perturb_snapping = True
        try:
            self._perturb_vars["contrast"].set(1.0)
            self._perturb_vars["brightness"].set(1.0)
            self._perturb_vars["noise"].set(0.0)
            self._jpeg_on.set(False)
        finally:
            self._perturb_snapping = False
        self._apply_perturb()

    def _update_perturb_status(self):
        arrow = "▼" if self._perturb_expanded else "▶"
        if not self.perturb.active:
            self.perturb_header_var.set(f"{arrow} Feed perturbation  ·  clean")
            self.perturb_status_var.set("Feed is the exact training tensor.")
            return
        try:
            diff = self.perturb.mean_abs_diff(self.feed.rgb(self._observed_frame))
        except Exception:
            diff = float("nan")
        # The number that makes sensitivity legible: input change on the same
        # 0-255 scale as the pixels, next to the action change it produces. It
        # rides in the header too, so a perturbed feed cannot hide behind a
        # collapsed panel.
        self.perturb_header_var.set(f"{arrow} Feed perturbation  ·  {self.perturb.describe()}")
        self.perturb_status_var.set(f"→ {diff:.2f}/255 mean pixel diff vs the training tensor")

    def _observation_ready(self):
        """The dataset is the observation source, so the camera must not gate a run."""
        if self.feed is None:
            return "No dataset loaded."
        if not 0 <= self.args.start_frame < self.feed.length:
            return f"--start-frame must be in 0..{self.feed.length - 1}."
        return None

    def _policy_observation(self):
        """The frame at the cursor, advancing it, tagged with the index submitted."""
        if not self._advance_cursor():
            return None, None
        lookahead = max(0, int(self._lookahead_var.get()))
        self._submitted_frame = min(self._observed_frame + lookahead, self.feed.length - 1)
        return self._deploy_path_frame(self._submitted_frame), self._submitted_frame

    def _deploy_path_frame(self, index):
        """A dataset frame carried in through deploy_policy.py's own camera path.

        Deliberately not the dataset tensor handed straight to the policy: the
        point is to exercise the preprocessing deployment actually runs, so that a
        mismatch in it shows up here rather than only on the arm. When the stored
        size equals image_size the round trip is exact, which is what
        _check_observation_fidelity() confirms at startup; when it is not, this
        rescales rather than silently feeding the wrong shape.

        The feed perturbation lands here, before the deploy path rather than after
        it, because that is where a camera's own degradations live -- they are in
        the pixels the resize is handed, not something applied to the tensor.
        """
        rgb = self.perturb.apply(self.feed.rgb(index))
        return resize_keep_aspect(rgb[:, :, ::-1].copy(), self.image_size)

    def _extra_log_fields(self):
        source = {"source": self._source_var.get()}
        # A run opens and closes with entries that record a command but no frame --
        # the handover stop, and the manual state reset afterwards. Tagging those
        # with a demo frame and its action would put a ground-truth comparison in
        # the log for ticks where the arm was not being driven from one.
        if not self._is_replay_tick:
            return source
        truth = self.feed.action(self._observed_frame)
        return {
            **source,
            "observed_frame": int(self._observed_frame),
            "predicted_from_frame": (None if self._action_meta is None
                                     else int(self._action_meta)),
            "lag_ticks": self._lag_ticks(),
            "ground_truth": [round(float(v), 3) for v in truth],
            **({"perturbation": self.perturb.as_dict()} if self.perturb.active else {}),
        }

    def _run_metadata(self):
        return {
            **super()._run_metadata(),
            "replay": {
                "repo_id": self.args.repo_id,
                "root": self.args.root,
                "episode": self.args.episode,
                "start_frame": self.args.start_frame,
                "lookahead": int(self._lookahead_var.get()),
                "loop": bool(self._loop_var.get()),
                "source": self._source_var.get(),
                "dataset_task": self.feed.task,
                "dataset_fps": self.feed.fps,
                "episode_frames": self.feed.length,
            },
            # Recorded because a run under a degraded feed is a different
            # experiment, and the video alone will not say by how much.
            "feed_perturbation": self.perturb.as_dict(),
        }

    def _draw_extra_plots(self, now):
        """The demonstration's own actions, dashed, on the same axes as the commands.

        Trimmed against the same window as _history rather than sharing its buffer:
        ground truth only exists on policy ticks, while _history also carries manual
        control and stop entries, so the two are not sample-for-sample parallel.
        """
        if not self._gt_history:
            return
        while len(self._gt_history) > 1 and self._gt_history[1][0] < now - PLOT_WINDOW_S:
            self._gt_history.popleft()
        x = np.fromiter((t - now for t, _ in self._gt_history), dtype=np.float64,
                        count=len(self._gt_history))
        data = np.stack([a for _, a in self._gt_history])
        x = np.append(x, 0.0)
        data = np.vstack([data, data[-1]])
        for j, line in enumerate(self._gt_lines):
            line.set_data(x, data[:, j])

    def _build_plot_panel(self, parent):
        super()._build_plot_panel(parent)
        self._gt_lines = [
            ax.plot([], [], lw=1.0, ls="--", alpha=0.75, color=PALETTE["muted"],
                    drawstyle="steps-post")[0]
            for ax in self.axes
        ]
        self.axes[0].legend(
            [self._lines[0], self._gt_lines[0]], ["commanded", "demonstration"],
            fontsize=6, loc="upper right", framealpha=0.3,
        )

    # -- replay clock ------------------------------------------------------

    def _advance_cursor(self):
        """Step the cursor one frame. False when the episode is over and stopping."""
        if self._cursor >= self.feed.length:
            if not self._loop_var.get():
                self._log(f"Reached the end of episode {self.args.episode} "
                          f"({self.feed.length} frames).")
                self._stop_policy(reason="episode ended")
                return False
            self._cursor = self.args.start_frame
            self._log(f"Looping back to frame {self.args.start_frame}.")
        self._observed_frame = self._cursor
        self._cursor += 1
        return True

    def _lag_ticks(self):
        """How stale the dispatched action is, in frames, or None if untagged."""
        if self._action_meta is None:
            return None
        return int(self._observed_frame - self._action_meta)

    def _start_policy(self):
        if self._policy_running:
            return
        self._cursor = self.args.start_frame
        self._observed_frame = self._submitted_frame = self.args.start_frame
        self._stats = ReplayStats()
        self._gt_history.clear()

        if self._source_var.get() == "dataset":
            self._start_dataset_replay()
            return
        super()._start_policy()

    def _start_dataset_replay(self):
        """Dispatch the recorded actions, with no policy in the loop at all."""
        if self.link is None:
            self._log("Connect to the board before starting the replay.", level="warn")
            return
        self.bus.cancel_pending()
        self.bus.send_now("stop_all")
        self._current_action[0:3] = 0.0
        self._policy_running = True
        self._held = None
        self._held_ticks = 0
        self._action_meta = None
        self._last_dispatched = None  # stop_all bypassed the bus; re-send everything
        self._underrun_reported = False
        self._priming = False
        self._tick_index = 0
        self._note_action(raw=None, dispatched=self._current_action, underrun=False)
        self._set_policy_buttons_state()
        self._log(f"Replaying recorded actions from episode {self.args.episode} "
                  "-- the policy is not in the loop.", level="connected")
        self._next_deadline = time.monotonic()
        self._control_tick()

    def _control_tick(self):
        if self._source_var.get() != "dataset":
            super()._control_tick()
            return
        if not self._policy_running:
            return
        self._next_deadline += self._period
        delay = max(1, int((self._next_deadline - time.monotonic()) * 1000))
        self.after(delay, self._control_tick)

        if not self._advance_cursor():
            return
        self._submitted_frame = self._observed_frame
        truth = self.feed.action(self._observed_frame)
        # Snapped/clamped like a prediction would be. The recorded values are
        # already on the demonstrated grid, so snapping is a no-op -- which is
        # itself worth having pass through the same code path.
        action = snap_to_levels(truth) if self._snap_var.get() else clamp_to_limits(truth)
        dispatch_action(self.bus, action, CHANNELS, last=self._last_dispatched)
        self._last_dispatched = np.asarray(action, dtype=np.float32).copy()
        self._current_action[:] = action
        self._note_action(raw=truth, dispatched=action, underrun=False,
                          frame=self._deploy_path_frame(self._observed_frame))
        self._tick_index += 1
        if self._tick_index % max(1, self.args.fps // 5) == 0:
            self._update_status_label()

    def _stop_policy(self, reason="stopped"):
        if not self._policy_running:
            return
        super()._stop_policy(reason=reason)
        self._write_summary()
        self._update_replay_status()

    def _write_summary(self):
        summary = self._stats.summary()
        if not summary.get("ticks"):
            return
        self._log(
            f"Replay: {summary['ticks']} ticks  ·  match {summary['match_rate'] * 100:.0f}%  ·  "
            f"normalized MSE {summary['mse_normalized']:.4f}"
            + (f"  ·  lag {summary['lag_ticks_mean']:.1f} ticks mean"
               if summary["lag_ticks_mean"] is not None else ""),
            level="connected",
        )
        if self._log_file is not None:
            self._log_file.write(json.dumps({"run_summary": summary}) + "\n")
            self._log_file.flush()

    # -- history, status, preview -----------------------------------------

    def _note_action(self, raw, dispatched, underrun, frame=None):
        # Only ticks where the arm was driven from a dataset frame have a ground
        # truth to compare against. `frame is None` is what separates those from
        # manual moves and from the entry a run opens with, before its first tick.
        # Set before super(), which is what calls _extra_log_fields().
        self._is_replay_tick = self._policy_running and frame is not None
        super()._note_action(raw, dispatched, underrun, frame=frame)
        if not self._is_replay_tick:
            return
        truth = self.feed.action(self._observed_frame)
        self._gt_history.append((time.time(), truth.copy()))
        self._stats.add(dispatched, truth, self._lag_ticks())

    def _update_status_label(self):
        super()._update_status_label()
        self._update_replay_status()

    def _update_replay_status(self):
        last = max(0, self.feed.length - 1)
        text = f"frame {self._observed_frame}/{last}"
        if self._policy_running and self._source_var.get() == "policy":
            text += f"  ·  submitted {self._submitted_frame}"
            lag = self._lag_ticks()
            if lag is not None:
                text += f"  ·  lag {lag} ticks ({lag / self.args.fps * 1000:.0f}ms)"
        self.replay_status_var.set(text)
        self._update_perturb_status()
        if self._stats.ticks:
            s = self._stats
            self.replay_match_var.set(
                f"match {s.match_rate * 100:.0f}%  ·  MSE "
                f"{s.sq_error.sum() / (s.ticks * ACTION_DIM):.4f} over {s.ticks} ticks"
            )
        else:
            self.replay_match_var.set(
                "Match the arm to the demo frame, then Start." if not self._policy_running else ""
            )

    def _on_scrub(self, value):
        if self._policy_running:
            return  # the control loop owns the cursor during a run
        index = int(float(value))
        if index == self._observed_frame:
            return
        self._cursor = self._observed_frame = self._submitted_frame = index
        self._show_dataset_frame(index)
        self._update_replay_status()

    def _on_episode(self, event=None):
        if self._policy_running:
            self._log("Stop the policy before switching episode.", level="warn")
            self.episode_var.set(str(self.args.episode))
            return
        episode = int(self.episode_var.get())
        if episode == self.args.episode:
            return
        from lone_data.dataset_feed import DatasetFeed

        try:
            feed = DatasetFeed(self.args.repo_id, root=self.args.root, episode=episode)
        except Exception as e:
            self._log(f"Could not open episode {episode}: {e}", level="error")
            self.episode_var.set(str(self.args.episode))
            return
        self.feed.close()
        self.feed = feed
        self.args.episode = episode
        self.args.start_frame = min(self.args.start_frame, feed.length - 1)
        self._cursor = self._observed_frame = self._submitted_frame = self.args.start_frame
        self._stats = ReplayStats()
        self._gt_history.clear()
        self.frame_scale.configure(to=max(0, feed.length - 1))
        self.frame_scale.set(self._cursor)
        if feed.task:
            self.task_var.set(feed.task)
        self._log(f"Episode {episode}: {feed.length} frames, task {feed.task!r}")
        self._show_dataset_frame(self._cursor)
        self._update_replay_status()

    def _replay_preview_tick(self):
        """Keep the dataset pane on the cursor while a run advances it.

        _show_dataset_frame() redraws only when the cursor actually moved. Unlike a
        camera, a recorded frame does not change on its own, and decoding, perturbing
        and rescaling the same one 20x a second is work done on the thread that runs
        control -- and, at startup, work taken directly from the thread loading 3B
        parameters onto the GPU.
        """
        self.after(50, self._replay_preview_tick)
        if self._policy_running:
            self._show_dataset_frame(self._observed_frame)
            self.frame_scale.set(self._observed_frame)

    def _show_dataset_frame(self, index, force=False):
        index = max(0, min(index, self.feed.length - 1))
        if index == self._shown_frame and not force:
            return
        self._shown_frame = index
        try:
            # Perturbed, not clean: the pane has to show what the policy is given.
            rgb = self.perturb.apply(self.feed.rgb(index))
        except Exception as e:
            self._log(f"Could not decode frame {index}: {e}", level="error")
            return
        image = Image.fromarray(rgb).resize(
            (PANE_WIDTH, max(1, int(self.feed.actual_height * PANE_WIDTH / self.feed.actual_width)))
        )
        photo = ImageTk.PhotoImage(image)
        self.dataset_label.configure(image=photo)
        self.dataset_label.image = photo
        self.frame_readout_var.set(
            f"demo frame {index}  ·  t={index / self.feed.fps:.2f}s"
        )

    # -- teardown ----------------------------------------------------------

    def _shutdown(self):
        super()._shutdown()
        if self.feed is not None:
            self.feed.close()


def main():
    args = parse_args()
    problem = validate_checkpoint(args.checkpoint)
    if problem and args.source == "policy":
        print(f"warning: {problem}")
    app = ReplayApp(args)
    try:
        app.mainloop()
    finally:
        app._shutdown()


if __name__ == "__main__":
    main()
