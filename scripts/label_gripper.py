"""Click the gripper keypoints on the frames select_label_frames.py chose.

    python scripts/label_gripper.py \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval

Each frame is shown one view at a time. Click three points, in order:

    body   centre of the black gripper servo (the one that opens the jaws)
    tip_1  the end of one jaw
    tip_2  the end of the other jaw (either order)

After the third click the next view/frame comes up. Hover over the big image to
aim the zoom panel; clicking in the zoom panel is the precise way to place a
point. The other view is shown small for context.

    left click     place the active point (visible)
    right click    place it as occluded -- hidden, but you're sure where it is
    u              occluded, no idea where -> no position
    o              outside the frame
    1 / 2 / 3      choose body / tip_1 / tip_2 (to fix one)
    backspace      clear the active point
    n / right      next view or frame         p / left   previous
    N / up         next episode               P / down   previous episode
    q              quit (every click is already saved)

Labels are written to labels/gripper_gt_<dataset>.json after every edit; rerun
to resume at the first incomplete frame. Label the pilot episodes first (the
ones listed first): the benchmark can run on them alone.

--relabel N re-labels N random already-labelled views into a separate file,
without showing the first labels -- do it on another day, to measure how
precisely a human can place these points at all.
"""

import argparse
import os
import random

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lone_track.episodes import EpisodeIndex  # noqa: E402
from lone_track.label_session import LabelSession, Task, build_tasks  # noqa: E402
from lone_track.labels import KEYPOINT_HELP, LabelSet  # noqa: E402

ZOOM_HALF = 24          # zoom window is 48x48 source pixels
STYLE = {               # marker, colour per keypoint
    "body": ("o", "#ff8c00"),
    "tip_1": ("^", "#00c8ff"),
    "tip_2": ("v", "#ff3ce0"),
}
GRIPPER_LABEL = {30.0: "CLOSED", 120.0: "OPEN"}


class EpisodeCache:
    """Both views of the current episode, decoded once on entry (~0.5 s)."""

    def __init__(self, index):
        self.index = index
        self.episode = None
        self.frames = {}
        self.actions = None

    def get(self, episode, view, frame):
        if episode != self.episode:
            self.frames = {v: self.index.frames(episode, v) for v in self.index.views}
            self.actions = self.index.actions(episode)
            self.episode = episode
        return self.frames[view][frame]


class LabelApp:
    def __init__(self, session, index, out_path):
        self.s = session
        self.index = index
        self.out = out_path
        self.cache = EpisodeCache(index)
        self.hover = None

        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[key] = []
        self.fig = plt.figure(figsize=(16, 8.6))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.15, 1], height_ratios=[1.25, 1],
                                   left=0.02, right=0.99, top=0.93, bottom=0.03,
                                   wspace=0.04, hspace=0.12)
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_zoom = self.fig.add_subplot(gs[0, 1])
        self.ax_ctx = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_main, self.ax_zoom, self.ax_ctx):
            ax.set_xticks([])
            ax.set_yticks([])
        blank = np.zeros((index.height, index.width, 3), np.uint8)
        self.im_main = self.ax_main.imshow(blank, interpolation="nearest")
        self.im_zoom = self.ax_zoom.imshow(blank, interpolation="nearest")
        self.im_ctx = self.ax_ctx.imshow(blank, interpolation="bilinear")
        self.cross = [self.ax_zoom.axhline(0, color="w", lw=0.6, alpha=0.7),
                      self.ax_zoom.axvline(0, color="w", lw=0.6, alpha=0.7)]
        self.artists = []
        self.title = self.fig.suptitle("", fontsize=12, family="monospace")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.redraw()

    # -- drawing -------------------------------------------------------------

    def other_view(self):
        views = self.index.views
        return views[(views.index(self.s.task.view) + 1) % len(views)] if len(views) > 1 else None

    def redraw(self):
        t = self.s.task
        img = self.cache.get(t.episode, t.view, t.frame)
        self.im_main.set_data(img)
        self.im_zoom.set_data(img)
        ctx_view = self.other_view()
        if ctx_view:
            self.im_ctx.set_data(self.cache.get(t.episode, ctx_view, t.frame))
            self.ax_ctx.set_title(f"context: {ctx_view}", fontsize=9)
        for a in self.artists:
            a.remove()
        self.artists = []
        self._draw_points(t, [self.ax_main, self.ax_zoom], big=True)
        if ctx_view:
            self._draw_points(Task(t.episode, t.frame, ctx_view), [self.ax_ctx], big=False)
        self._aim_zoom(self.hover)

        grip = self.cache.actions[t.frame, 3]
        moving = np.any(self.cache.actions[t.frame, :3] != 0)
        n_done, n_all = self.s.n_complete(), len(self.s.tasks)
        kp = self.s.active
        self.title.set_text(
            f"ep {t.episode}  frame {t.frame}  view {t.view.upper()}   "
            f"gripper {GRIPPER_LABEL.get(float(grip), grip)}  {'MOVING' if moving else 'still'}   "
            f"[{n_done}/{n_all} done]\n"
            f"click: {kp} -- {KEYPOINT_HELP[kp]}   "
            "(right-click occluded, u unknown, o out, 1/2/3 pick, n/p nav, q quit)"
        )
        self.title.set_color(STYLE[kp][1])
        self.fig.canvas.draw_idle()

    def _draw_points(self, task, axes, big):
        for kp in self.s.keypoints:
            e = self.s.entry(kp, task)
            if e is None or e["xy"] is None:
                continue
            marker, colour = STYLE[kp]
            filled = e["vis"] == "visible"
            for ax in axes:
                (a,) = ax.plot(
                    *e["xy"], marker=marker, ms=(11 if big else 6), mew=1.6,
                    mfc=(colour if filled else "none"), mec=colour, ls="none",
                    alpha=0.9,
                )
                self.artists.append(a)

    def _aim_zoom(self, xy):
        if xy is None:
            xy = (self.index.width / 2, self.index.height / 2)
        x, y = xy
        self.ax_zoom.set_xlim(x - ZOOM_HALF, x + ZOOM_HALF)
        self.ax_zoom.set_ylim(y + ZOOM_HALF, y - ZOOM_HALF)
        self.cross[0].set_ydata([y, y])
        self.cross[1].set_xdata([x, x])

    # -- events --------------------------------------------------------------

    def on_move(self, event):
        if event.inaxes is self.ax_main and event.xdata is not None:
            self.hover = (event.xdata, event.ydata)
            self._aim_zoom(self.hover)
            self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes not in (self.ax_main, self.ax_zoom) or event.xdata is None:
            return
        vis = "occluded" if event.button == 3 else "visible"
        if event.button not in (1, 3):
            return
        self.s.place((event.xdata, event.ydata), vis=vis)
        self.commit()

    def on_key(self, event):
        k = event.key
        if k == "q":
            self.s.labels.save(self.out)
            plt.close(self.fig)
            return
        if k == "u":
            self.s.flag("unknown")
            return self.commit()
        if k == "o":
            self.s.flag("out")
            return self.commit()
        if k in ("1", "2", "3"):
            self.s.select(self.s.keypoints[int(k) - 1])
        elif k == "backspace":
            self.s.clear()
            self.s.labels.save(self.out)
        elif k in ("n", "right"):
            self.s.next()
        elif k in ("p", "left"):
            self.s.prev()
        elif k in ("N", "up"):
            self.s.next_episode()
        elif k in ("P", "down"):
            self.s.prev_episode()
        else:
            return
        self.redraw()

    def commit(self):
        self.s.labels.save(self.out)
        self.redraw()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--frames", default=None,
                   help="selection file (default labels/label_frames_<dataset>.json)")
    p.add_argument("--out", default=None, help="default labels/gripper_gt_<dataset>.json")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="only these episodes (default: the selection's pilot-first order)")
    p.add_argument("--relabel", type=int, default=0, metavar="N",
                   help="blind re-label N random labelled views into <out>_relabel.json")
    args = p.parse_args()

    import json

    name = args.repo_id.split("/")[-1]
    sel_path = args.frames or os.path.join("labels", f"label_frames_{name}.json")
    out = args.out or os.path.join("labels", f"gripper_gt_{name}.json")
    with open(sel_path) as f:
        selection = json.load(f)
    index = EpisodeIndex(args.repo_id, args.root)
    tasks = build_tasks(selection, index.views, args.episodes)

    if args.relabel:
        first = LabelSet.load(out)
        done = [t for t in tasks if first.is_complete(t.episode, t.frame, t.view)]
        if len(done) < args.relabel:
            raise SystemExit(f"only {len(done)} labelled views; label more first")
        tasks = sorted(random.Random(0).sample(done, args.relabel),
                       key=lambda t: (t.episode, t.frame, t.view))
        out = out.replace(".json", "_relabel.json")

    labels = LabelSet.load_or_new(out, args.repo_id)
    session = LabelSession(labels, tasks)
    print(f"{len(tasks)} views to label, {session.n_complete()} done; writing {out}")
    LabelApp(session, index, out)
    plt.show()
    print(f"{session.n_complete()}/{len(tasks)} views labelled -> {out}")


if __name__ == "__main__":
    main()
