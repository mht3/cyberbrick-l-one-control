"""The labelling tool's state machine, kept free of any GUI so it can be tested.

A session is an ordered list of tasks -- one (episode, frame, view) each -- and a
cursor. Within a task the annotator places the keypoints in order (body, tip_1,
tip_2); placing the last one moves to the next task, so the common case is three
clicks per view and no keyboard at all. Every edit is written to disk straight
away by the caller (LabelSet.save is atomic).
"""

from dataclasses import dataclass

from lone_track.labels import HAS_XY, LabelSet


@dataclass(frozen=True)
class Task:
    episode: int
    frame: int
    view: str


def build_tasks(selection, views, episodes=None):
    """Tasks in labelling order from a select_label_frames.py document."""
    order = episodes if episodes is not None else selection["order"]
    tasks = []
    for ep in order:
        for frame in selection["frames"][str(ep)]:
            for view in views:
                tasks.append(Task(int(ep), int(frame), view))
    return tasks


class LabelSession:
    def __init__(self, labels: LabelSet, tasks, start_at_first_incomplete=True):
        if not tasks:
            raise ValueError("no frames to label")
        self.labels = labels
        self.tasks = list(tasks)
        self.cursor = 0
        if start_at_first_incomplete:
            self.cursor = self.first_incomplete()
        self.active = self._first_missing_kp()

    # -- where we are --------------------------------------------------------

    @property
    def task(self):
        return self.tasks[self.cursor]

    @property
    def keypoints(self):
        return self.labels.keypoints

    def first_incomplete(self):
        for i, t in enumerate(self.tasks):
            if not self.labels.is_complete(t.episode, t.frame, t.view):
                return i
        return len(self.tasks) - 1

    def n_complete(self):
        return sum(self.labels.is_complete(t.episode, t.frame, t.view) for t in self.tasks)

    def entry(self, kp, task=None):
        t = task or self.task
        return self.labels.get(t.episode, t.frame, t.view, kp)

    def _first_missing_kp(self):
        for kp in self.keypoints:
            if self.entry(kp) is None:
                return kp
        return self.keypoints[0]

    # -- edits ---------------------------------------------------------------

    def place(self, xy, vis="visible"):
        """Put the active keypoint at xy; returns True if the cursor moved on."""
        if vis not in HAS_XY:
            raise ValueError(f"{vis!r} has no position; use flag()")
        t = self.task
        self.labels.set(t.episode, t.frame, t.view, self.active, xy=xy, vis=vis)
        return self._after_edit()

    def flag(self, vis):
        """Mark the active keypoint as having no position ('unknown' or 'out')."""
        t = self.task
        self.labels.set(t.episode, t.frame, t.view, self.active, xy=None, vis=vis)
        return self._after_edit()

    def clear(self):
        t = self.task
        self.labels.clear(t.episode, t.frame, t.view, self.active)

    def select(self, kp):
        if kp not in self.keypoints:
            raise KeyError(kp)
        self.active = kp

    def _after_edit(self):
        """Advance to the next keypoint still missing here, else the next task."""
        missing = [kp for kp in self.keypoints if self.entry(kp) is None]
        if missing:
            # Prefer the next one after the active keypoint, wrapping around.
            i = self.keypoints.index(self.active)
            ring = self.keypoints[i + 1:] + self.keypoints[:i + 1]
            self.active = next(kp for kp in ring if kp in missing)
            return False
        if self.cursor < len(self.tasks) - 1:
            self.goto(self.cursor + 1)
            return True
        return False

    # -- navigation ----------------------------------------------------------

    def goto(self, index):
        self.cursor = max(0, min(len(self.tasks) - 1, index))
        self.active = self._first_missing_kp()

    def next(self):
        self.goto(self.cursor + 1)

    def prev(self):
        self.goto(self.cursor - 1)

    def next_episode(self):
        ep = self.task.episode
        for i in range(self.cursor + 1, len(self.tasks)):
            if self.tasks[i].episode != ep:
                return self.goto(i)

    def prev_episode(self):
        ep = self.task.episode
        starts = [i for i, t in enumerate(self.tasks)
                  if t.episode != ep and (i == 0 or self.tasks[i - 1].episode != t.episode)
                  and i < self.cursor]
        if starts:
            self.goto(starts[-1])
