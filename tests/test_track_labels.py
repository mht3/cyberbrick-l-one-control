"""Label schema, JSON round trip, frame selection and the labelling state machine.

No hardware, dataset or checkpoint is touched; the GUI itself is not exercised,
only the GUI-free session logic it drives.
"""

import importlib.util
import os

import numpy as np
import pytest

from lone_track.label_session import LabelSession, Task, build_tasks
from lone_track.labels import LabelSet


def _load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_round_trip_and_grasp_midpoint(tmp_path):
    labels = LabelSet(repo_id="lone/x")
    labels.set(3, 10, "front", "body", (100, 50))
    labels.set(3, 10, "front", "tip_1", (90, 80))
    labels.set(3, 10, "front", "tip_2", (110, 84), vis="occluded")
    labels.set(3, 20, "front", "tip_1", None, vis="unknown")
    path = tmp_path / "l.json"
    labels.save(path)
    back = LabelSet.load(path)
    assert back.frames == labels.frames
    frames, grasp = back.grasp(3, "front")
    assert frames.tolist() == [10, 20]
    assert np.allclose(grasp[0], [100, 82]) and np.isnan(grasp[1]).all()
    _, xy, vis = back.arrays(3, "front", "tip_1")
    assert vis.tolist() == ["visible", "unknown"] and np.isnan(xy[1]).all()


def test_set_rejects_inconsistent_visibility():
    labels = LabelSet(repo_id="lone/x")
    with pytest.raises(ValueError):
        labels.set(0, 0, "front", "body", None, vis="visible")
    with pytest.raises(ValueError):
        labels.set(0, 0, "front", "body", (1, 2), vis="out")


def test_wrong_dataset_is_refused(tmp_path):
    path = tmp_path / "l.json"
    LabelSet(repo_id="lone/a").save(path)
    with pytest.raises(ValueError):
        LabelSet.load_or_new(path, "lone/b")


def test_frame_selection_keeps_grasp_moments_and_ends():
    sel = _load_script("select_label_frames")
    actions = np.zeros((300, 4), np.float32)
    actions[:, 3] = 120
    actions[150:, 3] = 30                  # gripper closes at t=150
    frames = sel.select_frames(actions, every=60, min_gap=6)
    assert {0, 149, 160, 299} <= set(frames)
    assert all(b - a >= 6 for a, b in zip(frames, frames[1:]))
    assert sel.pilot_first_order(list(range(20)), 5)[:5] == [0, 5, 10, 14, 19]


def test_session_advances_through_keypoints_then_views():
    labels = LabelSet(repo_id="lone/x")
    tasks = build_tasks({"order": [0], "frames": {"0": [0, 40]}}, ["front", "side"])
    assert tasks[:2] == [Task(0, 0, "front"), Task(0, 0, "side")]
    s = LabelSession(labels, tasks)
    assert s.active == "body"
    assert not s.place((1, 1))
    assert s.active == "tip_1"
    s.flag("unknown")
    assert s.place((3, 3), vis="occluded")        # third point -> next view
    assert s.task == Task(0, 0, "side") and s.active == "body"
    assert s.n_complete() == 1


def test_session_resumes_at_first_incomplete_and_fixes_one_point():
    labels = LabelSet(repo_id="lone/x")
    for kp in labels.keypoints:
        labels.set(0, 0, "front", kp, (5, 5))
    tasks = build_tasks({"order": [0], "frames": {"0": [0]}}, ["front", "side"])
    s = LabelSession(labels, tasks)
    assert s.task.view == "side"
    s.prev()
    s.select("tip_2")
    s.place((9, 9))                                # all set already -> moves on
    assert labels.get(0, 0, "front", "tip_2")["xy"] == [9.0, 9.0]
    assert s.task.view == "side"
