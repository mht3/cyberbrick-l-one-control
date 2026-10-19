"""Hand-labelled gripper keypoints: schema, JSON I/O, and the derived grasp point.

Only physical points are clicked -- a tracker can follow a surface point but not
the empty space between open jaws:

    body   centre of the black gripper servo (the one that opens the jaws)
    tip_1  one fingertip
    tip_2  the other fingertip

The jaws have no visible identity, so the tips are an *unordered* pair: scoring
matches them to predictions by the cheaper of the two pairings
(metrics.match_tip_pair). The grasp point -- what has to reach the marker -- is
the midpoint of the two tips. For a ~3 cm gripper half a metre from the camera,
the midpoint of the two projections and the projection of the 3D midpoint
differ by far less than a pixel, so it is computed per view, no calibration.

Coordinates are pixels in the stored 640x360 frame, OpenCV convention: the
centre of the top-left pixel is (0, 0). matplotlib's imshow uses the same
convention, which is what the labelling tool clicks in.

Each labelled point carries a visibility flag:
    visible   seen, clicked where it is
    occluded  hidden, but the annotator is confident where it is (scored separately)
    unknown   hidden, no estimate (no coordinates)
    out       outside the frame (no coordinates)

On disk (labels/<name>.json):
    {"schema": 1, "repo_id": ..., "keypoints": [...],
     "frames": {"<episode>:<frame>": {"<view>": {"<kp>": {"xy": [x, y] | null,
                                                          "vis": flag}}}}}
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 1
KEYPOINTS = ("body", "tip_1", "tip_2")
KEYPOINT_HELP = {
    "body": "centre of the black gripper servo",
    "tip_1": "a fingertip (either jaw)",
    "tip_2": "the other fingertip",
}
VIS_FLAGS = ("visible", "occluded", "unknown", "out")
# Flags that come with a position.
HAS_XY = ("visible", "occluded")


def frame_key(episode, frame):
    return f"{int(episode)}:{int(frame)}"


def parse_frame_key(key):
    episode, frame = key.split(":")
    return int(episode), int(frame)


@dataclass
class LabelSet:
    repo_id: str
    keypoints: tuple = KEYPOINTS
    frames: dict = field(default_factory=dict)

    # -- editing -----------------------------------------------------------

    def set(self, episode, frame, view, kp, xy=None, vis="visible"):
        if kp not in self.keypoints:
            raise KeyError(f"unknown keypoint {kp!r}")
        if vis not in VIS_FLAGS:
            raise ValueError(f"unknown visibility {vis!r}")
        if (vis in HAS_XY) != (xy is not None):
            raise ValueError(f"visibility {vis!r} {'needs' if vis in HAS_XY else 'takes no'} xy")
        entry = {"xy": None if xy is None else [float(xy[0]), float(xy[1])], "vis": vis}
        self.frames.setdefault(frame_key(episode, frame), {}).setdefault(view, {})[kp] = entry

    def clear(self, episode, frame, view, kp):
        views = self.frames.get(frame_key(episode, frame), {})
        views.get(view, {}).pop(kp, None)

    def get(self, episode, frame, view, kp):
        return self.frames.get(frame_key(episode, frame), {}).get(view, {}).get(kp)

    def is_complete(self, episode, frame, view):
        got = self.frames.get(frame_key(episode, frame), {}).get(view, {})
        return all(kp in got for kp in self.keypoints)

    # -- reading -----------------------------------------------------------

    def episodes(self):
        return sorted({parse_frame_key(k)[0] for k in self.frames})

    def labelled_frames(self, episode):
        return sorted(f for e, f in map(parse_frame_key, self.frames) if e == episode)

    def arrays(self, episode, view, kp):
        """(frames, xy (N,2) with NaN where no position, vis flags (N,)) for one track."""
        frames = self.labelled_frames(episode)
        xy = np.full((len(frames), 2), np.nan)
        vis = np.full(len(frames), "", dtype=object)
        for i, f in enumerate(frames):
            entry = self.get(episode, f, view, kp)
            if entry is None:
                continue
            vis[i] = entry["vis"]
            if entry["xy"] is not None:
                xy[i] = entry["xy"]
        return np.asarray(frames), xy, vis

    def grasp(self, episode, view):
        """(frames, xy) of the tip midpoint; NaN unless both tips have a position."""
        frames, a, _ = self.arrays(episode, view, "tip_1")
        _, b, _ = self.arrays(episode, view, "tip_2")
        return frames, (a + b) / 2.0

    # -- I/O ---------------------------------------------------------------

    def to_json(self):
        return {"schema": SCHEMA, "repo_id": self.repo_id,
                "keypoints": list(self.keypoints), "frames": self.frames}

    @classmethod
    def from_json(cls, data):
        if data.get("schema") != SCHEMA:
            raise ValueError(f"label schema {data.get('schema')!r}, expected {SCHEMA}")
        return cls(repo_id=data["repo_id"], keypoints=tuple(data["keypoints"]),
                   frames=data["frames"])

    def save(self, path):
        """Atomic write: the labelling tool saves after every click."""
        tmp = f"{path}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f, indent=1, sort_keys=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_json(json.load(f))

    @classmethod
    def load_or_new(cls, path, repo_id):
        if os.path.exists(path):
            labels = cls.load(path)
            if labels.repo_id != repo_id:
                raise ValueError(f"{path} labels {labels.repo_id}, not {repo_id}")
            return labels
        return cls(repo_id=repo_id)
