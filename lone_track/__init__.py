"""Vision-based gripper tracking for the L-ONE.

The arm has no encoders, so the cameras are the only source of state. This
package locates the gripper in both views -- off-the-shelf point trackers,
SAM2, and later a distilled per-frame detector -- and scores them against
hand-labelled frames of a recorded dataset, as the feasibility study for an
end-effector action space closed by a PD loop on 2D keypoints.

Importing lone_data first is deliberate: it sets the HF offline flags before
anything imports lerobot (see lone_data/__init__.py).
"""

import lone_data  # noqa: F401
