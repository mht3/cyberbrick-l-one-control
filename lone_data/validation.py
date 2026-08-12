"""Data-integrity checks for a LoneReplayBuffer dataset.

Both functions return a list of human-readable warning strings (empty
list == all checks passed). Checks are non-fatal by design -- the
recorder logs them but doesn't block saving, since a corrupt episode is
still useful to know about after the fact.
"""

import numpy as np

from lone_data.replay_buffer import ACTION_DIM

MP4_FRAME_COUNT_TOLERANCE = 2  # frames


def validate_episode(replay_buffer, start_idx, end_idx, mp4_path=None):
    warnings = []
    ep_len = end_idx - start_idx
    if ep_len <= 0:
        return [f"Episode [{start_idx}:{end_idx}] has non-positive length {ep_len}"]

    img_len = replay_buffer.img.shape[0]
    action_len = replay_buffer.action.shape[0]
    ts_len = replay_buffer.timestamp.shape[0]
    if not (img_len == action_len == ts_len):
        warnings.append(
            f"Array length mismatch: img={img_len} action={action_len} timestamp={ts_len}"
        )

    timestamps = replay_buffer.timestamp[start_idx:end_idx]
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        warnings.append(f"Episode [{start_idx}:{end_idx}] timestamps are not strictly increasing")

    action_shape = replay_buffer.action.shape[1:]
    if action_shape != (ACTION_DIM,):
        warnings.append(f"Action dim {action_shape} != expected ({ACTION_DIM},)")

    if mp4_path is not None:
        try:
            import cv2

            cap = cv2.VideoCapture(mp4_path)
            mp4_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if abs(mp4_frames - ep_len) > MP4_FRAME_COUNT_TOLERANCE:
                warnings.append(
                    f"MP4 frame count {mp4_frames} differs from recorded step count {ep_len} "
                    f"by more than {MP4_FRAME_COUNT_TOLERANCE}"
                )
        except Exception as e:
            warnings.append(f"Could not verify MP4 frame count for {mp4_path}: {e}")

    return warnings


def validate_dataset(replay_buffer):
    warnings = []
    ends = replay_buffer.episode_ends
    n_steps = replay_buffer.n_steps

    if len(ends) == 0:
        if n_steps > 0:
            warnings.append(f"{n_steps} steps recorded but 0 episodes in meta/episode_ends")
        return warnings

    if not np.all(np.diff(ends) > 0) or ends[0] <= 0:
        warnings.append(f"meta/episode_ends is not strictly increasing: {ends.tolist()}")

    if ends[-1] != n_steps:
        warnings.append(
            f"Last episode_ends value {ends[-1]} != total step count {n_steps} "
            "(dataset may have orphaned/untracked rows)"
        )

    img_dtype = replay_buffer.img.dtype
    if img_dtype != np.uint8:
        warnings.append(f"data/img dtype is {img_dtype}, expected uint8")

    action_dtype = replay_buffer.action.dtype
    if action_dtype != np.float32:
        warnings.append(f"data/action dtype is {action_dtype}, expected float32")

    ts_dtype = replay_buffer.timestamp.dtype
    if ts_dtype != np.float64:
        warnings.append(f"data/timestamp dtype is {ts_dtype}, expected float64")

    if replay_buffer.action.shape[1:] != (ACTION_DIM,):
        warnings.append(f"data/action shape[1:] is {replay_buffer.action.shape[1:]}, expected ({ACTION_DIM},)")

    return warnings
