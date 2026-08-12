"""Dataset curation helpers -- deleting a reviewed episode.

Ties together the three places an episode index shows up: the Zarr
arrays (LoneReplayBuffer), metadata.json (DatasetMetadata), and the raw
per-episode MP4 backup on disk. All three are re-indexed together so
episode N always means the same thing across all of them after a delete.
"""

import os


def delete_episode(replay_buffer, metadata, raw_dir, episode_idx):
    """Deletes episode `episode_idx` and shifts every later episode's
    index down by one everywhere (Zarr arrays, metadata.json, raw MP4
    filenames). Returns the number of steps removed."""
    n_episodes_before = replay_buffer.n_episodes
    removed = replay_buffer.delete_episode(episode_idx)
    metadata.remove_episode(episode_idx)

    old_path = os.path.join(raw_dir, f"episode_{episode_idx:06d}.mp4")
    if os.path.exists(old_path):
        os.remove(old_path)

    # Shift every later episode's video down by one index to match.
    for i in range(episode_idx + 1, n_episodes_before):
        src = os.path.join(raw_dir, f"episode_{i:06d}.mp4")
        dst = os.path.join(raw_dir, f"episode_{i - 1:06d}.mp4")
        if os.path.exists(src):
            os.replace(src, dst)

    metadata.save()
    return removed
