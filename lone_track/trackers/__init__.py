"""Tracker wrappers behind one interface (see base.py), built by name.

Imports are lazy: each wrapper pulls in its own third-party package only when
built, so lone_track stays importable in an env that has none of them.
"""

import os

CKPT_DIR = os.path.join("outputs", "track", "ckpts")


def build(name, device="cuda", ckpt_dir=CKPT_DIR):
    """A tracker from its benchmark name, e.g. 'cotracker3_offline' or 'sam2_large_online'."""
    ck = lambda f: os.path.join(ckpt_dir, f)  # noqa: E731
    if name == "cotracker3_offline":
        from lone_track.trackers.cotracker import CoTracker3Offline
        return CoTracker3Offline(ck("cotracker3_offline.pth"), device)
    if name == "cotracker3_offline_lg3":
        from lone_track.trackers.cotracker import CoTracker3Offline
        return CoTracker3Offline(ck("cotracker3_offline.pth"), device, local_grid=3)
    if name == "cotracker3_online":
        from lone_track.trackers.cotracker import CoTracker3Online
        return CoTracker3Online(ck("cotracker3_online.pth"), device)
    if name == "bootstapir":
        from lone_track.trackers.tapir import BootsTAPIR
        return BootsTAPIR(ck("bootstapir_checkpoint_v2.pt"), device)
    if name == "bootstapir_512x512":
        from lone_track.trackers.tapir import BootsTAPIR
        return BootsTAPIR(ck("bootstapir_checkpoint_v2.pt"), device, size=(512, 512))
    if name == "bootstapir_online":
        from lone_track.trackers.tapir import OnlineBootsTAPIR
        return OnlineBootsTAPIR(ck("causal_bootstapir_checkpoint.pt"), device)
    if name == "tapnextpp":
        from lone_track.trackers.tapnext import TAPNextPP
        return TAPNextPP(ck("tapnextpp_ckpt.pt"), device)
    if name.startswith("sam2_"):
        from lone_track.trackers.sam2 import SAM2Video
        size = name.split("_")[1]
        return SAM2Video(ck(f"sam2.1_hiera_{size}.pt"), size=size, device=device,
                         causal=name.endswith("_online"))
    if name == "lk":
        from lone_track.trackers.lk import LucasKanade
        return LucasKanade()
    raise KeyError(f"unknown tracker {name!r}; known: {', '.join(NAMES)}")


NAMES = (
    "cotracker3_offline", "cotracker3_offline_lg3", "cotracker3_online",
    "bootstapir", "bootstapir_512x512", "bootstapir_online", "tapnextpp",
    "sam2_large", "sam2_large_online", "sam2_small_online", "lk",
)
