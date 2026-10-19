"""Draw tracks on frames: overlay videos for eyeballing, contact sheets for reports."""

import cv2
import numpy as np

# RGB, one per keypoint (body, tip_1, tip_2), then the cluster.
KP_COLOURS = [(255, 140, 0), (0, 200, 255), (255, 60, 224)]
CLUSTER_COLOUR = (120, 255, 120)
LABEL_COLOUR = (255, 255, 255)


def draw(frame, xy, vis=None, n_kp=3, labels_xy=None, text=None, radius=4):
    """One frame with keypoints (filled if visible, hollow if not), cluster dots,
    and optional labels (white crosses)."""
    img = np.ascontiguousarray(frame.copy())
    for i, (x, y) in enumerate(np.asarray(xy)):
        if not np.isfinite(x):
            continue
        c = KP_COLOURS[i] if i < n_kp else CLUSTER_COLOUR
        r = radius if i < n_kp else 1
        thick = -1 if (vis is None or vis[i]) else 1
        cv2.circle(img, (int(round(x)), int(round(y))), r, c, thick, cv2.LINE_AA)
    if labels_xy is not None:
        for x, y in np.asarray(labels_xy):
            if np.isfinite(x):
                cv2.drawMarker(img, (int(round(x)), int(round(y))), LABEL_COLOUR,
                               cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    if text:
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1,
                    cv2.LINE_AA)
    return img


def contact_sheet(rows, frame_ids, crop=None, scale=1.0):
    """rows: list of (name, frames, TrackResult); one row per tracker, one column per frame."""
    tiles = []
    for name, frames, res in rows:
        row = []
        for t in frame_ids:
            img = draw(frames[t], res.xy[t], res.vis[t], text=f"{name} t={t}")
            if crop is not None:
                x0, y0, x1, y1 = crop
                img = img[y0:y1, x0:x1]
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            row.append(img)
        tiles.append(np.concatenate(row, 1))
    return np.concatenate(tiles, 0)


def write_video(path, frames, fps=25):
    """H.264 mp4 via PyAV (already a dependency through lerobot)."""
    import av

    with av.open(path, "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20"}
        for f in frames:
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
