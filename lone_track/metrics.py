"""Scoring tracked points against hand labels. Pure numpy: no GPU, no dataset.

All errors are Euclidean pixel distances in the stored 640x360 frame. NaN means
"no comparison possible" (no label position, or no prediction) and is dropped
by every summary rather than counted as either a hit or a miss.
"""

import numpy as np

PCK_THRESHOLDS = (5.0, 10.0, 20.0)
FAIL_PX = 30.0


def point_errors(pred, gt):
    """(F,) distances between (F,2) predictions and labels; NaN where either is missing."""
    return np.linalg.norm(np.asarray(pred, float) - np.asarray(gt, float), axis=-1)


def match_tip_pair(pred_1, pred_2, gt_1, gt_2):
    """Per frame, pair the two predicted tips with the two labelled tips the cheaper way.

    The jaws have no visible identity, so a label's tip_1 may be the tracker's
    tip_2. Returns (matched_pred_1, matched_pred_2, swapped) where matched_pred_k
    is the prediction compared against gt_k. A frame with one tip missing is
    paired by the available one.
    """
    p1, p2, g1, g2 = (np.asarray(a, float) for a in (pred_1, pred_2, gt_1, gt_2))
    keep = np.nansum([point_errors(p1, g1), point_errors(p2, g2)], axis=0)
    swap = np.nansum([point_errors(p2, g1), point_errors(p1, g2)], axis=0)
    swapped = swap < keep
    m1 = np.where(swapped[:, None], p2, p1)
    m2 = np.where(swapped[:, None], p1, p2)
    return m1, m2, swapped


def summarize(errors):
    """Headline numbers for a set of errors (px)."""
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    if not len(e):
        return {"n": 0}
    out = {
        "n": int(len(e)),
        "median": float(np.median(e)),
        "mean": float(e.mean()),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
        "fail_rate": float(np.mean(e > FAIL_PX)),
    }
    for t in PCK_THRESHOLDS:
        out[f"pck{int(t)}"] = float(np.mean(e <= t))
    return out


def bootstrap_ci(errors_by_episode, stat=np.median, n_boot=2000, alpha=0.05, seed=0):
    """CI for `stat` over all errors, resampling whole episodes (errors within an
    episode are correlated, so frames are not independent samples)."""
    groups = [np.asarray(e, float)[np.isfinite(e)] for e in errors_by_episode]
    groups = [g for g in groups if len(g)]
    if not groups:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(stat(np.concatenate([groups[i] for i in pick])))
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def drift_curve(errors, frames_since_init, edges=(0, 25, 50, 100, 200, 300, 500), fps=25):
    """Median error in bins of time since the tracker was initialised."""
    e = np.asarray(errors, float)
    d = np.asarray(frames_since_init, float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = np.isfinite(e) & (d > lo) & (d <= hi)
        rows.append({"from_s": lo / fps, "to_s": hi / fps, "n": int(sel.sum()),
                     "median": float(np.median(e[sel])) if sel.any() else np.nan,
                     "fail_rate": float(np.mean(e[sel] > FAIL_PX)) if sel.any() else np.nan})
    return rows


def time_to_failure(errors, frames, start, fail_px=FAIL_PX):
    """Frames from `start` to the first labelled frame with error > fail_px (None if never)."""
    e = np.asarray(errors, float)
    f = np.asarray(frames)
    bad = np.flatnonzero((f > start) & (e > fail_px))
    return int(f[bad[0]] - start) if len(bad) else None


def runs(mask):
    """(start, stop) index pairs of the True runs in a boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def still_jitter(track, still, min_len=5):
    """How much a track moves while the arm is commanded still. Label-free.

    Returns {"std": RMS deviation from each still window's mean position,
             "step": median frame-to-frame displacement,
             "flicker": median |x_t - x_{t-2}| / median |x_t - x_{t-1}| --
                        ~1 for white noise, << 1 when frames alternate (the AV1
                        stream has a keyframe every 2 frames)}
    over still windows of at least `min_len` frames. The arm may still creep
    under a zero command, so this bounds tracker noise from above.
    """
    xy = np.asarray(track, float)
    devs, steps, steps2 = [], [], []
    for a, b in runs(still):
        if b - a < min_len:
            continue
        w = xy[a:b]
        w = w[np.all(np.isfinite(w), axis=1)]
        if len(w) < min_len:
            continue
        devs.append(w - w.mean(0))
        steps.append(np.linalg.norm(np.diff(w, axis=0), axis=1))
        steps2.append(np.linalg.norm(w[2:] - w[:-2], axis=1))
    if not devs:
        return {"std": np.nan, "step": np.nan, "flicker": np.nan, "n_windows": 0}
    d = np.concatenate(devs)
    s1 = np.median(np.concatenate(steps))
    s2 = np.median(np.concatenate(steps2))
    return {"std": float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))),
            "step": float(s1), "flicker": float(s2 / s1) if s1 > 0 else np.nan,
            "n_windows": len(devs)}


def auroc(score, positive):
    """Area under the ROC of `score` separating `positive` (True) from negative.

    Used with score = -confidence and positive = (error > 20 px): how well a
    tracker's own confidence flags its failures. Mann-Whitney U, ties halved.
    """
    s = np.asarray(score, float)
    y = np.asarray(positive, bool)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
