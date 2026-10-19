"""Two-view geometry without calibration. Pure numpy/cv2.

The two cameras are fixed, so one fundamental matrix F relates every pair of
(front, side) observations of the same 3D point: x_side^T F x_front = 0. F is
fitted from labelled correspondences -- no checkerboard -- and then the
distance of a point from its epipolar line is a label-free health check on any
tracker output: a point tracked correctly in both views lies on its partner's
line, a point that jumped to the forearm in one view generally does not.

Also here: a robust 2D similarity fit, used to carry a keypoint along with a
cluster of tracked points on the rigid gripper body.
"""

import cv2
import numpy as np


def fit_fundamental(pts_a, pts_b, method="lmeds"):
    """F with pts_b^T F pts_a = 0, from (N,2) correspondences (N >= 8). Returns (F, inlier mask)."""
    a = np.asarray(pts_a, np.float64)
    b = np.asarray(pts_b, np.float64)
    keep = np.all(np.isfinite(a), 1) & np.all(np.isfinite(b), 1)
    a, b = a[keep], b[keep]
    if len(a) < 8:
        raise ValueError(f"need >= 8 correspondences, got {len(a)}")
    flag = {"lmeds": cv2.FM_LMEDS, "ransac": cv2.FM_RANSAC, "8point": cv2.FM_8POINT}[method]
    F, inl = cv2.findFundamentalMat(a, b, flag, 1.0, 0.999)
    if F is None or F.shape != (3, 3):
        raise ValueError("fundamental matrix fit failed (degenerate correspondences?)")
    mask = np.zeros(len(keep), bool)
    mask[np.flatnonzero(keep)] = inl.ravel().astype(bool) if inl is not None else True
    return F / np.linalg.norm(F), mask


def _h(p):
    p = np.asarray(p, np.float64)
    return np.concatenate([p, np.ones(p.shape[:-1] + (1,))], -1)


def epipolar_distance(F, pts_a, pts_b):
    """Symmetric point-to-epipolar-line distance in px, (N,): mean of the two images."""
    a, b = _h(pts_a), _h(pts_b)
    la = a @ F.T          # lines in image b
    lb = b @ F            # lines in image a
    db = np.abs(np.sum(la * b, -1)) / np.linalg.norm(la[..., :2], axis=-1)
    da = np.abs(np.sum(lb * a, -1)) / np.linalg.norm(lb[..., :2], axis=-1)
    return (da + db) / 2


def sampson_distance(F, pts_a, pts_b):
    """First-order geometric error (px) of each correspondence, (N,)."""
    a, b = _h(pts_a), _h(pts_b)
    Fa = a @ F.T
    Ftb = b @ F
    num = np.sum(b * Fa, -1) ** 2
    den = Fa[..., 0] ** 2 + Fa[..., 1] ** 2 + Ftb[..., 0] ** 2 + Ftb[..., 1] ** 2
    return np.sqrt(num / den)


def fit_similarity(src, dst, thresh=3.0):
    """Robust 2D similarity (rotation, uniform scale, translation) src -> dst.

    Returns (2x3 matrix or None, inlier mask). Needs >= 2 finite pairs.
    """
    s = np.asarray(src, np.float64)
    d = np.asarray(dst, np.float64)
    keep = np.all(np.isfinite(s), 1) & np.all(np.isfinite(d), 1)
    mask = np.zeros(len(s), bool)
    if keep.sum() < 2:
        return None, mask
    M, inl = cv2.estimateAffinePartial2D(s[keep], d[keep], method=cv2.RANSAC,
                                         ransacReprojThreshold=thresh, maxIters=500)
    if M is None:
        return None, mask
    mask[np.flatnonzero(keep)] = inl.ravel().astype(bool)
    return M, mask


def apply_affine(M, pts):
    p = np.asarray(pts, np.float64)
    return p @ M[:, :2].T + M[:, 2]


def carry_keypoints(cluster_xy, cluster_vis, keypoints_q, t_q, thresh=3.0, min_inliers=3):
    """Per frame, map keypoints given on frame t_q through the similarity fitted
    from the cluster's positions on t_q to its positions on that frame.

    cluster_xy (T,K,2), cluster_vis (T,K) bool, keypoints_q (N,2).
    Returns (T,N,2) with NaN where fewer than `min_inliers` cluster points
    survived, and (T,) inlier counts.
    """
    T = len(cluster_xy)
    out = np.full((T, len(keypoints_q), 2), np.nan)
    n_in = np.zeros(T, int)
    src = cluster_xy[t_q]
    for t in range(T):
        dst = np.where(cluster_vis[t][:, None], cluster_xy[t], np.nan)
        M, inl = fit_similarity(src, dst, thresh=thresh)
        n_in[t] = inl.sum()
        if M is not None and n_in[t] >= min_inliers:
            out[t] = apply_affine(M, keypoints_q)
    return out, n_in
