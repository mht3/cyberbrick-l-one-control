"""lone_track.geometry on a synthetic two-camera rig with known projections.

No hardware, dataset or checkpoint is touched.
"""

import numpy as np
import pytest

from lone_track.geometry import (
    carry_keypoints,
    epipolar_distance,
    fit_fundamental,
    fit_similarity,
    sampson_distance,
)


def _project(P, X):
    x = np.c_[X, np.ones(len(X))] @ P.T
    return x[:, :2] / x[:, 2:]


def _rig():
    """A front and a side camera ~90 degrees apart looking at a 30 cm workspace."""
    K = np.array([[460.0, 0, 320], [0, 460, 180], [0, 0, 1]])
    def cam(yaw, t):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        return K @ np.c_[R, t]
    return cam(0.0, [0, 0, 0.6]), cam(np.pi / 2, [0, 0, 0.6])


def test_fundamental_from_correspondences_puts_points_on_their_lines():
    Pa, Pb = _rig()
    rng = np.random.default_rng(0)
    X = rng.uniform(-0.15, 0.15, (60, 3))
    a, b = _project(Pa, X), _project(Pb, X)
    F, inl = fit_fundamental(a + rng.normal(0, 0.3, a.shape), b + rng.normal(0, 0.3, b.shape))
    assert inl.mean() > 0.8
    Xt = rng.uniform(-0.15, 0.15, (20, 3))
    at, bt = _project(Pa, Xt), _project(Pb, Xt)
    assert np.median(epipolar_distance(F, at, bt)) < 1.0
    assert np.median(sampson_distance(F, at, bt)) < 1.0
    # A point that jumped 25 px in one view is far from its line.
    assert np.median(epipolar_distance(F, at + [25, -25], bt)) > 5


def test_fundamental_needs_eight_points():
    with pytest.raises(ValueError):
        fit_fundamental(np.zeros((5, 2)), np.zeros((5, 2)))


def test_similarity_recovers_rotation_and_rejects_an_outlier():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 50, (10, 2))
    th = 0.3
    R = 1.1 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = src @ R.T + [5, -3]
    dst[0] += 40  # one point slipped onto the forearm
    M, inl = fit_similarity(src, dst)
    assert not inl[0] and inl[1:].all()
    assert np.allclose(M[:, :2], R, atol=1e-6) and np.allclose(M[:, 2], [5, -3], atol=1e-6)


def test_carry_keypoints_follows_a_translating_cluster():
    T, K = 5, 6
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 40, (K, 2))
    shift = np.arange(T)[:, None, None] * np.array([2.0, 1.0])
    cluster = base[None] + shift
    vis = np.ones((T, K), bool)
    vis[3, :4] = False                     # too few points survive on frame 3
    kp = np.array([[20.0, 20.0]])
    out, n_in = carry_keypoints(cluster, vis, kp, t_q=0)
    assert np.allclose(out[4, 0], kp[0] + [8, 4])
    assert np.isnan(out[3]).all() and n_in[3] == 2
