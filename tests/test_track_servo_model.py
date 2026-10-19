"""lone_track.servo_model on a simulated L-ONE seen by two pinhole cameras.

A yaw/shoulder/elbow arm with speed-commanded joints, driven by bang-bang
single-joint commands like the teleop data, with a known 2-tick command delay,
projected into a front and a side camera. The analysis must recover the image
Jacobian, the delay, and flag the upright pose (gripper on the yaw axis) as
singular. No hardware, dataset or checkpoint is touched.
"""

import numpy as np

from lone_track import servo_model as S

FPS = 25
DELAY = 2
RATES = np.array([1.0, 0.8, 0.8])          # rad/s at full command
L1, L2, H0 = 0.12, 0.12, 0.08
K = np.array([[460.0, 0, 320], [0, 460, 180], [0, 0, 1]])


def _camera(C, look):
    """3x4 projection for a camera at C looking along `look`, z up in the world."""
    f = np.asarray(look, float) / np.linalg.norm(look)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f])
    return K @ np.c_[R, -R @ np.asarray(C, float)]


CAMS = [_camera([0.55, 0.0, 0.16], [-1, 0, -0.15]), _camera([0.0, -0.55, 0.16], [0, 1, -0.15])]


def ee(q):
    q0, q1, q2 = q
    r = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    z = H0 + L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return np.array([r * np.cos(q0), r * np.sin(q0), z])


def observe(q):
    X = np.r_[ee(q), 1.0]
    out = []
    for P in CAMS:
        x = P @ X
        out += [x[0] / x[2], x[1] / x[2]]
    return np.array(out)


def true_J(q, eps=1e-5):
    cols = []
    for j in range(3):
        dq = np.zeros(3)
        dq[j] = eps
        cols.append((observe(q + dq) - observe(q - dq)) / (2 * eps) * RATES[j])
    return np.stack(cols, 1)


def simulate(n_episodes=30, T=300, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    states, acts = [], []
    for _ in range(n_episodes):
        q = np.array([rng.uniform(-0.6, 0.6), rng.uniform(0.6, 1.1), rng.uniform(-1.6, -0.9)])
        a = np.zeros((T, 4))
        t = 0
        while t < T:
            t += rng.integers(3, 16)
            j, n = rng.integers(0, 3), rng.integers(6, 20)
            a[t:t + n, j] = rng.choice([-1, 1]) * S.FULL[j]
            t += n
        s = np.zeros((T, 4))
        for i in range(T):
            s[i] = observe(q)
            u = a[i - DELAY, :3] / S.FULL if i >= DELAY else np.zeros(3)
            q = q + u * RATES / FPS
            q[1] = np.clip(q[1], 0.4, 1.3)
            q[2] = np.clip(q[2], -1.9, -0.7)
        states.append(s + rng.normal(0, noise, s.shape))
        acts.append(a)
    return states, acts


def _fit(states, acts):
    vel = [S.velocity(s, FPS) for s in states]
    samples = [S.single_joint_samples(a, settle=DELAY + 2) for a in acts]
    return S.JacobianField(k=12).fit(states, vel, samples), vel


def test_jacobian_columns_recovered_without_calibration():
    states, acts = simulate()
    field, _ = _fit(states, acts)
    assert all(n > 100 for n in field.n_samples().values())
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(40):
        q = np.array([rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0), rng.uniform(-1.5, -1.0)])
        Jt, Je = true_J(q), field.J(observe(q)[None])[0]
        errs.append(np.linalg.norm(Je - Jt, axis=0) / np.linalg.norm(Jt, axis=0))
    assert np.median(errs) < 0.15


def test_open_loop_prediction_beats_static_baseline():
    states, acts = simulate()
    field, _ = _fit(states[:20], acts[:20])
    scores = S.prediction_scores(field, states[20:], acts[20:], delay=DELAY, horizons=(10, 25))
    assert scores[10]["n"] > 50
    assert scores[10]["ratio"] < 0.3 and scores[25]["ratio"] < 0.4
    # The wrong delay predicts worse than the right one.
    wrong = S.prediction_scores(field, states[20:], acts[20:], delay=DELAY + 4, horizons=(10,))
    assert wrong[10]["model_median"] > scores[10]["model_median"]


def test_step_response_finds_the_command_delay():
    states, acts = simulate()
    vel = [S.velocity(s, FPS) for s in states]
    r = S.step_response(vel, acts, joint=2)
    assert r["n_onsets"] > 20
    assert DELAY - 1 <= r["onset_lag"] <= DELAY + 1
    assert DELAY - 1 <= r["coast"] <= DELAY + 1     # the sim stops dead after the delay


def test_upright_pose_is_flagged_singular():
    q_reach = np.array([0.0, 0.8, -1.2])
    q_upright = np.array([0.0, np.pi / 2, 0.0])      # gripper on the yaw axis
    sv_reach = S.conditioning(true_J(q_reach)[None])[0]
    sv_up = S.conditioning(true_J(q_upright)[None])[0]
    assert sv_reach[2] > 20                          # px/s: all three joints visible
    assert sv_up[2] < 0.05 * sv_reach[2]             # base yaw moves nothing


def test_gain_bounds():
    g = S.gain_bounds(0.15)
    assert np.isclose(g["k_max"], np.pi / 0.3) and g["k_60deg"] < g["k_max"]


def test_fill_gaps_bridges_short_runs_only():
    s = np.c_[np.arange(10.0), np.arange(10.0)]
    s[3:5, 0] = np.nan          # 2-tick gap: filled
    s[1:6, 1] = np.nan          # 5-tick gap: left alone
    out = S.fill_gaps(s, max_gap=3)
    assert np.allclose(out[:, 0], np.arange(10.0))
    assert np.isnan(out[1:6, 1]).all()
