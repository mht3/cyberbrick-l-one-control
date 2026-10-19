"""Can a PD loop on 2D keypoints from two cameras drive the arm? The offline evidence.

The model being tested is the one a two-view image-based visual servo assumes:

    s = (u_front, v_front, u_side, v_side)   one gripper keypoint, both views
    ds/dt = J(s) @ (u / u_full)              u: the three speed commands

J(s) is 4x3: column j is the pixel velocity joint j produces at full command.
The teleop data are bang-bang and almost always move one joint at a time, so
every frame where exactly one joint has been commanded for a while measures one
column of J at the current s -- sign included, since -100 should move the
opposite way to +100. Columns are interpolated over s with distance-weighted
nearest neighbours. Nothing here needs camera calibration or depth.

What decides feasibility:
  - J well-conditioned where the task happens (sigma_min clearly above zero):
    the 4 image coordinates then determine 3 joint rates, so a pixel error can
    be turned into speed commands.
  - The model predicting motion open-loop over the next 0.2-1 s much better
    than "nothing moves": then it is good enough to drive a P(D) law or a
    Kalman filter's prediction step.
  - Delay and coast, which bound the gains a loop on these actuators tolerates.

Pure numpy. Frame indices are recorder ticks at `fps`.
"""

import numpy as np

from lone_track.metrics import runs

FULL = np.array([900.0, 100.0, 100.0])  # the teleop's full-speed level per joint
JOINTS = ("base", "upper_arm", "lower_arm")


def fill_gaps(s, max_gap=3):
    """Linear interpolation over NaN runs of at most `max_gap` ticks, per column.

    Used for duplicate (stale) frames: the tick has no fresh image, so its
    position is unknown, not equal to the previous tick's. Longer gaps stay NaN.
    """
    s = np.array(s, float)
    t = np.arange(len(s))
    for c in range(s.shape[1]):
        ok = np.isfinite(s[:, c])
        if ok.sum() < 2:
            continue
        filled = np.interp(t, t[ok], s[ok, c])
        for a, b in runs(~ok):
            if a > 0 and b < len(s) and b - a <= max_gap:
                s[a:b, c] = filled[a:b]
    return s


def velocity(s, fps=25, valid=None):
    """Central-difference velocity in px/s, (T,4); NaN at ends and invalid neighbours."""
    s = np.asarray(s, float)
    v = np.full_like(s, np.nan)
    v[1:-1] = (s[2:] - s[:-2]) * fps / 2
    if valid is not None:
        ok = np.asarray(valid, bool)
        bad = ~(np.r_[False, ok[:-2] & ok[2:], False])
        v[bad] = np.nan
    return v


def single_joint_samples(actions, settle=4):
    """(frame, joint, sign) for frames where exactly one joint is at full command and
    has been for `settle` ticks (past the start-up transient)."""
    a = np.asarray(actions, float)[:, :3]
    out = []
    for j in range(3):
        cmd = np.sign(a[:, j]) * (np.abs(a[:, j]) >= FULL[j] * 0.99)
        others = np.all(a[:, [k for k in range(3) if k != j]] == 0, axis=1)
        for sign in (-1, 1):
            on = (cmd == sign) & others
            for start, stop in runs(on):
                for f in range(start + settle, stop):
                    out.append((f, j, sign))
    return np.array(out, int).reshape(-1, 3)


class JacobianField:
    """Columns of J(s), each a distance-weighted kNN over measured samples."""

    def __init__(self, k=12):
        self.k = k
        self.s = {}
        self.c = {}

    def fit(self, states, velocities, samples):
        """states/velocities (T,4) per episode lists; samples (frame, joint, sign) per episode."""
        S = {j: [] for j in range(3)}
        C = {j: [] for j in range(3)}
        for s, v, smp in zip(states, velocities, samples):
            for f, j, sign in smp:
                if np.all(np.isfinite(s[f])) and np.all(np.isfinite(v[f])):
                    S[j].append(s[f])
                    C[j].append(v[f] * sign)  # velocity per unit +full command
        for j in range(3):
            self.s[j] = np.asarray(S[j]).reshape(-1, 4)
            self.c[j] = np.asarray(C[j]).reshape(-1, 4)
        return self

    def n_samples(self):
        return {JOINTS[j]: len(self.s[j]) for j in range(3)}

    def column(self, j, s):
        """J column j at states s (N,4)."""
        s = np.atleast_2d(np.asarray(s, float))
        if len(self.s[j]) == 0:
            return np.full((len(s), 4), np.nan)
        d = np.linalg.norm(self.s[j][None] - s[:, None], axis=-1)  # (N, M)
        k = min(self.k, d.shape[1])
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d, idx, 1)
        w = 1.0 / (dk + 5.0)          # 5 px softening: neighbours within a few px weigh alike
        w /= w.sum(1, keepdims=True)
        return np.einsum("nk,nkc->nc", w, self.c[j][idx])

    def J(self, s):
        """(N,4,3) Jacobians at states s (N,4)."""
        return np.stack([self.column(j, s) for j in range(3)], axis=-1)


def conditioning(J):
    """Singular values of (N,4,3) Jacobians: (N,3) descending, px/s per full command."""
    J = np.asarray(J, float)
    ok = np.all(np.isfinite(J.reshape(len(J), -1)), axis=1)
    sv = np.full((len(J), 3), np.nan)
    if ok.any():
        sv[ok] = np.linalg.svd(J[ok], compute_uv=False)
    return sv


def predict_open_loop(field, s0, commands, fps=25):
    """Integrate s forward from s0 through the commands acting on the plant, one per
    tick (zero-order hold). Returns (H+1,4) with s[0] = s0."""
    s = np.array(s0, float)
    out = [s.copy()]
    for u in np.asarray(commands, float)[:, :3] / FULL:
        if np.any(u != 0):
            s = s + (field.J(s[None])[0] @ u) / fps
        out.append(s.copy())
    return np.array(out)


def prediction_scores(field, states, actions_list, delay, horizons=(5, 10, 25), stride=5, fps=25):
    """Open-loop prediction error vs the zero-motion baseline, per horizon H (ticks).

    The plant at tick t moves under the command sent at t - delay, so predicting
    s(t+H) from s(t) integrates commands t-delay .. t+H-delay-1. Start frames every
    `stride` ticks, skipping windows with nothing commanded (both predictors are
    then trivially right). Errors are RMS over the two views' 2D errors, in px.
    """
    out = {}
    for H in horizons:
        em, e0 = [], []
        for s, a in zip(states, actions_list):
            for t in range(delay, len(s) - H, stride):
                if not (np.all(np.isfinite(s[t])) and np.all(np.isfinite(s[t + H]))):
                    continue
                cmds = a[t - delay:t + H - delay]
                if not np.any(cmds[:, :3] != 0):
                    continue
                pred = predict_open_loop(field, s[t], cmds, fps)
                em.append(np.linalg.norm(pred[H] - s[t + H]) / np.sqrt(2))
                e0.append(np.linalg.norm(s[t] - s[t + H]) / np.sqrt(2))
        em, e0 = np.array(em), np.array(e0)
        ok = len(em) and np.median(e0) > 0
        out[H] = {"n": int(len(em)),
                  "model_median": float(np.median(em)) if len(em) else np.nan,
                  "static_median": float(np.median(e0)) if len(e0) else np.nan,
                  "ratio": float(np.median(em) / np.median(e0)) if ok else np.nan}
    return out


def step_response(states_vel, actions_list, joint, pre=4, post=10, settle_len=10):
    """Speed (px/s, norm over all 4 coords) aligned on command onsets and offsets of
    one joint, averaged over clean events: an onset counts if every joint was
    still for `pre` ticks before it, an offset if every joint stays still for
    `post` ticks after it, and the joint's run lasted at least `settle_len`.

    onset_lag: ticks from the command until speed first crosses halfway from the
    still floor to the steady speed. coast: ticks from the stop command until it
    falls back below halfway. Halfway, not 10%: differentiated tracker noise puts
    a floor under "still" that a small fraction of steady speed would sit below.
    """
    on_prof, off_prof = [], []
    for v, a in zip(states_vel, actions_list):
        speed = np.linalg.norm(v, axis=1)
        a = np.asarray(a)[:, :3]
        any_moving = np.any(a != 0, axis=1)
        alone = (a[:, joint] != 0) & np.all(np.delete(a, joint, axis=1) == 0, axis=1)
        for start, stop in runs(alone):
            if stop - start < settle_len:
                continue
            if start >= pre and not any_moving[start - pre:start].any():
                on_prof.append(speed[start - pre:start + settle_len])
            if stop + post <= len(speed) and not any_moving[stop:stop + post].any():
                off_prof.append(speed[stop - pre:stop + post])
    on = np.nanmean(on_prof, axis=0) if on_prof else None
    off = np.nanmean(off_prof, axis=0) if off_prof else None
    lag = coast = None
    if on is not None:
        floor, steady = np.nanmedian(on[:pre]), np.nanmedian(on[pre + settle_len // 2:])
        above = np.flatnonzero(on[pre:] >= (floor + steady) / 2)
        lag = int(above[0]) if len(above) else None
    if off is not None:
        steady, floor = np.nanmedian(off[:pre]), np.nanmedian(off[-(post // 2):])
        below = np.flatnonzero(off[pre:] <= (floor + steady) / 2)
        coast = int(below[0]) if len(below) else None
    return {"onset_lag": lag, "coast": coast, "n_onsets": len(on_prof), "n_offsets": len(off_prof),
            "onset_profile": on, "offset_profile": off}


def gain_bounds(delay_s):
    """For an integrator plant with pure delay tau under P control (rate K, 1/s):
    stable for K < pi/(2 tau); ~60 degrees phase margin at K = (pi/6)/tau."""
    return {"delay_s": delay_s, "k_max": np.pi / (2 * delay_s), "k_60deg": (np.pi / 6) / delay_s}
