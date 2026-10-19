"""Is a PD loop on one gripper keypoint seen by both cameras feasible? Offline evidence.

    python scripts/analyze_servoing.py \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval \
        --run cotracker3_offline --keypoint body

Uses one tracker run's tracks (run_tracker_bench.py) of one keypoint in both
views, s = (u_front, v_front, u_side, v_side), and the recorded commands. No
calibration, no depth, no controller is built -- see lone_track/servo_model.py
for the model and why bang-bang single-joint teleop measures it directly.

Writes outputs/track/<dataset>/servo/<run>_<keypoint>/{summary.json,summary.md,*.png}:
  - full-speed pixel velocity per joint and view, and how many samples measured it
  - singular values of J(s) over every visited state: is any direction unobservable?
  - open-loop prediction (leave-one-episode-out) vs "nothing moves", per delay:
    the best delay is the effective command-to-motion latency
  - step response per joint: onset lag and coast after a stop
  - drift while commanded still (creep, bounded above by tracker jitter)
  - the P-gain bounds that delay implies
"""

import argparse
import json
import os

import numpy as np

from lone_track import servo_model as S
from lone_track.episodes import EpisodeIndex, duplicate_frames, still_mask
from lone_track.labels import KEYPOINTS
from lone_track.trackers.base import TrackResult

FPS = 25


def load_states(run_dir, index, episodes, keypoint):
    """Per episode: s (T,4) and the actions. NaN before both views are initialised;
    a view's duplicate (stale) frames are treated as missing and interpolated."""
    k = KEYPOINTS.index(keypoint) if keypoint != "grasp" else None
    states, acts, used = [], [], []
    for ep in episodes:
        paths = [os.path.join(run_dir, f"ep{ep:03d}_{v}.npz") for v in ("front", "side")]
        if not all(os.path.exists(p) for p in paths):
            continue
        cols, start = [], 0
        for p, view in zip(paths, ("front", "side")):
            res = TrackResult.load(p)
            t_q = int(np.load(p)["meta_t_q"])
            start = max(start, t_q)
            xy = res.xy[:, k] if k is not None else (res.xy[:, 1] + res.xy[:, 2]) / 2
            xy = xy.astype(float).copy()
            xy[duplicate_frames(index.frames(ep, view))] = np.nan
            cols.append(xy)
        s = S.fill_gaps(np.concatenate(cols, axis=1))
        s[:start] = np.nan
        states.append(s)
        acts.append(index.actions(ep))
        used.append(ep)
    return states, acts, used


def fit(states, acts, settle=4):
    vel = [S.velocity(s, FPS) for s in states]
    samples = [S.single_joint_samples(a, settle=settle) for a in acts]
    return S.JacobianField().fit(states, vel, samples), vel


def loeo_prediction(states, acts, delays, horizons):
    """Leave-one-episode-out open-loop prediction for each candidate delay."""
    out = {}
    for d in delays:
        per_h = {h: {"model": [], "static": []} for h in horizons}
        for i in range(len(states)):
            train = [j for j in range(len(states)) if j != i] or [i]
            field, _ = fit([states[j] for j in train], [acts[j] for j in train])
            sc = S.prediction_scores(field, [states[i]], [acts[i]], delay=d, horizons=horizons)
            for h in horizons:
                if sc[h]["n"]:
                    per_h[h]["model"].append(sc[h]["model_median"])
                    per_h[h]["static"].append(sc[h]["static_median"])
        out[d] = {h: {"model_median": float(np.median(v["model"])) if v["model"] else np.nan,
                      "static_median": float(np.median(v["static"])) if v["static"] else np.nan}
                  for h, v in per_h.items()}
        for h in horizons:
            r = out[d][h]
            r["ratio"] = r["model_median"] / r["static_median"] if r["static_median"] else np.nan
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--run", required=True, help="run name under outputs/track/<dataset>/runs/")
    p.add_argument("--keypoint", default="body", choices=list(KEYPOINTS) + ["grasp"])
    p.add_argument("--out", default=None, help="tracking output root (default outputs/track/<dataset>)")
    p.add_argument("--episodes", type=int, nargs="*", default=None)
    args = p.parse_args()

    name = args.repo_id.split("/")[-1]
    base = args.out or os.path.join("outputs", "track", name)
    run_dir = os.path.join(base, "runs", args.run)
    index = EpisodeIndex(args.repo_id, args.root)
    episodes = args.episodes if args.episodes is not None else [e.index for e in index.episodes]
    states, acts, used = load_states(run_dir, index, episodes, args.keypoint)
    if not states:
        raise SystemExit(f"no episode has both views tracked in {run_dir}")
    out_dir = os.path.join(base, "servo", f"{args.run}_{args.keypoint}")
    os.makedirs(out_dir, exist_ok=True)

    field, vel = fit(states, acts)
    allS = np.concatenate([s for s in states])
    allS = allS[np.all(np.isfinite(allS), axis=1)]
    sv = S.conditioning(field.J(allS))
    ratio = sv[:, 2] / sv[:, 0]

    columns = {}
    for j, jn in enumerate(S.JOINTS):
        c = field.c[j]
        columns[jn] = {
            "n_samples": int(len(c)),
            "front_px_s_median": float(np.median(np.linalg.norm(c[:, :2], axis=1))) if len(c) else None,
            "side_px_s_median": float(np.median(np.linalg.norm(c[:, 2:], axis=1))) if len(c) else None,
        }

    delays = list(range(0, 7))
    horizons = (5, 10, 25)
    pred = loeo_prediction(states, acts, delays, horizons)
    best_delay = min(delays, key=lambda d: pred[d][10]["model_median"])

    steps = {jn: S.step_response(vel, acts, j) for j, jn in enumerate(S.JOINTS)}
    still_speed = []
    for s, v, a in zip(states, vel, acts):
        still = still_mask(a, settle=6)
        sp = np.linalg.norm(v[still], axis=1) / np.sqrt(2)
        still_speed.append(sp[np.isfinite(sp)])
    still_speed = np.concatenate(still_speed)

    lag_ticks = [steps[j]["onset_lag"] for j in steps if steps[j]["onset_lag"] is not None]
    delay_s = (best_delay if best_delay > 0 else (np.median(lag_ticks) if lag_ticks else 1)) / FPS
    summary = {
        "run": args.run, "keypoint": args.keypoint, "episodes": used,
        "states": int(len(allS)),
        "jacobian_columns": columns,
        "singular_values_px_s": {
            "sigma_min_median": float(np.nanmedian(sv[:, 2])),
            "sigma_min_p05": float(np.nanpercentile(sv[:, 2], 5)),
            "sigma_max_median": float(np.nanmedian(sv[:, 0])),
            "inverse_condition_median": float(np.nanmedian(ratio)),
            "frac_states_inv_cond_below_0.1": float(np.nanmean(ratio < 0.1)),
        },
        "prediction_loeo": {str(d): {str(h): v for h, v in pred[d].items()} for d in delays},
        "best_delay_ticks": best_delay,
        "step_response": {j: {k: v for k, v in r.items() if not k.endswith("profile")}
                          for j, r in steps.items()},
        "still_speed_px_s": {"median": float(np.median(still_speed)),
                             "p90": float(np.percentile(still_speed, 90))},
        # Tracker latency and the command link (~50 ms) add to the recorded delay
        # at deployment; this is the motion delay alone, the optimistic bound.
        "gain_bounds_motion_delay_only": S.gain_bounds(max(delay_s, 1 / FPS)),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    lines = [f"# Two-view servoing feasibility: {args.run}, keypoint `{args.keypoint}`", "",
             f"{len(used)} episodes, {len(allS)} tracked states.", "",
             "## Full-speed pixel velocity per joint (one column of J)", "",
             "| joint | samples | front px/s | side px/s |", "|---|---|---|---|"]
    for jn, c in columns.items():
        lines.append(f"| {jn} | {c['n_samples']} | {c['front_px_s_median']:.0f} | "
                     f"{c['side_px_s_median']:.0f} |" if c["n_samples"] else f"| {jn} | 0 | - | - |")
    svs = summary["singular_values_px_s"]
    lines += ["", "## Observability", "",
              f"sigma_min of J: median {svs['sigma_min_median']:.1f} px/s, 5th pct "
              f"{svs['sigma_min_p05']:.1f}; sigma_max median {svs['sigma_max_median']:.1f}. "
              f"{100 * svs['frac_states_inv_cond_below_0.1']:.1f}% of visited states have "
              "sigma_min/sigma_max < 0.1 (one joint nearly invisible there).", "",
              "## Open-loop prediction, leave-one-episode-out (median px)", "",
              "| delay (ticks) | " + " | ".join(f"H={h} model / static" for h in horizons) + " |",
              "|---|" + "---|" * len(horizons)]
    for d in delays:
        lines.append(f"| {d}{' *' if d == best_delay else ''} | " + " | ".join(
            f"{pred[d][h]['model_median']:.1f} / {pred[d][h]['static_median']:.1f}" for h in horizons) + " |")
    lines += ["", "## Step response (ticks at 25 Hz)", "",
              "| joint | onsets | onset lag | offsets | coast |", "|---|---|---|---|---|"]
    for jn, r in steps.items():
        lines.append(f"| {jn} | {r['n_onsets']} | {r['onset_lag']} | {r['n_offsets']} | {r['coast']} |")
    g = summary["gain_bounds_motion_delay_only"]
    lines += ["", f"Speed while commanded still: median {summary['still_speed_px_s']['median']:.1f} "
              f"px/s (tracker jitter included).", "",
              f"P-gain bounds from the motion delay alone ({1000 * g['delay_s']:.0f} ms): "
              f"stable below {g['k_max']:.1f}/s, ~60 deg margin at {g['k_60deg']:.1f}/s. "
              "Deployment adds tracker and link latency."]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    plot(out_dir, steps, sv, allS, pred, horizons, index, used[0])
    print(f"wrote {out_dir}/summary.md")


def plot(out_dir, steps, sv, allS, pred, horizons, index, ep_for_bg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for jn, r in steps.items():
        for ax, key in zip(axs, ("onset_profile", "offset_profile")):
            if r[key] is not None:
                ax.plot(np.arange(len(r[key])) - 4, r[key], marker="o", ms=3, label=jn)
    for ax, title in zip(axs, ("command on at 0", "command off at 0")):
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("ticks (40 ms)")
        ax.set_ylabel("keypoint speed, px/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_response.png"), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, view, (iu, iv) in zip(axs, ("front", "side"), ((0, 1), (2, 3))):
        ax.imshow(index.video(ep_for_bg, view).frame(0), alpha=0.45)
        sc = ax.scatter(allS[:, iu], allS[:, iv], c=sv[:, 2], s=3, cmap="viridis")
        ax.set_title(f"{view}: sigma_min of J (px/s) at each visited state")
        ax.axis("off")
        fig.colorbar(sc, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "observability.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
