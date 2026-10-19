"""Score saved tracker runs against the hand labels; write tables and plots.

    python scripts/score_tracker_bench.py \
        --repo-id lone/l_one_manipulation_multiview_eval \
        --root data/lerobot/lone/l_one_manipulation_multiview_eval

Reads every run under outputs/track/<dataset>/runs/ (run_tracker_bench.py) and
writes outputs/track/<dataset>/report/{summary.json, summary.md, *.png}.
Needs no GPU: everything comes from the saved .npz files.

What is scored, per run and view, on labelled frames *after* the query frame:
  body    the gripper servo, tracked directly
  tips    the two fingertips, paired to the labels the cheaper way
  grasp   midpoint of the two tips -- the point that has to reach the marker
Point-tracker runs are also scored with the "cluster" strategy: keypoints
carried by a similarity fit to 24 points tracked on the gripper body.

Label-based: error statistics with episode-bootstrap CIs, drift vs time since
initialisation, time to first failure, and how well the tracker's confidence
flags its own failures (AUROC). Label-free, on every frame: jitter while the arm
is commanded still, and the epipolar residual between views -- F is fitted from
the labelled correspondences on still frames, no calibration -- which is
checked against the labels as a failure detector.
"""

import argparse
import glob
import json
import os

import numpy as np

from lone_track import metrics as M
from lone_track.episodes import EpisodeIndex, still_mask
from lone_track.geometry import carry_keypoints, epipolar_distance, fit_fundamental
from lone_track.labels import KEYPOINTS, LabelSet
from lone_track.trackers.base import TrackResult

TARGETS = ("body", "tips", "grasp")
ERR_FLAG_PX = 20.0  # an "error" for the AUROC/epipolar failure-detection checks


def predictions(res, n_kp, t_q, strategy):
    """(T,3,2) keypoint predictions and (T,3) confidences for one strategy."""
    if strategy == "direct":
        return res.xy[:, :n_kp], res.conf[:, :n_kp]
    cluster, cvis = res.xy[:, n_kp:], res.vis[:, n_kp:]
    kp, n_in = carry_keypoints(cluster, cvis, res.xy[t_q, :n_kp], t_q)
    frac = n_in / max(1, cluster.shape[1])
    return kp, np.repeat(frac[:, None], n_kp, 1)


def score_view(labels, ep, view, xy, conf, t_q):
    """Per-target arrays on labelled frames after t_q: errors, conf, frames, gt vis."""
    out = {}
    frames, body_gt, body_vis = labels.arrays(ep, view, "body")
    _, t1_gt, t1_vis = labels.arrays(ep, view, "tip_1")
    _, t2_gt, t2_vis = labels.arrays(ep, view, "tip_2")
    sel = frames > t_q
    f = frames[sel]
    body_err = M.point_errors(xy[f, 0], body_gt[sel])
    m1, m2, _ = M.match_tip_pair(xy[f, 1], xy[f, 2], t1_gt[sel], t2_gt[sel])
    tip_err = np.concatenate([M.point_errors(m1, t1_gt[sel]), M.point_errors(m2, t2_gt[sel])])
    grasp_pred = (xy[f, 1] + xy[f, 2]) / 2
    grasp_gt = (t1_gt[sel] + t2_gt[sel]) / 2
    grasp_err = M.point_errors(grasp_pred, grasp_gt)
    tip_vis = np.where((t1_vis[sel] == "visible") & (t2_vis[sel] == "visible"), "visible",
                       np.where(np.isnan(grasp_gt[:, 0]), "", "occluded"))
    out["body"] = dict(err=body_err, conf=conf[f, 0], frames=f, vis=body_vis[sel])
    out["tips"] = dict(err=tip_err, conf=np.concatenate([conf[f, 1], conf[f, 2]]),
                       frames=np.concatenate([f, f]), vis=np.concatenate([t1_vis[sel], t2_vis[sel]]))
    out["grasp"] = dict(err=grasp_err, conf=np.minimum(conf[f, 1], conf[f, 2]), frames=f,
                        vis=tip_vis)
    return out


def fit_label_F(labels, index, episodes):
    """F from labelled (front, side) pairs on still frames (the views are not synced)."""
    a, b = [], []
    for ep in episodes:
        still = still_mask(index.actions(ep))
        for f in labels.labelled_frames(ep):
            if not still[f]:
                continue
            for kp in KEYPOINTS:
                ea = labels.get(ep, f, "front", kp)
                eb = labels.get(ep, f, "side", kp)
                if ea and eb and ea["vis"] == "visible" and eb["vis"] == "visible":
                    a.append(ea["xy"])
                    b.append(eb["xy"])
    if len(a) < 8:
        return None, len(a)
    F, inl = fit_fundamental(np.array(a), np.array(b))
    resid = epipolar_distance(F, np.array(a), np.array(b))
    return {"F": F, "n": len(a), "inlier_frac": float(inl.mean()),
            "label_residual_median": float(np.median(resid))}, len(a)


def score_run(run_dir, labels, index, F):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    is_point = not cfg["tracker"].startswith("sam2")
    strategies = ["direct", "cluster"] if is_point else ["direct"]
    rows = {}
    for strategy in strategies:
        per = {tgt: {"err": [], "conf": [], "since": [], "ep_err": [], "ttf": []} for tgt in TARGETS}
        occluded = {tgt: [] for tgt in TARGETS}
        jitter = {"body": [], "grasp": []}
        epi, epi_flag, epi_err = [], [], []
        latency, lag = [], []
        views_done = 0
        for ep in labels.episodes():
            tracks = {}
            for view in index.views:
                path = os.path.join(run_dir, f"ep{ep:03d}_{view}.npz")
                if not os.path.exists(path):
                    continue
                res = TrackResult.load(path)
                z = np.load(path)
                t_q, n_kp = int(z["meta_t_q"]), int(z["meta_n_kp"])
                xy, conf = predictions(res, n_kp, t_q, strategy)
                tracks[view] = (xy, t_q)
                views_done += 1
                scored = score_view(labels, ep, view, xy, conf, t_q)
                for tgt, s in scored.items():
                    vis_sel = s["vis"] == "visible"
                    per[tgt]["err"].append(s["err"][vis_sel])
                    per[tgt]["conf"].append(s["conf"][vis_sel])
                    per[tgt]["since"].append(s["frames"][vis_sel] - t_q)
                    per[tgt]["ep_err"].append(s["err"][vis_sel])
                    ttf = M.time_to_failure(s["err"][vis_sel], s["frames"][vis_sel], t_q)
                    per[tgt]["ttf"].append(ttf)
                    occluded[tgt].append(s["err"][s["vis"] == "occluded"])
                still = still_mask(index.actions(ep))
                still[:t_q + 1] = False
                jitter["body"].append(M.still_jitter(xy[:, 0], still))
                jitter["grasp"].append(M.still_jitter((xy[:, 1] + xy[:, 2]) / 2, still))
                if res.latency_ms is not None:
                    latency.append(res.latency_ms[np.isfinite(res.latency_ms)][1:])
                    lag.append(res.lag_frames[np.isfinite(res.lag_frames)])
            if F is not None and len(tracks) == 2:
                (xf, tf), (xs, ts) = tracks["front"], tracks["side"]
                start = max(tf, ts) + 1
                d = epipolar_distance(F, xf[start:, 0], xs[start:, 0])
                epi.append(d[np.isfinite(d)])
                # Against the labels: does a large residual flag a body error in either view?
                frames, gf, _ = labels.arrays(ep, "front", "body")
                _, gs, _ = labels.arrays(ep, "side", "body")
                sel = frames >= start
                f = frames[sel]
                e = np.fmax(M.point_errors(xf[f, 0], gf[sel]), M.point_errors(xs[f, 0], gs[sel]))
                epi_flag.append(epipolar_distance(F, xf[f, 0], xs[f, 0]))
                epi_err.append(e)

        row = {"views": views_done}
        for tgt in TARGETS:
            err = np.concatenate(per[tgt]["err"]) if per[tgt]["err"] else np.array([])
            conf = np.concatenate(per[tgt]["conf"]) if per[tgt]["conf"] else np.array([])
            since = np.concatenate(per[tgt]["since"]) if per[tgt]["since"] else np.array([])
            r = M.summarize(err)
            if r["n"]:
                r["median_ci"] = M.bootstrap_ci(per[tgt]["ep_err"])
                r["fail_ci"] = M.bootstrap_ci(per[tgt]["ep_err"],
                                              stat=lambda e: np.mean(e > M.FAIL_PX))
                r["drift"] = M.drift_curve(err, since)
                r["conf_auroc"] = M.auroc(-conf, err > ERR_FLAG_PX)
                ttf = [t for t in per[tgt]["ttf"] if t is not None]
                r["views_with_failure"] = len(ttf)
                r["median_frames_to_failure"] = float(np.median(ttf)) if ttf else None
            occ = np.concatenate(occluded[tgt]) if occluded[tgt] else np.array([])
            r["occluded"] = M.summarize(occ)
            row[tgt] = r
        for tgt, js in jitter.items():
            vals = [j for j in js if j["n_windows"]]
            row[f"jitter_{tgt}"] = {k: float(np.nanmedian([j[k] for j in vals])) if vals else None
                                    for k in ("std", "step", "flicker")}
        if epi:
            d = np.concatenate(epi)
            fe, ee = np.concatenate(epi_flag), np.concatenate(epi_err)
            row["epipolar_body"] = {
                "median": float(np.median(d)), "frac_gt_10px": float(np.mean(d > 10)),
                "flags_errors_auroc": M.auroc(fe, ee > ERR_FLAG_PX)}
        if latency:
            lat = np.concatenate(latency)
            row["latency_ms"] = {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}
            lg = np.concatenate(lag)
            row["lag_frames"] = {"mean": float(lg.mean()), "max": float(lg.max())}
        rows[strategy] = row
    return cfg, rows


def fmt(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_markdown(path, results, F_info, labels):
    lines = ["# Gripper tracker benchmark", "",
             f"Labelled episodes: {labels.episodes()}  ",
             f"Fundamental matrix: " + (
                 f"{F_info['n']} still-frame correspondences, label residual median "
                 f"{F_info['label_residual_median']:.2f} px" if F_info else "not enough labels"),
             "", "Errors in px at 640x360, labelled-visible frames after the query frame. "
             "CI = 95% episode bootstrap. fail = error > 30 px.", ""]
    for tgt in TARGETS:
        lines += [f"## {tgt}", "",
                  "| run | strategy | n | median [CI] | p95 | PCK@5 | PCK@10 | fail [CI] | "
                  "conf AUROC | occluded median |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        order = sorted(((r[s][tgt].get("median", np.inf), run, s) for run, (_, r) in results.items()
                        for s in r), key=lambda x: x[0])
        for _, run, s in order:
            d = results[run][1][s][tgt]
            if not d.get("n"):
                continue
            ci = d["median_ci"]
            fci = d["fail_ci"]
            lines.append(
                f"| {run} | {s} | {d['n']} | {fmt(d['median'])} [{fmt(ci[0])}, {fmt(ci[1])}] | "
                f"{fmt(d['p95'])} | {fmt(d['pck5'], 2)} | {fmt(d['pck10'], 2)} | "
                f"{fmt(100 * d['fail_rate'])}% [{fmt(100 * fci[0])}, {fmt(100 * fci[1])}] | "
                f"{fmt(d['conf_auroc'], 2)} | {fmt(d['occluded'].get('median'))} |")
        lines.append("")
    lines += ["## Label-free and timing", "",
              "| run | strategy | still jitter body (std px) | flicker | epipolar body median | "
              "epi > 10 px | epi flags errors (AUROC) | latency ms median/p95 | lag frames mean/max |",
              "|---|---|---|---|---|---|---|---|---|"]
    for run, (_, r) in sorted(results.items()):
        for s, d in r.items():
            j = d["jitter_body"]
            e = d.get("epipolar_body", {})
            lat = d.get("latency_ms", {})
            lag = d.get("lag_frames", {})
            lines.append(
                f"| {run} | {s} | {fmt(j['std'], 2)} | {fmt(j['flicker'], 2)} | "
                f"{fmt(e.get('median'))} | {fmt(100 * e['frac_gt_10px']) + '%' if e else '-'} | "
                f"{fmt(e.get('flags_errors_auroc'), 2)} | "
                f"{fmt(lat.get('median'))}/{fmt(lat.get('p95'))} | "
                f"{fmt(lag.get('mean'))}/{fmt(lag.get('max'), 0)} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_drift(path, results, target="body"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run, (_, r) in sorted(results.items()):
        for s, d in r.items():
            dr = d[target].get("drift")
            if not dr:
                continue
            x = [(b["from_s"] + b["to_s"]) / 2 for b in dr]
            y = [b["median"] for b in dr]
            ax.plot(x, y, marker="o", ms=3, lw=1.2, ls="-" if s == "direct" else "--",
                    label=f"{run} ({s})")
    ax.set_xlabel("seconds since initialisation")
    ax.set_ylabel(f"median {target} error (px)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--out", default=None, help="tracking output root (default outputs/track/<dataset>)")
    args = p.parse_args()

    name = args.repo_id.split("/")[-1]
    labels = LabelSet.load(args.labels or os.path.join("labels", f"gripper_gt_{name}.json"))
    index = EpisodeIndex(args.repo_id, args.root)
    base = args.out or os.path.join("outputs", "track", name)
    report = os.path.join(base, "report")
    os.makedirs(report, exist_ok=True)

    F_info, n_corr = fit_label_F(labels, index, labels.episodes())
    F = F_info["F"] if F_info else None
    results = {}
    for run_dir in sorted(glob.glob(os.path.join(base, "runs", "*"))):
        if os.path.exists(os.path.join(run_dir, "config.json")):
            results[os.path.basename(run_dir)] = score_run(run_dir, labels, index, F)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    doc = {"labelled_episodes": labels.episodes(), "fundamental": clean(F_info),
           "runs": {k: {"config": c, "scores": clean(r)} for k, (c, r) in results.items()}}
    with open(os.path.join(report, "summary.json"), "w") as f:
        json.dump(doc, f, indent=1)
    write_markdown(os.path.join(report, "summary.md"), results, F_info, labels)
    for tgt in ("body", "grasp"):
        plot_drift(os.path.join(report, f"drift_{tgt}.png"), results, tgt)
    if F_info is None:
        print(f"note: only {n_corr} still-frame correspondences; no epipolar scores")
    print(f"wrote {report}/summary.md ({len(results)} runs)")


if __name__ == "__main__":
    main()
