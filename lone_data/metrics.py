"""Scoring for a bang-bang action space.

Teleop emits only the discrete levels in features.ACTION_LEVELS -- dimensions 0-2
are drive/stop/drive and the gripper is a two-position latch -- so every recorded
action lands on one of them exactly. Scoring a policy against that with squared
error measures the wrong thing:

  * Dimension 0 is stopped in 90.6% of frames. A policy that ignores the image and
    emits a constant 0 sits at the predict-the-mean baseline, so "no better than
    mean" and "learned nothing" become indistinguishable from "learned nothing"
    alone -- the metric cannot tell them apart because they are the same point.
  * The levels are +/-900 on dim 0 and 90 apart on dim 3. Getting the direction
    right but the switch a frame late costs 810000 on dim 0 and 8100 on dim 3, so
    the same behavioural error is scored 100x differently by dimension.
  * Squared error is minimized by hedging toward the conditional mean, and on a
    rare high-amplitude class the conditional mean is close to the majority level.
    The objective actively rewards not committing.

Snapping predictions to the levels and scoring the confusion fixes all three, and
it is also what deployment does: dispatch.snap_to_levels quantizes onto this same
set before anything reaches the board, so the snapped prediction is the action the
arm would have been driven with.
Balanced accuracy -- mean recall over the classes present in the ground truth --
is the number to read: a constant predictor scores exactly 1/k on it whatever the
class balance, so chance is a fixed, interpretable floor.

These functions are deliberately free of torch, lerobot and any checkpoint, so
they can be tested on recorded actions alone.
"""

import numpy as np

# Below this many examples of the rarest present class, a per-dimension verdict is
# sampling noise rather than a measurement. Dimension 0 is non-zero in under 10% of
# frames and splits that across two directions, so a 200-frame sample leaves single
# digits per class.
THIN_SUPPORT = 30


def snap_to_level_index(values, levels):
    """Index of the level each value is nearest.

    Returns class indices, not values -- dispatch.snap_to_levels is the one that
    returns a dispatchable action. Both round a continuous prediction to the same
    nearest level, which is the point: deployment snaps before it drives the arm,
    so scoring the snapped prediction scores what the arm would actually have done.

    Ground truth lands on a level exactly. Midpoints go to the lower level, and that
    tie-break is arbitrary on purpose -- a prediction sitting on a midpoint is
    uninformative whichever way it falls, and the prediction-spread table in
    eval_policy.py is where that shows up.
    """
    values = np.asarray(values, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    return np.abs(values[..., None] - levels[None, :]).argmin(-1)


def confusion(true_cls, pred_cls, k):
    """Counts of (ground truth, prediction) pairs. Rows are truth."""
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (np.asarray(true_cls), np.asarray(pred_cls)), 1)
    return cm


def classification_scores(cm):
    """accuracy, balanced accuracy, chance, majority accuracy, per-class recall.

    Classes absent from the ground truth are excluded from the balanced average:
    recall is undefined for them, and scoring them as zero would penalize a
    dimension for which levels the sampled frames happened to contain rather than
    for anything the policy did. `chance` is 1/k over the classes that are present,
    which is both the uniform-random score and the constant-predictor score.
    """
    cm = np.asarray(cm)
    support = cm.sum(1)
    present = support > 0
    correct = np.diag(cm)
    total = int(cm.sum())
    recall = np.full(len(support), np.nan)
    if total:
        recall[present] = correct[present] / support[present]
    k = int(present.sum())
    return {
        "accuracy": float(correct.sum() / total) if total else float("nan"),
        "balanced_accuracy": float(recall[present].mean()) if k else float("nan"),
        "chance": 1.0 / k if k else float("nan"),
        "majority_accuracy": float(support.max() / total) if total else float("nan"),
        "recall": recall.tolist(),
        "support": support.tolist(),
        "classes_present": k,
    }


def verdict(scores):
    """What the balanced accuracy licenses saying, and nothing more."""
    if scores["classes_present"] < 2:
        return "only one class in sample -- uninformative"
    lift = scores["balanced_accuracy"] - scores["chance"]
    if lift <= 0.02:
        text = "at chance"
    elif lift <= 0.10:
        text = "barely above chance"
    elif lift <= 0.25:
        text = "learned something"
    else:
        text = "learned this dimension"
    # A thin sample caveats every verdict rather than only the negative ones: "at
    # chance" on nine examples of a class is not evidence the policy learned
    # nothing, and "learned this dimension" on nine is not evidence it did.
    if min(s for s in scores["support"] if s > 0) < THIN_SUPPORT:
        return text + " -- too few examples to call"
    if text == "at chance":
        return text + " -- learned nothing"
    return text


def score_dimension(true_values, pred_values, levels):
    """Everything above, for one action dimension."""
    k = len(levels)
    cm = confusion(snap_to_level_index(true_values, levels),
                   snap_to_level_index(pred_values, levels), k)
    scores = classification_scores(cm)
    scores["confusion"] = cm.tolist()
    scores["levels"] = [float(v) for v in levels]
    return scores
