"""The properties that make balanced accuracy the verdict metric, asserted.

No hardware, dataset or checkpoint is touched -- lone_data.metrics is pure numpy
precisely so these can run anywhere.
"""

import numpy as np
import pytest

from lone_data.features import ACTION_LEVEL_LABELS, ACTION_LEVELS, ACTION_NAMES
from lone_data.metrics import (
    classification_scores,
    confusion,
    score_dimension,
    snap_to_level_index,
    verdict,
)

BASE = ACTION_LEVELS[0]          # (-900, 0, 900)
GRIPPER = ACTION_LEVELS[3]       # (30, 120)

# Dimension 0 as recorded: stopped in ~90% of frames, the rest split between the
# two directions. This is the class balance that makes squared error useless.
# Scaled up 40x from the recorded proportions so every class clears THIN_SUPPORT
# and the verdicts under test are the confident ones; test_thin_sample_is_flagged
# covers the small-sample path separately.
SKEWED = np.array([0.0] * 3600 + [-900.0] * 280 + [900.0] * 120)


def test_levels_and_labels_agree():
    assert len(ACTION_LEVELS) == len(ACTION_LEVEL_LABELS) == len(ACTION_NAMES)
    for levels, labels in zip(ACTION_LEVELS, ACTION_LEVEL_LABELS):
        assert len(levels) == len(labels)
        assert list(levels) == sorted(levels), "levels must be ascending"


def test_snap_picks_the_nearest_level():
    got = snap_to_level_index([-900.0, -600.0, -100.0, 0.0, 40.0, 500.0, 900.0], BASE)
    assert got.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_snap_is_exact_on_recorded_actions():
    """Ground truth is always a level, so it must never be misassigned."""
    for j, levels in enumerate(ACTION_LEVELS):
        truth = np.array(levels)
        assert snap_to_level_index(truth, levels).tolist() == list(range(len(levels)))


def test_constant_predictor_scores_exactly_chance():
    """The whole point. A policy ignoring the image gets 1/k however skewed the dim."""
    for constant in BASE:
        scores = score_dimension(SKEWED, np.full_like(SKEWED, constant), BASE)
        assert scores["classes_present"] == 3
        assert scores["balanced_accuracy"] == pytest.approx(1 / 3)
        assert scores["chance"] == pytest.approx(1 / 3)
        assert verdict(scores) == "at chance -- learned nothing"


def test_constant_predictor_looks_good_on_plain_accuracy():
    """Why plain accuracy cannot be the verdict: the do-nothing policy scores 0.9."""
    scores = score_dimension(SKEWED, np.zeros_like(SKEWED), BASE)
    assert scores["accuracy"] == pytest.approx(0.90)
    assert scores["accuracy"] == pytest.approx(scores["majority_accuracy"])
    assert scores["balanced_accuracy"] == pytest.approx(1 / 3)


def test_constant_predictor_matches_the_mse_baseline():
    """And why MSE cannot either: constant-zero is at the predict-the-mean baseline.

    Squared error scores the do-nothing policy within a hair of the baseline a
    policy is supposed to beat, so the two are not distinguishable by that metric.
    """
    mse_constant = float(((0.0 - SKEWED) ** 2).mean())
    mse_baseline = float(((SKEWED.mean() - SKEWED) ** 2).mean())
    assert mse_constant == pytest.approx(mse_baseline, rel=0.03)


def test_perfect_predictor():
    scores = score_dimension(SKEWED, SKEWED.copy(), BASE)
    assert scores["balanced_accuracy"] == pytest.approx(1.0)
    assert scores["accuracy"] == pytest.approx(1.0)
    assert verdict(scores) == "learned this dimension"


def test_noisy_predictions_still_snap_to_the_right_class():
    """A policy need not hit the level exactly, only land in its basin."""
    rng = np.random.default_rng(0)
    noisy = SKEWED + rng.uniform(-200, 200, SKEWED.shape)
    scores = score_dimension(SKEWED, noisy, BASE)
    assert scores["balanced_accuracy"] == pytest.approx(1.0)


def test_direction_flip_scores_at_chance_and_the_confusion_says_why():
    """A policy that knows when to move but not which way scores 1/3 -- the same as
    a do-nothing policy, since it only ever gets the stop class right. The two are
    separated by the confusion matrix, not by the headline number, which is why the
    matrix is printed for every dimension rather than only on request.
    """
    scores = score_dimension(SKEWED, -SKEWED, BASE)
    assert scores["balanced_accuracy"] == pytest.approx(1 / 3)
    assert scores["recall"] == pytest.approx([0.0, 1.0, 0.0])
    cm = np.array(scores["confusion"])
    assert cm[0, 2] == 280 and cm[2, 0] == 120   # every move predicted backwards
    assert cm[1, 1] == 3600                      # every stop predicted correctly


def test_scoring_below_chance_is_possible():
    """Nothing clamps the metric at chance: wrong on every class scores zero."""
    always_moving = np.where(SKEWED == 0.0, 900.0, -SKEWED)
    scores = score_dimension(SKEWED, always_moving, BASE)
    assert scores["balanced_accuracy"] == pytest.approx(0.0)
    assert scores["balanced_accuracy"] < scores["chance"]


def test_absent_classes_are_excluded_not_scored_zero():
    """A sample containing only two of three levels is scored out of two."""
    truth = np.array([0.0] * 10 + [900.0] * 10)
    scores = score_dimension(truth, truth.copy(), BASE)
    assert scores["classes_present"] == 2
    assert scores["chance"] == pytest.approx(0.5)
    assert scores["balanced_accuracy"] == pytest.approx(1.0)
    assert np.isnan(scores["recall"][0])      # -900 never occurred
    assert scores["support"] == [0, 10, 10]


def test_single_class_sample_is_called_uninformative():
    truth = np.zeros(50)
    scores = score_dimension(truth, truth.copy(), BASE)
    assert scores["classes_present"] == 1
    assert verdict(scores) == "only one class in sample -- uninformative"


def test_thin_sample_is_flagged():
    """16 base-motion events in 200 frames is what the old default produced."""
    thin = np.array([0.0] * 184 + [-900.0] * 11 + [900.0] * 5)
    thin_verdict = verdict(score_dimension(thin, thin.copy(), BASE))
    assert thin_verdict.endswith("too few examples to call")

    thick = np.array([0.0] * 1840 + [-900.0] * 110 + [900.0] * 50)
    assert verdict(score_dimension(thick, thick.copy(), BASE)) == "learned this dimension"


def test_thin_sample_caveats_a_negative_verdict_too():
    """Not enough data is not evidence of failure any more than of success."""
    thin = np.array([0.0] * 184 + [-900.0] * 11 + [900.0] * 5)
    scores = score_dimension(thin, np.zeros_like(thin), BASE)
    assert scores["balanced_accuracy"] == pytest.approx(scores["chance"])
    assert verdict(scores) == "at chance -- too few examples to call"


def test_binary_gripper_dimension():
    truth = np.array([30.0] * 60 + [120.0] * 40)
    scores = score_dimension(truth, np.full_like(truth, 30.0), GRIPPER)
    assert scores["chance"] == pytest.approx(0.5)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
    assert scores["accuracy"] == pytest.approx(0.6)


def test_confusion_rows_are_ground_truth():
    truth = np.array([0.0, 0.0, 900.0])
    pred = np.array([0.0, 900.0, 900.0])
    cm = confusion(snap_to_level_index(truth, BASE), snap_to_level_index(pred, BASE), 3)
    assert cm[1, 1] == 1 and cm[1, 2] == 1 and cm[2, 2] == 1
    assert cm.sum() == 3


def test_empty_sample_does_not_raise():
    scores = classification_scores(np.zeros((3, 3), dtype=int))
    assert np.isnan(scores["accuracy"])
    assert scores["classes_present"] == 0


def test_eval_and_deployment_snap_identically():
    """The invariant the shared ACTION_LEVELS exists to protect.

    dispatch.snap_to_levels is what runs before a command reaches the board. If the
    eval snapped differently it would be scoring an action the arm would never have
    been driven with, and nothing else in the repo would notice.
    """
    from lone_data.dispatch import DEMONSTRATED_LEVELS
    from lone_data.dispatch import snap_to_levels as dispatch_snap

    assert DEMONSTRATED_LEVELS == ACTION_LEVELS

    rng = np.random.default_rng(1)
    actions = np.stack([rng.uniform(lo, hi, 500) for lo, hi in
                        [(-900, 900), (-100, 100), (-100, 100), (30, 120)]], axis=1)
    dispatched = np.stack([dispatch_snap(a) for a in actions])
    for j, levels in enumerate(ACTION_LEVELS):
        scored = np.asarray(levels)[snap_to_level_index(actions[:, j], levels)]
        assert scored == pytest.approx(dispatched[:, j])
