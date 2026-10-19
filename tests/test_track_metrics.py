"""lone_track.metrics: the numbers the tracker benchmark reports, on known inputs.

No hardware, dataset or checkpoint is touched -- the module is pure numpy.
"""

import numpy as np

from lone_track.metrics import (
    auroc,
    bootstrap_ci,
    drift_curve,
    match_tip_pair,
    point_errors,
    still_jitter,
    summarize,
    time_to_failure,
)


def test_errors_propagate_missing_as_nan():
    e = point_errors([[0, 0], [3, 4], [np.nan, 0]], [[0, 0], [0, 0], [1, 1]])
    assert e[0] == 0 and e[1] == 5 and np.isnan(e[2])


def test_summary_ignores_nan_and_counts_failures():
    s = summarize([1.0, 2.0, 3.0, 50.0, np.nan])
    assert s["n"] == 4
    assert s["median"] == 2.5
    assert s["fail_rate"] == 0.25
    assert s["pck5"] == 0.75 and s["pck20"] == 0.75


def test_tip_pairing_undoes_a_label_swap():
    p1 = np.array([[10.0, 10.0], [10.0, 10.0]])
    p2 = np.array([[30.0, 10.0], [30.0, 10.0]])
    g1 = np.array([[11.0, 10.0], [29.0, 10.0]])   # frame 1: annotator swapped
    g2 = np.array([[29.0, 10.0], [11.0, 10.0]])
    m1, m2, swapped = match_tip_pair(p1, p2, g1, g2)
    assert swapped.tolist() == [False, True]
    assert np.allclose(point_errors(m1, g1), 1.0) and np.allclose(point_errors(m2, g2), 1.0)


def test_tip_pairing_with_one_tip_unlabelled():
    m1, _, swapped = match_tip_pair([[0, 0]], [[20, 0]], [[19, 0]], [[np.nan, np.nan]])
    assert swapped[0] and np.allclose(m1, [[20, 0]])


def test_bootstrap_ci_brackets_the_median():
    rng = np.random.default_rng(1)
    eps = [rng.normal(5, 1, 30) for _ in range(10)]
    lo, hi = bootstrap_ci(eps)
    assert lo < np.median(np.concatenate(eps)) < hi


def test_drift_curve_bins_by_time_since_init():
    rows = drift_curve([1, 1, 9, 9], [10, 20, 260, 280])
    by = {r["from_s"]: r for r in rows}
    assert by[0.0]["median"] == 1 and by[8.0]["median"] == 9


def test_time_to_failure():
    assert time_to_failure([1, 2, 40, 1], [0, 10, 20, 30], start=0) == 20
    assert time_to_failure([1, 2], [0, 10], start=0) is None


def test_still_jitter_measures_noise_and_detects_two_frame_flicker():
    rng = np.random.default_rng(0)
    noise = np.c_[rng.normal(0, 1, 200), rng.normal(0, 1, 200)] + 100
    still = np.ones(200, bool)
    j = still_jitter(noise, still)
    assert 1.2 < j["std"] < 1.6          # sqrt(2) for unit noise per axis
    assert 0.8 < j["flicker"] < 1.2
    alternating = np.c_[100 + np.tile([0.0, 1.0], 100), np.full(200, 50.0)]
    assert still_jitter(alternating, still)["flicker"] < 0.1
    # Moving frames are ignored.
    assert still_jitter(noise, np.zeros(200, bool))["n_windows"] == 0


def test_auroc_extremes_and_chance():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert np.isnan(auroc([1, 2], [True, True]))
