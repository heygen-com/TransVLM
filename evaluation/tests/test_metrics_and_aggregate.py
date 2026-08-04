# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the metric definitions and the aggregation conventions.

Most of these lock down behaviour that a reviewer would reasonably want to "fix". Each test
says which published number would move if it were changed.
"""

from __future__ import annotations

import pytest

from transvlm_eval.aggregate import (
    AGGREGATE_METRIC_NAMES,
    collect_metrics_v3,
    get_mean_metrics,
    mean_over_tolerances,
)
from transvlm_eval.metrics import (
    calculate_abe,
    calculate_metrics_v3_from_proc_data,
    calculate_ti,
    score_video,
)

DURATION = 100.0


def test_ti_is_clamped_at_zero():
    assert calculate_ti((0.0, 1.0), (2.0, 3.0)) == 0
    assert calculate_ti((0.0, 2.0), (1.0, 3.0)) == 1.0


def test_abe_is_the_mean_of_absolute_endpoint_offsets():
    assert calculate_abe((1.0, 2.0), (1.5, 2.5)) == 0.5
    assert calculate_abe((1.0, 5.0), (1.0, 5.0)) == 0.0


def test_cl_tp_uses_min_of_distinct_sides_not_a_matching_size():
    """One prediction spanning three GT segments scores cl_tp=1, cl_fn=2, not 3 TPs."""
    gt = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    pred = [(0.0, 7.0)]
    pairs = [(0, 0), (0, 1), (0, 2)]
    m = calculate_metrics_v3_from_proc_data(gt, pred, pairs)
    assert (m['cl_tp'], m['cl_fp'], m['cl_fn']) == (1, 0, 2)


def test_cl_tp_symmetric_case_three_preds_in_one_gt():
    gt = [(0.0, 7.0)]
    pred = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    pairs = [(0, 0), (1, 0), (2, 0)]
    m = calculate_metrics_v3_from_proc_data(gt, pred, pairs)
    assert (m['cl_tp'], m['cl_fp'], m['cl_fn']) == (1, 2, 0)


def test_abe_count_counts_pairs_not_matched_segments():
    """Many-to-many duplicates all contribute, so abe_count is a pair count."""
    gt = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    pred = [(0.0, 7.0)]
    pairs = [(0, 0), (0, 1), (0, 2)]
    m = calculate_metrics_v3_from_proc_data(gt, pred, pairs)
    assert m['abe_count'] == 3
    assert m['cl_tp'] == 1


def test_touching_segments_score_a_tp_with_zero_time_overlap():
    """A touching pair adds to cl_tp and abe_sum but contributes 0 to tl_tp."""
    gt = [(2.0, 3.0)]
    pred = [(1.0, 2.0)]
    m = calculate_metrics_v3_from_proc_data(gt, pred, [(0, 0)])
    assert m['cl_tp'] == 1
    assert m['tl_tp'] == 0
    assert m['abe_sum'] > 0


def test_none_metrics_when_nothing_predicted_and_nothing_in_gt():
    m = calculate_metrics_v3_from_proc_data([], [], [])
    assert m['cl_precision'] is None and m['cl_recall'] is None and m['cl_f1'] is None
    assert m['abe_mabe'] is None


def test_negative_count_raises_instead_of_dropping_into_a_debugger():
    """The original calls breakpoint() here; a batch run needs an exception."""
    with pytest.raises(AssertionError):
        # 2 distinct matched preds but only 1 pred in the list -> cl_fp goes negative.
        calculate_metrics_v3_from_proc_data([(1.0, 2.0), (3.0, 4.0)], [(1.0, 4.0)], [(0, 0), (1, 1)])


def test_tiny_negative_count_is_tolerated_by_the_negative_epsilon():
    """eps is -1e-3, so a float count just below zero must NOT raise.

    Pins the sign and magnitude of that epsilon: setting it to 0.0 would make float noise in
    tl_fp/tl_fn abort a whole run, and existing tests only cover the raising side.
    """
    gt = [(0.0, 1.0)]
    pred = [(0.0, 1.0 - 1e-12)]
    m = calculate_metrics_v3_from_proc_data(gt, pred, [(0, 0)])
    # tl_fp lands at a tiny negative value here; -1e-12 > -1e-3, so it is tolerated.
    assert m['tl_fp'] <= 0
    assert m['tl_fp'] > -1.0e-3


def test_pred_is_list1_so_fp_and_fn_are_not_swapped():
    """Pins the argument order into match_lists_v3 via a deliberately asymmetric case.

    Two predictions against one GT must give cl_fp=1, cl_fn=0. Passing gt as list1 instead
    would silently transpose these two columns, and every other metric test hand-feeds
    matched_idx_pairs so none of them would notice.
    """
    gt = [{'start_time': 10.0, 'end_time': 10.5}]
    pred = [{'start_time': 10.0, 'end_time': 10.5}, {'start_time': 50.0, 'end_time': 50.5}]
    m = score_video(gt, pred, DURATION, t_tolerance=0.0)['metrics_results_v3']
    assert (m['cl_tp'], m['cl_fp'], m['cl_fn']) == (1, 1, 0)


def test_touching_segments_match_end_to_end_after_expansion():
    """A pred ending exactly where a GT begins scores a TP, through the real expansion path."""
    gt = [{'start_time': 10.0, 'end_time': 11.0}]
    pred = [{'start_time': 9.0, 'end_time': 10.0}]
    m = score_video(gt, pred, DURATION, t_tolerance=0.0)['metrics_results_v3']
    assert m['cl_tp'] == 1
    assert m['tl_tp'] == 0


def test_score_video_end_to_end_perfect_match():
    gt = [{'start_time': 10.0, 'end_time': 10.5}]
    pred = [{'start_time': 10.0, 'end_time': 10.5}]
    rec = score_video(gt, pred, DURATION, t_tolerance=0.0)
    m = rec['metrics_results_v3']
    assert (m['cl_tp'], m['cl_fp'], m['cl_fn']) == (1, 0, 0)
    assert m['cl_f1'] == 1.0
    assert m['abe_mabe'] == 0.0


def test_aggregation_is_micro_not_macro():
    """Summing counts then computing F1 once differs from averaging per-video F1."""
    records = [
        {'metrics_results_v3': _m(cl_tp=1, cl_fp=0, cl_fn=0), 'video_duration': 10.0, 'pred_time_taken': None},
        {'metrics_results_v3': _m(cl_tp=0, cl_fp=9, cl_fn=9), 'video_duration': 10.0, 'pred_time_taken': None},
    ]
    agg = collect_metrics_v3(records)
    # Micro: P = R = 1/10 -> F1 = 0.1. A mean of per-video F1 would be (1.0 + 0.0)/2 = 0.5.
    assert agg['cl_f1'] == pytest.approx(0.1)


def test_fps_is_none_when_no_video_reported_a_time():
    records = [{'metrics_results_v3': _m(), 'video_duration': 10.0, 'pred_time_taken': None}]
    assert collect_metrics_v3(records)['fps'] is None


def test_fps_counts_duration_of_untimed_videos():
    """A video with no timing still adds its duration, which inflates fps. Preserved."""
    records = [
        {'metrics_results_v3': _m(), 'video_duration': 10.0, 'pred_time_taken': 1.0},
        {'metrics_results_v3': _m(), 'video_duration': 10.0, 'pred_time_taken': None},
    ]
    assert collect_metrics_v3(records)['fps'] == pytest.approx(20.0 * 25.0 / 1.0)


def test_get_mean_metrics_drops_none_from_both_sum_and_count():
    data = {0: {'x': 1.0}, 1: {'x': None}, 2: {'x': 3.0}}
    assert get_mean_metrics(data, 'x') == 2.0
    assert get_mean_metrics({0: {'x': None}}, 'x') is None


def test_ttmean_averages_counts_too_so_the_row_is_self_inconsistent():
    """A ttmean row's cl_f1 is NOT recomputable from its own cl_tp/cl_fp/cl_fn."""
    per_tt = {
        0.0: {n: (1.0 if n in ('cl_tp', 'cl_fp', 'cl_fn') else 0.5) for n in AGGREGATE_METRIC_NAMES},
        0.5: {n: (3.0 if n in ('cl_tp', 'cl_fp', 'cl_fn') else 0.9) for n in AGGREGATE_METRIC_NAMES},
    }
    row = mean_over_tolerances(per_tt)
    assert row['cl_tp'] == 2.0
    assert row['cl_f1'] == pytest.approx(0.7)
    # F1 recomputed from the averaged counts would be 0.5, not 0.7.
    recomputed = 2 * (2.0 / 4.0) * (2.0 / 4.0) / ((2.0 / 4.0) + (2.0 / 4.0))
    assert row['cl_f1'] != pytest.approx(recomputed)


def _m(**overrides):
    base = {
        'cl_tp': 0,
        'cl_fp': 0,
        'cl_fn': 0,
        'tl_tp': 0.0,
        'tl_fp': 0.0,
        'tl_fn': 0.0,
        'abe_sum': 0.0,
        'abe_count': 0,
    }
    base.update(overrides)
    return base
