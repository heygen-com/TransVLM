# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Cover the public :func:`transvlm_eval.score` entry point.

The scoring internals are already covered elsewhere; what is new here is the dataset-agnostic
surface, and specifically its refusal to score a mismatched set of ids. Quietly scoring the
intersection of three dicts is how a partial run gets reported as a complete one, and the
resulting number is higher than the truth, which is the direction nobody double-checks.
"""

from __future__ import annotations

import pytest

from transvlm_eval import score


def _seg(start: float, end: float) -> dict[str, float]:
    return {'start_time': start, 'end_time': end}


def test_a_perfect_prediction_scores_one():
    gt = {'a': [_seg(1.0, 1.04)], 'b': [_seg(2.0, 2.08)]}
    per_tolerance, ttmean = score(gt, dict(gt), {'a': 10.0, 'b': 10.0})
    assert per_tolerance[0.00]['cl_f1'] == 1.0
    assert ttmean['cl_f1'] == 1.0


def test_no_predictions_still_counts_the_ground_truth_as_missed():
    """An empty list must be present, not absent -- absence is the error path below.

    Note ``cl_f1`` is ``None``, not ``0.0``: with zero predictions the precision denominator is
    zero, and this scorer keeps the reference implementation's None-on-zero-denominator
    convention rather than substituting a zero. Anything aggregating these rows has to handle
    ``None``.
    """
    gt = {'a': [_seg(1.0, 1.04)]}
    per_tolerance, _ = score(gt, {'a': []}, {'a': 10.0})
    assert per_tolerance[0.00]['cl_fn'] == 1
    assert per_tolerance[0.00]['cl_tp'] == 0
    assert per_tolerance[0.00]['cl_f1'] is None


def test_a_missing_prediction_raises_rather_than_scoring_the_intersection():
    gt = {'a': [_seg(1.0, 1.04)], 'b': [_seg(2.0, 2.04)]}
    with pytest.raises(ValueError, match='same set'):
        score(gt, {'a': [_seg(1.0, 1.04)]}, {'a': 10.0, 'b': 10.0})


def test_a_missing_duration_raises():
    gt = {'a': [_seg(1.0, 1.04)]}
    with pytest.raises(ValueError, match='same set'):
        score(gt, dict(gt), {})


def test_predicting_for_an_unknown_video_raises():
    """Extra predictions are as wrong as missing ones: they inflate nothing but hide a wiring bug."""
    gt = {'a': [_seg(1.0, 1.04)]}
    with pytest.raises(ValueError, match='same set'):
        score(gt, {'a': [_seg(1.0, 1.04)], 'ghost': []}, {'a': 10.0})


def test_duration_clamps_the_tolerance_window():
    """The clamp bound is load-bearing, which is why the caller must supply it explicitly.

    Asserted on the expansion itself rather than on an f1: whether a clamp flips a *match* depends
    on contrived spacing, but that it clips the window is unconditional, and clipping is what
    propagates into every time-level figure.
    """
    from transvlm_eval import tolerance as tol

    seg = [_seg(9.7, 9.74)]
    generous = tol.to_tuples(tol.parse_sb_list_time_only_with_t_tolerance(seg, 0.50, 100.0, eps=tol.DEFAULT_EPS))
    clipped = tol.to_tuples(tol.parse_sb_list_time_only_with_t_tolerance(seg, 0.50, 9.95, eps=tol.DEFAULT_EPS))
    assert generous[0][1] > clipped[0][1], 'a shorter duration must clip the expanded window'
    assert clipped[0][1] == 9.95, 'the window is clipped exactly at the duration'
    assert generous[0][0] == clipped[0][0], 'only the upper bound is affected here'
