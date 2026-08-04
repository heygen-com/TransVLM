# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""End-to-end scoring: raw per-clip model text in, paper-format aggregate rows out.

Shared by every scoring entry point so that all reported numbers are produced by one
implementation. Any difference between two reported figures then has to come from the
predictions, never from the scoring.
"""

from __future__ import annotations

import logging
from typing import Any

from transvlm_eval import aggregate as agg
from transvlm_eval import metrics as mt
from transvlm_eval import tolerance as tol

logger = logging.getLogger(__name__)


Segments = list[dict[str, float]]


def score(
    ground_truth: dict[Any, Segments],
    predictions: dict[Any, Segments],
    durations: dict[Any, float],
) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    """Score whole-video predictions against ground truth. Dataset-agnostic entry point.

    Args:
        ground_truth: video id -> list of ``{'start_time': float, 'end_time': float}`` in seconds.
        predictions: same shape, same ids. A video with no predicted transitions maps to ``[]``;
            it must still be present, or its ground truth would go uncounted and recall would come
            out too high.
        durations: video id -> the clamp bound for tolerance expansion, in seconds. **Read the
            note below before choosing this value.**

    Returns:
        ``(per_tolerance, ttmean)``. ``per_tolerance`` maps each of the 11 tolerances to its
        aggregate row; ``ttmean`` is the mean of those rows.

    Raises:
        ValueError: if the three dicts do not cover exactly the same set of video ids. Silently
            scoring an intersection is how a partial run gets mistaken for a complete one.

    **On ``durations``.** Tolerance expansion is clamped to ``[0, duration]``, so this value moves
    every number. It is not automatically the container duration. For the published TransVLM
    figures it is the end of the last sliding window on the 25 fps timeline, cast to float32 --
    which matched the reference for 198 of 198 videos where the container duration matched only 2.
    If you are scoring your own predictions on your own data, the container duration is a
    reasonable choice; just be aware that it is a choice, and use the same one on both sides of
    any comparison.
    """
    ids = set(ground_truth)
    if set(predictions) != ids or set(durations) != ids:
        missing_pred = sorted(ids - set(predictions))[:5]
        missing_dur = sorted(ids - set(durations))[:5]
        extra_pred = sorted(set(predictions) - ids)[:5]
        raise ValueError(
            f'ground_truth has {len(ids)} ids, predictions {len(predictions)}, durations '
            f'{len(durations)} -- all three must cover the same set. '
            f'No prediction for: {missing_pred}. No duration for: {missing_dur}. '
            f'Predicted but no ground truth: {extra_pred}.'
        )

    video_ids = sorted(ids, key=str)
    per_tolerance: dict[float, dict[str, Any]] = {}
    for t_tolerance in tol.T_TOLERANCES:
        records = [
            mt.score_video(ground_truth[vid], predictions[vid], durations[vid], t_tolerance) for vid in video_ids
        ]
        per_tolerance[t_tolerance] = agg.collect_metrics_v3(records)
    return per_tolerance, agg.mean_over_tolerances(per_tolerance)


def headline(per_tolerance: dict[float, dict[str, Any]], ttmean: dict[str, Any]) -> dict[str, Any]:
    """The few figures worth printing in a summary, named as the paper names them."""
    return {
        'tt0.00_cl_f1': per_tolerance[0.00]['cl_f1'],
        'tt0.00_cl_tp': per_tolerance[0.00]['cl_tp'],
        'tt0.00_cl_fp': per_tolerance[0.00]['cl_fp'],
        'tt0.00_cl_fn': per_tolerance[0.00]['cl_fn'],
        'ttmean_cl_f1': ttmean['cl_f1'],
        'ttmean_tl_f1': ttmean['tl_f1'],
        'ttmean_abe_mabe': ttmean['abe_mabe'],
    }
