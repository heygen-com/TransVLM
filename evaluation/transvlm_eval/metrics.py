"""Per-video v3 metrics: case-level, time-level and mean absolute boundary error.

Ported from the reference implementation the published numbers were computed with:
its metric math and its matching driver.

Quirks that are load-bearing — reproduce, do not repair:

* ``cl_tp = min(#distinct matched pred, #distinct matched gt)``. This is *not* the size of a
  maximum bipartite matching. One prediction spanning three GT segments scores
  ``cl_tp = 1, cl_fn = 2``; three predictions inside one GT score ``cl_tp = 1, cl_fp = 2``.
* Matching is many-to-many, so ``tl_*`` and ``abe_*`` iterate **every** overlapping pair.
  ``abe_count`` therefore counts pairs, not matched GT segments, and mABE is weighted
  toward segments with many overlaps.
* The overlap predicate inside ``match_lists_v3`` is ``<=``, so segments that merely touch
  count as matched. Such a pair adds to ``cl_tp`` and contributes a full (usually large)
  value to ``abe_sum`` while adding exactly ``0`` to ``tl_tp``.
* ``pred_remains_idx`` / ``gt_remains_idx`` are computed, passed in, and never read.

The original calls ``breakpoint()`` at two sites when a count goes negative. That hangs, or
raises ``BdbQuit``, in a batch run, so it is replaced here by a raised assertion carrying the
offending values. The trigger condition and the negative epsilon are
unchanged, so this only affects *how* the invariant violation surfaces, never whether one is
detected.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from transvlm_eval._vendored.list_match import match_lists_v3

from transvlm_eval.tolerance import (
    DEFAULT_EPS,
    parse_sb_list_time_only_with_t_tolerance,
    to_tuples,
)

logger = logging.getLogger(__name__)

Segment = tuple[float, float]

# Column order is the literal dict order of the ported metric functions, which is what the
# paper's CSV writer emits. Keep it exactly.
PER_VIDEO_METRIC_NAMES = (
    'cl_tp',
    'cl_fp',
    'cl_fn',
    'cl_precision',
    'cl_recall',
    'cl_f1',
    'tl_tp',
    'tl_fp',
    'tl_fn',
    'tl_precision',
    'tl_recall',
    'tl_f1',
    'abe_sum',
    'abe_count',
    'abe_mabe',
)


def calculate_ti(pred_seg: Segment, gt_seg: Segment) -> float:
    """Temporal intersection, clamped at zero.

    Ported from the reference implementation.
    """
    sp, ep = pred_seg
    sg, eg = gt_seg
    intersection = max(0, min(ep, eg) - max(sp, sg))
    return intersection


def calculate_abe(pred_seg: Segment, gt_seg: Segment) -> float:
    """Mean of the absolute start and end offsets for one matched pair.

    Ported from the reference implementation. Operates on the tolerance-expanded,
    NMS-merged segments, so it is affected by clamping at 0 / video_duration and by merging.
    """
    return 0.5 * (math.fabs(pred_seg[0] - gt_seg[0]) + math.fabs(pred_seg[1] - gt_seg[1]))


def _prf(tp: float, fp: float, fn: float) -> tuple[float | None, float | None, float | None]:
    """Precision / recall / F1 with the paper's None-on-zero-denominator convention."""
    precision = None if (tp + fp) == 0.0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0.0 else tp / (tp + fn)
    if precision is not None and recall is not None:
        f1 = None if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return precision, recall, f1


def calculate_metrics_v3_from_proc_data(
    gt_tuple_tolerance: list[Segment],
    pred_tuple_tolerance: list[Segment],
    matched_idx_pairs: list[tuple[int, int]],
    pred_remains_idx: list[int] | None = None,
    gt_remains_idx: list[int] | None = None,
    eps: float = -1.0e-3,
) -> dict[str, Any]:
    """Compute one video's cl_*, tl_* and abe_* figures from matched pairs.

    Ported from the reference implementation. ``pred_remains_idx`` and
    ``gt_remains_idx`` are accepted only to mirror the original signature; the original
    never reads them either.
    """
    del pred_remains_idx, gt_remains_idx  # unread in the original too

    # Case Level F1. Matching is many-to-many, so collapse to distinct indices per side.
    matched_pred_idx_list = []
    matched_gt_idx_list = []
    for cur_matched_idx_pair in matched_idx_pairs:
        matched_pred_idx_list.append(cur_matched_idx_pair[0])
        matched_gt_idx_list.append(cur_matched_idx_pair[1])
    matched_pred_idx_list = list(set(matched_pred_idx_list))
    matched_gt_idx_list = list(set(matched_gt_idx_list))

    cl_tp = min(len(matched_pred_idx_list), len(matched_gt_idx_list))
    cl_fp = len(pred_tuple_tolerance) - cl_tp
    cl_fn = len(gt_tuple_tolerance) - cl_tp
    if cl_fp < eps or cl_fn < eps:
        raise AssertionError(f'negative case-level count: cl_tp={cl_tp} cl_fp={cl_fp} cl_fn={cl_fn}')
    cl_precision, cl_recall, cl_f1 = _prf(cl_tp, cl_fp, cl_fn)

    # Time Level F1. Every matched pair contributes, duplicates included.
    tl_tp = 0.0
    for cur_matched_idx_pair in matched_idx_pairs:
        cur_pred_seg = pred_tuple_tolerance[cur_matched_idx_pair[0]]
        cur_gt_seg = gt_tuple_tolerance[cur_matched_idx_pair[1]]
        tl_tp += calculate_ti(cur_pred_seg, cur_gt_seg)
    pred_duration = sum([i[1] - i[0] for i in pred_tuple_tolerance])
    gt_duration = sum([i[1] - i[0] for i in gt_tuple_tolerance])
    tl_fp = pred_duration - tl_tp
    tl_fn = gt_duration - tl_tp
    if tl_fp < eps or tl_fn < eps:
        raise AssertionError(f'negative time-level count: tl_tp={tl_tp} tl_fp={tl_fp} tl_fn={tl_fn}')
    tl_precision, tl_recall, tl_f1 = _prf(tl_tp, tl_fp, tl_fn)

    # MABE, again over all pairs, so abe_count is a pair count.
    abe_sum = 0.0
    abe_count = 0
    for cur_matched_idx_pair in matched_idx_pairs:
        cur_pred_seg = pred_tuple_tolerance[cur_matched_idx_pair[0]]
        cur_gt_seg = gt_tuple_tolerance[cur_matched_idx_pair[1]]
        abe_sum += calculate_abe(cur_pred_seg, cur_gt_seg)
        abe_count += 1
    abe_mabe = None if abe_count == 0 else abe_sum / abe_count

    return {
        'cl_tp': cl_tp,
        'cl_fp': cl_fp,
        'cl_fn': cl_fn,
        'cl_precision': cl_precision,
        'cl_recall': cl_recall,
        'cl_f1': cl_f1,
        'tl_tp': tl_tp,
        'tl_fp': tl_fp,
        'tl_fn': tl_fn,
        'tl_precision': tl_precision,
        'tl_recall': tl_recall,
        'tl_f1': tl_f1,
        'abe_sum': abe_sum,
        'abe_count': abe_count,
        'abe_mabe': abe_mabe,
    }


def calculate_metrics_v3(
    gt_res: list[dict[str, Any]],
    pred_parsed: list[dict[str, Any]],
    pred_time_taken: float | None,
    video_duration: float,
    t_tolerance: float = 0.1,
    eps: float = DEFAULT_EPS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Expand + merge both sides, then match them, returning the intermediate state.

    Ported from the reference implementation. GT and predictions go through the
    identical expansion, which is what makes the GT segment count a usable check on this
    pipeline.

    Note the argument order into ``match_lists_v3``: predictions are ``list1`` and GT is
    ``list2``, so ``matched_idx_pairs`` is ``(pred_idx, gt_idx)``.
    """
    gt_dict_tolerance = parse_sb_list_time_only_with_t_tolerance(
        gt_res, t_tolerance, video_duration, eps, verbose=verbose
    )
    pred_dict_tolerance = parse_sb_list_time_only_with_t_tolerance(
        pred_parsed, t_tolerance, video_duration, eps, verbose=verbose
    )

    gt_tuple_tolerance = to_tuples(gt_dict_tolerance)
    pred_tuple_tolerance = to_tuples(pred_dict_tolerance)

    matched_idx_pairs, pred_remains_idx, gt_remains_idx = match_lists_v3(
        pred_tuple_tolerance,
        gt_tuple_tolerance,
        get_start=lambda sb: sb[0],
        get_end=lambda sb: sb[1],
        show_progress=False,
        show_info=False,
    )

    return dict(
        gt_tuple_tolerance=gt_tuple_tolerance,
        pred_tuple_tolerance=pred_tuple_tolerance,
        matched_idx_pairs=matched_idx_pairs,
        pred_remains_idx=pred_remains_idx,
        gt_remains_idx=gt_remains_idx,
        video_duration=video_duration,
        pred_time_taken=pred_time_taken,
        t_tolerance=t_tolerance,
        eps=eps,
    )


def score_video(
    gt_res: list[dict[str, Any]],
    pred_parsed: list[dict[str, Any]],
    video_duration: float,
    t_tolerance: float,
    pred_time_taken: float | None = None,
    eps: float = DEFAULT_EPS,
) -> dict[str, Any]:
    """Convenience wrapper: match then score one video at one tolerance.

    Returns a record shaped like the one the paper's aggregator consumes, i.e. carrying
    ``metrics_results_v3`` alongside ``video_duration`` and ``pred_time_taken``.
    """
    proc = calculate_metrics_v3(gt_res, pred_parsed, pred_time_taken, video_duration, t_tolerance=t_tolerance, eps=eps)
    metrics = calculate_metrics_v3_from_proc_data(
        proc['gt_tuple_tolerance'],
        proc['pred_tuple_tolerance'],
        proc['matched_idx_pairs'],
        proc['pred_remains_idx'],
        proc['gt_remains_idx'],
    )
    return {
        'metrics_results_v3': metrics,
        'video_duration': proc['video_duration'],
        'pred_time_taken': proc['pred_time_taken'],
        'n_gt': len(proc['gt_tuple_tolerance']),
        'n_pred': len(proc['pred_tuple_tolerance']),
    }
