"""Tolerance expansion and NMS, as the paper applies them to GT and predictions alike.

Ported from the reference implementation the published numbers were computed with.

Both sides of the comparison go through the *same* function: GT is expanded, clamped and
merged exactly like predictions are (the reference implementation). That symmetry is why the
GT segment count at a given tolerance is itself a meaningful check on this module.

Behaviour that looks wrong but is load-bearing — do not change any of it:

* The zero-width point expansion by ``eps`` fires **only when ``t_tolerance == 0.0``
  exactly**. At every other tolerance a point boundary simply becomes ``[t-tt, t+tt]``.
* The NMS overlap predicate compares ``to_segment_level(0.02)`` copies, so segments
  shorter than ``2*eps`` are widened to exactly ``0.04 s`` about their midpoint *for the
  test only*, while the merger combines the **undilated** times. Two point boundaries up
  to 0.04 s apart therefore merge, and the merged segment can still have zero width.
* ``merge_segments`` absorbs forward over indices only. It behaves as a full transitive
  closure here purely because the list is pre-sorted by start time; remove the sort and it
  silently stops being one.
* Entries whose ``start_time``/``end_time`` is ``None`` are dropped, not raised on.

Callers must pass numeric ``start_time``/``end_time``. The swap branch below compares the
raw dict values, so a string timestamp (as the model emits, e.g. ``"7.26"``) would be
compared lexicographically and then fail on the arithmetic. The benchmark avoids this by
casting during clip lifting; :mod:`.assembly` does the same.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from transvlm_eval._vendored.segment_operations import merge_segments
from transvlm_eval._vendored.shot_boundary_types import ShotBoundary, ShotBoundaryComparer, ShotBoundaryMerger

logger = logging.getLogger(__name__)

# Half a frame at 25 fps. The benchmark hardcodes this same value in two distinct roles:
# as the `eps` used to give zero-width boundaries a body, and as the dilation applied
# inside the NMS overlap predicate.
DEFAULT_EPS = 0.02

# The 11 tolerances the paper sweeps, and over which `ttmean` is averaged.
T_TOLERANCES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def convert_dict_to_sb_time_only_with_t_tolerance(
    i: dict[str, Any], t_tolerance: float, video_duration: float, eps: float = 0.0
) -> ShotBoundary:
    """Expand one interval by ``t_tolerance`` and clamp it to ``[0, video_duration]``.

    Ported from the reference implementation.

    ``label`` and ``direction`` are forcibly nulled, which is why the paper's v3 metrics
    ignore transition type entirely and match on time alone.
    """
    i_new = i.copy()
    i_new['label'] = None
    i_new['direction'] = None
    if i_new['start_time'] > i_new['end_time']:
        logger.warning(
            'start_time (%s) is greater than end_time (%s), swap them', i_new['start_time'], i_new['end_time']
        )
        i_new['start_time'], i_new['end_time'] = (i_new['end_time'], i_new['start_time'])
    elif i_new['start_time'] == i_new['end_time']:
        if t_tolerance == 0.0:  # set offset
            if i_new['start_time'] < eps:
                i_new['start_time'] = 0.0
                i_new['end_time'] = 2.0 * eps
            elif i_new['start_time'] > video_duration - eps:
                i_new['start_time'] = video_duration - 2.0 * eps
                i_new['end_time'] = video_duration
            else:
                i_new['start_time'] = i_new['start_time'] - eps
                i_new['end_time'] = i_new['end_time'] + eps
    i_new['start_time'] = max(0, i_new['start_time'] - t_tolerance)
    i_new['end_time'] = min(video_duration, i_new['end_time'] + t_tolerance)
    i_new['start_time'] = min(video_duration, max(0, i_new['start_time']))  # avoid out of range
    i_new['end_time'] = min(video_duration, max(0, i_new['end_time']))  # avoid out of range
    return ShotBoundary.from_dict(i_new)


def parse_sb_list_time_only_with_t_tolerance(
    sb_dict_list: list[dict[str, Any]],
    t_tolerance: float,
    video_duration: float,
    eps: float = 0.0,
    verbose: bool = False,
) -> list[ShotBoundary]:
    """Expand every interval, sort by start time, then merge overlaps (NMS).

    Ported from the reference implementation.

    The sort is what makes the forward-only ``merge_segments`` behave as a transitive
    closure, and it is stable, so the input order decides which segment seeds each merge.
    """
    if verbose:
        logger.info('length of sb_dict_list: %d', len(sb_dict_list))
    sb_list = []
    for i in sb_dict_list:
        if i['start_time'] is None or i['end_time'] is None:
            logger.warning('start_time or end_time is None for item: %s', i)
            continue
        sb_list.append(convert_dict_to_sb_time_only_with_t_tolerance(i, t_tolerance, video_duration, eps))
    if verbose:
        logger.info('length of sb_list: %d', len(sb_list))
    sb_list_sorted = sorted(sb_list, key=lambda x: x.start_time if x.start_time is not None else float('inf'))
    sb_cmpr = ShotBoundaryComparer()
    sb_cmpr_func = partial(
        sb_cmpr,
        mode='ss',
        eps=0.02,  # FPS=25, 1 / 25 / 2 = 0.02s
        check_label=False,
        check_direction=False,
    )
    sb_mrg = ShotBoundaryMerger()
    sb_mrg_func = sb_mrg
    sb_list_merged = merge_segments(sb_list_sorted, sb_cmpr_func, sb_mrg_func)
    if verbose:
        logger.info('length of sb_list_merged: %d', len(sb_list_merged))
    return sb_list_merged


def to_tuples(sb_list: list[ShotBoundary]) -> list[tuple[float, float]]:
    """Reduce merged boundaries to the ``(start, end)`` pairs the metrics operate on.

    Ported from the reference implementation. Merged boundaries carry list-valued
    ``label``/``direction`` (the merger concatenates them), which nothing downstream reads.
    """
    return [(sb.start_time, sb.end_time) for sb in sb_list]
