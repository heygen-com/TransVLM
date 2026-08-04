# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Lift per-clip predictions into video time, the way the paper assembles them.

Ported from the reference implementation
(``get_row_result_from_benchmark_results``).

The windows are 10 s wide on a 9 s stride, so consecutive clips overlap by 1 s and can both
report the same boundary. Nothing de-duplicates here: the duplicates are removed later by
the NMS inside :func:`..tolerance.parse_sb_list_time_only_with_t_tolerance`. Do not add a
merge step, or boundaries near window seams will be merged twice.

Faithful details worth naming:

* ``float()`` is applied during lifting. The model emits string timestamps, and the
  downstream expansion compares raw values, so this cast is what keeps that comparison
  numeric rather than lexicographic.
* There is an ``"end"`` fallback for entries missing ``"end_time"``.
* A per-entry failure is logged and skipped, not raised. Clips with unparseable output
  therefore contribute nothing rather than failing the video.
* ``elapsed`` is always ``None`` in the original, which is why the paper's ``fps`` column is
  empty for this path. Preserved.
* The original passes the lifted list through ``convert_to_pred_result_ours_llm`` before
  scoring, which only adds ``label`` / ``direction`` / ``attributes`` keys. That hop is skipped
  here because it cannot change any number:
  ``convert_dict_to_sb_time_only_with_t_tolerance`` nulls ``label``/``direction`` regardless and
  ``ShotBoundary.from_dict`` defaults the rest. Named so a future line-by-line diff against the
  original does not read it as an omission.

**The one deliberate deviation from the original.** The original skips any clip whose
``.mp4`` is missing from disk, and identifies entries by ``sha256(clip_video_path)``. Those
paths need not exist where scoring runs, so replicating the check could silently drop every
clip. Instead every manifest clip must resolve to a prediction entry, and a miss raises. That
is strictly stronger than the original — it cannot silently lose a clip — and it is the only
intentional divergence in this package.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from transvlm_eval.parser import parse_model_output_benchmark_results

logger = logging.getLogger(__name__)


def lift_clip_predictions(
    clips: list[tuple[int, float, float]],
    get_raw_output: Callable[[int], str],
) -> dict[str, Any]:
    """Assemble one video's predictions from its clips' raw model outputs.

    Args:
        clips: ``(chunk_id, clip_st, clip_et)`` for every clip of this video.
        get_raw_output: maps a ``chunk_id`` to that clip's raw model text. Must not return
            ``None``; a missing clip is a hard error here, unlike in the original.

    Returns:
        ``{'pred': [{'start_time', 'end_time'}, ...], 'elapsed': None}`` with timestamps in
        absolute video time, in clip order (unsorted overall, which the later sort handles).
    """
    video_result_pred: list[dict[str, float]] = []
    video_result_elapsed = None

    for chunk_id, clip_st, _clip_et in clips:
        clip_response = get_raw_output(chunk_id)
        if clip_response is None:
            # Deviation from the original, which would `continue` here. See module docstring.
            raise KeyError(f'no prediction for chunk_id={chunk_id}')
        clip_result_parsed = parse_model_output_benchmark_results(clip_response)

        # Adjust timestamps and aggregate
        for r in clip_result_parsed:
            try:
                start_time = float(r['start_time']) + clip_st
                if 'end_time' not in r.keys():
                    end_time = float(r['end']) + clip_st
                else:
                    end_time = float(r['end_time']) + clip_st
                video_result_pred.append({'start_time': start_time, 'end_time': end_time})
            except Exception as e:  # noqa: BLE001 - per-entry skip is the original behaviour
                logger.warning('Error adjusting timestamps: %s (entry %r)', e, r)
                continue

    return {'pred': video_result_pred, 'elapsed': video_result_elapsed}
