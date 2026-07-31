"""Cross-video micro aggregation, the ttmean average, and the paper's CSV layout.

Ported from the reference implementation the published numbers were computed with
(``get_mean_metrics`` and ``collect_metrics_v3``), plus the export shape of its scoring script.

Two things here are routinely got wrong when reimplemented:

* **Aggregation is micro, not macro.** Raw counts are summed across videos and precision /
  recall / F1 are computed exactly once at the end. The per-video ``cl_f1`` values are
  computed and stored but never averaged. Likewise ``abe_mabe`` is ``sum(abe_sum) /
  sum(abe_count)`` over *pairs*, not the mean of per-video mABE.
* **ttmean is the mean of already-aggregated values, not F1 recomputed from pooled counts.**
  It averages every column independently across the 11 tolerances, count columns included —
  hence the non-integer counts in a ttmean row, and hence a ttmean row's ``cl_f1`` is
  deliberately inconsistent with the ``cl_tp``/``cl_fp``/``cl_fn`` printed beside it.
  ``None`` cells are dropped from both numerator and denominator, **per metric**, so two
  columns in the same row can have different effective denominators.

Videos with empty GT are not excluded anywhere: such a video contributes its predictions as
false positives. That is intentional in the original and preserved here.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# The aggregated row carries one column the per-video record does not: fps.
AGGREGATE_METRIC_NAMES = (
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
    'fps',
)


def get_mean_metrics(metrics_dict: dict[Any, dict[str, Any]], k: str) -> float | None:
    """Mean of metric ``k`` across the outer dict's entries, skipping ``None``.

    Ported from the reference implementation. The skip applies to both the sum and the
    count, so an all-``None`` metric yields ``None`` rather than 0.
    """
    all_values = []
    for kk in metrics_dict.keys():
        vv = metrics_dict[kk][k]
        if vv is not None:
            all_values.append(vv)
    if len(all_values) == 0:
        return None
    return sum(all_values) / len(all_values)


def _prf(tp: float, fp: float, fn: float) -> tuple[float | None, float | None, float | None]:
    precision = None if (tp + fp) == 0.0 else tp / (tp + fp)
    recall = None if (tp + fn) == 0.0 else tp / (tp + fn)
    if precision is not None and recall is not None:
        f1 = None if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return precision, recall, f1


def collect_metrics_v3(per_video_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-aggregate per-video records into one row.

    Ported from the reference implementation. The original takes
    ``(dict_of_lists, method_name)``; this takes the already-selected list, which is the
    same computation with one less indirection.

    ``fps`` uses a hardcoded 25.0 and is ``None`` when no video reported a time. On the
    benchmark-results path ``pred_time_taken`` is always ``None``, so ``fps`` is ``None``
    there — which is why the paper's CSV has an empty fps cell for these methods.
    """
    total_cl_tp, total_cl_fp, total_cl_fn = 0.0, 0.0, 0.0
    total_tl_tp, total_tl_fp, total_tl_fn = 0.0, 0.0, 0.0
    total_abe_sum, total_abe_count = 0.0, 0.0
    total_pred_time_taken, total_video_duration = 0.0, 0.0

    for record in per_video_records:
        metrics = record['metrics_results_v3']
        total_cl_tp += metrics['cl_tp']
        total_cl_fp += metrics['cl_fp']
        total_cl_fn += metrics['cl_fn']
        total_tl_tp += metrics['tl_tp']
        total_tl_fp += metrics['tl_fp']
        total_tl_fn += metrics['tl_fn']
        total_abe_sum += metrics['abe_sum']
        total_abe_count += metrics['abe_count']

        # A video with no timing still contributes its duration, which inflates fps. That
        # asymmetry is in the original.
        if record['pred_time_taken'] is not None:
            total_pred_time_taken += record['pred_time_taken']
        total_video_duration += record['video_duration']

    total_cl_precision, total_cl_recall, total_cl_f1 = _prf(total_cl_tp, total_cl_fp, total_cl_fn)
    total_tl_precision, total_tl_recall, total_tl_f1 = _prf(total_tl_tp, total_tl_fp, total_tl_fn)
    total_abe_mabe = None if total_abe_count == 0 else total_abe_sum / total_abe_count
    total_fps = None if total_pred_time_taken == 0 else total_video_duration * 25.0 / total_pred_time_taken

    return {
        'cl_tp': total_cl_tp,
        'cl_fp': total_cl_fp,
        'cl_fn': total_cl_fn,
        'cl_precision': total_cl_precision,
        'cl_recall': total_cl_recall,
        'cl_f1': total_cl_f1,
        'tl_tp': total_tl_tp,
        'tl_fp': total_tl_fp,
        'tl_fn': total_tl_fn,
        'tl_precision': total_tl_precision,
        'tl_recall': total_tl_recall,
        'tl_f1': total_tl_f1,
        'abe_sum': total_abe_sum,
        'abe_count': total_abe_count,
        'abe_mabe': total_abe_mabe,
        'fps': total_fps,
    }


def mean_over_tolerances(per_tolerance: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """Average each aggregated column across tolerances — the paper's ``ttmean`` row.

    Ported from the reference implementation, which calls
    ``get_mean_metrics`` once per metric name.
    """
    return {name: get_mean_metrics(per_tolerance, name) for name in AGGREGATE_METRIC_NAMES}


def write_metrics_csv(path: str, rows: dict[str, dict[str, Any]]) -> None:
    """Write one aggregated row per method, in the paper's column order.

    ``None`` becomes an empty cell, matching ``csv.writer``'s behaviour in the original.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method_name'] + list(AGGREGATE_METRIC_NAMES))
        for method_name in sorted(rows):
            writer.writerow([method_name] + [rows[method_name].get(m) for m in AGGREGATE_METRIC_NAMES])
    logger.info('Saved: %s', path)


def read_metrics_csv(path: str) -> dict[str, dict[str, float | None]]:
    """Read a metrics CSV back into ``{method_name: {metric: value}}``.

    Used to compare against the committed reference values. Empty cells become ``None``; the
    files were written by ``csv.writer`` and may carry CRLF endings, which the csv module
    handles.
    """
    out: dict[str, dict[str, float | None]] = {}
    with open(path, 'r', newline='', encoding='utf-8-sig') as handle:
        for record in csv.DictReader(handle):
            method_name = record.pop('method_name')
            out[method_name] = {k: (None if v in (None, '') else float(v)) for k, v in record.items()}
    return out


def write_score_tree(
    output_dir: str,
    method_name: str,
    per_tolerance: dict[float, dict[str, Any]],
    ttmean: dict[str, Any],
    *,
    data_source: str,
) -> None:
    """Write the 11-tolerance CSV tree plus ``summary.json``.

    Shared by every scoring entry point so the whole-video scorer
    emits a byte-identical tree instead of carrying a
    second copy. ``runtime/`` is deliberately not importable, so shared logic has to live here.

    ``data_source`` names the CSV file inside each tolerance directory; it is a parameter rather
    than a constant because a caller may be scoring something other than autoshot.
    """
    import json  # noqa: PLC0415 -- only needed on this path
    import os  # noqa: PLC0415

    for t_tolerance, row in per_tolerance.items():
        write_metrics_csv(os.path.join(output_dir, f'tt{t_tolerance:.2f}', f'{data_source}.csv'), {method_name: row})
    write_metrics_csv(os.path.join(output_dir, 'ttmean', f'{data_source}.csv'), {method_name: ttmean})
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'method_name': method_name,
                'per_tolerance': {f'{t:.2f}': row for t, row in per_tolerance.items()},
                'ttmean': ttmean,
            },
            handle,
            indent=2,
        )
