#!/usr/bin/env python
# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Score shot-transition predictions with the metric the TransVLM paper reports.

Reads two JSON files keyed by video id and writes one CSV per time tolerance plus a ``ttmean``
row, matching the layout the published tables were built from.

Input format, both files::

    {"<video id>": {"duration": <seconds>, "segments": [{"start_time": s, "end_time": s}, ...]}}

``duration`` is required in the ground-truth file and ignored in the predictions file. A video with
nothing to report still needs an entry with ``"segments": []`` -- see ``--help`` on that, because
omitting it is the one mistake that makes your scores look better than they are.

Usage:
    python score.py --ground-truth gt.json --predictions preds.json --output-dir scores/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ground-truth', type=Path, required=True, help='JSON: id -> {duration, segments}.')
    p.add_argument(
        '--predictions',
        type=Path,
        required=True,
        help='JSON: id -> {segments}. Must cover exactly the same ids as the ground truth, '
        'including videos with no predicted transitions (use an empty list). Leaving a video out '
        'drops its ground truth from the totals and inflates recall.',
    )
    p.add_argument('--output-dir', type=Path, required=True, help='Where the CSV tree is written.')
    p.add_argument('--method-name', default='transvlm', help='Row label in the emitted CSVs.')
    p.add_argument(
        '--data-source',
        default='custom',
        help='Filename stem for each CSV, so several datasets can share one output tree.',
    )
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


def _load(path: Path, *, need_duration: bool) -> tuple[dict[str, list[dict[str, float]]], dict[str, float]]:
    """Read one side into ``(segments_by_id, durations_by_id)``.

    Validates eagerly. A malformed segment that reaches the scorer produces a number rather than an
    error, and a number is indistinguishable from a correct one after the fact.
    """
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f'{path}: not valid JSON ({exc})') from exc
    if not isinstance(raw, dict):
        raise SystemExit(f'{path}: expected a JSON object keyed by video id, got {type(raw).__name__}')

    segments: dict[str, list[dict[str, float]]] = {}
    durations: dict[str, float] = {}
    for vid, entry in raw.items():
        if not isinstance(entry, dict):
            raise SystemExit(f'{path}: entry for {vid!r} must be an object, got {type(entry).__name__}')
        items = entry.get('segments')
        if items is None:
            raise SystemExit(f'{path}: entry for {vid!r} has no "segments" key (use [] if there are none)')
        parsed = []
        for i, seg in enumerate(items):
            try:
                start, end = float(seg['start_time']), float(seg['end_time'])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f'{path}: {vid!r} segment {i} needs numeric start_time/end_time ({exc})') from exc
            if end < start:
                raise SystemExit(f'{path}: {vid!r} segment {i} ends ({end}) before it starts ({start})')
            parsed.append({'start_time': start, 'end_time': end})
        segments[vid] = parsed
        if need_duration:
            if 'duration' not in entry:
                raise SystemExit(f'{path}: entry for {vid!r} has no "duration"; see score.py --help')
            durations[vid] = float(entry['duration'])
    return segments, durations


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format='%(levelname)s %(name)s: %(message)s')

    from transvlm_eval import aggregate as agg
    from transvlm_eval import score
    from transvlm_eval.pipeline import headline

    ground_truth, durations = _load(args.ground_truth, need_duration=True)
    predictions, _ = _load(args.predictions, need_duration=False)
    logger.info('%d video(s) in the ground truth, %d in the predictions', len(ground_truth), len(predictions))

    per_tolerance, ttmean = score(ground_truth, predictions, durations)

    out_dir = args.output_dir / args.method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    agg.write_score_tree(str(out_dir), args.method_name, per_tolerance, ttmean, data_source=args.data_source)

    head = headline(per_tolerance, ttmean)
    logger.info('cl_f1  tt0.00 = %s', head['tt0.00_cl_f1'])
    logger.info('cl_f1  ttmean = %s', head['ttmean_cl_f1'])
    logger.info('wrote %s', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
