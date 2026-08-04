# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the clip -> video lifting behaviour.

None of this was covered before, which meant the ``"end"`` key fallback or the string-to-float
cast could have been removed without a single test going red.
"""

from __future__ import annotations

import json

import pytest

from transvlm_eval.assembly import lift_clip_predictions

CLIPS = [(0, 0.0, 10.0), (1, 9.0, 19.0), (2, 18.0, 28.0)]


def _lift(by_chunk: dict[int, str]):
    return lift_clip_predictions(CLIPS, lambda chunk_id: by_chunk[chunk_id])


def test_timestamps_are_shifted_by_the_clip_start():
    out = _lift(
        {
            0: json.dumps([{'start_time': 1.0, 'end_time': 1.5}]),
            1: json.dumps([{'start_time': 2.0, 'end_time': 2.5}]),
            2: json.dumps([]),
        }
    )
    assert out['pred'] == [
        {'start_time': 1.0, 'end_time': 1.5},
        {'start_time': 11.0, 'end_time': 11.5},
    ]


def test_string_timestamps_are_cast_to_float():
    """The model emits strings; without this cast the tolerance expansion would compare
    lexicographically and then fail on the arithmetic."""
    out = _lift({0: json.dumps([{'start_time': '1.98', 'end_time': '2.02'}]), 1: '[]', 2: '[]'})
    (seg,) = out['pred']
    assert isinstance(seg['start_time'], float) and seg['start_time'] == pytest.approx(1.98)


def test_end_key_is_accepted_as_a_fallback_for_end_time():
    out = _lift({0: json.dumps([{'start_time': 1.0, 'end': 1.5}]), 1: '[]', 2: '[]'})
    assert out['pred'] == [{'start_time': 1.0, 'end_time': 1.5}]


def test_unusable_entry_is_skipped_not_raised():
    """A per-entry failure drops that entry and keeps the rest of the clip."""
    raw = json.dumps(
        [
            {'start_time': 1.0, 'end_time': 1.5},
            {'start_time': 'not-a-number', 'end_time': 2.0},
            {'end_time': 3.0},
            {'start_time': 4.0, 'end_time': 4.5},
        ]
    )
    out = _lift({0: raw, 1: '[]', 2: '[]'})
    assert out['pred'] == [
        {'start_time': 1.0, 'end_time': 1.5},
        {'start_time': 4.0, 'end_time': 4.5},
    ]


def test_overlapping_windows_keep_duplicates_for_the_later_nms():
    """The 1 s window overlap can report the same boundary twice; lifting must NOT dedupe.

    De-duplication is the NMS's job further down. Merging here as well would merge boundaries
    near a window seam twice and move the numbers.
    """
    out = _lift(
        {
            0: json.dumps([{'start_time': 9.5, 'end_time': 9.6}]),
            1: json.dumps([{'start_time': 0.5, 'end_time': 0.6}]),
            2: '[]',
        }
    )
    assert out['pred'] == [
        {'start_time': 9.5, 'end_time': 9.6},
        {'start_time': 9.5, 'end_time': 9.6},
    ]


def test_missing_clip_raises_rather_than_being_skipped():
    """The deliberate deviation from the benchmark, which would `continue` here."""
    with pytest.raises(KeyError):
        lift_clip_predictions(CLIPS, lambda chunk_id: None)


def test_elapsed_is_always_none():
    """Preserved from the original, and the reason the paper's fps column is empty."""
    assert _lift({0: '[]', 1: '[]', 2: '[]'})['elapsed'] is None


def test_unparseable_clip_contributes_nothing_rather_than_failing():
    """Parser tier 2 returns [] for text with no complete object, and that is a success."""
    out = _lift({0: 'I could not find any transitions.', 1: '[]', 2: '[]'})
    assert out['pred'] == []
