# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the tolerance expansion and the NMS dilation semantics.

The numbers the paper reports depend on these edges, so each test names the behaviour it
locks down rather than just asserting a value.
"""

from __future__ import annotations

from transvlm_eval.tolerance import (
    DEFAULT_EPS,
    T_TOLERANCES,
    convert_dict_to_sb_time_only_with_t_tolerance,
    parse_sb_list_time_only_with_t_tolerance,
    to_tuples,
)

DURATION = 100.0


def _convert(start, end, tt, *, duration=DURATION, eps=DEFAULT_EPS):
    sb = convert_dict_to_sb_time_only_with_t_tolerance({'start_time': start, 'end_time': end}, tt, duration, eps)
    return (sb.start_time, sb.end_time)


def test_eleven_tolerances():
    assert T_TOLERANCES == (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def test_point_expansion_only_fires_at_zero_tolerance():
    """A zero-width boundary gets an eps body ONLY when t_tolerance == 0.0 exactly."""
    assert _convert(50.0, 50.0, 0.0) == (50.0 - DEFAULT_EPS, 50.0 + DEFAULT_EPS)
    # At any other tolerance it becomes [t-tt, t+tt] with no eps applied.
    assert _convert(50.0, 50.0, 0.05) == (49.95, 50.05)


def test_point_expansion_near_zero_shifts_rather_than_truncates():
    assert _convert(0.0, 0.0, 0.0) == (0.0, 2.0 * DEFAULT_EPS)


def test_point_expansion_near_duration_shifts_rather_than_truncates():
    assert _convert(DURATION, DURATION, 0.0) == (DURATION - 2.0 * DEFAULT_EPS, DURATION)


def test_reversed_interval_is_swapped_not_rejected():
    assert _convert(5.0, 3.0, 0.0) == (3.0, 5.0)


def test_clamped_to_video_duration():
    assert _convert(0.0, DURATION, 0.5) == (0.0, DURATION)
    assert _convert(0.01, 0.02, 0.5) == (0.0, 0.52)


def test_none_endpoints_are_dropped_not_raised():
    entries = [
        {'start_time': None, 'end_time': 1.0},
        {'start_time': 1.0, 'end_time': None},
        {'start_time': 2.0, 'end_time': 2.5},
    ]
    merged = parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS)
    assert to_tuples(merged) == [(2.0, 2.5)]


def test_dilation_merges_points_within_four_hundredths():
    """The 0.02 predicate dilation makes two points 0.03 s apart merge, at tt=0.

    to_segment_level widens each sub-0.04 s segment to exactly 0.04 s about its midpoint for
    the overlap test only, so these two touch and collapse into one.
    """
    entries = [{'start_time': 10.0, 'end_time': 10.0}, {'start_time': 10.03, 'end_time': 10.03}]
    merged = parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS)
    assert len(merged) == 1


def test_distant_points_do_not_merge():
    entries = [{'start_time': 10.0, 'end_time': 10.0}, {'start_time': 20.0, 'end_time': 20.0}]
    merged = parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS)
    assert len(merged) == 2


def test_merge_spans_min_start_to_max_end():
    entries = [{'start_time': 1.0, 'end_time': 3.0}, {'start_time': 2.0, 'end_time': 5.0}]
    merged = parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS)
    assert to_tuples(merged) == [(1.0, 5.0)]


def test_merge_is_transitive_via_the_growing_segment():
    """A-B and B-C overlap while A-C do not; all three still collapse into one.

    The predicate is evaluated against the growing merged segment, not the original seed,
    which is what makes the forward-only scan a closure on a start-sorted list.
    """
    entries = [
        {'start_time': 1.0, 'end_time': 2.0},
        {'start_time': 1.9, 'end_time': 3.0},
        {'start_time': 2.9, 'end_time': 4.0},
    ]
    merged = parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS)
    assert to_tuples(merged) == [(1.0, 4.0)]


def test_output_is_sorted_and_disjoint():
    entries = [
        {'start_time': 30.0, 'end_time': 31.0},
        {'start_time': 10.0, 'end_time': 11.0},
        {'start_time': 20.0, 'end_time': 21.0},
    ]
    merged = to_tuples(parse_sb_list_time_only_with_t_tolerance(entries, 0.0, DURATION, DEFAULT_EPS))
    assert merged == sorted(merged)
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        assert prev_end < next_start


def test_empty_input():
    assert parse_sb_list_time_only_with_t_tolerance([], 0.0, DURATION, DEFAULT_EPS) == []
