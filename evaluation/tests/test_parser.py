"""Pin the parser's four tiers, including the fallbacks that fail silently by design.

These tests exist to stop a well-meaning cleanup from changing how many segments a clip
contributes. Several assertions below encode behaviour that looks like a bug; each says so.
"""

from __future__ import annotations

import pytest

from transvlm_eval.parser import (
    extract_dicts_from_string,
    fix_truncated_json_list,
    loose_parse_from_string,
    parse_model_output_benchmark_results,
)


def test_tier1_plain_json_list():
    out = parse_model_output_benchmark_results('[{"start_time": "1.0", "end_time": "2.0"}]')
    assert out == [{'start_time': '1.0', 'end_time': '2.0'}]


def test_tier1_keeps_string_timestamps():
    """The parser does not coerce types; lifting is what casts to float."""
    out = parse_model_output_benchmark_results('[{"start_time": "7.26", "end_time": "7.70"}]')
    assert isinstance(out[0]['start_time'], str)


def test_json_fence_is_stripped():
    raw = 'Here you go:\n```json\n[{"start_time": 1.0, "end_time": 2.0}]\n```\nDone.'
    assert parse_model_output_benchmark_results(raw) == [{'start_time': 1.0, 'end_time': 2.0}]


def test_bare_fence_is_not_stripped_but_tier3_recovers():
    """A ``` fence without the `json` tag is left in place, so tier 1/2 fail."""
    raw = '```\n[{"start_time": 1.0, "end_time": 2.0}]\n```'
    assert parse_model_output_benchmark_results(raw) == [{'start_time': 1.0, 'end_time': 2.0}]


def test_think_tag_split_takes_index_one_not_last():
    """With two </think> markers the tail is discarded — split(...)[1], not [-1]."""
    raw = 'a</think>[{"start_time": 1.0, "end_time": 2.0}]</think>[{"start_time": 9.0, "end_time": 9.5}]'
    out = parse_model_output_benchmark_results(raw)
    assert out == [{'start_time': 1.0, 'end_time': 2.0}], 'the segment after the 2nd </think> must be lost'


def test_tier2_drops_truncated_trailing_entry():
    raw = '[{"start_time": 1.0, "end_time": 2.0}, {"start_time": 3.0, "end_ti'
    assert fix_truncated_json_list(raw) == [{'start_time': 1.0, 'end_time': 2.0}]


def test_tier2_returns_empty_list_as_success():
    """No complete {...} means zero segments, NOT a parse error."""
    assert fix_truncated_json_list('total garbage, no braces') == []
    # And through the full ladder, garbage therefore yields no predictions rather than raising.
    assert parse_model_output_benchmark_results('total garbage, no braces') == []


def test_tier2_brace_counting_ignores_braces_inside_strings():
    raw = '[{"note": "a } brace", "start_time": 1.0, "end_time": 2.0}]'
    assert fix_truncated_json_list(raw) == [{'note': 'a } brace', 'start_time': 1.0, 'end_time': 2.0}]


def test_tier3_ignores_nested_objects():
    """The flat {[^{}]*} pattern cannot match an entry containing a nested object."""
    assert extract_dicts_from_string('[{"a": {"b": 1}}]') == [{'b': 1}]


def test_tier4_requires_all_four_keys():
    partial = '"start_time": 1.0, "end_time": 2.0'
    assert loose_parse_from_string(partial) == []
    full = '"start_time": 1.0, "end_time": 2.0, "transition_type": "cut", "direction": "global"'
    assert loose_parse_from_string(full) == [
        {'start_time': 1.0, 'end_time': 2.0, 'transition_type': 'cut', 'direction': 'global'}
    ]


def test_tier4_skips_entries_with_negative_times():
    """A negative start_time makes the whole entry vanish at tier 4.

    The pattern is anchored on the literal "start_time", and neither `\\s*` nor `[\\d.]+`
    can consume the minus, so the match fails at that anchor and there is nowhere else to
    retry — the entry is dropped rather than being recovered with the sign stripped.
    """
    raw = '"start_time": -1.0, "end_time": 2.0, "transition_type": "cut", "direction": "global"'
    assert loose_parse_from_string(raw) == []


def test_dict_output_trips_the_assertion():
    """A bare object parses at tier 1, then fails the isinstance check outside the ladder."""
    with pytest.raises(AssertionError):
        parse_model_output_benchmark_results('{"start_time": 1.0, "end_time": 2.0}')


def test_empty_list_round_trips():
    assert parse_model_output_benchmark_results('[]') == []
