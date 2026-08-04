# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Parse the model's raw JSON output into per-window ``SegmentPrediction`` lists.

Ports the 4-level fallback from the training pipeline's output parser, so a reply this
repository accepts is one the published numbers also accepted. Malformed output is common
enough that the tiers matter: a parser that only tries ``json.loads`` silently scores a
recoverable answer as an empty prediction.

Times reported by the model are window-relative; callers
supply ``window_offset`` (== ``WindowSpec.start_time``) to translate them to
absolute video times.

The four tiers, in order of decreasing strictness:

    1. ``json.loads`` on the raw string.
    2. ``fix_truncated_json_list`` — clip trailing garbage, close arrays.
    3. ``extract_dicts_from_string`` — regex-extract balanced ``{...}`` blocks.
    4. ``loose_parse_from_string`` — freeform ``start_time``/``end_time`` keys.

Each function returns ``list[dict[str, Any]]``. ``parse_model_output`` wraps
them and converts to ``SegmentPrediction``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from transvlm.data.types import SegmentPrediction

logger = logging.getLogger(__name__)


def fix_truncated_json_list(s: str) -> list[dict[str, Any]]:
    """Recover the longest prefix that parses as a JSON list of dicts.

    Handles the common failure mode where ``max_new_tokens`` truncates the
    response mid-object; walks the string once counting braces (skipping
    inside strings + escapes) to find the last complete dict boundary.
    """
    in_string = False
    escape = False
    brace_stack: list[str] = []
    last_valid_pos = -1

    for i, ch in enumerate(s):
        if ch == '"' and not escape:
            in_string = not in_string
        if in_string:
            escape = ch == '\\' and not escape
            continue
        escape = False
        if ch == '{':
            brace_stack.append('{')
        elif ch == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    last_valid_pos = i

    if last_valid_pos == -1:
        return []

    fixed = s[: last_valid_pos + 1].rstrip()
    if not fixed.startswith('['):
        fixed = '[' + fixed
    if not fixed.endswith(']'):
        fixed = fixed + ']'
    return json.loads(fixed)


def extract_dicts_from_string(s: str) -> list[dict[str, Any]]:
    """Regex-extract balanced ``{...}`` blocks and json-parse the concatenation."""
    pattern = re.compile(r'\{[^{}]*\}')
    dicts = pattern.findall(s)
    if not dicts:
        return []
    return json.loads('[' + ','.join(dicts) + ']')


def loose_parse_from_string(s: str) -> list[dict[str, Any]]:
    """Regex-match ``start_time``/``end_time`` (+optional label / direction) blocks."""
    # First, try the strict pattern with transition_type + direction.
    strict = re.findall(
        r'"start_time"\s*:\s*([\d.]+).*?'
        r'"end_time"\s*:\s*([\d.]+).*?'
        r'"transition_type"\s*:\s*"([^"]+)".*?'
        r'"direction"\s*:\s*"([^"]+)"',
        s,
        re.S,
    )
    if strict:
        return [
            {'start_time': float(a), 'end_time': float(b), 'transition_type': c, 'direction': d}
            for a, b, c, d in strict
        ]

    # Fallback to the training-time schema (only start_time / end_time).
    loose = re.findall(
        r'"start_time"\s*:\s*([\d.]+)[^}]*?"end_time"\s*:\s*([\d.]+)',
        s,
        re.S,
    )
    return [{'start_time': float(a), 'end_time': float(b)} for a, b in loose]


def _normalize_model_text(text: str) -> str:
    """Strip markdown fences, ``</think>`` prefix, and collapse whitespace.

    Ports training pipeline ``parse_model_output_from_llm`` normalization (see
    the training pipeline's output parser).
    Without this, models that emit ``` ```json [...] ``` ``` fences or
    ``<think>...</think>[...]`` prefixes silently return ``[]`` from tier 1
    even when the actual JSON is valid.
    """
    # Whitespace collapse — newlines to spaces, multi-space to single.
    text = ' '.join(text.replace('\n', ' ').split())
    # Strip ```json ... ``` fence.
    if '```json' in text:
        text = text.split('```json')[1]
        if '```' in text:
            text = text.split('```')[0]
    # Strip </think> prefix.
    if '</think>' in text:
        text = text.split('</think>')[1]
    return text.strip()


def parse_json_dicts(text: str) -> list[dict[str, Any]]:
    """Run the 4-tier fallback; return whatever tier succeeds first.

    Returns an empty list only if all four tiers turn up nothing.
    """
    # Normalize BEFORE the 4-tier parse chain.
    text = _normalize_model_text(text)

    # Tier 1 — plain json.loads.
    # The training pipeline `parse_model_output_from_llm` asserts
    # the result is a list AFTER successful parse; if it's a dict/str/number
    # it returns `[]` rather than falling through. Match that: bare object
    # from the model should NOT become `[{...}]` via tier-2 (which would
    # inject a phantom segment).
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    else:
        return parsed if isinstance(parsed, list) else []

    try:
        parsed = fix_truncated_json_list(text)
        if parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    try:
        parsed = extract_dicts_from_string(text)
        if parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    parsed = loose_parse_from_string(text)
    if parsed:
        return parsed
    logger.warning('All 4 parse tiers returned nothing for text: %r', text[:200])
    return []


def parse_model_output(
    text: str,
    *,
    window_offset: float,
    window_index: int | None = None,
) -> list[SegmentPrediction]:
    """Parse one window's model output and translate to absolute video time.

    Args:
        text: raw JSON-ish string emitted by the model.
        window_offset: seconds to add to each returned ``start_time`` / ``end_time``
            (usually the window's own ``start_time``).
        window_index: optional window id to stamp on ``source_window``.

    Returns:
        A list of ``SegmentPrediction`` with times in absolute video seconds.
        Malformed entries (missing keys, non-numeric times, ``end < start``) are
        dropped with a warning.
    """
    raw = parse_json_dicts(text)
    out: list[SegmentPrediction] = []
    for entry in raw:
        try:
            start = float(entry['start_time']) + window_offset
            end = float(entry['end_time']) + window_offset
        except (KeyError, TypeError, ValueError):
            logger.debug('Dropping malformed entry: %r', entry)
            continue
        if end < start:
            start, end = end, start
        out.append(
            SegmentPrediction(
                start_time=start,
                end_time=end,
                confidence=None,
                source_window=window_index,
            )
        )
    return out
