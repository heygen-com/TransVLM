# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""The paper's model-output parser: a four-tier ladder with deliberate silent fallbacks.

Ported from the reference implementation the published numbers were computed with: the
preprocessing step, the ladder, and all three recovery tiers.

Because the model emits free text, this parser decides how many predicted segments each
clip contributes, so its quirks move the published numbers directly. The ones that matter:

* **Whitespace is collapsed before anything else**, including inside string literals.
* Only a ``` ```json ``` fence is stripped; a bare ``` ``` ``` fence is not, so such output
  survives only via tier 3 or 4.
* Both the fence and ``</think>`` are split with index ``[1]``, not ``[-1]``: with two
  markers the tail is discarded.
* **Tier 2 returning ``[]`` counts as success**, so output containing no complete ``{...}``
  yields zero segments rather than an error. A truncated trailing entry is dropped silently.
* Tiers 2 and 3 are all-or-nothing; only tier 4 is per-entry, and it requires all four keys,
  so an entry with just start/end is skipped there.
* A JSON object (rather than a list) parses at tier 1 and then trips the final
  ``isinstance`` assertion, which sits **outside** the try ladder and so propagates.

Do not add tiers, reorder them, or make the failure modes louder: every one of these paths
was exercised while producing the reference results.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def fix_truncated_json_list(s: str) -> list[Any]:
    """Tier 2: cut at the last complete top-level ``{...}`` and re-close the list.

    Ported from the reference implementation. Brace counting is string- and
    escape-aware. Returns ``[]`` — a *success* — when no complete object exists.
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
    fixed = s[: last_valid_pos + 1]
    fixed = fixed.rstrip()
    if not fixed.startswith('['):
        fixed = '[' + fixed
    if not fixed.endswith(']'):
        fixed = fixed + ']'
    return json.loads(fixed)


def extract_dicts_from_string(s: str) -> list[Any]:
    """Tier 3: harvest every flat ``{...}`` and parse them as one list.

    Ported from the reference implementation. The pattern does not nest, so an entry
    containing a nested object is missed.
    """
    pattern = re.compile(r'\{[^{}]*\}')
    dicts = pattern.findall(s)
    return json.loads('[' + ','.join(dicts) + ']')


def loose_parse_from_string(s: str) -> list[dict[str, Any]]:
    """Tier 4: regex out the four expected keys, per entry.

    Ported from the reference implementation. Requires all four keys in order, and
    a negative timestamp makes the whole entry fail to match, so such entries are dropped.
    """
    blocks = re.findall(
        r'"start_time"\s*:\s*([\d.]+).*?'
        r'"end_time"\s*:\s*([\d.]+).*?'
        r'"transition_type"\s*:\s*"([^"]+)".*?'
        r'"direction"\s*:\s*"([^"]+)"',
        s,
        re.S,
    )
    return [
        {'start_time': float(a), 'end_time': float(b), 'transition_type': c, 'direction': d} for a, b, c, d in blocks
    ]


def parse_model_output_benchmark_results(result: str) -> list[Any]:
    """Turn one clip's raw model text into a list of interval dicts.

    Ported from the reference implementation.

    Raises:
        Exception: when all four tiers fail.
        AssertionError: when the output parsed to something other than a list.
    """
    result = ' '.join(result.replace('\n', ' ').split())
    if '```json' in result:
        result = result.split('```json')[1]
        if '```' in result:
            result = result.split('```')[0]
    if '</think>' in result:
        result = result.split('</think>')[1]
    result = result.strip()
    parsed_result: Any = None
    parsed_flag = False
    parsed_errors: list[str] = []

    if not parsed_flag:
        try:
            parsed_result = json.loads(result)
            parsed_flag = True
        except Exception as e:  # noqa: BLE001 - tier demotion is the control flow
            parsed_flag = False
            parsed_errors.append(str(e))

    if not parsed_flag:
        try:
            parsed_result = fix_truncated_json_list(result)
            parsed_flag = True
        except Exception as e:  # noqa: BLE001
            parsed_flag = False
            parsed_errors.append(str(e))

    if not parsed_flag:
        try:
            parsed_result = extract_dicts_from_string(result)
            parsed_flag = True
        except Exception as e:  # noqa: BLE001
            parsed_flag = False
            parsed_errors.append(str(e))

    if not parsed_flag:
        try:
            parsed_result = loose_parse_from_string(result)
            parsed_flag = True
        except Exception as e:  # noqa: BLE001
            parsed_flag = False
            parsed_errors.append(str(e))

    if not parsed_flag:
        raise Exception('Failed to parse result:\n' + '\n'.join(parsed_errors))

    assert isinstance(parsed_result, list), f'Result is not a list, got {type(parsed_result)}'
    return parsed_result
