# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the two timestamp patches. Training used .2f; both layers must deliver it.

There are two because vLLM does not route timestamp expansion through the HF processor at
all — it reimplements it with the format hardcoded. Patching only one layer leaves a backend
silently on .1f, which is the bug these tests exist to prevent recurring.
"""

from __future__ import annotations

import importlib.util

import pytest

from transvlm.models import processor_patch as pp

vllm_missing = importlib.util.find_spec('vllm') is None


def test_two_decimal_float_ignores_the_requested_precision():
    """The trick that reaches vLLM's hardcoded f'<{t:.1f} seconds>'."""
    t = pp._TwoDecimalFloat(0.02)
    assert f'<{t:.1f} seconds>' == '<0.02 seconds>'
    assert f'{t:.0f}' == '0.02'
    assert f'{t:.3f}' == '0.02'


def test_two_decimal_float_is_still_a_normal_float_elsewhere():
    """Only fixed-point formatting is hijacked; arithmetic and other specs are untouched."""
    t = pp._TwoDecimalFloat(1.5)
    assert t + 1 == 2.5
    assert float(t) == 1.5
    assert f'{t:g}' == '1.5'
    assert f'{t}' == '1.5'


class _FakeTokenizer:
    def __init__(self, text: str) -> None:
        self._text = text

    def decode(self, _ids: object) -> str:
        return self._text


class _FakeProcessor:
    def __init__(self, text: str) -> None:
        self.tokenizer = _FakeTokenizer(text)


def _clear_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate a test from whatever format an earlier test left the module in."""
    monkeypatch.setattr(pp, '_ACTIVE_FORMAT', None)


def test_expected_decimals_covers_both_formats():
    assert pp.expected_decimals('1f') == 1
    assert pp.expected_decimals('2f') == 2
    with pytest.raises(ValueError):
        pp.expected_decimals('3f')


def test_apply_timestamp_format_rejects_an_unknown_format(monkeypatch):
    _clear_active(monkeypatch)
    with pytest.raises(ValueError):
        pp.apply_timestamp_format('0.2f', include_vllm=False)


def test_apply_timestamp_format_refuses_to_switch_mid_process(monkeypatch):
    """The patch mutates a class, so a second format would split one run across two formats."""
    monkeypatch.setattr(pp, '_ACTIVE_FORMAT', '2f')
    with pytest.raises(RuntimeError, match='cannot change'):
        pp.apply_timestamp_format('1f', include_vllm=False)


def test_apply_timestamp_format_is_idempotent_for_the_same_format(monkeypatch):
    monkeypatch.setattr(pp, '_ACTIVE_FORMAT', '1f')
    monkeypatch.setattr(pp, '_assert_two_decimal_patch_absent', lambda: None)
    assert pp.apply_timestamp_format('1f', include_vllm=False) == '1f'


def test_requesting_1f_fails_when_the_2f_patch_is_installed(monkeypatch):
    """The whole point of the .1f branch: absence of a call is not proof of absence of a patch."""
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    _clear_active(monkeypatch)
    monkeypatch.setattr(Qwen3VLProcessor, pp._MARKER, True, raising=False)
    with pytest.raises(SystemExit, match='patch is installed on'):
        pp.apply_timestamp_format('1f', include_vllm=False)


def test_requesting_1f_succeeds_on_a_clean_processor(monkeypatch):
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    _clear_active(monkeypatch)
    monkeypatch.setattr(Qwen3VLProcessor, pp._MARKER, False, raising=False)
    assert pp.apply_timestamp_format('1f', include_vllm=False) == '1f'
    assert pp.active_timestamp_format() == '1f'


def test_assert_prompt_timestamp_format_accepts_a_matching_prompt():
    processor = _FakeProcessor('<0.02 seconds><0.10 seconds>')
    assert pp.assert_prompt_timestamp_format(processor, {'input_ids': [[1]]}, '2f') == 2


def test_assert_prompt_timestamp_format_catches_the_silent_failure():
    """A .1f prompt when .2f was requested is exactly the bug that cost a full 872-clip run."""
    processor = _FakeProcessor('<0.0 seconds><0.1 seconds>')
    with pytest.raises(SystemExit, match='expected 2'):
        pp.assert_prompt_timestamp_format(processor, {'input_ids': [[1]]}, '2f')


def test_assert_prompt_timestamp_format_rejects_a_prompt_with_no_stamps():
    """No stamps means the video never expanded — a different bug, but still fatal."""
    processor = _FakeProcessor('just text')
    with pytest.raises(SystemExit, match='no <x.x seconds> stamps'):
        pp.assert_prompt_timestamp_format(processor, {'input_ids': [[1]]}, '1f')


def test_timestamp_decimals_reads_a_prompt():
    assert pp.timestamp_decimals('<0.02 seconds><|vision_start|>') == 2
    assert pp.timestamp_decimals('<0.0 seconds><0.1 seconds>') == 1
    assert pp.timestamp_decimals('no stamps here') is None


def test_timestamp_decimals_reports_the_widest_stamp():
    """A mixed prompt should not read as .1f just because the first stamp is short."""
    assert pp.timestamp_decimals('<0.0 seconds><0.18 seconds>') == 2


def test_hf_level_patch_is_idempotent_and_marks_the_class():
    """NOTE: this installs the real .2f patch for the rest of the pytest session.

    Class mutation is not reversible, so any later test that builds a real Qwen3VLProcessor will
    see .2f behaviour regardless of what it asked for. No test does today; if one is added, it
    needs its own process (`pytest-forked`) rather than relying on ordering.
    """
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    assert pp.install_timestamp_patch() is True
    assert getattr(Qwen3VLProcessor, pp._MARKER, False) is True
    assert pp.install_timestamp_patch() is True  # second call is a no-op


@pytest.mark.skipif(vllm_missing, reason='vllm not installed in this venv')
def test_vllm_patch_targets_a_named_method_not_a_copied_body():
    """The point of the float trick: no ~120-line method body is duplicated.

    If a future vLLM stops routing through _get_video_second_idx, install_vllm_timestamp_patch
    raises at its self-check rather than letting prompts revert to .1f.
    """
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo

    assert hasattr(Qwen3VLProcessingInfo, '_get_video_second_idx')
    assert pp.install_vllm_timestamp_patch() is True
    assert getattr(Qwen3VLProcessingInfo, pp._MARKER, False) is True
    assert pp.install_vllm_timestamp_patch() is True


@pytest.mark.skipif(vllm_missing, reason='vllm not installed in this venv')
def test_vllm_still_hardcodes_one_decimal_upstream():
    """The reason the patch is needed, asserted so an upgrade that fixes it gets noticed."""
    import inspect

    from vllm.model_executor.models import qwen3_vl

    source = inspect.getsource(qwen3_vl)
    assert 'seconds>' in source
    assert ':.1f} seconds>' in source, (
        'vLLM no longer hardcodes .1f — re-check whether install_vllm_timestamp_patch is still '
        'needed, and whether it now double-applies'
    )


# --- the class-resolution bug (found in review, 2026-07-30) -----------------------
# In a vLLM process the vendored sitecustomize rebinds the submodule attribute to
# Qwen3VLProcessorOurs before this module runs, so importing `Qwen3VLProcessor` by name yields
# the SUBCLASS while AutoProcessor keeps handing out the original. Patching or checking only the
# name therefore touches a class nothing instantiates.


def test_processor_classes_reaches_the_base_when_the_name_resolves_to_a_subclass(monkeypatch):
    """The MRO walk is what makes the patch reach instances AutoProcessor actually builds."""
    import transformers.models.qwen3_vl as pkg
    import transformers.models.qwen3_vl.processing_qwen3_vl as sub
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor as original

    class Qwen3VLProcessorOursStub(original):
        pass

    # Reproduce exactly what sitecustomize does.
    monkeypatch.setattr(sub, 'Qwen3VLProcessor', Qwen3VLProcessorOursStub)

    found = pp._processor_classes()
    assert original in found, 'the base class AutoProcessor uses must be a patch target'
    assert Qwen3VLProcessorOursStub in found, 'the subclass the name resolves to must be too'
    assert getattr(pkg, 'Qwen3VLProcessor') in found


def test_the_1f_check_sees_a_marker_hidden_on_the_base_class(monkeypatch):
    """A marker on the base must not be invisible just because the name points elsewhere."""
    import transformers.models.qwen3_vl.processing_qwen3_vl as sub
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor as original

    class Qwen3VLProcessorOursStub(original):
        pass

    _clear_active(monkeypatch)
    monkeypatch.setattr(sub, 'Qwen3VLProcessor', Qwen3VLProcessorOursStub)
    monkeypatch.setattr(Qwen3VLProcessorOursStub, pp._MARKER, False, raising=False)
    monkeypatch.setattr(original, pp._MARKER, True, raising=False)

    with pytest.raises(SystemExit, match='patch is installed on'):
        pp.apply_timestamp_format('1f', include_vllm=False)


def test_the_1f_check_inspects_vllm_without_importing_it(monkeypatch):
    """`sys.modules` lookup, so the check cannot break activate_6ch_patch's import ordering."""
    import sys
    import types

    _clear_active(monkeypatch)
    for cls in pp._processor_classes():
        monkeypatch.setattr(cls, pp._MARKER, False, raising=False)

    fake = types.ModuleType('vllm.model_executor.models.qwen3_vl')
    fake.Qwen3VLProcessingInfo = type('Qwen3VLProcessingInfo', (), {pp._MARKER: True})
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.models.qwen3_vl', fake)

    with pytest.raises(SystemExit, match="vLLM's Qwen3VLProcessingInfo"):
        pp.apply_timestamp_format('1f', include_vllm=False)
