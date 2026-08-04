# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the SGLang engine's contract and the parity-critical metadata handling.

Split by import cost: the pure-helper tests run in any venv (the module imports no sglang at
module scope), while anything needing sglang itself skips when it is absent — this file has
to stay green in the vLLM venv too.

The metadata tests are the important ones. An earlier implementation synthesized
``frames_indices=list(range(n_frames))``, which is a silent parity break: the prompt's
``<x.xx seconds>`` stamps are computed from that field, and the real index set is not always
contiguous because ``smart_nframes`` floors the count to a multiple of 2 and then samples
with ``linspace``.
"""

from __future__ import annotations

import importlib.util
import inspect

import pytest

from transvlm.inference import sglang_clip_engine as sce
from transvlm.inference.clip_engine import SAMPLING_KWARGS

# `ckpt_dir` is required rather than defaulted, so every engine construction has to supply one.
# These tests never load weights -- they assert on constructor-assembled kwargs and on guard
# behaviour -- so a path that does not exist is exactly right: if a test ever starts touching the
# filesystem here, it will fail loudly instead of silently reading someone's real checkpoint.
_CKPT = '/nonexistent/transvlm-ckpt'


sglang_missing = importlib.util.find_spec('sglang') is None


class _FakeTensor:
    """Stands in for the (N, 6, H, W) decoded tensor; only ``.shape`` is read."""

    def __init__(self, n: int, h: int, w: int) -> None:
        self.shape = (n, 6, h, w)


def test_real_frames_indices_are_forwarded_not_synthesized():
    """A non-contiguous index set must survive verbatim.

    linspace sampling produces gaps, so range(n) would silently shift every timestamp.
    """
    indices = [0, 2, 5, 9]
    meta = {'fps': 25.0, 'frames_indices': indices, 'total_num_frames': 250}
    out = sce._video_metadata_for_processor(meta, _FakeTensor(4, 1152, 640), fps=25.0)
    assert out[0]['frames_indices'] == indices


def test_real_fps_is_forwarded_not_defaulted():
    """Without metadata the processor falls back to 24 fps, shifting every timestamp."""
    meta = {'fps': 25.0, 'frames_indices': [0, 1], 'total_num_frames': 2}
    out = sce._video_metadata_for_processor(meta, _FakeTensor(2, 64, 64), fps=99.0)
    assert out[0]['fps'] == 25.0, 'metadata fps must win over the engine default'


def test_total_num_frames_comes_from_metadata_not_the_window():
    """total_num_frames describes the source video, not the decoded window."""
    meta = {'fps': 25.0, 'frames_indices': [100, 102], 'total_num_frames': 1144}
    out = sce._video_metadata_for_processor(meta, _FakeTensor(2, 64, 64), fps=25.0)
    assert out[0]['total_num_frames'] == 1144


def test_height_width_come_from_the_tensor():
    """The tensor is post-resize, so it is the only honest source for H/W."""
    meta = {'fps': 25.0, 'frames_indices': [0], 'total_num_frames': 1}
    out = sce._video_metadata_for_processor(meta, _FakeTensor(1, 1152, 640), fps=25.0)
    assert (out[0]['height'], out[0]['width']) == (1152, 640)


def test_missing_frames_indices_falls_back_to_range():
    """Defensive: a decoder that omits the field should not crash the run."""
    out = sce._video_metadata_for_processor({'fps': 25.0}, _FakeTensor(3, 64, 64), fps=25.0)
    assert out[0]['frames_indices'] == [0, 1, 2]


def test_context_length_matches_the_vllm_max_model_len():
    """Comparability requires the same context window on both engines.

    An earlier implementation used 200000 against vLLM's 16384, which is a different
    engine configuration and would invalidate any A/B comparison.
    """
    from transvlm.inference.clip_engine import ENGINE_KWARGS as VLLM_KWARGS

    assert sce.ENGINE_KWARGS['context_length'] == VLLM_KWARGS['max_model_len'] == 16384


def test_greedy_sampling_is_fully_explicit():
    """Every field set, so a changed upstream default cannot masquerade as a regression."""
    assert sce.SAMPLING_KWARGS_SGLANG['temperature'] == 0.0
    assert sce.SAMPLING_KWARGS_SGLANG['top_p'] == 1.0
    assert sce.SAMPLING_KWARGS_SGLANG['top_k'] == -1
    assert sce.SAMPLING_KWARGS_SGLANG['max_new_tokens'] == SAMPLING_KWARGS['max_tokens']


def test_no_prefix_caching_disables_the_radix_cache():
    """SGLang's RadixCache is the counterpart of vLLM's prefix caching."""
    engine = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT, no_prefix_caching=True)
    assert engine.engine_kwargs['disable_radix_cache'] is True

    default = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT)
    assert 'disable_radix_cache' not in default.engine_kwargs


def test_env_knobs_override_engine_kwargs(monkeypatch):
    """The diagnostic ladder must be reachable without editing code."""
    monkeypatch.setenv(sce.ENV_MEM_FRACTION, '0.55')
    monkeypatch.setenv(sce.ENV_CONTEXT_LENGTH, '32768')
    monkeypatch.setenv(sce.ENV_ATTENTION_BACKEND, 'triton')
    monkeypatch.setenv(sce.ENV_KV_CACHE_DTYPE, 'fp8_e4m3')
    engine = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT)
    assert engine.engine_kwargs['mem_fraction_static'] == 0.55
    assert engine.engine_kwargs['context_length'] == 32768
    assert engine.engine_kwargs['attention_backend'] == 'triton'
    assert engine.engine_kwargs['kv_cache_dtype'] == 'fp8_e4m3'


def test_attention_backend_absent_unless_requested():
    """Do not pass the key at all by default, so sglang picks its own per-arch default."""
    assert 'attention_backend' not in sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT).engine_kwargs


def test_infer_before_load_raises():
    engine = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT)
    with pytest.raises(RuntimeError, match='load'):
        engine.infer('a.mp4', 'b.mp4')


def test_prepare_environment_exports_pythonpath_for_spawned_subprocesses(monkeypatch):
    """SGLang spawns Scheduler subprocesses; sys.path alone would fail silently there."""
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.delenv('SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE', raising=False)
    sce.prepare_environment()
    import os

    assert os.environ['SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE'] == sce.PATCH_PKG
    from transvlm._paths import PACKAGE_PARENT

    assert str(PACKAGE_PARENT) in os.environ['PYTHONPATH'].split(os.pathsep)


def test_engine_shares_the_vllm_preprocessing_helpers():
    """Both backends must go through one preprocessing chain, not two copies of it."""
    from transvlm.inference import clip_engine

    assert sce.build_messages is clip_engine.build_messages
    assert sce.read_user_input is clip_engine.read_user_input
    assert sce._rebased is clip_engine._rebased
    assert sce.ClipResult is clip_engine.ClipResult


@pytest.mark.skipif(sglang_missing, reason='sglang not installed in this venv')
def test_upstream_processor_api_still_matches_the_patch():
    """The patch transcribes upstream's method; catch an upstream signature change early."""
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor, preprocess_video

    assert hasattr(QwenVLImageProcessor, 'process_mm_data_async')
    assert hasattr(QwenVLImageProcessor, 'models')

    import asyncio

    assert asyncio.iscoroutinefunction(preprocess_video), 'patch awaits preprocess_video'


@pytest.mark.skipif(sglang_missing, reason='sglang not installed in this venv')
def test_patch_subclass_is_registered_for_qwen3_vl():
    from transvlm.models._sglang_patches import QwenVLImageProcessorOurs

    names = [c.__name__ for c in QwenVLImageProcessorOurs.models]
    assert any('Qwen3VL' in n for n in names), names


@pytest.mark.skipif(sglang_missing, reason='sglang not installed in this venv')
def test_six_channel_patch_embed_needs_no_fork():
    """The reason there is no model-side patch here, asserted rather than assumed."""
    import inspect as _inspect

    from sglang.srt.models.qwen3_vl import Qwen3VLVisionPatchEmbed

    assert 'in_channels' in _inspect.getsource(Qwen3VLVisionPatchEmbed.__init__)


def test_processor_is_called_with_video_kwargs_not_do_resize_false():
    """The single most consequential line in this file. Guard it here, not only in the HF twin.

    vLLM leaves do_resize at its default so the processor resizes again on top of
    process_vision_info's resize. Passing do_resize=False (which an earlier implementation
    did, and which reads as the more sensible choice) produced a 91,423-token prompt against a
    16,384 context — a 5.6x overflow that is very likely the historical "valid JSON but empty
    []" collapse.
    """
    from transvlm.inference import processor_inputs

    # The call itself now lives in one shared place, so assert there and assert the engine
    # actually routes through it rather than hand-rolling its own.
    call = inspect.getsource(processor_inputs.run_client_side_processor)
    assert '**video_kwargs' in call, 'must forward what process_vision_info returned'
    assert 'do_resize=' not in call.split('return processor(')[1], 'must not override do_resize'
    assert 'run_client_side_processor' in inspect.getsource(sce.SGLangClipInferenceEngine.infer)


def test_load_takes_the_timestamp_format_from_the_instance():
    """The `.2f` behaviour must never be unconditional here.

    This started life pinning an env gate (`TRANSVLM_SGLANG_REBIND_PROCESSOR`). The gate is
    gone, replaced by the `timestamp_format` ctor argument, but the invariant it protected is
    the same one: hardcoding either format in `load()` would make this backend disagree with
    the other two, which is how all three silently drifted apart the first time.
    """
    source = inspect.getsource(sce.SGLangClipInferenceEngine.load)
    assert 'apply_timestamp_format(self.timestamp_format' in source, 'format must come from the instance'
    assert 'install_timestamp_patch()' not in source, 'the .2f patch must not be installed unconditionally'


def test_rebind_helper_is_documented_as_unused():
    """`rebind_transformers_processor` is reference material, not a code path.

    If a future edit wires it back into `load()`, the timestamp format would be set in two
    places at once. Its docstring is what tells the next reader that; keep them in sync.
    """
    assert 'Nothing calls this' in (sce.rebind_transformers_processor.__doc__ or '')
    assert 'rebind_transformers_processor()' not in inspect.getsource(sce.SGLangClipInferenceEngine.load)


def test_close_calls_shutdown_not_just_drops_the_reference():
    """SGLang leaks sglang::scheduler subprocesses holding tens of GiB if not shut down."""
    calls = []

    class _FakeEngine:
        def shutdown(self):
            calls.append('shutdown')

    engine = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT)
    engine._engine = _FakeEngine()
    engine.close()
    assert calls == ['shutdown']
    assert engine._engine is None


def test_close_survives_a_shutdown_that_raises():
    """Teardown must not mask the real failure that led to it."""

    class _Boom:
        def shutdown(self):
            raise RuntimeError('ipc gone')

    engine = sce.SGLangClipInferenceEngine(ckpt_dir=_CKPT)
    engine._engine = _Boom()
    engine.close()
    assert engine._engine is None


def test_empty_text_without_a_stop_finish_reason_is_rejected():
    """An aborted request returns empty text and no exception.

    Recording that as a legitimate empty prediction is indistinguishable from "no transition
    here" and is exactly the signature of the historical silent collapse.
    """
    source = inspect.getsource(sce.SGLangClipInferenceEngine.infer)
    assert 'finish_reason' in source
    assert 'refusing to record it as an empty prediction' in source


@pytest.mark.skipif(sglang_missing, reason='sglang not installed in this venv')
def test_patch_passes_the_real_grids_into_get_rope_index():
    """The one deliberate divergence from upstream, and it is a bug fix.

    Upstream passes getattr(ret, 'video_grid_thw', None); on the processor_output path ret is
    None, so get_rope_index falls back to plain sequential positions and ~11k video tokens
    lose Qwen3-VL's 3D M-RoPE. Verified empirically: with grids None all three position rows
    equal arange; with the grids passed they do not, and the three axes differ.
    """
    from transvlm.models._sglang_patches import qwen_vl_processor_6ch as patch

    source = inspect.getsource(patch.QwenVLImageProcessorOurs.process_mm_data_async)
    call = source.split('get_rope_index(')[1]
    assert 'video_grid_thw=video_grid_thw' in call, 'must pass the computed grid, not getattr(ret, ...)'
    assert 'image_grid_thw=image_grid_thw' in call
    assert "video_grid_thw=getattr(ret" not in call, 'upstream form reintroduces the M-RoPE bug'
