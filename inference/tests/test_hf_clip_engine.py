# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the HF engine's contract. No model is loaded, so this runs anywhere.

The HF backend is a correctness reference (~145x slower than vLLM), so what matters is that
it goes through the same preprocessing as the other backends and reports the same
``ClipResult`` — not that it is fast.
"""

from __future__ import annotations

import inspect

import pytest

from transvlm.inference import hf_clip_engine as hce
from transvlm.inference.clip_engine import SAMPLING_KWARGS

# `ckpt_dir` is required rather than defaulted, so every engine construction has to supply one.
# These tests never load weights -- they assert on constructor-assembled kwargs and on guard
# behaviour -- so a path that does not exist is exactly right: if a test ever starts touching the
# filesystem here, it will fail loudly instead of silently reading someone's real checkpoint.
_CKPT = '/nonexistent/transvlm-ckpt'



def test_shares_the_preprocessing_helpers():
    """One preprocessing chain across all three backends, not three copies of it."""
    from transvlm.inference import clip_engine

    assert hce.build_messages is clip_engine.build_messages
    assert hce.read_user_input is clip_engine.read_user_input
    assert hce._rebased is clip_engine._rebased
    assert hce.ClipResult is clip_engine.ClipResult


def test_uses_the_shared_processor_call():
    """Both client-side backends must invoke the processor through one function.

    The metadata builder is reached through it, so asserting the call site is the stronger
    check: it also guarantees the preprocessing dump tool validates what this engine actually sends.
    """
    from transvlm.inference import processor_inputs

    assert hce.run_client_side_processor is processor_inputs.run_client_side_processor


def test_processor_is_called_with_video_kwargs_not_do_resize_false():
    """Passing do_resize=False skips a resize vLLM performs and inflates the prompt ~7x.

    An earlier implementation did exactly that, so this asserts the corrected form survives.
    """
    from transvlm.inference import processor_inputs

    # The call itself now lives in one shared place, so assert there and assert the engine
    # actually routes through it rather than hand-rolling its own.
    call = inspect.getsource(processor_inputs.run_client_side_processor)
    assert '**video_kwargs' in call, 'must forward what process_vision_info returned'
    assert 'do_resize=' not in call.split('return processor(')[1], 'must not override do_resize'
    assert 'run_client_side_processor' in inspect.getsource(hce.HFClipInferenceEngine.infer)


def test_greedy_by_default_without_dangling_sampling_params():
    """do_sample=False makes temperature/top_p meaningless; transformers warns if they are set."""
    assert hce.GENERATE_KWARGS['do_sample'] is False
    assert 'temperature' not in hce.GENERATE_KWARGS
    assert 'top_p' not in hce.GENERATE_KWARGS


def test_max_new_tokens_matches_the_other_backends():
    engine = hce.HFClipInferenceEngine(ckpt_dir=_CKPT)
    assert engine.generate_kwargs['max_new_tokens'] == SAMPLING_KWARGS['max_tokens']


def test_no_prefix_caching_is_accepted_and_ignored():
    """Plain transformers has no prefix cache; the flag exists so 70/71 stay backend-agnostic."""
    engine = hce.HFClipInferenceEngine(ckpt_dir=_CKPT, no_prefix_caching=True)
    assert engine.no_prefix_caching is True


def test_infer_before_load_raises():
    with pytest.raises(RuntimeError, match='load'):
        hce.HFClipInferenceEngine(ckpt_dir=_CKPT).infer('a.mp4', 'b.mp4')


def test_imports_no_vllm_or_sglang_at_module_scope():
    """It runs in the vLLM venv, so it must not depend on either serving engine."""
    header = inspect.getsource(hce).split('class HFClipInferenceEngine')[0]
    for forbidden in ('import vllm', 'import sglang', 'from vllm', 'from sglang'):
        assert forbidden not in header
