# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the backend registry and the cross-backend contract.

These tests must pass in **either** venv, so nothing here imports vllm or sglang: the whole
point of the registry is that only the requested backend's module is ever imported.
"""

from __future__ import annotations

import inspect

import pytest

from transvlm.inference import backends


def test_known_backends():
    assert backends.BACKENDS == ('vllm', 'sglang', 'hf')


def test_unknown_backend_exits_rather_than_returning_none():
    """A typo must fail before an engine loads, not midway through a batch."""
    with pytest.raises(SystemExit):
        backends.resolve_engine_cls('vLLM')  # case matters
    with pytest.raises(SystemExit):
        backends.resolve_engine_cls('nonesuch')


def test_registry_does_not_import_engines_at_module_scope():
    """vLLM and SGLang cannot coexist in one venv, so imports must stay inside the function.

    A module-scope import would break whichever backend is absent from the current venv.
    """
    source = inspect.getsource(backends)
    header = source.split('def resolve_engine_cls')[0]
    for forbidden in ('clip_engine', 'sglang_clip_engine', 'hf_clip_engine', 'import vllm', 'import sglang'):
        assert forbidden not in header, f'{forbidden!r} must not be imported at module scope'


def test_resolve_returns_a_class_with_the_shared_surface():
    """The vLLM engine is always importable here; check the contract on it.

    The other two are checked the same way by their own tests, which skip when their
    package is absent.
    """
    cls = backends.resolve_engine_cls('vllm')
    for name in ('load', 'infer', 'close'):
        assert callable(getattr(cls, name)), f'{cls.__name__} is missing {name}()'

    init = inspect.signature(cls.__init__)
    for kwarg in ('ckpt_dir', 'prompt_file', 'fps', 'max_new_tokens', 'no_prefix_caching'):
        assert kwarg in init.parameters, f'{cls.__name__}.__init__ is missing {kwarg}'

    infer = inspect.signature(cls.infer)
    for kwarg in ('video_start', 'video_end', 'rebase_frame_indices'):
        assert kwarg in infer.parameters, f'{cls.__name__}.infer is missing {kwarg}'
