# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Guard `infer_video.py` against the name-resolution bug that broke every video.

The `selected_video_backend` import was placed inside
`main()` while the call site lives in `_process_video`, which resolves it as a global. Every
video raised `NameError`, the per-item `except Exception` swallowed it, the output jsonl came
out empty, and the whole test suite stayed green because **nothing tested the entry script at all**.

These tests exist because of that. `infer_video.py` is a command beside the package, not part of it,
so it is loaded by path. Note `sys.modules` registration before `exec_module` is required --
`@dataclass` looks the module up by `cls.__module__` and fails with a confusing
`AttributeError: 'NoneType' object has no attribute '__dict__'` without it.
"""

from __future__ import annotations

import importlib.util
import logging
import symtable
import sys
from pathlib import Path

import pytest

# `infer_video.py` sits beside the package as a command, not inside it, so it is loaded by path.
_ENTRY_DIR = Path(__file__).resolve().parents[1]
_SCRIPT = _ENTRY_DIR / 'infer_video.py'


@pytest.fixture(scope='module')
def source() -> str:
    return _SCRIPT.read_text()


def _load_module():
    spec = importlib.util.spec_from_file_location('_test_infer_video', _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_module_imports_without_pulling_in_a_backend():
    """Cheap enough to stay at module scope: `video_backend` imports only stdlib eagerly.

    If this starts failing because torch/vllm/sglang appear, the module-level import has grown
    a heavy transitive dependency and `--help` just got slow -- move it back into a function
    AND move the call site with it.

    **Only meaningful when this file runs on its own.** In a full-suite run an earlier test module
    has already imported torch, so the set difference is empty regardless. Kept because the
    standalone signal is the one that matters here, but do not read a green full-suite run as
    evidence.
    """
    before = {'torch', 'vllm', 'sglang'} & set(sys.modules)
    _load_module()
    grown = ({'torch', 'vllm', 'sglang'} & set(sys.modules)) - before
    assert not grown, f'module-level import now pulls in {sorted(grown)}'


def test_selected_video_backend_resolves_at_module_scope():
    """The actual bug: a function-local import cannot serve a call site in another function."""
    module = _load_module()
    assert callable(getattr(module, 'selected_video_backend', None))


def test_every_function_can_see_every_global_it_references(source: str):
    """Catch the whole class of bug, across every function -- not just the one that had it.

    The original defect was an import inside `main()` serving a call site in `_process_video`. The
    same mistake in `_resample`, `_maybe_resize`, `_read_pairs` or any other sibling is identical in
    kind, so every top-level function is checked. symtable already knows all of them, so scoping
    this to one function would have been a smaller test for no less work.
    """
    table = symtable.symtable(source, str(_SCRIPT), 'exec')
    module_names = {s.get_name() for s in table.get_symbols()}
    builtin_names = set(vars(__import__('builtins')))

    problems: dict[str, list[str]] = {}
    for child in table.get_children():
        if child.get_type() != 'function':
            continue
        missing = sorted(
            s.get_name()
            for s in child.get_symbols()
            if s.is_global() and s.get_name() not in module_names and s.get_name() not in builtin_names
        )
        if missing:
            problems[child.get_name()] = missing
    assert not problems, f'functions referencing globals that do not exist at module scope: {problems}'


def test_all_failed_returns_nonzero_and_names_the_fault(monkeypatch, tmp_path, caplog):
    """Behavioural, not a source-text match.

    A string check on 'pipeline fault' would pass on the COMMENT alone -- delete the log call and
    keep the comment and it stays green. So drive `main` with a `_process_video` that always
    raises and assert on what it actually does.
    """
    module = _load_module()

    def _always_raises(*_args, **_kwargs):
        raise RuntimeError('synthetic')

    video = tmp_path / 'a.mp4'
    video.write_bytes(b'')
    flow = tmp_path / 'a_flow.mp4'
    flow.write_bytes(b'')
    out = tmp_path / 'out.jsonl'
    monkeypatch.setattr(module, '_process_video', _always_raises)
    monkeypatch.setattr(module, '_resolve_engine_cls', lambda _b: _StubEngineCls)
    # --flow supplied so main() takes the 'flow given' branch instead of building an
    # OnlineFlowComputer, which would load NeuFlow weights for a test that never decodes a frame.
    # The file is never read: _maybe_flow lives inside the stubbed _process_video.
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'infer_video.py',
            '--video',
            str(video),
            '--flow',
            str(flow),
            '--output-jsonl',
            str(out),
            '--backend',
            'hf',
            '--ckpt-dir',
            str(tmp_path / 'ckpt'),
        ],
    )

    with caplog.at_level(logging.ERROR):
        assert module.main() == 1
    assert out.read_text() == '', 'nothing should have been written'
    assert 'pipeline fault' in caplog.text, 'the log must say this is not bad input'


class _StubEngineCls:
    """Minimal stand-in so `main` can reach its loop without loading a real model."""

    model_load_sec = 0.0

    def __init__(self, **_kwargs) -> None:
        self.fps = 25.0

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_entry_dir_is_not_a_package():
    """The entry scripts are commands, not modules, so their directory is deliberately not a package.

    If someone adds one, these tests should be rewritten as ordinary imports rather than
    silently keeping the fragile path-loading.
    """
    assert not (_ENTRY_DIR / '__init__.py').exists()
