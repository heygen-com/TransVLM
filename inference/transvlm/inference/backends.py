# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Backend registry: map a ``--backend`` name to its engine class.

Lives in the package because both entry scripts need it, and entry scripts must never import
each other.

Every engine class exposes the same surface — ``load()``, ``infer(...) -> ClipResult``,
``close()``, plus ``fps`` and ``model_load_sec`` — so the callers stay backend-agnostic and
the output jsonl schema is identical whichever backend produced it. That identity is what
lets ``a sharding launcher``, ``the comparison tool`` and
``the scoring entry point`` work unchanged across backends.

The import is deferred per backend, and that is load-bearing rather than stylistic: vLLM and
SGLang pin incompatible ``flashinfer-python`` versions (0.5.3 vs 0.6.1) and are declared
mutually exclusive in ``pyproject.toml``, so only one of them exists in any given venv.
Importing at module scope would break the other backend's venv on a missing package or a
flashinfer symbol mismatch.
"""

from __future__ import annotations

from typing import Any

BACKENDS = ('vllm', 'sglang', 'hf')


def validate_ckpt_dir(ckpt_dir: Any) -> Any:
    """Fail at load time, in the user's terms, when ``--ckpt-dir`` is not a checkpoint.

    Called from each engine's ``load()`` rather than its ``__init__`` so that constructing an
    engine stays free of filesystem requirements: tests build one against a path that will never
    exist, and a caller may legitimately construct before fetching weights.

    Without this the first thing to touch the path is ``AutoProcessor.from_pretrained``, which
    treats anything that is not an existing directory as a Hub repo id and reports "Repo id must
    be in the form 'repo_name' or 'namespace/repo_name'". For someone who passed a local path
    that message points at the wrong problem entirely.

    Raises:
        SystemExit: if the path is missing, is not a directory, or has no ``config.json``.
    """
    from pathlib import Path  # noqa: PLC0415

    path = Path(ckpt_dir)
    if not path.exists():
        raise SystemExit(f'--ckpt-dir does not exist: {path}')
    if not path.is_dir():
        raise SystemExit(f'--ckpt-dir is not a directory: {path}')
    if not (path / 'config.json').exists():
        raise SystemExit(
            f'--ckpt-dir has no config.json, so it is not a checkpoint directory: {path}\n'
            'Download one with: hf download HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct '
            '--local-dir ./pretrained/TransVLM-v1'
        )
    return path


def missing_backend_exit(backend: str, package: str, exc: ImportError) -> SystemExit:
    """Turn a bare ImportError for an optional backend into an actionable message.

    vLLM and SGLang are optional extras that cannot share a venv, so "not installed" is an
    ordinary state rather than a broken install, and it deserves an instruction instead of a
    traceback.
    """
    return SystemExit(
        f'--backend {backend} needs the {package!r} package, which is not installed in this '
        f'environment ({exc}).\n'
        'vLLM and SGLang pin incompatible flashinfer versions and need separate venvs; see the '
        '"Optional: vLLM or SGLang" section of inference/README.md. The default --backend hf '
        'needs no extra install.'
    )


def resolve_engine_cls(backend: str) -> Any:
    """Return the engine class for ``backend``, importing only that backend's module.

    Raises:
        SystemExit: on an unknown backend name, so a typo fails before the engine loads
            rather than midway through a batch.
    """
    if backend == 'vllm':
        from transvlm.inference.clip_engine import ClipInferenceEngine  # noqa: PLC0415

        return ClipInferenceEngine
    if backend == 'sglang':
        from transvlm.inference.sglang_clip_engine import (  # noqa: PLC0415
            SGLangClipInferenceEngine,
        )

        return SGLangClipInferenceEngine
    if backend == 'hf':
        from transvlm.inference.hf_clip_engine import HFClipInferenceEngine  # noqa: PLC0415

        return HFClipInferenceEngine
    raise SystemExit(f'unknown backend {backend!r}; expected one of {BACKENDS}')


def assert_output_backend_matches(output_jsonl: str, backend: str) -> None:
    """Refuse to append to a jsonl produced by a different backend.

    Resume keys on the ``video`` path alone, so pointing ``--backend sglang`` at a file a
    vLLM run wrote would silently skip those clips and produce a mixed-backend file that
    scores as if it came from one engine. Records written before ``config.backend`` existed
    have no backend recorded; those are let through with no claim either way.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    if not os.path.isfile(output_jsonl):
        return
    with open(output_jsonl, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                existing = (json.loads(line).get('config') or {}).get('backend')
            except ValueError:
                return  # a truncated first line says nothing about the backend
            if existing is not None and existing != backend:
                raise SystemExit(
                    f'{output_jsonl} already holds {existing!r} records but --backend is '
                    f'{backend!r}; appending would produce a mixed-backend file. Use a '
                    'different --output-jsonl.'
                )
            return
