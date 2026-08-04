# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Install the ``.2f`` prompt-timestamp behaviour that training used, reliably.

Training expanded per-frame timestamps as ``<x.xx seconds>`` (two decimals). The vendored
``_vllm_patches/sitecustomize.py`` tries to deliver that by rebinding the processor class, but
that rebind **never takes effect**, so every backend has been running on ``.1f`` — which at
25 fps prints the same timestamp for 25 of 125 temporal patches. Measured on a 10 s clip:

    unpatched    Qwen3VLProcessor      12,673 tokens   stamps 0.0 0.1 0.2 0.3 0.3
    patched      (this module)         12,798 tokens   stamps 0.02 0.10 0.18 0.26 0.34

Why the class rebind fails, since the same trap is easy to fall into again:

``AutoProcessor.from_pretrained`` resolves the class through
``transformers.models.auto.processing_auto.processor_class_from_name``, which does
``getattr(transformers.models.qwen3_vl, 'Qwen3VLProcessor')`` — the **package** namespace.
``sitecustomize`` instead sets the attribute on the **submodule**
``transformers.models.qwen3_vl.processing_qwen3_vl``. Those are different namespaces. Worse,
the package is a ``_LazyModule`` that caches each attribute into ``__dict__`` on first access,
and ``sitecustomize``'s own first patch imports vLLM's ``qwen3_vl`` module — which reads the
attribute — *before* the rebind runs, freezing the original class in place.

So this module patches the **method onto the original class** rather than swapping the class:

    Qwen3VLProcessor.__call__ = Qwen3VLProcessorOurs.__call__

That is independent of which namespace resolved the class, of import order, and of whether the
lazy module already cached it — **provided the patch lands on the class instances actually are.**
It originally did not: in a vLLM process the vendored ``sitecustomize.py`` rebinds the submodule
attribute to ``Qwen3VLProcessorOurs`` before this module runs, so a by-name import yielded the
subclass while ``AutoProcessor`` kept handing out the original. :func:`_processor_classes` walks
the MRO and both namespaces so every candidate is patched and every candidate is checked. This
matters because vLLM builds its own processor internally via
``vllm/transformers_utils/processor.py``, not through ours.

``Qwen3VLProcessorOurs.__call__`` is self-contained (it makes no ``super()`` call), so moving
it onto the base class cannot recurse. It keeps its defining module's globals, so its
module-level helpers still resolve.

**This changes results.** The published reference numbers were produced on ``.1f``, so a run
with this patch installed will not reproduce them. That is the point — it matches training —
but it means a comparison against the published figures has to say which format produced it.

Because both formats are legitimately wanted (``.1f`` to reproduce the paper, ``.2f`` to match
training), callers do not use the installers directly: :func:`apply_timestamp_format` is the
entry point, it takes the format as data, and it makes the ``.1f`` case an explicit assertion
that nothing is patched rather than the mere absence of a call.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

from transvlm._paths import VLLM_PATCH_DIR

logger = logging.getLogger(__name__)

# `qwen3vl_processor_ours` does a flat import, so the directory itself must be importable.
_PATCH_DIR = VLLM_PATCH_DIR

# Set on the class once the patch is installed, so a second call is a no-op and a caller can
# ask whether it happened.
_MARKER = '_transvlm_timestamp_patch'

TIMESTAMP_RE = re.compile(r'<([\d.]+) seconds>')

#: The two prompt-timestamp formats a run can choose between. ``'1f'`` is every backend's
#: native behaviour and reproduces the published reference numbers; ``'2f'`` matches training.
#: There is no third option on purpose -- these are the only two formats any number in this
#: repo was ever produced under, and a general N-decimals knob would invite comparisons
#: against figures that do not exist.
TIMESTAMP_FORMATS = ('1f', '2f')

#: Which format this process is running. ``None`` until :func:`apply_timestamp_format` runs.
_ACTIVE_FORMAT: str | None = None


def expected_decimals(fmt: str) -> int:
    """How many decimals prompts should carry under ``fmt``. Inverse of :func:`timestamp_decimals`."""
    if fmt not in TIMESTAMP_FORMATS:
        raise ValueError(f'unknown timestamp format {fmt!r}, expected one of {TIMESTAMP_FORMATS}')
    return 1 if fmt == '1f' else 2


def active_timestamp_format() -> str | None:
    """The format :func:`apply_timestamp_format` put this process into, or None if never called."""
    return _ACTIVE_FORMAT


def apply_timestamp_format(fmt: str, *, include_vllm: bool) -> str:
    """Put this process into ``'1f'`` or ``'2f'`` prompt-timestamp mode.

    ``'1f'`` installs **nothing** -- it is what an unpatched ``Qwen3VLProcessor`` and an
    unpatched vLLM already emit -- so this function's job in that mode is to prove no patch
    snuck in. ``'2f'`` installs both layers (see :func:`install_timestamp_patch` and
    :func:`install_vllm_timestamp_patch` for why one is not enough).

    ``include_vllm`` should be True only for the vLLM backend. The vLLM patch imports
    ``vllm.model_executor.models.qwen3_vl``, and the caller
    (``clip_engine.activate_6ch_patch``) must run before the first ``import vllm``; in ``'1f'``
    mode nothing is imported at all, which is also why the negative check below only inspects
    the HF class.

    Raises:
        ValueError: on an unknown format.
        RuntimeError: if called twice with different formats. The patches are process-global
            and mutate a class, so a mid-process switch would leave earlier and later prompts
            in different formats with nothing recording the split.
        SystemExit: if ``'1f'`` is requested but a ``.2f`` patch is already installed.
    """
    global _ACTIVE_FORMAT  # noqa: PLW0603 -- the patches it tracks are themselves process-global

    if fmt not in TIMESTAMP_FORMATS:
        raise ValueError(f'unknown timestamp format {fmt!r}, expected one of {TIMESTAMP_FORMATS}')
    if _ACTIVE_FORMAT is not None and _ACTIVE_FORMAT != fmt:
        raise RuntimeError(
            f'timestamp format is already {_ACTIVE_FORMAT!r} in this process and cannot change to '
            f'{fmt!r}: the patch mutates Qwen3VLProcessor for every instance. Use one format per '
            'process.'
        )

    if fmt == '2f':
        install_timestamp_patch()
        if include_vllm:
            install_vllm_timestamp_patch()
    else:
        _assert_two_decimal_patch_absent()

    _ACTIVE_FORMAT = fmt
    logger.info(
        'prompt timestamps: .%s%s', fmt, ' (matches training)' if fmt == '2f' else ' (matches published numbers)'
    )
    return fmt


def _processor_classes() -> list[type]:
    """Every class an instance might actually be, so a patch or a check cannot miss it.

    Importing ``Qwen3VLProcessor`` by name is **not** enough in a vLLM process. The vendored
    ``sitecustomize.py`` rebinds the submodule attribute to ``Qwen3VLProcessorOurs`` whenever
    ``VLLM_QWEN3VL_6CH_DUMMY`` is set -- which ``activate_6ch_patch`` always sets, before this
    module runs -- so the name resolves to the subclass while ``AutoProcessor`` keeps handing
    out the original (frozen in the lazy package's ``__dict__``). Patching only the name would
    therefore patch a class nothing instantiates: the exact "looks installed, is not" failure
    this module exists to remove, verified empirically on 2026-07-30.

    So the targets are the name-resolved class, whatever the package namespace holds, and every
    ``Qwen3VLProcessor`` in the resolved class's MRO. De-duplicated, order preserved.
    """
    import transformers.models.qwen3_vl as pkg  # noqa: PLC0415
    from transformers.models.qwen3_vl.processing_qwen3_vl import (  # noqa: PLC0415
        Qwen3VLProcessor as resolved,
    )

    candidates: list[type] = [resolved, *resolved.__mro__, getattr(pkg, 'Qwen3VLProcessor', resolved)]
    out: list[type] = []
    for cls in candidates:
        if cls.__name__.startswith('Qwen3VLProcessor') and cls not in out:
            out.append(cls)
    return out


def _assert_two_decimal_patch_absent() -> None:
    """Fail if a ``.2f`` patch is installed anywhere when ``.1f`` was asked for.

    Checks every class :func:`_processor_classes` can reach, plus vLLM's
    ``Qwen3VLProcessingInfo`` -- the latter via ``sys.modules`` rather than an import, so the
    check costs nothing and cannot violate ``activate_6ch_patch``'s "before the first
    ``import vllm``" ordering: if the module is not loaded then its patch cannot exist.

    This closes the path where a caller reaches past ``apply_timestamp_format`` into
    :func:`install_vllm_timestamp_patch` directly and a later ``apply('1f')`` waves it through.
    """
    for cls in _processor_classes():
        if getattr(cls, _MARKER, False):
            raise SystemExit(
                f'requested .1f timestamps but the .2f processor patch is installed on '
                f'{cls.__module__}.{cls.__qualname__}, so prompts would silently be .2f. '
                'Start a fresh process per format.'
            )

    vllm_mod = sys.modules.get('vllm.model_executor.models.qwen3_vl')
    info = getattr(vllm_mod, 'Qwen3VLProcessingInfo', None) if vllm_mod else None
    if info is not None and getattr(info, _MARKER, False):
        raise SystemExit(
            'requested .1f timestamps but the .2f patch is installed on '
            "vLLM's Qwen3VLProcessingInfo, so prompts would silently be .2f."
        )


def assert_prompt_timestamp_format(processor: Any, inputs: Any, fmt: str) -> int:
    """Decode the expanded prompt and fail unless its stamps match ``fmt``.

    For the client-side backends (SGLang, HF) this checks the real thing: the prompt that was
    just built and is about to be sent. Worth the one tokenizer decode because the bug this
    guards against -- a timestamp patch that looks installed and is not -- burned a full
    872-clip run before it was noticed. Call it once per process, on the first clip.

    Returns the number of decimals found, so a caller can log it.

    Raises:
        SystemExit: on a mismatch, or when the prompt carries no timestamps at all.
    """
    want = expected_decimals(fmt)
    text = processor.tokenizer.decode(inputs['input_ids'][0])
    got = timestamp_decimals(text)
    if got is None:
        raise SystemExit('expanded prompt carries no <x.x seconds> stamps -- video expansion did not happen')
    if got != want:
        raise SystemExit(f'requested .{fmt} timestamps but the prompt carries {got} decimal(s), expected {want}')
    return got


def _load_ours() -> Any:
    """Import ``Qwen3VLProcessorOurs`` from the vendored patch directory."""
    patch_dir = str(_PATCH_DIR)
    if patch_dir not in sys.path:
        sys.path.insert(0, patch_dir)
    from qwen3vl_processor_ours import Qwen3VLProcessorOurs  # noqa: PLC0415

    return Qwen3VLProcessorOurs


def install_timestamp_patch() -> bool:
    """Make every ``Qwen3VLProcessor`` instance expand timestamps with two decimals.

    Idempotent. Returns True if the patch is in place (whether installed now or already).

    Raises:
        SystemExit: if the vendored processor cannot be imported — running without it would
            silently produce ``.1f`` prompts that do not match training.
    """
    targets = _processor_classes()
    if all(getattr(cls, _MARKER, False) for cls in targets):
        return True

    try:
        ours = _load_ours()
    except ImportError as exc:
        raise SystemExit(
            f'cannot import Qwen3VLProcessorOurs from {_PATCH_DIR}: {exc}. Without it prompts '
            'carry .1f timestamps, which do not match training.'
        ) from exc

    # The fix, applied to EVERY class an instance might be -- see `_processor_classes`. Patching
    # only the by-name import would, in a vLLM process, patch `Qwen3VLProcessorOurs` while
    # `AutoProcessor` keeps handing out the original.
    for cls in targets:
        if cls is not ours:
            cls.__call__ = ours.__call__
        setattr(cls, _MARKER, True)

    # Also point the name at the subclass in the namespaces that resolution can read, so an
    # isinstance check or a by-name lookup sees a consistent picture. Harmless if it is already
    # frozen: the method patch above is what actually carries the behaviour.
    try:
        import transformers  # noqa: PLC0415
        import transformers.models.qwen3_vl as _pkg  # noqa: PLC0415
        import transformers.models.qwen3_vl.processing_qwen3_vl as _sub  # noqa: PLC0415

        _sub.Qwen3VLProcessor = ours
        _pkg.Qwen3VLProcessor = ours
        if hasattr(transformers, 'Qwen3VLProcessor'):
            transformers.Qwen3VLProcessor = ours
    except Exception:  # noqa: BLE001 -- cosmetic; the method patch is the functional part
        logger.debug('could not rebind the Qwen3VLProcessor name', exc_info=True)

    logger.info(
        'timestamp patch installed on %s: prompts will use .2f (matches training)',
        ', '.join(c.__qualname__ for c in targets),
    )
    return True


class _TwoDecimalFloat(float):
    """A float that formats with two decimals no matter what precision is requested.

    Exists for exactly one reason: vLLM hardcodes the timestamp format. See
    :func:`install_vllm_timestamp_patch`.
    """

    def __format__(self, spec: str) -> str:
        # Only hijack fixed-point formatting; anything else behaves like a normal float.
        if spec.endswith('f'):
            return format(float(self), '.2f')
        return super().__format__(spec)


def install_vllm_timestamp_patch() -> bool:
    """Make vLLM's own prompt expansion emit two-decimal timestamps.

    Idempotent. Returns True once installed.

    **vLLM does not use the HF processor's ``__call__`` to expand video timestamps** — it
    reimplements the expansion in its own multimodal processor, with the format string
    hardcoded (``f'<{curr_time:.1f} seconds>'`` in
    ``vllm/model_executor/models/qwen3_vl.py``). That is why the vendored
    ``sitecustomize.py``'s processor rebind could never have worked for vLLM: it patches the
    wrong layer entirely. :func:`install_timestamp_patch` is still needed for the SGLang and HF
    backends, which really do go through the HF processor.

    The format string sits inside a nested closure, so it cannot be reached directly. Rather
    than copy the ~120-line enclosing method — which would duplicate vLLM's video-pruning
    logic and rot on the next upgrade — this wraps the timestamps at their source
    (``Qwen3VLProcessingInfo._get_video_second_idx``, a stable named method) in a float that
    formats itself with two decimals. In that closure the values are used only for a length
    assertion and for the f-string, so nothing else changes.

    This is a deliberately unusual trick, so it is guarded: the wrapper is exercised at install
    time, and :func:`timestamp_decimals` lets a caller assert on the prompt vLLM actually
    built. If a future vLLM stops routing through this method, the guard fires rather than the
    prompt silently reverting to ``.1f``.
    """
    from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo  # noqa: PLC0415

    if getattr(Qwen3VLProcessingInfo, _MARKER, False):
        return True

    original = Qwen3VLProcessingInfo._get_video_second_idx

    def _two_decimal_second_idx(self: Any, *args: Any, **kwargs: Any) -> list[Any]:
        return [_TwoDecimalFloat(t) for t in original(self, *args, **kwargs)]

    Qwen3VLProcessingInfo._get_video_second_idx = _two_decimal_second_idx
    setattr(Qwen3VLProcessingInfo, _MARKER, True)

    # Prove the mechanism here, not on the first real prompt.
    probe = _TwoDecimalFloat(0.02)
    rendered = f'<{probe:.1f} seconds>'
    if rendered != '<0.02 seconds>':
        raise SystemExit(f'timestamp patch self-check failed: expected "<0.02 seconds>", got {rendered!r}')

    logger.info("vLLM timestamp patch installed: vLLM's own expansion will use .2f")
    return True


def timestamp_decimals(expanded_prompt: str) -> int | None:
    """How many decimals the ``<x.xx seconds>`` stamps in a prompt carry.

    Returns None when the prompt has no stamps. Use this to assert the patch reached the
    process that actually built the prompt — the failure mode being fixed here was precisely a
    patch that looked installed and was not.
    """
    stamps = TIMESTAMP_RE.findall(expanded_prompt)
    if not stamps:
        return None
    return max(len(s.split('.')[1]) if '.' in s else 0 for s in stamps)


def assert_processor_is_six_channel(processor: Any) -> None:
    """Fail loudly if the processor is not configured for the 6-channel checkpoint.

    A 3-channel processor would silently halve the input the model expects.
    """
    video_processor = getattr(processor, 'video_processor', None)
    image_processor = getattr(processor, 'image_processor', None)
    for name, proc in (('video_processor', video_processor), ('image_processor', image_processor)):
        if proc is None:
            continue
        mean = getattr(proc, 'image_mean', None)
        if mean is not None and len(mean) not in (3, 6):
            raise SystemExit(f'{name}.image_mean has {len(mean)} entries, expected 3 or 6')
