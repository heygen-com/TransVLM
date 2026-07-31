"""One SGLang engine, reused across many clips or windows — the vLLM engine's sibling.

Mirrors :class:`transvlm.inference.clip_engine.ClipInferenceEngine`'s public
surface (``load`` / ``infer`` / ``close``, plus the ``fps`` and ``model_load_sec``
attributes) and returns the same :class:`ClipResult`, so ``infer_clips.py`` and
``infer_video.py`` can swap backends without changing their output schema. That
matters: the scoring entry point and the comparison script both key off that schema.

**Preprocessing is shared, not reimplemented.** The backend-neutral helpers
(``build_messages``, ``read_user_input``, ``_rebased``) and the whole
``process_vision_info`` chain are imported from ``clip_engine``, so both backends decode,
resize (``smart_resize`` / pixel budget) and channel-concatenate through the *same* code.
Raw video never reaches the engine. Only the final hand-off differs, and it has to:

* vLLM takes the ``(N, 6, H, W)`` tensor plus its metadata and runs the HF processor
  **server-side** inside the EngineCore worker.
* SGLang 0.5.8 cannot do that, so the processor runs **client-side** here and the engine
  receives ``pixel_values_videos`` + ``video_grid_thw`` + already-expanded ``input_ids``.

One measurement consequence to keep in mind when reading RTF numbers: this backend's
``preprocess_sec`` includes patchify and timestamp expansion, while vLLM's does not — that
cost sits inside vLLM's ``generate_sec`` instead. Only end-to-end wall clock is directly
comparable across the two.

Two facts about SGLang worth stating because they are the opposite of vLLM's:

* **No model fork is needed for 6 channels.** ``Qwen3VLVisionPatchEmbed`` reads
  ``in_channels`` from the config, so the checkpoint's 6-channel patch embed loads natively.
* **SGLang has no multimodal dummy profiling**, so vLLM's ``sitecustomize.py`` dummy-video
  patch has no counterpart and is not needed. The flip side is that SGLang never measures
  the ViT activation peak, so ``mem_fraction_static`` is a static guess that does not know
  about it — which is why that knob is exposed here.

What *is* still needed is the mm-processor override: upstream's ``preprocess_video`` assumes
a decord ``VideoReader`` and dies on our pre-computed dicts, and upstream also drops the video
grid on this path, which silently costs the video tokens their 3D M-RoPE (see
``_sglang_patches/qwen_vl_processor_6ch.py``).

Prompt timestamp precision is a per-run choice (``timestamp_format``, wired to
``--timestamp-format``), identical to the other two backends. ``.1f`` is this processor's
native behaviour and reproduces the published reference numbers; ``.2f`` matches training. The
class *rebind* that ``Qwen3VLProcessorOurs`` represents is never applied — measurement showed
it cannot take effect — so ``.2f`` is delivered by patching the method instead; see
:func:`rebind_transformers_processor` for that measurement and
``transvlm/models/processor_patch.py`` for the mechanism.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from transvlm.inference.clip_engine import (
    DEFAULT_FPS,
    DEFAULT_PROMPT_FILE,
    DEFAULT_TIMESTAMP_FORMAT,
    SAMPLING_KWARGS,
    ClipResult,
    _rebased,
    build_messages,
    read_user_input,
)
from transvlm.inference.processor_inputs import (
    run_client_side_processor,
    video_metadata_for_processor as _video_metadata_for_processor,
)
from transvlm.models.processor_patch import (
    apply_timestamp_format,
    assert_prompt_timestamp_format,
)

from transvlm._paths import PACKAGE_PARENT

logger = logging.getLogger(__name__)

# SGLang's tokenizer_manager reads this package name from
# SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE and scans it for processor subclasses.
PATCH_PKG = 'transvlm.models._sglang_patches'

# Engine kwargs chosen to mirror the vLLM engine's posture rather than to be fast.
#
# `context_length` matches vLLM's `max_model_len` (16384) exactly. The archived
# implementation used 200000, which is a different engine configuration and would make any
# comparison meaningless.
#
# `mem_fraction_static` has no vLLM equivalent worth copying: vLLM's
# `gpu_memory_utilization=0.2` works because vLLM *measures* the ViT peak during profiling and
# budgets around it, whereas this is a blind static split. Tunable because the historical
# high-resolution failure was first misdiagnosed as pure OOM.
#
# 0.45 rather than the inherited 0.70, measured on this GPU 2026-07-30. At 0.70 the engine's
# peak was 88.4 GiB of the 97 GiB card -- 91%, roughly twice vLLM's 45.0 GiB -- leaving 6.6 GiB
# free, which is less than the ~6.1-6.9 GiB the GPU flow visualisation needs. 29 of 30 clips
# therefore fell through `OnlineFlowComputer._fits_on_gpu` onto the CPU path and silently lost
# the 19.5x speedup that vLLM was getting from the same code. Note 0.70 does not mean 68 GiB:
# the fraction covers weights + KV cache only, and activations plus CUDA context stack on top.
#
# At 0.45 (and 0.30, identical in measurement) the fallback disappears: flow_to_image median
# 4.257 s -> 0.218 s and the whole flow stage 227.7 s -> 95.7 s over 30 clips, against a ~5%
# rise in generate (122.6 s -> 128.8 s) from the smaller KV cache. 0.45 is the larger of the
# two working values, so it keeps the most KV headroom that still frees the visualisation.
#
# The 0.70 figure came from a 44 GiB card, where it was 31 GiB. Anything comparing against the
# 2026-07-29 SGLang run  should know that run was at 0.70.
ENGINE_KWARGS: dict[str, Any] = {
    'dtype': 'bfloat16',
    'trust_remote_code': False,
    'mem_fraction_static': 0.45,
    'context_length': 16384,
    'kv_cache_dtype': 'auto',
    'log_level': 'info',
    # Skips graph capture. Kept off deliberately: capture buys throughput, not peak memory,
    # and window widths vary, so recapture per shape is a known source of surprises here.
    'disable_cuda_graph': True,
    # Single sequential requests need no scheduler overlap, and overlap has been observed to
    # spin the Scheduler subprocess while the caller waits on a stalled IPC round-trip.
    'disable_overlap_schedule': True,
}

# Sampling params. vLLM's `seed` / `top_k` / `stop_token_ids` have no per-request equivalent
# in SGLang, so greedy is expressed the way SGLang expresses it. Every field is set
# explicitly rather than left to a default, because a silent default change between versions
# would look like a model regression.
SAMPLING_KWARGS_SGLANG: dict[str, Any] = {
    'temperature': 0.0,
    'top_p': 1.0,
    'top_k': -1,
    'max_new_tokens': SAMPLING_KWARGS['max_tokens'],
}

# Env knobs for the diagnostic ladder, so a rung can be tried without editing code.
ENV_MEM_FRACTION = 'TRANSVLM_SGLANG_MEM_FRACTION_STATIC'
ENV_CONTEXT_LENGTH = 'TRANSVLM_SGLANG_CONTEXT_LENGTH'
ENV_ATTENTION_BACKEND = 'TRANSVLM_SGLANG_ATTENTION_BACKEND'
ENV_KV_CACHE_DTYPE = 'TRANSVLM_SGLANG_KV_CACHE_DTYPE'

# `TRANSVLM_SGLANG_REBIND_PROCESSOR` used to gate the .2f timestamp patch here. It is gone:
# the format is now the `timestamp_format` ctor argument, wired to `--timestamp-format` on the
# runtime scripts, so all three backends express it the same way. An env knob on one backend
# only was exactly the asymmetry that let the three drift apart unnoticed.


def prepare_environment() -> None:
    """Install the mm-processor override and make it importable from subprocesses.

    Must run before ``sglang.Engine(...)``. SGLang calls
    ``mp.set_start_method('spawn', force=True)``, so a parent-process monkey-patch would
    fail **silently** in the Scheduler subprocesses — the repo root has to be on their
    ``PYTHONPATH``, not merely on this process's ``sys.path``.
    """
    os.environ['SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE'] = PATCH_PKG

    # PyTorch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF; set both so the
    # setting applies regardless of which name this build honours.
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    root = str(PACKAGE_PARENT)
    if root not in sys.path:
        sys.path.insert(0, root)
    existing = os.environ.get('PYTHONPATH', '')
    entries = existing.split(os.pathsep) if existing else []
    if root not in entries:
        os.environ['PYTHONPATH'] = os.pathsep.join([root, *entries]) if entries else root


def rebind_transformers_processor() -> None:
    """Point ``transformers``' Qwen3VLProcessor at our 6-channel subclass. Idempotent.

    **Nothing calls this.** It is kept because its measurement below is the evidence for why
    the class rebind cannot deliver ``.2f`` timestamps — the conclusion that
    ``transvlm/models/processor_patch.py`` acts on by patching the *method* onto the
    original class instead. Timestamp format is now chosen per run via
    ``apply_timestamp_format``; the ``TRANSVLM_SGLANG_REBIND_PROCESSOR`` env knob this function
    used to sit behind is gone.

    vLLM's ``sitecustomize.py`` performs this same rebind and its docstring says the point is
    to move prompt timestamps from ``.1f`` to ``.2f``. Measured on a 10 s clip, that rebind
    **has no effect on what ``AutoProcessor.from_pretrained`` returns** — AutoProcessor
    resolves the class through its own registry, which already holds a direct reference to the
    original, so patching the module attribute afterwards is too late:

        no patch          Qwen3VLProcessor      12,673 tokens   stamps 0.0 0.1 0.2 0.3 0.3
        sitecustomize     Qwen3VLProcessor      12,673 tokens   stamps 0.0 0.1 0.2 0.3 0.3
        this rebind       Qwen3VLProcessorOurs  12,798 tokens   stamps 0.02 0.10 0.18 0.26

    And vLLM's returned ``prompt_token_ids`` for that clip is 12,673 with ``.1f`` stamps,
    matching the unpatched expansion token for token — so the reference results this project
    reproduces were produced with ``.1f``, duplicate stamps and all. Calling this function
    would therefore make SGLang diverge from vLLM rather than agree with it.

    Kept because it is the only way to test the ``.2f`` hypothesis, and because silently
    dropping it would hide the discrepancy from whoever reads the vLLM docstring next.

    The subclass lives under ``_vllm_patches/`` but is pure ``transformers`` code with no
    vLLM import, so sharing it across backends is deliberate, not an accident of layout.
    """
    import transformers  # noqa: PLC0415
    import transformers.models.qwen3_vl.processing_qwen3_vl as _mod  # noqa: PLC0415

    from transvlm.models._vllm_patches.qwen3vl_processor_ours import (  # noqa: PLC0415
        Qwen3VLProcessorOurs,
    )

    _mod.Qwen3VLProcessor = Qwen3VLProcessorOurs
    if hasattr(transformers, 'Qwen3VLProcessor'):
        transformers.Qwen3VLProcessor = Qwen3VLProcessorOurs


def _warn_on_vllm_coexistence() -> None:
    """Warn if vllm is importable: the two pin incompatible flashinfer versions.

    ``pyproject.toml`` declares the groups mutually exclusive, so this state means uv's
    solver was bypassed and SGLang may die on a flashinfer symbol mismatch.
    """
    import importlib.util  # noqa: PLC0415

    if importlib.util.find_spec('vllm') is not None:
        logger.warning(
            'both vllm and sglang are importable in this environment; they pin different '
            'flashinfer-python versions (0.5.3 vs 0.6.1) and sglang may crash on a symbol '
            'mismatch. Use the dedicated SGLang venv described in the inference README.'
        )


class SGLangClipInferenceEngine:
    """Owns the SGLang engine and the client-side processor for a process's lifetime."""

    def __init__(
        self,
        *,
        ckpt_dir: str | Path,
        prompt_file: str | Path = DEFAULT_PROMPT_FILE,
        fps: float = DEFAULT_FPS,
        max_new_tokens: int = SAMPLING_KWARGS['max_tokens'],
        no_prefix_caching: bool = False,
        timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.prompt_file = Path(prompt_file)
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        self.no_prefix_caching = no_prefix_caching
        self.timestamp_format = timestamp_format
        # Verified once, on the first clip: the failure this guards against is a patch that
        # looks installed and is not, which cost a full 872-clip run before it was spotted.
        self._timestamp_format_verified = False
        self.user_input = read_user_input(self.prompt_file)

        self.engine_kwargs: dict[str, Any] = dict(ENGINE_KWARGS, model_path=str(self.ckpt_dir))
        self.engine_kwargs['mem_fraction_static'] = float(
            os.environ.get(ENV_MEM_FRACTION, ENGINE_KWARGS['mem_fraction_static'])
        )
        self.engine_kwargs['context_length'] = int(os.environ.get(ENV_CONTEXT_LENGTH, ENGINE_KWARGS['context_length']))
        self.engine_kwargs['kv_cache_dtype'] = os.environ.get(ENV_KV_CACHE_DTYPE, ENGINE_KWARGS['kv_cache_dtype'])
        attention_backend = os.environ.get(ENV_ATTENTION_BACKEND)
        if attention_backend:
            self.engine_kwargs['attention_backend'] = attention_backend
        if no_prefix_caching:
            # SGLang's RadixCache is the counterpart of vLLM's prefix caching. vLLM's was
            # measured moving a boundary 1.98 -> 1.99 when the same request was resubmitted,
            # so any A/B comparison needs it off on both sides.
            self.engine_kwargs['disable_radix_cache'] = True

        self.sampling_kwargs: dict[str, Any] = dict(SAMPLING_KWARGS_SGLANG, max_new_tokens=max_new_tokens)
        self._engine: Any = None
        self._processor: Any = None
        self.model_load_sec = 0.0

    def load(self) -> None:
        """Build the engine. Idempotent."""
        if self._engine is not None:
            return

        _warn_on_vllm_coexistence()
        prepare_environment()
        # include_vllm=False: this backend runs the HF processor client-side, so the HF-level
        # patch is the whole story here. Importing vLLM's module would also pull vLLM into a
        # process that is deliberately kept free of it.
        apply_timestamp_format(self.timestamp_format, include_vllm=False)

        from transformers import AutoProcessor  # noqa: PLC0415

        started = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(str(self.ckpt_dir))

        # Imported only after the env vars are set, so tokenizer_manager picks up the
        # external mm-processor override.
        import sglang  # noqa: PLC0415

        logger.info('sglang engine kwargs: %s', self.engine_kwargs)
        self._engine = sglang.Engine(**self.engine_kwargs)
        self.model_load_sec = time.perf_counter() - started
        logger.info('engine ready in %.1f s', self.model_load_sec)

    def infer(
        self,
        rgb_video: str | Path,
        flow_video: str | Path,
        *,
        video_start: float | None = None,
        video_end: float | None = None,
        rebase_frame_indices: bool = False,
    ) -> ClipResult:
        """Run one request. One request per generate call.

        Single requests are mandatory, not merely tidy: batching several videos into one
        ``Engine.generate`` on 0.5.8 raises ``Mismatch: More 'VIDEO' tokens found than
        corresponding data provided`` and returns empty text for all but the last request.
        It also matches the vLLM path, which sends one request for determinism.

        ``rebase_frame_indices`` has the same meaning as on the vLLM engine: a range-decoded
        window returns ABSOLUTE frame numbers, and prompt timestamps derive from that field,
        so without rebasing the model would see absolute video time where a pre-cut clip
        shows window-local time.
        """
        if self._engine is None:
            raise RuntimeError('call load() before infer()')

        # Routed through the loader so the decoder choice (torchcodec, else decord with a
        # warning) is settled before the vendored module captures it at import time.
        from transvlm.data.video_backend import load_process_vision_info  # noqa: PLC0415

        process_vision_info = load_process_vision_info()

        pre_started = time.perf_counter()
        messages = build_messages(
            rgb_video,
            flow_video,
            self.user_input,
            self.fps,
            video_start=video_start,
            video_end=video_end,
        )
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if not video_inputs:
            raise RuntimeError('process_vision_info returned no video input')
        tensor, metadata = video_inputs[0]
        if rebase_frame_indices:
            metadata = _rebased(metadata)

        # Client-side processing, with EXACTLY the kwargs vLLM sends server-side:
        # `{'do_sample_frames': False}` and nothing else. In particular `do_resize` is left at
        # its default of True, so the processor applies its own resize on top of the one
        # `process_vision_info` already did. That double resize looks redundant but it is the
        # protocol the reference results were produced with, and skipping it is catastrophic
        # rather than merely different: measured on a 1152x640 clip,
        #
        #     do_resize=False  ->  grid [125, 72, 40], 91,423 prompt tokens
        #     do_resize=True   ->  grid [125, 26, 14], 12,798 prompt tokens  (vLLM: 12,673)
        #
        # and `context_length` is 16384, so the first form overflows the window by 5.6x. The
        # earlier implementation passed do_resize=False, which is very likely the real cause
        # of the historical "valid but empty []" collapse at high resolution that was
        # attributed to an upstream vision-encoder regression.
        inputs = run_client_side_processor(self._processor, text, tensor, metadata, self.fps, video_kwargs)
        if not self._timestamp_format_verified:
            decimals = assert_prompt_timestamp_format(self._processor, inputs, self.timestamp_format)
            logger.info('prompt timestamps verified: %d decimal(s)', decimals)
            self._timestamp_format_verified = True
        input_ids = inputs['input_ids'][0].tolist()
        video_data = [
            {
                'format': 'processor_output',
                'pixel_values_videos': inputs['pixel_values_videos'],
                'video_grid_thw': inputs['video_grid_thw'],
            }
        ]
        preprocess_sec = time.perf_counter() - pre_started

        gen_started = time.perf_counter()
        result = self._engine.generate(
            input_ids=input_ids,
            sampling_params=self.sampling_kwargs,
            video_data=video_data,
        )
        generate_sec = time.perf_counter() - gen_started

        if isinstance(result, list):
            if len(result) != 1:
                raise RuntimeError(f'expected exactly 1 output, got {len(result)}')
            result = result[0]

        meta_info = result.get('meta_info') or {}
        # An aborted request (context overflow, scheduler abort) comes back with empty text
        # and no exception, which would be written to the jsonl as a legitimate empty
        # prediction — indistinguishable from "the model saw no transition" and exactly the
        # signature of the historical silent-collapse. Refuse to report it as a result.
        finish_type = (meta_info.get('finish_reason') or {}).get('type')
        if finish_type not in (None, 'stop', 'length'):
            raise RuntimeError(f'sglang returned finish_reason={finish_type!r} for {rgb_video}')
        if not result.get('text') and finish_type != 'stop':
            raise RuntimeError(
                f'sglang returned empty text with finish_reason={finish_type!r} for {rgb_video}; '
                'refusing to record it as an empty prediction'
            )

        return ClipResult(
            raw_text=result.get('text', ''),
            preprocess_sec=preprocess_sec,
            generate_sec=generate_sec,
            n_input_tokens=int(meta_info.get('prompt_tokens') or len(input_ids)),
            n_output_tokens=int(meta_info.get('completion_tokens') or 0),
            n_frames=int(tensor.shape[0]),
            height=int(tensor.shape[2]),
            width=int(tensor.shape[3]),
        )

    def close(self) -> None:
        """Shut the engine down so the Scheduler subprocesses actually exit.

        SGLang leaves ``sglang::scheduler`` processes holding tens of GiB of VRAM if the
        engine is merely dropped, and they survive a ``pkill`` aimed at the parent — so this
        calls ``shutdown()`` rather than just clearing the reference.
        """
        if self._engine is not None:
            shutdown = getattr(self._engine, 'shutdown', None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:  # noqa: BLE001 -- teardown must not mask a real failure
                    logger.warning('engine shutdown raised; scheduler may need killing by name')
        self._engine = None
