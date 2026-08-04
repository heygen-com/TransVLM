"""One vLLM engine, reused across many clips or windows.

Shared by both entry points so the clip script and the whole-video script take exactly one
code path into the model. Engine startup is ~25 s, so it must be paid
once per process, never per item.

The protocol here is the one the published numbers were produced with: a
``{'type': 'video', 'rgb_and_of': {...}}`` message carrying two FILE PATHS, handed to the
vendored `_vision_process_of.process_vision_info`, which decodes both clips, resizes them
and concatenates on the channel axis into ``(N, 6, H, W)``.

Sliding windows ride the same protocol via ``video_start`` / ``video_end`` — see
:meth:`ClipInferenceEngine.infer`. Nothing is cut on disk.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transvlm._paths import DEFAULT_PROMPT_FILE, VLLM_PATCH_DIR

logger = logging.getLogger(__name__)

# The directory ITSELF goes on the path, not its parent: sitecustomize does a flat
# `from qwen3vl_processor_ours import ...`.
PATCH_DIR = VLLM_PATCH_DIR

# There is deliberately no default checkpoint directory. The weights are ~9 GB and live wherever
# the user downloaded them, so `--ckpt-dir` is required rather than defaulted -- a stale default
# pointing at someone else's filesystem fails in a far more confusing way than a missing argument.
# See the inference README for how to fetch them.

# Appended to the prompt file's contents.
USER_INPUT_SUFFIX = (
    'Please output the shot transition information for the provided video. Your output MUST be a single, '
    'raw JSON array of objects. Do not wrap it in markdown like ```json or add any explanatory text.'
)

# Engine kwargs, taken verbatim from the reference implementation. Deliberately NOT tuned for
# any particular host: keeping them identical is what makes results comparable.
# `limit_mm_per_prompt` must keep image=0 — the 6-channel patch only replaces the dummy
# VIDEO builder, so a dummy IMAGE would still be 3-channel and would crash the shared patch
# embed during profiling. `dtype` is intentionally absent: the reference implementation does not pass it and vLLM
# infers bfloat16 from the checkpoint config.
ENGINE_KWARGS: dict[str, Any] = {
    'seed': 42,
    'mm_encoder_tp_mode': 'data',
    'enable_expert_parallel': False,
    'tensor_parallel_size': 1,
    'data_parallel_size': 1,
    'gpu_memory_utilization': 0.2,
    # vLLM refuses to read a local media path outside this root. It is '/' because the two
    # inputs live in unrelated places by design: the RGB/flow files are written under the temp
    # dir, while --no-pre-resize feeds the caller's original video paths directly, and those can
    # be on any filesystem. vLLM only ever opens paths this process hands it, so this is not a
    # remote-read surface, but if your videos and temp dir share a root you can tighten it by
    # editing this dict before constructing the engine.
    'allowed_local_media_path': '/',
    'max_model_len': 16384,
    'max_num_seqs': 2,
    'max_num_batched_tokens': 16384,
    'limit_mm_per_prompt': {'video': 1, 'image': 0},
    'media_io_kwargs': {'video': {'num_frames': -1}},
    'attention_backend': 'FLASH_ATTN',
}

# Sampling params, verbatim from the reference implementation. temperature=0 for determinism.
SAMPLING_KWARGS: dict[str, Any] = {
    'seed': 42,
    'temperature': 0,
    'max_tokens': 2048,
    'top_k': -1,
    'stop_token_ids': [],
}

DEFAULT_FPS = 25.0

#: Prompt-timestamp format when a caller does not choose. ``'1f'`` is every backend's native
#: behaviour and is what produced the published reference numbers, so it is the default for a
#: repo whose job is reproducing them; ``'2f'`` prints two decimal places and must be asked for.
#: The two are not interchangeable: at 25 fps the ``.1f`` form prints identical timestamps for 25
#: of 125 temporal patches, so scores from the two formats must never be compared.
DEFAULT_TIMESTAMP_FORMAT = '1f'

#: Per-frame pixel budget for the ffmpeg pre-resize. 524,288 rather than the paper's 786,432,
#: measured over all 872 clips on 2026-07-30: it is better on BOTH metrics (ttmean cl_f1
#: 0.782401 vs 0.777722, tt0.00 0.608677 vs 0.587988 -- and above the published 0.779161) while
#: being 1.18x faster.
#:
#: Note what this knob really does: the processor renormalises to the same grid either way, so
#: the token count does not change (735 of 872 clips are identical) and generate time is flat.
#: It is a picture-quality knob, and the measured effect of the downscale-upscale round trip is
#: fewer spurious frame-level candidates. Pass 786432 to reproduce the paper's configuration.
DEFAULT_MAX_PIXELS = 524288

#: Override `ENGINE_KWARGS['attention_backend']`, mirroring the SGLang engine's equivalent knob.
#: Exists for one diagnostic: SGLang and vLLM disagree by |delta ttmean cl_f1| = 0.0042 while
#: running different kernels (triton vs FLASH_ATTN, the former forced because flashinfer cannot
#: JIT for sm_120 here), and the only way to tell whether the kernel explains the residual is to
#: run vLLM on triton too. Recorded in each jsonl's `backend_env`.
ENV_VLLM_ATTENTION_BACKEND = 'TRANSVLM_VLLM_ATTENTION_BACKEND'


@dataclass
class ClipResult:
    """One generate call's output plus the two timings the RTF bases need."""

    raw_text: str
    preprocess_sec: float
    generate_sec: float
    n_input_tokens: int
    n_output_tokens: int
    n_frames: int
    height: int
    width: int


def activate_6ch_patch(timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT) -> Path:
    """Install the 6-channel patches. MUST run before the first `import vllm`.

    `PYTHONPATH` is the load-bearing half: vLLM spawns its EngineCore workers, and a
    spawned interpreter auto-imports `sitecustomize` from `PYTHONPATH` at startup.
    `sys.path` plus the explicit import below covers this already-booted process.

    Without this, memory profiling builds a 3-channel dummy video, feeds it to the
    checkpoint's `in_channels=6` Conv3d patch embed, and dies before the weights finish
    loading.

    sitecustomize's *second* patch is meant to change prompt timestamps from `.1f` to `.2f`,
    which is what training used — at 25 fps the `.1f` form prints identical timestamps for 25
    of 125 temporal patches. That patch never worked (it targets the wrong namespace and runs
    after the lazy package has cached the original class), so the behaviour is installed here
    instead, via `apply_timestamp_format`. See
    `transvlm/models/processor_patch.py` for why the class rebind cannot work.

    `timestamp_format` selects which format this process runs: `'1f'` reproduces the published
    reference numbers, `'2f'` matches training. It is threaded from the CLI rather than fixed
    here because both are legitimately wanted, and leaving it implicit is what let all three
    backends silently disagree.
    """
    if not (PATCH_DIR / 'sitecustomize.py').is_file():
        raise SystemExit(f'6-channel patch not found at {PATCH_DIR}. Inference would crash during profiling.')

    patch_str = str(PATCH_DIR)
    existing = os.environ.get('PYTHONPATH', '')
    if patch_str not in existing.split(os.pathsep):
        os.environ['PYTHONPATH'] = f'{patch_str}{os.pathsep}{existing}' if existing else patch_str
    if patch_str not in sys.path:
        sys.path.insert(0, patch_str)

    os.environ['VLLM_QWEN3VL_6CH_DUMMY'] = '1'
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    # vLLM's telemetry thread raises a JSONDecodeError out of cpuinfo on this GPU.
    os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')

    import sitecustomize  # noqa: F401,PLC0415 -- the patch applies as an import side effect

    # sitecustomize's second patch (.1f -> .2f timestamps) does not actually take effect: it
    # rebinds the submodule attribute, while AutoProcessor resolves the class from the package
    # namespace, and the lazy package has already cached the original by the time it runs.
    # Install the behaviour by patching the method instead, which is order-independent.
    from transvlm.models.processor_patch import apply_timestamp_format  # noqa: PLC0415

    # include_vllm because vLLM does NOT use the HF processor's __call__ to expand timestamps —
    # it reimplements the expansion with the format hardcoded, so `.2f` needs both layers. The
    # HF-level one still matters for anything in this process that calls the processor directly.
    apply_timestamp_format(timestamp_format, include_vllm=True)

    logger.info('6-channel patches activated from %s', PATCH_DIR)
    return PATCH_DIR


def read_user_input(prompt_file: str | Path = DEFAULT_PROMPT_FILE) -> str:
    """Prompt file contents + the training-time suffix.

    `.strip()` is load-bearing: the file ends with a newline, so without it the suffix
    lands after a blank line, which is a different prompt and shifts every predicted
    boundary.
    """
    return f'{Path(prompt_file).read_text().strip()}\n{USER_INPUT_SUFFIX}'


def build_messages(
    rgb_video: str | Path,
    flow_video: str | Path,
    user_input: str,
    fps: float,
    *,
    video_start: float | None = None,
    video_end: float | None = None,
) -> list[dict[str, Any]]:
    """Build the `rgb_and_of` message, as in the reference implementation, plus time ranges.

    The outer block keeps `type: 'video'` so the chat template still emits the
    vision-start / video-pad / vision-end triple, but carries no `'video'` key — that is
    what routes it to `process_vision_info`'s `rgb_and_of` branch.

    `video_start` / `video_end` are read by `fetch_video` and select a sub-range without
    cutting anything on disk. Callers doing sliding windows must pass an end compensated by
    half a frame — see `RANGE_END_EPSILON_NOTE` in `infer_video.py`.
    """

    def _one(path: str | Path) -> dict[str, Any]:
        block: dict[str, Any] = {
            'type': 'video',
            'video': str(path),
            'fps': fps,
            'sample_fps': fps,
            'raw_fps': fps,
        }
        if video_start is not None:
            block['video_start'] = video_start
        if video_end is not None:
            block['video_end'] = video_end
        return block

    return [
        {
            'role': 'user',
            'content': [
                {'type': 'video', 'rgb_and_of': {'rgb_video': _one(rgb_video), 'of_video': _one(flow_video)}},
                {'type': 'text', 'text': user_input},
            ],
        }
    ]


class ClipInferenceEngine:
    """Owns the vLLM engine and the processor for the lifetime of a process."""

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
        # Verified once per process, on the first clip, against what vLLM itself expanded.
        self._timestamp_format_verified = False
        self.user_input = read_user_input(self.prompt_file)
        self.engine_kwargs: dict[str, Any] = dict(ENGINE_KWARGS, model=str(self.ckpt_dir))
        attention_backend = os.environ.get(ENV_VLLM_ATTENTION_BACKEND)
        if attention_backend:
            self.engine_kwargs['attention_backend'] = attention_backend
            logger.info('attention_backend overridden to %s', attention_backend)
        if no_prefix_caching:
            # With caching on, resubmitting the same request to one engine changes the
            # answer (measured: a boundary moved 1.98 -> 1.99 on call #2), which would
            # contaminate any A/B comparison. Costs ~0.4% wall clock here because only
            # ~2.3% of the prompt is cacheable.
            self.engine_kwargs['enable_prefix_caching'] = False
        self.sampling_kwargs: dict[str, Any] = dict(SAMPLING_KWARGS, max_tokens=max_new_tokens)
        self._llm: Any = None
        self._processor: Any = None
        self._sampling: Any = None
        self.model_load_sec = 0.0

    def load(self) -> None:
        """Build the engine. Idempotent."""
        if self._llm is not None:
            return
        activate_6ch_patch(self.timestamp_format)

        from transformers import AutoProcessor  # noqa: PLC0415
        try:
            from vllm import LLM, SamplingParams  # noqa: PLC0415
        except ImportError as exc:
            from transvlm.inference.backends import missing_backend_exit  # noqa: PLC0415

            raise missing_backend_exit('vllm', 'vllm', exc) from exc

        started = time.perf_counter()
        from transvlm.inference.backends import validate_ckpt_dir  # noqa: PLC0415

        validate_ckpt_dir(self.ckpt_dir)
        self._processor = AutoProcessor.from_pretrained(str(self.ckpt_dir))
        self._llm = LLM(**self.engine_kwargs)
        self._sampling = SamplingParams(**self.sampling_kwargs)
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
        """Run one request. One request per generate call, for determinism.

        `rebase_frame_indices` shifts the decoded `frames_indices` so they start at 0.
        Required whenever a time range is used: a range-decoded window comes back with
        ABSOLUTE frame numbers, and prompt timestamps are computed from that field, so the
        model would see absolute video time where a pre-cut clip shows window-local time.
        Rebasing (not resetting to `range(n)`) is what reproduces the pre-cut clip exactly,
        because `smart_nframes` floors the count to a multiple of 2 and then samples with
        `linspace`, so the index set is not always contiguous.
        """
        if self._llm is None:
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
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            if rebase_frame_indices:
                video_inputs = [(t, _rebased(m)) for t, m in video_inputs]
            mm_data['video'] = video_inputs
        request = {'prompt': text, 'multi_modal_data': mm_data, 'mm_processor_kwargs': video_kwargs}
        preprocess_sec = time.perf_counter() - pre_started

        gen_started = time.perf_counter()
        outputs = self._llm.generate([request], sampling_params=self._sampling)
        generate_sec = time.perf_counter() - gen_started
        if len(outputs) != 1:
            raise RuntimeError(f'expected exactly 1 output, got {len(outputs)}')

        if not self._timestamp_format_verified:
            # vLLM expands the prompt inside the engine, so the install-time self-check proves
            # the mechanism exists but not that it reached the path that actually built this
            # prompt. `prompt_token_ids` is that prompt, so decode it once and assert. Without
            # this the reference backend is the only one with no runtime guard -- and it is the
            # one whose numbers get compared to the paper.
            from transvlm.models.processor_patch import (  # noqa: PLC0415
                assert_prompt_timestamp_format,
            )

            decimals = assert_prompt_timestamp_format(
                self._processor, {'input_ids': [outputs[0].prompt_token_ids or []]}, self.timestamp_format
            )
            logger.info('prompt timestamps verified: %d decimal(s)', decimals)
            self._timestamp_format_verified = True

        completion = outputs[0].outputs[0]
        payload = mm_data.get('video') or []
        tensor = payload[0][0] if payload else None
        n_frames, height, width = (
            (int(tensor.shape[0]), int(tensor.shape[2]), int(tensor.shape[3])) if tensor is not None else (0, 0, 0)
        )
        return ClipResult(
            raw_text=completion.text,
            preprocess_sec=preprocess_sec,
            generate_sec=generate_sec,
            n_input_tokens=len(outputs[0].prompt_token_ids or []),
            n_output_tokens=len(completion.token_ids),
            n_frames=n_frames,
            height=height,
            width=width,
        )

    def close(self) -> None:
        """Drop the engine so NCCL / EngineCore teardown happens before interpreter exit."""
        self._llm = None
        self._sampling = None


def _rebased(metadata: dict[str, Any]) -> dict[str, Any]:
    """Shift `frames_indices` so it starts at 0, leaving everything else alone.

    `fps` deliberately stays as-is; the timestamp path reads only `frames_indices` and
    `fps`, and a pre-cut clip's metadata also reports the true source fps.
    """
    out = dict(metadata)
    idx = list(out.get('frames_indices') or [])
    if idx:
        base = idx[0]
        out['frames_indices'] = [i - base for i in idx]
    return out
