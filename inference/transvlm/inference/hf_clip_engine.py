"""Plain-``transformers`` engine — a correctness reference, not a serving path.

Same public surface as the vLLM and SGLang engines (``load`` / ``infer`` -> ``ClipResult`` /
``close``, plus ``fps`` and ``model_load_sec``) so ``infer_clips.py`` can select it
with ``--backend hf``.

It exists to answer "does the checkpoint produce sane output through the most boring
possible path", so use it on a handful of clips, not for a benchmark. `70`'s own resume logic
makes an interrupted smoke run cheap to continue.

On speed: an earlier measurement on a different GPU was far slower. That did **not** reproduce here — 4 short
clips on a Blackwell GPU came out at RTF 0.52 end-to-end. The earlier figure was a
whole-video run with 5 windows rather than short clips, so the two are not measuring the same
work; treat neither as a benchmark.

It runs in the vLLM venv: it imports no vllm and needs nothing SGLang pins, and
``transformers`` 4.57.6 plus ``torchcodec`` 0.9.1 are already there. So there is no third
venv and no install.

Preprocessing is the shared chain, exactly as in the SGLang engine — ``build_messages`` ->
``process_vision_info`` -> :func:`..processor_inputs.run_client_side_processor`, which is the
single place the processor is invoked so the two client-side backends and
the preprocessing dump tool cannot drift apart.

Unlike SGLang, this backend needs no M-RoPE fix: ``transformers`` computes position ids inside
``generate``, from the real grids.
"""

from __future__ import annotations

import logging
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
from transvlm.inference.processor_inputs import run_client_side_processor
from transvlm.models.processor_patch import (
    apply_timestamp_format,
    assert_prompt_timestamp_format,
)

logger = logging.getLogger(__name__)

# Greedy, to match the other two backends. `do_sample=False` makes temperature and top_p
# meaningless, and transformers warns if they are set anyway, so they are omitted rather than
# passed as None.
GENERATE_KWARGS: dict[str, Any] = {
    'do_sample': False,
}

# bfloat16 matches the checkpoint's own `dtype` and what vLLM infers.
DTYPE = 'bfloat16'
# sdpa rather than flash_attention_2: it is always available, and this backend is a
# correctness reference where portability beats speed.
ATTN_IMPLEMENTATION = 'sdpa'


class HFClipInferenceEngine:
    """Owns a plain transformers model + processor for a process's lifetime."""

    def __init__(
        self,
        *,
        ckpt_dir: str | Path,
        prompt_file: str | Path = DEFAULT_PROMPT_FILE,
        fps: float = DEFAULT_FPS,
        max_new_tokens: int = SAMPLING_KWARGS['max_tokens'],
        no_prefix_caching: bool = False,
        device: str = 'cuda:0',
        timestamp_format: str = DEFAULT_TIMESTAMP_FORMAT,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.prompt_file = Path(prompt_file)
        self.fps = fps
        self.max_new_tokens = max_new_tokens
        # Accepted and ignored: plain transformers has no prefix cache to disable, so the flag
        # is a no-op here rather than an error. Keeping the signature identical is what lets
        # 70 and 71 treat the three backends interchangeably.
        self.no_prefix_caching = no_prefix_caching
        self.device = device
        self.timestamp_format = timestamp_format
        # Verified once, on the first clip -- see the SGLang engine for why.
        self._timestamp_format_verified = False
        self.user_input = read_user_input(self.prompt_file)
        self.generate_kwargs: dict[str, Any] = dict(GENERATE_KWARGS, max_new_tokens=max_new_tokens)
        self._model: Any = None
        self._processor: Any = None
        self.model_load_sec = 0.0

    def load(self) -> None:
        """Load the model and processor. Idempotent."""
        if self._model is not None:
            return

        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        # include_vllm=False: this backend never imports vLLM, and it runs the HF processor
        # itself, so the HF-level patch is the whole story.
        apply_timestamp_format(self.timestamp_format, include_vllm=False)

        started = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(str(self.ckpt_dir))
        self._model = AutoModelForImageTextToText.from_pretrained(
            str(self.ckpt_dir),
            dtype=getattr(torch, DTYPE),
            attn_implementation=ATTN_IMPLEMENTATION,
            device_map={'': self.device},
        )
        self._model.eval()
        self.model_load_sec = time.perf_counter() - started

        in_channels = getattr(self._model.config.vision_config, 'in_channels', None)
        if in_channels != 6:
            raise SystemExit(
                f'checkpoint reports vision_config.in_channels={in_channels}, expected 6; '
                'this is not the 6-channel TransVLM v1 checkpoint'
            )
        logger.info('model ready in %.1f s (in_channels=%d)', self.model_load_sec, in_channels)

    def infer(
        self,
        rgb_video: str | Path,
        flow_video: str | Path,
        *,
        video_start: float | None = None,
        video_end: float | None = None,
        rebase_frame_indices: bool = False,
    ) -> ClipResult:
        """Run one clip. Expect tens of minutes per clip, not seconds."""
        if self._model is None:
            raise RuntimeError('call load() before infer()')

        import torch  # noqa: PLC0415

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

        inputs = run_client_side_processor(self._processor, text, tensor, metadata, self.fps, video_kwargs)
        if not self._timestamp_format_verified:
            decimals = assert_prompt_timestamp_format(self._processor, inputs, self.timestamp_format)
            logger.info('prompt timestamps verified: %d decimal(s)', decimals)
            self._timestamp_format_verified = True
        inputs = inputs.to(self.device)
        n_input_tokens = int(inputs['input_ids'].shape[1])
        preprocess_sec = time.perf_counter() - pre_started

        gen_started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **self.generate_kwargs)
        generate_sec = time.perf_counter() - gen_started

        # Drop the echoed prompt; generate() returns prompt + completion.
        completion_ids = generated[0][n_input_tokens:]
        raw_text = self._processor.tokenizer.decode(completion_ids, skip_special_tokens=True)

        return ClipResult(
            raw_text=raw_text,
            preprocess_sec=preprocess_sec,
            generate_sec=generate_sec,
            n_input_tokens=n_input_tokens,
            n_output_tokens=int(completion_ids.shape[0]),
            n_frames=int(tensor.shape[0]),
            height=int(tensor.shape[2]),
            width=int(tensor.shape[3]),
        )

    def close(self) -> None:
        """Release the model and empty the CUDA cache."""
        self._model = None
        self._processor = None
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 -- teardown must not mask a real failure
            logger.debug('torch.cuda.empty_cache() failed during close', exc_info=True)
