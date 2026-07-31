"""SGLang QwenVLImageProcessor subclass for pre-decoded 6-channel video inputs.

Upstream ``sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor``
iterates ``base_output.videos`` calling ``preprocess_video(video, video_config)``
which requires a decord-style ``VideoReader`` (crashes on dicts with
``AttributeError: 'dict' object has no attribute 'get_avg_fps'``).

For our 6-channel checkpoint we compute ``pixel_values_videos`` +
``video_grid_thw`` client-side using ``Qwen3VLProcessorOurs`` and pass them via
``video_data=[{'format': 'processor_output', 'pixel_values_videos': ...,
'video_grid_thw': ...}]``. These dict items are recognized by
``base_processor.load_mm_data`` (base_processor.py:409-420) and forwarded to
``base_output.videos`` as-is. Only the ``preprocess_video`` call above trips
on them — this subclass skips it and lets
``process_and_combine_mm_data`` (base_processor.py:1023-1042) handle the
dict items through its ``collect_mm_items_from_processor_output`` branch.

Register the containing package via the environment variable
``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=transvlm.models._sglang_patches``
BEFORE calling ``sglang.Engine(...)`` — SGLang's tokenizer_manager scans the
package for subclasses whose ``models`` class attribute lists the loaded
``hf_config.architectures`` entry (see
``sglang/srt/managers/tokenizer_manager.py:267-269``).
"""
from __future__ import annotations

import logging
import time

import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    preprocess_video,
)

logger = logging.getLogger(__name__)


class QwenVLImageProcessorOurs(QwenVLImageProcessor):
    """QwenVLImageProcessor variant that tolerates pre-decoded dict videos."""

    # SGLang dispatches by hf_config.architectures name -> processor via the
    # ``models`` list; ``import_processors(pkg, overwrite=True)`` at
    # tokenizer_manager.py:267-269 overrides the upstream registration
    # for the same class names.
    models = QwenVLImageProcessor.models

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        entry_time = time.perf_counter()
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            video_data=request_obj.video_data,
            audio_data=request_obj.audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        load_time = time.perf_counter()
        rid = getattr(request_obj, 'rid', 'anonymous_rid')

        video_metadata = None
        if base_output.videos:
            processed_videos: list = []
            processed_metadata: list = []
            for video in base_output.videos:
                if isinstance(video, dict):
                    # processor_output / precomputed_embedding passthrough.
                    # Downstream process_and_combine_mm_data recognizes it
                    # via its dict-handling branch (base_processor.py:1023).
                    processed_videos.append(video)
                else:
                    v, m = await preprocess_video(video, video_config=self.video_config)
                    processed_videos.append(v)
                    processed_metadata.append(m)
            base_output.videos = processed_videos
            video_metadata = processed_metadata if processed_metadata else None

        preprocess_time = time.perf_counter()

        # Below is verbatim from the upstream Qwen3VL branch.
        if self.hf_config.model_type in ('qwen3_vl', 'qwen3_vl_moe'):
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output,
                self.mm_tokens,
                video_metadata=video_metadata,
                do_sample_frames=False,
            )
        else:
            mm_items, input_ids, ret = self.process_and_combine_mm_data(
                base_output, self.mm_tokens,
            )

        audio_feature_lengths = None
        if self.model_type == 'qwen3_omni_moe':
            audio_item = next((mm for mm in mm_items if mm.is_audio()), None)
            if audio_item:
                audio_feature_lengths = torch.sum(
                    audio_item.feature_attention_mask, dim=1,
                )

        second_per_grid_ts = getattr(ret, 'second_per_grid_ts', None)
        if second_per_grid_ts is None:
            second_per_grid_ts = getattr(ret, 'video_second_per_grid', None)

        process_time = time.perf_counter()

        input_ids = input_ids.flatten()

        image_grid_thw = None
        if hasattr(ret, 'image_grid_thw'):
            image_grid_thw = ret.image_grid_thw
        if image_grid_thw is None and image_data and isinstance(image_data[0], dict):
            image_grid_thw = image_data[0].get('image_grid_thw')

        video_grid_thw = None
        if hasattr(ret, 'video_grid_thw'):
            video_grid_thw = ret.video_grid_thw
        if video_grid_thw is None and request_obj.video_data:
            first_video = request_obj.video_data[0]
            if isinstance(first_video, dict):
                video_grid_thw = first_video.get('video_grid_thw')

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, 'tokens_per_second', None,
            ),
            input_ids=input_ids.unsqueeze(0),
            # DELIBERATE DIVERGENCE FROM UPSTREAM — the one place this subclass is not a
            # faithful transcription, and it is a bug fix.
            #
            # Upstream passes `getattr(ret, 'video_grid_thw', None)` here. On the
            # processor_output path `ret` is None (process_and_combine_mm_data returns None
            # when there are no raw videos to process), so upstream hands get_rope_index
            # video_grid_thw=None. get_rope_index then falls through to its text branch and
            # assigns plain sequential positions — roughly 12k video tokens lose Qwen3-VL's
            # 3D M-RoPE entirely. The grids computed just above are upstream's own dead code:
            # correct values that are never passed anywhere.
            #
            # This is silent rather than loud: the timestamps are also present as
            # `<x.xx seconds>` text in the prompt, so output stays plausible while position
            # encoding is wrong. Nothing downstream recomputes it — mrope_positions rides
            # MultimodalInputs straight into the model forward.
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_audio_in_video=False,
            audio_seqlens=audio_feature_lengths,
            audio_token_id=getattr(self.hf_config, 'audio_token_id', None),
            audio_start_token_id=self.audio_start_token_id,
            position_id_per_seconds=getattr(
                self.hf_config, 'position_id_per_seconds', None,
            ),
        )
        mrope_positions = mrope_positions.squeeze(1)
        get_rope_index_time = time.perf_counter()

        logger.debug(
            '[QwenVLProcessorOurs Perf] rid=%s '
            'load=%.2fms preprocess=%.2fms process=%.2fms rope=%.2fms total=%.2fms',
            rid,
            (load_time - entry_time) * 1000,
            (preprocess_time - load_time) * 1000,
            (process_time - preprocess_time) * 1000,
            (get_rope_index_time - process_time) * 1000,
            (get_rope_index_time - entry_time) * 1000,
        )

        return {
            'input_ids': input_ids.tolist(),
            'mm_items': mm_items,
            'im_start_id': self.vision_start_token_id,
            'im_end_id': self.vision_end_token_id,
            'im_token_id': self.mm_tokens.image_token_id,
            'video_token_id': self.mm_tokens.video_token_id,
            'audio_token_id': self.mm_tokens.audio_token_id,
            'mrope_positions': mrope_positions,
            'mrope_position_delta': mrope_position_delta,
        }
