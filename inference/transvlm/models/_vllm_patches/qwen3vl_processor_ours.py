"""
Lightweight Qwen3VLProcessorOurs class for runtime monkey-patching.

This module contains ONLY the custom processor class with minimal dependencies
(only transformers and numpy) to avoid pulling in heavy modules like vLLM,
PIL, torch.distributed, etc. when loaded at Python startup via sitecustomize.

This is functionally identical to the class defined in
the reference implementation, but extracted here
so that sitecustomize.py (and vLLM spawned child processes) can import it
without triggering the entire models_vllm.py import chain.
"""

import logging
from typing import Union

import numpy as np

from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
    Qwen3VLProcessorKwargs,
    ImageInput,
    VideoInput,
    TextInput,
    PreTokenizedInput,
    Unpack,
    BatchFeature,
)

logger = logging.getLogger(__name__)

# Track whether the fps warning has been emitted to avoid spamming logs
_fps_warning_emitted = False


class Qwen3VLProcessorOurs(Qwen3VLProcessor):
    """
    Custom Qwen3VL processor that handles video timestamp processing
    correctly for vLLM.

    Key difference from the base Qwen3VLProcessor:
    When replacing video tokens with per-frame timestamp placeholders,
    this class checks for BOTH the wrapped pattern
    (``<|vision_start|><|video_pad|><|vision_end|>``) and the bare
    ``<|video_pad|>`` token.  vLLM may pass the bare token directly,
    which the upstream implementation does not handle.
    """

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput,
            PreTokenizedInput,
            list[TextInput],
            list[PreTokenizedInput],
        ] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Prepare one or several sequence(s) and image(s) / video(s) for the
        model.  See the base-class docstring for full parameter descriptions.
        """
        global _fps_warning_emitted

        logger.debug("Qwen3VLProcessorOurs __call__")
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # ----- image processing -----
        if images is not None:
            image_inputs = self.image_processor(
                images=images, **output_kwargs["images_kwargs"]
            )
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # ----- video processing -----
        if videos is not None:
            videos_inputs = self.video_processor(
                videos=videos, **output_kwargs["videos_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        # ----- text token replacement -----
        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        # Replace image tokens with the correct number of placeholders
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = (
                        image_grid_thw[index].prod() // merge_length
                    )
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Replace video tokens with per-frame timestamp placeholders
        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size ** 2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        if not _fps_warning_emitted:
                            logger.warning(
                                "Qwen3VL requires frame timestamps to "
                                "construct prompts, but the `fps` of the "
                                "input video could not be inferred. Probably "
                                "`video_metadata` was missing from inputs and "
                                "you passed pre-sampled frames. Defaulting to "
                                "`fps=24`. Please provide `video_metadata` "
                                "for more accurate results."
                            )
                            _fps_warning_emitted = True
                        metadata.fps = (
                            24 if metadata.fps is None else metadata.fps
                        )

                    # Calculate per-frame timestamps
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = (
                        video_grid_thw[index][1:].prod() // merge_length
                    )
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.2f} seconds>"
                        video_placeholder += (
                            self.vision_start_token
                            + "<|placeholder|>" * frame_seqlen
                            + self.vision_end_token
                        )

                    # Check for the fully-wrapped pattern first
                    wrapped_pattern = (
                        f"{self.vision_start_token}"
                        f"{self.video_token}"
                        f"{self.vision_end_token}"
                    )
                    if wrapped_pattern in text[i]:
                        text[i] = text[i].replace(
                            wrapped_pattern,
                            video_placeholder,
                            1,
                        )
                    else:
                        # vLLM may input the video token directly (bare)
                        text[i] = text[i].replace(
                            self.video_token, video_placeholder, 1
                        )
                    index += 1

                text[i] = text[i].replace(
                    "<|placeholder|>", self.video_token
                )

        # ----- tokenization -----
        return_tensors = output_kwargs["text_kwargs"].pop(
            "return_tensors", None
        )
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop(
            "return_mm_token_type_ids", None
        )
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(
            text, text_inputs, modalities=["image", "video"]
        )

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs},
            tensor_type=return_tensors,
        )
