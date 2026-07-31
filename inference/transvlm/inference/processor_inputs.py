"""Shared client-side processor input construction.

Both the SGLang and HuggingFace backends run the HF processor client-side (vLLM runs it
server-side inside its EngineCore worker), so they need identical ``video_metadata``. Keeping
it here rather than in either backend means a parity fix lands for both at once.
"""

from __future__ import annotations

from typing import Any


def video_metadata_for_processor(metadata: dict[str, Any], tensor: Any, fps: float) -> list[dict[str, Any]]:
    """Reshape ``process_vision_info``'s metadata into what the HF processor expects.

    Two fields are parity-critical, and earlier implementations got both wrong:

    * ``frames_indices`` — forwarded verbatim. Substituting ``list(range(n_frames))`` looks
      harmless but silently shifts every prompt timestamp: ``smart_nframes`` floors the frame
      count to a multiple of 2 and then samples with ``linspace``, so the real index set is
      not always contiguous, and the ``<x.xx seconds>`` stamps are computed from exactly this
      field.
    * ``fps`` — taken from the decode. With no metadata the processor falls back to 24 fps,
      which shifts every stamp as well.

    Explicit metadata is also load-bearing for a third reason specific to this checkpoint:
    transformers 4.57.x infers the channel dimension when metadata is absent, and
    ``infer_channel_dimension_format`` only accepts 1, 3 or 4 channels, so it trips on the
    6-channel tensor.
    """
    n_frames = int(tensor.shape[0])
    frames_indices = list(metadata.get('frames_indices') or range(n_frames))
    meta_fps = float(metadata.get('fps') or fps)
    return [
        {
            'fps': meta_fps,
            'total_num_frames': int(metadata.get('total_num_frames') or n_frames),
            'duration': float(n_frames) / meta_fps if meta_fps else 0.0,
            'frames_indices': frames_indices,
            'height': int(tensor.shape[2]),
            'width': int(tensor.shape[3]),
        }
    ]


def run_client_side_processor(
    processor: Any,
    text: str,
    tensor: Any,
    metadata: dict[str, Any],
    fps: float,
    video_kwargs: dict[str, Any],
) -> Any:
    """Expand the prompt and patchify the video, exactly as vLLM does server-side.

    Single implementation on purpose. This call used to be duplicated in the SGLang engine,
    the HF engine and a preprocessing dump tool; the duplication meant the tool verified its own
    copy rather than what the engines actually send, so one engine could drift while 76 still
    reported parity.

    ``video_kwargs`` is forwarded verbatim — it is what ``process_vision_info`` returned and
    what vLLM passes as ``mm_processor_kwargs``. Do **not** add ``do_resize=False`` here: the
    processor is meant to resize again on top of ``process_vision_info``'s resize. Measured on
    a 1152x640 clip, suppressing it yields grid [125, 72, 40] and 91,423 prompt tokens against
    a 16,384 context, versus [125, 26, 14] and 12,798 with it left alone.
    """
    return processor(
        text=[text],
        videos=[tensor],
        video_metadata=video_metadata_for_processor(metadata, tensor, fps),
        return_tensors='pt',
        **video_kwargs,
    )
