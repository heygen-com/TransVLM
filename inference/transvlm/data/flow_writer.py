"""Encode flow-visualization frames to an H.264-encoded MP4.

Matches the training pipeline ``save_tensor_as_video`` semantics used to produce the
training-time sibling ``_flow.mp4`` files: libx264, yuv420p (default from
torchvision), no audio. Input is a ``(N, 3, H, W) uint8`` tensor; the writer
permutes to ``(N, H, W, C)`` before handing off to ``torchvision.io.write_video``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.io


#: Encoder settings per fidelity mode.
#:
#: ``libx264`` is the pinned training-time setting and is **lossy in a way that matters here**:
#: torchvision defaults to ``yuv420p``, so chroma is 4:2:0 subsampled. Flow visualisation encodes
#: direction as *hue*, so that subsampling attacks the signal directly -- measured on two autoshot
#: clips, round-tripping costs up to 121/255 peak and ~1.3 mean absolute error per channel.
#:
#: ``lossless`` uses ``libx264rgb`` at ``crf=0``, which round-trips bit-exactly (verified: max abs
#: error 0). It costs ~25x the bytes (20 MB vs 0.8 MB for a 10 s clip).
FLOW_CODECS: dict[str, tuple[str, dict[str, str] | None]] = {
    'libx264': ('libx264', None),
    'lossless': ('libx264rgb', {'crf': '0'}),
}


def write_flow_mp4(
    frames: torch.Tensor,
    output_path: str | Path,
    *,
    fps: float,
    video_codec: str = 'libx264',
    options: dict[str, str] | None = None,
) -> Path:
    """Write flow-visualization frames to an mp4.

    Args:
        frames: ``(N, 3, H, W) uint8`` tensor. Values already in ``[0, 255]``
            (typically the output of ``torchvision.utils.flow_to_image``).
        output_path: destination filename. Parent directory is created if
            missing.
        fps: frame rate to embed in the container.
        video_codec: passed straight to ``torchvision.io.write_video``. See
            :data:`FLOW_CODECS` for the two calibrated choices.
        options: encoder options forwarded to ``torchvision.io.write_video``.

    Returns:
        The output path (as ``Path``).

    Raises:
        ValueError: on shape / dtype mismatch.
    """
    if frames.ndim != 4:
        raise ValueError(f'Expected 4-D tensor (N, C, H, W). Got ndim={frames.ndim}')
    if frames.shape[1] != 3:
        raise ValueError(f'Expected 3 channels (BGR/RGB flow viz). Got channels={frames.shape[1]}')
    if frames.dtype != torch.uint8:
        raise ValueError(f'Expected torch.uint8 frames (0..255). Got dtype={frames.dtype}')

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # torchvision.io.write_video expects (T, H, W, C) uint8 on CPU.
    frames_thwc = frames.permute(0, 2, 3, 1).contiguous().cpu()
    torchvision.io.write_video(str(out), frames_thwc, fps=fps, video_codec=video_codec, options=options)
    return out
