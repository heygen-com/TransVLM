# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Frame resolution helper — mirrors Qwen3-VL smart_resize behavior.

Ported verbatim (down to constants) from
the preprocessing the checkpoint was trained with.
Keeping this in lock-step with the training pipeline implementation is what
guarantees the pixel budget the model saw during training is preserved at
inference time.

Do NOT tune the constants without a paper-alignment re-run.
"""

from __future__ import annotations

import math

# Aspect ratio guard — anything more extreme than this is rejected as pathological.
MAX_RATIO = 200

# Spatial merge factor built into the Qwen3-VL vision encoder.
SPATIAL_MERGE_SIZE = 2

# Token budgets for images vs videos.
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768

# Frame sampling budget for video inputs.
FRAME_FACTOR = 2
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: float, factor: int) -> int:
    """Return the closest integer to ``number`` divisible by ``factor``."""
    return round(number / factor) * factor


def ceil_by_factor(number: float, factor: int) -> int:
    """Return the smallest integer ≥ ``number`` divisible by ``factor``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: float, factor: int) -> int:
    """Return the largest integer ≤ ``number`` divisible by ``factor``."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Compute a rescaled (H, W) that satisfies the Qwen3-VL invariants.

    Guarantees:
      1. Both dimensions are divisible by ``factor``.
      2. Total pixels stay within ``[min_pixels, max_pixels]``.
      3. Aspect ratio is preserved as closely as possible.

    ``min_pixels`` / ``max_pixels`` default to the image token budget
    (``IMAGE_MIN_TOKEN_NUM * factor**2`` and ``IMAGE_MAX_TOKEN_NUM * factor**2``).
    Callers that want video-scale limits should compute those from
    ``VIDEO_MIN_TOKEN_NUM`` / ``VIDEO_MAX_TOKEN_NUM`` themselves.
    """
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * factor**2
    if min_pixels is None:
        min_pixels = IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError(f'max_pixels ({max_pixels}) must be >= min_pixels ({min_pixels})')

    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(f'aspect ratio {ratio:.3f} exceeds cap {MAX_RATIO}')

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def smart_nframes(
    total_frames: int,
    video_fps: float,
    *,
    target_fps: float,
    min_frames: int = FPS_MIN_FRAMES,
    max_frames: int = FPS_MAX_FRAMES,
) -> int:
    """Pick the number of frames to sample from a video to match ``target_fps``.

    Enforces:
      * Result is divisible by ``FRAME_FACTOR`` (== 2).
      * Result is within ``[min_frames, min(max_frames, total_frames)]``.
      * Result is within ``[FRAME_FACTOR, total_frames]``.

    Raises ``ValueError`` if the invariants cannot be satisfied.
    """
    if video_fps <= 0:
        raise ValueError(f'video_fps must be positive, got {video_fps}')
    if total_frames <= 0:
        raise ValueError(f'total_frames must be positive, got {total_frames}')

    min_frames_aligned = ceil_by_factor(min_frames, FRAME_FACTOR)
    max_frames_aligned = floor_by_factor(min(max_frames, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * target_fps
    nframes = min(max(nframes, min_frames_aligned), max_frames_aligned)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not FRAME_FACTOR <= nframes <= total_frames:
        raise ValueError(
            f'nframes must be in [{FRAME_FACTOR}, {total_frames}], got {nframes} '
            f'(total_frames={total_frames}, video_fps={video_fps}, target_fps={target_fps})'
        )
    return nframes
