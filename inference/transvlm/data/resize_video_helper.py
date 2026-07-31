"""ffmpeg-based video resize + full training-side smart_resize formula.

Mirrors the training pipeline's ``resize_video_for_qwen3vl`` step
used by the training pipeline
so that flow generation lands on the identical resized-video substrate
the training pipeline used.

Two functions:

* ``smart_resize_for_qwen3vl(h, w, *, nframes=250, image_patch_size=16)``:
  reproduces the parent's ``get_qwen3vl_suitable_hw_of_video`` — the version
  with the ``MODEL_SEQ_LEN``-driven ``total_pixels / nframes * FRAME_FACTOR``
  penalty. Returns ``(h_bar, w_bar)`` guaranteed to be multiples of 32.

* ``resize_video_ffmpeg(src, dst, h, w)``: subprocess call mirroring
  ``resize_video_ffmpeg`` (libx264 crf=17 preset=fast, scale filter with
  default bicubic when we pass ``scale=W:H``).

``nframes=250`` is hardcoded upstream in training data prep (10s window ×
25 fps). Callers that use different fps/window combos should still pass
``nframes=250`` here to keep the resized-dim identical to the training
distribution — otherwise the ``max_pixels`` penalty term shifts.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from transvlm.data.smart_resize import (
    FRAME_FACTOR,
    SPATIAL_MERGE_SIZE,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_MIN_TOKEN_NUM,
    smart_resize,
)

logger = logging.getLogger(__name__)

# Training-side default. The training pipeline reads env var ``MODEL_SEQ_LEN`` but with a
# 128000 fallback (the training pipeline); we honor the same env var so
# tests can dial it, but default to 128000 to match the training pipeline.
_MODEL_SEQ_LEN_ENV = 'MODEL_SEQ_LEN'
_DEFAULT_MODEL_SEQ_LEN = 128_000

# Training pipeline hardcodes 250 frames per 10-second window. Callers
# should NOT override unless they know why — this value drives the
# ``max_pixels`` term of smart_resize.
DEFAULT_NFRAMES_FOR_RESIZE = 250

# ffmpeg command constants — kept identical to
# ``resize_video_ffmpeg`` in the training pipeline for bit-close output.
_FFMPEG_CRF = '17'
_FFMPEG_PRESET = 'fast'
_FFMPEG_THREADS = '8'


def _get_model_seq_len() -> int:
    raw = os.environ.get(_MODEL_SEQ_LEN_ENV)
    if raw is None:
        return _DEFAULT_MODEL_SEQ_LEN
    return int(float(raw))


def smart_resize_for_qwen3vl(
    height: int,
    width: int,
    *,
    nframes: int = DEFAULT_NFRAMES_FOR_RESIZE,
    image_patch_size: int = 16,
    max_pixels_override: int | None = None,
) -> tuple[int, int]:
    """Full training-side smart_resize formula (with seq_len penalty).

    Verbatim port of ``get_qwen3vl_suitable_hw_of_video`` from the training
    pipeline. Guarantees:

    * Result is a multiple of ``image_patch_size * SPATIAL_MERGE_SIZE`` (32
      for the default patch=16).
    * ``h*w`` is within ``[min_pixels, max_pixels]`` where max_pixels is
      the min of ``VIDEO_MAX_TOKEN_NUM * factor**2`` and the seq-len-driven
      per-frame budget.
    * Aspect ratio preserved.

    Args:
        height, width: source video dimensions.
        nframes: number of sampled frames per window; drives the seq-len
            penalty. Use ``250`` (default) to match training.
        image_patch_size: vision-model patch size; ``16`` for Qwen3-VL.
        max_pixels_override: if set, use this value as ``max_pixels``
            directly, bypassing the seq-len penalty formula. Used by the
            max_pixels sweep experiment (post-Plan-A ablation) to characterize
            accuracy/speed vs pixel-budget trade-off. When None (default) the training
            formula is applied verbatim.
    """
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE  # = 32
    min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    if max_pixels_override is not None:
        max_pixels = int(max_pixels_override)
        # Guard: overrides below min_pixels floor silently degenerate to floor
        # for all callers — warn once so downstream sweep reports don't hide
        # a config that never actually shrunk the target.
        #
        # 2026-07-29: this used to warn and then fall through, at which point
        # smart_resize raised ValueError('max_pixels must be >= min_pixels') —
        # contradicting the message right above, which promises a clamp. Clamp for
        # real so the documented behaviour is the actual behaviour.
        if max_pixels < min_pixels:
            logger.warning(
                'max_pixels_override=%d is below min_pixels floor=%d; clamping to the '
                'floor. This config is degenerate w.r.t. all overrides <= floor.',
                max_pixels,
                min_pixels,
            )
            max_pixels = min_pixels
    else:
        total_pixels = _get_model_seq_len() * image_factor * image_factor * 0.9
        max_pixels_ceiling = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
        max_pixels_per_frame = total_pixels / nframes * FRAME_FACTOR
        max_pixels = int(
            max(
                min(max_pixels_ceiling, max_pixels_per_frame),
                int(min_pixels * 1.05),
            )
        )

    return smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def resize_video_ffmpeg(
    src: str | Path,
    dst: str | Path,
    *,
    target_h: int,
    target_w: int,
    threads: int = 8,
    overwrite: bool = True,
) -> None:
    """Resize a video via ``ffmpeg -filter:v scale=w:h``.

    Mirrors ``resize_video_ffmpeg`` in the training pipeline:
    libx264 encoder, crf=17, preset=fast, no audio re-encode overrides
    beyond what ffmpeg picks by default. The default ffmpeg ``scale`` filter
    uses bicubic interpolation for spatial downsampling.

    Raises:
        FileNotFoundError: if ffmpeg is not on PATH.
        RuntimeError: if ffmpeg exits non-zero.
    """
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError(
            'ffmpeg is not on PATH. Install it (conda-forge, your system package '
            'manager, or a static build) and make sure the process can see it.'
        )
    src = str(src)
    dst = str(dst)
    if not overwrite and os.path.exists(dst):
        # Must be checked here, not left to ffmpeg's `-n`: that flag now guards the temp name,
        # which is always fresh, so ffmpeg would happily encode and `os.replace` would clobber
        # `dst` regardless -- silently inverting the caller's request.
        logger.info('skip resize, destination exists and overwrite=False: %s', dst)
        return
    # Encode to a sibling temp file and rename into place. ffmpeg writing `dst` directly means a
    # concurrent reader (or a second run of a different arm needing the same `_resized.mp4`) can
    # decode a half-written file and compute flow on garbage frames, with nothing downstream
    # noticing. os.replace is atomic within a filesystem, and the temp name carries the pid so two
    # writers cannot collide on it either.
    tmp_dst = f'{dst}.tmp{os.getpid()}.mp4'
    cmd = [
        'ffmpeg',
        '-threads',
        str(threads),
        '-filter_threads',
        str(threads),
        '-y' if overwrite else '-n',
        '-v',
        'error',  # quiet unless something's wrong
        '-i',
        src,
        '-filter:v',
        f'scale={target_w}:{target_h}',
        '-c:v',
        'libx264',
        '-x264-params',
        f'threads={threads}',
        '-crf',
        _FFMPEG_CRF,
        '-preset',
        _FFMPEG_PRESET,
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        tmp_dst,
    ]
    logger.info(
        'ffmpeg scale %s -> %s (%dx%d)',
        Path(src).name,
        Path(dst).name,
        target_h,
        target_w,
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg failed (rc={result.returncode}) on {src}\n' f'stderr:\n{result.stderr}')
        os.replace(tmp_dst, dst)
    finally:
        # A failed encode leaves the temp behind; a partial file lying around next to the real one
        # is exactly the confusion this change exists to prevent.
        if os.path.exists(tmp_dst):
            os.unlink(tmp_dst)


def probe_video_fps_exact(path: str | Path) -> Fraction:
    """Return the first video stream's ``avg_frame_rate`` as an exact ``Fraction``.

    A float is not good enough here. ``torchvision.io.write_video`` rounds a float rate to the
    nearest integer -- ``fps=24.964387`` lands in the container as ``25/1`` -- while ffmpeg's own
    scale pass preserves the exact ``17525/702``. That mismatch is what made two whole videos fail
    the ``rgb_video_sample_fps == of_video_sample_fps`` assertion in the pinned
    ``_vision_process_of.py``: the RGB and flow siblings declared different rates, so
    ``smart_nframes`` sampled a different number of frames from each.

    Passing a ``Fraction`` through instead survives verbatim (verified: ``Fraction(17525, 702)``
    round-trips as ``17525/702``), which lets the flow sibling declare exactly what its RGB source
    declares. For the common ``25/1`` case this is a no-op -- both the float and the Fraction write
    ``25/1`` -- so existing outputs are unchanged.
    """
    if shutil.which('ffprobe') is None:
        raise FileNotFoundError('ffprobe is not on PATH (install ffmpeg and ensure it is on PATH).')
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=avg_frame_rate',
        '-of',
        'default=nw=1:nk=1',
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    try:
        rate = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{path}: cannot parse avg_frame_rate {raw!r}') from exc
    if rate <= 0:
        # Streams with no usable rate report 0/0. Callers previously fell back to 25.0 on a
        # missing float, so keep that behaviour rather than crashing a whole batch.
        raise ValueError(f'{path}: non-positive avg_frame_rate {raw!r}')
    return rate


def probe_video_hw(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` of the first video stream via ffprobe."""
    if shutil.which('ffprobe') is None:
        raise FileNotFoundError('ffprobe is not on PATH (install ffmpeg and ensure it is on PATH).')
    cmd = [
        'ffprobe',
        '-v',
        'error',
        '-select_streams',
        'v:0',
        '-show_entries',
        'stream=width,height',
        '-of',
        'csv=p=0:s=x',
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w_str, h_str = result.stdout.strip().split('x')
    return int(h_str), int(w_str)
