# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``smart_resize`` + ``smart_nframes``."""

from __future__ import annotations

import pytest

from transvlm.data.smart_resize import (
    FRAME_FACTOR,
    IMAGE_MAX_TOKEN_NUM,
    IMAGE_MIN_TOKEN_NUM,
    MAX_RATIO,
    SPATIAL_MERGE_SIZE,
    VIDEO_MAX_TOKEN_NUM,
    VIDEO_MIN_TOKEN_NUM,
    ceil_by_factor,
    floor_by_factor,
    round_by_factor,
    smart_nframes,
    smart_resize,
)


# ---------------------------------------------------------------------------
# Constant invariants — locked to the training-distribution values.
# ---------------------------------------------------------------------------
def test_constants_locked_to_training_distribution() -> None:
    assert SPATIAL_MERGE_SIZE == 2
    assert IMAGE_MIN_TOKEN_NUM == 4
    assert IMAGE_MAX_TOKEN_NUM == 16384
    assert VIDEO_MIN_TOKEN_NUM == 128
    assert VIDEO_MAX_TOKEN_NUM == 768
    assert FRAME_FACTOR == 2
    assert MAX_RATIO == 200


# ---------------------------------------------------------------------------
# round / ceil / floor by factor.
# ---------------------------------------------------------------------------
def test_round_by_factor_basic() -> None:
    assert round_by_factor(10, 4) == 8
    assert round_by_factor(11, 4) == 12
    assert round_by_factor(14, 4) == 16


def test_ceil_by_factor_basic() -> None:
    assert ceil_by_factor(10, 4) == 12
    assert ceil_by_factor(12, 4) == 12
    assert ceil_by_factor(13, 4) == 16


def test_floor_by_factor_basic() -> None:
    assert floor_by_factor(10, 4) == 8
    assert floor_by_factor(12, 4) == 12
    assert floor_by_factor(15, 4) == 12


# ---------------------------------------------------------------------------
# smart_resize — token budget + divisibility + aspect ratio preservation.
# ---------------------------------------------------------------------------
@pytest.fixture(name='video_factor')
def _video_factor() -> int:
    # image_patch_size (14) × SPATIAL_MERGE_SIZE (2) — the training pipeline default.
    return 14 * SPATIAL_MERGE_SIZE


def test_smart_resize_divisible_by_factor(video_factor: int) -> None:
    for h, w in [(720, 1280), (1080, 1920), (480, 640), (224, 224)]:
        h_bar, w_bar = smart_resize(
            h,
            w,
            factor=video_factor,
            min_pixels=VIDEO_MIN_TOKEN_NUM * video_factor**2,
            max_pixels=VIDEO_MAX_TOKEN_NUM * video_factor**2,
        )
        assert h_bar % video_factor == 0
        assert w_bar % video_factor == 0
        assert h_bar >= video_factor
        assert w_bar >= video_factor


def test_smart_resize_pixel_budget(video_factor: int) -> None:
    min_pixels = VIDEO_MIN_TOKEN_NUM * video_factor**2
    max_pixels = VIDEO_MAX_TOKEN_NUM * video_factor**2
    # A large video frame gets shrunk to fit under max_pixels.
    h_bar, w_bar = smart_resize(1080, 1920, factor=video_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    assert h_bar * w_bar <= max_pixels
    # A tiny video frame gets upscaled to reach min_pixels.
    h_small, w_small = smart_resize(64, 96, factor=video_factor, min_pixels=min_pixels, max_pixels=max_pixels)
    assert h_small * w_small >= min_pixels


def test_smart_resize_preserves_aspect_ratio_approximately(video_factor: int) -> None:
    max_pixels = VIDEO_MAX_TOKEN_NUM * video_factor**2
    h, w = 720, 1280
    h_bar, w_bar = smart_resize(h, w, factor=video_factor, min_pixels=0, max_pixels=max_pixels)
    original_ratio = w / h
    new_ratio = w_bar / h_bar
    # Ratio should drift by less than 5% under the factor-alignment constraint.
    assert abs(new_ratio - original_ratio) / original_ratio < 0.05


def test_smart_resize_rejects_extreme_ratio(video_factor: int) -> None:
    with pytest.raises(ValueError, match='aspect ratio'):
        smart_resize(10, 5000, factor=video_factor)


def test_smart_resize_defaults_to_image_budget(video_factor: int) -> None:
    # Passing None uses IMAGE_MIN/MAX_TOKEN_NUM * factor**2 as defaults.
    h_bar, w_bar = smart_resize(720, 1280, factor=video_factor)
    assert h_bar * w_bar <= IMAGE_MAX_TOKEN_NUM * video_factor**2
    assert h_bar * w_bar >= IMAGE_MIN_TOKEN_NUM * video_factor**2


# ---------------------------------------------------------------------------
# smart_nframes — sample budget for a given target fps.
# ---------------------------------------------------------------------------
def test_smart_nframes_matches_target_fps() -> None:
    # 10-second clip at 25 fps sampled to 25 fps → all frames (rounded down to FRAME_FACTOR).
    n = smart_nframes(total_frames=250, video_fps=25.0, target_fps=25.0)
    assert n == 250


def test_smart_nframes_downsamples() -> None:
    # 10s @ 25fps down to 5fps → ~50 frames.
    n = smart_nframes(total_frames=250, video_fps=25.0, target_fps=5.0)
    assert n == 50


def test_smart_nframes_respects_max_frames() -> None:
    # A very long clip with target_fps that would exceed FPS_MAX_FRAMES is clamped.
    n = smart_nframes(total_frames=10_000, video_fps=25.0, target_fps=25.0)
    assert n <= 768  # FPS_MAX_FRAMES.
    assert n % FRAME_FACTOR == 0


def test_smart_nframes_respects_min_frames() -> None:
    # A very short clip is bumped up to FPS_MIN_FRAMES if possible.
    n = smart_nframes(total_frames=4, video_fps=25.0, target_fps=1.0)
    assert n >= 4  # FPS_MIN_FRAMES.


def test_smart_nframes_result_divisible_by_frame_factor() -> None:
    for total_frames in [50, 100, 245, 763]:
        for fps in [25.0, 30.0]:
            for target in [1.0, 2.0, 5.0, 25.0]:
                n = smart_nframes(total_frames=total_frames, video_fps=fps, target_fps=target)
                assert n % FRAME_FACTOR == 0, (total_frames, fps, target, n)


def test_smart_nframes_rejects_bad_metadata() -> None:
    with pytest.raises(ValueError, match='video_fps'):
        smart_nframes(total_frames=100, video_fps=0.0, target_fps=25.0)
    with pytest.raises(ValueError, match='total_frames'):
        smart_nframes(total_frames=0, video_fps=25.0, target_fps=25.0)
