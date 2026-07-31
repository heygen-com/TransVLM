"""Tests for the training-side smart_resize wrapper + ffmpeg helpers.

Focus is the ``max_pixels_override`` ablation flag added for the post-Plan-A
pixel-budget sweep (paper Ours vs 4-point curve).
"""

from __future__ import annotations

import pytest

from transvlm.data.resize_video_helper import smart_resize_for_qwen3vl


class TestMaxPixelsOverride:
    """When ``max_pixels_override`` is set, output shrinks accordingly."""

    def test_default_matches_training_formula_autoshot(self) -> None:
        """No override → 1080x1920 → 640x1152 (post-Plan-A observed target)."""
        h, w = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=None)
        assert (h, w) == (640, 1152)

    def test_override_786432_matches_default(self) -> None:
        """Explicit 786432 = training-formula max_pixels for autoshot on this seq_len."""
        h_none, w_none = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=None)
        h_786, w_786 = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=786432)
        assert (h_none, w_none) == (h_786, w_786) == (640, 1152)

    @pytest.mark.parametrize(
        'override, expected_hw',
        [
            (524288, (512, 960)),
            (393216, (448, 832)),
            (262144, (384, 672)),
        ],
    )
    def test_override_shrinks_autoshot(self, override: int, expected_hw: tuple[int, int]) -> None:
        h, w = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=override)
        assert (h, w) == expected_hw
        assert h * w <= override, f'{h}*{w}={h*w} exceeds override {override}'

    def test_monotonic_shrink(self) -> None:
        """Larger max_pixels → equal or larger output dim."""
        dims = [smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=mp) for mp in [262144, 393216, 524288, 786432]]
        for i in range(1, len(dims)):
            assert dims[i][0] * dims[i][1] >= dims[i - 1][0] * dims[i - 1][1]

    def test_all_multiples_of_32(self) -> None:
        """Vision patch=16, spatial_merge=2 → dim must divide 32."""
        for mp in [None, 786432, 524288, 393216, 262144]:
            h, w = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=mp)
            assert h % 32 == 0 and w % 32 == 0

    def test_small_input_floored_by_min_pixels(self) -> None:
        """Tiny inputs (like clipshots 128x128) upsample to min_pixels floor;
        max_pixels_override below the floor is a no-op — the sweep degenerates."""
        outputs = {
            mp: smart_resize_for_qwen3vl(128, 128, max_pixels_override=mp)
            for mp in [None, 786432, 524288, 393216, 262144]
        }
        # All configs should produce identical output when input is smaller than min_pixels.
        distinct = set(outputs.values())
        assert len(distinct) == 1, f'Expected 1 distinct output, got {outputs}'
        # And that single output is 384x384 (known constant for min_pixels floor).
        assert outputs[None] == (384, 384)


class TestOverrideBelowFloorClamps:
    """Regression for the 2026-07-29 clamp fix.

    An override below ``min_pixels`` used to log "the resize target will clamp at the
    floor" and then fall through to ``smart_resize``, which raised
    ``ValueError('max_pixels must be >= min_pixels')`` — so the documented behaviour and
    the actual behaviour disagreed. It now really clamps.
    """

    def test_override_below_floor_does_not_raise(self) -> None:
        # min_pixels at patch=16 is 128 * 32**2 = 131072; ask for well under it.
        h, w = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=100_000)
        assert h % 32 == 0 and w % 32 == 0

    def test_override_below_floor_lands_on_the_floor(self) -> None:
        floor = 128 * 32 * 32
        at_floor = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=floor)
        below = smart_resize_for_qwen3vl(1080, 1920, max_pixels_override=floor // 4)
        assert below == at_floor, f'clamped result {below} should equal the floor result {at_floor}'


class TestSiblingCacheTagging:
    """Downstream flow_computer._sibling_paths uses ``_mp<N>`` tag when override set;
    verified separately in test_flow_computer_sibling_tag.py."""

    def test_stub(self) -> None:
        """Placeholder — full behavior lives in flow_computer's own tests."""
        pass
