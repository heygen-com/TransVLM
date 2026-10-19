# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training-side smart_resize wrapper + ffmpeg helpers.

Focus is the ``max_pixels_override`` ablation flag added for the post-Plan-A
pixel-budget sweep (paper Ours vs 4-point curve).
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from transvlm.data import resize_video_helper
from transvlm.data.resize_video_helper import probe_video_hw, smart_resize_for_qwen3vl


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


class TestProbeVideoHw:
    """``probe_video_hw`` must survive the side-data tail ffprobe appends.

    The legacy invocation was ``-of csv=p=0:s=x`` followed by ``split('x')``; when the
    CSV line carries a trailing side-data field the split raises ``ValueError``.

    ffprobe is faked in the unit tests below, so they pin the contract and the regression,
    not the byte layout of any particular ffprobe build.
    """

    def test_normal_output(self, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            assert 'json' in cmd, f'expected JSON output, got {cmd}'
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({'streams': [{'width': 1920, 'height': 1080}]}))

        monkeypatch.setattr(resize_video_helper.shutil, 'which', lambda _: '/usr/bin/ffprobe')
        monkeypatch.setattr(resize_video_helper.subprocess, 'run', fake_run)
        assert probe_video_hw('in.mp4') == (1080, 1920)

    def test_icc_side_data_is_ignored(self, monkeypatch) -> None:
        """JSON keeps side data in its own list, so width/height are read cleanly."""

        def fake_run(cmd, **kwargs):
            payload = {
                'streams': [
                    {
                        'width': 64,
                        'height': 48,
                        'side_data_list': [{'side_data_type': 'ICC Profile', 'size': 132}],
                    }
                ]
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload))

        monkeypatch.setattr(resize_video_helper.shutil, 'which', lambda _: '/usr/bin/ffprobe')
        monkeypatch.setattr(resize_video_helper.subprocess, 'run', fake_run)
        assert probe_video_hw('icc.mp4') == (48, 64)

    def test_missing_ffprobe(self, monkeypatch) -> None:
        monkeypatch.setattr(resize_video_helper.shutil, 'which', lambda _: None)
        with pytest.raises(FileNotFoundError, match='ffprobe is not on PATH'):
            probe_video_hw('in.mp4')

    def test_ffprobe_failure_propagates(self, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, stderr='moov atom not found')

        monkeypatch.setattr(resize_video_helper.shutil, 'which', lambda _: '/usr/bin/ffprobe')
        monkeypatch.setattr(resize_video_helper.subprocess, 'run', fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            probe_video_hw('broken.mp4')

    def test_unparseable_json_raises_value_error(self, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout='{"streams": []}')

        monkeypatch.setattr(resize_video_helper.shutil, 'which', lambda _: '/usr/bin/ffprobe')
        monkeypatch.setattr(resize_video_helper.subprocess, 'run', fake_run)
        with pytest.raises(ValueError, match='cannot parse video dimensions'):
            probe_video_hw('in.mp4')


@pytest.mark.skipif(
    shutil.which('ffprobe') is None or shutil.which('ffmpeg') is None,
    reason='ffprobe/ffmpeg not installed',
)
def test_probe_video_hw_on_real_media(tmp_path) -> None:
    """Real ffprobe, real container -- the path the fakes above stand in for."""
    src = tmp_path / 'in.mp4'
    subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
         '-i', 'testsrc=size=128x64:rate=25:duration=1', str(src)],
        check=True,
    )  # fmt: skip
    assert probe_video_hw(src) == (64, 128)


class TestSiblingCacheTagging:
    """Downstream flow_computer._sibling_paths uses ``_mp<N>`` tag when override set;
    verified separately in test_flow_computer_sibling_tag.py."""

    def test_stub(self) -> None:
        """Placeholder — full behavior lives in flow_computer's own tests."""
        pass
