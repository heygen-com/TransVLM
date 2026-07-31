"""Cover the atomic-rename resize and the `overwrite=False` contract it nearly broke.

`resize_video_ffmpeg` now encodes to a temp sibling and `os.replace`s it into position, so a
concurrent reader cannot decode a half-written `_resized.mp4` and compute optical flow on garbage
frames. Moving the encode to a temp name also moved ffmpeg's `-n` guard onto that temp name, which
is always fresh -- so `overwrite=False` had to become an explicit check or it would have silently
started overwriting.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from transvlm.data.resize_video_helper import resize_video_ffmpeg


@pytest.fixture(scope='module')
def tiny_source(tmp_path_factory) -> Path:
    src = tmp_path_factory.mktemp('src') / 'in.mp4'
    subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
         '-i', 'testsrc=size=128x64:rate=25:duration=1', str(src)],
        check=True,
    )  # fmt: skip
    return src


def test_resize_produces_the_target_dims(tiny_source, tmp_path):
    dst = tmp_path / 'out.mp4'
    resize_video_ffmpeg(tiny_source, dst, target_h=32, target_w=64)
    assert dst.exists() and dst.stat().st_size > 0


def test_no_temp_file_is_left_behind(tiny_source, tmp_path):
    """A stray `.tmpNNN.mp4` beside the real file is the confusion this change prevents."""
    dst = tmp_path / 'out.mp4'
    resize_video_ffmpeg(tiny_source, dst, target_h=32, target_w=64)
    assert not list(tmp_path.glob('*.tmp*')), 'temp artefact survived a successful encode'


def test_overwrite_false_leaves_the_existing_file_untouched(tiny_source, tmp_path):
    """The contract that the temp-file change would otherwise have inverted."""
    dst = tmp_path / 'out.mp4'
    dst.write_bytes(b'sentinel')
    resize_video_ffmpeg(tiny_source, dst, target_h=32, target_w=64, overwrite=False)
    assert dst.read_bytes() == b'sentinel'


def test_overwrite_true_replaces_it(tiny_source, tmp_path):
    dst = tmp_path / 'out.mp4'
    dst.write_bytes(b'sentinel')
    resize_video_ffmpeg(tiny_source, dst, target_h=32, target_w=64, overwrite=True)
    assert dst.read_bytes() != b'sentinel'
