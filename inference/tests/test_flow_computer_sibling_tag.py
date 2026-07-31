"""Sibling filename tagging when max_pixels_override is set.

The heavy NeuFlow load is skipped by not calling ``__init__``; we exercise
just ``_sibling_paths`` which is a pure string-manipulation method.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from transvlm.data.flow_computer import OnlineFlowComputer


def _make_stub(max_pixels_override: int | None, flow_codec: str = 'libx264') -> OnlineFlowComputer:
    """Bypass NeuFlow load — attach only the fields ``_sibling_paths`` reads."""
    stub = OnlineFlowComputer.__new__(OnlineFlowComputer)
    stub._cfg = SimpleNamespace(sibling_suffix='_flow.mp4', flow_codec=flow_codec)  # noqa: SLF001
    stub._max_pixels_override = max_pixels_override  # noqa: SLF001
    return stub


class TestSiblingPaths:
    def test_no_override_untagged(self) -> None:
        stub = _make_stub(None)
        r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert r.name == 'video_resized.mp4'
        assert f.name == 'video_flow.mp4'

    def test_override_gets_mp_tag(self) -> None:
        stub = _make_stub(393216)
        r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert r.name == 'video_resized_mp393216.mp4'
        assert f.name == 'video_flow_mp393216.mp4'

    def test_different_overrides_dont_collide(self) -> None:
        for mp in [524288, 393216, 262144]:
            stub = _make_stub(mp)
            r, _f = stub._sibling_paths(Path('/tmp/v.mp4'))  # noqa: SLF001
            assert f'_mp{mp}' in r.name
        # Default (None) is DISTINCT from any override.
        default = _make_stub(None)._sibling_paths(Path('/tmp/v.mp4'))[0].name  # noqa: SLF001
        assert '_mp' not in default

    def test_custom_flow_suffix_still_tagged(self) -> None:
        """User overriding FlowConfig.sibling_suffix (e.g. _flow_neuflow.mp4)
        still gets the _mp<N> tag inserted before the extension."""
        stub = OnlineFlowComputer.__new__(OnlineFlowComputer)
        stub._cfg = SimpleNamespace(sibling_suffix='_flow_custom.mp4', flow_codec='libx264')  # noqa: SLF001
        stub._max_pixels_override = 393216  # noqa: SLF001
        _r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert f.name == 'video_flow_custom_mp393216.mp4'


class TestFlowCodecTag:
    """The `_ll` tag is what stops the two encode fidelities clobbering one another."""

    def test_lossless_tags_only_the_flow_file(self) -> None:
        stub = _make_stub(None, flow_codec='lossless')
        r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert f.name == 'video_flow_ll.mp4'
        # `_resized.mp4` is ffmpeg output and identical under either codec, so tagging it would
        # force a pointless re-encode and stop the two arms sharing it.
        assert r.name == 'video_resized.mp4'

    def test_libx264_is_untagged_so_existing_caches_still_resolve(self) -> None:
        stub = _make_stub(None, flow_codec='libx264')
        _r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert f.name == 'video_flow.mp4'

    def test_lossless_composes_with_the_pixel_budget_tag(self) -> None:
        stub = _make_stub(393216, flow_codec='lossless')
        _r, f = stub._sibling_paths(Path('/tmp/video.mp4'))  # noqa: SLF001
        assert f.name == 'video_flow_mp393216_ll.mp4'
