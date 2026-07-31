"""Pin the exact-rate flow write that fixes the RGB/flow `sample_fps` assertion.

Two whole videos failed the pinned `_vision_process_of.py` assertion
`rgb_video_sample_fps == of_video_sample_fps`. The cause was not the resample and not the frame
count (both siblings had 250 frames) -- it was that `torchvision.io.write_video` rounds a *float*
rate to the nearest integer, so a flow sibling computed from a 17525/702 fps RGB source declared
25/1. `smart_nframes` then sampled a different frame count from each sibling.

Passing a `Fraction` through survives verbatim, which is what these tests hold in place. The
second test is the important one: it is what makes the fix provably a no-op for every video whose
rate is already integral, i.e. the ones that already worked.
"""

from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction

import pytest
import torch

from transvlm.data.flow_writer import write_flow_mp4
from transvlm.data.resize_video_helper import probe_video_fps_exact


def _frames(n: int = 10) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (n, 3, 64, 64), dtype=torch.uint8)


def test_fractional_rate_survives_the_container(tmp_path):
    """A float would land as 25/1 here -- that rounding is the whole bug."""
    out = write_flow_mp4(_frames(), tmp_path / 'frac.mp4', fps=Fraction(17525, 702))
    assert probe_video_fps_exact(out) == Fraction(17525, 702)


def test_float_rate_is_rounded_which_is_why_a_fraction_is_required(tmp_path):
    """Documents the failure mode so nobody 'simplifies' the Fraction back to a float."""
    out = write_flow_mp4(_frames(), tmp_path / 'flt.mp4', fps=24.964387)
    assert probe_video_fps_exact(out) == Fraction(25, 1), 'torchvision no longer rounds -- revisit the fix'


def test_integral_rate_is_byte_identical_either_way(tmp_path):
    """The no-op guarantee for videos that already worked.

    If Fraction(25, 1) and 25.0 produced different bytes, this change would silently perturb every
    previously-successful video instead of only the two that failed.
    """
    a = write_flow_mp4(_frames(), tmp_path / 'a.mp4', fps=25.0)
    b = write_flow_mp4(_frames(), tmp_path / 'b.mp4', fps=Fraction(25, 1))
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_probe_rejects_an_unusable_rate(tmp_path):
    """0/0 must raise so the caller's explicit fallback runs, rather than yielding fps=0."""
    bad = tmp_path / 'bad.mp4'
    bad.write_bytes(b'not a video')
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        probe_video_fps_exact(bad)
