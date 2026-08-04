# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the decoder-choice behaviour: torchcodec by default, decord as the announced fallback.

The vendored ``_vision_process_of`` cannot be edited (sha256-pinned, "DO NOT EDIT"), so the
behaviour lives in a wrapper. These tests exercise the wrapper against fakes rather than
against real decoders: both venvs happen to have torchcodec *and* decord installed, so the
interesting cases (torchcodec absent, torchcodec raising) can only be reached by faking.
"""

from __future__ import annotations

import logging

import pytest

from transvlm.data import video_backend as vb


@pytest.fixture(autouse=True)
def _no_forced_reader(monkeypatch):
    """Start every test from an unforced state; the env var leaks between tests otherwise."""
    monkeypatch.delenv(vb.ENV_FORCE_READER, raising=False)


def test_torchcodec_is_preferred_when_present(monkeypatch):
    monkeypatch.setattr(vb, '_available', lambda name: True)
    assert vb.select_video_backend() == 'torchcodec'
    # Preferring the default must not pin it via the env var; that would defeat an explicit
    # override in a later process started from the same environment.
    assert vb.ENV_FORCE_READER not in __import__('os').environ


def test_missing_torchcodec_warns_and_selects_decord(monkeypatch, caplog):
    monkeypatch.setattr(vb, '_available', lambda name: name != 'torchcodec')
    with caplog.at_level(logging.WARNING, logger=vb.logger.name):
        assert vb.select_video_backend() == 'decord'
    assert 'torchcodec is NOT installed' in caplog.text
    assert 'decord' in caplog.text
    # The env var is how the vendored module learns about it -- setting it is the actual fix,
    # the warning is only the paper trail.
    assert __import__('os').environ[vb.ENV_FORCE_READER] == 'decord'


def test_missing_both_warns_harder_and_falls_to_torchvision(monkeypatch, caplog):
    monkeypatch.setattr(vb, '_available', lambda name: name not in ('torchcodec', 'decord'))
    with caplog.at_level(logging.WARNING, logger=vb.logger.name):
        assert vb.select_video_backend() == 'torchvision'
    assert 'neither torchcodec nor decord' in caplog.text


def test_an_explicit_force_wins_and_is_logged(monkeypatch, caplog):
    monkeypatch.setenv(vb.ENV_FORCE_READER, 'torchvision')
    monkeypatch.setattr(vb, '_available', lambda name: True)
    with caplog.at_level(logging.INFO, logger=vb.logger.name):
        assert vb.select_video_backend() == 'torchvision'
    assert vb.ENV_FORCE_READER in caplog.text


class _FakeVendored:
    """Stand-in for the vendored module: just the one dict the redirect touches."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.VIDEO_READER_BACKENDS = {
            'decord': self._decord,
            'torchvision': self._torchvision,
            'torchcodec': self._torchcodec,
        }

    def _decord(self, _ele):
        self.calls.append('decord')
        return 'decord-frames'

    def _torchvision(self, _ele):
        self.calls.append('torchvision')
        return 'torchvision-frames'

    def _torchcodec(self, _ele):
        self.calls.append('torchcodec')
        raise RuntimeError('torchcodec exploded')


def test_reader_failure_falls_back_to_decord_not_torchvision(monkeypatch):
    """The gap being closed: upstream sends a torchcodec failure straight to torchvision."""
    monkeypatch.setattr(vb, '_available', lambda name: True)
    fake = _FakeVendored()
    vb._install_decord_fallback(fake)

    # `fetch_video` reaches the fallback by looking up the name 'torchvision', so that entry is
    # what has to change -- this call is what the vendored code does.
    assert fake.VIDEO_READER_BACKENDS['torchvision']({}) == 'decord-frames'
    assert fake.calls == ['decord']


def test_the_real_torchvision_stays_reachable_when_decord_also_fails(monkeypatch, caplog):
    """A decord failure used to be survivable via torchvision; it must stay that way."""
    monkeypatch.setattr(vb, '_available', lambda name: True)
    fake = _FakeVendored()
    fake.VIDEO_READER_BACKENDS['decord'] = lambda _ele: (_ for _ in ()).throw(RuntimeError('decord too'))
    vb._install_decord_fallback(fake)

    with caplog.at_level(logging.WARNING, logger=vb.logger.name):
        assert fake.VIDEO_READER_BACKENDS['torchvision']({}) == 'torchvision-frames'
    assert 'decord fallback also failed' in caplog.text


def test_installing_the_fallback_twice_does_not_nest(monkeypatch):
    """Without the marker the second install would wrap the first, doubling the chain."""
    monkeypatch.setattr(vb, '_available', lambda name: True)
    fake = _FakeVendored()
    vb._install_decord_fallback(fake)
    first = fake.VIDEO_READER_BACKENDS['torchvision']
    vb._install_decord_fallback(fake)
    assert fake.VIDEO_READER_BACKENDS['torchvision'] is first


def test_no_fallback_installed_without_decord(monkeypatch):
    """Nothing to redirect to; leave the vendored dict exactly as it was."""
    monkeypatch.setattr(vb, '_available', lambda name: name != 'decord')
    fake = _FakeVendored()
    original = fake.VIDEO_READER_BACKENDS['torchvision']
    vb._install_decord_fallback(fake)
    assert fake.VIDEO_READER_BACKENDS['torchvision'] is original


def test_loader_returns_the_real_process_vision_info():
    """The whole point of the loader: callers get the function, ordering handled for them."""
    fn = vb.load_process_vision_info()
    assert fn.__name__ == 'process_vision_info'
