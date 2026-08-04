# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Choose the video decoder, and make the fallback chain say what it does.

The vendored ``_vision_process_of`` already prefers torchcodec, but its fallback behaviour is
not what this project wants and cannot be fixed in place -- that file is pinned byte-for-byte
by sha256 and its header says never to patch it. Two gaps:

* **torchcodec missing** -> it silently drops to decord. The only trace is a bare
  ``print(...)`` to stderr, which does not reach the log files that runs are judged from.
  A run could therefore be decoded by a different backend than every number it is compared
  against, with nothing on record.
* **torchcodec present but raising** -> it falls back to **torchvision**, not decord. decord is
  the closer match to torchcodec's output and is installed in both venvs, so torchvision as the
  first fallback is a larger, unannounced change of signal.

Rather than add a "remember to call this first" convention -- ``FORCE_QWENVL_VIDEO_READER`` is
read at *import* time and ``get_video_reader_backend`` is ``lru_cache``d, so calling it late is
a silent no-op -- this module exposes :func:`load_process_vision_info`, which settles the
choice and then imports. Ordering cannot be got wrong because there is nothing to order.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The env var the vendored module reads, at import time, into a module-level global.
ENV_FORCE_READER = 'FORCE_QWENVL_VIDEO_READER'

_VENDORED_MODULE = 'transvlm.data._vision_process_of'

#: Set on the vendored module once the fallback redirect is installed, so a second call is a
#: no-op and the chain cannot nest inside itself.
_REDIRECT_MARKER = '_transvlm_decord_fallback'


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def select_video_backend() -> str:
    """Pick the decoder and set ``FORCE_QWENVL_VIDEO_READER`` so the vendored module agrees.

    Preference is torchcodec > decord > torchvision, which is the vendored default; what this
    adds is a **log line at warning level** whenever the preferred backend is not the one in
    use. An explicit ``FORCE_QWENVL_VIDEO_READER`` always wins and is logged as such.

    Returns the backend name that will be used.
    """
    forced = os.environ.get(ENV_FORCE_READER)
    if forced:
        logger.info('video reader forced to %r via %s', forced, ENV_FORCE_READER)
        return forced

    if _available('torchcodec'):
        logger.info('video reader: torchcodec')
        return 'torchcodec'

    if _available('decord'):
        logger.warning(
            'torchcodec is NOT installed -- falling back to decord for video decoding. decoded '
            'frames may differ from runs made with torchcodec, so numbers from this run are not '
            'directly comparable to those.'
        )
        os.environ[ENV_FORCE_READER] = 'decord'
        return 'decord'

    logger.warning(
        'neither torchcodec nor decord is installed -- falling back to torchvision, which is the '
        'least faithful of the three. Install torchcodec before producing any reported number.'
    )
    return 'torchvision'


def _install_decord_fallback(vpo: Any) -> None:
    """Point the vendored module's hardcoded torchvision fallback at decord first.

    ``fetch_video`` catches a reader failure and calls ``VIDEO_READER_BACKENDS['torchvision']``
    by name, so replacing that entry is the only way to change the fallback without editing the
    pinned file. The replacement still ends at the real torchvision reader: a decord failure
    used to be survivable via torchvision, and turning it into a hard shard failure would lose
    work.

    Note the vendored module logs "use torchvision as default" before calling this; the wrapper
    logs what actually ran, so the pair reads slightly redundantly but never wrongly.
    """
    if not _available('decord'):
        return
    if getattr(vpo, _REDIRECT_MARKER, False):
        return

    real_torchvision: Callable[..., Any] = vpo.VIDEO_READER_BACKENDS['torchvision']
    decord_reader: Callable[..., Any] = vpo.VIDEO_READER_BACKENDS['decord']

    def _decord_then_torchvision(ele: Any) -> Any:
        # Logged at warning level on the SUCCESS path too. The vendored code has already printed
        # "use torchvision as default" to stderr by the time this runs -- which is both
        # unreachable from the log files runs are judged from, and wrong about what ran. Staying
        # silent here would leave the failure path this module exists to fix with no record.
        logger.warning('torchcodec reader raised; decoding this item with decord instead')
        try:
            return decord_reader(ele)
        except Exception:  # noqa: BLE001 -- last-resort chain; the real reader is tried next
            logger.warning('decord fallback also failed, using torchvision', exc_info=True)
            return real_torchvision(ele)

    vpo.VIDEO_READER_BACKENDS['torchvision'] = _decord_then_torchvision
    setattr(vpo, _REDIRECT_MARKER, True)
    logger.debug('reader-failure fallback redirected to decord')


#: The decoder :func:`load_process_vision_info` settled on, for a caller that wants to record it.
#: ``None`` until it has run.
_SELECTED_BACKEND: str | None = None


def selected_video_backend() -> str | None:
    """Which decoder this process is using, so a run record can name it.

    A decord-fallback run and a torchcodec run produce different frames; a jsonl that does not
    say which one ran is not self-describing, which defeats the point of this module.
    """
    return _SELECTED_BACKEND


def load_process_vision_info() -> Callable[..., Any]:
    """Return ``process_vision_info`` with the decoder choice already settled.

    Use this instead of importing ``_vision_process_of`` directly. Importing it first would
    freeze the backend before :func:`select_video_backend` could influence it.
    """
    global _SELECTED_BACKEND  # noqa: PLW0603 -- the decoder choice is process-global by design

    already_imported = _VENDORED_MODULE in sys.modules
    backend = select_video_backend()
    _SELECTED_BACKEND = backend

    from transvlm.data import _vision_process_of as vpo  # noqa: PLC0415

    if already_imported and os.environ.get(ENV_FORCE_READER) != vpo.FORCE_QWENVL_VIDEO_READER:
        # The env var is captured into a module global at import time, so a later change is
        # invisible to it. Loud, because this is the same shape of bug as a timestamp patch that
        # looks installed and is not.
        logger.warning(
            '%s was already imported with FORCE_QWENVL_VIDEO_READER=%r; the requested %r cannot '
            'take effect in this process.',
            _VENDORED_MODULE,
            vpo.FORCE_QWENVL_VIDEO_READER,
            os.environ.get(ENV_FORCE_READER),
        )

    if backend == 'torchcodec':
        _install_decord_fallback(vpo)

    return vpo.process_vision_info
