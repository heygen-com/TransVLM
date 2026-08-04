# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Optical-flow generation policy.

A plain dataclass rather than a pydantic model. The original was a ``BaseModel``, but nothing
here uses validation, serialisation or schema generation -- only attribute access -- so pydantic
was the only non-torch dependency in an otherwise torch-only inference path.

What the pydantic version *did* give for free was rejecting a bad ``mode`` or ``flow_codec``.
That matters: ``OnlineFlowComputer`` dispatches with ``if mode == 'sibling' ... elif mode ==
'online_with_cache'``, so a typo like ``'onlin'`` would silently fall through and behave like
``'online'`` -- recomputing flow every call and never caching, with no error. ``__post_init__``
keeps that check without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Recompute every call and do not persist / require a pre-existing sibling / compute once and
#: cache next to the input.
FLOW_MODES = ('online', 'sibling', 'online_with_cache')

#: See ``flow_writer.FLOW_CODECS`` for the measured round-trip error of each.
FLOW_CODEC_NAMES = ('libx264', 'lossless')


@dataclass
class FlowConfig:
    """How optical flow is produced and cached.

    Args:
        mode: ``online`` recomputes on every call and persists nothing. ``sibling`` requires
            ``<basename>_flow.mp4`` to already exist and errors if it does not. ``online_with_cache``
            computes once, writes the sibling, and reuses it afterwards.
        sibling_suffix: filename suffix for the cached flow video.
        flow_codec: encode fidelity. ``libx264`` is the setting the checkpoint was trained
            against and is the default. ``lossless`` round-trips bit-exactly at roughly 25x the
            bytes; its siblings carry an ``_ll`` tag so the two fidelities never overwrite each
            other. Measured on this checkpoint, lossless is not an accuracy win -- see the
            evaluation README.
        neuflow_weights: HuggingFace Hub id for NeuFlow v2, fetched on first use.
        downscale: reserved; only ``1`` is supported, because the training pipeline ran flow at
            full resolution and any other value shifts the motion-magnitude distribution.
        mini_batch_size: NeuFlow's internal batch. 32 suits ~97 GiB cards on inputs inside the
            default pixel budget; drop to 8 on smaller cards or at higher resolution. An explicit
            constructor argument to ``OnlineFlowComputer`` still wins over this value.
    """

    mode: str = 'online_with_cache'
    sibling_suffix: str = '_flow.mp4'
    flow_codec: str = 'libx264'
    neuflow_weights: str = 'Study-is-happy/neuflow-v2'
    downscale: int = 1
    mini_batch_size: int = 32

    def __post_init__(self) -> None:
        if self.mode not in FLOW_MODES:
            raise ValueError(f'mode={self.mode!r} is not one of {FLOW_MODES}')
        if self.flow_codec not in FLOW_CODEC_NAMES:
            raise ValueError(f'flow_codec={self.flow_codec!r} is not one of {FLOW_CODEC_NAMES}')
        if self.mini_batch_size < 1:
            raise ValueError(f'mini_batch_size={self.mini_batch_size} must be >= 1')
