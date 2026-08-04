# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Frozen dataclasses passed between pipeline stages.

Kept intentionally minimal — no methods, just data. Downstream code that needs
to compute things over these (e.g. IoU, string formatting) should live in the
appropriate ``source/<purpose>/`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoPipelineInput:
    """Everything ``InferencePipeline.run`` needs to process one video.

    * ``video_path``: the source mp4 (may be at any resolution).
    * ``resized_path`` / ``flow_path``: both ``None`` (default) → the
      pipeline calls ``OnlineFlowComputer.compute_and_cache`` which
      materializes both siblings at the training-side smart-resize target
      dim. Either can be pre-supplied to bypass caching; when doing so,
      **both** should be pre-supplied (and at the same dim) — the pipeline
      no longer runs any python-side resize on their outputs.
    * ``output_dir`` may be ``None``; when set, per-video artifacts
      (segments JSON, timing log) go there.
    """

    video_path: Path
    resized_path: Path | None = None
    flow_path: Path | None = None
    output_dir: Path | None = None


@dataclass(frozen=True)
class WindowSpec:
    """One sliding-window slice of a video."""

    index: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class SegmentPrediction:
    """One detected shot transition segment.

    ``confidence`` is optional because the generative backbone does not
    natively emit a per-segment score; it is populated only when a scoring
    strategy (e.g. log-prob aggregation) is enabled.
    """

    start_time: float
    end_time: float
    confidence: float | None = None
    source_window: int | None = None
