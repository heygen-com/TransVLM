# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Runtime patches applied when running the vLLM backend.

Two patches, vendored from the reference implementation:

1. ``sitecustomize.py`` — auto-loaded by Python at interpreter startup **when
   this directory is on PYTHONPATH**. Overrides vLLM's Qwen3-VL dummy video
   builder to produce 6-channel torch tensors (default 3 channels would break
   profiling of our 6-channel patch embedding). Only activates when
   ``VLLM_QWEN3VL_6CH_DUMMY=1``.
2. ``qwen3vl_processor_ours.py`` — replaces
   ``transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor`` with
   a variant that handles the bare ``<|video_pad|>`` token vLLM passes at
   inference time.

Callers must put this directory on ``sys.path`` **and** on ``PYTHONPATH`` before
the first ``import vllm``, then ``import sitecustomize`` explicitly (the
interpreter has already booted by then). ``PYTHONPATH`` is the load-bearing one:
vLLM spawns its EngineCore workers, and only the environment survives a spawn.
See ``infer_clips.py``.

Both files are byte-identical to the reference implementation as of
2026-07-29 (sha256 ``444e9224…`` / ``d1385233…``, verified against
Do not edit them here -- re-vendor from upstream instead, so the
two copies cannot drift apart silently.
"""
from __future__ import annotations
