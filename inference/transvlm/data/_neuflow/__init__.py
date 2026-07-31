"""Vendored NeuFlow v2 network (Zhang et al., 2024).

Upstream: https://github.com/neufieldrobotics/NeuFlow_v2
Model weights: HuggingFace hub ``Study-is-happy/neuflow-v2`` (Apache-2.0).

Vendored here so the inference pipeline is self-contained. Do NOT edit the
individual submodule files (backbone_v7 / corr / matching / neuflow / …) —
they are the upstream implementation verbatim, with only the top-of-file
``from NeuFlow import X`` lines rewritten to relative ``from . import X``.

Only the ``NeuFlow`` class is meant to be imported directly:

    from transvlm.data._neuflow.neuflow import NeuFlow

See ``LICENSE`` in this directory for the upstream license terms.
"""
from __future__ import annotations
