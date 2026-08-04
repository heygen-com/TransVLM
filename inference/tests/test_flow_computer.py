# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Pin the flow-timing split.

The split exists so this project's optical-flow cost can be compared against the paper's
published 0.22, which covers **only** the NeuFlow forward -- decode, visualisation and the mp4
encode are excluded from that figure. Getting the arithmetic or the key names wrong here would
silently produce a wrong comparison rather than an error, which is why they are pinned.
"""

from __future__ import annotations

import inspect

import pytest

from transvlm.data.flow_computer import FlowTimings, OnlineFlowComputer


def test_total_is_the_sum_of_the_parts():
    t = FlowTimings(decode_sec=1.0, neuflow_sec=2.0, flow_to_image_sec=0.5, encode_sec=0.25)
    assert t.total_sec == pytest.approx(3.75)


def test_defaults_to_zero():
    """A supplied --flow means nothing is computed; the record must not claim otherwise."""
    assert FlowTimings().total_sec == 0.0


def test_as_dict_keys_are_stable():
    """These key names land in every jsonl record; renaming one orphans every past run."""
    assert set(FlowTimings().as_dict()) == {
        'flow_decode_sec',
        'flow_neuflow_sec',
        'flow_to_image_sec',
        'flow_encode_sec',
    }


def test_as_dict_reports_every_part_it_was_given():
    t = FlowTimings(decode_sec=1.5, neuflow_sec=2.25, flow_to_image_sec=0.125, encode_sec=0.0625)
    assert t.as_dict() == {
        'flow_decode_sec': 1.5,
        'flow_neuflow_sec': 2.25,
        'flow_to_image_sec': 0.125,
        'flow_encode_sec': 0.0625,
    }


def test_compute_still_works_without_a_timings_sink():
    """Two report scripts call `compute(frames)` positionally; the sink must stay optional."""
    sig = inspect.signature(OnlineFlowComputer.compute)
    assert sig.parameters['timings'].default is None
    assert sig.parameters['timings'].kind is inspect.Parameter.KEYWORD_ONLY


def test_compute_actually_writes_into_the_sink_it_is_handed():
    """The behavioural version of "the wiring is there".

    A source-text assertion would pass while `compute_flow_only` quietly stopped forwarding
    `timings=`, which would zero every flow_neuflow_sec / flow_to_image_sec in every jsonl and
    make the flow-timing tool's paper-comparable RTF read ~0 -- silently destroying the very split these exist
    for. So drive the real method with a stub model instead.
    """
    import torch

    computer = OnlineFlowComputer.__new__(OnlineFlowComputer)  # no NeuFlow download
    computer._mini_batch_size = 2
    computer._device = torch.device('cpu')
    computer._device_str = 'cpu'
    computer._viz_on_gpu = False
    computer._bhwd = (2, 32, 32)
    # NeuFlow returns a list of flow pyramids; only the last is used.
    computer._model = lambda prev, cur: [torch.zeros(prev.shape[0], 2, prev.shape[2], prev.shape[3])]

    frames = torch.zeros(4, 3, 32, 32, dtype=torch.uint8)
    sink = FlowTimings()
    out = computer.compute(frames, timings=sink)

    assert out.shape == (4, 3, 32, 32)
    assert sink.neuflow_sec > 0.0, 'compute must record the forward into the sink it was given'
    assert sink.flow_to_image_sec > 0.0, 'compute must record the visualisation too'
    # compute() never sees the file, so these stay untouched -- that is compute_flow_only's job.
    assert sink.decode_sec == 0.0
    assert sink.encode_sec == 0.0


def test_compute_flow_only_forwards_the_sink_and_returns_it():
    """Pins the two halves a source-text check cannot: the forward, and the return shape."""
    sig = inspect.signature(OnlineFlowComputer.compute_flow_only)
    assert sig.return_annotation == 'tuple[Path, FlowTimings]'
    src = inspect.getsource(OnlineFlowComputer.compute_flow_only)
    assert 'timings=timings' in src, 'the sink must reach compute(), or every part reads 0.0'
