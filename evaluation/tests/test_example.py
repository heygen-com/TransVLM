"""Re-run the shipped example and compare against its committed output.

This is the end-to-end regression for the whole scoring path: loader, tolerance expansion, NMS
merge, matcher, aggregation, CSV writer. Unit tests cover each stage, but only this one catches a
change in how they compose.

It doubles as a check that the example in `examples/README.md` still works, so the documentation
cannot rot away from the code.

The expected numbers are hand-derivable from the three synthetic videos, and the docstrings below
say what each one is, so a diff here can be reasoned about instead of just re-blessed.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parents[1]
_EXAMPLES = _EVAL_DIR / 'examples'
_EXPECTED = _EXAMPLES / 'expected' / 'transvlm'


def _rows(path: Path) -> dict[str, str]:
    with open(path, newline='') as handle:
        return next(iter(csv.DictReader(handle)))


@pytest.fixture(scope='module')
def produced(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp('scored')
    result = subprocess.run(
        [
            sys.executable, str(_EVAL_DIR / 'score.py'),
            '--ground-truth', str(_EXAMPLES / 'ground_truth.json'),
            '--predictions', str(_EXAMPLES / 'predictions.json'),
            '--output-dir', str(out),
            '--data-source', 'example',
        ],
        capture_output=True, text=True, cwd=_EVAL_DIR,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    return out / 'transvlm'


def test_every_tolerance_csv_matches_the_committed_one(produced: Path):
    expected_files = sorted(p.relative_to(_EXPECTED) for p in _EXPECTED.rglob('*.csv'))
    assert expected_files, 'no committed CSVs to compare against'
    for rel in expected_files:
        assert _rows(produced / rel) == _rows(_EXPECTED / rel), f'{rel} changed'


def test_summary_json_matches(produced: Path):
    assert json.loads((produced / 'summary.json').read_text()) == json.loads((_EXPECTED / 'summary.json').read_text())


def test_the_counts_are_the_ones_the_example_was_designed_to_produce():
    """Anchors the expected files to hand-reasoning rather than to whatever ran first.

    At tt0.00 across the three videos: clip_a contributes one exact match plus one overlapping
    match (2 TP), clip_b's prediction is nowhere near its ground truth (1 FP, 1 FN), and clip_c
    has a prediction with no ground truth at all (1 FP). So 2 TP / 2 FP / 1 FN.
    """
    row = _rows(_EXPECTED / 'tt0.00' / 'example.csv')
    assert (float(row['cl_tp']), float(row['cl_fp']), float(row['cl_fn'])) == (2.0, 2.0, 1.0)


def test_a_video_absent_from_the_predictions_is_refused():
    """The mistake the example README warns about, checked at the CLI boundary."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        partial = Path(tmp) / 'partial.json'
        full = json.loads((_EXAMPLES / 'predictions.json').read_text())
        full.pop('clip_c')
        partial.write_text(json.dumps(full))
        result = subprocess.run(
            [
                sys.executable, str(_EVAL_DIR / 'score.py'),
                '--ground-truth', str(_EXAMPLES / 'ground_truth.json'),
                '--predictions', str(partial),
                '--output-dir', tmp,
            ],
            capture_output=True, text=True, cwd=_EVAL_DIR,
        )  # fmt: skip
        assert result.returncode != 0
        assert 'same set' in result.stderr
