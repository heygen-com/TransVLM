# TransVLM Real-Video Shot-Transition Labels

Version: eccv2026-paper (v1)

These are the interval-level shot-transition labels used to evaluate on six real-video evaluation subsets in
*TransVLM: A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions* (ECCV 2026,
[arXiv:2604.27975](https://arxiv.org/abs/2604.27975)).

This package contains **labels and metadata only**. It does not include any videos, images or audio,
and it does not separately bundle the upstream annotation files. Get the media from each source yourself,
under that source's own terms.

A revised annotation of the same evaluation set is in progress and will be released as a separate
version.

## Contents

```
README.md
sources.json                    # upstream entry points, upstream terms (quoted), ID rules, counts
CHECKSUMS.sha256                # shasum -a 256 -c CHECKSUMS.sha256
labels/<subset>/ground_truth.json
labels/<subset>/windows.json
labels/<subset>/videos.json
```

## Annotation provenance

Manual re-annotation covered `rai`, `bbc` and `autoshot_test` only, starting from the original
labels. `clipshots_test`, `movieshots2_test` and `sportsshot_val` keep the original labels, converted
to interval format on the same time base; no manual relabeling was applied to those three.

## Statistics

The three count columns are reported separately:

- **raw intervals**: the number of intervals stored in `ground_truth.json`.
- **normalized intervals**: the number of ground-truth intervals the public scorer matches against at
  tolerance 0.00 s with eps = 0.02 s, after expansion and NMS merging.
- **paper Table 1**: the count printed in the paper.

| subset | split | videos | evaluation windows | videos with no transition | raw intervals | normalized intervals | paper Table 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `rai` | test | 10 | 660 | 0 | 1,036 | 1,018 | 1,036 |
| `bbc` | test | 11 | 3,604 | 0 | 4,943 | 4,851 | 4,943 |
| `autoshot_test` | test | 200 | 872 | 1 | 2,100 | 2,060 | 2,065 |
| `clipshots_test` | test | 500 | 13,335 | 158 | 7,209 | 7,179 | 6,923 |
| `movieshots2_test` | test | 282 | 8,401 | 0 | 14,767 | 14,767 | 14,767 |
| `sportsshot_val` | validation | 240 | 3,863 | 0 | 5,045 | 5,044 | 5,045 |
| **total** | | **1,243** | **30,735** | **159** | **35,100** | **34,919** | **34,779** |

The total in the paper Table 1 column above sums only these six public subsets; the paper's overall
total also includes a synthetic subset not included here.

The dataset statistics reported in the paper contain discrepancies; the labels and counts released in this repository are authoritative.

## File formats

**`ground_truth.json`** maps a video ID to its labels:

```json
{"<video_id>": {"duration": 366.96, "segments": [{"start_time": 12.36, "end_time": 12.84}]}}
```

- Times are in seconds on the video timeline.
- `duration` is the scoring duration: the end of the last evaluation window, stored as float32. It
  is not the container duration.
- `segments: []` means the video was annotated and has no transition.
- Values are exactly the ones used for scoring. A few anomalous intervals are kept as they are:
  one reversed interval in `clipshots_test`, and 34 intervals in `autoshot_test` that end past the
  stored scoring duration. Of those 34, 30 exceed it only by float32 rounding and 4 are meaningful
  overshoots (3 of them by more than 0.02 s). All are kept.

**`windows.json`** maps a video ID to the exact list of evaluation windows used in the paper:
`[{"window": 0, "start_time": 0.0, "end_time": 10.0}, ...]`. Windows are 10 s long with a 9 s
stride, and the short final window is kept as is. Use this list directly instead of generating your
own windows.

**`videos.json`** gives per-video metadata: `num_frames`, `fps`, `container_duration` and `duration`
(the scoring duration).

The paper scores on a 25 fps time grid with eps = 0.02 s.

## Scoring

The public scorer is the `evaluation/` package in this repository. It needs Python 3.10 or newer
and has no third-party dependencies; there is nothing new to install. You can either run its
command-line entry point from the repository root, or put `evaluation/` on `PYTHONPATH` and call
the library.

The CLI reads the two JSON files and writes one CSV per tolerance:

```bash
python evaluation/score.py --ground-truth benchmark/labels/rai/ground_truth.json \
                           --predictions predictions.json --output-dir scores/
```

`score.py` validates its inputs and rejects a reversed interval. `clipshots_test` therefore does not
run through the CLI as-is because of its single reversed interval; the other five subsets do. For all
six subsets, call the library directly. The library copies the inputs, swaps reversed endpoints,
clamps intervals to `[0, duration]`, applies NMS, and never writes back to the ground truth.

Save the snippet below as `score_labels.py` in the repository root. It uses `clipshots_test` to
exercise that reversed interval, and reads `predictions.json` from the same root:

```python
"""score_labels.py -- run from the repository root."""
import json

from transvlm_eval import score  # no reimplementation of normalization

SUBSET = 'clipshots_test'
with open(f'benchmark/labels/{SUBSET}/ground_truth.json') as f:
    gt = json.load(f)
with open('predictions.json') as f:
    pred = json.load(f)

ground_truth = {vid: entry['segments'] for vid, entry in gt.items()}
predictions = {vid: entry['segments'] for vid, entry in pred.items()}
durations = {vid: entry['duration'] for vid, entry in gt.items()}

per_tolerance, ttmean = score(ground_truth, predictions, durations)
print(ttmean['cl_f1'])
```

Run it as:

```bash
PYTHONPATH=evaluation python score_labels.py
```

`predictions.json` maps every video ID in the ground truth to `{"segments": [...]}`; a video with no
predicted transition still needs an entry with an empty list. `score` raises if the three dicts do
not cover exactly the same video IDs.

## Getting the videos and matching IDs

`sources.json` lists the official entry point of each source. For `autoshot_test`, `clipshots_test`,
`movieshots2_test` and `sportsshot_val`, the video ID is the first 16 hex characters of the SHA-256
of the upstream file name, including `.mp4`:

```python
import hashlib
video_id = hashlib.sha256("BAgcx7pSWQo.mp4".encode()).hexdigest()[:16]   # -> "0003454f8958907f"
```

For `rai`, the IDs are `1` to `10` (from `1.mp4` to `10.mp4`). For `bbc`, they are `bbc_01` to
`bbc_11`, one per BBC Planet Earth episode.

## License

The labels and metadata we contribute in this package are released under **CC BY 4.0**. That
license covers only our contributions. It does not replace or relax the terms of the original
annotations or of the videos, which stay under each source's own terms, quoted in `sources.json`.
SportsShot is CC BY-NC 4.0 and gated; neither restriction is lifted by this package.

## Citation

If you use these labels, please cite the TransVLM paper (ECCV 2026, arXiv:2604.27975).

## Questions

If you have questions or find a problem, please open an issue on GitHub.
