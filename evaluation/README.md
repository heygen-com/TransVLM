# TransVLM evaluation

The scoring code behind the numbers in the paper. Segment-level, swept over 11 time tolerances.

It is dataset-agnostic: give it ground truth, predictions and per-video durations, and it reports
the same metric the paper does. Nothing here is specific to shot transitions beyond the fact that
predictions are time spans.

## Install

Pure Python, no GPU, no torch.

```bash
cd evaluation
uv venv && source .venv/bin/activate
uv sync
```

## Run

```bash
python score.py \
  --ground-truth examples/ground_truth.json \
  --predictions examples/predictions.json \
  --data-source example \
  --output-dir scores/
```

`--data-source` is the CSV filename stem (default `custom`) and `--method-name` is the row label
inside them, so several datasets and several methods can share one output tree. The value above is
the one `examples/expected/` was generated with, which is what makes `diff -r` against it work.

Both files are JSON keyed by video id:

```json
{
  "clip_a": {"duration": 12.0, "segments": [{"start_time": 2.0, "end_time": 2.04}]},
  "clip_b": {"duration": 8.0,  "segments": []}
}
```

`duration` is read from the ground-truth file and ignored in the predictions file.

Output is one CSV per tolerance (`tt0.00` through `tt0.50`) plus `ttmean`, and a `summary.json`.

### Every video needs an entry on both sides

A video with nothing to report still needs `"segments": []`. Omit it from the predictions and its
ground truth stops being counted, so recall comes out higher than it is -- the direction nobody
double-checks. `score.py` refuses a mismatched set of ids rather than scoring the intersection.

## The metric

For each of 11 tolerances from 0.00 s to 0.50 s in 0.05 s steps:

1. Expand every ground-truth and predicted span by the tolerance on each side, clamped to
   `[0, duration]`. A zero-width span at tolerance 0 is first given a half-frame body, so
   `tt0.00` means "within half a frame", not "exactly equal".
2. NMS-merge overlapping spans on each side independently.
3. Match predictions to ground truth by **bare temporal overlap** -- no IoU threshold, many-to-many.
4. Count clip-level TP/FP/FN, and time-level intersection over summed durations.

Counts are summed across all videos and precision/recall/F1 computed once from the totals (micro),
not averaged per video. `ttmean` averages each metric across the 11 tolerances, which means
`ttmean_f1` is *not* the F1 of the mean precision and recall -- it is the mean of the F1s, as the
paper reports it.

Precision, recall and F1 are `None`, not `0.0`, when their denominator is zero. That is the
reference implementation's convention and is kept deliberately; anything consuming these rows has to
handle `None`.

## `durations` is the one input you cannot guess

It is the clamp bound for tolerance expansion, so it moves every number, and **it is not
automatically the container duration.**

For the published TransVLM figures it is the end of the last sliding window on the 25 fps timeline,
cast to float32. Checked against the 198 per-video duration records for AutoShot-test, the
window-end value matched the reference for 198 of 198 records, while the container duration matched
only 2. The two differ because a video is chunked on a 25 fps grid and the last chunk ends where the
chunking ends, not where the file ends.

Scoring your own predictions on your own data, the container duration is a perfectly reasonable
choice. Just make it a choice, and use the same one on both sides of any comparison.

## As a library

```python
from transvlm_eval import score

per_tolerance, ttmean = score(
    ground_truth={'a': [{'start_time': 2.0, 'end_time': 2.04}]},
    predictions={'a': [{'start_time': 2.0, 'end_time': 2.04}]},
    durations={'a': 12.0},
)
print(per_tolerance[0.00]['cl_f1'], ttmean['cl_f1'])
```

Clip-level predictions can be lifted to video time first with
`transvlm_eval.assembly.lift_clip_predictions`, which takes your clip spans and a callback fetching
each clip's raw model output. It has no opinion about where the clips came from.

## Reproducing the paper's numbers

You can reproduce the scoring procedure from this repository. Matching the published figures
exactly still needs care, because the inputs differ as follows:

- **The benchmark ground truth.** The frozen paper-evaluation labels are now provided in
  [`../benchmark/`](../benchmark/README.md) (version `eccv2026-paper`, v1); no videos are
  distributed, so bring your own copies from the original sources. In particular, AutoShot-test uses
  a corrected re-annotation, not the upstream labels: it differs on most videos, so scoring against
  the upstream labels gives a different number -- not a worse one, a *different* one, which is
  harder to interpret than no number. The frozen ClipShots JSON contains one reversed-interval
  segment; the strict CLI validation rejects it, so score that dataset through the public library
  entry point described in [`benchmark/README.md`](../benchmark/README.md).
- **The pre-computed optical flow, and the pixel budget.** The published run used flow generated by
  the training data pipeline and a max-pixels setting of 786432 (applied per frame); `../inference` computes flow
  at inference time and defaults to 524288 pixels. The shipped defaults were measured to land
  slightly *above* the published figures only on AutoShot-test (200 videos, 872 windows) with the
  HuggingFace backend. Optical flow and pixel budget differ together, so that difference cannot be
  attributed to the flow source alone.

What you can do today is score your own predictions with exactly the metric the paper used, and
compare methods against each other on equal terms.

## Example and tests

```bash
python -m pytest tests -q
```

`examples/` holds three hand-written videos exercising an exact hit, a partial overlap, a complete
miss, and a video with no ground truth at all. Its committed output under `examples/expected/` is
re-derived and compared on every test run, so the example is also the end-to-end regression for the
whole scoring path. The expected counts (2 TP / 2 FP / 1 FN at `tt0.00`) are derivable by hand from
the inputs -- see `examples/README.md` -- so a diff there can be reasoned about instead of just
re-blessed.

## `_vendored/` is our own code, not third-party

Despite the directory name, `transvlm_eval/_vendored/` is HeyGen's own work: three files --
the matcher, the NMS merge, and the shot-boundary value types -- copied byte-for-byte from an
internal shared library rather than written for this repository. They are covered by this
repository's LICENSE like everything else; no third-party attribution applies.

They are copied rather than reimplemented because they decide which prediction counts as
matching which ground truth, and a lookalike that rounds or orders differently produces numbers
that look right and are not. Their hashes are pinned by `tests/test_vendored_integrity.py`.

One intentional change: `tqdm` moved from a module-level import into the progress branches this
path never takes, so the scorer does not depend on it.

See `../NOTICE`.
