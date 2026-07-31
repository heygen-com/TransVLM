# Synthetic example

Three videos, hand-written, exercising the cases that matter:

- `clip_a` — one exact hit (a single-frame cut at 2.00 s) and one partial overlap (a 0.8 s gradual
  transition predicted 0.4 s wide). Tests that overlap counts as a match at the clip level while
  the time level sees the shortfall.
- `clip_b` — one ground-truth transition, one prediction, nowhere near each other. One false
  positive and one false negative, not a match.
- `clip_c` — no ground-truth transitions but a prediction anyway. Present in all three input dicts
  with an empty ground-truth list, which is how a video with nothing to find must be supplied. Omit
  it instead and its false positive goes uncounted, making precision look better than it is.

Run it:

```bash
python score.py --ground-truth examples/ground_truth.json \
                --predictions examples/predictions.json \
                --output-dir /tmp/transvlm_eval_example
```

`expected/` holds the scores this produces. `tests/test_example.py` re-runs the example and
compares, so the example is also the end-to-end regression test for the scoring path.
