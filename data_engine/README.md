# Data engine: shot-transition video synthesis

This directory contains the self-contained transition synthesis engine used to
build training clips from user-supplied shots. It consumes an explicit ordered
list of local clip files and writes one composed video plus two JSON files: the
transition labels and the sampled parameters.

The engine supports **59 transition names**: one hard `cut` plus 58 FFmpeg
`xfade` transitions. It does not mine clean shots, does not index footage, and
does not contact any network service. You provide the clean shots; the engine
samples windows, joins them, and records labels.

## Requirements

- Python 3.10+ with `numpy` (already a project dependency).
- `ffmpeg` and `ffprobe` on `PATH`. Transition rendering uses the `xfade`,
  `acrossfade`, `concat`, `settb`, `setpts`, `fps`, `format` and `setsar`
  filters.

`tqdm` is not required by this engine.

## Usage

```bash
# Probe a file (duration and FPS)
python -m data_engine.cli --probe inputs/shot_01.mp4

# Compose from explicit paths, with a seed for reproducibility
python -m data_engine.cli \
    --video inputs/shot_01.mp4 --video inputs/shot_02.mp4 --video inputs/shot_03.mp4 \
    --output-dir outputs --seed 7 --threads 2

# Compose from a JSON config (see configs/example.json)
python -m data_engine.cli --config data_engine/configs/example.json

# Validate and print the sampled plan without running FFmpeg
python -m data_engine.cli --config data_engine/configs/example.json --dry-run

# Restrict the transition pool: all 59 (default), the 58 gradual ones, or a list
python -m data_engine.cli ... --transition-types long
python -m data_engine.cli ... --transition-types cut,fade
```

Outputs in `save_dir`:

| File | Content |
| --- | --- |
| `<video_id>.mp4` | Composed video (H.264 / AAC). |
| `<video_id>.json` | Label list, one entry per transition: `start_time`, `end_time`, `transition_type` (seconds). |
| `<video_id>_params.json` | Inputs, sampled shot windows, transition parameters, and sampling constants. |

`video_id` is taken from the config or `--video-id`; otherwise it is the first
8 hex digits of a SHA-256 over the parameter record.

## Configuration keys

`video_files`, `save_dir`, `eps` (default `0.01`), `shot_times`, `transition_params`,
`min_shot_duration` (`2.0`), `min_shot_interval` (`0.5`),
`max_transition_duration_ratio` (`0.95`), `min_transition_duration_ratio` (`0.01`),
`fps` (`25.0`), `threads` (`8`), `video_id`, `transition_types`, `seed`.
Unknown keys are rejected.

## Behaviour preserved from the original engine

These are intentional and must not be "fixed" silently:

- **Encoder settings.** `libx264`, `-crf 17`, `-preset fast`, AAC 192k, and the
  same thread flags for trimming, concat and transition steps.
- **Shot count.** The engine places no lower or upper shot-count bound beyond
  requiring at least two inputs. The CLI consumes an explicit, ordered local
  list and never samples or selects 2-7 shots; 2-7 was the historical external
  batch driver default, not a default of this CLI.
- **Shot windows.** For each input of duration `D` and minimum shot duration `m`,
  if `D < m` the whole clip is used; otherwise the window length is
  `U(m, D)` and the start is `U(0, D - length)`. Windows are floor-rounded to
  three decimals when requested.
- **Non-cut duration.** With
  `L = min(d0 - prev_duration, d1) - min_shot_interval`:
  `upper = 0.95 * L`, `lower = max(3 / fps, 0.01 * L)`, and
  `duration = U(lower, upper)`. The offset is
  `prev_offset + d0 - duration`. Both are floor-rounded to three decimals.
- **Hard cut.** Duration is exactly `0.0` and the clips are concatenated
  (no `xfade`), so labels are zero-length intervals.
- **RNG order.** Shot windows first, then for each pair the transition type, then
  (non-cut only) the duration. A seed reproduces the same plan only with the
  same inputs, the same probed durations and frame rates, and the same Python
  environment; probed values feed the RNG boundaries, so a different probe
  result yields a different plan.
- **Transition rendering.** Every non-cut name is passed verbatim to
  `xfade`; audio uses `acrossfade` with the same duration. Per-step offsets are
  recomputed from the actual duration of the composed prefix, floored to three
  decimals, exactly as before.
- **JSON format.** Four-space indentation, non-ASCII characters written raw.

## Differences from the ECCV paper text (disclosed, not tuned)

The paper's appendix describes a different sampling configuration. This release
faithfully ships the historical engine and does **not** adjust values to make
the text match:

| Item | This engine | Paper text |
| --- | --- | --- |
| Encoder quality | CRF 17 | CRF 20 |
| Shots per video | 2-7 (historical external batch driver default; this CLI takes an explicit ordered list) | 2-5 |
| Non-cut duration | `max(3/fps, 0.01*L) .. 0.95*L` | `U(0, T_max)` |
| Cut duration | 0 | 0 (matches) |

The transition name set (59) and the zero-length cut labels match the paper.

The ratios are configurable through `min_transition_duration_ratio` and
`max_transition_duration_ratio`, but the `3 / fps` floor is hard-coded in the
sampling function, so setting the ratio to zero does **not** reproduce the
paper's `U(0, T_max)`, and reducing the ratios cannot. The encoder quality is
fixed at CRF 17 / preset `fast`; CRF 20 is not reachable through the current
interface. Reproducing the paper-text sampling would require changing the
engine, which this release intentionally does not do.

## Scope and limitations

- The engine does **not** reproduce the full historical training-set pipeline.
  Clean-shot mining, candidate selection, dataset indexing and storage layouts
  are out of scope. Only synthesis from an existing local shot list is provided.
- `--dry-run` previews sampled plans but cannot know the exact duration of
  trimmed clips without encoding; it uses the requested window length. Rendered
  output uses probed durations.
- Outputs are not guaranteed byte-identical across FFmpeg or x264 versions.
  Version-independent statements are limited to the transition table and the
  sampling formula. Reproducing sampled values, timestamps or bytes requires
  the same inputs, the same probe results and the same FFmpeg/x264 versions.
  Per-step offsets are derived from the actual probed duration of the composed
  prefix, so a different probe result shifts the labels.
- Every input must have a video stream and a frame rate within `eps` of the
  configured `fps` (default 25). Containers without audio are not supported by
  the current filter graph, which maps an audio stream from each input.

## Tests

```bash
python -m unittest discover -s data_engine/tests -v
```

Unit tests cover the transition table, rational FPS parsing, stream selection,
duration fallbacks, numpy-aware JSON output, rounding, sampling bounds, seeded
reproducibility, cut semantics, degenerate inputs, CLI configuration and CLI
forwarding of the transition pool. They do not invoke FFmpeg.

`data_engine/tests/test_ffmpeg_smoke.py` is a public CPU smoke test. It
generates two colour clips with tone audio in a temporary directory, then
renders a `cut` and a `fade` through the real CLI and checks the output video
and audio streams plus the label type and span. The fixture requests one
libx264 encoding thread and one filter thread, and calls the engine with
`threads=1`; these are thread-count requests, not a hard cap on every internal
thread FFmpeg or x264 may create. The test skips when `ffmpeg` or `ffprobe` is
missing. No media is written into the repository.

```bash
python -m unittest data_engine.tests.test_ffmpeg_smoke -v
```

## Licensing

This directory is released under the Apache License 2.0, the same license as
the repository root (see `LICENSE` and `NOTICE`). It is a self-contained
implementation and does not vendor internal helper libraries.

Third-party components are **not** relicensed by this project:

- **FFmpeg / x264** — separate projects with their own licenses; users are
  responsible for complying with them (FFmpeg builds may be LGPL or GPL
  depending on configuration, and libx264 is GPL).
- **NumPy** — BSD-3-Clause.
- **Python standard library** — PSF License.

No third-party source code is included in this directory.
