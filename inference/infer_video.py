"""TransVLM v1 inference over whole videos: resample -> resize -> flow -> sliding windows.

Companion to `infer_clips.py`, sharing its engine (`transvlm/inference/clip_engine.py`) so
the two cannot drift apart in protocol or in what their timings mean.

Stage order is fixed and each stage is announced and timed:

1. **fps -> 25** if needed. The check is rational-exact against `25/1`, so 30000/1001
   counts as unequal, and a video whose `avg_frame_rate` differs from `r_frame_rate` (VFR)
   is resampled too. 59 of 200 videos in one benchmark we measured need this.
2. **resize** to the smart_resize target (skipped with a notice when already there).
3. **flow** for the WHOLE video, in one NeuFlow call.
4. **sliding windows**, as time ranges — nothing is cut on disk.

Two subtleties in step 4, both measured rather than assumed:

* The range calculator is end-INCLUSIVE (`ceil(s*fps)` .. `floor(e*fps)`), so asking for
  `[9, 19]` spans 251 frames where a pre-cut 10 s clip has 250. Windows are therefore
  requested as `[s, min(e - 0.5/fps, duration)]`.
* A range-decoded window reports ABSOLUTE frame indices, and prompt timestamps derive from
  that field, so the model would see absolute video time where a pre-cut clip shows
  window-local time. `infer(..., rebase_frame_indices=True)` shifts them to start at 0.
  Rebasing rather than resetting to `range(n)` is what matters: `smart_nframes` floors the
  count to a multiple of 2 and then samples with `linspace`, so the index set is not always
  contiguous.

Verified on `<video-id>` (45.76 s, 25 fps) against its pre-cut clips: with the
half-frame compensation, windows 0/1/2 all give 250 frames AND a rebased index set equal to
the clip's; without it the frame count still matches but the sampling phase does not.

Whole-video flow is not the same signal as per-clip flow even on identical frames, because
`flow_to_image` normalises by the maximum flow magnitude over its whole input. That is a
property of the requested pipeline order, not a defect, and it belongs in any comparison
against per-clip references.

Usage:
    python infer_video.py \\
        --video-list videos.txt --no-prefix-caching \\
        --output-jsonl b1.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

# Module scope, NOT inside main(): `_process_video` resolves this name as a global. Moving the
# import into main() makes every video raise NameError, which the per-item `except Exception`
# swallows into "each video failed" while the output file comes out empty. `tests/
# test_infer_video_module.py` checks every top-level function against module scope for this reason.
from transvlm.data.video_backend import selected_video_backend

logger = logging.getLogger(__name__)

# System temp rather than a path inside the repo: intermediates here are resized clips and
# optical-flow videos, which are large and are deleted per item. Writing them into the checkout
# fills the working tree and, on a shared machine, puts them on whatever filesystem the repo
# happens to live on. Override with --temp-dir to place them on a specific fast disk.
_DEFAULT_TEMP_ROOT = Path(tempfile.gettempdir()) / 'transvlm_infer'

# The paper protocol's fps. Kept as documentation of the default; the actual target now comes
# from --fps so the model's sampling fps and the container's fps cannot drift apart.
TARGET_FPS = Fraction(25, 1)


@dataclass
class VideoTimings:
    resample_sec: float = 0.0
    resize_sec: float = 0.0
    flow_sec: float = 0.0
    window_plan_sec: float = 0.0
    preprocess_sec_total: float = 0.0
    generate_sec_total: float = 0.0
    parse_merge_sec: float = 0.0
    wall_sec: float = 0.0
    per_window: list[dict[str, Any]] = field(default_factory=list)
    # Breakdown of flow_sec; only the NeuFlow forward is what the paper's 0.22 measured.
    # `None` when flow was supplied rather than computed.
    flow_parts: dict[str, float] | None = None


def _parse_args() -> argparse.Namespace:
    from transvlm.inference.clip_engine import (  # noqa: PLC0415
        DEFAULT_FPS,
        DEFAULT_MAX_PIXELS,
        DEFAULT_PROMPT_FILE,
        DEFAULT_TIMESTAMP_FORMAT,
        SAMPLING_KWARGS,
    )
    from transvlm.models.processor_patch import TIMESTAMP_FORMATS  # noqa: PLC0415

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', type=Path)
    src.add_argument('--video-list', type=Path)
    p.add_argument('--flow', type=Path, help='Whole-video flow for --video. Omit to compute.')
    p.add_argument('--flow-list', type=Path, help='Line-aligned with --video-list. Omit to compute.')
    p.add_argument('--window-size', type=float, default=10.0)
    p.add_argument('--stride', type=float, default=9.0)
    p.add_argument('--strict-tail', action='store_true', help='Right-anchor the tail window (default: shorter tail).')
    p.add_argument('--merge-eps', type=float, default=0.02)
    p.add_argument('--no-pre-resize', action='store_true', help='Feed the video without the smart_resize step.')
    p.add_argument(
        '--max-pixels-override',
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f'Per-frame pixel ceiling for the pre-resize. Default {DEFAULT_MAX_PIXELS}; pass '
        '786432 for the paper configuration. See infer_clips.py.',
    )
    p.add_argument('--image-patch-size', type=int, default=16)
    p.add_argument('--nframes-for-resize', type=int, default=250)
    p.add_argument('--no-fps-resample', action='store_true', help='Skip the 25 fps conversion.')
    p.add_argument(
        '--backend',
        choices=('vllm', 'sglang', 'hf'),
        default='hf',
        help='Inference backend; see infer_clips.py --backend for why hf is the default.',
    )
    p.add_argument(
        '--fps',
        type=float,
        default=DEFAULT_FPS,
        help=f'Sampling fps handed to the model AND used for the ffmpeg resample. Default '
        f'{DEFAULT_FPS} is the paper protocol; changing it also changes the optical-flow signal '
        '(flow_to_image normalises over the whole clip, so the frame count matters). '
        'Previously hardcoded in three places.',
    )
    p.add_argument(
        '--timestamp-format',
        choices=TIMESTAMP_FORMATS,
        default=DEFAULT_TIMESTAMP_FORMAT,
        help='Prompt timestamp precision; see infer_clips.py --timestamp-format.',
    )
    p.add_argument(
        '--flow-codec',
        choices=('libx264', 'lossless'),
        default='libx264',
        help=(
            'Flow-visualisation encode fidelity. libx264 is the pinned training-time setting and '
            'stays the default. It is lossy where it hurts: torchvision writes yuv420p, so chroma '
            'is 4:2:0 subsampled while flow encodes direction as hue -- measured up to 121/255 '
            'peak and 1.30 mean absolute error per channel round-tripping two autoshot clips. '
            'lossless (libx264rgb crf=0) round-trips bit-exactly at ~25x the bytes; its siblings '
            'carry an _ll tag so the two fidelities cache independently.'
        ),
    )
    p.add_argument(
        '--flow-viz-device',
        choices=('gpu', 'cpu'),
        default='gpu',
        help=(
            'Where to run flow_to_image, the flow stage biggest cost (60%% of it). gpu is the '
            'default: measured 1.91x faster end-to-end over 872 clips with |delta ttmean| = '
            '0.00033, inside the 0.002 parity gate. cpu reproduces the pre-2026-07-30 byte '
            'stream exactly -- the two devices differ by max 1/255 on ~1e-6 of pixels, so the '
            'encoded flow mp4 is not identical.'
        ),
    )
    p.add_argument('--flow-mini-batch-size', type=int, default=32)
    p.add_argument(
        '--ckpt-dir',
        type=Path,
        required=True,
        help='Directory holding the TransVLM checkpoint (config.json + safetensors shards).',
    )
    p.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT_FILE)
    p.add_argument('--max-new-tokens', type=int, default=SAMPLING_KWARGS['max_tokens'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-jsonl', type=Path, required=True)
    p.add_argument('--temp-dir', type=Path, default=_DEFAULT_TEMP_ROOT)
    p.add_argument('--keep-temp', action='store_true')
    p.add_argument('--no-prefix-caching', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


def _ffprobe(path: Path, entries: str) -> list[str]:
    out = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            entries,
            '-of',
            'default=nw=1:nk=1',
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.split()


def _frame_rates(path: Path) -> tuple[Fraction, Fraction]:
    avg, real = _ffprobe(path, 'stream=avg_frame_rate,r_frame_rate')[:2]
    return Fraction(avg), Fraction(real)


def _duration_sec(path: Path) -> float:
    return float(_ffprobe(path, 'format=duration')[0])


def _needs_resample(path: Path, target: Fraction) -> tuple[bool, str]:
    """True when the video is not exactly ``target`` CFR. Rational-exact, so 30000/1001 counts."""
    avg, real = _frame_rates(path)
    if avg != real:
        return True, f'VFR (avg {avg} != r {real})'
    if avg != target:
        return True, f'{avg} fps'
    return False, f'{target} CFR'


def _resample(src: Path, work: Path, timings: VideoTimings, target: Fraction) -> Path:
    """ffmpeg-resample to ``target`` fps.

    ``target`` is threaded from ``--fps`` rather than hardcoded: the model's sampling fps and
    the container's fps have to agree, and when 25 was fixed in three separate places they
    could not be changed together.
    """
    dst = work / f'{src.stem}_fps{float(target):g}.mp4'
    cmd = [
        'ffmpeg',
        '-y',
        '-v',
        'error',
        '-i',
        str(src),
        '-filter:v',
        f'fps={target}',
        '-c:v',
        'libx264',
        '-crf',
        '17',
        '-preset',
        'fast',
        '-an',
        str(dst),
    ]
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg fps={target} failed on {src}\n{result.stderr}')
    timings.resample_sec = time.perf_counter() - started
    return dst


def _maybe_resize(src: Path, work: Path, args: argparse.Namespace, timings: VideoTimings) -> Path:
    if args.no_pre_resize:
        return src
    from transvlm.data.resize_video_helper import (  # noqa: PLC0415
        probe_video_hw,
        resize_video_ffmpeg,
        smart_resize_for_qwen3vl,
    )

    src_h, src_w = probe_video_hw(src)
    target_h, target_w = smart_resize_for_qwen3vl(
        src_h,
        src_w,
        nframes=args.nframes_for_resize,
        image_patch_size=args.image_patch_size,
        max_pixels_override=args.max_pixels_override,
    )
    if (target_h, target_w) == (src_h, src_w):
        logger.info('[resize] %s already %dx%d, skipping re-encode', src.name, src_h, src_w)
        return src
    dst = work / f'{src.stem}_resized.mp4'
    logger.info('[resize] %s %dx%d -> %dx%d', src.name, src_h, src_w, target_h, target_w)
    started = time.perf_counter()
    resize_video_ffmpeg(src, dst, target_h=target_h, target_w=target_w, overwrite=True)
    timings.resize_sec = time.perf_counter() - started
    return dst


def _read_pairs(args: argparse.Namespace) -> list[tuple[Path, Path | None]]:
    if args.video is not None:
        if args.flow_list is not None:
            raise SystemExit('--flow-list goes with --video-list')
        return [(args.video, args.flow)]
    if args.flow is not None:
        raise SystemExit('--flow goes with --video; use --flow-list')
    rgb = [Path(x) for x in args.video_list.read_text().split()]
    if not rgb:
        raise SystemExit(f'{args.video_list} is empty')
    if args.flow_list is None:
        return [(r, None) for r in rgb]
    flow = [Path(x) for x in args.flow_list.read_text().split()]
    if len(flow) != len(rgb):
        raise SystemExit(f'--flow-list has {len(flow)} line(s) but --video-list has {len(rgb)}')
    return list(zip(rgb, flow))


def _load_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)['video'])
        except (ValueError, KeyError):
            logger.warning('skipping unparseable output line during resume scan')
    return done


def _process_video(
    rgb_src: Path,
    flow_given: Path | None,
    work: Path,
    *,
    args: argparse.Namespace,
    engine: Any,
    flow_computer: Any,
) -> dict[str, Any]:
    from transvlm.data.sliding_window import get_sliding_window_segments  # noqa: PLC0415
    from transvlm.inference.merger import merge_shots  # noqa: PLC0415
    from transvlm.inference.parser import parse_model_output  # noqa: PLC0415

    timings = VideoTimings()
    item_started = time.perf_counter()

    # 1. fps -> args.fps
    # 10000, not 1000: NTSC is 30000/1001 and a 1000 cap cannot represent it (it lands on
    # 19001/634). Nothing breaks at 1000 -- container and engine.fps stay consistent -- but exact
    # NTSC is unreachable, so a true 29.97 source is always re-encoded even when asked to match.
    target_fps = Fraction(args.fps).limit_denominator(10000)
    resampled_from = None
    model_src = rgb_src
    if not args.no_fps_resample:
        needed, why = _needs_resample(rgb_src, target_fps)
        if needed:
            logger.info('[resample] %s is %s -> converting to %s fps', rgb_src.name, why, target_fps)
            model_src = _resample(rgb_src, work, timings, target_fps)
            resampled_from = why
        else:
            logger.info('[resample] %s already %s, skipping', rgb_src.name, why)

    # 2. resize
    model_rgb = _maybe_resize(model_src, work, args, timings)

    # 3. whole-video flow
    if flow_given is not None:
        model_flow = flow_given
        flow_provenance = 'given'
    else:
        logger.info('[flow] computing whole-video NeuFlow for %s', model_rgb.name)
        model_flow = work / f'{model_rgb.stem}_flow.mp4'
        started = time.perf_counter()
        _flow_out, flow_parts = flow_computer.compute_flow_only(model_rgb, model_flow)
        timings.flow_sec = time.perf_counter() - started
        timings.flow_parts = flow_parts.as_dict()
        flow_provenance = 'computed'

    # 4. window plan
    duration = _duration_sec(model_rgb)
    started = time.perf_counter()
    windows = get_sliding_window_segments(
        start=0.0,
        end=duration,
        window_size=args.window_size,
        stride=args.stride,
        strict_tail=args.strict_tail,
    )
    timings.window_plan_sec = time.perf_counter() - started
    logger.info('[windows] %s: %.2f s -> %d window(s)', model_rgb.name, duration, len(windows))

    # 5. per-window inference, as time ranges
    half_frame = 0.5 / engine.fps
    all_segments = []
    for idx, (w_start, w_end) in enumerate(windows):
        requested_end = min(w_end - half_frame, duration)
        result = engine.infer(
            model_rgb,
            model_flow,
            video_start=w_start,
            video_end=requested_end,
            rebase_frame_indices=True,
        )
        timings.preprocess_sec_total += result.preprocess_sec
        timings.generate_sec_total += result.generate_sec
        timings.per_window.append(
            {
                'idx': idx,
                'start': round(w_start, 4),
                'end': round(w_end, 4),
                'requested_end': round(requested_end, 4),
                'n_frames': result.n_frames,
                'raw_output': result.raw_text,
                'n_output_tokens': result.n_output_tokens,
                'n_input_tokens': result.n_input_tokens,
                'preprocess_sec': round(result.preprocess_sec, 4),
                'generate_sec': round(result.generate_sec, 4),
            }
        )
        all_segments.extend(parse_model_output(result.raw_text, window_offset=w_start, window_index=idx))

    # 6. merge
    started = time.perf_counter()
    merged = merge_shots(all_segments, eps=args.merge_eps)
    timings.parse_merge_sec = time.perf_counter() - started
    timings.wall_sec = time.perf_counter() - item_started

    return {
        'video': str(rgb_src),
        'resampled_from': resampled_from,
        'model_rgb': str(model_rgb),
        'flow_source': flow_provenance,
        'duration_sec': duration,
        'n_windows': len(windows),
        'n_raw_segments': len(all_segments),
        'n_segments': len(merged),
        'segments': [
            {'start_time': s.start_time, 'end_time': s.end_time, 'source_window': s.source_window} for s in merged
        ],
        'time_base': 'absolute_video',
        'windows': timings.per_window,
        'timings': {
            'resample_sec': round(timings.resample_sec, 4),
            'resize_sec': round(timings.resize_sec, 4),
            'flow_sec': round(timings.flow_sec, 4),
            **(timings.flow_parts or {}),
            # There is no separate cutting step: windows are time ranges, so the
            # "sliding-window split" cost is the plan (microseconds) plus the per-window
            # sparse decode, which lives in preprocess_sec_total.
            'window_plan_sec': round(timings.window_plan_sec, 6),
            'preprocess_sec_total': round(timings.preprocess_sec_total, 4),
            'generate_sec_total': round(timings.generate_sec_total, 4),
            'parse_merge_sec': round(timings.parse_merge_sec, 4),
            'wall_sec': round(timings.wall_sec, 4),
        },
        'config': {
            'window_size': args.window_size,
            'stride': args.stride,
            'strict_tail': bool(args.strict_tail),
            'merge_eps': args.merge_eps,
            'pre_resize': not args.no_pre_resize,
            'max_pixels_override': args.max_pixels_override,
            'fps_resample': not args.no_fps_resample,
            'max_new_tokens': args.max_new_tokens,
            'prefix_caching': not args.no_prefix_caching,
            'ckpt_dir': str(args.ckpt_dir),
            'backend': args.backend,
            # Recorded because it changes the prompt and therefore every predicted boundary.
            'timestamp_format': args.timestamp_format,
            'flow_viz_on_gpu': args.flow_viz_device == 'gpu',
            # Changes the flow signal the model sees (measured 121/255 peak error between the
            # two), and computed flow lives in a temp dir that is deleted, so the jsonl is
            # the only surviving record of which encode produced these numbers.
            'flow_codec': args.flow_codec,
            # Which decoder produced the frames. A decord-fallback run and a torchcodec run
            # are different signals, so a record that omits this is not self-describing.
            'video_decoder': selected_video_backend(),
        },
    }


def _resolve_engine_cls(backend: str) -> Any:
    """Import the requested backend's engine class. See transvlm/inference/backends.py."""
    from transvlm.inference.backends import resolve_engine_cls  # noqa: PLC0415

    return resolve_engine_cls(backend)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    pairs = _read_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    mode = 'single' if args.video is not None else f'batch (N={len(pairs)})'
    flow_mode = 'given' if (args.flow or args.flow_list) else 'will compute'
    logger.info(
        '[mode] %s | [flow] %s | [resize] %s | [resample] %s | [windows] w%.1f/s%.1f strict_tail=%s',
        mode,
        flow_mode,
        'off' if args.no_pre_resize else f'on (max_pixels={args.max_pixels_override or "default"})',
        'off' if args.no_fps_resample else f'on (to {args.fps:g} fps when needed)',
        args.window_size,
        args.stride,
        args.strict_tail,
    )

    for rgb, flow in pairs:
        if not rgb.is_file():
            raise SystemExit(f'input not found: {rgb}')
        if flow is not None and not flow.is_file():
            raise SystemExit(f'flow not found: {flow}')

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    from transvlm.inference.backends import assert_output_backend_matches  # noqa: PLC0415

    assert_output_backend_matches(str(args.output_jsonl), args.backend)
    done = _load_done(args.output_jsonl)
    pending = [(r, f) for r, f in pairs if str(r) not in done]
    logger.info('[resume] %d done, %d pending', len(pairs) - len(pending), len(pending))
    if not pending:
        return 0

    engine_cls = _resolve_engine_cls(args.backend)
    engine = engine_cls(
        ckpt_dir=args.ckpt_dir,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        no_prefix_caching=args.no_prefix_caching,
        timestamp_format=args.timestamp_format,
        fps=args.fps,
    )
    engine.load()

    # engine.close() must run even on an abnormal exit. The SIGTERM handler installed
    # below turns the signal into KeyboardInterrupt, which is NOT an Exception subclass,
    # so it escapes the per-item `except Exception` and would skip the teardown. For
    # SGLang that means orphaned sglang::scheduler subprocesses holding tens of GiB.
    try:

        flow_computer: Any = None
        if flow_mode == 'will compute':
            from transvlm.data.flow_config import FlowConfig  # noqa: PLC0415
            from transvlm.data.flow_computer import OnlineFlowComputer  # noqa: PLC0415

            flow_computer = OnlineFlowComputer(
                FlowConfig(mode='online', flow_codec=args.flow_codec),
                device=args.device,
                mini_batch_size=args.flow_mini_batch_size,
                max_pixels_override=args.max_pixels_override,
                viz_on_gpu=args.flow_viz_device == 'gpu',
            )

        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        n_ok = n_fail = 0
        started_all = time.perf_counter()
        with (
            tempfile.TemporaryDirectory(prefix='transvlm_video_', dir=args.temp_dir) as temp_root,
            args.output_jsonl.open('a') as sink,
        ):
            for i, (rgb, flow) in enumerate(pending, 1):
                work = Path(temp_root) / f'item{i:06d}'
                work.mkdir(parents=True, exist_ok=True)
                try:
                    record = _process_video(rgb, flow, work, args=args, engine=engine, flow_computer=flow_computer)
                    sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                    sink.flush()
                    n_ok += 1
                except Exception:
                    n_fail += 1
                    logger.exception('video %d (%s) failed', i, rgb.name)
                finally:
                    if not args.keep_temp:
                        shutil.rmtree(work, ignore_errors=True)
                if i % 10 == 0 or i == len(pending):
                    logger.info('[%d/%d] videos', i, len(pending))

        logger.info(
            'Done in %.1f s — %d ok, %d failed (engine load %.1f s, excluded)',
            time.perf_counter() - started_all,
            n_ok,
            n_fail,
            engine.model_load_sec,
        )
    finally:
        engine.close()

    if pending and n_ok == 0:
        # Categorically different from "some videos failed": nothing was written at all, which
        # is a pipeline fault, not a data problem. Said explicitly because the failure this
        # guards against produced exactly this shape -- a NameError swallowed by the per-item
        # `except Exception`, one traceback per video, an empty jsonl, and a log that read like
        # 200 bad videos.
        logger.error(
            'every one of %d video(s) failed and %s is empty -- treat this as a pipeline fault, '
            'not bad input. Check the first traceback above.',
            len(pending),
            args.output_jsonl,
        )
        return 1
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
