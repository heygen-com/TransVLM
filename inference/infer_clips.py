"""TransVLM v1 inference over clips: one clip or a list, flow given or computed.

A faithful port of the reference implementation the published numbers came from. Only the
parameter passing changes: that script drives itself off two index JSONs which do not exist
on this GPU, and only `resized_video_path` and `optical_flow_path` ever reach the model
(`video_path` is read into a dead local at the reference implementation), so CLI flags replace both files
losing nothing. Every value that moves the numbers — engine kwargs, sampling params, fps 25,
the prompt and its suffix, the `rgb_and_of` message shape — is copied, not reinvented; see
`transvlm/inference/clip_engine.py`.

Three optional stages, each announced on stderr and each timed separately:

* **pre-resize** (`--pre-resize`): compute the target with `smart_resize_for_qwen3vl` and
  ffmpeg-scale into a temp file. Skipped with a notice when the target already equals the
  source, so a clip that is already inside the pixel budget is not re-encoded for nothing.
* **auto flow** (no `--flow` / `--flow-list` given): run NeuFlow into a temp file. Flow is
  always computed from the file about to be fed to the model, so a `--max-pixels-override`
  cannot desynchronise the RGB and flow dimensions.
* **batch** (`--video-list`): one engine for the whole list. `--flow-list` must be
  line-aligned with it.

Temp files live under `--temp-dir` on the big disk and are deleted per item in a `finally`,
so peak usage is one item's footprint rather than the whole run's.

Usage:
    # single clip, flow given
    python infer_clips.py \\
        --video <clip>_resized.mp4 --flow <clip>_flow.mp4 --output-jsonl out.jsonl

    # batch, flow computed, halved pixel budget
    python infer_clips.py \\
        --video-list list_clips_plain.txt --pre-resize --max-pixels-override 393216 \\
        --output-jsonl out.jsonl --no-prefix-caching
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# System temp rather than a path inside the repo: intermediates here are resized clips and
# optical-flow videos, which are large and are deleted per item. Writing them into the checkout
# fills the working tree and, on a shared machine, puts them on whatever filesystem the repo
# happens to live on. Override with --temp-dir to place them on a specific fast disk.
_DEFAULT_TEMP_ROOT = Path(tempfile.gettempdir()) / 'transvlm_infer'


@dataclass
class ItemTimings:
    resize_sec: float = 0.0
    flow_sec: float = 0.0
    preprocess_sec: float = 0.0
    generate_sec: float = 0.0
    wall_sec: float = 0.0
    # Breakdown of flow_sec. Kept separate rather than replacing it because the paper's
    # published flow RTF (0.22) measured only the NeuFlow forward -- decode, visualisation and
    # the mp4 encode are excluded from that figure. `None` when flow was supplied via --flow.
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
    src.add_argument('--video', type=Path, help='Single clip (RGB).')
    src.add_argument('--video-list', type=Path, help='Text file, one clip path per line.')
    p.add_argument('--flow', type=Path, help='Flow for --video. Omit to compute it.')
    p.add_argument('--flow-list', type=Path, help='Flow paths, line-aligned with --video-list. Omit to compute.')
    p.add_argument(
        '--no-pre-resize',
        action='store_true',
        help='Skip the ffmpeg resize and feed each clip at its on-disk resolution. Needed with '
        '--flow-list, whose pre-computed flow is at the paper resolution.',
    )
    p.add_argument(
        '--max-pixels-override',
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f'Per-frame pixel ceiling for the pre-resize. Default {DEFAULT_MAX_PIXELS} beats the '
        "paper's 786432 on both metrics and is faster; pass 786432 for the paper configuration.",
    )
    p.add_argument('--image-patch-size', type=int, default=16)
    p.add_argument('--nframes-for-resize', type=int, default=250)
    p.add_argument('--flow-mini-batch-size', type=int, default=32)
    p.add_argument(
        '--ckpt-dir',
        type=Path,
        required=True,
        help='Directory holding the TransVLM checkpoint (config.json + safetensors shards).',
    )
    p.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT_FILE)
    p.add_argument('--fps', type=float, default=DEFAULT_FPS)
    p.add_argument('--max-new-tokens', type=int, default=SAMPLING_KWARGS['max_tokens'])
    p.add_argument('--device', default='cuda:0', help='Device for NeuFlow (vLLM uses CUDA_VISIBLE_DEVICES).')
    p.add_argument('--output-jsonl', type=Path, required=True)
    p.add_argument('--temp-dir', type=Path, default=_DEFAULT_TEMP_ROOT)
    p.add_argument('--keep-temp', action='store_true', help='Leave per-item temp files for debugging.')
    p.add_argument('--no-prefix-caching', action='store_true', help='Required for reproducible A/B comparisons.')
    p.add_argument(
        '--backend',
        choices=('vllm', 'sglang', 'hf'),
        default='hf',
        help=(
            'Inference backend. hf is the default because it is the fastest here: measured '
            '71.7 min single-card-serial over 872 clips against vLLM 152.1 and SGLang 102.2, '
            'for ttmean cl_f1 within 0.0004 of vLLM. This workload is one request at a time '
            "with no batching, which is exactly where vLLM's scheduler is pure overhead. vllm "
            'remains the reference that produced the published numbers. sglang needs its own '
            'venv (incompatible flashinfer pin) -- see the inference README.'
        ),
    )
    p.add_argument(
        '--timestamp-format',
        choices=TIMESTAMP_FORMATS,
        default=DEFAULT_TIMESTAMP_FORMAT,
        help=(
            'Prompt timestamp precision. "1f" is every backend native behaviour and reproduces '
            'the published reference numbers; "2f" matches what training used. The two are not '
            'interchangeable, so their scores must never be compared.'
        ),
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
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--log-level', default='INFO')
    return p.parse_args()


def _reject_supplied_flow_with_resize(args: argparse.Namespace) -> None:
    """Refuse supplied flow together with a resize, because their dimensions cannot agree.

    Supplied flow is at the paper's 640x1152. Resizing the RGB to any other budget
    leaves ``process_vision_info``'s ``torch.cat`` on the channel axis with two different
    spatial shapes -- it raises, but deep inside the vendored preprocessing where the message
    says nothing about pixel budgets. Computed flow has no such problem: ``_maybe_flow`` derives
    it from the very file about to be fed to the model, so the two always match.
    """
    if args.flow is None and args.flow_list is None:
        return
    if args.no_pre_resize:
        return
    raise SystemExit(
        'supplied flow (--flow / --flow-list) is at the paper resolution, so it cannot '
        f'be combined with the pre-resize to max_pixels={args.max_pixels_override}. Either drop '
        'the flow argument and let NeuFlow compute matching flow, or add --no-pre-resize to feed '
        'both at their on-disk resolution.'
    )


def _read_pairs(args: argparse.Namespace) -> list[tuple[Path, Path | None]]:
    """Resolve the input into a list of (rgb, flow-or-None), validating alignment."""
    if args.video is not None:
        if args.flow_list is not None:
            raise SystemExit('--flow-list goes with --video-list, not --video')
        return [(args.video, args.flow)]

    if args.flow is not None:
        raise SystemExit('--flow goes with --video, not --video-list; use --flow-list')
    rgb = [Path(x) for x in args.video_list.read_text().split()]
    if not rgb:
        raise SystemExit(f'{args.video_list} is empty')
    if args.flow_list is None:
        return [(r, None) for r in rgb]
    flow = [Path(x) for x in args.flow_list.read_text().split()]
    if len(flow) != len(rgb):
        raise SystemExit(
            f'--flow-list has {len(flow)} line(s) but --video-list has {len(rgb)}. They must correspond 1:1.'
        )
    return list(zip(rgb, flow))


def _probe(path: Path, entries: str) -> list[str]:
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


def _duration_sec(path: Path) -> float:
    return float(_probe(path, 'format=duration')[0])


def _maybe_resize(src: Path, work: Path, args: argparse.Namespace, timings: ItemTimings) -> Path:
    """ffmpeg-scale to the smart_resize target, or return src unchanged."""
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
        # Re-encoding a file that is already at the target only perturbs pixels.
        logger.info('[resize] %s already %dx%d, skipping re-encode', src.name, src_h, src_w)
        return src
    dst = work / f'{src.stem}_resized{src.suffix}'
    logger.info('[resize] %s %dx%d -> %dx%d', src.name, src_h, src_w, target_h, target_w)
    started = time.perf_counter()
    resize_video_ffmpeg(src, dst, target_h=target_h, target_w=target_w, overwrite=True)
    timings.resize_sec = time.perf_counter() - started
    return dst


def _maybe_flow(
    rgb: Path, given: Path | None, work: Path, flow_computer: Any, timings: ItemTimings
) -> tuple[Path, str]:
    """Return (flow path, provenance). Computes from `rgb` so dimensions always agree."""
    if given is not None:
        if not given.is_file():
            raise SystemExit(f'--flow path not found: {given}')
        return given, 'given'
    logger.info('[flow] computing NeuFlow for %s', rgb.name)
    dst = work / f'{rgb.stem}_flow.mp4'
    started = time.perf_counter()
    _out, parts = flow_computer.compute_flow_only(rgb, dst)
    timings.flow_sec = time.perf_counter() - started
    timings.flow_parts = parts.as_dict()
    return dst, 'computed'


def _load_done(path: Path) -> set[str]:
    """Video paths already present in the output, for resume."""
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


def _resolve_engine_cls(backend: str) -> Any:
    """Import the requested backend's engine class. See transvlm/inference/backends.py."""
    from transvlm.inference.backends import resolve_engine_cls  # noqa: PLC0415

    return resolve_engine_cls(backend)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('c_tar.reader').setLevel(logging.ERROR)

    _reject_supplied_flow_with_resize(args)

    pairs = _read_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    mode = 'single' if args.video is not None else f'batch (N={len(pairs)})'
    flow_mode = 'given' if (args.flow or args.flow_list) else 'will compute'
    resize_mode = 'off' if args.no_pre_resize else f'on (max_pixels={args.max_pixels_override})'
    logger.info('[mode] %s | [flow] %s | [resize] %s', mode, flow_mode, resize_mode)

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

    # Imported here rather than at module scope so --help stays fast.
    from transvlm.data.video_backend import selected_video_backend  # noqa: PLC0415

    engine_cls = _resolve_engine_cls(args.backend)
    engine = engine_cls(
        ckpt_dir=args.ckpt_dir,
        prompt_file=args.prompt_file,
        fps=args.fps,
        max_new_tokens=args.max_new_tokens,
        no_prefix_caching=args.no_prefix_caching,
        timestamp_format=args.timestamp_format,
    )
    engine.load()

    # engine.close() must run even on an abnormal exit. The SIGTERM handler installed
    # below turns the signal into KeyboardInterrupt, which is NOT an Exception subclass,
    # so it escapes the per-item `except Exception` and would skip the teardown. For
    # SGLang that means orphaned sglang::scheduler subprocesses holding tens of GiB.
    try:

        flow_computer: Any = None
        if flow_mode == 'will compute':
            # Built after the engine so it cannot disturb vLLM's memory profiling.
            from transvlm.data.flow_config import FlowConfig  # noqa: PLC0415
            from transvlm.data.flow_computer import OnlineFlowComputer  # noqa: PLC0415

            flow_computer = OnlineFlowComputer(
                FlowConfig(mode='online', flow_codec=args.flow_codec),
                device=args.device,
                mini_batch_size=args.flow_mini_batch_size,
                max_pixels_override=args.max_pixels_override,
                viz_on_gpu=args.flow_viz_device == 'gpu',
            )

        # Let SIGTERM raise so the TemporaryDirectory context manager unwinds normally.
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

        args.temp_dir.mkdir(parents=True, exist_ok=True)
        n_ok = n_fail = 0
        started_all = time.perf_counter()
        with (
            tempfile.TemporaryDirectory(prefix='transvlm_clip_', dir=args.temp_dir) as temp_root,
            args.output_jsonl.open('a') as sink,
        ):
            for i, (rgb, flow) in enumerate(pending, 1):
                work = Path(temp_root) / f'item{i:06d}'
                work.mkdir(parents=True, exist_ok=True)
                timings = ItemTimings()
                item_started = time.perf_counter()
                try:
                    model_rgb = _maybe_resize(rgb, work, args, timings)
                    model_flow, flow_provenance = _maybe_flow(
                        rgb=model_rgb, given=flow, work=work, flow_computer=flow_computer, timings=timings
                    )
                    result = engine.infer(model_rgb, model_flow)
                    timings.preprocess_sec = result.preprocess_sec
                    timings.generate_sec = result.generate_sec
                    timings.wall_sec = time.perf_counter() - item_started
                    duration = _duration_sec(model_rgb)
                    record: dict[str, Any] = {
                        'video': str(rgb),
                        'model_rgb': str(model_rgb) if model_rgb != rgb else None,
                        'flow_source': flow_provenance,
                        'flow_path': str(flow) if flow is not None else None,
                        'duration_sec': duration,
                        'n_frames': result.n_frames,
                        'height': result.height,
                        'width': result.width,
                        'inference_result': result.raw_text,
                        'n_input_tokens': result.n_input_tokens,
                        'n_output_tokens': result.n_output_tokens,
                        'hit_token_cap': result.n_output_tokens >= args.max_new_tokens,
                        'time_base': 'chunk_local',
                        'timings': {
                            'resize_sec': round(timings.resize_sec, 4),
                            'flow_sec': round(timings.flow_sec, 4),
                            'preprocess_sec': round(timings.preprocess_sec, 4),
                            'generate_sec': round(timings.generate_sec, 4),
                            'wall_sec': round(timings.wall_sec, 4),
                            **(timings.flow_parts or {}),
                        },
                        'config': {
                            'pre_resize': not args.no_pre_resize,
                            'max_pixels_override': args.max_pixels_override,
                            'fps': args.fps,
                            'max_new_tokens': args.max_new_tokens,
                            'prefix_caching': not args.no_prefix_caching,
                            'ckpt_dir': str(args.ckpt_dir),
                            'backend': args.backend,
                            # Recorded because it changes the prompt and therefore every
                            # predicted boundary; a jsonl without it cannot be compared to
                            # anything.
                            'timestamp_format': args.timestamp_format,
                            'flow_viz_on_gpu': args.flow_viz_device == 'gpu',
                            # Changes the flow signal the model sees (measured 121/255 peak
                            # error between the two), and computed flow lives in a temp dir
                            # that is deleted, so the jsonl is the only surviving record.
                            'flow_codec': args.flow_codec,
                            # Which decoder produced the frames. A decord-fallback run and a torchcodec run
                            # are different signals, so a record that omits this is not self-describing.
                            'video_decoder': selected_video_backend(),
                            # Engine-visible knobs, so a jsonl is self-describing about which
                            # kernels produced it. On this host flashinfer's JIT cannot build for
                            # sm_120 (system nvcc is CUDA 12.0, max compute_90), so SGLang runs
                            # need attention_backend=triton and that must be recorded.
                            'backend_env': {
                                # Widened from TRANSVLM_SGLANG_ once vLLM gained an
                                # attention-backend override: a jsonl has to record every knob
                                # that can change kernels, not just one backend's.
                                k: v
                                for k, v in sorted(os.environ.items())
                                if k.startswith('TRANSVLM_')
                            },
                        },
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + '\n')
                    sink.flush()
                    n_ok += 1
                except Exception:
                    n_fail += 1
                    logger.exception('item %d (%s) failed', i, rgb.name)
                finally:
                    if not args.keep_temp:
                        shutil.rmtree(work, ignore_errors=True)
                if i % 25 == 0 or i == len(pending):
                    logger.info('[%d/%d] clips', i, len(pending))

        elapsed = time.perf_counter() - started_all
        logger.info(
            'Done in %.1f s — %d ok, %d failed (engine load %.1f s, excluded from item timings)',
            elapsed,
            n_ok,
            n_fail,
            engine.model_load_sec,
        )
    finally:
        engine.close()
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
