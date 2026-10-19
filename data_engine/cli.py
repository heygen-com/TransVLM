"""Command-line interface for the transition synthesis data engine.

Inputs are always explicit local paths. The CLI does not discover footage,
does not query any storage service and does not contact a network endpoint.

Usage:
    python -m data_engine.cli --config configs/example.json
    python -m data_engine.cli --video a.mp4 --video b.mp4 --video c.mp4 \
        --output-dir outputs --seed 7
    python -m data_engine.cli --config configs/example.json --dry-run
    python -m data_engine.cli --probe a.mp4
"""

import argparse
import json
import random
import sys
from typing import Any, Dict, List, Optional

from .engine import generate_transition_video, validate_shot_times, validate_transition_params
from .media import ProbeError, get_video_info
from .transitions import ALL_TRANSITIONS, LONG_TRANSITIONS, is_supported

CONFIG_KEYS = {
    "video_files",
    "save_dir",
    "eps",
    "shot_times",
    "transition_params",
    "min_shot_duration",
    "min_shot_interval",
    "max_transition_duration_ratio",
    "min_transition_duration_ratio",
    "fps",
    "threads",
    "video_id",
    "transition_types",
    "seed",
}

ENGINE_KEYS = (
    "eps",
    "shot_times",
    "transition_params",
    "min_shot_duration",
    "min_shot_interval",
    "max_transition_duration_ratio",
    "min_transition_duration_ratio",
    "fps",
    "threads",
)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return config


def _parse_transition_types(value: str) -> List[str]:
    if value == "all":
        return list(ALL_TRANSITIONS)
    if value == "long":
        return list(LONG_TRANSITIONS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("transition-types must not be empty")
    unsupported = [name for name in names if not is_supported(name)]
    if unsupported:
        raise ValueError(f"Unsupported transitions: {', '.join(unsupported)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data-engine",
        description="Compose videos with cut and xfade transitions (59 names).",
    )
    parser.add_argument("--config", help="Path to a JSON configuration file.")
    parser.add_argument("--video", action="append", default=None, help="Input video; repeat in shot order.")
    parser.add_argument("--output-dir", help="Directory for the composed video and JSON files.")
    parser.add_argument("--seed", type=int, help="Seed the standard random module before sampling.")
    parser.add_argument(
        "--transition-types",
        help="Comma-separated transition subset, or 'all' (default) / 'long' (58 gradual).",
    )
    parser.add_argument("--fps", type=float, help="Required input frame rate (default 25).")
    parser.add_argument("--threads", type=int, help="FFmpeg thread count per process.")
    parser.add_argument("--video-id", help="Output basename; defaults to a hash of the parameters.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and validate, print the plan as JSON, and do not run FFmpeg.",
    )
    parser.add_argument("--probe", metavar="FILE", help="Print duration and FPS for one file, then exit.")
    return parser


def _merged_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    if args.config:
        settings.update(_load_config(args.config))
    if args.video:
        settings["video_files"] = list(args.video)
    if args.output_dir:
        settings["save_dir"] = args.output_dir
    if args.seed is not None:
        settings["seed"] = args.seed
    if args.transition_types:
        settings["transition_types"] = _parse_transition_types(args.transition_types)
    if args.fps is not None:
        settings["fps"] = args.fps
    if args.threads is not None:
        settings["threads"] = args.threads
    if args.video_id is not None:
        settings["video_id"] = args.video_id
    return settings


def _dry_run_plan(settings: Dict[str, Any]) -> Dict[str, Any]:
    video_files = settings.get("video_files")
    if not video_files or len(video_files) < 2:
        raise ValueError("At least two input videos are required")
    seed = settings.get("seed")
    if seed is not None:
        random.seed(seed)
    fps = settings.get("fps", 25.0)
    eps = settings.get("eps", 0.01)
    shot_times = validate_shot_times(
        video_files,
        settings.get("shot_times"),
        settings.get("min_shot_duration", 2.0),
        fps,
        eps,
        do_round=True,
    )
    # Without re-encoding there are no trimmed files to probe; use the requested
    # window length as a stand-in. This approximation is documented in README.md
    # and only affects the dry-run preview, never the rendered result.
    shot_durations = [end - start for start, end in shot_times]
    transition_types = settings.get("transition_types") or list(ALL_TRANSITIONS)
    transition_params = validate_transition_params(
        shot_durations,
        settings.get("min_shot_interval", 0.5),
        settings.get("max_transition_duration_ratio", 0.95),
        settings.get("min_transition_duration_ratio", 0.01),
        fps,
        transition_types=transition_types,
        transition_params=settings.get("transition_params"),
    )
    return {
        "video_files": video_files,
        "fps": fps,
        "shot_times": shot_times,
        "transition_params": transition_params,
        "transition_types": transition_types,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.probe:
        try:
            duration, fps = get_video_info(args.probe)
        except ProbeError as exc:
            print(f"probe failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"path": args.probe, "duration": duration, "fps": fps}, indent=4))
        return 0

    try:
        settings = _merged_settings(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        try:
            plan = _dry_run_plan(settings)
        except (AssertionError, ProbeError, ValueError) as exc:
            print(f"plan validation failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(plan, indent=4))
        return 0

    save_dir = settings.pop("save_dir", None)
    if not save_dir:
        print("missing --output-dir (or save_dir in the config)", file=sys.stderr)
        return 2
    video_files = settings.pop("video_files", None)
    if not video_files or len(video_files) < 2:
        print("at least two --video inputs (or video_files in the config) are required", file=sys.stderr)
        return 2

    seed = settings.pop("seed", None)
    if seed is not None:
        random.seed(seed)
    video_id = settings.pop("video_id", None)
    transition_types = settings.pop("transition_types", None)
    kwargs = {key: settings[key] for key in ENGINE_KEYS if key in settings}

    ok = generate_transition_video(
        video_files,
        save_dir,
        video_id_str=video_id,
        transition_types=transition_types,
        **kwargs,
    )
    if not ok:
        print("synthesis failed", file=sys.stderr)
        return 1
    print(f"wrote output to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
