"""Container probing and JSON output helpers.

This module intentionally depends only on the Python standard library,
``numpy`` (for numpy-aware JSON conversion) and the ``ffprobe`` binary. It
does not import any internal or third-party video-library helper.
"""

import json
import subprocess
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class ProbeError(ValueError):
    """Raised when a media file cannot be probed or lacks required metadata."""


def parse_frame_rate(frame_rate: Optional[str]) -> Optional[float]:
    """Parse an ffprobe rational frame rate such as ``"30000/1001"``.

    Returns ``None`` for missing or degenerate rates, matching the historical
    helper. Raises ``ValueError`` if the value is not a ``num/den`` rational.
    """
    if frame_rate is None:
        return None
    try:
        num, den = frame_rate.split("/")
        if num == "0" or den == "0":
            return None
        return float(Decimal(num) / Decimal(den))
    except (ValueError, ArithmeticError) as exc:
        raise ValueError(f"Malformed frame rate: {frame_rate!r}") from exc


def get_video_info_from_probe(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the video stream from an ffprobe dictionary.

    A single video stream is returned directly. When several exist, the stream
    whose ``disposition.default`` flag is highest wins; ties keep probe order,
    which reproduces the historical selection semantics.
    """
    video_streams = []  # [(default flag, stream)]
    for stream in probe["streams"]:
        if stream["codec_type"] == "video":
            video_streams.append([stream.get("disposition", {}).get("default", 0), stream])
    if len(video_streams) == 1:
        return video_streams[0][1]
    if len(video_streams) > 1:
        return sorted(video_streams, key=lambda item: item[0], reverse=True)[0][1]
    return None


def get_duration_attribute(stream_info: Dict[str, Any], format_info: Dict[str, Any]) -> float:
    """Return stream duration if present, otherwise container duration."""
    if "duration" in stream_info:
        return float(stream_info["duration"])
    if "duration" in format_info:
        return float(format_info["duration"])
    raise ProbeError("No duration attribute")


def get_video_fps(video_info: Dict[str, Any]) -> Optional[float]:
    """Return average frame rate, falling back to the time base rate."""
    avg_fps = parse_frame_rate(video_info.get("avg_frame_rate"))
    tbr_fps = parse_frame_rate(video_info.get("r_frame_rate"))
    return avg_fps or tbr_fps


def probe_video(video_path: str) -> Dict[str, Any]:
    """Run ``ffprobe`` and return the parsed JSON document."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise ProbeError(f"ffprobe failed for {video_path}: {detail}") from exc
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {video_path}") from exc


def get_video_info(video_path: str) -> Tuple[float, float]:
    """Return ``(duration_seconds, fps)`` for a media file."""
    probe = probe_video(video_path)
    video_info = get_video_info_from_probe(probe)
    if video_info is None:
        raise ProbeError(f"No video stream found in {video_path}")
    fps = get_video_fps(video_info)
    if fps is None:
        raise ProbeError(f"Failed to extract FPS info from video: {video_path}")
    duration = get_duration_attribute(video_info, probe["format"])
    return duration, float(fps)


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars and arrays to plain Python containers."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def save_json_file(path: str, data: Any, indent: int = 4, ensure_ascii: bool = False) -> None:
    """Write JSON with numpy-aware conversion.

    The historical engine wrote four-space indented JSON and allowed raw
    non-ASCII characters. Numpy values are converted explicitly through their
    public ``item`` / ``tolist`` interfaces before encoding.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_builtin(data), handle, indent=indent, ensure_ascii=ensure_ascii)


# Alias kept for callers that used the old helper's exported name.
save_json = save_json_file
