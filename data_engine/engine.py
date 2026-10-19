"""Synthesis core for shot-transition videos.

The sampling logic, rounding, FFmpeg filters and encoder settings here follow
the historical implementation used for the paper. See README.md for the
documented differences from the paper text; do not "fix" them silently.
"""

import hashlib
import os
import os.path as osp
import random
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .media import get_video_info, save_json_file
from .transitions import ALL_TRANSITIONS


def round_with_floor(x: float, decimal_places: int = 3) -> float:
    """Floor-round to *decimal_places* digits, as the historical engine did."""
    return round(float(np.floor(x * 10**decimal_places) / 10**decimal_places), decimal_places)


def check_shot_time(shot_time: Tuple[float, float], video_duration: float) -> bool:
    start_time, end_time = shot_time
    assert start_time >= 0, f"Start time must be >= 0: {start_time}"
    assert end_time > start_time, f"End time must be greater than start time: {end_time} <= {start_time}"
    assert end_time <= video_duration, f"End time exceeds video duration: {end_time} > {video_duration}"
    return True


def generate_shot_time(
    video_duration: float, min_shot_duration: float, do_round: bool = False
) -> Tuple[float, float]:
    assert video_duration > 0, f"Video duration must be > 0: {video_duration}"
    assert min_shot_duration > 0, f"Minimum shot duration must be > 0: {min_shot_duration}"
    if video_duration < min_shot_duration:
        selected_shot_duration = video_duration
        start_time = 0.0
    else:
        selected_shot_duration = random.uniform(min_shot_duration, video_duration)
        max_start_time = video_duration - selected_shot_duration
        start_time = random.uniform(0, max_start_time)
    end_time = start_time + selected_shot_duration
    if do_round:
        start_time = round_with_floor(start_time, 3)
        end_time = round_with_floor(end_time, 3)
    return (start_time, end_time)


def check_video_fps(video_fps: float, fps: float, eps: float = 0.01) -> bool:
    assert abs(video_fps - fps) < eps, f"Video FPS mismatch: {video_fps} != {fps}"
    return True


def validate_shot_times(
    video_files: List[str],
    shot_times: Optional[List[Tuple[float, float]]] = None,
    min_shot_duration: float = 2.0,
    fps: float = 25,
    eps: float = 0.01,
    do_round: bool = False,
) -> List[Tuple[float, float]]:
    """Validate provided shot times or sample them with the configured RNG."""
    if shot_times is None:
        shot_times = []
        for video_file in video_files:
            video_duration, video_fps = get_video_info(video_file)
            assert check_video_fps(video_fps, fps, eps), f"Video FPS mismatch: {video_fps} != {fps}"
            shot_time = generate_shot_time(video_duration, min_shot_duration, do_round)
            assert check_shot_time(shot_time, video_duration), f"Invalid shot_time: {shot_time}"
            shot_times.append(shot_time)
    else:
        for video_file, shot_time in zip(video_files, shot_times):
            video_duration, video_fps = get_video_info(video_file)
            assert check_video_fps(video_fps, fps, eps), f"Video FPS mismatch: {video_fps} != {fps}"
            assert check_shot_time(shot_time, video_duration), f"Invalid shot_time: {shot_time}"

    assert len(shot_times) == len(
        video_files
    ), f"shot_times length ({len(shot_times)}) does not match video_files length ({len(video_files)})"
    return shot_times


def trim_video_ffmpeg(
    input_video_path: str,
    output_video_path: str,
    start_time: float,
    end_time: float,
    threads: int = 8,
    vfilter: Optional[str] = None,
) -> bool:
    """Trim a clip and re-encode it with the historical encoder settings."""
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-threads",
        str(threads),
        "-filter_threads",
        str(threads),
        "-y",
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-i",
        str(input_video_path),
    ]

    if vfilter is not None:
        cmd.append("-filter:v")
        cmd.append(vfilter)

    cmd = cmd + [
        "-c:v",
        "libx264",
        "-x264-params",
        f"threads={threads}",
        "-crf",
        "17",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg trim failed with return code {exc.returncode}")
        return False
    except Exception as exc:  # noqa: BLE001 - preserve historical failure contract
        print(f"Error running FFmpeg trim: {exc}")
        return False


def generate_transition_duration_and_offset(
    shot_duration0: float,
    shot_duration1: float,
    prev_transition_duration: float,
    prev_transition_offset: float,
    min_shot_interval: float,
    max_transition_duration_ratio: float,
    min_transition_duration_ratio: float,
    fps: float,
    do_round: bool = False,
) -> Tuple[float, float]:
    max_shot_overlap = (
        min(shot_duration0 - prev_transition_duration, shot_duration1) - min_shot_interval
    )
    max_transition_duration = max_transition_duration_ratio * max_shot_overlap
    min_transition_duration = min_transition_duration_ratio * max_shot_overlap
    min_transition_duration = max(3.0 / fps, min_transition_duration)
    transition_duration = random.uniform(min_transition_duration, max_transition_duration)
    transition_offset = prev_transition_offset + shot_duration0 - transition_duration
    if do_round:
        transition_duration = round_with_floor(transition_duration, 3)
        transition_offset = round_with_floor(transition_offset, 3)
    return (transition_duration, transition_offset)


def generate_transition_type(transition_types: List[str]) -> str:
    return random.choice(transition_types)


def generate_transition_params(
    shot_duration0: float,
    shot_duration1: float,
    prev_transition_duration: float,
    prev_transition_offset: float,
    min_shot_interval: float,
    max_transition_duration_ratio: float,
    min_transition_duration_ratio: float,
    fps: float,
    transition_types: List[str],
    transition_type: Optional[str] = None,
    do_round: bool = False,
) -> Dict[str, Any]:
    if transition_type is None:
        transition_type = generate_transition_type(transition_types)
    else:
        assert transition_type in transition_types, f"Invalid transition_type: {transition_type}"
    if transition_type == "cut":
        transition_duration = 0.0
        transition_offset = prev_transition_offset + shot_duration0
    else:
        transition_duration, transition_offset = generate_transition_duration_and_offset(
            shot_duration0,
            shot_duration1,
            prev_transition_duration,
            prev_transition_offset,
            min_shot_interval,
            max_transition_duration_ratio,
            min_transition_duration_ratio,
            fps,
            do_round,
        )
    return {
        "transition": transition_type,
        "duration": transition_duration,
        "offset": transition_offset,
    }


def check_transition_params(
    transition_params: Dict[str, Any],
    fps: float,
    transition_types: List[str],
) -> bool:
    transition_duration = transition_params["duration"]
    transition_offset = transition_params["offset"]
    transition_type = transition_params["transition"]
    assert transition_type in transition_types, f"Invalid transition_type: {transition_type}"
    if transition_type == "cut":
        assert transition_duration == 0.0, f"transition_duration must be 0.0: {transition_duration}"
    else:
        assert transition_duration > 1 / fps, f"transition_duration must exceed 1 / fps: {transition_duration}"
    assert transition_offset >= 0, f"transition_offset must be >= 0: {transition_offset}"
    return True


def validate_transition_params(
    shot_durations: List[float],
    min_shot_interval: float,
    max_transition_duration_ratio: float,
    min_transition_duration_ratio: float,
    fps: float,
    transition_types: List[str],
    transition_params: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Validate provided transition params or sample them."""
    if transition_params is None:
        prev_transition_duration = 0.0
        prev_transition_offset = 0.0
        transition_params = []
        for i in range(len(shot_durations) - 1):
            shot_duration0 = shot_durations[i]
            shot_duration1 = shot_durations[i + 1]
            transition_param = generate_transition_params(
                shot_duration0,
                shot_duration1,
                prev_transition_duration,
                prev_transition_offset,
                min_shot_interval,
                max_transition_duration_ratio,
                min_transition_duration_ratio,
                fps,
                transition_types,
                do_round=False,
            )
            prev_transition_duration = transition_param["duration"]
            prev_transition_offset = transition_param["offset"]
            transition_params.append(transition_param)
    else:
        prev_transition_duration = 0.0
        prev_transition_offset = 0.0
        transition_params_ = []
        for i in range(len(shot_durations) - 1):
            transition_param = transition_params[i]
            assert "transition" in transition_param, f"transition_param must contain transition: {transition_param}"
            if "duration" in transition_param and "offset" in transition_param:
                transition_params_.append(transition_param)
            else:
                shot_duration0 = shot_durations[i]
                shot_duration1 = shot_durations[i + 1]
                transition_param = generate_transition_params(
                    shot_duration0,
                    shot_duration1,
                    prev_transition_duration,
                    prev_transition_offset,
                    min_shot_interval,
                    max_transition_duration_ratio,
                    min_transition_duration_ratio,
                    fps,
                    transition_types,
                    transition_param["transition"],
                    do_round=True,
                )
                transition_params_.append(transition_param)
            prev_transition_duration = transition_param["duration"]
            prev_transition_offset = transition_param["offset"]
        transition_params = transition_params_
    for transition_param in transition_params:
        transition_param["duration"] = round_with_floor(transition_param["duration"], 3)
        transition_param["offset"] = round_with_floor(transition_param["offset"], 3)
        assert check_transition_params(
            transition_param, fps, transition_types
        ), f"Invalid transition_param: {transition_param}"
    assert len(transition_params) == len(
        shot_durations
    ) - 1, (
        f"transition_params length ({len(transition_params)}) does not match "
        f"shot_durations length - 1 ({len(shot_durations) - 1})"
    )
    return transition_params


def check_video_duration(video_duration: float, target_duration: float, eps: float = 0.01) -> bool:
    assert abs(video_duration - target_duration) < eps, (
        f"Video duration mismatch: {video_duration} != {target_duration}"
    )
    return True


def generate_concat_ffmpeg(
    video_file1: str,
    video_file2: str,
    output_video_path: str,
    threads: int = 8,
) -> bool:
    """Concatenate two clips without a transition."""
    if os.path.dirname(output_video_path):
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    filter_complex = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,fps=25,format=yuv420p,setsar=1[v0];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS,fps=25,format=yuv420p,setsar=1[v1];"
        "[0:a]asetpts=PTS-STARTPTS[a0];"
        "[1:a]asetpts=PTS-STARTPTS[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )

    cmd = [
        "ffmpeg",
        "-threads",
        str(threads),
        "-filter_threads",
        str(threads),
        "-y",
        "-i",
        str(video_file1),
        "-i",
        str(video_file2),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-x264-params",
        f"threads={threads}",
        "-crf",
        "17",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg concat failed with return code {exc.returncode}")
        return False
    except Exception as exc:  # noqa: BLE001 - preserve historical failure contract
        print(f"Error running FFmpeg concat: {exc}")
        return False


def generate_transition_ffmpeg(
    video_file1: str,
    video_file2: str,
    output_video_path: str,
    transition: str = "fade",
    duration: float = 1.0,
    offset: float = 1.0,
    threads: int = 8,
) -> bool:
    """Join two clips with an FFmpeg xfade video transition and audio crossfade."""
    if os.path.dirname(output_video_path):
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    filter_complex = (
        "[0:v]settb=AVTB,setpts=PTS-STARTPTS,fps=25,format=yuv420p,setsar=1[v0];"
        "[1:v]settb=AVTB,setpts=PTS-STARTPTS,fps=25,format=yuv420p,setsar=1[v1];"
        "[0:a]asetpts=PTS-STARTPTS[a0];"
        "[1:a]asetpts=PTS-STARTPTS[a1];"
        f"[v0][v1]xfade=transition={transition}:duration={duration}:offset={offset}[vxf];"
        f"[a0][a1]acrossfade=d={duration}[axf]"
    )

    cmd = [
        "ffmpeg",
        "-threads",
        str(threads),
        "-filter_threads",
        str(threads),
        "-y",
        "-i",
        str(video_file1),
        "-i",
        str(video_file2),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vxf]",
        "-map",
        "[axf]",
        "-c:v",
        "libx264",
        "-x264-params",
        f"threads={threads}",
        "-crf",
        "17",
        "-preset",
        "fast",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_video_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg transition failed with return code {exc.returncode}")
        return False
    except Exception as exc:  # noqa: BLE001 - preserve historical failure contract
        print(f"Error running FFmpeg transition: {exc}")
        return False


def generate_transition_video(
    video_files: List[str],
    save_dir: str,
    eps: float = 0.01,
    shot_times: Optional[List[Tuple[float, float]]] = None,
    transition_params: Optional[List[Dict[str, Any]]] = None,
    min_shot_duration: float = 2.0,
    min_shot_interval: float = 0.5,
    max_transition_duration_ratio: float = 0.95,
    min_transition_duration_ratio: float = 0.01,
    fps: float = 25.0,
    threads: int = 8,
    video_id_str: Optional[str] = None,
    transition_types: Optional[List[str]] = None,
) -> bool:
    """Generate one composed video plus label and parameter JSON files.

    Returns ``True`` on success and ``False`` on any failure, matching the
    historical contract. Random draws occur in the original order: shot times,
    then per-transition type, then duration (non-cut only).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            assert len(video_files) > 1, f"video_files length must exceed 1: {len(video_files)}"
            timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
            temp_dir = Path(temp_dir)

            shot_times = validate_shot_times(
                video_files, shot_times, min_shot_duration, fps, eps, do_round=True
            )

            shot_files = []
            for shot_index, video_file, shot_time in zip(range(len(video_files)), video_files, shot_times):
                video_name = osp.splitext(osp.basename(video_file))[0]
                shot_file = temp_dir / f"{timestamp_str}_{shot_index}_{video_name}.mp4"
                if shot_file.exists():
                    shot_files.append(str(shot_file))
                else:
                    trim_video_ffmpeg(video_file, str(shot_file), shot_time[0], shot_time[1], threads)
                    shot_files.append(str(shot_file))

            shot_durations = []
            for shot_file in shot_files:
                assert osp.exists(shot_file), f"shot_file does not exist: {shot_file}"
                shot_duration, shot_fps = get_video_info(shot_file)
                assert check_video_fps(shot_fps, fps, eps), f"shot_fps mismatch: {shot_fps} != {fps}"
                shot_durations.append(shot_duration)

            if transition_types is None:
                transition_types = ALL_TRANSITIONS

            transition_params = validate_transition_params(
                shot_durations,
                min_shot_interval,
                max_transition_duration_ratio,
                min_transition_duration_ratio,
                fps,
                transition_types=transition_types,
                transition_params=transition_params,
            )

            prev_segment_video_path = shot_files[0]
            prev_segment_video_duration, prev_segment_video_fps = get_video_info(
                str(prev_segment_video_path)
            )
            assert check_video_fps(
                prev_segment_video_fps, fps, eps
            ), f"prev_segment_video_fps mismatch: {prev_segment_video_fps} != {fps}"
            segment_video_paths = []
            transition_params_ = []
            for shot_file, transition_index, transition_param in zip(
                shot_files[1:], range(len(transition_params)), transition_params
            ):
                transition_type = transition_param["transition"]
                transition_duration = transition_param["duration"]
                transition_offset = prev_segment_video_duration - transition_duration
                transition_offset = round_with_floor(transition_offset, 3)
                transition_params_.append(
                    {
                        "transition": transition_type,
                        "duration": transition_duration,
                        "offset": transition_offset,
                    }
                )
                output_segment_video_path = (
                    temp_dir / f"{timestamp_str}_transition_{transition_index}.mp4"
                )
                if transition_type == "cut":
                    flag = generate_concat_ffmpeg(
                        str(prev_segment_video_path),
                        str(shot_file),
                        str(output_segment_video_path),
                        threads=threads,
                    )
                else:
                    flag = generate_transition_ffmpeg(
                        str(prev_segment_video_path),
                        str(shot_file),
                        str(output_segment_video_path),
                        transition_type,
                        transition_duration,
                        transition_offset,
                        threads=threads,
                    )
                assert flag, f"Failed to generate transition video: {output_segment_video_path}"
                output_segment_video_duration, output_segment_video_fps = get_video_info(
                    str(output_segment_video_path)
                )
                assert check_video_fps(
                    output_segment_video_fps, fps, eps
                ), f"output_segment_video_fps mismatch: {output_segment_video_fps} != {fps}"
                prev_segment_video_duration = output_segment_video_duration
                prev_segment_video_path = output_segment_video_path
                segment_video_paths.append(output_segment_video_path)

            assert len(transition_params) == len(
                transition_params_
            ), f"transition_params length mismatch: {len(transition_params)} != {len(transition_params_)}"
            assert len(segment_video_paths) == len(
                transition_params
            ), f"segment_video_paths length mismatch: {len(segment_video_paths)} != {len(transition_params)}"
            for segment_video_path in segment_video_paths:
                assert osp.exists(segment_video_path), f"segment_video_path does not exist: {segment_video_path}"

            transition_params = transition_params_
            transition_labels = []
            for transition_param in transition_params:
                duration = transition_param["duration"]
                offset = transition_param["offset"]
                transition_type = transition_param["transition"]
                start_time = offset
                end_time = start_time + duration
                transition_labels.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "transition_type": transition_type,
                    }
                )

            params_data = {
                "video_files": video_files,
                "save_dir": save_dir,
                "eps": eps,
                "shot_times": shot_times,
                "transition_params": transition_params,
                "min_shot_duration": min_shot_duration,
                "min_shot_interval": min_shot_interval,
                "max_transition_duration_ratio": max_transition_duration_ratio,
                "min_transition_duration_ratio": min_transition_duration_ratio,
                "fps": fps,
                "threads": threads,
            }

            if video_id_str is None:
                video_id_str = str(params_data)
                video_id = hashlib.sha256(video_id_str.encode()).hexdigest()[:8]
            else:
                video_id = video_id_str

            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

            output_video = save_dir_path / f"{video_id}.mp4"
            output_json = save_dir_path / f"{video_id}.json"
            output_params = save_dir_path / f"{video_id}_params.json"

            shutil.copy(str(segment_video_paths[-1]), str(output_video))
            save_json_file(str(output_json), transition_labels)
            save_json_file(str(output_params), params_data)
        except Exception as exc:  # noqa: BLE001 - preserve historical failure contract
            print(f"Error generating transition video: {exc}")
            return False

    return True
