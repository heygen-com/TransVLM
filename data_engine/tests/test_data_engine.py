"""Unit tests for sampling, probe parsing and JSON helpers.

These tests never invoke FFmpeg or ffprobe. They run with the standard
library runner:

    python -m unittest discover -s data_engine/tests -v
"""

import json
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_engine import media  # noqa: E402
from data_engine import transitions  # noqa: E402
from data_engine import engine  # noqa: E402
from data_engine.cli import main, _parse_transition_types, _load_config  # noqa: E402


class TransitionTableTest(unittest.TestCase):
    def test_counts_and_order(self):
        self.assertEqual(len(transitions.LONG_TRANSITIONS), 58)
        self.assertEqual(len(transitions.ALL_TRANSITIONS), 59)
        self.assertEqual(len(set(transitions.ALL_TRANSITIONS)), 59)
        self.assertEqual(transitions.ALL_TRANSITIONS[0], "cut")
        self.assertEqual(transitions.LONG_TRANSITIONS[0], "fade")
        self.assertEqual(transitions.LONG_TRANSITIONS[-1], "revealdown")
        self.assertNotIn("cut", transitions.LONG_TRANSITIONS)

    def test_mapping(self):
        self.assertEqual(transitions.XFRAME_MAPPING["cut"], "concat")
        self.assertEqual(transitions.xfade_name("fade"), "fade")
        self.assertEqual(transitions.xfade_name("revealdown"), "revealdown")
        with self.assertRaises(ValueError):
            transitions.xfade_name("cut")
        with self.assertRaises(ValueError):
            transitions.xfade_name("not-a-transition")

    def test_all_names_are_xfade_names_except_cut(self):
        for name in transitions.LONG_TRANSITIONS:
            self.assertEqual(transitions.TRANSITION_KIND[name], "xfade")


class FrameRateTest(unittest.TestCase):
    def test_rational_rates(self):
        self.assertAlmostEqual(media.parse_frame_rate("25/1"), 25.0)
        self.assertAlmostEqual(media.parse_frame_rate("30000/1001"), 30000 / 1001)
        self.assertEqual(media.parse_frame_rate("0/0"), None)
        self.assertEqual(media.parse_frame_rate("0/1"), None)
        self.assertEqual(media.parse_frame_rate("25/0"), None)
        self.assertEqual(media.parse_frame_rate(None), None)

    def test_degenerate_rates(self):
        with self.assertRaises(ValueError):
            media.parse_frame_rate("25")
        with self.assertRaises(ValueError):
            media.parse_frame_rate("abc/def")

    def test_fps_prefers_average(self):
        info = {"avg_frame_rate": "24000/1001", "r_frame_rate": "25/1"}
        self.assertAlmostEqual(media.get_video_fps(info), 24000 / 1001)
        info = {"avg_frame_rate": "0/0", "r_frame_rate": "25/1"}
        self.assertAlmostEqual(media.get_video_fps(info), 25.0)
        self.assertIsNone(media.get_video_fps({"avg_frame_rate": "0/0"}))


class ProbeSelectionTest(unittest.TestCase):
    def test_single_stream(self):
        stream = {"codec_type": "video", "index": 0}
        probe = {"streams": [{"codec_type": "audio"}, stream]}
        self.assertIs(media.get_video_info_from_probe(probe), stream)

    def test_prefers_default_stream(self):
        first = {"codec_type": "video", "disposition": {"default": 0}, "index": 0}
        second = {"codec_type": "video", "disposition": {"default": 1}, "index": 3}
        probe = {"streams": [first, second]}
        self.assertIs(media.get_video_info_from_probe(probe), second)

    def test_ties_keep_probe_order(self):
        first = {"codec_type": "video", "disposition": {"default": 1}, "index": 0}
        second = {"codec_type": "video", "disposition": {"default": 1}, "index": 1}
        probe = {"streams": [first, second]}
        self.assertIs(media.get_video_info_from_probe(probe), first)

    def test_no_video_stream(self):
        probe = {"streams": [{"codec_type": "audio"}]}
        self.assertIsNone(media.get_video_info_from_probe(probe))

    def test_duration_fallbacks(self):
        self.assertEqual(media.get_duration_attribute({"duration": "3.5"}, {"duration": "9"}), 3.5)
        self.assertEqual(media.get_duration_attribute({}, {"duration": "9"}), 9.0)
        with self.assertRaises(media.ProbeError):
            media.get_duration_attribute({}, {})


class JsonHelperTest(unittest.TestCase):
    def test_numpy_scalars_and_arrays(self):
        payload = {
            "int": np.int64(3),
            "float": np.float64(1.25),
            "array": np.array([1, 2, 3], dtype=np.int32),
            "nested": [np.float32(0.5), (np.int32(7),)],
            "python": {"ok": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.json")
            media.save_json_file(path, payload)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            loaded = json.loads(text)
        self.assertEqual(loaded["int"], 3)
        self.assertEqual(loaded["float"], 1.25)
        self.assertEqual(loaded["array"], [1, 2, 3])
        self.assertEqual(loaded["nested"], [0.5, [7]])
        self.assertIsInstance(loaded["int"], int)
        self.assertIsInstance(loaded["float"], float)
        # Four-space indent and non-ASCII preservation are historical behaviour.
        self.assertIn('\n    "int": 3', text)

    def test_non_ascii_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.json")
            media.save_json_file(path, {"text": "café"})
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        self.assertIn("café", text)
        self.assertNotIn("\\u00e9", text)

    def test_unicode_escape_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.json")
            media.save_json_file(path, {"text": "café"}, ensure_ascii=True)
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        self.assertIn("\\u00e9", text)
        self.assertNotIn("café", text)


class RoundingTest(unittest.TestCase):
    def test_floor_rounding(self):
        self.assertEqual(engine.round_with_floor(1.2349, 3), 1.234)
        self.assertEqual(engine.round_with_floor(1.2, 3), 1.2)
        self.assertEqual(engine.round_with_floor(0.0009, 3), 0.0)


class SamplingTest(unittest.TestCase):
    def test_cut_is_zero_duration(self):
        params = engine.generate_transition_params(
            3.0, 3.0, 0.0, 0.0, 0.5, 0.95, 0.01, 25.0, ["cut"], transition_type="cut"
        )
        self.assertEqual(params["transition"], "cut")
        self.assertEqual(params["duration"], 0.0)
        self.assertEqual(params["offset"], 3.0)

    def test_duration_bounds_and_reproducibility(self):
        kwargs = dict(
            shot_durations=[3.0, 3.0, 3.0],
            min_shot_interval=0.5,
            max_transition_duration_ratio=0.95,
            min_transition_duration_ratio=0.01,
            fps=25.0,
            transition_types=["fade"],
        )
        random.seed(11)
        first = engine.validate_transition_params(**kwargs)
        random.seed(11)
        second = engine.validate_transition_params(**kwargs)
        self.assertEqual(first, second)
        overlap = min(3.0 - 0.0, 3.0) - 0.5
        for index, param in enumerate(first):
            lower = max(3.0 / 25.0, 0.01 * (overlap if index == 0 else min(3.0 - first[index - 1]["duration"], 3.0) - 0.5))
            self.assertGreaterEqual(param["duration"], round(lower, 3))
            self.assertLessEqual(param["duration"], round(0.95 * (overlap if index == 0 else min(3.0 - first[index - 1]["duration"], 3.0) - 0.5), 3))
            self.assertGreaterEqual(param["offset"], 0.0)

    def test_fixed_transition_type_still_samples_duration(self):
        random.seed(3)
        params = engine.validate_transition_params(
            [3.0, 3.0],
            0.5,
            0.95,
            0.01,
            25.0,
            ["fade"],
            transition_params=[{"transition": "fade"}],
        )
        self.assertEqual(params[0]["transition"], "fade")
        self.assertGreater(params[0]["duration"], 1 / 25.0)

    def test_provided_params_are_respected_and_rounded(self):
        params = engine.validate_transition_params(
            [3.0, 3.0],
            0.5,
            0.95,
            0.01,
            25.0,
            ["cut"],
            transition_params=[{"transition": "cut", "duration": 0.0, "offset": 2.9999}],
        )
        self.assertEqual(params, [{"transition": "cut", "duration": 0.0, "offset": 2.999}])

    def test_invalid_provided_params_raise(self):
        with self.assertRaises(AssertionError):
            engine.validate_transition_params(
                [3.0, 3.0], 0.5, 0.95, 0.01, 25.0, ["cut"],
                transition_params=[{"transition": "fade", "duration": 0.5, "offset": 1.0}],
            )


class DegenerateInputTest(unittest.TestCase):
    def test_video_shorter_than_minimum_uses_whole_video(self):
        start, end = engine.generate_shot_time(1.0, 2.0, do_round=True)
        self.assertEqual((start, end), (0.0, 1.0))

    def test_non_positive_duration_rejected(self):
        with self.assertRaises(AssertionError):
            engine.generate_shot_time(0.0, 2.0)
        with self.assertRaises(AssertionError):
            engine.generate_shot_time(5.0, 0.0)

    def test_check_shot_time_rejects_invalid_windows(self):
        with self.assertRaises(AssertionError):
            engine.check_shot_time((-1.0, 2.0), 5.0)
        with self.assertRaises(AssertionError):
            engine.check_shot_time((3.0, 3.0), 5.0)
        with self.assertRaises(AssertionError):
            engine.check_shot_time((1.0, 6.0), 5.0)

    def test_fps_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            engine.check_video_fps(30.0, 25.0)
        self.assertTrue(engine.check_video_fps(25.005, 25.0))

    def test_single_video_is_rejected_by_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.mp4")
            self.assertFalse(engine.generate_transition_video([missing], tmp))

    def test_missing_input_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(
                engine.generate_transition_video(
                    [os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")], tmp
                )
            )


class CliConfigTest(unittest.TestCase):
    def test_transition_type_selection(self):
        self.assertEqual(len(_parse_transition_types("all")), 59)
        self.assertEqual(len(_parse_transition_types("long")), 58)
        self.assertEqual(_parse_transition_types("cut, fade"), ["cut", "fade"])
        with self.assertRaises(ValueError):
            _parse_transition_types("cut,bogus")
        with self.assertRaises(ValueError):
            _parse_transition_types(" ")

    def test_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"video_files": [], "surprise": 1}, handle)
            with self.assertRaises(ValueError):
                _load_config(path)

    def test_example_config_is_valid(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config = _load_config(os.path.join(here, "configs", "example.json"))
        self.assertGreaterEqual(len(config["video_files"]), 2)
        self.assertNotIn("/", config["video_id"])


class CliMainWiringTest(unittest.TestCase):
    """Regression: the CLI must pass the transition pool to the engine."""

    def _run(self, extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("data_engine.cli.generate_transition_video", return_value=True) as fake:
                code = main(
                    [
                        "--video", os.path.join(tmp, "a.mp4"),
                        "--video", os.path.join(tmp, "b.mp4"),
                        "--output-dir", os.path.join(tmp, "out"),
                        "--seed", "7",
                    ]
                    + extra_args
                )
        self.assertEqual(code, 0)
        return fake.call_args

    def test_main_forwards_cut_pool(self):
        args, kwargs = self._run(["--transition-types", "cut"])
        self.assertEqual(kwargs["transition_types"], ["cut"])

    def test_main_forwards_long_pool(self):
        args, kwargs = self._run(["--transition-types", "long"])
        self.assertEqual(len(kwargs["transition_types"]), 58)
        self.assertNotIn("cut", kwargs["transition_types"])

    def test_main_forwards_fade_pool(self):
        args, kwargs = self._run(["--transition-types", "fade"])
        self.assertEqual(kwargs["transition_types"], ["fade"])

    def test_main_defaults_to_none_for_engine_default(self):
        args, kwargs = self._run([])
        self.assertIn("transition_types", kwargs)
        self.assertIsNone(kwargs["transition_types"])

    def test_main_forwards_config_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "cfg.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "video_files": [os.path.join(tmp, "a.mp4"), os.path.join(tmp, "b.mp4")],
                        "save_dir": os.path.join(tmp, "out"),
                        "transition_types": ["cut", "fade"],
                        "seed": 1,
                    },
                    handle,
                )
            with mock.patch("data_engine.cli.generate_transition_video", return_value=True) as fake:
                code = main(["--config", config_path])
        self.assertEqual(code, 0)
        self.assertEqual(fake.call_args.kwargs["transition_types"], ["cut", "fade"])


if __name__ == "__main__":
    unittest.main()
