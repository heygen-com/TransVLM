"""CPU smoke tests that exercise real FFmpeg rendering through the CLI.

Two small colour clips with tone audio are generated inside a temporary
directory; nothing is written to the repository and no media is committed.
The fixture requests one libx264 encoding thread and one filter thread, and the
engine is called with ``threads=1``. These are thread-count requests; they are
not a hard cap on every internal thread FFmpeg or x264 may create.

Skipped (not failed) when ``ffmpeg`` or ``ffprobe`` is unavailable.

Run:
    python -m unittest data_engine.tests.test_ffmpeg_smoke -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(HERE))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from data_engine.cli import main  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_clip(path, color, seconds=1.2):
    cmd = [
        "ffmpeg",
        "-threads", "1",
        "-filter_threads", "1",
        "-y",
        "-f", "lavfi",
        "-i", f"color=c={color}:s=320x240:r=25:d={seconds}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
        # Output-side encoder thread request (the -threads above precedes -i and
        # is an input option, so it does not limit the libx264 encoder threads).
        "-threads", "1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        path,
    ]
    subprocess.run(
        cmd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout.decode("utf-8"))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg and ffprobe are required for rendering smoke tests")
class CliRenderSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        cls.inputs = os.path.join(cls.root, "inputs")
        os.makedirs(cls.inputs)
        cls.clip_a = os.path.join(cls.inputs, "clip_a.mp4")
        cls.clip_b = os.path.join(cls.inputs, "clip_b.mp4")
        _make_clip(cls.clip_a, "red")
        _make_clip(cls.clip_b, "blue")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _render(self, name, pool):
        outdir = os.path.join(self.root, "outputs", name)
        code = main(
            [
                "--video", self.clip_a,
                "--video", self.clip_b,
                "--output-dir", outdir,
                "--seed", "3",
                "--threads", "1",
                "--fps", "25",
                "--transition-types", pool,
                "--video-id", name,
            ]
        )
        self.assertEqual(code, 0)
        return outdir

    def _assert_outputs(self, outdir, name, expected_type):
        video = os.path.join(outdir, name + ".mp4")
        labels_path = os.path.join(outdir, name + ".json")
        params_path = os.path.join(outdir, name + "_params.json")
        for path in (video, labels_path, params_path):
            self.assertTrue(os.path.exists(path), path)

        streams = _probe(video)["streams"]
        self.assertTrue(any(s["codec_type"] == "video" for s in streams))
        self.assertTrue(any(s["codec_type"] == "audio" for s in streams))

        with open(labels_path, "r", encoding="utf-8") as handle:
            labels = json.load(handle)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["transition_type"], expected_type)
        span = labels[0]["end_time"] - labels[0]["start_time"]
        if expected_type == "cut":
            self.assertEqual(span, 0)
        else:
            self.assertGreater(span, 0)

        with open(params_path, "r", encoding="utf-8") as handle:
            params = json.load(handle)
        self.assertEqual([item["transition"] for item in params["transition_params"]], [expected_type])

    def test_cli_cut_renders(self):
        outdir = self._render("smoke_cut", "cut")
        self._assert_outputs(outdir, "smoke_cut", "cut")

    def test_cli_fade_renders(self):
        outdir = self._render("smoke_fade", "fade")
        self._assert_outputs(outdir, "smoke_fade", "fade")


if __name__ == "__main__":
    unittest.main()
