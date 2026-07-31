# TransVLM inference

Shot transition detection. Takes a video, returns the time spans where transitions happen.

TransVLM is Qwen3-VL-4B-Instruct with the vision patch embed widened from 3 to 6 channels, so
optical flow goes in alongside RGB at the input stage. Motion is what distinguishes a dissolve from
a camera pan, and a model that only sees appearance has to infer it. The flow is computed for you at
inference time; you supply a video.

## Install

Python 3.12, a CUDA GPU, and `ffmpeg` on `PATH`. We use [uv](https://docs.astral.sh/uv/), but any
tool that reads `pyproject.toml` works.

```bash
cd inference
uv venv && source .venv/bin/activate
uv sync --group cu130 --group dev        # cu128 if your driver is older than 570
```

Then pick a backend. The HuggingFace backend is the default and needs nothing extra, so if you just
want predictions you can stop here.

### cuDNN must be 9.16 or newer, and `uv sync` will undo it

This is the step people skip. Do not skip it.

```bash
uv pip install nvidia-cudnn-cu13==9.16.0.29     # cu130
# uv pip install nvidia-cudnn-cu12==9.16.0.29   # cu128 / cu126

python -c "import torch; print(torch.backends.cudnn.version())"   # must print 91600
```

Two reasons, and both bite hard:

- PyTorch 2.9.1's `nn.Conv3d` has a [known bug](https://github.com/pytorch/pytorch/issues/168167) on
  cuDNN below 9.15 that badly degrades speed and memory. The 6-channel patch embed *is* a `Conv3d`,
  so it is on the hot path of every forward pass.
- SGLang 0.5.8 refuses to start on that combination at all. It raises rather than warning.

**The package name changes with the CUDA flavour, and installing the wrong one is a silent no-op** --
it lands an unused package while torch keeps loading the other family. That is why the check above
prints a number rather than trusting the install to have worked. Re-run this after every
`uv sync`, because a sync resolves cuDNN back down.

**`uv pip` needs the venv activated.** Unlike `uv sync`, it ignores `UV_PROJECT_ENVIRONMENT` and
installs into whatever environment is active -- with none active, that can be an ambient conda env,
silently, while the venv keeps its old cuDNN. Either activate first (as above) or pass
`--python .venv/bin/python`.

### Optional: vLLM or SGLang

These two cannot share a venv. vLLM 0.14.1 pins `flashinfer-python==0.5.3`, SGLang 0.5.8 pins
`0.6.1`, and installing the second breaks the first. `pyproject.toml` declares them as conflicting
so `uv` will stop you, but the fix is two venvs, not one clever resolution.

```bash
# vLLM, in the venv you already have
uv sync --group cu130 --group dev --group vllm
uv pip install nvidia-cudnn-cu13==9.16.0.29     # again

# SGLang, in a second venv. Note UV_PROJECT_ENVIRONMENT, not VIRTUAL_ENV.
uv venv .venv-sglang
UV_PROJECT_ENVIRONMENT=.venv-sglang uv sync --group cu130 --group dev --group sglang \
  --extra-index-url https://docs.sglang.ai/whl/cu130/ --index-strategy unsafe-best-match
UV_PROJECT_ENVIRONMENT=.venv-sglang uv pip install --python .venv-sglang/bin/python \
  nvidia-cudnn-cu13==9.16.0.29
```

**`UV_PROJECT_ENVIRONMENT` is not interchangeable with `VIRTUAL_ENV` here, and getting it wrong is
destructive.** `uv sync` manages *the project environment*, which defaults to `.venv`; it does not
follow an activated `VIRTUAL_ENV`. Point it with `VIRTUAL_ENV` and it will cheerfully resolve the
sglang group into `.venv` instead, uninstalling vLLM and replacing flashinfer 0.5.3 with 0.6.1 --
leaving you with one broken venv and one empty one. If that happens, rebuild with
`uv sync --group cu130 --group dev --group vllm` and re-pin cuDNN.

**On Blackwell (sm_120) SGLang needs the Triton attention backend:**

```bash
export TRANSVLM_SGLANG_ATTENTION_BACKEND=triton
```

Not a tuning preference. flashinfer JIT-compiles kernels for `sm_120a` using the system `nvcc`, and
if that `nvcc` is CUDA 12.0 it tops out at `compute_90` and fails with
`nvcc fatal: Unsupported gpu architecture 'compute_120a'`. Without the variable the engine starts
and then dies deep inside, which reads like a model problem rather than a toolchain one.

### torchcodec needs to find FFmpeg's shared libraries

If your `ffmpeg` came from the system package manager, its libraries are already on the loader path
and there is nothing to do. If it came from conda, a module system, or anywhere else non-standard,
torchcodec will fail at import with a wall of

```
FFmpeg version 7: Could not load this library: .../libtorchcodec_core7.so
```

for every FFmpeg major version it tries. It is looking for `libavutil.so.*` and cannot find it.
Point the loader at whichever directory holds them:

```bash
# find them
ldd "$(which ffmpeg)" | grep libavutil

# then, using that directory
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib:/path/to/ffmpeg/lib:$LD_LIBRARY_PATH"
```

The `torch/lib` entry belongs there too: torchcodec links against libtorch, and the loader has to
resolve both from the same process.

### GPU video decoding does not work, and that is fine

The PyPI torchcodec wheels are built without NVDEC/NPP, so `device='cuda'` decoding raises
`RuntimeError: Unsupported device: cuda:0`. CPU decode with GPU inference is the supported path
everywhere and costs about 0.5%.

## Weights

```bash
hf download HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct --local-dir ./pretrained/TransVLM-v1
```

There is no default checkpoint path: `--ckpt-dir` is required. A default pointing at a directory you
do not have fails in a much more confusing way than a missing argument.

**A second model is fetched on first use.** Optical flow comes from NeuFlow v2
(`Study-is-happy/neuflow-v2`, ~30 MB), pulled from the Hub the first time flow is computed rather
than shipped here. On a machine without Hub access the run gets as far as loading TransVLM and then
fails at the flow step; pre-fetch it, or point `FlowConfig.neuflow_weights` at a local copy.

## Run

### One video

```bash
python infer_video.py \
  --video /path/to/video.mp4 \
  --ckpt-dir ./pretrained/TransVLM-v1 \
  --output-jsonl out.jsonl
```

One JSON object per video, with `segments` as `{start_time, end_time}` pairs in seconds on the
original video's timeline, plus timings and the full configuration that produced them.

The video is cut into 10 s windows with 9 s stride, each window is inferred, and overlapping
predictions are merged. Nothing is written to disk except under `--temp-dir` (system temp by
default), and that is cleaned up per item.

**Keep videos reasonably short.** Every window is a forward pass, so cost is linear in duration and
a feature-length film is hours of GPU time. For anything long, split it up first and offset the
returned timestamps yourself:

```bash
ffmpeg -i long.mp4 -c copy -f segment -segment_time 300 part_%03d.mp4
```

### Many pre-cut clips

If your clips are already cut, `infer_clips.py` skips the windowing and infers each one directly.
This is the route the published numbers were produced with, and its output feeds `../evaluation`.

```bash
python infer_clips.py \
  --video-list clips.txt \
  --ckpt-dir ./pretrained/TransVLM-v1 \
  --output-jsonl clips.jsonl
```

### Choosing a backend

`--backend hf` (default), `vllm`, or `sglang`.

**Use `hf` for anything you intend to report.** It is the default, it needs no extra install, and
it is the only backend whose scores match or exceed the paper's. Measured over the same 872 clips
with everything else identical:

| backend | tt0.00 cl_F1 | ttmean cl_F1 |
|---|---|---|
| `hf` | 0.6083 | 0.7822 |
| `vllm` | 0.6006 | 0.7775 |
| `sglang` | 0.5880 | 0.7717 |

The spread is not noise: `sglang` sits about 0.02 below `hf` at the tightest tolerance. The three
run different attention kernels -- on Blackwell, `sglang` is forced onto Triton because flashinfer
cannot JIT for `sm_120` -- and that is enough to move predictions near a decision boundary. Treat
`vllm` and `sglang` as throughput options for bulk processing, and re-measure on your own data
before quoting a number from either.

## Two settings that change the numbers

**`--max-pixels-override`** defaults to `524288`, which is *not* the paper's `786432`. Measured
across 872 clips, the lower budget is better on both metrics and 1.18x faster, so it is the
recommended default and what we ship. If you want the paper's configuration exactly, pass
`786432` -- but note that alone does not reproduce the published figures, because those also used
pre-computed optical flow rather than flow computed at inference time.

**`--timestamp-format`** is `1f` by default: prompt timestamps are printed with one decimal place.
`2f` prints two. They are not interchangeable and their scores must never be compared: at 25 fps the
`1f` form prints the same timestamp for 25 of 125 temporal patches, so the model sees a coarser time
signal. `1f` is what the released checkpoint's reported numbers used.

## Tests

```bash
python -m pytest tests -q
```

The vLLM and SGLang tests skip when those packages are absent, so a HuggingFace-only install still
gets a green run.

## Layout

```
inference/
├── infer_video.py          one video in, merged transitions out
├── infer_clips.py          pre-cut clips in, per-clip predictions out
├── transvlm/
│   ├── _paths.py           where the package's own files live
│   ├── assets/prompts/     the prompt the checkpoint was evaluated with
│   ├── data/               decoding, optical flow, the resize protocol
│   │   └── _neuflow/       NeuFlow v2, vendored (Apache-2.0, own LICENSE)
│   ├── inference/          the three backends, output parser, segment merger
│   └── models/             6-channel and timestamp processor patches
└── tests/
```

Two files carry a "do not edit" note: `transvlm/data/_vision_process_of.py` and the contents of
`transvlm/models/_vllm_patches/`. They are vendored, and the preprocessing they perform is part of
what the checkpoint expects. To change their behaviour, wrap them --
`transvlm/data/video_backend.py` is an example of doing exactly that without touching a byte.

See `../NOTICE` for vendored components and their upstreams.
