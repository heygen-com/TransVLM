<h1 align="left">TransVLM
  <a href="https://www.heygen.com/research">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/heygen-logo-white.png">
      <img align="right" height="44" src="assets/heygen-logo-black.png" alt="HeyGen Research">
    </picture>
  </a>
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2604.27975"><img src="https://img.shields.io/badge/arXiv-2604.27975-b31b1b.svg" alt="arXiv"></a>
  <a href="https://heygen-com.github.io/TransVLM/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-TransVLM-ffce1c.svg" alt="Hugging Face Model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-3da639.svg" alt="License"></a>
</p>

<p align="center">
  <b>A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions</b>
</p>

<p align="center">
  <b>Accepted to ECCV 2026 (Poster)</b>
</p>

---

## 📰 News

- *TransVLM* was accepted to **ECCV 2026** as a Poster.
- Model weights, inference code and evaluation code are released. See [Release Progress](#-release-progress).
- A write-up of the method for a general audience is in [`blog/hf/transvlm.md`](blog/hf/transvlm.md).

## Abstract

Traditional Shot Boundary Detection (SBD) inherently struggles with complex transitions by formulating the task around isolated cut points, frequently yielding corrupted video shots. We address this fundamental limitation by formalizing the Shot Transition Detection (STD) task. Rather than searching for ambiguous points, STD explicitly detects the continuous temporal segments of transitions. To tackle this, we propose **TransVLM**, a Vision-Language Model (VLM) framework for STD. Unlike regular VLMs that predominantly rely on spatial semantics and struggle with fine-grained inter-shot dynamics, our method explicitly injects optical flow as a critical motion prior at the input stage. Through a simple yet effective feature-fusion strategy, TransVLM directly processes concatenated color and motion representations, significantly enhancing its temporal awareness without incurring any additional visual token overhead on the language backbone. To overcome the severe class imbalance in public data, we design a scalable data engine to synthesize diverse transition videos for robust training, alongside a comprehensive benchmark for STD. Extensive experiments demonstrate that TransVLM achieves superior overall performance, outperforming traditional heuristic methods, specialized spatiotemporal networks, and top-tier VLMs.

Qualitative results and the comparison against baselines are on the [project page](https://heygen-com.github.io/TransVLM/).

## 🗓️ Release Progress

- [x] Model weights
- [x] Inference code
- [x] Evaluation code
- [x] Data engine code ([`data_engine/README.md`](data_engine/README.md))
- [x] Re-annotated dataset labels ([`benchmark/README.md`](benchmark/README.md))

All components are released.

## 🚀 Getting Started

You need Python 3.12, a CUDA GPU, and `ffmpeg` on `PATH`. Install first: setup has a few sharp
edges (a cuDNN version floor, mutually exclusive backend venvs, and FFmpeg library discovery),
so follow [`inference/README.md`](inference/README.md) rather than guessing:

```bash
cd inference
uv venv && source .venv/bin/activate
uv sync --group cu130 --group dev
uv pip install nvidia-cudnn-cu13==9.16.0.29   # required, and every uv sync undoes it
```

Then fetch the checkpoint and run detection on one video:

```bash
hf download HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct --local-dir ./pretrained/TransVLM-v1

python infer_video.py \
  --video /path/to/video.mp4 \
  --ckpt-dir ./pretrained/TransVLM-v1 \
  --output-jsonl out.jsonl
```

> [!IMPORTANT]
> This checkpoint takes **6-channel RGB + optical-flow** input, so a plain `transformers`
> Qwen3-VL pipeline will not produce correct results. Use the code in `inference/`.

Every option, the output format and the three backends are documented here:

| Topic | Document |
|---|---|
| Inference | [`inference/README.md`](inference/README.md) |
| Evaluation | [`evaluation/README.md`](evaluation/README.md) |

## Repository Layout

| Path | Contents |
|---|---|
| `inference/` | Inference pipeline and its two entry points, plus the vendored NeuFlow v2 optical-flow code |
| `evaluation/` | Evaluation code for the STD benchmark |
| `benchmark/` | Frozen timestamp labels, version `eccv2026-paper` (v1), for the six public datasets of the STD benchmark; labels only, no videos |
| `data_engine/` | Standalone FFmpeg-based transition-video synthesis for building training data; generates clips from clean shots and ships no media |
| `docs/` | Project page sources (figures, videos, interactive comparison widgets) |
| `blog/` | Blog write-ups of the method, with the figures they ship with |
| `assets/` | Logos used by this README |
| `NOTICE` | Vendored third-party components and their upstream attribution |

## ✏️ Citation

```bibtex
@inproceedings{chen2026transvlm,
  title={TransVLM: A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions},
  author={Chen, Ce and Ren, Yi and Li, Yuanming and Goriachko, Viktor and
          Ye, Zhenhui and Guo, Zujin and Hong, Zhibin and Gong, Mingming},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

## Updates

The STD benchmark is intended as a **living resource**. We are actively exploring **smaller and more efficient VLM backbones** for the STD task, and the resulting checkpoints, training recipes, and evaluation results will be open-sourced here as they become available.

## Community Contributions

We **welcome community submissions** of smaller-model baselines on the STD benchmark. If you have a more compact or more efficient method for STD, please open an issue or pull request with: (i) the model description and total parameter count; (ii) per-method evaluation results following the format on the [project page](https://heygen-com.github.io/TransVLM/); and (iii) reproduction instructions. We welcome discussion of reproducible experiments and results through issues and pull requests.

## Acknowledgements

TransVLM builds on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and uses
[NeuFlow v2](https://github.com/neufieldrobotics/NeuFlow_v2) for optical-flow estimation. We thank
both teams for releasing their work.

## License

TransVLM's original code is released under the [Apache License 2.0](LICENSE). Third-party
components retain their respective licenses; attribution and details are in [`NOTICE`](NOTICE).
The released weights derive from Qwen3-VL-4B-Instruct and are distributed under Apache-2.0 as
well.

The annotation contributions released in [`benchmark/`](benchmark/README.md) -- the corrected
re-annotations produced for this benchmark -- are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This does not extend to the original
annotations or media of the six public datasets, which remain under their own original terms; see
[`benchmark/README.md`](benchmark/README.md) for per-dataset provenance and licensing. The
benchmark release contains labels only, not videos.

TransVLM was developed as academic research and is published at ECCV 2026. It is not used in any
HeyGen product; HeyGen's production system for this task is a different model. The training data behind these
weights, and the licences of the public datasets in it, are listed on the
[model card](https://huggingface.co/HeyGenAI/TransVLM-Qwen3-VL-4B-Instruct).
