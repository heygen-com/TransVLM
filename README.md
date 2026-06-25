# TransVLM

**A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions**

Ce Chen, Yi Ren, Yuanming Li, Viktor Goriachko, Zhenhui Ye, Zujin Guo, Zhibin Hong, Mingming Gong

University of Melbourne · HeyGen Research · Nanyang Technological University

## Project Page

https://chence17.github.io/TransVLM/

## Abstract

Traditional Shot Boundary Detection (SBD) inherently struggles with complex transitions by formulating the task around isolated cut points, frequently yielding corrupted video shots. We address this fundamental limitation by formalizing the Shot Transition Detection (STD) task. Rather than searching for ambiguous points, STD explicitly detects the continuous temporal segments of transitions. To tackle this, we propose **TransVLM**, a Vision-Language Model (VLM) framework for STD. Unlike regular VLMs that predominantly rely on spatial semantics and struggle with fine-grained inter-shot dynamics, our method explicitly injects optical flow as a critical motion prior at the input stage. Through a simple yet effective feature-fusion strategy, TransVLM directly processes concatenated color and motion representations, significantly enhancing its temporal awareness without incurring any additional visual token overhead on the language backbone. To overcome the severe class imbalance in public data, we design a scalable data engine to synthesize diverse transition videos for robust training, alongside a comprehensive benchmark for STD. Extensive experiments demonstrate that TransVLM achieves superior overall performance, outperforming traditional heuristic methods, specialized spatiotemporal networks, and top-tier VLMs.

## Repository Layout

- `docs/` — GitHub Pages source (project page, figures, videos, interactive comparison widgets).

Code release is in progress.

## Citation

```bibtex
@article{chen2026transvlm,
  title   = {TransVLM: A Vision-Language Framework and Benchmark for Detecting Any Shot Transitions},
  author  = {Chen, Ce and Ren, Yi and Li, Yuanming and Goriachko, Viktor and
             Ye, Zhenhui and Guo, Zujin and Hong, Zhibin and Gong, Mingming},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Updates

The STD benchmark is intended as a **living resource**. We are actively exploring **smaller and more efficient VLM backbones** for the STD task, and the resulting checkpoints, training recipes, and evaluation results will be open-sourced here as they become available.

## Community Contributions

We **welcome community submissions** of smaller-model baselines on the STD benchmark. If you have a more compact or more efficient method for STD, please open an issue or pull request with: (i) the model description and total parameter count; (ii) per-method evaluation results following the format on the [project page](https://chence17.github.io/TransVLM/); and (iii) reproduction instructions. Approved submissions will be added to the leaderboard on the project page.
