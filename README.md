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
