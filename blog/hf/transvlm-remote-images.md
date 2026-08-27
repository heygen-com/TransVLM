# TransVLM: Detecting Any Shot Transition with Vision-Language Models

**Accepted at ECCV 2026 — HeyGen Research & University of Melbourne**

📄 [Read the technical report](https://dynamic.heygen.ai/www/Paper%20Links/TransVLM_HeyGen.pdf) · 🌐 [Project page](https://heygen-com.github.io/TransVLM/) · 📝 [Original post](https://www.heygen.com/research/transvlm)

We reformulate shot boundary detection as Shot Transition Detection (STD)—finding complete transition segments, not just cut points. TransVLM fuses optical flow with color frames in a vision-language model to detect all transition types: cuts, dissolves, and special effects. It achieves 78.3% segment F1 on public data and 89.5% on synthetic data, outperforming all existing methods.

---

When you watch a movie dissolve from one scene to the next, you perceive a smooth blend. But ask an AI to find that transition in a raw video file, and most methods either miss it entirely or mark it at the wrong frame. For hard cuts this problem is mostly solved. For everything else—dissolves, fades, wipes, special effects—existing methods fall apart.

This matters because inaccurate transition detection corrupts downstream pipelines: video retrieval, captioning, action recognition, and especially text-to-video generation, where bad shot boundaries in training data cause generative models to produce unintended transitions.

## From Points to Segments

Traditional Shot Boundary Detection (SBD) treats the problem as finding isolated cut points—single frames where one shot ends and another begins. This works for abrupt cuts but fundamentally cannot represent gradual transitions that span dozens or hundreds of frames. A dissolve has a start time and an end time; reducing it to a single point discards exactly the information you need.

We reformulate the problem as Shot Transition Detection (STD): explicitly detecting the continuous temporal segments of transitions, including their precise start and end timestamps. Each transition is represented as a tuple (start, end) rather than a single frame index. This representation naturally unifies abrupt cuts (where start ≈ end) and gradual transitions (where start < end), and it aligns perfectly with VLM-style structured output—the model simply returns a JSON array of segments.

## The Problem with Existing Approaches

Two paradigms dominate today. Traditional SBD methods (PySceneDetect, TransNet, AutoShot) work well on abrupt cuts but fail on gradual transitions—dissolves, fades, and wipes produce ambiguous frame-level probabilities that no threshold can cleanly separate. General-purpose Vision-Language Models (Qwen3-VL, Gemini) handle complex transitions better but miss simple cuts entirely, because their sparse, low-frame-rate inputs cannot capture instantaneous changes.

Neither paradigm detects the full temporal span of a transition. They look for points, not segments. TransVLM detects segments.

![A grid of video frames demonstrating various video transitions, with colored bars below each sequence comparing the detection performance of different methods: Ground Truth, SBD, VLM, and Ours.](https://cdn.sanity.io/images/pdhqcmb1/production/3a9aa50b031bfc5ad6832c0256058ee90ac2df83-7902x4176.jpg?w=1600&fit=max&auto=format)

*Figure 1: Limitations of existing shot transition detection methods. SBD methods (red) detect normal cuts but fail on gradual and special transitions. VLMs (blue) handle gradual transitions but miss cuts. TransVLM (green) detects all types.*

## Optical Flow as Motion Prior

The key insight: VLMs are biased toward static spatial semantics and miss fine-grained inter-frame motion dynamics. But transitions are fundamentally about motion—abrupt pixel changes for cuts, smooth blending for dissolves. Optical flow captures exactly this information.

The difference is visible at the token level. When we visualize the internal visual tokens that Qwen3-VL produces for color frames versus optical flow frames, the flow-derived tokens show dramatically sharper contrast at transition boundaries—for both normal cuts and subtle cuts that color tokens cannot distinguish.

![A comparative image showing video analysis for "Normal Cut" and "Subtle Cut" scenes, each displaying original color frames, optical flow frames, and corresponding visual token heatmaps for both color and optical flow data.](https://cdn.sanity.io/images/pdhqcmb1/production/4a5f4d15ff719e2f6d5db43462f02b0afcda37c6-3286x1266.png?w=1600&fit=max&auto=format)

*Figure 2: Visual token visualization for color vs. optical flow. Left: normal cut. Right: subtle cut. Top rows show raw color and optical flow frames. Bottom rows show the internal visual tokens produced by Qwen3-VL. The optical flow tokens exhibit dramatically sharper contrast at transition boundaries, making both cut types clearly distinguishable.*

We inject optical flow directly into the vision encoder through a feature-fusion strategy. Color frames and optical flow frames are concatenated along the channel dimension and processed together by the Vision Patch Embedding layer. The input channels expand from 3 to 6, but the output token count stays identical—zero additional computational burden on the language model. A zero-padding initialization strategy ensures the optical flow channels don't disrupt the pre-trained color representations during early training.

## TransVLM Framework

![Diagram illustrating a video analysis pipeline with three main sections: model architecture, data synthesis, and arbitrary video inference.](https://cdn.sanity.io/images/pdhqcmb1/production/edaec750b9a21c9a6353101667d847f0892ca9db-4248x1522.png?w=1600&fit=max&auto=format)

*Figure 3: TransVLM Framework. (a) Optical flow is fused with color frames at the Vision Patch Embed layer. (b) A scalable data engine synthesizes training videos with diverse transitions. (c) Sliding-window inference enables arbitrary-length video processing.*

TransVLM comprises three core components: a modified vision-language architecture with optical flow fusion, a scalable data engine for training, and a sliding-window inference pipeline for arbitrary-length videos.

### Sliding-Window Inference

TransVLM is trained on short clips, but real videos can be hours long. Directly processing a full-length video would cause memory overflow and distribution mismatch. We partition the video into overlapping temporal windows, generate local segment-level predictions for each, then merge them into a continuous global output via temporal Non-Maximum Suppression. This handles videos of arbitrary length while maintaining training-inference consistency.

### Scalable Data Engine

Annotated transition data is scarce, and public datasets have noisy, point-level labels that don't capture transition segments. We built a data engine that automatically synthesizes diverse training videos: given clean shots, it randomly applies one of 59 distinct transition effects (powered by FFmpeg), producing videos with precise segment-level labels. Combined with re-annotated public data and quality-aware sampling, this gives TransVLM a training set of 233,000 videos containing 690,000 transitions across four quality tiers.

## STD Benchmark

To standardize evaluation of the Shot Transition Detection task, we constructed a comprehensive benchmark: 5,215 videos (100+ hours) containing 45,239 transitions with segment-level ground truth. The benchmark spans cuts (&lt;0.1s), normal transitions (≤1s), and long transitions (&gt;1s), with multi-dimensional metrics including segment-level F1, frame-level F1, Absolute Boundary Error, and Real-Time Factor.

## Results

TransVLM establishes a new state of the art across the board, outperforming traditional SBD methods, specialized deep learning models, and general-purpose VLMs on both public and synthetic evaluation data.

![Table comparing performance metrics for various scene detection methods, showing TransVLM (Ours) achieves the highest F1 scores on public and synthetic datasets, and the lowest ABE (s).](https://cdn.sanity.io/images/pdhqcmb1/production/d889441075e0445d05a489c474a7dc78e631ba14-1540x674.png?w=1600&fit=max&auto=format)

*Table 1: Quantitative comparison on the STD benchmark. Segment-level and frame-level F1 on both public and synthetic evaluation data. ABE = Absolute Boundary Error (lower is better). RTF = Real-Time Factor (below 1.0 = faster than real-time).*

On public data, TransVLM achieves 78.3% segment-level F1—surpassing specialized deep learning models like AutoShot (75.1%). On synthetic data containing complex gradual transitions, the gap widens dramatically: TransVLM reaches 89.5% segment F1 while AutoShot drops to 29.9%. The Absolute Boundary Error of just 0.11 seconds demonstrates precise temporal localization.

Traditional SBD methods are faster (AutoShot at RTF 0.03) but fundamentally unable to handle gradual transitions. General-purpose VLMs like Gemini show reasonable performance on complex transitions but struggle with cuts. TransVLM is the only method that robustly handles all transition types at practical inference speeds (RTF 0.50).

## What Matters Most

Ablation studies validate each design decision:

- **Optical flow is critical.** Removing it drops public segment F1 from 78.3% to 69.4% and synthetic frame F1 from 93.8% to 89.2%—confirming that VLMs' inherent insensitivity to low-level temporal cues is a real bottleneck.
- **Feature fusion beats separate encoders.** Processing color and flow as separate streams achieves similar accuracy (94.1% frame F1 on synthetic) but doubles the visual token count, inflating RTF from 0.50 to 1.29—no longer real-time.
- **Mixed data is essential.** Training on synthetic data alone collapses to 41.6% segment F1 on public data. Training on public data alone collapses on complex transitions. The quality-aware mixed strategy bridges the domain gap.
- **Zero-padding initialization prevents catastrophic forgetting.** Naive weight duplication for the optical flow channels drops segment F1 from 78.3% to 65.5% by destabilizing the pre-trained spatial representations.

## Why This Matters

Shot transition detection is infrastructure. Every downstream video understanding task—retrieval, captioning, action recognition—depends on accurate shot segmentation. For HeyGen's video generation pipeline, the quality of training data directly determines the quality of generated videos. Inaccurate transition labels in training data cause generative models to produce unintended cuts and artifacts. TransVLM provides the precise, segment-level annotations that clean this signal at the source.

---

All videos shown in this report are for research demonstration purposes only. HeyGen's platform enforces consent verification for all digital twin creation.
