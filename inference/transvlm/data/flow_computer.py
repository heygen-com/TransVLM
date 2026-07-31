"""NeuFlow v2 optical-flow computer with sibling-file caching.

Reproduces the FULL flow-generation pipeline of
the training pipeline's flow generation, including the pre-step ``resize_video_for_qwen3vl``
(ffmpeg scale to a Qwen3-VL-friendly resolution) that the training pipeline
did BEFORE feeding NeuFlow. This ordering is important — running NeuFlow at
the original resolution then bicubic-resizing the HSV-encoded flow output
would (a) shift the position-vector magnitude distribution and (b) pollute
HSV hue values via non-linear bicubic interpolation around 0°/360°. See
the inference README for why this ordering matters.

Sibling layout (both files at Qwen3-VL-friendly resolution):
    <name>_resized.mp4  # RGB video after ffmpeg scale
    <name>_flow.mp4     # NeuFlow output on the resized video (HSV-encoded)

Design decisions (locked to keep training-distribution parity):

* Frame pairing: for ``N`` decoded frames, ``(prev, cur)`` pairs are built
  with ``prev[0] = cur[0]`` (self-pair on frame 0), matches parent.
* Model precision: ``model.half()`` after ``fuse_conv_and_bn`` — same as parent.
* Batch dim padding: pad up to a multiple of ``mini_batch_size`` (default 32).
* Visualization: ``torchvision.utils.flow_to_image`` (HSV-encoded RGB uint8).
* Encoder: libx264 via ``torchvision.io.write_video``.
* Pre-resize target: ``smart_resize_for_qwen3vl`` — the FULL formula with
  ``MODEL_SEQ_LEN``-driven ``total_pixels/nframes*FRAME_FACTOR`` penalty
  (matches ``video_tools.py::get_qwen3vl_suitable_hw_of_video``).
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import flow_to_image

from transvlm.data.flow_config import FlowConfig
from transvlm.data._neuflow.backbone_v7 import ConvBlock
from transvlm.data._neuflow.neuflow import NeuFlow
from transvlm.data.flow_writer import FLOW_CODECS, write_flow_mp4
from transvlm.data.resize_video_helper import (
    DEFAULT_NFRAMES_FOR_RESIZE,
    probe_video_fps_exact,
    probe_video_hw,
    resize_video_ffmpeg,
    smart_resize_for_qwen3vl,
)

logger = logging.getLogger(__name__)


@dataclass
class FlowTimings:
    """Where one ``compute_flow_only`` call's time went, split four ways.

    Split rather than summed because the four parts are not comparable to each other and are
    not always all wanted. NeuFlow's forward pass is the model cost; decoding is I/O; the flow
    visualisation and its mp4 encode exist only so the flow can be inspected or cached, and a
    pipeline that keeps flow in memory does not pay them at all. Quoting a single total against a
    figure that measured only one of the four is the mistake this dataclass is meant to prevent.

    The split is honest about GPU asynchrony without needing explicit syncs: the NeuFlow loop
    calls ``.cpu()`` on every mini-batch, which blocks until that batch is done, and
    ``flow_to_image`` then runs on CPU tensors.
    """

    decode_sec: float = 0.0
    neuflow_sec: float = 0.0
    flow_to_image_sec: float = 0.0
    encode_sec: float = 0.0

    @property
    def total_sec(self) -> float:
        """Sum of the parts. Equals what callers previously recorded as ``flow_sec``."""
        return self.decode_sec + self.neuflow_sec + self.flow_to_image_sec + self.encode_sec

    def as_dict(self) -> dict[str, float]:
        """Rounded, for writing next to the other per-item timings in a jsonl."""
        return {
            'flow_decode_sec': round(self.decode_sec, 4),
            'flow_neuflow_sec': round(self.neuflow_sec, 4),
            'flow_to_image_sec': round(self.flow_to_image_sec, 4),
            'flow_encode_sec': round(self.encode_sec, 4),
        }


@dataclass(frozen=True)
class ResizedAndFlow:
    """Pair of sibling video paths produced by :class:`OnlineFlowComputer`.

    Both files are at the training-side Qwen3-VL-friendly resolution
    (multiples of 32, within the ``smart_resize_for_qwen3vl`` pixel budget).

    * ``resized_path``: ``<name>_resized.mp4`` — RGB video after ffmpeg scale.
    * ``flow_path``: ``<name>_flow.mp4`` — NeuFlow output HSV visualization.
    * ``target_h`` / ``target_w``: nominal target dims (may differ from the
      actual decoded dims by at most rounding; both files should agree).
    """

    resized_path: Path
    flow_path: Path
    target_h: int
    target_w: int


def _insert_before_ext(name: str, tag: str) -> str:
    """Insert ``tag`` before the file extension in ``name``.

    ``'_flow.mp4'`` + ``'_mp393216'`` → ``'_flow_mp393216.mp4'``.
    ``'_flow_custom'`` (no ext) + ``'_mp393216'`` → ``'_flow_custom_mp393216'``.
    Robust to any/no extension — no more silent tag-drop when suffix isn't ``.mp4``.
    """
    if '.' not in name:
        return name + tag
    stem, _, ext = name.rpartition('.')
    return f'{stem}{tag}.{ext}'


# ---------------------------------------------------------------------------
# Helper: fuse Conv2d + BatchNorm2d — ported verbatim from parent.
# ---------------------------------------------------------------------------
def _exact_fps_or(path: Path, decoded_fps: float | None) -> Fraction | float:
    """Exact container rate for ``path``, falling back to the decoder's float.

    Split out because both flow-writing paths need it and the fallback must stay identical: a
    stream with an unusable ``avg_frame_rate`` (``0/0``) previously landed on ``25.0``, and
    changing that silently would alter timestamps on inputs that currently work.
    """
    try:
        return probe_video_fps_exact(path)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        fallback = float(decoded_fps or 25.0)
        logger.warning('exact fps probe failed for %s (%s); falling back to %s', path, exc, fallback)
        return fallback


def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold ``bn`` into ``conv`` so a single Conv2d encodes both operations."""
    fused = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))

    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.matmul(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


# ---------------------------------------------------------------------------
# OnlineFlowComputer — sibling-cache friendly flow producer.
# ---------------------------------------------------------------------------
class OnlineFlowComputer:
    """Compute optical flow for a video with NeuFlow v2.

    Lifecycle:
        1. Instantiate once per process (loads the ~30 MB NeuFlow weights).
        2. Call ``compute_and_cache(video_path)`` per input video.
        3. Underlying ``model.init_bhwd`` is only re-invoked when the batch
           dim or resolution changes.
    """

    # NeuFlow's feature-pyramid arithmetic (s8 vs s16 upsampling) requires H
    # and W divisible by 16. Inputs whose dims are not multiples of 16 are
    # zero-padded to the next boundary and cropped after inference. With the
    # ffmpeg pre-resize (Plan A) landing dims are always multiples of 32
    # (image_factor), so the pad in practice is 0.
    _ALIGN: int = 16

    # Sibling filename suffixes (both files at Qwen3-VL-friendly resolution
    # after the ffmpeg pre-resize; matches parent training-side naming).
    _RESIZED_SUFFIX: str = '_resized.mp4'
    _FLOW_SUFFIX: str = '_flow.mp4'

    def __init__(
        self,
        cfg: FlowConfig,
        *,
        device: str = 'cuda:0',
        # ``None`` means "take FlowConfig.mini_batch_size" so the value can come
        # from a YAML. An explicit int still wins, which keeps
        # ``41_bench_worker.py --flow-mini-batch-size`` behaving as before.
        mini_batch_size: int | None = None,
        nframes_for_resize: int = DEFAULT_NFRAMES_FOR_RESIZE,
        image_patch_size: int = 16,
        num_ffmpeg_threads: int = 8,  # forwarded as VideoDecoder(..., num_ffmpeg_threads=...)
        max_pixels_override: int | None = None,  # ablation flag — see resize_video_helper
        max_pixels: int | None = None,  # noqa: ARG002 — kept for backwards CLI compat; ignored
        resize_factor: int | None = None,  # noqa: ARG002 — same
        # `flow_to_image` is 60% of the flow stage's cost. Running it on device measured 1.91x
        # faster end-to-end over the full 872-clip set (34m35s -> 18m06s, 8 GPUs) for
        # |delta ttmean cl_f1| = 0.00033, inside the 0.002 parity gate -- hence the default.
        # The two devices do not agree bit-for-bit (max abs diff 1/255 on ~1e-6 of pixels, from
        # per-element rounding; the max_norm reduction itself matches exactly), so the encoded
        # flow mp4 differs and `viz_on_gpu=False` is kept to reproduce the older byte stream.
        viz_on_gpu: bool = True,
    ) -> None:
        if cfg.downscale != 1:
            raise NotImplementedError(
                'FlowConfig.downscale != 1 not supported yet — training pipeline used the '
                'ffmpeg-resized flow, so any downscale would break paper parity.'
            )
        self._cfg = cfg
        self._device_str = device
        self._device = torch.device(device)
        self._mini_batch_size = cfg.mini_batch_size if mini_batch_size is None else mini_batch_size
        self._nframes_for_resize = nframes_for_resize
        self._image_patch_size = image_patch_size
        self._num_ffmpeg_threads = num_ffmpeg_threads
        self._max_pixels_override = max_pixels_override
        self._viz_on_gpu = viz_on_gpu and self._device.type == 'cuda'

        logger.info('Loading NeuFlow (%s) onto %s', cfg.neuflow_weights, device)
        self._model = NeuFlow.from_pretrained(cfg.neuflow_weights).to(self._device)
        self._fuse_bn()
        self._model.eval()
        self._model.half()

        # Cache the last (batch, H, W) init so redundant re-inits are avoided.
        self._bhwd: tuple[int, int, int] | None = None

    def _fuse_bn(self) -> None:
        """Fold every ``ConvBlock``'s BN into its Conv2d — halves per-step latency."""
        for module in self._model.modules():
            if isinstance(module, ConvBlock):
                module.conv1 = _fuse_conv_and_bn(module.conv1, module.norm1)
                module.conv2 = _fuse_conv_and_bn(module.conv2, module.norm2)
                delattr(module, 'norm1')
                delattr(module, 'norm2')
                module.forward = module.forward_fuse

    def _ensure_bhwd(self, batch: int, height: int, width: int) -> None:
        key = (batch, height, width)
        if self._bhwd == key:
            return
        self._model.init_bhwd(batch, height, width, self._device_str)
        self._bhwd = key
        logger.debug('NeuFlow init_bhwd: batch=%d, H=%d, W=%d', batch, height, width)

    # ---- Core computation -------------------------------------------------
    @torch.no_grad()
    def compute(self, frames: torch.Tensor, *, timings: FlowTimings | None = None) -> torch.Tensor:
        """Compute per-frame flow visualization from ``(N, 3, H, W) uint8``.

        Returns a ``(N, 3, H, W) uint8`` tensor (HSV-encoded flow), matching the
        exact format the training pipeline consumed.

        Pass ``timings`` to have the NeuFlow forward and the visualisation recorded
        separately; only the forward is what the paper's 0.22 RTF measured.

        H and W are internally zero-padded to the next multiple of ``_ALIGN``
        because NeuFlow's s8/s16 feature-pyramid arithmetic requires the input
        dimensions divisible by 16 (otherwise ``merge_s8`` fails on a
        one-off-by-one shape mismatch between ``H/8`` and ``2 * (H/16)``).
        The padded region is dropped after inference. For inputs already
        aligned (e.g. the training pipeline's ``resize_video_for_qwen3vl`` output),
        this is a zero-cost no-op that preserves numerical parity.
        """
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f'Expected (N, 3, H, W) tensor. Got shape {tuple(frames.shape)}')
        if frames.dtype != torch.uint8:
            raise ValueError(f'Expected uint8 frames. Got {frames.dtype}')

        n_valid, _, h, w = frames.shape

        # Note: pre-resize was moved OUT of compute() into compute_and_cache
        # (via ffmpeg scale before decoding). Frames arriving here are
        # expected to be at the smart-resize target dim already. See
        # commit-log for the 3-backend-benchmark rework.

        pad_h = (self._ALIGN - h % self._ALIGN) % self._ALIGN
        pad_w = (self._ALIGN - w % self._ALIGN) % self._ALIGN
        h_pad, w_pad = h + pad_h, w + pad_w
        self._ensure_bhwd(self._mini_batch_size, h_pad, w_pad)

        # (prev, cur) pairs — self-pair the first frame so N pairs give N flow maps.
        prev = torch.cat([frames[0:1], frames[:-1]], dim=0).half()
        cur = frames.half()

        # Pad H, W dims to the alignment boundary.
        if pad_h or pad_w:
            prev = nn.functional.pad(prev, (0, pad_w, 0, pad_h))
            cur = nn.functional.pad(cur, (0, pad_w, 0, pad_h))

        # Pad batch dim up to a multiple of mini_batch_size.
        pad_b = (-n_valid) % self._mini_batch_size
        if pad_b:
            prev = nn.functional.pad(prev, (0, 0, 0, 0, 0, 0, 0, pad_b))
            cur = nn.functional.pad(cur, (0, 0, 0, 0, 0, 0, 0, pad_b))

        viz_on_gpu = self._viz_on_gpu and self._fits_on_gpu(n_valid, h, w)

        neuflow_started = time.perf_counter()
        chunks: list[torch.Tensor] = []
        for start in range(0, prev.shape[0], self._mini_batch_size):
            end = start + self._mini_batch_size
            flow_list = self._model(prev[start:end].to(self._device), cur[start:end].to(self._device))
            # Staying on device when the visualisation will also run there saves the round
            # trip, but it also means this loop no longer synchronises per mini-batch, so the
            # timing split below shifts GPU time from neuflow_sec into flow_to_image_sec.
            # The explicit sync keeps the two attributable.
            chunks.append(flow_list[-1].detach() if viz_on_gpu else flow_list[-1].detach().cpu())
        flows = torch.cat(chunks, dim=0)[:n_valid]  # trim batch padding.
        # Trim spatial padding.
        if pad_h or pad_w:
            flows = flows[..., :h, :w]
        if viz_on_gpu:
            torch.cuda.synchronize(self._device)
        neuflow_sec = time.perf_counter() - neuflow_started

        viz_started = time.perf_counter()
        viz = flow_to_image(flows.float())
        if viz_on_gpu:
            viz = viz.cpu()
        if timings is not None:
            timings.neuflow_sec += neuflow_sec
            timings.flow_to_image_sec += time.perf_counter() - viz_started
        return viz

    def _fits_on_gpu(self, n_frames: int, h: int, w: int) -> bool:
        """Whether the whole-clip flow + visualisation will fit in device memory.

        ``flow_to_image`` normalises over its entire input, so it cannot be chunked without
        changing the signal (measured: max abs diff 99/255 on 16% of pixels). That makes this a
        whole-tensor decision: either everything fits or the visualisation runs on CPU. A 250
        frame clip needs ~2 GB, but a 296 s whole video needs ~50 GB, so long videos fall back
        rather than OOM mid-run.

        **The 0.5 is carrying an underestimate, not providing spare margin.** Reviewed
        2026-07-30 against this venv's ``torchvision._normalized_flow_to_image``: its two
        int64 index tensors alone are 16 B/px, which already exceeds the whole ``3 * 4``
        allowance below, and with the float copies, the normalized flow and the per-channel
        temporaries the real peak is ~60-70 B/px against the formula's 46. The guard is safe
        only because ``< free * 0.5`` pushes the effective threshold to ~92 B/px. **Do not
        raise 0.5 toward 1.0 without first measuring the true peak working set** -- the
        formula, not the factor, is the thing that would need fixing.
        """
        # float32 flow (2ch) + float32 intermediate + uint8 output. Known to underestimate; see
        # the docstring.
        needed = n_frames * h * w * (2 * 4 + 3 * 4 + 3) * 2
        # driver-free plus what this process already holds but is not using. Without the
        # caching-allocator terms, the first clip's cached blocks depress driver-free and every
        # later clip of the same size sticks on the CPU path -- a one-way fallback that would
        # look like the guard working.
        free, _total = torch.cuda.mem_get_info(self._device)
        reserved = torch.cuda.memory_reserved(self._device)
        allocated = torch.cuda.memory_allocated(self._device)
        free = free + max(0, reserved - allocated)
        if needed < free * 0.5:
            return True
        logger.info(
            'flow visualisation falling back to CPU: needs ~%.1f GiB, %.1f GiB free',
            needed / 2**30,
            free / 2**30,
        )
        return False

    # ---- Sibling-cache facade --------------------------------------------
    def _sibling_paths(self, video_path: Path) -> tuple[Path, Path]:
        """Compose the ``_resized.mp4`` + ``_flow.mp4`` sibling paths.

        When ``max_pixels_override`` is set, a ``_mp<N>`` tag is inserted
        before both suffixes so that different pixel-budget configs cache
        independently (no clobbering of the default 786432-pixel siblings).
        """
        flow_suffix = self._cfg.sibling_suffix
        resized_suffix = self._RESIZED_SUFFIX
        if self._max_pixels_override is not None:
            tag = f'_mp{self._max_pixels_override}'
            # Insert tag before the file extension, robust to non-.mp4 suffixes.
            flow_suffix = _insert_before_ext(flow_suffix, tag)
            resized_suffix = _insert_before_ext(resized_suffix, tag)
        if self._cfg.flow_codec != 'libx264':
            # Only the flow file changes with the codec -- `_resized.mp4` is produced by ffmpeg and
            # is identical either way, so tagging it too would force a pointless re-encode and
            # break reuse across the two arms. The dim check reads both, so they must stay in sync
            # on everything except this.
            flow_suffix = _insert_before_ext(flow_suffix, '_ll')
        flow_path = video_path.with_name(video_path.stem + flow_suffix)
        resized_path = video_path.with_name(video_path.stem + resized_suffix)
        return resized_path, flow_path

    def _existing_siblings_match_target(
        self,
        resized_path: Path,
        flow_path: Path,
        target_h: int,
        target_w: int,
    ) -> bool:
        """True iff both siblings exist AND are at the exact target dims."""
        if not (resized_path.exists() and flow_path.exists()):
            return False
        try:
            rh, rw = probe_video_hw(resized_path)
            fh, fw = probe_video_hw(flow_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning('sibling dim probe failed (%s), regenerating', exc)
            return False
        return (rh, rw) == (target_h, target_w) and (fh, fw) == (target_h, target_w)

    @torch.no_grad()
    def compute_flow_only(self, video_path: str | Path, out_path: str | Path) -> tuple[Path, FlowTimings]:
        """Run NeuFlow on an already-correctly-sized video, writing flow to ``out_path``.

        This is the decoupled half of :meth:`compute_and_cache`: no ffmpeg resize, no
        sibling-cache naming, output goes exactly where the caller says. Added 2026-07-29
        because the runtime scripts orchestrate the resize themselves and write into a temp
        directory that is deleted per item — the sibling-cache behaviour fights both.

        The input is assumed to already be at the smart-resize target (the caller either
        resized it or is using a dataset file that was resized upstream). Nothing is
        rescaled here.

        **All frames are processed in one call, deliberately, and this cannot be chunked.**
        ``torchvision.utils.flow_to_image`` normalises by the maximum flow magnitude over
        its whole input (``max_norm = sum(flow**2, dim=1).sqrt().max()``, torchvision
        ``utils.py:626``), so the visualisation of any given frame depends on the entire
        batch. A chunked version was written first and measured against a single-shot run
        on a 250-frame clip: max abs difference 99/255 with 16% of pixels changed. Chunking
        is therefore not an optimisation, it is a different signal. Memory scales with
        duration — roughly 76 GiB of host RAM for the longest video in this dataset
        (296 s / 7,400 frames), which is why callers running several of these in parallel
        need to watch total RAM rather than only VRAM.

        A consequence worth carrying into any comparison: whole-video flow and per-clip
        flow are **not** the same signal even for identical frames, because the normaliser
        is computed over different populations.
        """
        from torchcodec.decoders import VideoDecoder  # noqa: PLC0415 -- keeps import cost off --help

        video_path, out_path = Path(video_path), Path(out_path)
        timings = FlowTimings()

        decode_started = time.perf_counter()
        decoder = VideoDecoder(
            str(video_path),
            num_ffmpeg_threads=self._num_ffmpeg_threads,
            dimension_order='NCHW',
        )
        total = int(decoder.metadata.num_frames or 0)
        if total <= 0:
            raise RuntimeError(f'torchcodec reports no frames for {video_path}')
        # Exact rational, not a float: torchvision rounds a float rate to an integer, which made
        # the flow sibling declare 25/1 while its ffmpeg-produced RGB source declared 17525/702.
        # See `probe_video_fps_exact`. Falls back to the float on an unparseable rate.
        fps = _exact_fps_or(video_path, decoder.metadata.average_fps)

        rgb = decoder.get_frames_at(indices=list(range(total))).data
        timings.decode_sec = time.perf_counter() - decode_started

        flow_viz = self.compute(rgb, timings=timings)
        if flow_viz.shape[0] != total:
            raise RuntimeError(f'flow has {flow_viz.shape[0]} frames but the video has {total}')

        out_path.parent.mkdir(parents=True, exist_ok=True)
        encode_started = time.perf_counter()
        codec, options = FLOW_CODECS[self._cfg.flow_codec]
        write_flow_mp4(flow_viz, out_path, fps=fps, video_codec=codec, options=options)
        timings.encode_sec = time.perf_counter() - encode_started
        return out_path, timings

    def compute_and_cache(
        self,
        video_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ResizedAndFlow:
        """Materialize the ``_resized.mp4`` + ``_flow.mp4`` sibling pair.

        Both siblings are written at the Qwen3-VL-friendly resolution
        picked by ``smart_resize_for_qwen3vl`` (multiples of 32, seq-len
        penalty applied, ``nframes=250`` default). Returns a
        :class:`ResizedAndFlow` dataclass with both paths + target dims.

        Behavior by ``self._cfg.mode``:

        * ``sibling``: refuse to compute. Both siblings must exist AND match
          the recomputed target dims, otherwise raise ``FileNotFoundError``.
        * ``online``: always recompute both.
        * ``online_with_cache``: reuse if both exist and dims match; else
          regenerate both.
        """
        video_path = Path(video_path)
        resized_path, flow_path = self._sibling_paths(video_path)

        # Probe source dims → compute target dims via FULL smart_resize
        # formula (nframes=250 + seq_len penalty; matches training pipeline).
        src_h, src_w = probe_video_hw(video_path)
        target_h, target_w = smart_resize_for_qwen3vl(
            src_h,
            src_w,
            nframes=self._nframes_for_resize,
            image_patch_size=self._image_patch_size,
            max_pixels_override=self._max_pixels_override,
        )
        logger.info(
            'Source %s (%dx%d) → smart_resize target %dx%d (nframes=%d, patch=%d, mp_override=%s)',
            video_path.name,
            src_h,
            src_w,
            target_h,
            target_w,
            self._nframes_for_resize,
            self._image_patch_size,
            self._max_pixels_override,
        )

        if self._cfg.mode == 'sibling':
            if not self._existing_siblings_match_target(
                resized_path,
                flow_path,
                target_h,
                target_w,
            ):
                raise FileNotFoundError(
                    f"FlowConfig.mode='sibling' requires both {resized_path} and "
                    f"{flow_path} to exist AT dim ({target_h}, {target_w}). "
                    'Precompute the sibling files first, or switch to '
                    "mode='online_with_cache'."
                )
            return ResizedAndFlow(resized_path, flow_path, target_h, target_w)

        if (
            not overwrite
            and self._cfg.mode == 'online_with_cache'
            and self._existing_siblings_match_target(resized_path, flow_path, target_h, target_w)
        ):
            logger.info('Reusing cached siblings: %s, %s', resized_path.name, flow_path.name)
            return ResizedAndFlow(resized_path, flow_path, target_h, target_w)

        # Step 1: ffmpeg scale the source to target dims → _resized.mp4.
        # Matches parent's ``resize_video_for_qwen3vl`` byte-for-byte on
        # command flags (see resize_video_helper.py).
        logger.info('Step 1/2: ffmpeg scale → %s', resized_path.name)
        resize_video_ffmpeg(
            video_path,
            resized_path,
            target_h=target_h,
            target_w=target_w,
            threads=self._num_ffmpeg_threads,
            overwrite=True,
        )

        # Step 2: decode _resized.mp4, run NeuFlow, write _flow.mp4.
        # (Both files end up at exactly the target dim — no downstream resize
        # needed in pipeline.)
        logger.info('Step 2/2: NeuFlow on resized → %s', flow_path.name)
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(
            str(resized_path),
            num_ffmpeg_threads=self._num_ffmpeg_threads,
            dimension_order='NCHW',
        )
        total = int(decoder.metadata.num_frames or 0)
        if total <= 0:
            raise RuntimeError(f'torchcodec reports no frames for {resized_path}')

        rgb = decoder.get_frames_at(indices=list(range(total))).data
        # Exact rational, not a float: torchvision rounds a float rate to an integer, which made
        # the flow sibling declare 25/1 while its ffmpeg-produced RGB source declared 17525/702.
        # See `probe_video_fps_exact`. Falls back to the float on an unparseable rate.
        fps = _exact_fps_or(resized_path, decoder.metadata.average_fps)
        # Sanity: the ffmpeg-produced _resized.mp4 dim should match target.
        if rgb.shape[-2:] != (target_h, target_w):
            logger.warning(
                'Decoded _resized.mp4 dim %s != target (%d,%d); proceeding anyway',
                tuple(rgb.shape[-2:]),
                target_h,
                target_w,
            )
        flow_viz = self.compute(rgb)
        codec, options = FLOW_CODECS[self._cfg.flow_codec]
        write_flow_mp4(flow_viz, flow_path, fps=fps, video_codec=codec, options=options)

        return ResizedAndFlow(resized_path, flow_path, target_h, target_w)
