"""
Runtime patches for vLLM + custom Qwen3VL (6-channel, custom processor).

This file is automatically imported by Python at startup when its parent
directory is on PYTHONPATH.  It is designed to work with ``spawn``
multiprocessing so that both the main process and EngineCore child processes
apply the patches.

Patches applied (when VLLM_QWEN3VL_6CH_DUMMY=1):
  1. **6-channel dummy video**: Override vLLM's Qwen3VL dummy video builder
     to produce torch.Tensor inputs with the correct number of channels
     (default 6) so that memory profiling does not crash on the modified
     Qwen3VL patch embedding layer.
  2. **Processor monkey patch**: Replace the upstream
     ``transformers.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor``
     with ``Qwen3VLProcessorOurs`` which correctly handles video timestamp
     placeholders when vLLM passes bare video tokens.

Activation:
    Set the environment variable VLLM_QWEN3VL_6CH_DUMMY=1 to enable both
    patches.  Without this variable (or with any other value), no patching
    occurs.

Usage:
    PYTHONPATH=/path/to/runtime_patches:$PYTHONPATH \\
    VLLM_QWEN3VL_6CH_DUMMY=1 \\
    python your_script.py
"""

import os
import sys

_ENABLE_FLAG = os.environ.get("VLLM_QWEN3VL_6CH_DUMMY", "0")

if _ENABLE_FLAG == "1":
    try:
        import torch

        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.model_executor.models.qwen3_vl import (
            Qwen3VLForConditionalGeneration,
            Qwen3VLMultiModalProcessor,
            Qwen3VLProcessingInfo,
            Qwen3VLDummyInputsBuilder,
        )

        # Number of input channels expected by the modified model
        _NUM_CHANNELS = int(os.environ.get("VLLM_QWEN3VL_NUM_CHANNELS", "6"))

        class Qwen3VL6ChDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
            """
            Overrides dummy video generation to produce **torch.Tensor**
            dummy videos in channels-first layout (T, C, H, W) with the
            correct number of channels (default 6).

            Using torch.Tensor instead of np.ndarray avoids the Transformers
            ``infer_channel_dimension_format`` path which only accepts 1 or 3
            channels for numpy arrays, thereby preventing the ValueError
            during vLLM's memory profiling pass.
            """

            def _get_dummy_videos(
                self,
                *,
                width: int,
                height: int,
                num_frames: int,
                num_videos: int,
            ):
                # Build channels-first torch tensor: (T, C, H, W)
                # RGB channels filled with 1.0, remaining channels (e.g. OF)
                # filled with 0.0.  Using float32 to match typical processor
                # output dtype.
                video = torch.zeros(
                    (num_frames, _NUM_CHANNELS, height, width),
                    dtype=torch.float32,
                )
                # Fill RGB channels (first 3) with white (1.0)
                video[:, 3:, :, :] = 255.0

                video_items = []
                for _ in range(num_videos):
                    video_metadata = {
                        "fps": 2.0,
                        "duration": num_frames / 2.0,
                        "total_num_frames": num_frames,
                        "frames_indices": list(range(num_frames)),
                        "video_backend": "opencv",
                        "do_sample_frames": False,
                    }
                    video_items.append((video.clone(), video_metadata))
                return video_items

        # Re-register the processor factory on the model class, replacing
        # only the dummy_inputs builder while keeping info and processor
        # identical to the upstream defaults.
        MULTIMODAL_REGISTRY.register_processor(
            Qwen3VLMultiModalProcessor,
            info=Qwen3VLProcessingInfo,
            dummy_inputs=Qwen3VL6ChDummyInputsBuilder,
        )(Qwen3VLForConditionalGeneration)

        print(
            f"[sitecustomize] Qwen3VL 6-channel dummy patch applied "
            f"(channels={_NUM_CHANNELS}, dtype=torch.float32, "
            f"layout=channels-first).",
            file=sys.stderr,
        )

    except ImportError:
        # vLLM is not installed in this environment; skip silently.
        pass
    except Exception as exc:
        print(
            f"[sitecustomize] WARNING: Failed to apply Qwen3VL 6-channel "
            f"dummy patch: {exc}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Patch 2: Replace transformers Qwen3VLProcessor with our custom
    # version that handles bare video tokens from vLLM correctly.
    # ------------------------------------------------------------------
    try:
        from qwen3vl_processor_ours import (  # noqa: E402
            Qwen3VLProcessorOurs,
        )
        import transformers.models.qwen3_vl.processing_qwen3_vl as _qwen3_vl_mod

        _qwen3_vl_mod.Qwen3VLProcessor = Qwen3VLProcessorOurs

        print(
            f"[sitecustomize] Qwen3VLProcessor monkey patch applied "
            f"(replaced with {Qwen3VLProcessorOurs.__name__}).",
            file=sys.stderr,
        )

    except ImportError as exc:
        print(
            f"[sitecustomize] WARNING: Could not apply Qwen3VLProcessor "
            f"monkey patch (import error): {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[sitecustomize] WARNING: Failed to apply Qwen3VLProcessor "
            f"monkey patch: {exc}",
            file=sys.stderr,
        )
