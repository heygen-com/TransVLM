# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""SGLang 6-channel adapter patches — external multimodal processor package.

When SGLang boots with the environment variable
``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=transvlm.models._sglang_patches``
(set by ``transvlm.inference.sglang_clip_engine.prepare_environment``),
``sglang/srt/managers/multimodal_processor.py::import_processors`` scans this
package for subclasses whose ``models`` class attribute lists the loaded
``hf_config.architectures`` entry, and uses them in place of the upstream
processor.
"""
# Import the subclass so SGLang's package walker picks it up during
# ``import_processors(_PATCH_PKG, overwrite=True)``.
from transvlm.models._sglang_patches.qwen_vl_processor_6ch import (
    QwenVLImageProcessorOurs,
)

__all__ = ['QwenVLImageProcessorOurs']
