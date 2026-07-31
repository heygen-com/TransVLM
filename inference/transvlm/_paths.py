"""Where the package's own files live.

Single source of truth, anchored on the package directory rather than a repo root.

The original code derived everything from ``Path(__file__).resolve().parents[N]`` with a different
N per module, which encodes the directory nesting into arithmetic: move a file one level and the
constant silently points somewhere else. Anchoring on ``__file__.parent`` of this module removes
the arithmetic entirely and keeps working when the package is installed into site-packages, where
no "repo root" exists at all.

The prompt lives inside the package for the same reason. It is not optional documentation -- change
the prompt and you change the task, so a checkpoint's numbers are only reproducible with the prompt
it was evaluated against. Shipping it as package data means it travels with the import.
"""

from __future__ import annotations

from pathlib import Path

#: Directory containing the ``transvlm`` package.
PACKAGE_DIR = Path(__file__).resolve().parent

#: Directory that must be on ``sys.path`` / ``PYTHONPATH`` for ``import transvlm`` to resolve.
#: SGLang spawns subprocesses that import the mm-processor package by name, so they need this too.
PACKAGE_PARENT = PACKAGE_DIR.parent

#: Runtime patches injected into vLLM via ``PYTHONPATH`` / ``sitecustomize``. Referenced as a
#: *path* rather than imported, so packaging tools that only follow imports will miss it.
VLLM_PATCH_DIR = PACKAGE_DIR / 'models' / '_vllm_patches'

#: The system prompt the released checkpoint was evaluated with.
DEFAULT_PROMPT_FILE = PACKAGE_DIR / 'assets' / 'prompts' / 'prompt_only_timestamps.txt'
