# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""Our copyright goes on our files, and nowhere else.

Claiming copyright over vendored upstream code is worse than claiming nothing, so the boundary
is worth pinning: the next person to run a bulk header script over the tree should fail here
rather than assert ownership of NeuFlow.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

#: Paths holding third-party code. Their attribution belongs to the upstream project.
_VENDORED = (
    'inference/transvlm/data/_neuflow/',
    'inference/transvlm/data/_vision_process_of.py',
    'inference/transvlm/models/_vllm_patches/qwen3vl_processor_ours.py',
    'inference/transvlm/models/_vllm_patches/sitecustomize.py',
    'inference/transvlm/models/_sglang_patches/qwen_vl_processor_6ch.py',
)
_NOTICE = 'Copyright © 2026 HeyGen'


def _tracked_python_files() -> list[str]:
    out = subprocess.run(
        ['git', '-C', str(_ROOT), 'ls-files', '*.py'], capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def test_vendored_files_do_not_carry_our_copyright():
    claimed = [
        f
        for f in _tracked_python_files()
        if any(v in f for v in _VENDORED) and _NOTICE in (_ROOT / f).read_text()
    ]
    assert not claimed, f'our copyright asserted over vendored code: {claimed}'


def test_our_own_files_do_carry_it():
    missing = [
        f
        for f in _tracked_python_files()
        if not any(v in f for v in _VENDORED)
        and 'inference/' in f
        and _NOTICE not in (_ROOT / f).read_text()
    ]
    assert not missing, f'our files without a copyright header: {missing}'


def test_the_upstream_licence_copy_is_left_verbatim():
    """The Apache-2.0 appendix placeholder stays a placeholder in NeuFlow's own copy.

    Filling it with our name would read as a claim over NeuFlow's code. Our own LICENSE is a
    different file and does carry the notice.
    """
    upstream = (_ROOT / 'inference/transvlm/data/_neuflow/LICENSE').read_text()
    assert 'Copyright [yyyy] [name of copyright owner]' in upstream
    assert 'HeyGen' not in upstream
