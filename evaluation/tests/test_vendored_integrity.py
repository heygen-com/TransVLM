"""Pin the vendored files so an edit cannot pass unnoticed.

These three files decide which predictions count as matching which ground truth. An edit that
looks harmless -- a reordered comparison, a changed rounding -- shifts every number this package
reports, and the shift is invisible because the result still looks like a plausible score. So the
bytes are hashed rather than trusted.

If a hash fails: do not update the constant to make it green. Work out what changed and why. The
only legitimate reason to update these is deliberately re-vendoring from upstream, and then the
whole scoring path needs re-verifying against a known reference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_VENDORED = Path(__file__).resolve().parents[1] / 'transvlm_eval' / '_vendored'

#: sha256 of each vendored file as shipped.
EXPECTED = {
    'segment_operations.py': 'f846fa33416440ab1176dd7d3c21858f9c72df443dfad9353dd8ef9b27b040a2',
    'shot_boundary_types.py': '8a302fb3fdd79bc8680e9366b185cdac300e06725d3dd0c2fcc9198350ee122a',
}


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_vendored_file_is_unmodified(name: str):
    digest = hashlib.sha256((_VENDORED / name).read_bytes()).hexdigest()
    assert digest == EXPECTED[name], f'{name} changed; see this module docstring before touching the constant'


def test_list_match_has_no_module_level_tqdm_import():
    """The one deliberate deviation from byte-identical, asserted so it stays deliberate.

    ``tqdm`` was a module-scope import, making it a hard dependency of the scorer even though every
    call on this path passes ``show_progress=False``. It now lives inside those unreachable
    branches. Hashing this file would pin the deviation but not explain it, so the property is
    asserted instead.
    """
    source = (_VENDORED / 'list_match.py').read_text()
    lines = source.splitlines()
    module_level = [ln for ln in lines if ln.startswith(('import ', 'from '))]
    assert not any('tqdm' in ln for ln in module_level), 'tqdm is back at module scope'
    assert source.count('from tqdm import tqdm') == 3, 'expected one lazy import per progress branch'


def test_the_matcher_is_actually_reachable():
    """Guard against the hashes passing while the imports point somewhere else entirely."""
    from transvlm_eval._vendored.list_match import match_lists_v3
    from transvlm_eval._vendored.segment_operations import merge_segments
    from transvlm_eval._vendored.shot_boundary_types import ShotBoundary

    assert callable(match_lists_v3)
    assert callable(merge_segments)
    assert ShotBoundary is not None
