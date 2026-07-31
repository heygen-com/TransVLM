"""Keep ``dependencies = []`` honest.

`pyproject.toml` declares no runtime dependencies. That is easy to state and easy to break: one
convenience import of pandas at module scope inside any module the public API touches, and every
user needs it, while the metadata still says otherwise. Nobody notices, because the developer's
environment has it.

So the import is exercised with the usual suspects blocked. This already caught one real breakage:
`pipeline.py` imported the benchmark adapter at module scope for a type annotation, and that adapter
imported duckdb, so `from transvlm_eval import score` required an optional extra.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parents[1]

#: Anything a scorer might plausibly reach for that would silently become mandatory.
BLOCKED = ('duckdb', 'pandas', 'pyarrow', 'numpy', 'torch', 'tqdm', 'scipy')


def test_the_public_api_imports_with_every_optional_package_blocked():
    """Run in a subprocess: blocking imports in-process would leak into other tests."""
    program = textwrap.dedent(
        f"""
        import sys
        for name in {BLOCKED!r}:
            sys.modules[name] = None      # any import of these now raises

        from transvlm_eval import score
        gt = {{'v': [{{'start_time': 1.0, 'end_time': 1.04}}]}}
        per_tolerance, ttmean = score(gt, dict(gt), {{'v': 10.0}})
        assert per_tolerance[0.00]['cl_f1'] == 1.0
        print('OK')
        """
    )
    result = subprocess.run(
        [sys.executable, '-c', program], capture_output=True, text=True, cwd=_EVAL_DIR
    )
    assert result.returncode == 0, f'a blocked package is actually required:\n{result.stderr}'
    assert 'OK' in result.stdout


def test_the_block_list_would_actually_catch_something():
    """Guard the guard: if `sys.modules[name] = None` stopped raising, the test above is vacuous."""
    program = 'import sys; sys.modules["json"] = None; import json'
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    assert result.returncode != 0, 'blocking a module no longer raises; the test above proves nothing'
