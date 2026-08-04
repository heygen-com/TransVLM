# Copyright © 2026 HeyGen
# SPDX-License-Identifier: Apache-2.0

"""The mistakes a first-time user makes, and the messages they get for them.

Each of these produced a raw traceback or a message pointing at the wrong problem before these
checks existed. They are cheap to break again: any of them regresses the moment someone moves a
validation call or lets an optional import escape its try block.
"""

from __future__ import annotations

import pytest

from transvlm.inference.backends import missing_backend_exit, validate_ckpt_dir


def test_a_missing_ckpt_dir_is_named_as_such(tmp_path):
    """Not "Repo id must be in the form 'namespace/repo_name'", which is what from_pretrained says.

    That message sends someone who passed a local path off to look for a Hub name.
    """
    with pytest.raises(SystemExit) as exc:
        validate_ckpt_dir(tmp_path / 'not-there')
    assert 'does not exist' in str(exc.value)
    assert '--ckpt-dir' in str(exc.value)


def test_a_file_is_rejected_as_a_ckpt_dir(tmp_path):
    f = tmp_path / 'weights.safetensors'
    f.write_bytes(b'')
    with pytest.raises(SystemExit) as exc:
        validate_ckpt_dir(f)
    assert 'not a directory' in str(exc.value)


def test_a_directory_without_config_json_says_so_and_how_to_get_one(tmp_path):
    """The likely case: someone pointed at the parent of the checkpoint, or an empty download."""
    with pytest.raises(SystemExit) as exc:
        validate_ckpt_dir(tmp_path)
    msg = str(exc.value)
    assert 'config.json' in msg
    assert 'hf download' in msg


def test_a_real_checkpoint_layout_passes(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    assert validate_ckpt_dir(tmp_path) == tmp_path


def test_validation_happens_at_load_not_construction():
    """Constructing an engine must not require the weights to be on disk yet.

    The tests around these engines build one against a path that will never exist, and a caller
    may legitimately construct before fetching. Moving this check into __init__ broke nine tests
    the first time it was written; this pins the boundary.
    """
    from transvlm.inference.hf_clip_engine import HFClipInferenceEngine

    engine = HFClipInferenceEngine(ckpt_dir='/nonexistent/transvlm-ckpt')
    assert str(engine.ckpt_dir) == '/nonexistent/transvlm-ckpt'


def test_a_missing_optional_backend_points_at_the_readme():
    exc = missing_backend_exit('sglang', 'sglang', ImportError('No module named sglang'))
    msg = str(exc)
    assert 'sglang' in msg
    assert 'README' in msg
    assert '--backend hf' in msg, 'the message should name the backend that needs no install'
