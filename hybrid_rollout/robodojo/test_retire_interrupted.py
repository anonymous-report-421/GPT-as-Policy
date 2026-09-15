from pathlib import Path

import pytest

from ..annotation_dashboard import Catalog
from .case_ledger import build_ledger
from .retire_interrupted import move_recoverably
from .test_case_ledger import attempt
from .test_evaluation import fake_panel_file


def test_retirement_keeps_ledger_and_raw_bytes_but_hides_video_catalog(tmp_path):
    _,_,panel=fake_panel_file(tmp_path)
    batch=attempt(tmp_path,panel,panel['cases'][6],state='failed')
    result=tmp_path/'results/exp'; trash=tmp_path/'results/.retired';trash.mkdir()
    before=build_ledger(tmp_path,panel)
    raw=batch.read_bytes()
    move_recoverably(result,trash)
    assert result.is_symlink() and batch.read_bytes()==raw
    after=build_ledger(tmp_path,panel)
    before.pop('updated_utc');after.pop('updated_utc')
    assert after==before
    assert Catalog(tmp_path/'results').scan()==[]
    with pytest.raises(ValueError):move_recoverably(result,trash)


def test_nonhidden_or_cross_root_destination_rejected(tmp_path):
    result=tmp_path/'exp';result.mkdir();trash=tmp_path/'trash';trash.mkdir()
    with pytest.raises(ValueError):move_recoverably(result,trash)
    assert result.is_dir() and not result.is_symlink()
