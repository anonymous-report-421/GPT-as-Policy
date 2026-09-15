import copy
import json

import pytest

from .score_table import build, export, score_summary


def test_missing_score_is_not_imputed_or_silently_dropped():
    rows=[dict(native_success=True,native_score=1),dict(native_success=None,native_score=None,
        evaluation_success=False,native_complete=False,evaluation_failure_reason='simulator_rpc_idle_timeout_900s')]
    r=score_summary(rows)
    assert r==dict(n=2,score_n=1,native_complete=1,successes=1,idle_failures=1,
        mean_score_100=None,available_mean_score_100=100)


@pytest.mark.parametrize('v',[True,-.1,1.1,float('nan'),float('inf')])
def test_invalid_scores_rejected(v):
    with pytest.raises(ValueError):score_summary([dict(native_score=v,native_success=False)])


def test_complete_ten_task_table_and_average_without_overwrite(tmp_path):
    from .test_two_stage import certificate
    _,_,d=certificate(tmp_path);left=d['selection'];right=copy.deepcopy(left)
    for r in left['cases']+right['cases']:r['native_score']=.5
    right['cases'][0].update(native_score=None,native_success=None,evaluation_success=False,
        native_complete=False,evaluation_failure_reason='simulator_rpc_idle_timeout_900s')
    right['native_successes']=7
    report=dict(hybrid=left,direct=right);rows,_=build(report)
    assert len(rows)==11 and rows[-1]['direct_score_n']==49
    assert rows[-1]['direct_mean_score_100'] is None and rows[-1]['hybrid_mean_score_100']==50
    source=tmp_path/'results.json';source.write_text(json.dumps(report))
    out=tmp_path/'table';export(source,out)
    assert 'N/A' in (out/'README.md').read_text() and '49/50' in (out/'README.md').read_text()
    with pytest.raises(ValueError,match='Preserve'):export(source,out)
