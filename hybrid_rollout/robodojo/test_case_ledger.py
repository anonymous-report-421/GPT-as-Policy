import json
from pathlib import Path

import pytest

from .case_ledger import build_ledger, refresh, require_available
from .evaluation import case_identity, selected_cases
from .test_evaluation import fake_panel_file


def attempt(tmp_path, panel, case, success=True, state='completed'):
    archive = tmp_path/'results/exp/archive'
    (archive/'controller').mkdir(parents=True)
    (archive/'sim').mkdir()
    identity = case_identity(panel, case)
    (archive/'controller/result.json').write_text(json.dumps(dict(
        complete=True, step_id=10, evaluation_case=identity)))
    (archive/'sim/evaluation_outcome.json').write_text(json.dumps(dict(
        complete=True, valid_for_success_rate=True, native_success=success, evaluation_case=identity)))
    path = tmp_path/'results/exp/_replicas/replica_1/attempt_0/batch.json'
    path.parent.mkdir(parents=True)
    row = dict(identity, state=state, archive=str(archive), artifact_status='verified',
               reset=dict(all_owned_processes_exited=True, ports_released=True))
    path.write_text(json.dumps(dict(panel_sha256=panel['panel_sha256'],
        state='completed' if state=='completed' else 'running', attempt='0',
        model='historical_model', provider='historical_gateway', episodes=[row])))
    return path


@pytest.mark.parametrize('success', [True, False])
def test_completed_case_is_not_repeated_and_identity_preserved(tmp_path, success):
    _, panel_path, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    attempt(tmp_path, panel, case, success)
    output, ledger = refresh(tmp_path, panel_path)
    assert output.is_file()
    assert ledger['status_counts'] == {'not_started': 59, 'completed': 1}
    assert ledger['next_case_id'] == panel['cases'][7]['case_id']
    row = ledger['cases'][6]
    assert row['attempts'][0]['provider'] == 'historical_gateway'
    assert row['attempts'][0]['native_success'] is success
    assert row['eval_seed'] == 0 and row['reset_seed'] == 0
    with pytest.raises(ValueError, match='must not be repeated'):
        require_available(tmp_path, panel, case['case_id'])


def test_running_is_not_assumed_dead(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    attempt(tmp_path, panel, case, state='starting')
    ledger = build_ledger(tmp_path, panel)
    assert ledger['cases'][6]['state'] == 'running_or_unreconciled'
    with pytest.raises(ValueError):
        require_available(tmp_path, panel, case['case_id'])


def test_explicit_second_case_cannot_default_to_first(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][7]
    assert selected_cases(panel, case['task'], 1, 1, case['case_id']) == [case]
    assert case['layout_id'] == case['reset_seed'] == 1 and case['eval_seed'] == 0
    with pytest.raises(ValueError):
        selected_cases(panel, case['task'], 1, 6, case['case_id'])
    with pytest.raises(ValueError):
        selected_cases(panel, case['task'], 8, 1, case['case_id'])


def test_uncertain_submission_is_not_retried(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][7]
    directory = tmp_path/'cluster/new/replica_1'
    directory.mkdir(parents=True)
    (directory/'submission_started.json').write_text('{}')
    (directory/'submission.json').write_text(json.dumps(dict(
        evaluation_panel_sha256=panel['panel_sha256'], evaluation_cases=[case],
        batch_result=str(tmp_path/'results/new/_replicas/replica_1/attempt_0/batch.json'),
        environment={'ROLLOUT_ATTEMPT': '0'}, model='gpt-6-astra')))
    with pytest.raises(ValueError, match='running_or_unreconciled'):
        require_available(tmp_path, panel, case['case_id'])
    require_available(tmp_path, panel, case['case_id'],
        tmp_path/'results/new/_replicas/replica_1/attempt_0/batch.json')
