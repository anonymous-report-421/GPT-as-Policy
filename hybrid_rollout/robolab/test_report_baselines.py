import json

import pytest

from hybrid_rollout.robolab.report_baselines import select_first


def fixture(root, order=(5, 3, 0, 4, 1, 2)):
    task = root / 'Task'
    task.mkdir()
    (task / 'env_cfg.json').write_text('{}')
    rows = []
    for i in order:
        success = i >= 2
        row = dict(task_name='Task', run=0, episode=i, env_id=i, policy='pi05',
                   instruction_type='specific', success=success, episode_step=10)
        rows.append(row)
        (task / f'log_0_env{i}.json').write_text(json.dumps(dict(
            task='Task', run=0, env_id=i, success=success, final_step=10)))
    (root / 'episode_results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return rows


def test_first_five_uses_identifiers_and_retains_failures(tmp_path):
    fixture(tmp_path)
    result = select_first(tmp_path, ['Task'], 'pi05')['tasks'][0]
    assert [r['episode'] for r in result['trials']] == [0, 1, 2, 3, 4]
    assert result['successes'] == 3
    assert result['episodes'] == 5


def test_missing_trials_are_not_silently_filled(tmp_path):
    fixture(tmp_path, order=(0, 1, 2, 3))
    with pytest.raises(ValueError, match='needs 5 episodes'):
        select_first(tmp_path, ['Task'], 'pi05')


def test_duplicate_identifiers_are_rejected(tmp_path):
    fixture(tmp_path, order=(0, 1, 2, 3, 4, 4))
    with pytest.raises(ValueError, match='duplicate episode identifiers'):
        select_first(tmp_path, ['Task'], 'pi05')


def test_summary_log_disagreement_is_rejected(tmp_path):
    fixture(tmp_path)
    p = tmp_path / 'Task/log_0_env0.json'
    log = json.loads(p.read_text())
    log['success'] = True
    p.write_text(json.dumps(log))
    with pytest.raises(ValueError, match='disagrees with terminal log'):
        select_first(tmp_path, ['Task'], 'pi05')


def test_wrong_policy_or_missing_outcome_is_rejected(tmp_path):
    rows = fixture(tmp_path)
    with pytest.raises(ValueError, match='unexpected policy'):
        select_first(tmp_path, ['Task'], 'dreamzero')
    rows[2]['success'] = None
    (tmp_path / 'episode_results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='non-boolean outcome'):
        select_first(tmp_path, ['Task'], 'pi05')
