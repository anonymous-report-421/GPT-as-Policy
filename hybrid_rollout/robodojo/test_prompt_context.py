"""Prompt-only metadata and future-only migration, no model/cluster/physics."""
import copy
import hashlib
import json

import pytest

from . import campaign
from .context_migration import revise
from .io import write_json
from .io import sha256
from .prompt_context import CONTEXT_VERSION
from .robodojo_server.client import RoboDojoTools
from .test_contract import BenchmarkSim, Student, invoke, response


def test_context_budget_tracks_native_steps_without_new_queries(tmp_path):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=BenchmarkSim, max_decisions=0)
    obs = invoke(tools)
    assert (obs['step_id'], obs['max_episode_steps'], obs['remaining_steps'], obs['rollout_finished']) == (0, 2, 2, False)
    assert obs['task_context']['requires_arm_return'] is True
    assert 'digits' in obs['task_context']['process']
    proposal = invoke(tools)
    assert (proposal['step_id'], proposal['max_episode_steps'], proposal['remaining_steps'], proposal['rollout_finished']) == (0, 2, 2, False)
    decision = response(tools.request); decision['steps'] = 1
    obs = invoke(tools, response=decision)
    assert 'task_context' not in obs  # First observation only; no repetitive context cost.
    assert (obs['step_id'], obs['max_episode_steps'], obs['remaining_steps'], obs['rollout_finished']) == (1, 2, 1, False)
    invoke(tools)
    decision = response(tools.request); decision['steps'] = 1
    decision['assessment']['execution_status'] = 'progressing'
    final = invoke(tools, response=decision)
    assert (final['step_id'], final['max_episode_steps'], final['remaining_steps'], final['rollout_finished']) == (2, 2, 0, True)
    assert final['result']['complete'] and not final['result']['success']
    assert json.loads((tmp_path/'run.json').read_text())['context_version'] == CONTEXT_VERSION == 'v3'
    assert not any('score' in op or 'reward' in op for op in tools.sim.calls)
    assert tools.sim.calls.count('reset') == 1


def test_task_notes_home_exception_and_variants():
    from .prompt_context import task_context, TASK_NOTES
    from .evaluation import TASKS
    assert set(TASK_NOTES) == set(TASKS)
    for task in TASKS:
        note = task_context(task)
        assert note == task_context(task+'_random')
        assert note['requires_arm_return'] == (task != 'make_kong')
    assert task_context('unknown_task') == {}


def sample():
    plan = dict(experiment_prefix='campaign', skipped_completed=['prior'],
        queue=[dict(case=dict(case_id=f'case{i}', reset_seed=i), experiment_prefix='old', initial_attempt=0) for i in range(4)])
    state = dict(slots=[dict(id=0, index=0, status='running', assigned=[0], submission='launched', attempt=2),
        dict(id=1, index=1, status='ready', assigned=[1], submission='prepared-not-sent', attempt=0)])
    bindings = {'0': dict(source_sha256='v1-sha', dispatcher_source='/immutable/v1', context_version='v1')}
    return plan, state, bindings


def test_cutover_preserves_running_identity_and_only_replaces_unlaunched():
    plan, state, bindings = sample()
    before = copy.deepcopy((plan, state, bindings))
    p, s = revise(plan, state, {0}, bindings)
    assert (plan, state, bindings) == before
    assert s['slots'][0] == state['slots'][0]
    assert p['queue'][0] == plan['queue'][0]
    assert s['slots'][1]['submission'] is None and s['slots'][1]['attempt'] == 0
    assert p['context_case_versions'] == {'prior':'v1', 'case0':'v1', 'case1':'v2', 'case2':'v2', 'case3':'v2'}
    for old, new in zip(plan['queue'], p['queue']):
        assert old['case'] == new['case']
    assert p['case_runtime_overrides'] == bindings
    assert p['queue'][1]['experiment_prefix'] != 'old'


def test_started_retry_cannot_silently_change_context():
    plan, state, bindings = sample()
    state['slots'][0].update(status='ready', submission=None, attempt=3)
    p, s = revise(plan, state, {0}, bindings)
    assert s['slots'][0] == state['slots'][0]
    assert p['case_runtime_overrides']['0']['context_version'] == 'v1'
    with pytest.raises(ValueError, match='Missing immutable runtime'):
        revise(plan, state, {0}, {})


def test_case_binding_follows_case_not_slot_and_rejects_wrong_context(tmp_path, monkeypatch):
    plan = dict(queue=[dict(case={'case_id': 'old'}), dict(case={'case_id':'new'})],
        source_sha256='new', context_version='v2', cluster='c', workspace='w', worker_spec='sku', quota='reserved',
        case_runtime_overrides={'0':dict(source_sha256='old', context_version='v1', dispatcher_source='/old')})
    path = tmp_path/'submission.json'
    monkeypatch.setattr(campaign, 'submission_path', lambda p, s:path)
    for index, source, context in [(0, 'old', 'v1'), (1, 'new', 'v2')]:
        sub = dict(source_sha256=source, cluster='c', workspace='w', worker_spec='sku', quota='reserved',
            environment={'ROLLOUT_AUTH_PROFILE':'codex_b'}, evaluation_cases=[plan['queue'][index]['case']])
        if context == 'v2': sub['context_version'] = context
        write_json(path, sub)
        slot = dict(id=3, index=index, auth_profile='codex_b')
        assert campaign.ensure_submission(plan, slot) == path
        sub['context_version'] = 'v2' if context == 'v1' else 'v1'
        write_json(path, sub)
        with pytest.raises(ValueError, match='differs'):
            campaign.ensure_submission(plan, slot)


@pytest.mark.parametrize('tamper', [False, True])
def test_activation_preserves_slots_manual_stop_and_parent_backup(tmp_path, tamper):
    from .context_migration import activate, REVISION
    (tmp_path/'evaluation').mkdir()
    parent = dict(approved_sha256='parent', slots=[dict(id=i, auth_profile=n, status='running', index=i)
                  for i, n in campaign.B5_A2_SLOTS.items()])
    write_json(tmp_path/'state.json', parent)
    write_json(tmp_path/'STOP', {'reason':'context cutover'})
    write_json(tmp_path/'STOP_SLOT_2', {'reason':'manual'})
    p = dict(shared_root=str(tmp_path), context_revision=REVISION, parent_plan_sha256='parent',
        slots=[dict(id=i, auth_profile=n) for i,n in campaign.B5_A2_SLOTS.items()],
        context_parent_state_sha256=sha256(tmp_path/'state.json'),
        context_stop_sha256=sha256(tmp_path/'STOP'),
        context_slot_stops={'STOP_SLOT_2':sha256(tmp_path/'STOP_SLOT_2')},
        context_state_payload_sha256=hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest())
    path=tmp_path/'plan.json'; write_json(path,p)
    prepared=dict(parent,approved_sha256=sha256(path))
    if tamper:
        prepared['slots']=[]
    write_json(tmp_path/f'state_{REVISION}_prepared.json',prepared)
    if tamper:
        with pytest.raises(ValueError,match='Prepared state changed'):
            activate(path,sha256(path))
        assert json.loads((tmp_path/'state.json').read_text()) == parent
        assert (tmp_path/'STOP').exists()
    else:
        activate(path,sha256(path))
        assert json.loads((tmp_path/f'state_before_{REVISION}.json').read_text()) == parent
        assert json.loads((tmp_path/'state.json').read_text())['slots'] == parent['slots']
        assert not (tmp_path/'STOP').exists()
        assert (tmp_path/'STOP_SLOT_2').exists()
