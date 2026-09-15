"""Offline dispatcher fault injection; no ACP/model calls or real credentials."""
import copy
import json
from pathlib import Path

import pytest

from . import campaign
from .case_ledger import build_ledger, require_available
from .codex_backend.profiles import CHATGPT_PROFILES, initialize, validate_campaign_sessions
from .evaluation import archive_path, case_identity
from .io import write_json
from .test_case_ledger import attempt
from .test_evaluation import fake_panel_file


@pytest.mark.parametrize('text,expected', [
    ('Codex error: 429 Too Many Requests', 'retry_transport'),
    ('Codex error: stream disconnected before completion', 'retry_transport'),
    ('HTTP 502 bad gateway', 'retry_transport'),
    ('error sending request: connection reset', 'retry_transport'),
    ('HTTP 429 insufficient_quota', 'pause_quota'),
    ('HTTP 429 credit balance exhausted', 'pause_quota'),
    ('HTTP 401 unauthorized', 'pause_auth'),
    ('refresh token failed', 'pause_auth'),
    ('RuntimeError: simulator timeout', 'pause_unknown_failure'),
    ('FileNotFoundError: checkpoint missing', 'pause_unknown_failure'),
])
def test_classification(text, expected):
    assert campaign.transport_classification(text) == expected


def test_explicit_account_repair_preserves_other_slots():
    from .repair_accounts import repaired_state
    state = dict(slots=[dict(id=i, status='running' if i<3 else 'pause_unknown_failure',
        index=i, attempt=2, submission=f'old-{i}', job_id=f'job-{i}', assigned=[i]) for i in range(7)])
    snapshot = copy.deepcopy(state)
    result = repaired_state(state)
    assert state == snapshot
    assert result['slots'][:3] == state['slots'][:3]
    for slot in result['slots'][3:]:
        assert slot['attempt'] == 3 and slot['status'] == 'ready'
        assert slot['assigned'] == [slot['index']]
        assert slot['history'][-1]['action'] == 'user_authorized_config_repair'
    state['slots'][3]['status'] = 'stopped'
    with pytest.raises(ValueError):
        repaired_state(state)


def test_runtime_override_does_not_change_other_slots(tmp_path, monkeypatch):
    plan = dict(queue=[dict(case={}, initial_attempt=0)], shared_root=str(tmp_path),
        experiment_prefix='test', source_sha256='old', cluster='cluster', workspace='workspace',
        worker_spec='sku', quota='reserved',
        slot_runtime_overrides={'3': dict(source_sha256='fixed', dispatcher_source='/fixed', https_proxy='http://proxy:3128')})
    sub = tmp_path/'submission.json'
    monkeypatch.setattr(campaign, 'submission_path', lambda p, s: sub)
    for slot_id, source, proxy in [(0, 'old', None), (3, 'fixed', 'http://proxy:3128')]:
        env = {'ROLLOUT_AUTH_PROFILE': 'galbot' if slot_id==0 else 'codex_b'}
        if proxy: env['ROLLOUT_HTTPS_PROXY'] = proxy
        write_json(sub, dict(source_sha256=source, cluster='cluster', workspace='workspace',
            worker_spec='sku', quota='reserved', environment=env, evaluation_cases=[{}]))
        assert campaign.ensure_submission(plan, dict(id=slot_id, index=0, auth_profile=env['ROLLOUT_AUTH_PROFILE'])) == sub
    assert plan['source_sha256'] == 'old'


def test_native_failure_not_retried_even_with_network_errors(tmp_path):
    (tmp_path/'sim').mkdir()
    (tmp_path/'controller').mkdir()
    identity = dict(case_id='fixed-case')
    write_json(tmp_path/'sim/evaluation_outcome.json', dict(evaluation_case=identity,
        complete=True, valid_for_success_rate=True, native_success=False))
    write_json(tmp_path/'controller/result.json', dict(evaluation_case=identity, complete=True))
    write_json(tmp_path/'controller/failure.json', dict(error='HTTP 502'))
    batch = dict(final_reset=dict(all_owned_processes_exited=True, ports_released=True))
    assert campaign.assess(batch, tmp_path, 'RUNNING', identity) == 'wait'
    assert campaign.assess(batch, tmp_path, 'SUCCEEDED', identity) == 'pause_terminal_artifacts'
    write_json(tmp_path/'artifact_manifest.json', dict(status='verified'))
    assert campaign.assess(batch, tmp_path, 'SUCCEEDED', identity) == 'completed'
    assert campaign.assess(batch, tmp_path, 'STOPPED', identity) == 'stopped'
    assert campaign.assess(dict(exit_code=130), tmp_path, 'FAILED', identity) == 'stopped'


def test_backoff_is_bounded_not_attempt_count():
    assert [campaign.retry_delay(i) for i in range(1, 7)] == [300, 600, 1200, 2400, 3600, 3600]
    assert campaign.retry_delay(10000) == 3600


def fair_submission_fixture(tmp_path, monkeypatch):
    slots = [dict(id=i, auth_profile=f'account_{i}', status='ready',
                  index=i, attempt=0, next_retry_at=0, assigned=[i]) for i in (2, 7, 14)]
    state = dict(slots=slots)
    calls = []
    def prepare(plan, slot):
        # Fairness is durable even if preparation/create subsequently fails.
        saved = json.loads((tmp_path/'state.json').read_text())
        assert saved['submission_round_robin_after_slot'] == slot['id']
        calls.append(slot['id'])
        return tmp_path/f"slot_{slot['id']}"/'submission.json'
    monkeypatch.setattr(campaign, 'ensure_submission', prepare)
    monkeypatch.setattr(campaign.acp, 'submit', lambda *a, **kw: None)
    return state, calls


def test_capped_submissions_do_not_starve_later_slots_after_restart(tmp_path, monkeypatch):
    state, calls = fair_submission_fixture(tmp_path, monkeypatch)
    ids = [s['id'] for s in state['slots']]
    for _ in range(6):
        campaign.tick({}, state, tmp_path, [], only_slots=set(ids), max_new_submissions=1)
        state = json.loads((tmp_path/'state.json').read_text())
        # Even if every submitted low slot finishes before the next poll,
        # every higher slot must still get a turn, without reordering identities.
        for slot in state['slots']:
            slot['status'] = 'ready'
        assert [s['id'] for s in state['slots']] == ids
        assert [s['assigned'] for s in state['slots']] == [[i] for i in ids]
    assert calls == [2, 7, 14, 2, 7, 14]


@pytest.mark.parametrize('gate', ['quota', 'backoff', 'slot_stop', 'subset', 'global_stop'])
def test_fair_submissions_preserve_all_existing_gates(tmp_path, monkeypatch, gate):
    state, calls = fair_submission_fixture(tmp_path, monkeypatch)
    state['submission_round_robin_after_slot'] = 14
    only = {2, 7, 14}
    if gate == 'quota':state['quota_guard'] = dict(blocked_profiles=['account_2'])
    if gate == 'backoff':state['slots'][0]['next_retry_at'] = campaign.time.time() + 3600
    if gate == 'slot_stop':(tmp_path/'STOP_SLOT_2').touch()
    if gate == 'subset':only.remove(2)
    if gate == 'global_stop':(tmp_path/'STOP').touch()
    campaign.tick({}, state, tmp_path, [], only_slots=only, max_new_submissions=1)
    assert calls == ([] if gate == 'global_stop' else [7])


def test_fairness_caps_creates_and_handles_absent_previous_slot(tmp_path, monkeypatch):
    state, calls = fair_submission_fixture(tmp_path, monkeypatch)
    state['submission_round_robin_after_slot'] = 999
    campaign.tick({}, state, tmp_path, [], max_new_submissions=2)
    assert calls == [2, 7]
    assert state['submission_round_robin_after_slot'] == 7


def test_fairness_does_not_issue_an_ambiguous_create_again(tmp_path, monkeypatch):
    state, calls = fair_submission_fixture(tmp_path, monkeypatch)
    started = tmp_path/'slot_2'/'submission_started.json'
    started.parent.mkdir(); started.write_text('{}')
    submitted = []
    monkeypatch.setattr(campaign.acp, 'submit', lambda p, **kw: submitted.append(p))
    campaign.tick({}, state, tmp_path, [], max_new_submissions=1)
    assert calls == [2] and submitted == []
    assert state['slots'][0]['status'] == 'submitted'


def test_capacity_is_retryable_without_model_fallback():
    assert campaign.transport_classification('Selected model is at capacity. Please try a different model.') == 'retry_transport'
    assert campaign.transport_classification('codexErrorInfo: serverOverloaded') == 'retry_transport'


def test_timeout_requires_real_rpc_transport_error(tmp_path):
    workspace = tmp_path/'controller/codex_workspace'
    workspace.mkdir(parents=True)
    write_json(tmp_path/'controller/failure.json', dict(
        error='TimeoutError: Codex made no completed service or native tool call within the configured timeout'))
    rpc = workspace/'rpc_out.jsonl'
    rpc.write_text(json.dumps(dict(method='item/completed', params=dict(text='HTTP 502')))+'\n')
    assert campaign.assess({}, tmp_path, 'FAILED', {}) == 'pause_unknown_failure'
    with rpc.open('a') as stream:
        stream.write(json.dumps(dict(method='error', params=dict(error=dict(message='HTTP 502'))))+'\n')
    assert campaign.assess({}, tmp_path, 'FAILED', {}) == 'retry_transport'


def test_independent_subscription_sessions(tmp_path):
    for i, name in enumerate(CHATGPT_PROFILES):
        home = initialize(name, tmp_path)
        write_json(home/'auth.json', dict(auth_mode='chatgpt', tokens=dict(
            access_token='fake-access', id_token='fake-id', refresh_token=f'fake-{i}',
            account_id='account-current' if name.startswith('codex_a') else 'account-new')))
        (home/'auth.json').chmod(0o600)
    validate_campaign_sessions(tmp_path)
    home = initialize('codex_b_3', tmp_path)
    data = json.loads((home/'auth.json').read_text())
    data['tokens']['refresh_token'] = json.loads(
        (tmp_path/'private/auth_profiles/codex_b/codex_home/auth.json').read_text())['tokens']['refresh_token']
    write_json(home/'auth.json', data)
    (home/'auth.json').chmod(0o600)
    with pytest.raises(ValueError, match='independent logins'):
        validate_campaign_sessions(tmp_path)


def test_slot_assignment_is_unique_and_keeps_native_replica():
    plan = dict(queue=[dict(initial_attempt=2, case=dict(replica_id=5)) for _ in range(10)],
                experiment_prefix='test', shared_root='/tmp/not-real')
    state = dict(slots=[dict(id=i, assigned=[]) for i in range(7)])
    for slot in state['slots']:
        campaign.reserve(state, slot, plan)
        assert slot['attempt'] == 2
        assert campaign.submission_path(plan, slot).parent.name == 'replica_5'
    assert len({s['index'] for s in state['slots']}) == 7
    campaign.reserve(state, state['slots'][0], plan)
    assert state['slots'][0]['index'] == 7


def dispatcher_fixture(tmp_path, monkeypatch):
    _, panel_file, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    plan = dict(shared_root=str(tmp_path), experiment_prefix='robodojo_test', eval_manifest=str(panel_file),
                queue=[dict(case=case, initial_attempt=0)], source_sha256='fake-sha')
    slot = dict(id=4, auth_profile='codex_b_2', index=0, attempt=0, assigned=[0],
                status='submitted', retry_number=0, next_retry_at=0)
    path = campaign.submission_path(plan, slot)
    path.parent.mkdir(parents=True)
    exp = path.parents[1].name
    batch_path = tmp_path/'results'/exp/'_replicas/replica_1/attempt_0/batch.json'
    submission = dict(batch_result=str(batch_path), environment=dict(ROLLOUT_EXPERIMENT_ID=exp,
        ROLLOUT_REPLICA_ID='1'), evaluation_cases=[case], evaluation_panel_sha256=panel['panel_sha256'])
    write_json(path, submission)
    write_json(path.parent/'submission_started.json', {})
    slot['submission'] = str(path)
    state = dict(slots=[slot])
    job = dict(name='pt-fixture', display_name=exp+'_r1', ownership=dict(user_name='sujiayi'), state='FAILED')
    archive = archive_path(tmp_path, exp, case, 1, 0)
    (archive/'controller').mkdir(parents=True)
    write_json(archive/'controller/failure.json', dict(error='Codex error: 502 Bad Gateway'))
    monkeypatch.setattr(campaign.acp, 'submit', lambda _: pytest.fail('must not submit here'))
    return plan, state, job, panel, path, batch_path, archive


def test_terminal_transport_retry_and_durable_recovery(tmp_path, monkeypatch):
    plan, state, job, panel, path, batch_path, archive = dispatcher_fixture(tmp_path, monkeypatch)
    batch_path.parent.mkdir(parents=True)
    # Simulate container dying before batch could flush its final state.
    write_json(batch_path, dict(state='running', panel_sha256=panel['panel_sha256'], attempt='0', episodes=[dict(
        case_identity(panel, plan['queue'][0]['case']), state='starting', archive=str(archive))]))
    campaign.tick(plan, state, tmp_path, [job])
    slot = state['slots'][0]
    assert slot['status'] == 'ready' and slot['attempt'] == 1 and slot['retry_number'] == 1
    assert slot['auth_profile'] == 'codex_b_2'
    assert slot['next_retry_at'] > campaign.time.time()
    assert json.loads(batch_path.read_text())['state'] == 'running'  # never rewrite old evidence
    require_available(tmp_path, panel, plan['queue'][0]['case']['case_id'])
    recovered = json.loads((tmp_path/'state.json').read_text())
    campaign.tick(plan, recovered, tmp_path, [job])  # still in backoff; no duplicate attempt
    assert recovered['slots'][0]['attempt'] == 1


@pytest.mark.parametrize('stop_mode', ['file', 'platform'])
def test_explicit_stop_never_retries(tmp_path, monkeypatch, stop_mode):
    plan, state, job, *_ = dispatcher_fixture(tmp_path, monkeypatch)
    if stop_mode == 'file':
        write_json(tmp_path/'STOP_SLOT_4', {})
    else:
        job['state'] = 'STOPPED'
    campaign.tick(plan, state, tmp_path, [job])
    assert state['slots'][0]['status'] == 'stopped'
    assert state['slots'][0]['attempt'] == 0


def test_ambiguous_create_not_reissued(tmp_path, monkeypatch):
    plan, state, job, *_ = dispatcher_fixture(tmp_path, monkeypatch)
    state['slots'][0]['status'] = 'submitting'
    campaign.tick(plan, state, tmp_path, [])
    assert state['slots'][0]['status'] == 'pause_ambiguous_submission'


def test_native_complete_preserved_by_reconciliation(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    batch_path = attempt(tmp_path, panel, case, success=False)
    root = tmp_path/'cluster/exp/replica_1'
    root.mkdir(parents=True)
    write_json(root/'submission.json', dict(batch_result=str(batch_path), evaluation_cases=[case],
        evaluation_panel_sha256=panel['panel_sha256']))
    write_json(root/'scheduler_reconciliation.json', dict(
        schema='hybrid_rollout.robodojo.scheduler_reconciliation.v1', old_container_terminal=True,
        platform_state='FAILED', action='retry_transport', job_id='pt-fake',
        batch_path=str(batch_path), case_id=case['case_id']))
    assert build_ledger(tmp_path, panel)['cases'][6]['state'] == 'completed'


def test_reserved_quota_is_forwarded_for_new_attempt(tmp_path, monkeypatch):
    plan = dict(shared_root=str(tmp_path), experiment_prefix='reserved_test',
        queue=[dict(case=dict(replica_id=4, task='pack_objects_into_box', case_id='fixed'))],
        source_sha256='reviewed', eval_manifest='/fixture/panel.json', codex='/fixture/codex',
        sco='/fixture/sco', image='reviewed-image', workspace='galbot-foundation-model',
        cluster='galbot-vla', worker_spec='n5lp.nn.a80.2', quota='reserved')
    slot = dict(id=3, index=0, attempt=9, auth_profile='codex_b')
    def fake_prepare(args):
        assert args.quota_type == 'reserved'
        assert args.cluster == 'galbot-vla' and args.worker_spec == 'n5lp.nn.a80.2'
        assert args.replica_id == 4 and args.attempt == 9 and args.count == 1
        path = campaign.submission_path(plan, slot)
        path.parent.mkdir(parents=True)
        write_json(path, dict(source_sha256='reviewed'))
    monkeypatch.setattr(campaign.acp, 'prepare', fake_prepare)
    campaign.ensure_submission(plan, slot)


def test_invalid_quota_rejected_before_any_io():
    from argparse import Namespace
    with pytest.raises(ValueError, match='quota_type'):
        campaign.acp.prepare(Namespace(quota_type='auto'))


def test_fixed_stability_routing_and_new_namespace():
    plan = dict(experiment_prefix='base', shared_root='/tmp/fixture', queue=[
        dict(slot_id=3, initial_attempt=2, experiment_prefix='base_stable', case=dict(replica_id=2)),
        dict(slot_id=2, initial_attempt=0, experiment_prefix='base_stable', case=dict(replica_id=0))])
    state = dict(slots=[dict(id=2, assigned=[]), dict(id=3, assigned=[])])
    campaign.reserve(state, state['slots'][0], plan)
    campaign.reserve(state, state['slots'][1], plan)
    assert state['slots'][0]['index'] == 1 and state['slots'][1]['index'] == 0
    assert 'base_stable_c0_a2' in str(campaign.submission_path(plan, state['slots'][1]))


def test_dispatcher_fix_keeps_original_rollout_source(tmp_path, monkeypatch):
    plan = dict(shared_root=str(tmp_path), experiment_prefix='base',
        queue=[dict(case=dict(replica_id=4, task='pack_objects_into_box', case_id='fixed'))],
        source_sha256='rollout-original', dispatcher_source='/frozen/rollout',
        eval_manifest='/fixture/panel.json', codex='/fixture/codex', sco='/fixture/sco',
        image='same-image', workspace='galbot-foundation-model', cluster='galbot-vla',
        worker_spec='n5lp.nn.a80.2', quota='reserved')
    slot = dict(id=3, index=0, attempt=2, auth_profile='codex_b')
    def fake_run(argv, **kwargs):
        assert kwargs['cwd'] == '/frozen/rollout'
        assert kwargs['env']['PYTHONPATH'] == '/frozen/rollout'
        assert argv[argv.index('--quota-type')+1] == 'reserved'
        path = campaign.submission_path(plan, slot)
        path.parent.mkdir(parents=True)
        write_json(path, dict(source_sha256='rollout-original'))
    monkeypatch.setattr(campaign, 'source_digest', lambda: 'dispatcher-fixed')
    monkeypatch.setattr(campaign.subprocess, 'run', fake_run)
    campaign.ensure_submission(plan, slot)
