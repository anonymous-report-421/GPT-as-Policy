"""Offline tests: explicit v1/v2 comparisons and work-conserving scheduling."""
import copy
import json
from pathlib import Path

import pytest

from . import campaign, elastic_dispatch
from .case_ledger import build_ledger, make_rerun_reference, require_available
from .io import write_json
from .test_case_ledger import attempt
from .test_campaign import dispatcher_fixture
from .test_evaluation import fake_panel_file


def test_linked_v2_failure_is_complete_not_permission_for_third_run(tmp_path, monkeypatch):
    monkeypatch.setattr('hybrid_rollout.robodojo.prompt_context.CONTEXT_VERSION', 'v2')
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    batch = attempt(tmp_path, panel, case, success=False)
    old = tmp_path/'results/exp/archive'
    ref = make_rerun_reference(tmp_path, panel, case['case_id'], old)
    old_bytes = (old/'controller/result.json').read_bytes()
    require_available(tmp_path, panel, case['case_id'], rerun_of=ref)
    with pytest.raises(ValueError):
        require_available(tmp_path, panel, case['case_id'])
    with pytest.raises(ValueError):
        require_available(tmp_path, panel, case['case_id'], rerun_of=dict(ref, result_sha256='changed'))
    new = tmp_path/'results/new/archive'
    new.mkdir(parents=True)
    import shutil
    shutil.copytree(old/'controller', new/'controller')
    shutil.copytree(old/'sim', new/'sim')
    data = json.loads(batch.read_text())
    data.update(context_version='v2', rerun_of=ref, attempt='1')
    data['episodes'][0]['archive'] = str(new)
    (tmp_path/'results/new/_replicas/replica_1/attempt_1').mkdir(parents=True)
    write_json(tmp_path/'results/new/_replicas/replica_1/attempt_1/batch.json', data)
    row = build_ledger(tmp_path, panel)['cases'][6]
    assert row['state'] == 'completed' and row['paired_context_rerun'] is True
    assert row['completed_by_context'] == {'v1': 1, 'v2': 1}
    with pytest.raises(ValueError):
        require_available(tmp_path, panel, case['case_id'], rerun_of=ref)
    assert (old/'controller/result.json').read_bytes() == old_bytes
    data.pop('rerun_of')
    write_json(tmp_path/'results/new/_replicas/replica_1/attempt_1/batch.json', data)
    assert build_ledger(tmp_path, panel)['cases'][6]['state'] == 'duplicate_completed_review_required'


@pytest.mark.parametrize('success,state', [(True, 'completed'), (False, 'starting')])
def test_success_or_incomplete_cannot_authorize_rerun(tmp_path, success, state):
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    attempt(tmp_path, panel, case, success=success, state=state)
    with pytest.raises(ValueError, match='native v1 failure'):
        make_rerun_reference(tmp_path, panel, case['case_id'], tmp_path/'results/exp/archive')


def test_own_pending_submission_only_is_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr('hybrid_rollout.robodojo.prompt_context.CONTEXT_VERSION', 'v2')
    _, _, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    attempt(tmp_path, panel, case, success=False)
    ref = make_rerun_reference(tmp_path, panel, case['case_id'], tmp_path/'results/exp/archive')
    root = tmp_path/'cluster/new/replica_1'
    root.mkdir(parents=True)
    own = tmp_path/'results/new/_replicas/replica_1/attempt_1/batch.json'
    write_json(root/'submission.json', dict(evaluation_panel_sha256=panel['panel_sha256'],
        evaluation_cases=[case], batch_result=str(own), context_version='v2', rerun_of=ref,
        environment={'ROLLOUT_ATTEMPT': '1'}))
    write_json(root/'submission_started.json', {})
    with pytest.raises(ValueError):
        require_available(tmp_path, panel, case['case_id'], rerun_of=ref)
    require_available(tmp_path, panel, case['case_id'], own, rerun_of=ref)


def test_first_free_slot_and_backoff_do_not_pin_work(tmp_path, monkeypatch):
    plan, state, job, panel, path, batch, archive = dispatcher_fixture(tmp_path, monkeypatch)
    plan['dispatch_mode'] = 'work_conserving'
    other = copy.deepcopy(plan['queue'][0])
    other['case'] = panel['cases'][7]
    plan['queue'].append(other)
    # Network-failed case enters durable backoff. Same slot takes another case immediately.
    created = []
    def prepare(p, s):
        q = tmp_path/f"fake-{s['index']}.json"
        write_json(q, {})
        return q
    monkeypatch.setattr(campaign, 'ensure_submission', prepare)
    monkeypatch.setattr(campaign.acp, 'submit', lambda p, **kw: created.append(p))
    elastic_dispatch.tick(plan, state, tmp_path, [job])
    assert state['slots'][0]['index'] == 1 and state['slots'][0]['status'] == 'submitted'
    assert state['pending_retries'][0]['index'] == 0 and len(created) == 1
    free = dict(id=7, auth_profile='codex_a_2', status='idle', assigned=[])
    state['slots'].append(free)
    state['pending_retries'][0]['next_retry_at'] = 0
    elastic_dispatch.reserve(state, free, plan)
    assert free['index'] == 0 and free['attempt'] == 1
    assert not state['pending_retries']  # Another profile may safely pick it up.


def test_native_completion_refills_in_same_poll(tmp_path, monkeypatch):
    plan, state, job, *_ = dispatcher_fixture(tmp_path, monkeypatch)
    plan['dispatch_mode'] = 'work_conserving'
    new = copy.deepcopy(plan['queue'][0])
    new['case']['case_id'] = 'different'
    plan['queue'].append(new)
    monkeypatch.setattr(campaign, 'assess', lambda *a: 'completed')
    created = []
    monkeypatch.setattr(campaign, 'ensure_submission', lambda p, s: tmp_path/'new/submission.json')
    monkeypatch.setattr(campaign.acp, 'submit', lambda p, **kw: created.append(p))
    elastic_dispatch.tick(plan, state, tmp_path, [job])
    assert len(created) == 1 and state['slots'][0]['index'] == 1
    assert state['slots'][0]['status'] == 'submitted'


def test_conditional_rerun_does_not_block_and_same_case_never_overlaps(tmp_path):
    _, manifest, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    plan = dict(shared_root=str(tmp_path), eval_manifest=str(manifest), dispatch_mode='work_conserving', queue=[
        dict(case=case, initial_attempt=0, rerun_from_context='v1', priority=0),
        dict(case=panel['cases'][7], initial_attempt=0)])
    free = dict(id=2, assigned=[], status='idle')
    state = dict(slots=[free])
    elastic_dispatch.reserve(state, free, plan)
    assert free['index'] == 1
    attempt(tmp_path, panel, case, success=False)
    free['status'] = 'idle'
    elastic_dispatch.reserve(state, free, plan)
    assert free['index'] == 0 and free['attempt'] == 1
    assert free['rerun_v1_archive'] == str(tmp_path/'results/exp/archive')
    # The running original case prevents an overlapping v2 submission.
    state = dict(slots=[dict(id=3, assigned=[], status='running', index=0),
                       dict(id=4, assigned=[], status='idle')])
    elastic_dispatch.reserve(state, state['slots'][1], plan)
    assert state['slots'][1]['index'] == 1


def test_success_skips_conditional_rerun(tmp_path):
    _, manifest, panel = fake_panel_file(tmp_path)
    case = panel['cases'][6]
    attempt(tmp_path, panel, case, success=True)
    plan = dict(shared_root=str(tmp_path), eval_manifest=str(manifest), queue=[
        dict(case=case, initial_attempt=1, rerun_from_context='v1')])
    slot = dict(id=2, status='idle', assigned=[])
    state = dict(slots=[slot])
    elastic_dispatch.reserve(state, slot, plan)
    assert slot['status'] == 'drained'
    assert state['skipped_indices'] == {'0': 'v1_native_success_no_rerun'}


def test_expansion_preserves_active_seeds_and_adds_only_reviewed_cases(tmp_path):
    from .expand_campaign import revise
    _, _, panel = fake_panel_file(tmp_path)
    scope = json.loads(Path(__file__).with_name('eval_panels').joinpath('robodojo_panel50_scope_v2.json').read_text())
    selected = [cid for t in scope['tasks'][:6] for cid in t['case_ids']]
    by_id = {c['case_id']: c for c in panel['cases']}
    old = dict(queue=[dict(case=by_id[c], initial_attempt=0, slot_id=i%6) for i,c in enumerate(selected)],
        skipped_completed=[],experiment_prefix='base',source_sha256='v8',dispatcher_source='/frozen/v8',
        context_case_versions={c:('v1' if i<3 else 'v2') for i,c in enumerate(selected)},
        case_runtime_overrides={'0':dict(source_sha256='v1',dispatcher_source='/frozen/v1',context_version='v1')})
    state = dict(slots=[dict(id=0,auth_profile='codex_b_4',index=0,assigned=[0],status='running',
        attempt=2,submission='old',job_id='pt-old')])
    ledger = dict(errors=[],cases=[dict(case_id=c['case_id'],attempts=[]) for c in panel['cases']])
    ledger['cases'][0]['attempts']=[dict(state='completed',context_version='v1',native_success=False,attempt='2',archive='/old/fail')]
    ledger['cases'][1]['attempts']=[dict(state='completed',context_version='v1',native_success=True,attempt='0',archive='/old/success')]
    before=copy.deepcopy(state)
    new,after=revise(old,state,panel,scope,ledger)
    assert after==before and state==before
    assert len(new['added_case_ids'])==20 and new['primary_target']==50
    assert new['confirmed_v1_failure_reruns']==[selected[0]]
    assert new['conditional_v1_failure_reruns']==[selected[2]]
    assert new['minimum_complete_trajectories']==51 and new['maximum_complete_trajectories']==52
    assert len(new['queue'])==52
    assert all('slot_id' not in e for e in new['queue'])
    assert [e['case'] for e in new['queue'][:30]]==[e['case'] for e in old['queue']]
    assert new['case_runtime_overrides']['0']['context_version']=='v1'
    assert new['case_runtime_overrides']['1']['source_sha256']=='v8'
    with pytest.raises(ValueError,match='already planned'):
        revise(new,after,panel,scope,ledger)


@pytest.mark.parametrize('tamper', [False, True])
def test_expansion_activation_never_changes_active_assignments(tmp_path, tamper):
    import hashlib
    from .expand_campaign import activate, REVISION
    from .io import sha256
    (tmp_path/'evaluation').mkdir()
    state = dict(approved_sha256='old',slots=[dict(id=i,auth_profile=n,status='running',index=i)
                 for i,n in campaign.B5_A2_SLOTS.items()])
    write_json(tmp_path/'state.json',state)
    write_json(tmp_path/'STOP',{'reason':'migration'})
    write_json(tmp_path/'STOP_SLOT_1',{'reason':'retired gateway'})
    plan=dict(shared_root=str(tmp_path), expansion_revision=REVISION,parent_plan_sha256='old',
        slots=[dict(id=i,auth_profile=n) for i,n in campaign.B5_A2_SLOTS.items()],
        expansion_parent_state_sha256=sha256(tmp_path/'state.json'),
        expansion_stop_sha256=sha256(tmp_path/'STOP'),
        expansion_slot_stops={'STOP_SLOT_1':sha256(tmp_path/'STOP_SLOT_1')},
        expansion_state_payload_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest())
    path=tmp_path/'plan.json';write_json(path,plan)
    prepared=dict(state,approved_sha256=sha256(path))
    if tamper:prepared['slots']=[]
    write_json(tmp_path/f'state_{REVISION}_prepared.json',prepared)
    if tamper:
        with pytest.raises(ValueError,match='Prepared state changed'):activate(path,sha256(path))
        assert json.loads((tmp_path/'state.json').read_text())==state
    else:
        activate(path,sha256(path))
        assert json.loads((tmp_path/'state.json').read_text())['slots']==state['slots']
        assert json.loads((tmp_path/f'state_before_{REVISION}.json').read_text())==state
        assert not (tmp_path/'STOP').exists()
    assert (tmp_path/'STOP_SLOT_1').exists()
