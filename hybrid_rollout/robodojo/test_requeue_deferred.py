import copy

import pytest

from . import campaign
from . import requeue_deferred as module
from .io import write_json
from .test_input_guard_queue import fixture


def state_fixture():
    targets = {3: (2, 'old3'), 10: (0, 'old10')}
    state = dict(slots=[dict(id=0, status='running', index=49, attempt=1, job_id='live', assigned=[49]),
        dict(id=1, status='drained', index=3, assigned=[3,10])], pending_retries=[],
        input_guard_retry_counts={'3':1,'46':1}, deferred_cases={
            str(i):dict(original_slot=dict(index=i, attempt=a, job_id=j, status='pause_unknown_failure'),
                preserve='raw evidence') for i,(a,j) in targets.items()})
    state['deferred_cases']['46'] = {'unrelated': 'new interruption stays deferred'}
    return state, targets


def test_exact_replay_keeps_active_slots_and_old_evidence_without_resetting_limits():
    state,targets = state_fixture(); old = copy.deepcopy(state)
    new = module.transition(state, targets)
    assert state == old and new['slots'][0] == old['slots'][0]
    assert new['slots'][1]['status'] == 'idle'
    assert new['slots'][1]['assigned'] == [3,10]
    assert new['deferred_cases'] == {'46':old['deferred_cases']['46']}
    assert [(r['index'],r['attempt']) for r in new['pending_retries']] == [(3,3),(10,1)]
    assert all(r['next_retry_at']==0 for r in new['pending_retries'])
    assert new['input_guard_retry_counts']=={'3':1,'10':1,'46':1}
    assert new['manual_deferred_requeues'][0]['old_deferred']['3']==old['deferred_cases']['3']
    with pytest.raises(ValueError,match='already applied'): module.transition(new,targets)


@pytest.mark.parametrize('bad',['missing','different_job','different_attempt','already_queued','active','unreserved'])
def test_transition_rejects_changed_or_duplicate_targets_without_partial_mutation(bad):
    state,targets = state_fixture()
    if bad=='missing':state['deferred_cases'].pop('10')
    elif bad=='different_job':state['deferred_cases']['10']['original_slot']['job_id']='new'
    elif bad=='different_attempt':state['deferred_cases']['10']['original_slot']['attempt']=1
    elif bad=='already_queued':state['pending_retries'].append({'index':10})
    elif bad=='active':state['slots'].append(dict(id=3,status='running',index=10,assigned=[]))
    else:state['slots'][1]['assigned']=[3]
    before=copy.deepcopy(state)
    with pytest.raises(ValueError):module.transition(state,targets)
    assert state==before


def test_only_known_maintenance_stopped_slots_are_restored():
    state,targets=state_fixture();before=copy.deepcopy(state)
    state['slots'][0]['status']='stopped'
    new=module.transition(state,targets,before)
    assert new['slots'][0]==before['slots'][0]
    with pytest.raises(ValueError,match='unrelated'):module.transition(state,targets)
    state['slots'][0]['job_id']='changed'
    with pytest.raises(ValueError,match='unrelated'):module.transition(state,targets,before)


@pytest.mark.parametrize('bad',[None,'live','stopped','complete','wrong_seed','unclean','unverified','other_owner','unknown_error','later_attempt'])
def test_terminal_evidence_is_required_and_never_rewritten(tmp_path,monkeypatch,bad):
    plan,state,job,sub,batch,archive=fixture(tmp_path,monkeypatch)
    slot=state['slots'][0];slot['attempt']=0
    state['deferred_cases']={'0':{'original_slot':copy.deepcopy(slot)}}
    state['slots'][0]['status']='drained'
    if bad=='live':job['state']='RUNNING'
    elif bad=='stopped':job['state']='SUSPENDED'
    elif bad=='complete':write_json(archive/'controller/result.json',dict(campaign.optional(archive/'controller/result.json'),complete=True))
    elif bad=='wrong_seed':write_json(archive/'sim/evaluation_outcome.json',dict(complete=False,evaluation_case={'wrong':'seed'}))
    elif bad=='unclean':write_json(batch,dict(campaign.optional(batch),final_reset={}))
    elif bad=='unverified':write_json(archive/'artifact_manifest.json',{'status':'failed'})
    elif bad=='other_owner':job['ownership']['user_name']='other'
    elif bad=='unknown_error':write_json(archive/'controller/failure.json',{'error':'HTTP 502'})
    elif bad=='later_attempt':(tmp_path/'cluster'/f"{plan['experiment_prefix']}_c0_a1").mkdir()
    before={p:p.read_bytes() for p in archive.rglob('*') if p.is_file()}
    if bad is None:
        rows=module.verify_targets(plan,state,[job],{0:(0,job['name'])})
        assert rows[0]['next_attempt']==1
    else:
        with pytest.raises(ValueError):module.verify_targets(plan,state,[job],{0:(0,job['name'])})
    assert all(p.read_bytes()==value for p,value in before.items())
