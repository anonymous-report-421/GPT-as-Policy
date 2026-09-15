import copy
from contextlib import nullcontext
from pathlib import Path

import pytest

from . import campaign, refill_campaign as module
from .evaluation import case_identity
from .io import sha256, write_json
from .test_evaluation import fake_panel_file


def state_fixture():
    return dict(slots=[dict(id=0,status='running',index=1,assigned=[1],job_id='live'),
        dict(id=10,status='pause_unknown_failure',index=20,assigned=[10,20],attempt=1,
             job_id='pt-z85m9n1v',submission='old/submission.json',retry_number=1)],
        deferred_cases={'3':{'preserve':'old evidence'}},pending_retries=[])


def test_exact_retry_preserves_active_slots_and_uses_shared_queue():
    old=state_fixture();before=copy.deepcopy(old)
    new=module.retry_state(old,module.TARGET,set())
    assert old==before and new['slots'][0]==old['slots'][0]
    assert new['deferred_cases']==old['deferred_cases']
    slot=new['slots'][1]
    assert slot['status']=='idle' and slot['submission'] is None
    assert slot['assigned']==[10,20]
    assert slot['history'][-1]['submission']=='old/submission.json'
    assert new['pending_retries'][0]['index']==20 and new['pending_retries'][0]['attempt']==2
    assert new['pending_retries'][0]['next_retry_at']==0


@pytest.mark.parametrize('field,value',[('status','running'),('job_id','different'),('attempt',2),('index',3)])
def test_cannot_release_a_different_or_running_attempt(field,value):
    state=state_fixture();state['slots'][1][field]=value
    with pytest.raises(ValueError):module.retry_state(state,module.TARGET,set())


def test_duplicate_retry_is_rejected():
    state=state_fixture();state['pending_retries'].append(dict(index=20,attempt=2))
    with pytest.raises(ValueError):module.retry_state(state,module.TARGET,set())


def test_poll_override_keeps_plan_unchanged():
    plan={'poll_seconds':300};before=dict(plan)
    assert campaign.polling_interval(plan)==300
    assert campaign.polling_interval(plan,30)==30 and plan==before
    for value in [0,1,29,301]:
        with pytest.raises(ValueError):campaign.polling_interval(plan,value)


@pytest.mark.parametrize('native_complete', [False, True])
def test_activate_verifies_terminal_evidence_and_never_retries_native_complete(tmp_path,monkeypatch,native_complete):
    _,manifest,panel=fake_panel_file(tmp_path)
    case=panel['cases'][6];identity=case_identity(panel,case)
    target=dict(slot=10,index=0,attempt=1,job_id='pt-target',case_id=case['case_id'])
    monkeypatch.setattr(module,'TARGET',target)
    monkeypatch.setattr(module,'_lock',lambda plan:nullcontext())
    job=dict(name='pt-target',state='FAILED',display_name='experiment_r1',ownership={'user_name':'sujiayi'})
    monkeypatch.setattr(campaign,'job_list',lambda plan:[job])
    archive=tmp_path/'archive';archive.mkdir()
    for name in ['sim','controller']:(archive/name).mkdir()
    write_json(archive/'sim/evaluation_outcome.json',dict(evaluation_case=identity,complete=native_complete))
    write_json(archive/'controller/result.json',dict(evaluation_case=identity,complete=native_complete))
    write_json(archive/'controller/failure.json',{'error':'Five rejected tool calls'})
    write_json(archive/'artifact_manifest.json',{'status':'verified'})
    batch=tmp_path/'batch.json';write_json(batch,dict(exit_code=1,
        episodes=[dict(case_id=case['case_id'],archive=str(archive))],
        final_reset=dict(all_owned_processes_exited=True,ports_released=True)))
    sub=tmp_path/'submission.json';write_json(sub,dict(batch_result=str(batch),
        environment={'ROLLOUT_EXPERIMENT_ID':'experiment','ROLLOUT_REPLICA_ID':'1'}))
    write_json(sub.with_name('submission_started.json'),{'started':True})
    root=tmp_path/'campaign';root.mkdir();plan=root/'plan.json'
    write_json(plan,dict(eval_manifest=str(manifest),queue=[{'case':case}]))
    approved=sha256(plan)
    state=dict(approved_sha256=approved,slots=[dict(id=10,index=0,attempt=1,assigned=[0],
        status='pause_unknown_failure',job_id='pt-target',submission=str(sub))])
    write_json(root/'state.json',state)
    write_json(root/'operations_fill15_v1.json',dict(user_confirmed=True,approved_plan_sha256=approved,operations_source_sha256='frozen'))
    write_json(root/'STOP',{'reason':module.STOP_REASON})
    before={str(f):f.read_bytes() for f in archive.rglob('*') if f.is_file()}
    if native_complete:
        with pytest.raises(ValueError,match='terminal, incomplete'):module.activate(plan,approved)
        assert campaign.optional(root/'state.json')==state
    else:
        module.activate(plan,approved)
        assert campaign.optional(root/'state.json')['pending_retries'][0]['attempt']==2
        assert campaign.optional(root/'before_fill15_v1/state.json')==state
        assert not (root/'STOP').exists()
    assert sha256(plan)==approved and all(Path(name).read_bytes()==data for name,data in before.items())
