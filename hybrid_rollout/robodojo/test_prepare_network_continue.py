import hashlib
import json
from pathlib import Path

import pytest

from . import prepare_network_continue as module
from .io import sha256, write_json


def test_freeze_changes_only_network_runtime_files_not_original(tmp_path):
    original=tmp_path/'source_old';p=original/'hybrid_rollout/robodojo/skill/run.py'
    p.parent.mkdir(parents=True);p.write_text('# original runtime\n')
    other=original/'hybrid_rollout/robodojo/io.py';other.write_text('# unchanged IO\n')
    files={str(f.relative_to(original)):sha256(f) for f in [p,other]}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(original/'source_manifest.json',dict(files=files,source_sha256=digest))
    old=dict(context_version='v3',slots=list(range(15)),primary_target=50,
        dispatcher_source=str(original),source_sha256=digest,queue=[{'case_id':'same_seed'}])
    path=tmp_path/'old.json';write_json(path,old);before=path.read_bytes()
    module.prepare(path)
    new=json.loads((tmp_path/'campaign_continue_v3_1.json').read_text())
    assert path.read_bytes()==before and p.read_text()=='# original runtime\n'
    assert new['queue']==old['queue'] and new['slots']==old['slots']
    assert new['network_recovery']['consecutive_wakeups']==20
    assert new['network_recovery']['interval_seconds']==10
    assert (tmp_path/'source_continue_v3_1/hybrid_rollout/robodojo/io.py').read_bytes()==other.read_bytes()
    assert not (tmp_path/'state.json').exists()
    with pytest.raises(ValueError):module.prepare(path)


def test_operator_stop_or_missing_review_cannot_activate(tmp_path):
    p=tmp_path/'plan.json';write_json(p,{'network_recovery':{'revision':'continue_v3_1'}})
    write_json(tmp_path/'STOP',{'reason':'user stopped all rollout'})
    write_json(tmp_path/'authorization_continue_v3_1.json',dict(approved_sha256=sha256(p),user_confirmed=True))
    with pytest.raises(ValueError,match='explicit review'):module.activate(p,sha256(p))
    assert json.loads((tmp_path/'STOP').read_text())['reason']=='user stopped all rollout'


def test_activate_keeps_submitted_attempts_and_replaces_only_unsent_preparation(tmp_path, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(module, '_lock', lambda plan: nullcontext())
    root=tmp_path/'campaign';root.mkdir()
    submissions=tmp_path/'cluster';submissions.mkdir()
    slots=[]
    for i in range(2):
        folder=submissions/str(i);folder.mkdir()
        sub=folder/'submission.json'
        write_json(sub,dict(batch_result=str(folder/'batch_result.json')))
        slots.append(dict(id=i,status='submitted' if i==0 else 'submitting',
            submission=str(sub),index=i,attempt=3,assigned=[i]))
    write_json(submissions/'0/submission_started.json',{'started':True})
    original_bytes=(submissions/'0/submission.json').read_bytes()
    old=dict(approved_sha256='old',slots=slots,deferred_cases={'7':{'case_id':'keep'}})
    write_json(root/'state.json',old)
    write_json(root/'launch_authorization.json',{'approved_sha256':'old'})
    write_json(root/'STOP',{'reason':module.STOP_REASON})
    plan=root/'plan.json';write_json(plan,dict(previous_plan_sha256='old',previous_plan='old.json'))
    approved=sha256(plan)
    write_json(root/'authorization_continue_v3_1.json',dict(approved_sha256=approved,user_confirmed=True))
    module.activate(plan,approved)
    state=json.loads((root/'state.json').read_text())
    assert state['approved_sha256']==approved
    assert state['slots'][0]==slots[0]
    assert (submissions/'0/submission.json').read_bytes()==original_bytes
    assert not (submissions/'1').exists()
    assert state['slots'][1]['status']=='ready' and state['slots'][1]['submission'] is None
    assert state['slots'][1]['index']==1 and state['slots'][1]['attempt']==3
    assert state['deferred_cases']==old['deferred_cases']
    backup=root/'before_continue_v3_1'
    assert json.loads((backup/'state.json').read_text())==old
    assert (backup/'unsubmitted_slot_1/submission.json').exists()
    assert (backup/'STOP_released').exists() and not (root/'STOP').exists()


def test_query_fix_cannot_change_approved_plan_or_worker_source(tmp_path):
    original=tmp_path/'runtime'
    query=original/'hybrid_rollout/robodojo/acp_query.py'
    query.parent.mkdir(parents=True);query.write_text('# old query\n')
    worker=query.with_name('worker.py');worker.write_text('# approved worker\n')
    files={str(p.relative_to(original)):sha256(p) for p in [query,worker]}
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(original/'source_manifest.json',dict(source_sha256=digest,files=files))
    plan=tmp_path/'plan.json';write_json(plan,dict(dispatcher_source=str(original),source_sha256=digest))
    approved=sha256(plan)
    write_json(tmp_path/'authorization_continue_v3_1.json',dict(user_confirmed=True,approved_sha256=approved))
    module.freeze_query_fix(plan,approved)
    assert sha256(plan)==approved
    assert all(sha256(original/name)==value for name,value in files.items())
    ops=json.loads((tmp_path/'operations_continue_query_v1.json').read_text())
    frozen=Path(ops['operations_source'])
    assert ops['rollout_source_sha256']==digest
    assert ops['changed_files']==['hybrid_rollout/robodojo/acp_query.py']
    assert (frozen/'hybrid_rollout/robodojo/worker.py').read_bytes()==worker.read_bytes()
    with pytest.raises(ValueError,match='overwrite'):module.freeze_query_fix(plan,approved)
