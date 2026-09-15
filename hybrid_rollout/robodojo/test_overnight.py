import copy
from pathlib import Path

import pytest

from . import campaign, overnight
from .io import write_json
from .quota_guard import maybe_reset
from .test_quota_guard import RPC, limits
from .test_input_guard_queue import fixture
from .unattended_recovery import release_deferred


def test_one_new_b_reset_is_durable_idempotent_and_shared_between_phases():
    rpc = RPC([TimeoutError(), 'alreadyRedeemed'])
    journal = {}
    saved = []
    with pytest.raises(TimeoutError):
        maybe_reset(rpc, limits(), journal, lambda: saved.append(copy.deepcopy(journal)), max_resets=1)
    # Simulate restart/second phase reading the persisted intent, not allocating
    # a new reset when the response is lost.
    recovered = copy.deepcopy(saved[-1])
    maybe_reset(rpc, limits(80), recovered, lambda: None, max_resets=1)
    maybe_reset(rpc, limits(), recovered, lambda: None, max_resets=1)
    assert len(rpc.keys)==2 and rpc.keys[0]==rpc.keys[1]
    assert len(recovered['b_reset_attempts'])==1


def deferred_fixture(tmp_path, monkeypatch):
    plan, state, job, sub, batch, archive = fixture(tmp_path, monkeypatch)
    plan.update(shared_root=str(tmp_path), experiment_prefix='robodojo_test')
    state['deferred_cases']={'0':dict(original_slot=copy.deepcopy(state['slots'][0]))}
    state['slots'][0].update(status='drained',submission=None)
    return plan,state,job,sub,batch,archive


def test_deferred_recovery_requeues_once_preserves_seed_and_all_old_bytes(tmp_path,monkeypatch):
    plan,state,job,sub,batch,archive=deferred_fixture(tmp_path,monkeypatch)
    original={p:p.read_bytes() for p in (sub,batch,archive/'controller/failure.json')}
    queue=copy.deepcopy(plan['queue'])
    release_deferred(plan,state,tmp_path,[job])
    assert not state['deferred_cases']
    assert state['pending_retries'][0]['index']==0 and state['pending_retries'][0]['attempt']==1
    release_deferred(plan,state,tmp_path,[job])
    assert len(state['pending_retries'])==1 and plan['queue']==queue
    assert all(p.read_bytes()==b for p,b in original.items())


@pytest.mark.parametrize('condition',['native','live','reset_missing','stop','later_intent','duplicate_active'])
def test_no_speculative_recovery(tmp_path,monkeypatch,condition):
    plan,state,job,sub,batch,archive=deferred_fixture(tmp_path,monkeypatch)
    if condition=='native':
        write_json(archive/'controller/result.json',dict(complete=True))
    if condition=='live':job['state']='RUNNING'
    if condition=='reset_missing':write_json(batch,dict(campaign.optional(batch),final_reset={}))
    if condition=='stop':write_json(tmp_path/'STOP',{})
    if condition=='later_intent':
        path=tmp_path/'cluster/robodojo_test_c0_a1/replica_0/submission.json'
        path.parent.mkdir(parents=True)
        write_json(path,{})
    if condition=='duplicate_active':state['slots'][0]['status']='running'
    release_deferred(plan,state,tmp_path,[job])
    assert not state.get('pending_retries') and '0' in state['deferred_cases']


def test_direct_plan_preserves_exact_case_bytes_without_runtime_overrides():
    parent=dict(schema='test',context_version='v3',evaluation_method='pi05_plus_gpt',
        primary_case_ids=[str(i) for i in range(50)],
        queue=[dict(case=dict(case_id=str(i),seed=i),initial_attempt=3) for i in range(50)],
        case_runtime_overrides={'1':{'evaluation_method':'pi05_plus_gpt'}})
    keys=('shared_root eval_manifest panel_sha256 scope_file scope_file_sha256 poll_seconds retry_limit '
          'retry_only model effort fast max_decisions max_seconds gpus_per_replica codex sco image '
          'workspace cluster worker_spec quota https_proxy_file preparation_revision quota_protection '
          'account_groups primary_target network_recovery').split()
    parent.update({k:'test' for k in keys})
    parent.update(slots=[dict(id=k,auth_profile=v) for k,v in campaign.POOL15_SLOTS.items()],
                  max_concurrent_gpus=30,dispatch_mode='work_conserving')
    p=overnight.direct_plan(parent,'robodojo_direct',Path('/example'),Path('/source'),'hash')
    assert [r['case'] for r in p['queue']]==[r['case'] for r in parent['queue']]
    assert all(r['initial_attempt']==0 for r in p['queue'])
    assert p['action_space']=='eef_only' and p['context_version']=='v3'
    assert 'case_runtime_overrides' not in p


def test_v3_gate_rejects_old_context_and_accepts_native_failure(tmp_path):
    from .test_two_stage import certificate
    from .phase_gate import verify_certificate
    path,panel,doc=certificate(tmp_path)
    doc['schema']='robodojo.hybrid50_completion.v3'
    for row in doc['selection']['cases']:
        row.update(context_version='v3',reused_v1_success=False)
        write_json(Path(row['archive'])/'controller/run.json',dict(context_version='v3'))
    write_json(path,doc)
    assert verify_certificate(path,panel)['selection']['complete']
    doc['selection']['cases'][0]['context_version']='v2'
    write_json(path,doc)
    with pytest.raises(ValueError,match='v3'):verify_certificate(path,panel)


def test_phase_switch_waits_for_fifty_and_never_blocks_on_own_direct_jobs(tmp_path,monkeypatch):
    from .io import sha256
    first=tmp_path/'first/campaign.json';second=tmp_path/'second/campaign.json'
    first.parent.mkdir();second.parent.mkdir()
    quota=tmp_path/'quota.json';write_json(quota,{})
    plan=dict(experiment_prefix='robodojo_hybrid',panel_sha256='p',eval_manifest='m')
    write_json(first,plan)
    write_json(second,dict(plan,experiment_prefix='robodojo_direct',phase1_certificate=str(tmp_path/'cert.json')))
    wf=dict(phase1_plan=str(first),phase1_sha256=sha256(first),phase2_plan=str(second),
        phase2_sha256=sha256(second),quota_policy=str(quota),quota_sha256=sha256(quota))
    complete=False
    monkeypatch.setattr(overnight,'selection',lambda p,j:dict(
        complete=complete and p['experiment_prefix']=='robodojo_hybrid',errors=[]))
    jobs=[]
    monkeypatch.setattr(campaign,'job_list',lambda p:jobs)
    monkeypatch.setattr(overnight,'read_panel',lambda *a:dict(panel_sha256='p'))
    monkeypatch.setattr(overnight,'verify_certificate',lambda *a:None)
    assert overnight.cycle(wf,'wf',second.parent)['phase']==1
    assert not (tmp_path/'cert.json').exists()
    complete=True
    jobs.append(dict(name='pt-x',display_name='robodojo_hybrid_c1_a0_r1',state='RUNNING',
                     ownership=dict(user_name='sujiayi')))
    assert 'phase' not in overnight.cycle(wf,'wf',second.parent)
    write_json(first.parent/'quota_guard_state.json',dict(a_held=True))
    jobs[0]['display_name']='robodojo_direct_c1_a0_r1'
    assert overnight.cycle(wf,'wf',second.parent)['phase']==2
    assert campaign.optional(second.parent/'quota_guard_state.json')['a_held'] is True


def test_external_pool_policy_only_enables_b_and_cannot_cross_authorizations(tmp_path,monkeypatch):
    from .test_prepare_v3 import pool_plan, mock_accounts
    from .pool15_quota import Pool15QuotaGuard
    from .io import sha256
    p=pool_plan(tmp_path);p.update(shared_root=str(tmp_path),experiment_prefix='robodojo_test')
    policy=tmp_path/'quota.json'
    write_json(policy,dict(schema='robodojo.unattended_quota.v1',panel_sha256='test',
        campaigns=['robodojo_test'],b_max_resets=1,a_hold_remaining=30,a_stop_remaining=20,
        a_resume_remaining=50,reset_budget=str(tmp_path/'evaluation/budget.json')))
    obj=Pool15QuotaGuard(p,policy,sha256(policy))
    calls=[]
    monkeypatch.setattr(obj,'reset_b',lambda rpc,value,profile:(calls.append(profile) or value,[]))
    mock_accounts(obj,monkeypatch,dict(a=50,b=0,c=0))
    obj.tick(dict(slots=[]),tmp_path,[])
    assert len(calls)==1 and calls[0].startswith('codex_b')
    p['experiment_prefix']='unapproved'
    with pytest.raises(ValueError):Pool15QuotaGuard(p,policy,sha256(policy))


def operations_fixture(tmp_path, monkeypatch, fairness=False):
    from .io import sha256
    old, new = tmp_path/'worker', tmp_path/'ops'
    names = ('hybrid_rollout/robodojo/input_guard_queue.py',
             'hybrid_rollout/robodojo/overnight.py', 'hybrid_rollout/robodojo/skill/run.py',
             'hybrid_rollout/robodojo/campaign.py')
    for root in (old, new):
        for i, name in enumerate(names):
            p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('changed' if root==new and (i<2 or fairness and i==3) else 'original')
    def files(root):return {name:sha256(root/name) for name in names}
    write_json(old/'source_manifest.json', dict(source_sha256='workerhash',files=files(old)))
    write_json(new/'source_manifest.json', dict(source_sha256='opshash',
        parent_source_sha256='workerhash', files=files(new)))
    monkeypatch.setattr(overnight, '__file__', str(new/names[1]))
    monkeypatch.setattr(campaign, 'source_digest', lambda:'opshash')
    return dict(source=str(old),source_sha256='workerhash',phase2_plan='unchanged-plan',
        quota_policy='unchanged-quota',phase1_sha256='unchanged-certificate-owner'), old, new


@pytest.mark.parametrize('fairness', [False, True])
def test_operations_override_changes_only_dispatcher_source_not_workflow_identity(tmp_path, monkeypatch, fairness):
    wf, old, new = operations_fixture(tmp_path, monkeypatch, fairness)
    before=copy.deepcopy(wf)
    actual=overnight.operations_workflow(wf,'opshash')
    assert wf==before
    assert actual==dict(wf,source=str(new),source_sha256='opshash')
    assert overnight.operations_workflow(wf) is wf


@pytest.mark.parametrize('fault',['wrong_approval','wrong_parent','worker_change','mutated_parent'])
def test_operations_override_refuses_unapproved_or_worker_changes(tmp_path, monkeypatch, fault):
    wf, old, new = operations_fixture(tmp_path, monkeypatch)
    m=campaign.optional(new/'source_manifest.json')
    if fault=='wrong_parent':m['parent_source_sha256']='other'
    if fault=='worker_change':m['files']['hybrid_rollout/robodojo/skill/run.py']='other'
    if fault=='mutated_parent':(old/'hybrid_rollout/robodojo/skill/run.py').write_text('changed-after-freeze')
    write_json(new/'source_manifest.json',m)
    with pytest.raises(ValueError,match='operations-only'):
        overnight.operations_workflow(wf,'wrong' if fault=='wrong_approval' else 'opshash')


def test_supervisor_keeps_cycle_identity_but_launches_approved_operations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from .io import sha256
    first=tmp_path/'first.json';write_json(first,dict(shared_root=str(tmp_path)))
    second=tmp_path/'second.json';write_json(second,{})
    wf=dict(schema='robodojo.overnight_v3.v1',source='/worker',source_sha256='workerhash',
        phase1_plan=str(first),phase2_plan=str(second),poll_seconds=1)
    path=tmp_path/'workflow.json';write_json(path,wf);approved=sha256(path)
    ops=dict(wf,source='/ops',source_sha256='opshash');seen=[]
    monkeypatch.setattr(overnight,'operations_workflow',lambda original,sha:ops if sha=='opshash' else None)
    monkeypatch.setattr(campaign,'source_digest',lambda:'opshash')
    monkeypatch.setattr(overnight,'cycle',lambda original,digest,root:
        seen.append(('cycle',original,digest)) or dict(phase=2,state='complete'))
    monkeypatch.setattr(overnight,'dispatcher_unlocked',lambda shared:True)
    monkeypatch.setattr(overnight,'launch_dispatcher',lambda actual,phase,root:
        seen.append(('launch',actual,phase)) or SimpleNamespace(pid=123))
    overnight.run(path,approved,'opshash')
    assert seen==[('cycle',wf,approved),('launch',ops,2)]
    assert sha256(path)==approved
    assert campaign.optional(tmp_path/'progress.json')['operations_source_sha256']=='opshash'
