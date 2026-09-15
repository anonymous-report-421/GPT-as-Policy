import copy
from datetime import datetime, timezone
import json
import os

import pytest

from . import campaign
from .idle_timeout_policy import detect, adjudicate, verified_adjudication, SCHEMA, REASON
from .io import write_json, sha256


def fixture(tmp_path):
    archive = tmp_path/'archive'
    expected = {'case_id':'language_l1', 'reset_seed':1}
    def put(name, value):
        p = archive/name; p.parent.mkdir(parents=True, exist_ok=True); write_json(p, value)
    put('controller/failure.json', dict(error='record_codex_decision robodojo_server/protocol.py EOFError: RPC connection closed', step_id=257))
    put('controller/history.json', [dict(start_tick=252, end_tick=257, decision=55)])
    put('controller/run.json', dict(context_version='v3', evaluation_case=expected))
    put('sim/evaluation_outcome.json', dict(evaluation_case=expected, complete=False, status='incomplete',
        native_success=None, native_score=None, native_control_steps=257, native_step_limit=1100))
    put('sim/summary.json', dict(complete=False, reason='server_close', step_id=257))
    put('component_reset.json', dict(all_owned_processes_exited=True, ports_released=True))
    put('controller/debug_video/manifest.json', dict(status='completed', frames=258))
    rpc = archive/'source_snapshot/hybrid_rollout/robodojo/robodojo_server/rpc.py'
    rpc.parent.mkdir(parents=True); rpc.write_text('connection.settimeout(900)')
    timestamp = datetime(2026,9,13,4,33,27,tzinfo=timezone.utc).timestamp()
    os.utime(archive/'controller/history.json', (timestamp+2, timestamp+2))
    os.utime(archive/'sim/summary.json', (timestamp+903, timestamp+903))
    log=archive/'logs/sim.log';log.parent.mkdir()
    log.write_text(json.dumps(dict(schema='robodojo_rollout.codex_decision.v1', event='codex_decision',
        recorded_at='2026-09-13T04:33:27+00:00', decision=55, step_id=252))+
        '\n2026-09-13T04:48:31Z Warning\nSimulation App Shutting Down\n')
    plan=dict(shared_root=str(tmp_path),experiment_prefix='test', panel_sha256='panel')
    policy=dict(schema=SCHEMA,campaign='test',panel_sha256='panel',retry=False,idle_seconds=900,
        applies_from_utc='2026-09-13T05:03:00+00:00',explicit_archives=[str(archive)],rpc_source_sha256=sha256(rpc))
    root=tmp_path/'campaigns/test';root.mkdir(parents=True);write_json(root/'rpc_idle_timeout_policy.json',policy)
    batch=dict(exit_code=1,final_reset=dict(all_owned_processes_exited=True,ports_released=True))
    return plan,archive,expected,policy,batch,dict(name='pt-test',state='FAILED')


def test_confirmed_idle_failure_preserves_native_nulls_and_all_original_bytes(tmp_path):
    plan,a,expected,p,b,j=fixture(tmp_path)
    before={x:x.read_bytes() for x in a.rglob('*') if x.is_file()}
    row=adjudicate(plan,b,a,j,expected)
    assert row['reason']==REASON and row['evaluation_success'] is False and row['retry'] is False
    assert row['native_score'] is None and row['native_success'] is None and row['native_complete'] is False
    assert row['observed_idle_seconds']==901
    assert verified_adjudication(a)==row
    assert adjudicate(plan,b,a,j,expected)==row
    assert all(x.read_bytes()==v for x,v in before.items())
    assert not (a/'controller/result.json').exists()


@pytest.mark.parametrize('fault',['live','stopped','manual_exit','cleanup','native_complete','native_score',
    'wrong_identity','event_timeout_only','short_idle','long_idle','wrong_rpc','no_shutdown',
    'sim_crash','missing_frames','unrelated_old_attempt','different_last_decision'])
def test_never_conflate_unrelated_network_or_native_outcome_with_idle(tmp_path,fault):
    plan,a,e,p,b,j=fixture(tmp_path)
    if fault=='live':j['state']='RUNNING'
    if fault=='stopped':j['state']='STOPPED'
    if fault=='manual_exit':b['exit_code']=130
    if fault=='cleanup':b['final_reset']={}
    if fault in ('native_complete','native_score','wrong_identity'):
        f=a/'sim/evaluation_outcome.json';v=json.loads(f.read_text())
        v.update({'native_complete':{'complete':True},'native_score':{'native_score':0},
                  'wrong_identity':{'evaluation_case':{}}}[fault]);write_json(f,v)
    if fault=='event_timeout_only':write_json(a/'controller/failure.json',dict(error='900s Codex event timeout',step_id=257))
    if fault in ('short_idle','long_idle'):
        ts=(a/'controller/history.json').stat().st_mtime+(30 if fault=='short_idle' else 1800)
        os.utime(a/'sim/summary.json',(ts,ts))
    if fault=='wrong_rpc':p['rpc_source_sha256']='bad'
    if fault in ('no_shutdown','sim_crash','different_last_decision'):
        f=a/'logs/sim.log';s=f.read_text()
        if fault=='no_shutdown':s=s.replace('Simulation App Shutting Down','unrelated')
        if fault=='sim_crash':s+='Traceback (most recent call last)'
        if fault=='different_last_decision':s=s.replace('"decision": 55','"decision": 54')
        f.write_text(s)
    if fault=='missing_frames':write_json(a/'controller/debug_video/manifest.json',dict(status='completed',frames=9))
    if fault=='unrelated_old_attempt':
        p['explicit_archives']=[];os.utime(a/'controller/failure.json',(100,100))
    assert detect(a,b,j['state'],e,p) is None


def test_changed_raw_evidence_invalidates_adjudication(tmp_path):
    plan,a,e,p,b,j=fixture(tmp_path);adjudicate(plan,b,a,j,e)
    write_json(a/'sim/summary.json',dict(complete=True))
    with pytest.raises(ValueError,match='evidence changed'):verified_adjudication(a)


def test_comparison_counts_evaluation_failure_without_zero_imputing_score(tmp_path):
    from .test_two_stage import certificate
    from .paired_evaluation import comparison
    _,_,doc=certificate(tmp_path);left=doc['selection'];right=copy.deepcopy(left)
    for row in left['cases']+right['cases']:row['native_score']=1
    right['cases'][0].update(native_success=None,evaluation_success=False,native_score=None,
        evaluation_failure_reason=REASON,native_complete=False)
    right.update(native_successes=7,native_complete_count=49,adjudicated_failure_count=1)
    result=comparison(left,right)
    assert result['gpt_only_success_rate']==7/50
    assert result['gpt_only_mean_score'] is None
    assert result['direct_native_complete_count']==49
    assert result['direct_adjudicated_failure_count']==1


def test_dispatcher_releases_idle_failure_without_retry_and_refills_next_case(tmp_path,monkeypatch):
    from .test_campaign import dispatcher_fixture
    from . import elastic_dispatch, idle_timeout_policy
    plan,state,job,panel,path,batch,archive=dispatcher_fixture(tmp_path,monkeypatch)
    plan['dispatch_mode']='work_conserving'
    other=copy.deepcopy(plan['queue'][0]);other['case']=panel['cases'][7];plan['queue'].append(other)
    monkeypatch.setattr(idle_timeout_policy,'adjudicate',lambda *a:dict(reason=REASON))
    created=[]
    monkeypatch.setattr(campaign,'ensure_submission',lambda p,s:tmp_path/'new/submission.json')
    monkeypatch.setattr(campaign.acp,'submit',lambda p,**kw:created.append(p))
    elastic_dispatch.tick(plan,state,tmp_path,[job])
    assert state['adjudicated_failures']['0']['reason']==REASON
    assert state['skipped_indices']['0']=='failed_rpc_idle_timeout'
    assert not state.get('pending_retries')
    assert state['slots'][0]['index']==1 and state['slots'][0]['status']=='submitted' and len(created)==1
    assert state['slots'][0]['history'][-1]['action']=='failed_rpc_idle_timeout'


def test_display_label_distinguishes_idle_from_native_failure():
    from .robodojo_server.video_panel import terminal_badge
    assert terminal_badge(dict(complete=False,evaluation_failure_reason=REASON))=='Failed · RPC idle 900s'
    assert terminal_badge(dict(complete=True,terminated=True,success=False,evaluation_failure_reason=REASON))=='Task failed · native'
