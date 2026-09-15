"""Scoped maintenance for the user-approved simulator idle-timeout rule."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess

from . import campaign
from .io import write_json, sha256
from .prepare_input_recovery import verified_source

ROOT = Path('/mnt/rollout/robodojo_mixed_control/campaigns/robodojo_v3_eef50_20260912_01')
PLAN_SHA = 'f83775883343bef355cf66f60c5c3856590d85789b2ad21208d45cd28fa1bb41'
REASON = 'user_approved_rpc_idle_timeout_no_retry_handoff'
SOURCE_SHA = '909b3ae5629d0790da215d92cb86c68fc3971ddaa1536fb940a188d63c160c0c'
WORKFLOW_SHA = 'e1a9a015bcc376603b4e3887fe112abdcf56e379352cad6dd0adc4b28aeb4ca5'
PATCHES = ('campaign.py', 'overnight.py', 'paired_evaluation.py', 'two_stage.py', 'idle_timeout_policy.py')


def prepare(root=ROOT):
    from .idle_timeout_policy import SCHEMA
    worker = root/'source_input_recovery_v1'
    old = verified_source(worker, SOURCE_SHA)
    ops = root/'operations_rpc_idle_timeout_v1'
    policy_path = root/'rpc_idle_timeout_policy.json'
    if ops.exists() or policy_path.exists():
        raise ValueError('Never overwrite an existing operations snapshot/policy')
    if campaign.optional(root/'STOP').get('reason') != REASON:
        raise ValueError('Require this authorized maintenance STOP')
    archive = root.parents[1]/'results/robodojo_v3_eef50_20260912_01_c11_a7/classify_objects_by_language/standard/eval_seed_0/layout_1/replica_1/attempt_7'
    policy = dict(schema=SCHEMA, campaign=root.name, panel_sha256=campaign.optional(root/'campaign_input_recovery_v1.json')['panel_sha256'],
        retry=False, idle_seconds=900, native_score_imputation=None,
        applies_from_utc=campaign.optional(root/'STOP')['utc'], explicit_archives=[str(archive)],
        rpc_source_sha256=sha256(archive/'source_snapshot/hybrid_rollout/robodojo/robodojo_server/rpc.py'),
        authorization=str(root/'handoff/rpc_idle_timeout_v1/authorization.json'),
        qualification='Reviewed fixed RPC source + EOF at record_codex_decision + 895..915s ACK-to-close and 900..940s decision-to-shutdown, closed/reset/frames verified; native completion takes precedence.')
    write_json(policy_path, policy)
    shutil.copytree(worker/'hybrid_rollout', ops/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    repo = Path(__file__).resolve().parent
    for name in PATCHES:
        shutil.copy2(repo/name, ops/'hybrid_rollout/robodojo'/name)
    files={str(p.relative_to(ops)):sha256(p) for p in (ops/'hybrid_rollout').rglob('*') if p.is_file()}
    changed={p for p in files if files[p] != old['files'].get(p)}
    expected={'hybrid_rollout/robodojo/'+p for p in PATCHES}
    if changed != expected or set(old['files'])-set(files):
        raise ValueError('Unexpected operations diff')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(ops/'source_manifest.json',dict(files=files,source_sha256=digest,parent_source_sha256=SOURCE_SHA,
        idle_timeout_policy_sha256=sha256(policy_path),changed_files=sorted(changed),workers_unchanged=True))
    receipt=dict(utc=campaign.utc(),operations_source=str(ops),operations_source_sha256=digest,
        worker_source_sha256=SOURCE_SHA,workflow_sha256=WORKFLOW_SHA,plan_sha256=PLAN_SHA,
        policy_sha256=sha256(policy_path),status='prepared_not_activated')
    write_json(root/'handoff/rpc_idle_timeout_v1/prepared.json',receipt)
    return receipt


def activate(root=ROOT):
    from .activate_input_recovery import preflight, restore_slots
    from .evaluation import case_identity, read_panel
    from .idle_timeout_policy import adjudicate
    handoff=root/'handoff/rpc_idle_timeout_v1'
    prepared=campaign.optional(handoff/'prepared.json')
    if (sha256(root/'campaign_input_recovery_v1.json') != PLAN_SHA
            or sha256(root/'workflow_input_recovery_v1.json') != WORKFLOW_SHA
            or campaign.optional(root/'STOP').get('reason') != REASON
            or (handoff/'activation_intent.json').exists()):
        raise ValueError('Unexpected or already attempted handoff')
    verified_source(Path(prepared['operations_source']),prepared['operations_source_sha256'])
    verified_source(root/'source_input_recovery_v1',SOURCE_SHA)
    if sha256(root/'rpc_idle_timeout_policy.json') != prepared['policy_sha256']:
        raise ValueError('Policy changed')
    plan=campaign.optional(root/'campaign_input_recovery_v1.json')
    with (root/'supervisor.lock').open('a') as sup, (Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock').open('a') as disp:
        fcntl.flock(sup,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(disp,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=campaign.optional(root/'state.json')
        if state['approved_sha256'] != PLAN_SHA:
            raise ValueError('State plan changed')
        revised,restored=restore_slots(campaign.optional(handoff/'before_state.json'),state)
        retry=[r for r in revised.get('pending_retries',[]) if r['index']==11]
        if len(retry)!=1 or retry[0]['attempt']!=8 or any(s.get('index')==11 and s['status'] in
                ('ready','submitting','submitted','running') for s in revised['slots']):
            raise ValueError('The exact unstarted retry is no longer pending')
        fresh=preflight(plan)
        # preflight lists only ACTIVE jobs, so separately resolve this old job read-only.
        import subprocess
        response=subprocess.run([plan['sco'],'acp','jobs','describe',
            '--workspace-name='+plan['workspace'],'-o','json','pt-k8ljr7oc'],
            capture_output=True,text=True,check=True,timeout=60)
        job=json.loads(response.stdout)
        if job.get('name')!='pt-k8ljr7oc' or job.get('ownership',{}).get('user_name')!='sujiayi':
            raise ValueError('Unexpected job identity')
        subpath=Path(plan['shared_root'])/'cluster/robodojo_v3_eef50_20260912_01_c11_a7/replica_1/submission.json'
        sub=campaign.optional(subpath);batch=campaign.optional(sub['batch_result'])
        archive=Path(plan['shared_root'])/'results/robodojo_v3_eef50_20260912_01_c11_a7/classify_objects_by_language/standard/eval_seed_0/layout_1/replica_1/attempt_7'
        expected=case_identity(read_panel(plan['eval_manifest']),plan['queue'][11]['case'])
        if job.get('display_name')!=sub['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+sub['environment']['ROLLOUT_REPLICA_ID']:
            raise ValueError('Wrong attempt display identity')
        if campaign.optional(root/'state.json') != state or campaign.optional(root/'STOP').get('reason') != REASON:
            raise ValueError('Maintenance state changed')
        row=adjudicate(plan,batch,archive,job,expected)
        if not row:
            raise ValueError('Idle timeout no longer matches verified evidence')
        write_json(handoff/'stopped_state.json',state)
        write_json(handoff/'fresh_preflight.json',fresh)
        intent=dict(utc=campaign.utc(),withdrawn_pending=retry,restored_slots=restored,
            containers_stopped=0,operations_source_sha256=prepared['operations_source_sha256'],
            adjudication_sha256=sha256(archive/'evaluation_adjudication.json'))
        write_json(handoff/'activation_intent.json',intent)
        revised['pending_retries']=[r for r in revised['pending_retries'] if r['index']!=11]
        revised.setdefault('skipped_indices',{})['11']='failed_rpc_idle_timeout'
        revised.setdefault('adjudicated_failures',{})['11']=dict(archive=str(archive),attempt=7,
            job_id=job['name'],reason=row['reason'])
        revised.setdefault('adjudication_events',[]).append(dict(intent,index=11,action='failed_rpc_idle_timeout_no_retry'))
        revised['updated_utc']=campaign.utc()
        write_json(root/'state.json',revised)
        write_json(handoff/'activation.json',dict(intent,status='activated',state_sha256=sha256(root/'state.json')))
        (root/'STOP').rename(handoff/'STOP_released')
    return dict(intent,preflight=fresh,status='activated')


def launch(root=ROOT):
    handoff=root/'handoff/rpc_idle_timeout_v1'
    prepared=campaign.optional(handoff/'prepared.json')
    if (root/'STOP').exists() or not (handoff/'activation.json').exists() or (handoff/'launch_intent.json').exists():
        raise ValueError('Require activated handoff, no STOP, and no previous launch intent')
    source=Path(prepared['operations_source'])
    verified_source(source,prepared['operations_source_sha256'])
    if sha256(root/'workflow_input_recovery_v1.json') != WORKFLOW_SHA:
        raise ValueError('Unexpected workflow')
    shared=root.parents[1]
    with (root/'supervisor.lock').open('a') as sup, (shared/'evaluation/campaign_dispatch.lock').open('a') as disp:
        fcntl.flock(sup,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(disp,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tmux=['tmux','-L','robodojo-v3-overnight']
    if subprocess.run(tmux+['has-session','-t','supervisor'],capture_output=True).returncode==0:
        raise ValueError('Supervisor tmux already exists')
    argv=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+str(source),
        '/usr/bin/python3','-u','-m','hybrid_rollout.robodojo.overnight','run',
        str(root/'workflow_input_recovery_v1.json'),'--approved-sha256',WORKFLOW_SHA,
        '--operations-source-sha256',prepared['operations_source_sha256']]
    log=root/'supervisor_rpc_idle_timeout_v1.log'
    command='exec '+shlex.join(argv)+' >> '+shlex.quote(str(log))+' 2>&1'
    receipt=dict(utc=campaign.utc(),cwd=str(source),argv=argv,log=str(log),containers_stopped=0,
        operations_source_sha256=prepared['operations_source_sha256'])
    write_json(handoff/'launch_intent.json',receipt)
    subprocess.run(tmux+['new-session','-d','-s','supervisor','-c',str(source),command],check=True,timeout=15)
    write_json(handoff/'launch.json',dict(receipt,tmux_started=True))
    return dict(receipt,tmux_started=True)


def pause(root=ROOT):
    plan = root/'campaign_input_recovery_v1.json'
    if sha256(plan) != PLAN_SHA:
        raise ValueError('Unexpected plan')
    handoff = root/'handoff/rpc_idle_timeout_v1'
    if handoff.exists() or (root/'STOP').exists():
        raise ValueError('Do not replay an existing handoff or overwrite STOP')
    state = campaign.optional(root/'state.json')
    if state['approved_sha256'] != PLAN_SHA:
        raise ValueError('Unexpected state')
    if any(s['status'] in ('ready', 'submitting', 'submitted', 'running')
           and s['index'] == 11 for s in state['slots']):
        raise ValueError('Case 11 already active; do not interfere with a live container')
    retry = [r for r in state.get('pending_retries', []) if r['index'] == 11]
    if len(retry) != 1 or retry[0]['attempt'] != 8:
        raise ValueError('Expected exact pending case 11 attempt 8')
    handoff.mkdir(parents=True)
    write_json(handoff/'before_state.json', state)
    write_json(handoff/'authorization.json', dict(utc=campaign.utc(),
        instruction='Confirmed simulator RPC 900-second idle timeout counts as evaluation failure; do not retry; preserve evidence for video labels.',
        native_outcome_unchanged=True, native_score_if_missing=None,
        containers_unchanged=True, plan_sha256=PLAN_SHA))
    with (root/'STOP').open('x') as stream:
        json.dump(dict(reason=REASON, utc=campaign.utc(), containers_unchanged=True), stream)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(paused_external_dispatch=True, containers_stopped=0, pending_retry=retry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pause', 'prepare', 'activate', 'launch'))
    print(json.dumps(globals()[parser.parse_args().action](), indent=2))
