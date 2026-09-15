"""One-shot, user-authorized guard retry and faster operations-only slot polling."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import campaign
from .evaluation import case_identity, read_panel
from .io import sha256, write_json
from .subscription_migration import _lock

REVISION = 'fill15_v1'
STOP_REASON = 'user_authorized_guard_retry_and_fill15'
TARGET = dict(slot=10, index=20, attempt=1, job_id='pt-z85m9n1v',
              case_id='organize_table__standard__g0__l2')
GUARD_REVISION = 'fill15_v2'
GUARD_STOP_REASON = 'user_requested_keep_slots_full_bounded_guard_queue'


def prepare_guard_operations(path, approved, confirmation):
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    if sha256(path)!=approved or not confirmation or plan.get('primary_target')!=50 or len(plan['slots'])!=15:
        raise ValueError('Require actual authorization for the original 15-slot/50-case plan')
    target=root/f'operations_{GUARD_REVISION}'
    if target.exists():raise ValueError('Do not overwrite operations snapshot')
    old=campaign.optional(root/f'operations_{REVISION}.json')
    parent=Path(old['operations_source']);manifest=campaign.optional(parent/'source_manifest.json')
    if manifest['source_sha256']!=old['operations_source_sha256'] or any(sha256(parent/n)!=h for n,h in manifest['files'].items()):
        raise ValueError('Previous operations source changed')
    shutil.copytree(parent/'hybrid_rollout',target/'hybrid_rollout')
    names=['hybrid_rollout/robodojo/campaign.py','hybrid_rollout/robodojo/elastic_dispatch.py','hybrid_rollout/robodojo/input_guard_queue.py']
    repo=Path(__file__).resolve().parents[2]
    for name in names:shutil.copy2(repo/name,target/name)
    files={str(p.relative_to(target)):sha256(p) for p in target.rglob('*') if p.is_file()}
    if sorted(n for n,h in files.items() if manifest['files'].get(n)!=h)!=names or set(manifest['files'])-set(files):
        raise ValueError('Unexpected non-operations changes')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(target/'source_manifest.json',dict(manifest,files=files,source_sha256=digest,
        parent_source_sha256=manifest['source_sha256'],changed_files=names))
    value=dict(utc=campaign.utc(),user_confirmed=True,user_confirmation=confirmation,
        approved_plan_sha256=approved,operations_source=str(target),operations_source_sha256=digest,
        worker_source_sha256=plan['source_sha256'],poll_seconds=30,retry_input_guards_once=True,
        max_input_guard_retries_per_case=1,guard_retry_backoff_seconds=300,
        repeated_guard_action='defer_case_and_release_slot',running_or_native_complete_never_retried=True)
    write_json(root/f'operations_{GUARD_REVISION}.json',value)
    print(json.dumps(value))


def activate_guard_operations(path, approved):
    root=path.resolve().parent;plan=campaign.optional(path)
    receipt=campaign.optional(root/f'operations_{GUARD_REVISION}.json')
    if sha256(path)!=approved or receipt.get('approved_plan_sha256')!=approved or receipt.get('user_confirmed') is not True or campaign.optional(root/'STOP').get('reason')!=GUARD_STOP_REASON:
        raise ValueError('Require this exact authorized operations switch')
    with _lock(plan):
        state=campaign.optional(root/'state.json')
        if state['approved_sha256']!=approved or list(root.glob('STOP_SLOT_*')):
            raise ValueError('Another plan or explicit slot stop is active')
        for slot in state['slots']:
            if slot.get('submission') and Path(slot['submission']).with_name('submission_cancelled.json').exists():
                raise ValueError('Cancelled create needs separate reconciliation')
        backup=root/f'before_{GUARD_REVISION}';backup.mkdir()
        shutil.copy2(root/'state.json',backup/'state.json');shutil.copy2(root/'STOP',backup/'STOP')
        for slot in state['slots']:
            if slot['status']=='stopped':
                sub=slot.get('submission')
                slot['status']=('submitted' if Path(sub).with_name('submission_started.json').exists() else 'ready') if sub else 'idle'
        state['updated_utc']=campaign.utc();write_json(root/'state.json',state)
        write_json(root/f'activation_{GUARD_REVISION}.json',dict(utc=campaign.utc(),**{
            k:receipt[k] for k in ['approved_plan_sha256','operations_source_sha256','retry_input_guards_once','poll_seconds']},
            worker_source_unchanged=True,active_containers_unchanged=True))
        (root/'STOP').rename(backup/'STOP_released')
        print('Bounded guard recycling activated; start exactly one dispatcher with --retry-input-guards-once.')


def retry_state(state, target, started_paths):
    """Release only the authorized slot; put the next attempt in the shared queue."""
    result=copy.deepcopy(state)
    slot=next(s for s in result['slots'] if s['id']==target['slot'])
    if (slot['status']!='pause_unknown_failure' or slot['job_id']!=target['job_id']
            or slot['index']!=target['index'] or slot['attempt']!=target['attempt']
            or str(slot['index']) in result.get('deferred_cases', {})
            or any(r['index']==slot['index'] for r in result.get('pending_retries', []))):
        raise ValueError('Require the exact paused, unqueued attempt')
    if slot['index'] not in slot.get('assigned', []):
        raise ValueError('Retry must preserve the original case reservation')
    history=dict(index=slot['index'],attempt=slot['attempt'],job_id=slot['job_id'],
        submission=slot['submission'],action='user_authorized_input_guard_retry',
        finished_utc=campaign.utc(),next_attempt=slot['attempt']+1)
    slot.setdefault('history', []).append(history)
    result.setdefault('pending_retries', []).insert(0,dict(index=slot['index'],
        attempt=slot['attempt']+1,retry_number=slot.get('retry_number',0),next_retry_at=0,
        reason='explicit_user_retry_after_terminal_input_guard'))
    slot.update(status='idle',submission=None,job_id=None,platform_state=None,
        next_retry_at=0,last_action='user_authorized_input_guard_retry')
    for other in result['slots']:
        if other['status']=='stopped':
            sub=other.get('submission')
            other['status']=('submitted' if sub in started_paths else 'ready') if sub else 'idle'
    return result


def prepare(path, approved, confirmation):
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    if (sha256(path)!=approved or not confirmation or len(plan['slots'])!=15
            or plan['primary_target']!=50 or plan['context_version']!='v3'):
        raise ValueError('Require actual user authorization for the existing v3 plan')
    receipt=root/f'operations_{REVISION}.json'
    if receipt.exists():raise ValueError('Do not overwrite a preparation or authorization')
    previous=campaign.optional(root/'operations_continue_query_v1.json')
    original=Path(previous['operations_source']);manifest=campaign.optional(original/'source_manifest.json')
    if manifest['source_sha256']!=previous['operations_source_sha256'] or any(
            sha256(original/name)!=digest for name,digest in manifest['files'].items()):
        raise ValueError('Previous operations snapshot changed')
    frozen=root/f'operations_{REVISION}'
    if frozen.exists():raise ValueError('Do not overwrite a source snapshot')
    shutil.copytree(original/'hybrid_rollout',frozen/'hybrid_rollout')
    name='hybrid_rollout/robodojo/campaign.py'
    shutil.copy2(Path(__file__).resolve().parents[2]/name,frozen/name)
    files={str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*') if p.is_file()}
    changed=sorted(name for name,digest in files.items() if manifest['files'].get(name)!=digest)
    if changed!=[name] or set(files)!=set(manifest['files']):
        raise ValueError('Only the dispatcher cadence may change')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(frozen/'source_manifest.json',dict(manifest,files=files,source_sha256=digest,
        parent_source_sha256=manifest['source_sha256'],changed_files=changed))
    value=dict(utc=campaign.utc(),user_confirmation=confirmation,user_confirmed=True,
        approved_plan_sha256=approved,target=TARGET,poll_seconds=30,
        operations_source=str(frozen),operations_source_sha256=digest,
        worker_source_sha256=plan['source_sha256'],plan_unchanged=True,
        active_containers_unchanged=True,automatic_guard_retry_policy_changed=False)
    write_json(receipt,value)
    print(json.dumps(value))


def activate(path, approved):
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    receipt=campaign.optional(root/f'operations_{REVISION}.json')
    if (sha256(path)!=approved or receipt.get('approved_plan_sha256')!=approved
            or receipt.get('user_confirmed') is not True
            or campaign.optional(root/'STOP').get('reason')!=STOP_REASON):
        raise ValueError('Require authorization and this exact maintenance stop')
    with _lock(plan):
        state=campaign.optional(root/'state.json')
        if state['approved_sha256']!=approved or list(root.glob('STOP_SLOT_*')):
            raise ValueError('Another plan or explicit slot stop is active')
        slot=next(s for s in state['slots'] if s['id']==TARGET['slot'])
        # Validate the pure transition before examining external evidence.
        started={s['submission'] for s in state['slots'] if s.get('submission')
            and Path(s['submission']).with_name('submission_started.json').exists()}
        revised=retry_state(state,TARGET,started)
        for s in state['slots']:
            if s.get('submission') and Path(s['submission']).with_name('submission_cancelled.json').exists():
                raise ValueError('A cancelled create requires separate reconciliation; do not blindly recreate')
        jobs=campaign.job_list(plan);job=next(j for j in jobs if j['name']==TARGET['job_id'])
        sub=campaign.optional(slot['submission']);batch=campaign.optional(sub['batch_result'])
        if len(batch.get('episodes',[]))!=1:raise ValueError('Require exactly one episode')
        episode=batch['episodes'][0];archive=Path(episode['archive'])
        expected=case_identity(read_panel(plan['eval_manifest']),plan['queue'][TARGET['index']]['case'])
        outcome=campaign.optional(archive/'sim/evaluation_outcome.json')
        result=campaign.optional(archive/'controller/result.json')
        if (episode['case_id']!=TARGET['case_id'] or outcome.get('evaluation_case')!=expected
                or result.get('evaluation_case')!=expected or outcome.get('complete') is not False
                or result.get('complete') is not False or job['state'] not in campaign.TERMINAL
                or batch.get('exit_code')==130
                or job.get('ownership',{}).get('user_name')!='sujiayi'
                or job.get('display_name')!=sub['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+sub['environment']['ROLLOUT_REPLICA_ID']
                or not all(batch.get('final_reset',{}).get(k) is True for k in ['all_owned_processes_exited','ports_released'])
                or campaign.optional(archive/'artifact_manifest.json').get('status')!='verified'
                or 'Five rejected tool calls' not in campaign.optional(archive/'controller/failure.json').get('error','')):
            raise ValueError('Target is not a clean, verified, terminal, incomplete input-guard attempt')
        gpus={quota:sum(int(spec.get('limits',{}).get('nvidia.com/gpu',0))*int(spec.get('replicas',role.get('total_replicas',1)))
            for j in jobs if j['state'] in ['RUNNING','CREATING','STARTING','RECOVERING','RESTARTING','QUEUEING']
            and j.get('scheduling',{}).get('quota_type')==quota
            for role in j.get('roles',[]) for spec in role.get('resource_spec',[])) for quota in ['RESERVED','SPOT']}
        backup=root/f'before_{REVISION}';backup.mkdir()
        shutil.copy2(root/'state.json',backup/'state.json');shutil.copy2(root/'STOP',backup/'STOP')
        revised['updated_utc']=campaign.utc();write_json(root/'state.json',revised)
        write_json(root/f'activation_{REVISION}.json',dict(utc=campaign.utc(),approved_plan_sha256=approved,
            target=TARGET,next_attempt=TARGET['attempt']+1,shared_queue=True,raw_results_unchanged=True,
            model_prompt_guard_unchanged=True,created_jobs=0,gpu_preflight=gpus,
            operations_source_sha256=receipt['operations_source_sha256'],poll_seconds=30))
        (root/'STOP').rename(backup/'STOP_released')
        print(json.dumps(dict(retry_queued=TARGET,next_attempt=2,gpu_preflight=gpus,active_containers_unchanged=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','pause','activate','guard-prepare','guard-pause','guard-activate']);parser.add_argument('plan',type=Path)
    parser.add_argument('--approved-sha256',required=True);parser.add_argument('--confirmation');args=parser.parse_args()
    if args.action=='prepare':prepare(args.plan,args.approved_sha256,args.confirmation)
    elif args.action=='activate':activate(args.plan,args.approved_sha256)
    elif args.action=='guard-prepare':prepare_guard_operations(args.plan,args.approved_sha256,args.confirmation)
    elif args.action=='guard-activate':activate_guard_operations(args.plan,args.approved_sha256)
    else:
        revision=GUARD_REVISION if args.action=='guard-pause' else REVISION
        reason=GUARD_STOP_REASON if args.action=='guard-pause' else STOP_REASON
        root=args.plan.resolve().parent;r=campaign.optional(root/f'operations_{revision}.json')
        if sha256(args.plan)!=args.approved_sha256 or r.get('approved_plan_sha256')!=args.approved_sha256 or not r.get('user_confirmed') or (root/'STOP').exists():
            raise ValueError('Do not override another stop intent or unapproved plan')
        write_json(root/'STOP',dict(reason=reason,utc=campaign.utc()))


if __name__=='__main__':main()
