"""Freeze/review a same-thread network-wakeup runtime without editing live snapshots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import campaign
from .io import sha256, write_json
from .subscription_migration import _lock

REVISION = 'continue_v3_1'
CHANGED = ['hybrid_rollout/robodojo/skill/network_recovery.py', 'hybrid_rollout/robodojo/skill/run.py']
STOP_REASON = 'reviewed_network_continue_migration'


def prepare(path):
    path=path.resolve();root=path.parent;old=campaign.optional(path)
    if old.get('context_version')!='v3' or len(old['slots'])!=15 or old['primary_target']!=50:
        raise ValueError('Require the existing reviewed v3 plan')
    target=root/f'campaign_{REVISION}.json'
    if target.exists():raise ValueError('Do not overwrite a frozen revision')
    original=Path(old['dispatcher_source']);manifest=campaign.optional(original/'source_manifest.json')
    if manifest['source_sha256']!=old['source_sha256'] or any(
            sha256(original/f)!=digest for f,digest in manifest['files'].items()):
        raise ValueError('Original runtime changed')
    frozen=root/f'source_{REVISION}'
    shutil.copytree(original/'hybrid_rollout',frozen/'hybrid_rollout')
    repo=Path(__file__).resolve().parents[2]
    for name in CHANGED:shutil.copy2(repo/name,frozen/name)
    files={str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*') if p.is_file()}
    changed=sorted(f for f in files if files[f]!=manifest['files'].get(f))
    if changed!=CHANGED or set(manifest['files'])-set(files):raise ValueError('Unexpected runtime edits')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(frozen/'source_manifest.json',dict(manifest,files=files,source_sha256=digest,
        parent_source_sha256=old['source_sha256'],changed_files=changed))
    new=copy.deepcopy(old)
    new.update(source_sha256=digest,dispatcher_source=str(frozen),operations_source_sha256=digest,
        previous_plan=str(path),previous_plan_sha256=sha256(path),
        network_recovery=dict(revision=REVISION,mode='continue_same_thread_after_terminal_network_error',
            consecutive_wakeups=20,interval_seconds=10,reset_on_successful_service_call=True,
            new_thread=False,simulator_restart=False,action_replay=False,
            controller_error_budget_unchanged=True,active_snapshots_unchanged=True,
            fallback='existing same-seed container retry after wakeups exhausted'))
    write_json(target,new)
    print(json.dumps(dict(plan=str(target),sha256=sha256(target),source_sha256=digest,changed_files=changed)))


def activate(path,approved):
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    receipt=campaign.optional(root/f'authorization_{REVISION}.json')
    if (sha256(path)!=approved or receipt.get('approved_sha256')!=approved
            or receipt.get('user_confirmed') is not True
            or campaign.optional(root/'STOP').get('reason') != STOP_REASON):
        raise ValueError('Require explicit review of this revision and a paused dispatcher')
    with _lock(plan):
        state=campaign.optional(root/'state.json')
        if state['approved_sha256']!=plan['previous_plan_sha256']:
            raise ValueError('Another plan is already active')
        backup=root/f'before_{REVISION}';backup.mkdir()
        for name in ['state.json','launch_authorization.json','STOP']:shutil.copy2(root/name,backup/name)
        pending = []
        for slot in state['slots']:
            sub=Path(slot['submission']) if slot.get('submission') else None
            if sub and not sub.with_name('submission_started.json').exists():
                if Path(campaign.optional(sub)['batch_result']).exists():
                    raise ValueError('Unsubmitted preparation unexpectedly has runtime results')
                pending.append(sub)
        for slot in state['slots']:
            sub=Path(slot['submission']) if slot.get('submission') else None
            if sub in pending:
                # Only never-submitted preparation can be replaced. Preserve it outside cluster scan.
                destination=backup/f'unsubmitted_slot_{slot["id"]}'
                sub.parent.rename(destination)
                slot.update(status='ready',submission=None,job_id=None,platform_state=None,next_retry_at=0)
            elif slot['status']=='stopped':
                slot['status']='submitted' if sub else 'idle'
        state.update(approved_sha256=approved,updated_utc=campaign.utc())
        write_json(root/'state.json',state)
        write_json(root/'launch_authorization.json',dict(approved_sha256=approved,
            accounts_confirmed=True,user_confirmation=receipt))
        write_json(root/f'activation_{REVISION}.json',dict(utc=campaign.utc(),approved_sha256=approved,
            previous_plan=plan['previous_plan'],active_rollouts_unchanged=True))
        (root/'STOP').rename(backup/'STOP_released')
        print('Activated; start exactly one dispatcher from the newly reviewed source.')


def freeze_query_fix(path, approved):
    """Separate control-plane fix; the approved plan and worker bytes stay unchanged."""
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    receipt=campaign.optional(root/f'authorization_{REVISION}.json')
    if sha256(path)!=approved or receipt.get('approved_sha256')!=approved or not receipt.get('user_confirmed'):
        raise ValueError('Require reviewed rollout revision')
    original=Path(plan['dispatcher_source']);manifest=campaign.optional(original/'source_manifest.json')
    if manifest['source_sha256']!=plan['source_sha256'] or any(
            sha256(original/name)!=digest for name,digest in manifest['files'].items()):
        raise ValueError('Approved rollout source changed')
    target=root/'operations_continue_query_v1'
    if target.exists():raise ValueError('Do not overwrite an operations snapshot')
    shutil.copytree(original/'hybrid_rollout',target/'hybrid_rollout')
    name='hybrid_rollout/robodojo/acp_query.py'
    shutil.copy2(Path(__file__).resolve().parents[2]/name,target/name)
    files={str(p.relative_to(target)):sha256(p) for p in target.rglob('*') if p.is_file()}
    if sorted(f for f in files if files[f]!=manifest['files'].get(f))!=[name] or set(files)!=set(manifest['files']):
        raise ValueError('Operations snapshot must change only ACP read-only query')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(target/'source_manifest.json',dict(manifest,files=files,source_sha256=digest,
        parent_source_sha256=plan['source_sha256'],changed_files=[name]))
    value=dict(utc=campaign.utc(),approved_rollout_plan_sha256=approved,
        operations_source=str(target),operations_source_sha256=digest,
        rollout_source_sha256=plan['source_sha256'],rollout_plan_unchanged=True,
        changed_files=[name],reason='sco exact empty page: No jobs found; other malformed responses still fail closed')
    write_json(root/'operations_continue_query_v1.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','authorize','pause','activate','freeze-query-fix'])
    p.add_argument('plan',type=Path);p.add_argument('--approved-sha256');p.add_argument('--confirmation');args=p.parse_args()
    if args.action=='prepare':prepare(args.plan)
    elif args.action=='authorize':
        root=args.plan.resolve().parent;plan=campaign.optional(args.plan)
        if sha256(args.plan)!=args.approved_sha256 or not args.confirmation:
            raise ValueError('Exact plan hash and actual user confirmation required')
        receipt=root/f'authorization_{REVISION}.json'
        if receipt.exists():raise ValueError('Do not overwrite an approval receipt')
        jobs=campaign.job_list(plan);gpus={'RESERVED':0,'SPOT':0}
        for job in jobs:
            if job['state'] not in {'RUNNING','CREATING','STARTING','RECOVERING','RESTARTING','QUEUEING'}:continue
            count=sum(int(spec.get('limits',{}).get('nvidia.com/gpu',
                spec.get('requests',{}).get('nvidia.com/gpu',0)) or 0)*
                int(spec.get('replicas',role.get('total_replicas',1)))
                for role in job.get('roles',[]) for spec in role.get('resource_spec',[]))
            quota=job.get('scheduling',{}).get('quota_type','SPOT').upper()
            gpus[quota]+=count
        document=Path(__file__).resolve().parents[2]/'docs/operations/robodojo_v3_continue_confirmation.md'
        value=dict(approved_sha256=args.approved_sha256,user_confirmed=True,
            user_confirmation=args.confirmation,utc=campaign.utc(),configuration_document=str(document),
            configuration_document_sha256=sha256(document),gpu_preflight=gpus,
            max_concurrent_gpus=plan['max_concurrent_gpus'],new_capacity_requested=0,
            continue_policy=plan['network_recovery'],reset_enabled=False)
        write_json(receipt,value)
        print(json.dumps(dict(authorization=str(receipt),gpu_preflight=gpus,additional_capacity=0)))
    elif args.action=='freeze-query-fix':freeze_query_fix(args.plan,args.approved_sha256)
    elif args.action=='pause':
        root=args.plan.resolve().parent
        receipt=campaign.optional(root/f'authorization_{REVISION}.json')
        if (sha256(args.plan)!=args.approved_sha256 or receipt.get('approved_sha256')!=args.approved_sha256
                or receipt.get('user_confirmed') is not True or (root/'STOP').exists()):
            raise ValueError('Explicit revision approval required; do not override another stop intent')
        write_json(root/'STOP',dict(reason=STOP_REASON,utc=campaign.utc()))
    else:activate(args.plan,args.approved_sha256)
