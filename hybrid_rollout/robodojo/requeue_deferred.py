"""One operator-authorized replay of five exact deferred v3 cases, not a worker change."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from . import campaign
from .evaluation import case_identity, read_panel
from .io import sha256, write_json
from .subscription_migration import _lock

OPERATION = 'manual_requeue5_20260912_01'
TARGETS = {10: (0, 'pt-lzgql985'), 3: (2, 'pt-o1ctpbxm'),
           23: (2, 'pt-t61n3lga'), 36: (1, 'pt-ods4b7bd'), 45: (1, 'pt-nauv9yiu')}
PLAN_SHA = '8de77898e69a819fa4ad8624f9bad6608061d330d79eab95b3e96bc15477a2e2'
OPS_SHA = 'd064b1ea494bbca2d061f573239381f634353c98f196821e76b8c2a027609d3f'
ACTIVE = {'running', 'submitted', 'submitting', 'ready'}


def transition(state, targets, before_pause=None):
    """Pure all-or-nothing state transition; retain all other cases and active slots."""
    new = copy.deepcopy(state)
    if any(e.get('operation') == OPERATION for e in new.get('manual_deferred_requeues', [])):
        raise ValueError('This replay was already applied; never enqueue it twice')
    for index, (attempt, job) in targets.items():
        old = new.get('deferred_cases', {}).get(str(index), {}).get('original_slot', {})
        if (old.get('index') != index or old.get('attempt') != attempt or old.get('job_id') != job
                or old.get('status') != 'pause_unknown_failure'
                or not any(index in s.get('assigned', []) for s in new['slots'])
                or any(s.get('index') == index and s['status'] in ACTIVE for s in new['slots'])
                or any(r['index'] == index for r in new.get('pending_retries', []))):
            raise ValueError(f'Case {index} is not the exact unqueued deferred attempt')
    for slot in new['slots']:
        if slot['status'] == 'stopped':
            prior = next((s for s in (before_pause or {}).get('slots', []) if s['id'] == slot['id']), {})
            if prior.get('status') not in ACTIVE | {'idle', 'drained'} or any(
                    slot.get(k) != prior.get(k) for k in ['index', 'attempt', 'submission', 'job_id']):
                raise ValueError('Do not override an unrelated explicit stop or changed submission')
            slot['status'] = prior['status']
        if slot['status'] == 'drained':
            slot['status'] = 'idle'
    event = dict(operation=OPERATION, utc=campaign.utc(), cases=[], old_deferred={})
    for index, (attempt, job) in targets.items():
        event['old_deferred'][str(index)] = new['deferred_cases'].pop(str(index))
        event['cases'].append(dict(index=index, old_attempt=attempt, old_job_id=job, next_attempt=attempt+1))
        new.setdefault('pending_retries', []).append(dict(index=index, attempt=attempt+1,
            retry_number=0, next_retry_at=0, reason=OPERATION))
        # This is exactly one extra replay. Do not silently grant another automatic guard replay.
        counts = new.setdefault('input_guard_retry_counts', {})
        counts[str(index)] = max(1, int(counts.get(str(index), 0)))
    new.setdefault('manual_deferred_requeues', []).append(event)
    new['updated_utc'] = campaign.utc()
    return new


def verify_targets(plan, state, jobs, targets):
    panel = read_panel(plan['eval_manifest'])
    rows = []
    for index, (attempt, job_id) in targets.items():
        old = state['deferred_cases'][str(index)]['original_slot']
        case = plan['queue'][index]['case']
        sub = campaign.optional(old['submission']); batch = campaign.optional(sub['batch_result'])
        matches = [j for j in jobs if j.get('name') == job_id]
        if len(matches) != 1:
            raise ValueError('Require exactly one platform job for the old attempt')
        job = matches[0]
        if len(batch.get('episodes', [])) != 1:
            raise ValueError('Require one archived episode')
        archive = Path(batch['episodes'][0]['archive'])
        outcome = campaign.optional(archive/'sim/evaluation_outcome.json')
        result = campaign.optional(archive/'controller/result.json')
        error = campaign.optional(archive/'controller/failure.json').get('error', '')
        expected = case_identity(panel, case)
        if (old['attempt'] != attempt or old['job_id'] != job_id or old['index'] != index
                or job.get('state') != 'FAILED' or batch.get('state') != 'failed'
                or str(batch.get('attempt')) != str(attempt) or batch.get('exit_code') == 130
                or job.get('ownership', {}).get('user_name') != 'sujiayi'
                or job.get('display_name') != sub['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+sub['environment']['ROLLOUT_REPLICA_ID']
                or sub.get('evaluation_panel_sha256') != panel['panel_sha256']
                or sub.get('evaluation_cases') != [case]
                or outcome.get('evaluation_case') != expected or result.get('evaluation_case') != expected
                or outcome.get('complete') is not False or result.get('complete') is not False
                or 'InputError' not in error or 'Five rejected tool calls' not in error
                or campaign.optional(archive/'artifact_manifest.json').get('status') != 'verified'
                or not all(batch.get('final_reset', {}).get(k) is True for k in ['all_owned_processes_exited','ports_released'])):
            raise ValueError(f'Case {index} is not a verified, cleaned-up, native-incomplete guard failure')
        # Reject later submissions (including ambiguous or incomplete preparations), not just live slots.
        prefix = plan['queue'][index].get('experiment_prefix', plan['experiment_prefix'])
        for directory in (Path(plan['shared_root'])/'cluster').glob(f'{prefix}_c{index}_a*'):
            if int(directory.name.rsplit('_a', 1)[1]) > attempt:
                raise ValueError(f'Case {index} already has a later preparation/submission')
        rows.append(dict(index=index, case=case, old_attempt=attempt, next_attempt=attempt+1,
            old_job_id=job_id, old_archive=str(archive), old_submission=old['submission'],
            old_source_sha256=sub.get('source_sha256'), worker_source_sha256=plan['source_sha256']))
    return rows


def validate_plan(path):
    plan = campaign.optional(path)
    if (sha256(path) != PLAN_SHA or plan.get('primary_target') != 50
            or len(plan['slots']) != 15 or plan.get('model') != 'gpt-6-astra'
            or plan.get('effort') != 'xhigh' or plan.get('fast') is not False):
        raise ValueError('Require the unchanged, approved v3 50-case plan')
    ops = path.parent/'operations_fill15_v2'
    for source, digest in [(ops, OPS_SHA), (Path(plan['dispatcher_source']), plan['source_sha256'])]:
        manifest = campaign.optional(source/'source_manifest.json')
        if manifest.get('source_sha256') != digest or any(sha256(source/n) != h for n,h in manifest['files'].items()):
            raise ValueError('An immutable source snapshot changed')
    return plan


def inspect(path, state=None):
    plan = validate_plan(path); state = state or campaign.optional(path.parent/'state.json')
    transition(state, TARGETS)  # Validate without modifying shared state.
    jobs = campaign.job_list(plan); rows = verify_targets(plan, state, jobs, TARGETS)
    totals = {'RESERVED': 0, 'SPOT': 0}
    for job in jobs:
        if job.get('state') not in {'RUNNING','CREATING','STARTING','RECOVERING','RESTARTING','QUEUEING'}:
            continue
        quota = job.get('scheduling', {}).get('quota_type', 'SPOT')
        totals[quota] = totals.get(quota, 0) + sum(int(spec.get('limits', {}).get('nvidia.com/gpu',0))
            * int(spec.get('replicas',role.get('total_replicas',1)))
            for role in job.get('roles',[]) for spec in role.get('resource_spec',[]))
    if totals['RESERVED'] + 10 > plan['max_concurrent_gpus']:
        raise ValueError('The five requested replays do not fit the reviewed 30-GPU limit')
    return dict(utc=campaign.utc(), operation=OPERATION, plan_sha256=PLAN_SHA,
        operations_source_sha256=OPS_SHA, worker_source_sha256=plan['source_sha256'],
        targets=rows, gpu_current=totals, requested_gpus=10, projected_reserved=totals['RESERVED']+10,
        same_seed_and_v3_settings=True, active_containers_unchanged=True)


def pause(path, confirmation):
    root = path.parent; operation = root/OPERATION
    if not confirmation or (root/'STOP').exists() or list(root.glob('STOP_SLOT_*')) or operation.exists():
        raise ValueError('Require fresh explicit authorization and no other stop/operation')
    before = campaign.optional(root/'state.json'); report = inspect(path, before)
    operation.mkdir()
    write_json(operation/'authorization.json', dict(report, user_confirmation=confirmation))
    write_json(operation/'before_pause.json', before)
    write_json(root/'STOP', dict(reason=OPERATION, utc=campaign.utc()))
    print(json.dumps(report))


def activate(path):
    root = path.parent; operation = root/OPERATION; plan = validate_plan(path)
    auth = campaign.optional(operation/'authorization.json')
    if auth.get('plan_sha256') != PLAN_SHA or not auth.get('user_confirmation') or campaign.optional(root/'STOP').get('reason') != OPERATION or list(root.glob('STOP_SLOT_*')):
        raise ValueError('Require this exact authorized maintenance stop')
    with _lock(plan):
        state = campaign.optional(root/'state.json')
        if state.get('approved_sha256') != PLAN_SHA:
            raise ValueError('Active state belongs to another plan')
        revised = transition(state, TARGETS, campaign.optional(operation/'before_pause.json'))
        jobs = campaign.job_list(plan); rows = verify_targets(plan, state, jobs, TARGETS)
        if (operation/'before_apply.json').exists():
            raise ValueError('Prior application evidence exists; reconcile it before writing again')
        write_json(operation/'before_apply.json', state)
        shutil.copy2(root/'STOP', operation/'STOP.before')
        write_json(root/'state.json', revised)
        write_json(operation/'activation.json', dict(utc=campaign.utc(), operation=OPERATION,
            plan_sha256=PLAN_SHA, targets=rows, queued=True, new_creates=0,
            worker_and_operations_unchanged=True, old_results_preserved=True))
        (root/'STOP').rename(operation/'STOP.released')
        print('Five exact cases requeued once; resume the same immutable dispatcher, not a new worker build.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect','pause','activate'])
    parser.add_argument('plan', type=Path); parser.add_argument('--confirmation')
    args = parser.parse_args(); path = args.plan.resolve()
    if args.action == 'inspect': print(json.dumps(inspect(path)))
    elif args.action == 'pause': pause(path, args.confirmation)
    else: activate(path)


if __name__ == '__main__': main()
