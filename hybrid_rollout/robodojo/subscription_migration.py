"""Reviewed legacy-to-subscriptions migration; preserve active and paused jobs.

Drain the last Galbot container first, then briefly stop the external dispatcher.
Prepare/activate run under its exclusive lock. Native completed cases stay used;
retired pending cases return to the same panel queue with new attempts. Old
provider STOP markers and state are archived, not discarded. No implicit create.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import campaign
from .codex_backend.profiles import validate_campaign_sessions
from .evaluation import case_identity, read_panel
from .io import sha256, write_json

REVISION = 'subscriptions_v5'
TOPOLOGIES = {
    'a3_b3': (REVISION, campaign.SUBSCRIPTION_SLOTS),
    'b5_a1': ('subscriptions_b5_v6', campaign.B5_A1_SLOTS),
    'b5_a2': ('subscriptions_b5a2_v7', campaign.B5_A2_SLOTS),
}


def migration_spec(topology):
    if topology not in TOPOLOGIES:
        raise ValueError('Unknown subscription topology; no implicit account substitution')
    return TOPOLOGIES[topology]


def revised_state(plan, state, completed, next_attempts, topology='a3_b3'):
    """Pure transformation; callers must verify terminal jobs and ledger first."""
    if campaign.validate_topology(plan) != campaign.SLOTS:
        raise ValueError('Migration requires the original topology')
    _, target_slots = migration_spec(topology)
    old = {s['id']: s for s in state['slots']}
    if set(old) != set(range(7)) or any(old[i]['auth_profile'] != campaign.SLOTS[i] for i in old):
        raise ValueError('Unexpected source slots')
    replaced = (0, 2, 6) if topology == 'b5_a2' else (0, 2)
    if any(old[i]['index'] in completed for i in replaced):
        raise ValueError('Never rerun completed cases when replacing a slot')
    result, revised = copy.deepcopy(state), copy.deepcopy(plan)
    for entry in revised['queue']:
        entry.pop('slot_id', None)  # Pending work is shared across the reviewed subscription slots.
    retired = (1, 6) if topology == 'b5_a2' else (1,)
    result['retired_slots'] = result.get('retired_slots', []) + [copy.deepcopy(old[i]) for i in retired]
    result['slots'] = []
    for i, name in target_slots.items():
        source_id = 6 if i == 7 else i
        slot = copy.deepcopy(old[source_id])
        if source_id in replaced:
            slot.setdefault('history', []).append(dict(
                index=slot['index'], attempt=slot['attempt'], submission=slot['submission'],
                job_id=slot['job_id'], auth_profile=slot['auth_profile'], source_slot_id=source_id,
                action='user_authorized_subscription_migration', finished_utc=campaign.utc()))
            slot.update(id=i, auth_profile=name, attempt=next_attempts[slot['index']], status='ready',
                        submission=None, job_id=None, platform_state=None, next_retry_at=0,
                        retry_number=0, last_action='user_authorized_subscription_migration')
        result['slots'].append(slot)
    # Retired Galbot indices must not disappear: completed ones remain reserved;
    # incomplete ones are eligible for a NEW same-case attempt on an official slot.
    for index in old[1].get('assigned', []):
        if index in completed:
            result['slots'][0].setdefault('assigned', []).append(index)
        else:
            revised['queue'][index]['initial_attempt'] = next_attempts[index]
    revised['slots'] = [dict(id=i, auth_profile=n) for i, n in target_slots.items()]
    revised['max_concurrent_gpus'] = 12
    return revised, result


def _lock(plan):
    lock = (Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock


def prepare(path, topology='a3_b3'):
    path = path.resolve()
    root = path.parent
    revision, target_slots = migration_spec(topology)
    target = root/f'campaign_{revision}.json'
    if not (root/'STOP').is_file() or target.exists():
        raise ValueError('Require a stopped dispatcher and fresh revision; do not overwrite')
    plan = campaign.optional(path)
    with _lock(plan):
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != sha256(path):
            raise ValueError('Parent plan/state mismatch')
        validate_campaign_sessions(plan['shared_root'], tuple(target_slots.values()))
        panel = read_panel(plan['eval_manifest'], plan['panel_sha256'])
        if sha256(plan['scope_file']) != plan['scope_file_sha256']:
            raise ValueError('Frozen case selection changed')
        ledger = campaign.build_ledger(plan['shared_root'], panel)
        if ledger['errors']:
            raise ValueError('Resolve ledger errors before migration')
        rows = {r['case_id']: r for r in ledger['cases']}
        completed = {i for i,e in enumerate(plan['queue']) if rows[e['case']['case_id']]['state'] == 'completed'}
        next_attempts = {i: 1+max((int(a['attempt']) for a in rows[e['case']['case_id']]['attempts']), default=-1)
                         for i,e in enumerate(plan['queue'])}
        jobs = campaign.job_list(plan)
        evidence = []
        # Active B containers are adopted unchanged. Replacing the paused A
        # stream also requires authoritative termination and cleanup evidence.
        required_terminal = (0, 1, 2, 6) if topology == 'b5_a2' else (0, 1, 2)
        for slot in (s for s in state['slots'] if s['id'] in required_terminal):
            submission = campaign.optional(slot['submission'])
            name = submission['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+submission['environment']['ROLLOUT_REPLICA_ID']
            matches = [j for j in jobs if j.get('display_name') == name
                       and j.get('ownership', {}).get('user_name') == 'sujiayi']
            if len(matches) != 1 or matches[0]['state'] not in ('SUCCEEDED', 'FAILED'):
                raise ValueError('Replacement jobs must be terminal; never assume missing/manual-stop/live jobs are stopped')
            slot['job_id'] = matches[0]['name']
            batch = campaign.optional(submission['batch_result'])
            if not all(batch.get('final_reset', {}).get(k) is True
                       for k in ('all_owned_processes_exited', 'ports_released')):
                raise ValueError('Require API job cleanup evidence before replacing slots')
            episode = batch['episodes'][0]
            archive = Path(episode['archive'])
            result = campaign.optional(archive/'controller/result.json')
            expected = case_identity(panel, plan['queue'][slot['index']]['case'])
            if result.get('evaluation_case') != expected or episode['case_id'] != expected['case_id']:
                raise ValueError('Case/seed identity changed')
            action = campaign.assess(batch, archive, matches[0]['state'], expected)
            if slot['id'] == 0:
                if expected['task'] != 'organize_table' or result.get('complete') or result.get('reason') != 'model_stop':
                    raise ValueError('Slot 0 must be the reviewed incomplete model stop')
            elif slot['id'] == 2:
                failure = campaign.optional(archive/'controller/failure.json')
                if (expected['task'] != 'organize_table' or result.get('complete')
                        or '413' not in json.dumps(failure)):
                    raise ValueError('Slot 2 must be the reviewed incomplete intermediary413')
            elif slot['id'] == 6:
                if (expected['task'] != 'imitate_sorting_sequence' or result.get('complete')
                        or result.get('reason') != 'model_stop'):
                    raise ValueError('Old A replacement requires the reviewed incomplete model stop')
            elif action not in ('completed', 'retry_transport'):
                raise ValueError('Review the last Galbot outcome before migration')
            evidence.append(dict(slot=slot['id'], job_id=matches[0]['name'], platform_state=matches[0]['state'],
                result_path=str(archive/'controller/result.json'), result_sha256=sha256(archive/'controller/result.json'),
                assessment=action))
        # A manual STOP on another account must remain effective, not get lost.
        markers = {p.name: sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()}
        if 'STOP_SLOT_2' not in markers or any(n not in ('STOP_SLOT_1', 'STOP_SLOT_2') for n in markers):
            raise ValueError('Unexpected manual slot stop; reconcile explicitly')
        new_plan, new_state = revised_state(plan, state, completed, next_attempts, topology)
        fresh_slots = (0, 2, 7) if topology == 'b5_a2' else (0, 2)
        if topology == 'b5_a2':
            # Keep earlier, unsubmitted A attempt4/source intact. A new namespace
            # makes the different managed session and source explicit.
            for slot in new_state['slots']:
                if slot['id'] in fresh_slots:
                    new_plan['queue'][slot['index']]['experiment_prefix'] = f"{plan['experiment_prefix']}_{revision}"
        operation = root/({'a3_b3': 'subscription_source_v5', 'b5_a1': 'subscription_b5_source_v6',
                           'b5_a2': 'subscription_b5a2_source_v7'}[topology])
        repo = Path(__file__).resolve().parents[2]
        shutil.copytree(repo/'hybrid_rollout', operation/'hybrid_rollout',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        files = {str(p.relative_to(operation)): sha256(p) for p in sorted(operation.rglob('*')) if p.is_file()}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(operation/'source_manifest.json', dict(git_base_commit=commit,
            source_sha256=digest, includes_reviewed_worktree_changes=True, files=files))
        new_plan.pop('slot_runtime_overrides', None)
        for key in ('repair_submissions', 'repair_stop_hashes', 'repair_parent_state_sha256'):
            new_plan.pop(key, None)
        new_plan.update(parent_plan_sha256=sha256(path), source_sha256=digest,
            dispatcher_source=str(operation), operations_source=str(operation), operations_source_sha256=digest,
            https_proxy_file=str(Path(plan['shared_root'])/'private/company_https_proxy.url'),
            migration_revision=revision, migration_topology=topology,
            migration_parent_state_sha256=sha256(root/'state.json'),
            migration_stop_hashes=markers, migration_global_stop_sha256=sha256(root/'STOP'),
            migration_evidence=evidence, migration_prepared_utc=campaign.utc(),
            routing=dict(rule='Shared pending fixed-case queue across reviewed subscription slots',
                         preserve_running_containers=True, native_completed_cases_never_repeated=True),
            authorization=f'User authorized {topology} subscription routing and retirement of unreliable gateways; '
                          'keep all fixed cases/seeds, preserve active and paused official slots and old attempts.')
        new_plan['migration_submissions'] = [str(campaign.ensure_submission(new_plan, s))
                                            for s in new_state['slots'] if s['id'] in fresh_slots]
        new_plan['initial_submissions'] = list(new_plan['migration_submissions'])
        new_plan['adopted_submissions'] = [s.get('submission') for s in new_state['slots'] if s['id'] not in fresh_slots]
        new_plan['migration_state_payload_sha256'] = hashlib.sha256(
            json.dumps(new_state, sort_keys=True).encode()).hexdigest()
        write_json(target, new_plan)
        new_state['approved_sha256'] = sha256(target)
        write_json(root/f'state_{revision}_prepared.json', new_state)
        print(json.dumps(dict(plan=str(target), sha256=sha256(target), source_sha256=digest,
            submissions=new_plan['migration_submissions'], submitted=False), indent=2))


def activate(path, approved):
    path = path.resolve(); root = path.parent; plan = campaign.optional(path)
    revision, target_slots = migration_spec(plan.get('migration_topology', 'a3_b3'))
    if sha256(path) != approved or plan.get('migration_revision') != revision:
        raise ValueError('Migration review hash/revision mismatch')
    with _lock(plan):
        backup = root/f'state_before_{revision}.json'
        if backup.exists() or sha256(root/'state.json') != plan['migration_parent_state_sha256']:
            raise ValueError('Already activated or source state changed')
        if sha256(root/'STOP') != plan['migration_global_stop_sha256']:
            raise ValueError('Dispatcher stop intent changed')
        actual = {p.name: sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()}
        if actual != plan['migration_stop_hashes']:
            raise ValueError('Manual slot stop intent changed')
        state = campaign.optional(root/f'state_{revision}_prepared.json')
        if state.get('approved_sha256') != approved:
            raise ValueError('Prepared state mismatch')
        payload = dict(state, approved_sha256=plan['parent_plan_sha256'])
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan['migration_state_payload_sha256']:
            raise ValueError('Prepared migration state payload changed')
        if (campaign.validate_topology(plan) != tuple(target_slots.values())
                or len(state['slots']) != len(target_slots)
                or {s['id']: s['auth_profile'] for s in state['slots']} != target_slots):
            raise ValueError('Prepared topology differs from the reviewed migration')
        validate_campaign_sessions(plan['shared_root'], tuple(target_slots.values()))
        shutil.copy2(root/'state.json', backup)
        write_json(root/'state.json', state)
        for name in actual:
            (root/name).rename(root/f'{name}.before_{revision}.json')
        (root/'STOP').rename(root/f'STOP.before_{revision}.json')
        print('Six-subscription state activated. Start its frozen dispatcher once; no jobs created here.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('plan', type=Path)
    p.add_argument('--topology', choices=tuple(TOPOLOGIES), default='a3_b3')
    p = sub.add_parser('activate'); p.add_argument('plan', type=Path); p.add_argument('--approved-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.plan, args.topology)
    else: activate(args.plan, args.approved_sha256)


if __name__ == '__main__':
    main()
