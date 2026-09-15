"""Explicit one-time repair of the four paused subscription slots; never kills jobs.

Prepare while the dispatcher is stopped (STOP file), review generated submissions,
then activate with their plan hash. Other slots and all historical evidence stay
unchanged. This is not an automatic retry of arbitrary configuration failures.
"""
import argparse
import copy
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import campaign
from .codex_backend.profiles import account_home, account_lock, config_text, validate_campaign_sessions
from .codex_backend.validate import validate_home_config
from .io import sha256, write_json


def repaired_state(state, *, proxy_repair=False):
    result = copy.deepcopy(state)
    for slot in result['slots']:
        if slot['id'] not in (3, 4, 5, 6):
            continue
        allowed = ('stopped', 'submitted', 'pause_unknown_failure') if proxy_repair else ('pause_unknown_failure',)
        if slot['status'] not in allowed:
            raise ValueError('Only the four reviewed paused configuration failures may be repaired')
        slot.setdefault('history', []).append(dict(index=slot['index'], attempt=slot['attempt'],
            submission=slot['submission'], job_id=slot['job_id'],
            action='user_authorized_proxy_repair' if proxy_repair else 'user_authorized_config_repair',
            finished_utc=campaign.utc()))
        slot.update(attempt=slot['attempt']+1, status='ready', submission=None,
                    next_retry_at=0, job_id=None, platform_state=None,
                    last_action='user_authorized_config_repair')
    return result


def prepare(path, proxy=None, proxy_file=None):
    root = path.resolve().parent
    revision = 'accounts_v4' if proxy_file else 'accounts_v3'
    target = root/f'campaign_{revision}.json'
    if not (root/'STOP').exists() or target.exists():
        raise ValueError('Require stopped dispatcher and a fresh repair plan')
    plan = campaign.optional(path)
    lock_path = Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock'
    with lock_path.open('a') as lock, ExitStack() as sessions:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != sha256(path):
            raise ValueError('State/parent plan identity mismatch')
        jobs = {j['name']: j for j in campaign.job_list(plan)}
        validate_campaign_sessions(plan['shared_root'])
        for slot in state['slots'][3:]:
            submission = campaign.optional(slot['submission'])
            expected_name = submission['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'+submission['environment']['ROLLOUT_REPLICA_ID']
            matching = [j for j in jobs.values() if j.get('display_name') == expected_name
                        and j.get('ownership', {}).get('user_name') == 'sujiayi']
            if len(matching) != 1:
                raise ValueError('Need exactly one old submission; never assume a missing job is stopped')
            job = matching[0]
            slot['job_id'] = job['name']
            if job.get('state') != 'FAILED' or job.get('ownership', {}).get('user_name') != 'sujiayi':
                raise ValueError('Each repaired job must still be terminal FAILED and owned by this user')
            sessions.enter_context(account_lock(slot['auth_profile'], plan['shared_root']))
            batch_file = Path(submission['batch_result'])
            batch = campaign.optional(batch_file)
            if proxy_file:
                bootstrap = Path(slot['submission']).parent/'bootstrap.log'
                stop = root/f"STOP_SLOT_{slot['id']}"
                if (batch_file.exists() or not bootstrap.exists()
                        or '407 Proxy Authentication Required' not in bootstrap.read_text()
                        or not campaign.optional(stop).get('reason', '').startswith('Repair preflight received proxy407')):
                    raise ValueError('Require exact proxy407 pre-batch failure and its temporary stop marker')
                sidecar = Path(slot['submission']).parent/'scheduler_reconciliation.json'
                if sidecar.exists():
                    raise ValueError('Existing reconciliation requires review; do not overwrite')
                write_json(sidecar, dict(schema='hybrid_rollout.robodojo.scheduler_reconciliation.v1',
                    checked_utc=campaign.utc(), job_id=job['name'], platform_state=job['state'],
                    batch_path=str(batch_file), case_id=submission['evaluation_cases'][0]['case_id'],
                    action='user_authorized_proxy_repair', old_container_terminal=True,
                    reason='User authorized authenticated company proxy after pre-batch407; no rollout components launched'))
            elif (batch.get('exit_code') != 2 or not all(batch.get('final_reset', {}).get(k) is True
                    for k in ('all_owned_processes_exited', 'ports_released'))
                    or (batch_file.parent/'rollout_000.log').read_text().strip() !=
                    'Isolated CODEX_HOME/config.toml differs from the validated rollout config'):
                raise ValueError('Repair evidence differs from the reviewed startup failure')
            if any(e.get('evaluation_outcome') or e.get('result') for e in batch.get('episodes', [])):
                raise ValueError('Never restart a terminal native result for this repair')
        new_state = repaired_state(state, proxy_repair=bool(proxy_file))
        operation_root = root/('repair_source_v4' if proxy_file else 'repair_source_v3')
        repo = Path(__file__).resolve().parents[2]
        shutil.copytree(repo/'hybrid_rollout', operation_root/'hybrid_rollout',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        files = {str(p.relative_to(operation_root)): sha256(p)
                 for p in sorted(operation_root.rglob('*')) if p.is_file()}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(operation_root/'source_manifest.json', dict(git_base_commit=commit,
            source_sha256=digest, includes_reviewed_worktree_changes=True, files=files))
        plan.update(parent_plan_sha256=sha256(path), operations_source=str(operation_root),
                    operations_source_sha256=digest, repair_prepared_utc=campaign.utc(),
                    repair_revision=revision,
                    repair_parent_state_sha256=sha256(root/'state.json'))
        proxy_settings = dict(https_proxy_file=str(Path(proxy_file).resolve())) if proxy_file else dict(https_proxy=proxy)
        plan['slot_runtime_overrides'] = {str(i): dict(source_sha256=digest,
            dispatcher_source=str(operation_root), **proxy_settings) for i in (3, 4, 5, 6)}
        if proxy_file:
            plan['repair_stop_hashes'] = {str(i): sha256(root/f'STOP_SLOT_{i}') for i in (3,4,5,6)}
        submissions = []
        for slot in new_state['slots'][3:]:
            # Fresh expected settings, no rewriting/copying managed credentials.
            expected = operation_root/f"expected_{slot['auth_profile']}.toml"
            expected.write_text(config_text(slot['auth_profile']))
            validate_home_config(expected, account_home(slot['auth_profile'], plan['shared_root'])/'config.toml',
                                 slot['auth_profile'], plan['shared_root'])
            submissions.append(str(campaign.ensure_submission(plan, slot)))
        plan['repair_submissions'] = submissions
        write_json(target, plan)
        new_state['approved_sha256'] = sha256(target)
        write_json(root/f'state_{revision}_prepared.json', new_state)
        print(json.dumps(dict(plan=str(target), sha256=sha256(target), source_sha256=digest,
                              submissions=submissions, submitted=False), indent=2))


def activate(path, approved):
    if sha256(path) != approved:
        raise ValueError('Review hash mismatch')
    root = path.resolve().parent
    plan = campaign.optional(path)
    revision = plan.get('repair_revision', 'accounts_v3')
    if revision not in ('accounts_v3', 'accounts_v4'):
        raise ValueError('Unknown repair revision')
    with (Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root/'STOP').exists() or (root/f'state_before_{revision}.json').exists():
            raise ValueError('Dispatcher must be stopped and repair not already activated')
        if sha256(root/'state.json') != plan['repair_parent_state_sha256']:
            raise ValueError('State changed after review; do not overwrite it')
        prepared = campaign.optional(root/f'state_{revision}_prepared.json')
        if prepared['approved_sha256'] != approved:
            raise ValueError('Prepared state hash mismatch')
        for i, digest in plan.get('repair_stop_hashes', {}).items():
            if sha256(root/f'STOP_SLOT_{i}') != digest:
                raise ValueError('Slot stop intent changed since repair was reviewed')
        shutil.copy2(root/'state.json', root/f'state_before_{revision}.json')
        write_json(root/'state.json', prepared)
        for i in plan.get('repair_stop_hashes', {}):
            (root/f'STOP_SLOT_{i}').rename(root/f'STOP_SLOT_{i}.before_{revision}.json')
        (root/'STOP').rename(root/f'STOP.before_{revision}.json')
        print('Repair state activated; start the reviewed dispatcher once. No jobs created by activate.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare'); a.add_argument('plan', type=Path)
    group = a.add_mutually_exclusive_group(required=True)
    group.add_argument('--https-proxy'); group.add_argument('--https-proxy-file', type=Path)
    a = sub.add_parser('activate'); a.add_argument('plan', type=Path); a.add_argument('--approved-sha256', required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.plan, args.https_proxy, args.https_proxy_file)
    else:
        activate(args.plan, args.approved_sha256)


if __name__ == '__main__':
    main()
