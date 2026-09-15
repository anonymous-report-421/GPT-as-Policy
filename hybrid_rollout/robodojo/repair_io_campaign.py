"""Review-gated IO-only v3 repair; preserve running containers and raw attempts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import campaign
from .case_ledger import build_ledger
from .evaluation import read_panel
from .io import sha256, write_json
from .subscription_migration import _lock

REVISION = 'io_v3_1'
IO_FILE = 'hybrid_rollout/robodojo/io.py'


def repaired_state(state, *, retry_index, started_paths):
    """A temporary global STOP may mark reconciled/idle slots 'stopped'."""
    result = copy.deepcopy(state)
    paused = [s for s in result['slots'] if s.get('index') == retry_index
              and s['status'] == 'pause_unknown_failure']
    if len(paused) != 1:
        raise ValueError('Require exactly one paused IO case')
    retries = result.setdefault('pending_retries', [])
    if any(r['index'] == retry_index for r in retries):
        raise ValueError('Repair already queued')
    for slot in result['slots']:
        if slot['status'] != 'stopped':
            continue
        path = slot.get('submission')
        if path and path not in started_paths:
            raise ValueError('Reconcile unsubmitted prepared artifacts before repair')
        slot['status'] = 'submitted' if path else 'idle'
    slot = paused[0]
    # The review-only submission already fixes this one retry's auth profile.
    # Reserve it on its idle original slot, not in the free-slot retry queue:
    # another slot must not claim the prebuilt submission with different auth.
    slot.update(status='ready', attempt=slot['attempt']+1, next_retry_at=0,
        submission=None, job_id=None, platform_state=None,
        repair_reason='reviewed_startup_io_repair')
    return result


def prepare(path):
    path = path.resolve(); root = path.parent
    old = campaign.optional(path)
    if old.get('context_version') != 'v3' or not (root/'STOP').exists():
        raise ValueError('Pause only the v3 dispatcher first')
    if list(root.glob('STOP_SLOT_*')):
        raise ValueError('Explicit slot STOP needs separate reconciliation')
    target = root/f'campaign_{REVISION}.json'
    if target.exists():
        raise ValueError('Repair already prepared; do not overwrite')
    with _lock(old):
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != sha256(path):
            raise ValueError('Original plan/state mismatch')
        index = next(i for i,e in enumerate(old['queue'])
            if e['case']['case_id'] == 'make_kong__standard__g0__l0')
        slots = [s for s in state['slots'] if s.get('index') == index
                 and s['status'] == 'pause_unknown_failure']
        if len(slots) != 1:
            raise ValueError('Expected paused make_kong startup')
        slot = slots[0]; subpath = Path(slot['submission'])
        sub = campaign.optional(subpath); bp = Path(sub['batch_result'])
        batch = campaign.optional(bp)
        job = next(j for j in campaign.job_list(old) if j['name'] == slot['job_id'])
        if (job['state'] != 'FAILED' or batch.get('state') != 'failed'
                or 'cases.json.tmp' not in batch.get('error', '')
                or 'FileNotFoundError' not in batch.get('error', '')
                or not all(batch.get('final_reset', {}).get(k) is True
                    for k in ('ports_released', 'all_owned_processes_exited'))):
            raise ValueError('Not a verified terminal startup IO failure')
        archive = Path(batch['episodes'][0]['archive'])
        if archive.exists() or (bp.parent/'rollout_000.log').exists():
            raise ValueError('This repair is only for the pre-simulator failure')
        ledger = build_ledger(old['shared_root'], read_panel(old['eval_manifest']))
        if ledger['errors']:
            raise ValueError('Resolve ledger errors before repair')
        original = Path(old['dispatcher_source'])
        manifest = campaign.optional(original/'source_manifest.json')
        if manifest.get('source_sha256') != old['source_sha256'] or any(
                sha256(original/f) != digest for f,digest in manifest['files'].items()):
            raise ValueError('Original frozen runtime changed')
        frozen = root/f'source_{REVISION}'
        shutil.copytree(original/'hybrid_rollout', frozen/'hybrid_rollout')
        shutil.copy2(Path(__file__).with_name('io.py'), frozen/IO_FILE)
        files = {str(p.relative_to(frozen)): sha256(p) for p in frozen.rglob('*') if p.is_file()}
        changed = sorted(f for f in files if files[f] != manifest['files'].get(f))
        if set(files) != set(manifest['files']) or changed != [IO_FILE]:
            raise ValueError('Only io.py may change in the rollout package')
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        write_json(frozen/'source_manifest.json', dict(manifest, files=files, source_sha256=digest,
            parent_source_sha256=old['source_sha256'], changed_files=changed))
        new = copy.deepcopy(old)
        new.update(source_sha256=digest, dispatcher_source=str(frozen), operations_source_sha256=digest,
            io_repair=dict(revision=REVISION, original_plan=str(path), original_plan_sha256=sha256(path),
                original_state_sha256=sha256(root/'state.json'), stop_sha256=sha256(root/'STOP'),
                original_batch=str(bp), original_batch_sha256=sha256(bp), job_id=job['name'],
                retry_index=index, new_attempt=slot['attempt']+1, changed_files=changed,
                preserve_active_runtime=True, prompt_seed_settings_unchanged=True,
                paused_model_input_errors_unchanged=True, created_jobs=0))
        write_json(target, new)
        started = {s['submission'] for s in state['slots'] if s.get('submission')
            and Path(s['submission']).with_name('submission_started.json').exists()}
        prepared = repaired_state(state, retry_index=index, started_paths=started)
        prepared.update(approved_sha256=sha256(target), updated_utc=campaign.utc())
        write_json(root/f'state_{REVISION}_prepared.json', prepared)
        print(json.dumps(dict(plan=str(target), sha256=sha256(target), source_sha256=digest,
            retry_case=old['queue'][index]['case']['case_id'], attempt=slot['attempt']+1,
            prepared_only=True, created_jobs=0), indent=2))


def activate(path, approved):
    path = path.resolve(); root = path.parent; plan = campaign.optional(path)
    if sha256(path) != approved or plan.get('io_repair', {}).get('revision') != REVISION:
        raise ValueError('Explicit repair plan review required')
    repair = plan['io_repair']
    receipt = campaign.optional(root/f'authorization_{REVISION}.json')
    prepared_path = root/f'state_{REVISION}_prepared.json'
    if (receipt.get('approved_sha256') != approved or receipt.get('user_confirmed') is not True
            or receipt.get('prepared_state_sha256') != sha256(prepared_path)):
        raise ValueError('Missing user approval of exact repair/state')
    with _lock(plan):
        if (sha256(root/'state.json') != repair['original_state_sha256']
                or sha256(root/'STOP') != repair['stop_sha256']
                or sha256(repair['original_batch']) != repair['original_batch_sha256']):
            raise ValueError('State, stop intent or original evidence changed; re-review')
        backup = root/f'before_{REVISION}'
        backup.mkdir()  # Activation is one-shot, never overwrite recovery evidence.
        for name in ('state.json', 'launch_authorization.json', 'STOP'):
            shutil.copy2(root/name, backup/name)
        write_json(root/'state.json', campaign.optional(prepared_path))
        write_json(root/'launch_authorization.json', dict(approved_sha256=approved,
            accounts_confirmed=True, user_confirmation=receipt, previous_authorization=str(backup/'launch_authorization.json')))
        write_json(root/f'activation_{REVISION}.json', dict(activated_utc=campaign.utc(),
            plan=str(path), approved_sha256=approved, old_attempt_preserved=True))
        (root/'STOP').rename(backup/'STOP_released')
    print('Repair activated; start exactly one reviewed dispatcher. No jobs created here.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'activate'))
    p.add_argument('plan', type=Path)
    p.add_argument('--approved-sha256')
    args = p.parse_args()
    if args.action == 'prepare': prepare(args.plan)
    else: activate(args.plan, args.approved_sha256)


if __name__ == '__main__':
    main()
