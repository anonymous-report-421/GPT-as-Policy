"""Durable, explicitly reviewed seven-slot ACP dispatcher (one case per container).

Only transient transport failures retry, in a NEW container with the same case
and a new attempt. Native failures are results, not reasons to retry. This
process runs on the submitting host, not inside any rollout container.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

from . import acp
from .case_ledger import build_ledger, refresh
from .codex_backend.profiles import validate_credential, validate_campaign_sessions, POOL15_PROFILES
from .evaluation import case_identity, read_panel
from .io import sha256, write_json

TASKS = ('organize_table', 'classify_objects_by_language', 'imitate_sorting_sequence',
         'arrange_largest_number', 'pack_objects_into_box', 'classify_objects')
SLOTS = ('galbot', 'galbot', 'koozhan', 'codex_b', 'codex_b_2', 'codex_b_3', 'codex_a')
SUBSCRIPTION_SLOTS = {0: 'codex_a_2', 2: 'codex_a_3', 3: 'codex_b',
                      4: 'codex_b_2', 5: 'codex_b_3', 6: 'codex_a'}
B5_A1_SLOTS = {0: 'codex_b_4', 2: 'codex_b_5', 3: 'codex_b',
               4: 'codex_b_2', 5: 'codex_b_3', 6: 'codex_a'}
B5_A2_SLOTS = {0: 'codex_b_4', 2: 'codex_b_5', 3: 'codex_b',
               4: 'codex_b_2', 5: 'codex_b_3', 7: 'codex_a_2'}
POOL15_SLOTS = dict(enumerate(POOL15_PROFILES))
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED', 'CANCELLED', 'CANCELED', 'DELETED'}
MANUAL_STOP = {'STOPPED', 'CANCELLED', 'CANCELED', 'DELETED', 'SUSPENDED', 'STOPPING'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def source_digest():
    root = Path(__file__).resolve().parents[2]
    files = {str(p.relative_to(root)): sha256(p)
             for p in sorted((root/'hybrid_rollout').rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def validate_topology(plan):
    slots = plan['slots']
    mapping = {s['id']: s['auth_profile'] for s in slots}
    if len(mapping) != len(slots) or mapping not in (
            dict(enumerate(SLOTS)), SUBSCRIPTION_SLOTS, B5_A1_SLOTS, B5_A2_SLOTS, POOL15_SLOTS):
        raise ValueError('Unsupported reviewed account topology')
    if mapping == POOL15_SLOTS and (plan.get('context_version') != 'v3'
            or plan.get('max_concurrent_gpus') != 30 or plan.get('dispatch_mode') != 'work_conserving'):
        raise ValueError('The 15-session topology requires an explicit v3 shared-queue/30-GPU plan')
    return tuple(mapping.values())


def transport_classification(text):
    """Only pass structured controller errors, never task/model/tool prose here."""
    text = text.lower()
    if re.search(r'insufficient[_ ]quota|quota.{0,25}exhaust|usage limit|credit|balance|billing|payment', text):
        return 'pause_quota'
    if re.search(r'\b(401|403)\b|unauthorized|invalid.{0,12}(key|token)|refresh.token|login|log in', text):
        return 'pause_auth'
    if re.search(r'\b(429|500|502|503|504)\b|rate.limit|too many requests|stream disconnected|'
                 r'connection (reset|closed|refused)|network|error sending request|'
                 r'transport|stream.{0,30}(timeout|timed out)|serveroverloaded|selected model is at capacity', text):
        return 'retry_transport'
    return 'pause_unknown_failure'


def assess(batch, archive, platform_state, expected):
    """Require both platform termination and immutable native outcome evidence."""
    if platform_state in MANUAL_STOP or batch.get('exit_code') == 130:
        return 'stopped'
    if platform_state not in TERMINAL:
        return 'wait'
    archive = Path(archive)
    outcome = optional(archive/'sim/evaluation_outcome.json')
    result = optional(archive/'controller/result.json')
    if outcome.get('complete') or result.get('complete'):
        # Never rerun a native terminal case just to repair an artifact problem.
        verified = (outcome.get('evaluation_case') == expected
            and result.get('evaluation_case') == expected
            and outcome.get('complete') is True and result.get('complete') is True
            and outcome.get('valid_for_success_rate') is True
            and type(outcome.get('native_success')) is bool
            and optional(archive/'artifact_manifest.json').get('status') == 'verified'
            and all(batch.get('final_reset', {}).get(k) is True
                    for k in ('all_owned_processes_exited', 'ports_released')))
        return 'completed' if verified else 'pause_terminal_artifacts'
    failure = optional(archive/'controller/failure.json')
    # These fields are written by the trusted runner, not an LLM's assessment.
    error = '\n'.join(str(failure.get(k, '')) for k in ('error', 'exception', 'message', 'traceback'))
    if 'Codex made no completed service or native tool call' in error:
        rpc = archive/'controller/codex_workspace/rpc_out.jsonl'
        if rpc.exists():
            with rpc.open('rb') as stream:
                stream.seek(max(0, rpc.stat().st_size - 2 * 1024 * 1024))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
            last_error = None
            for line in lines:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if message.get('method') == 'error':
                    last_error = message.get('params', {}).get('error')
            if last_error:
                error += '\n'+json.dumps(last_error)
    return transport_classification(error)


def retry_delay(number):
    return min(3600, 300 * 2 ** min(max(number - 1, 0), 4))


def prepare(args):
    if args.quota_type not in ('spot', 'reserved'):
        raise ValueError('Explicit quota type required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.experiment_prefix):
        raise ValueError('Invalid campaign identifier')
    shared = args.shared_root.resolve()
    if not shared.is_relative_to('/mnt/rollout'):
        raise ValueError('Campaign logs and state must use the requested shared storage')
    panel = read_panel(args.eval_manifest)
    scope = optional(args.scope)
    if scope.get('source_panel_sha256') != panel['panel_sha256']:
        raise ValueError('Scope/panel identity mismatch')
    ledger = build_ledger(shared, panel)
    if ledger['errors']:
        raise ValueError('Reconcile ledger errors before preparing a campaign')
    rows = {r['case_id']: r for r in ledger['cases']}
    by_id = {c['case_id']: c for c in panel['cases']}
    groups = {t['task']: t['case_ids'] for t in scope['tasks'] if t['task'] in TASKS}
    if set(groups) != set(TASKS) or any(len(v) != 5 for v in groups.values()):
        raise ValueError('Require the six reviewed low-SR tasks, five frozen cases each')
    queue, skipped = [], []
    for i in range(5):
        for task in TASKS:
            cid = groups[task][i]
            row = rows[cid]
            if row['state'] == 'completed':
                skipped.append(cid)
                continue
            if row['state'] in ('running_or_unreconciled', 'duplicate_completed_review_required'):
                raise ValueError(f'Reconcile existing attempts first: {cid}')
            attempt = 1 + max((int(a['attempt']) for a in row['attempts']), default=-1)
            queue.append(dict(case=by_id[cid], initial_attempt=attempt))
    root = shared/'campaigns'/args.experiment_prefix
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='hybrid_rollout.robodojo.campaign.v1', created_utc=utc(),
        experiment_prefix=args.experiment_prefix, shared_root=str(shared),
        eval_manifest=str(args.eval_manifest.resolve()), panel_sha256=panel['panel_sha256'],
        scope_file=str(args.scope.resolve()), scope_file_sha256=sha256(args.scope),
        source_sha256=source_digest(), skipped_completed=skipped, queue=queue,
        slots=[dict(id=i, auth_profile=p) for i, p in enumerate(SLOTS)],
        poll_seconds=300, retry_base_seconds=300, retry_max_seconds=3600,
        retry_limit=None, retry_only='transient_controller_transport_failure',
        model='gpt-6-astra', effort='xhigh', fast=False, max_decisions=0, max_seconds=0,
        gpus_per_replica=2, max_concurrent_gpus=14,
        codex=str(args.codex.resolve()), sco=str(args.sco.resolve()), image=args.image,
        workspace=args.workspace, cluster=args.cluster, worker_spec=args.worker_spec, quota=args.quota_type,
        authorization='Prepared only. Run requires user review of this plan and its SHA256. '
                      'Retries preserve case/source/backend and create a new attempt. '
                      'Manual stop disables the slot; no cross-backend fallback.')
    preview_state = dict(slots=[dict(s, status='idle', assigned=[]) for s in plan['slots']])
    previews = []
    for slot in preview_state['slots']:
        reserve(preview_state, slot, plan)
        if slot['status'] != 'drained':
            previews.append(str(ensure_submission(plan, slot, review_only=True)))
    plan['initial_submissions'] = previews
    plan['dispatcher_source'] = optional(previews[0])['source_snapshot'] if previews else None
    write_json(root/'campaign.json', plan)
    print(json.dumps(dict(plan=str(root/'campaign.json'), sha256=sha256(root/'campaign.json'),
                         pending=len(queue), skipped_completed=skipped, submitted=False), indent=2))


def job_list(plan):
    from .acp_query import user_jobs
    return user_jobs(plan['sco'], plan['workspace'])


def reserve(state, slot, plan):
    if plan.get('dispatch_mode') == 'work_conserving':
        from .elastic_dispatch import reserve as elastic_reserve
        return elastic_reserve(state, slot, plan)
    used = {i for s in state['slots'] for i in s.get('assigned', [])}
    index = next((i for i, entry in enumerate(plan['queue']) if i not in used
                  and entry.get('slot_id', slot['id']) == slot['id']), None)
    if index is None:
        slot['status'] = 'drained'
        return
    slot.setdefault('assigned', []).append(index)
    slot.update(index=index, attempt=plan['queue'][index]['initial_attempt'],
                status='ready', retry_number=0, next_retry_at=0, submission=None,
                job_id=None, platform_state=None)


def submission_path(plan, slot):
    prefix = plan['queue'][slot['index']].get('experiment_prefix', plan['experiment_prefix'])
    experiment = f"{prefix}_c{slot['index']}_a{slot['attempt']}"
    replica = plan['queue'][slot['index']]['case']['replica_id']
    return Path(plan['shared_root'])/'cluster'/experiment/f'replica_{replica}'/'submission.json'


def ensure_submission(plan, slot, review_only=False):
    # A prompt cutover pins already-launched CASES, including their retries,
    # to their old immutable source. A later case on the same slot uses v2.
    case_override = plan.get('case_runtime_overrides', {}).get(str(slot['index']))
    if case_override:
        effective = dict(plan, **case_override)
        effective.pop('case_runtime_overrides', None)
        return ensure_submission(effective, slot, review_only=review_only)
    # Explicit reviewed repair for selected identities. Other slots keep their
    # original frozen runtime, even on subsequent automatic transport retries.
    override = plan.get('slot_runtime_overrides', {}).get(str(slot['id']))
    if override:
        effective = dict(plan, **override)
        effective.pop('slot_runtime_overrides', None)
        return ensure_submission(effective, slot, review_only=review_only)
    path = submission_path(plan, slot)
    if path.exists():
        submission = optional(path)
        if submission.get('evaluation_method', 'pi05_plus_gpt') != plan.get('evaluation_method', 'pi05_plus_gpt'):
            raise ValueError('Existing submission belongs to another evaluation method')
        if (submission['source_sha256'] != plan['source_sha256']
                or submission['cluster'] != plan['cluster']
                or submission['workspace'] != plan['workspace']
                or submission['quota'] != plan.get('quota', 'spot')
                or submission['worker_spec'] != plan['worker_spec']
                or submission['environment']['ROLLOUT_AUTH_PROFILE'] != slot['auth_profile']
                or submission['environment'].get('ROLLOUT_HTTPS_PROXY') != plan.get('https_proxy')
                or submission['environment'].get('ROLLOUT_HTTPS_PROXY_FILE') != plan.get('https_proxy_file')
                or ('context_version' in plan and submission.get('context_version', 'v1') != plan['context_version'])
                or ((submission.get('rerun_of') or {}).get('archive') !=
                    slot.get('rerun_v1_archive', plan['queue'][slot['index']].get('rerun_v1_archive')))
                or submission['evaluation_cases'] != [plan['queue'][slot['index']]['case']]):
            raise ValueError('Existing submission differs from the reviewed campaign')
        return path
    c = plan['queue'][slot['index']]['case']
    args = argparse.Namespace(experiment_id=path.parents[1].name, task=c['task'],
        replica_id=c['replica_id'], count=1, attempt=slot['attempt'], case_id=c['case_id'],
        eval_manifest=Path(plan['eval_manifest']), max_decisions=0, max_seconds=0,
        codex_image_max_edge=480, auth_profile=slot['auth_profile'], credential_file=None,
        codex=Path(plan['codex']), sco=Path(plan['sco']), image=plan['image'],
        workspace=plan['workspace'], cluster=plan['cluster'], worker_spec=plan['worker_spec'],
        quota_type=plan.get('quota', 'spot'),
        shared_root=Path(plan['shared_root']), review_without_credentials=review_only)
    if 'evaluation_method' in plan:
        args.evaluation_method = plan['evaluation_method']
    if plan.get('phase1_certificate'):
        args.phase1_certificate = Path(plan['phase1_certificate'])
    origin = slot.get('rerun_v1_archive', plan['queue'][slot['index']].get('rerun_v1_archive'))
    if origin:
        args.rerun_v1_archive = Path(origin)
    if plan.get('https_proxy'):
        args.https_proxy = plan['https_proxy']
    if plan.get('https_proxy_file'):
        args.https_proxy_file = plan['https_proxy_file']
    if plan.get('dispatcher_source') and source_digest() != plan['source_sha256']:
        # Operational dispatcher fixes must not change the rollout implementation.
        # Future attempts are still prepared by the ORIGINAL frozen rollout code.
        argv = [os.sys.executable, '-m', 'hybrid_rollout.robodojo.acp', 'prepare']
        for key, value in vars(args).items():
            if key in ('credential_file', 'review_without_credentials'):
                continue
            argv.extend(['--'+key.replace('_', '-'), str(value)])
        subprocess.run(argv, cwd=plan['dispatcher_source'],
            env=dict(os.environ, PYTHONPATH=plan['dispatcher_source'], PYTHONDONTWRITEBYTECODE='1'),
            check=True, timeout=300)
    else:
        acp.prepare(args)
    if optional(path)['source_sha256'] != plan['source_sha256']:
        raise ValueError('Source changed while preparing; submission was NOT sent')
    return path


def tick(plan, state, directory, jobs, *, only_slots=None, max_new_submissions=None):
    """One synchronous poll. Durable intent always precedes external create."""
    def save():
        state['updated_utc'] = utc()
        write_json(directory/'state.json', state)

    new_submissions = 0
    slots = state['slots']
    if max_new_submissions is not None:
        # A per-poll submission cap must not starve higher-numbered sessions
        # when earlier sessions keep finishing or retrying. Rotate only the
        # traversal, never the persisted slot identities or case queue.
        previous = state.get('submission_round_robin_after_slot')
        start = next((i + 1 for i, s in enumerate(slots) if s['id'] == previous), 0)
        slots = slots[start:] + slots[:start]
    for slot in slots:
        if only_slots is not None and slot['id'] not in only_slots:
            continue
        if (directory/'STOP').exists() or (directory/f"STOP_SLOT_{slot['id']}").exists():
            slot['status'] = 'stopped'
            save()
            continue  # Deliberately do not kill any active container.
        if slot['status'] in ('stopped', 'drained') or slot['status'].startswith('pause'):
            continue
        if (slot['status'] in ('idle', 'ready') and slot['auth_profile'] in
                state.get('quota_guard', {}).get('blocked_profiles', [])):
            if slot['status'] == 'idle' and plan.get('dispatch_mode') == 'work_conserving':
                used = {i for s in state['slots'] for i in s.get('assigned', [])}
                skipped = state.get('skipped_indices', {})
                if not state.get('pending_retries') and all(
                        i in used or str(i) in skipped for i in range(len(plan['queue']))):
                    slot['status'] = 'drained'
                    save()
            continue  # Reconcile running jobs, but preserve quota before creating more.
        if slot['status'] == 'idle':
            reserve(state, slot, plan)
            save()
        if slot['status'] == 'drained':
            continue
        if slot['status'] == 'ready' and time.time() >= slot['next_retry_at']:
            if max_new_submissions is not None and new_submissions >= max_new_submissions:
                continue
            new_submissions += 1
            if max_new_submissions is not None:
                # Persist before preparation/create so a process restart does
                # not give the same low-numbered slot priority again.
                state['submission_round_robin_after_slot'] = slot['id']
                save()
            path = ensure_submission(plan, slot)
            slot.update(submission=str(path), status='submitting')
            save()
            # An ambiguous create is never re-issued. Recover it via display name.
            if not (path.parent/'submission_started.json').exists():
                if (directory/'STOP').exists() or (directory/f"STOP_SLOT_{slot['id']}").exists():
                    slot['status'] = 'stopped'
                    save()
                    continue
                try:
                    acp.submit(path, cancelled=lambda: (directory/'STOP').exists()
                        or (directory/f"STOP_SLOT_{slot['id']}").exists())
                except InterruptedError:
                    slot['status'] = 'stopped'
                    save()
                    continue
                except SystemExit:
                    pass
                except subprocess.SubprocessError:
                    # A create timeout is ambiguous; intent stays persisted and
                    # the next poll resolves it rather than sending it again.
                    if not (path.parent/'submission_started.json').exists():
                        slot.update(status='ready', next_retry_at=time.time() + 300)
                        save()
                        continue
            slot['status'] = 'submitted'
            save()
            continue
        if slot['status'] not in ('submitted', 'submitting', 'running'):
            continue
        path = Path(slot['submission'])
        submission = optional(path)
        expected_name = (submission['environment']['ROLLOUT_EXPERIMENT_ID']+'_r'
                         + submission['environment']['ROLLOUT_REPLICA_ID'])
        matches = [j for j in jobs if j.get('display_name') == expected_name
                   and j.get('ownership', {}).get('user_name') == 'sujiayi']
        if len(matches) != 1:
            slot.update(status='pause_ambiguous_submission', reason='Expected exactly one ACP job; never re-create blindly')
            save()
            continue
        job = matches[0]
        slot.update(job_id=job['name'], platform_state=job['state'], status='running')
        batch = optional(submission['batch_result'])
        case = plan['queue'][slot['index']]['case']
        from .evaluation import archive_path
        archive = archive_path(Path(plan['shared_root']), submission['environment']['ROLLOUT_EXPERIMENT_ID'],
                               case, case['replica_id'], slot['attempt'])
        action = assess(batch, archive, job['state'], case_identity(read_panel(plan['eval_manifest']), case))
        if action == 'retry_transport':
            from .idle_timeout_policy import adjudicate
            idle_failure = adjudicate(plan, batch, archive, job,
                case_identity(read_panel(plan['eval_manifest']), case))
            if idle_failure:
                action = 'failed_rpc_idle_timeout'
                state.setdefault('adjudicated_failures', {})[str(slot['index'])] = dict(
                    archive=str(archive), attempt=slot['attempt'], job_id=job['name'], reason=idle_failure['reason'])
                state.setdefault('skipped_indices', {})[str(slot['index'])] = action
        if action == 'pause_unknown_failure' and not (archive/'controller/run.json').exists():
            bootstrap = path.parent/'bootstrap.log'
            if bootstrap.exists():
                # This exact trusted startup probe executes before any model.
                tail = bootstrap.read_text(errors='replace')[-16384:]
                if ('hybrid_rollout/robodojo/proxy_config.py' in tail and
                        'RuntimeError: connection_failed (connection details withheld)' in tail):
                    action = 'retry_transport'
        slot['last_action'] = action
        if action == 'wait':
            save()
            continue
        evidence = dict(schema='hybrid_rollout.robodojo.scheduler_reconciliation.v1',
            checked_utc=utc(), job_id=job['name'], platform_state=job['state'],
            batch_path=submission['batch_result'], case_id=case['case_id'], action=action,
            old_container_terminal=job['state'] in TERMINAL)
        # Sidecar only: do not rewrite old batches, observations, videos or errors.
        if not (path.parent/'scheduler_reconciliation.json').exists():
            write_json(path.parent/'scheduler_reconciliation.json', evidence)
        elif action == 'retry_transport' and optional(path.parent/'scheduler_reconciliation.json').get('action') != action:
            # Preserve the previous diagnosis; append a corrected disposition.
            corrected = path.parent/f"scheduler_reconciliation.retry_{slot['attempt']}.json"
            if not corrected.exists():
                write_json(corrected, evidence)
        refresh(plan['shared_root'], plan['eval_manifest'], method=plan.get('evaluation_method', 'pi05_plus_gpt'))
        slot.setdefault('history', []).append(dict(index=slot['index'], attempt=slot['attempt'],
            submission=str(path), job_id=job['name'], action=action, finished_utc=utc()))
        if action in ('completed', 'failed_rpc_idle_timeout'):
            slot['status'] = 'idle'
        elif action == 'retry_transport':
            slot['retry_number'] += 1
            slot['attempt'] += 1
            slot.update(status='ready', submission=None,
                        next_retry_at=time.time() + retry_delay(slot['retry_number']))
        else:
            slot['status'] = action
        save()


def polling_interval(plan, override=None):
    """Operations-only cadence; never modify the reviewed worker plan."""
    if override is None:
        return plan['poll_seconds']
    if not 30 <= override <= plan['poll_seconds']:
        raise ValueError('Poll override must be between 30 seconds and the reviewed interval')
    return override


def run(args):
    if sha256(args.plan) != args.approved_sha256:
        raise ValueError('Review hash mismatch; no task was submitted')
    plan = optional(args.plan)
    poll_seconds = polling_interval(plan, getattr(args, 'poll_seconds', None))
    if plan.get('preparation_revision') == 'context_v3_pool15':
        from .prepare_v3 import verify_ready
        verify_ready(args.plan, args.approved_sha256)
    if plan.get('evaluation_method') == 'gpt_only':
        from .phase_gate import verify_certificate
        verify_certificate(plan.get('phase1_certificate'), read_panel(plan['eval_manifest'], plan['panel_sha256']))
    quota_guard = None
    if plan.get('preparation_revision') == 'context_v3_pool15':
        from .pool15_quota import Pool15QuotaGuard
        quota_guard = Pool15QuotaGuard(plan, getattr(args, 'quota_policy', None),
                                      getattr(args, 'quota_policy_sha256', None))
    elif getattr(args, 'quota_policy', None):
        from .quota_guard import QuotaGuard
        quota_guard = QuotaGuard(args.quota_policy, args.quota_policy_sha256, plan)
    dispatcher_sha = (getattr(args, 'dispatcher_source_sha256', None)
                      or plan.get('operations_source_sha256') or plan['source_sha256'])
    directory = args.plan.resolve().parent
    profiles = validate_topology(plan)
    for name in set(profiles):
        validate_credential(name, plan['shared_root'])
    validate_campaign_sessions(plan['shared_root'], tuple(n for n in profiles if n.startswith('codex_')))
    state = optional(directory/'state.json') or dict(schema='hybrid_rollout.robodojo.campaign_state.v1',
        approved_sha256=args.approved_sha256, slots=[dict(s, status='idle', assigned=[]) for s in plan['slots']])
    if state['approved_sha256'] != args.approved_sha256:
        raise ValueError('Existing campaign belongs to a different plan')
    if {s['id']: s['auth_profile'] for s in state['slots']} != {s['id']: s['auth_profile'] for s in plan['slots']}:
        raise ValueError('State slots differ from reviewed topology')
    # One dispatcher for the shared panel, even across campaign IDs.
    with (Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            write_json(directory/'STOP', dict(reason='Scheduler interrupted; do not auto-restart', utc=utc()))
            raise KeyboardInterrupt
        old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        quota_next_check = 0.0
        try:
            while not (directory/'STOP').exists():
                external_stop = getattr(quota_guard, 'external', {}).get('stop_file')
                if external_stop and Path(external_stop).exists():
                    return
                if source_digest() != dispatcher_sha:
                    raise ValueError('Dispatcher source changed; pause and re-review before continuing')
                if sha256(plan['scope_file']) != plan['scope_file_sha256']:
                    raise ValueError('Frozen scope changed')
                read_panel(plan['eval_manifest'], plan['panel_sha256'])
                try:
                    jobs = job_list(plan)
                except (subprocess.SubprocessError, ValueError):
                    print(json.dumps(dict(utc=utc(), event='platform_query_failed_no_submission')), flush=True)
                else:
                    other = [j for j in jobs if j.get('ownership', {}).get('user_name') == 'sujiayi'
                        and j.get('display_name', '').startswith('robodojo_')
                        and not j.get('display_name', '').startswith(plan['experiment_prefix']+'_')
                        and j.get('state') not in TERMINAL | {'SUSPENDED'}]
                    if other:
                        raise ValueError('Another RoboDojo campaign is active; reconcile before adding concurrency')
                    # Only this dispatcher process; never modify the user shell environment.
                    for name in ('NO_PROXY', 'no_proxy'):
                        os.environ[name] = 'management.sensecoreapi.cn,aec2.cn-sh-01.sensecoreapi.cn'
                    if quota_guard is not None and time.monotonic() >= quota_next_check:
                        quota_guard.tick(state, directory, jobs)
                        # Faster slot polling must not multiply account RPC traffic.
                        quota_next_check = time.monotonic() + plan['poll_seconds']
                    try:
                        if getattr(args, 'recover_verified_guards', False):
                            if not getattr(quota_guard, 'external', {}).get('retry_verified_input_guards'):
                                raise ValueError('Guard recovery requires unattended authorization')
                            from .unattended_recovery import release_deferred
                            release_deferred(plan, state, directory, jobs)
                        if plan.get('dispatch_mode') == 'work_conserving':
                            from .elastic_dispatch import tick as elastic_tick
                            elastic_tick(plan, state, directory, jobs,
                                retry_input_guards_once=getattr(args, 'retry_input_guards_once', False),
                                max_new_submissions=1 if getattr(quota_guard, 'external', {}) else None)
                        else:
                            tick(plan, state, directory, jobs)
                    except (subprocess.SubprocessError, OSError) as error:
                        # Preparation/query transport failure is not evidence of
                        # a failed create. Retain durable intent and reconcile.
                        print(json.dumps(dict(utc=utc(), event='dispatch_io_error_retry_next_poll',
                                              error_type=type(error).__name__)), flush=True)
                    print(json.dumps(dict(utc=utc(), states=dict(Counter(s['status'] for s in state['slots'])))), flush=True)
                    if all(s['status'] == 'drained' for s in state['slots']):
                        return
                for _ in range(poll_seconds):
                    if (directory/'STOP').exists() or (external_stop and Path(external_stop).exists()):
                        return
                    time.sleep(1)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('eval-manifest', 'scope', 'codex', 'sco', 'shared-root'):
        prepare_parser.add_argument('--'+name, type=Path, required=True)
    for name in ('experiment-prefix', 'image', 'workspace', 'cluster'):
        prepare_parser.add_argument('--'+name, required=True)
    prepare_parser.add_argument('--quota-type', choices=('spot', 'reserved'), required=True)
    prepare_parser.add_argument('--worker-spec', required=True, help='Verified two-GPU single-node SKU')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('plan', type=Path)
    run_parser.add_argument('--approved-sha256', required=True)
    run_parser.add_argument('--dispatcher-source-sha256',
        help='Explicitly reviewed operations-only source; rollout source remains frozen in the plan')
    run_parser.add_argument('--poll-seconds', type=int,
        help='Operations-only faster refill interval (minimum 30 seconds); account checks keep their original cadence')
    run_parser.add_argument('--retry-input-guards-once', action='store_true',
        help='Release verified terminal input-guard slots; one same-seed retry per case, then defer for analysis')
    run_parser.add_argument('--quota-policy', type=Path)
    run_parser.add_argument('--recover-verified-guards', action='store_true',
        help='Authorized recovery of verified incomplete guard attempts, never native failures')
    run_parser.add_argument('--quota-policy-sha256')
    stop_parser = sub.add_parser('stop')
    stop_parser.add_argument('plan', type=Path)
    stop_parser.add_argument('--slot', type=int, choices=range(8))
    rebalance_parser = sub.add_parser('rebalance')
    rebalance_parser.add_argument('plan', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'run':
        run(args)
    elif args.action == 'rebalance':
        rebalance(args.plan)
    else:
        name = 'STOP' if args.slot is None else f'STOP_SLOT_{args.slot}'
        write_json(args.plan.resolve().parent/name, dict(utc=utc(), reason='User requested no new jobs or retries'))
        print('Dispatch/retries disabled; existing containers are not killed by this command.')


def rebalance(path):
    """User-approved routing revision; adopt only the already launched first case."""
    plan = optional(path)
    root = path.resolve().parent
    if (root/'state.json').exists() or (root/'STOP').exists():
        raise ValueError('Reconcile existing dispatcher state before changing routing')
    for p in plan['initial_submissions'][1:]:
        if Path(p).with_name('submission_started.json').exists():
            raise ValueError('Cannot reassign a case whose submission already started')
    first = optional(plan['initial_submissions'][0])
    if first['evaluation_cases'][0]['case_id'] != plan['queue'][0]['case']['case_id']:
        raise ValueError('Unexpected first-case identity')
    if plan['queue'][0]['case']['task'] != 'organize_table':
        raise ValueError('This revision must preserve the running organize_table case')
    target = root/'campaign_stability_v2.json'
    if target.exists():
        raise ValueError('Routing revision already exists; do not overwrite or reprepare')
    limits = {'organize_table': 1000, 'arrange_largest_number': 1050,
              'classify_objects': 1100, 'classify_objects_by_language': 1100,
              'pack_objects_into_box': 1300, 'imitate_sorting_sequence': 1600}
    loads = {i: 0 for i in range(7)}
    for i, entry in enumerate(plan['queue']):
        task = entry['case']['task']
        native = Path(plan['shared_root'])/'src/RoboDojo/task/RoboDojo/tasks'/f"{entry['case']['runtime_task']}.py"
        match = re.search(r'self\.step_lim\s*=\s*(\d+)', native.read_text())
        if not match or int(match[1]) != limits[task]:
            raise ValueError('Native horizon differs from the reviewed routing rule')
        candidates = ([0] if i == 0 else [2] if task == 'organize_table'
                      else [0, 1] if task == 'arrange_largest_number' else [3, 4, 5, 6])
        slot = min(candidates, key=lambda s: (loads[s], s))
        loads[slot] += limits[task]
        entry.update(slot_id=slot, native_step_limit=limits[task], native_limit_source_sha256=sha256(native))
        if i:
            entry['experiment_prefix'] = plan['experiment_prefix']+'_stable'
    plan['routing'] = dict(rule='User stability ranking; native horizon proxy; fixed slots, no retry migration',
        stability=['codex_a=codex_b', 'galbot', 'koozhan'], total_native_horizon_by_slot=loads,
        adopted_first_submission=plan['initial_submissions'][0])
    plan['parent_plan_sha256'] = sha256(path)
    plan['operations_source_sha256'] = source_digest()
    operation_root = root/'dispatcher_source_v2'
    shutil.copytree(Path(__file__).resolve().parents[2]/'hybrid_rollout', operation_root/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    plan['operations_source'] = str(operation_root)
    state = dict(slots=[dict(s, assigned=[]) for s in plan['slots']])
    previews = []
    for slot in state['slots']:
        reserve(state, slot, plan)
        previews.append(str(ensure_submission(plan, slot)))
    plan['initial_submissions'] = previews
    write_json(target, plan)
    print(json.dumps(dict(plan=str(target), sha256=sha256(target), operations_source=str(operation_root),
                         slot_loads=loads, submitted=False), indent=2))


if __name__ == '__main__':
    main()
