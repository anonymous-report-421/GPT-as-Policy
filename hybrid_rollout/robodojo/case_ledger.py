"""Rebuild a durable case-level index from immutable panel and persisted attempts.

Never infer completion from scheduler exit status. An unfinished batch remains
unreconciled even if its timestamp is old; cluster state must be checked before retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from .evaluation import case_identity, read_panel
from .io import sha256, write_json
from .settings import BASE_URL, EFFORT, MODEL, PROVIDER, WIRE_API


def read_optional(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def build_ledger(shared, panel, preferred_task=None, *, method='pi05_plus_gpt'):
    from .method import evaluation_method
    method = evaluation_method(method)
    shared = Path(shared)
    identities = {c['case_id']: case_identity(panel, c) for c in panel['cases']}
    cases = {c['case_id']: dict(identities[c['case_id']], replica_id=c['replica_id'],
             rollout_index=c['rollout_index'], attempts=[]) for c in panel['cases']}
    issues = []
    reconciled = {}
    for path in sorted((shared/'cluster').glob('*/replica_*/scheduler_reconciliation*.json')):
        try:
            record = read_optional(path)
            plan = read_optional(path.with_name('submission.json'))
            if (record.get('schema') == 'hybrid_rollout.robodojo.scheduler_reconciliation.v1'
                    and record.get('old_container_terminal') is True
                    and (record.get('platform_state') in ('FAILED', 'SUCCEEDED')
                         or (record.get('action') == 'user_authorized_quota_recovery'
                             and record.get('platform_state') in ('STOPPED', 'CANCELLED', 'CANCELED', 'SUSPENDED')))
                    and record.get('action') in ('retry_transport', 'user_authorized_proxy_repair',
                                                'user_authorized_quota_recovery')
                    and record.get('job_id', '').startswith('pt-')
                    and record.get('batch_path') == plan.get('batch_result')
                    and [c['case_id'] for c in plan.get('evaluation_cases', [])] == [record.get('case_id')]
                    and plan.get('evaluation_panel_sha256') == panel['panel_sha256']):
                reconciled[record['batch_path']] = record
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(dict(path=str(path), error=str(error)))
    for path in sorted((shared/'results').glob('*/_replicas/replica_*/attempt_*/batch.json')):
        try:
            batch = read_optional(path)
            if batch.get('panel_sha256') != panel['panel_sha256']:
                continue
            if batch.get('evaluation_method', 'pi05_plus_gpt') != method:
                continue
            for episode in batch.get('episodes', []):
                row = cases[episode['case_id']]
                expected = identities[row['case_id']]
                if any(episode.get(k) != v for k, v in expected.items()):
                    raise ValueError('Attempt case identity mismatch')
                archive = Path(episode['archive'])
                if not archive.resolve().is_relative_to((shared/'results').resolve()):
                    raise ValueError('Attempt archive outside results root')
                result = read_optional(archive/'controller/result.json')
                outcome = read_optional(archive/'sim/evaluation_outcome.json')
                progress = read_optional(archive/'controller/progress.json')
                reset = episode.get('reset') or {}
                complete = (episode.get('state') == 'completed'
                    and episode.get('artifact_status') == 'verified'
                    and reset.get('all_owned_processes_exited') is True
                    and reset.get('ports_released') is True
                    and result.get('evaluation_case') == expected
                    and outcome.get('evaluation_case') == expected
                    and result.get('complete') is True
                    and outcome.get('complete') is True
                    and outcome.get('valid_for_success_rate') is True
                    and type(outcome.get('native_success')) is bool)
                if complete:
                    state = 'completed'
                elif episode.get('state') == 'completed':
                    state = 'invalid_or_unverified'
                elif batch.get('state') == 'running' and str(path) not in reconciled:
                    state = 'running_or_unreconciled'
                else:
                    state = 'incomplete'
                row['attempts'].append(dict(state=state, attempt=batch.get('attempt'),
                    experiment=path.parents[3].name, batch_path=str(path), archive=str(archive),
                    provider=batch.get('provider'), model=batch.get('model'), effort=batch.get('effort'),
                    launch_identity=read_optional(archive/'launch_manifest.json').get('backend'),
                    context_version=read_optional(archive/'controller/run.json').get(
                        'context_version', batch.get('context_version', 'v1')),
                    rerun_of=batch.get('rerun_of'),
                    teacher_prompt_sha256=read_optional(archive/'controller/run.json').get('teacher_prompt_sha256'),
                    native_success=outcome.get('native_success') if complete else None,
                    native_status=outcome.get('status'), reason=outcome.get('reason'),
                    step_id=result.get('step_id', progress.get('step_id', 0)),
                    started_utc=episode.get('started_utc'), finished_utc=episode.get('finished_utc'),
                    exit_code=episode.get('exit_code'), artifact_status=episode.get('artifact_status'),
                    scheduler_reconciliation=reconciled.get(str(path))))
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(dict(path=str(path), error=str(error)))
    # A submit may have reached ACP before the container or its first batch row
    # exists. Do not treat such an ambiguous submission as permission to retry.
    for path in sorted((shared/'cluster').glob('*/replica_*/submission.json')):
        if not (path.parent/'submission_started.json').exists():
            continue
        try:
            plan = read_optional(path)
            if plan.get('evaluation_panel_sha256') != panel['panel_sha256']:
                continue
            if plan.get('evaluation_method', 'pi05_plus_gpt') != method:
                continue
            batch_path = str(Path(plan['batch_result']))
            for case in plan['evaluation_cases']:
                row = cases[case['case_id']]
                # Only the first selected case is reserved before a sequential
                # job starts. Later unstarted cases in a terminal batch stay pending.
                if row['case_id'] != plan['evaluation_cases'][0]['case_id']:
                    continue
                if any(a.get('batch_path') == batch_path for a in row['attempts']):
                    continue
                if Path(batch_path).exists() and read_optional(Path(batch_path)).get('state') in ('failed', 'completed'):
                    continue
                row['attempts'].append(dict(state='incomplete' if batch_path in reconciled else 'running_or_unreconciled',
                    batch_path=batch_path, submission_path=str(path),
                    submission_response_path=str(path.parent/'submission_response.json'),
                    model=plan.get('model'), effort=plan.get('effort'),
                    context_version=plan.get('context_version', 'v1'),
                    rerun_of=plan.get('rerun_of'),
                    attempt=plan['environment'].get('ROLLOUT_ATTEMPT'),
                    reason='Submission attempted; batch evidence pending; reconcile scheduler before retry'))
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(dict(path=str(path), error=str(error)))
    for row in cases.values():
        states = [a['state'] for a in row['attempts']]
        completed = [a for a in row['attempts'] if a['state'] == 'completed']
        row['completed_by_context'] = dict(Counter(a['context_version'] for a in completed))
        row['paired_context_rerun'] = valid_context_pair(completed)
        if states.count('completed') > 1 and not row['paired_context_rerun']:
            row['state'] = 'duplicate_completed_review_required'
        elif 'completed' in states:
            row['state'] = 'completed'
        elif 'running_or_unreconciled' in states:
            row['state'] = 'running_or_unreconciled'
        elif states:
            row['state'] = 'needs_review_before_retry'
        else:
            row['state'] = 'not_started'
    ordered = sorted(cases.values(), key=lambda c: (c['task'] != preferred_task, c['replica_id'], c['rollout_index']))
    remaining = [r for r in ordered if r['state'] != 'completed']
    return dict(schema='hybrid_rollout.robodojo.case_ledger.v1',
        evaluation_method=method,
        updated_utc=datetime.now(timezone.utc).isoformat(), panel_id=panel['panel_id'],
        panel_sha256=panel['panel_sha256'], errors=issues,
        status_counts=dict(Counter(c['state'] for c in cases.values())),
        next_case_id=remaining[0]['case_id'] if remaining and not issues else None,
        next_is_submission_authorization=False,
        proposed_backend=dict(provider=PROVIDER, base_url=BASE_URL, model=MODEL,
                              effort=EFFORT, wire_api=WIRE_API, source='current_configuration_not_run_evidence'),
        note='Preserve all attempts. Backend changes are recorded separately, not a homogeneous success-rate estimate. '
             'Running/stale/ambiguous records require cluster reconciliation; never auto-retry.',
        cases=list(cases.values()))


def refresh(shared, manifest, preferred_task='classify_objects_by_language', *, method='pi05_plus_gpt'):
    shared = Path(shared).resolve()
    panel = read_panel(manifest)
    suffix = '' if method == 'pi05_plus_gpt' else '_'+method
    directory = shared/'evaluation'/f'{panel["panel_id"]}{suffix}_progress'
    directory.mkdir(parents=True, exist_ok=True)
    # All writers use the same lock and atomic JSON replacement.
    with (directory/'ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = build_ledger(shared, panel, preferred_task, method=method)
        write_json(directory/'cases.json', ledger)
    return directory/'cases.json', ledger


def valid_context_pair(completed):
    """An authorized v1 failure + one linked v2 result is not an accidental duplicate."""
    if len(completed) != 2:
        return False
    old = [a for a in completed if a.get('context_version', 'v1') == 'v1'
           and a.get('native_success') is False]
    new = [a for a in completed if a.get('context_version') == 'v2']
    if len(old) != 1 or len(new) != 1 or not new[0].get('rerun_of'):
        return False
    ref = new[0]['rerun_of']
    archive = Path(old[0]['archive'])
    try:
        outcome = read_optional(archive/'sim/evaluation_outcome.json')
        identity = outcome['evaluation_case']
        return (ref.get('schema') == 'robodojo.v1_failure_v2_rerun.v1'
            and ref.get('archive') == str(archive) and ref.get('from_context') == 'v1'
            and ref.get('to_context') == 'v2' and ref.get('case_id') == identity['case_id']
            and ref.get('panel_sha256') == identity['panel_sha256']
            and ref.get('result_sha256') == sha256(archive/'controller/result.json')
            and ref.get('outcome_sha256') == sha256(archive/'sim/evaluation_outcome.json'))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def make_rerun_reference(shared, panel, case_id, archive):
    """Narrow permission: exactly one verified native v1 failure, never a success."""
    archive = Path(archive).resolve()
    if not archive.is_relative_to(Path(shared).resolve()/'results'):
        raise ValueError('Rerun origin must be inside shared results')
    ledger = build_ledger(shared, panel)
    if ledger['errors']:
        raise ValueError('Resolve ledger errors before a context rerun')
    row = next(r for r in ledger['cases'] if r['case_id'] == case_id)
    old = [a for a in row['attempts'] if a.get('archive') == str(archive)
           and a['state'] == 'completed' and a.get('context_version', 'v1') == 'v1'
           and a.get('native_success') is False]
    if len(old) != 1:
        raise ValueError('Rerun requires a verified native v1 failure')
    return dict(schema='robodojo.v1_failure_v2_rerun.v1', case_id=case_id,
        panel_sha256=panel['panel_sha256'], from_context='v1', to_context='v2', archive=str(archive),
        result_sha256=sha256(archive/'controller/result.json'),
        outcome_sha256=sha256(archive/'sim/evaluation_outcome.json'))


def require_available(shared, panel, case_id, own_submission_batch_path=None, *, rerun_of=None,
                      method='pi05_plus_gpt'):
    if rerun_of is not None and method != 'pi05_plus_gpt':
        raise ValueError('Context-pair exemptions apply only to hybrid evaluation')
    ledger = build_ledger(shared, panel, method=method)
    if ledger['errors']:
        raise ValueError('Case ledger has unreadable or inconsistent evidence; reconcile before launch')
    row = next(c for c in ledger['cases'] if c['case_id'] == case_id)
    attempts = [a for a in row['attempts'] if not (
        own_submission_batch_path is not None and a.get('submission_path')
        and a.get('batch_path') == str(own_submission_batch_path))]
    if rerun_of is not None:
        from .prompt_context import CONTEXT_VERSION
        if CONTEXT_VERSION != 'v2' or rerun_of != make_rerun_reference(
                shared, panel, case_id, rerun_of.get('archive', '')):
            raise ValueError('Context rerun reference changed or is not v2')
        # Only the explicitly named old native failure is exempted. Any new
        # completed result (including v2 failure) or uncertain submission blocks.
        attempts = [a for a in attempts if not (a['state'] == 'completed'
                    and a.get('archive') == rerun_of['archive'])]
    if any(a['state'] in ('completed', 'running_or_unreconciled') for a in attempts):
        raise ValueError(f'Case must not be repeated: {case_id}: {row["state"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path, required=True)
    parser.add_argument('--eval-manifest', type=Path, required=True)
    args = parser.parse_args()
    path, ledger = refresh(args.shared_root, args.eval_manifest)
    print(json.dumps(dict(ledger=str(path), status_counts=ledger['status_counts'],
                         next_case_id=ledger['next_case_id'], errors=ledger['errors']), indent=2))
