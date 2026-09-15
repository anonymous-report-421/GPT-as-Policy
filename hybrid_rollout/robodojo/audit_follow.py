"""Incremental, model-free deep audit of the existing overnight 50+50 workflow.

Only this observer's reports are written. Worker archives, plans, prompts,
credentials, scheduler state and existing audit reports are never modified.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from .archive_audit import audit
from .campaign import optional, utc
from .io import sha256, write_json


def load_plans(workflow_path, approved):
    if sha256(workflow_path) != approved:
        raise ValueError('Workflow changed')
    workflow = optional(workflow_path)
    plans = {}
    for phase, method in ((1, 'pi05_plus_gpt'), (2, 'gpt_only')):
        path = Path(workflow[f'phase{phase}_plan'])
        if sha256(path) != workflow[f'phase{phase}_sha256']:
            raise ValueError('Frozen phase plan changed')
        plan = optional(path)
        if plan['evaluation_method'] != method or plan['context_version'] != 'v3':
            raise ValueError('Wrong experiment')
        if len(plan['primary_case_ids']) != 50 or len(set(plan['primary_case_ids'])) != 50:
            raise ValueError('Require fifty unique cases per method')
        plans[method] = plan
    if plans['pi05_plus_gpt']['primary_case_ids'] != plans['gpt_only']['primary_case_ids']:
        raise ValueError('Paired cases differ')
    return plans


def seed_process_live(pid, seed_report):
    """A stale PID or stale in-progress JSON must not strand unaudited cases."""
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    except PermissionError:
        return True  # Cannot establish death; do not duplicate active work.
    return (b'hybrid_rollout.robodojo.archive_audit' in argv
            and str(seed_report).encode() in argv)


def selected_rows(progress, plans):
    rows = []
    for key, method in (('hybrid', 'pi05_plus_gpt'), ('direct', 'gpt_only')):
        selection = progress.get(key)
        if selection is None:
            continue
        if selection.get('errors') or not selection.get('platform_verified'):
            raise ValueError('Selection is not verified: ' + key)
        seen = set()
        for row in selection['cases']:
            if (row['case_id'] not in plans[method]['primary_case_ids']
                    or row['case_id'] in seen or row['method'] != method
                    or row['context_version'] != 'v3'):
                raise ValueError('Unexpected or duplicate selected case')
            seen.add(row['case_id'])
            rows.append(row)
    return rows


def step(workflow_path, approved, output, seed_report, seed_pid):
    plans = load_plans(workflow_path, approved)
    progress = optional(workflow_path.parent/'progress.json')
    selected = selected_rows(progress, plans)
    report = optional(output) or dict(schema='robodojo.deep_audit_100.v1',
        workflow_sha256=approved, started_utc=utc(), rows={}, target=100)
    if report['workflow_sha256'] != approved:
        raise ValueError('Refuse to reuse another workflow audit')
    seed = optional(seed_report)
    reserved = set()
    if seed.get('status') == 'in_progress' and seed_process_live(seed_pid, seed_report):
        snapshot = Path(seed['snapshot'])
        if sha256(snapshot) != seed['snapshot_sha256']:
            raise ValueError('Initial audit snapshot changed')
        reserved = {str(Path(row['archive']).resolve())
                    for row in optional(snapshot)['completed']}
    imported = {row['archive']: row for row in seed.get('rows', [])}
    errors = []
    for case in selected:
        root = Path(case['archive']).resolve()
        if output.resolve().is_relative_to(root):
            raise ValueError('Report must be outside original archives')
        key = case['method'] + ':' + case['case_id']
        existing = report['rows'].get(key)
        if existing and (existing['archive'] != str(root)
                or existing['selected_manifest_sha256'] != case['artifact_manifest_sha256']):
            raise ValueError('Selected complete trajectory changed: ' + key)
        if sha256(root/'artifact_manifest.json') != case['artifact_manifest_sha256']:
            errors.append('Manifest changed: ' + key)
            continue
        if existing is None:
            row = imported.get(str(root))
            if row is None and str(root) in reserved:
                continue
            if row is None:
                try:
                    if case.get('evaluation_failure_reason'):
                        from .adjudicated_archive_audit import audit_idle
                        row = audit_idle(case, plans[case['method']])
                    else:
                        row = audit(root)
                except Exception as error:
                    row = dict(archive=str(root), passed=False, checked_utc=utc(),
                        error_type=type(error).__name__, error=str(error)[:800])
            existing = dict(row, case_id=case['case_id'], method=case['method'],
                selected_manifest_sha256=case['artifact_manifest_sha256'])
            report['rows'][key] = existing
            # Save each expensive audit immediately, including failed evidence.
            report.update(updated_utc=utc(), status='in_progress')
            write_json(output, report)
        if not existing.get('passed'):
            errors.append('Deep audit failed: ' + key)
        elif (existing.get('artifact_manifest_sha256') != case['artifact_manifest_sha256']
                or existing.get('context_version') != 'v3'
                or existing.get('evaluation_method') != case['method']
                or existing.get('native_success') is not case['native_success']
                or existing.get('adjudication_sha256') != case.get('adjudication_sha256')):
            errors.append('Deep audit identity mismatch: ' + key)
    passed = sum(row.get('passed') is True for row in report['rows'].values())
    # No success-rate threshold: native failures are complete scientific results.
    complete = (not errors and len(selected) == 100 and passed == 100
        and all(progress.get(key, {}).get('complete') for key in ('hybrid', 'direct'))
        and progress.get('state') == 'complete')
    report.update(updated_utc=utc(), selected_count=len(selected), passed_count=passed,
        native_complete_passed_count=sum(r.get('passed') is True and r.get('native_complete') is not False
            for r in report['rows'].values()),
        adjudicated_failure_passed_count=sum(r.get('passed') is True and r.get('native_complete') is False
            for r in report['rows'].values()),
        errors=errors, status='passed' if complete else ('hold_audit_errors' if errors else 'waiting'),
        progress_updated_utc=progress.get('updated_utc'), original_data_modified=False,
        model_calls=0, simulation_runs=0)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--approved-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed-report', type=Path, required=True)
    parser.add_argument('--seed-pid', type=int, required=True)
    parser.add_argument('--poll-seconds', type=int, default=300)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                report = step(args.workflow.resolve(), args.approved_sha256,
                    args.output, args.seed_report.resolve(), args.seed_pid)
                print(json.dumps({k: v for k, v in report.items() if k != 'rows'}), flush=True)
                if report['status'] == 'passed':
                    return
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(json.dumps(dict(utc=utc(), state='waiting_recheck',
                    error_type=type(error).__name__, error=str(error)[:800])), flush=True)
            time.sleep(max(1, args.poll_seconds))


if __name__ == '__main__':
    main()
