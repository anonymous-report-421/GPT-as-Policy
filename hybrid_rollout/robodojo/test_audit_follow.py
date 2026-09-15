import copy
import json

import pytest

from . import audit_follow as subject
from .io import sha256


def fixture(tmp_path, monkeypatch, count=1):
    workflow = tmp_path/'workflow.json'
    workflow.write_text('{}')
    ids = [f'case_{i}' for i in range(50)]
    plans = {method: {'primary_case_ids': ids}
             for method in ('pi05_plus_gpt', 'gpt_only')}
    monkeypatch.setattr(subject, 'load_plans', lambda *_: plans)
    selected = []
    for i in range(count):
        root = tmp_path/f'archive_{i}'
        root.mkdir()
        (root/'artifact_manifest.json').write_text('{}')
        method = 'pi05_plus_gpt' if i < 50 else 'gpt_only'
        selected.append(dict(archive=str(root), case_id=ids[i % 50], method=method,
            context_version='v3', native_success=False,
            artifact_manifest_sha256=sha256(root/'artifact_manifest.json')))
    progress = dict(state='complete' if count == 100 else 'running_hybrid50')
    for key, method in (('hybrid', 'pi05_plus_gpt'), ('direct', 'gpt_only')):
        rows = [r for r in selected if r['method'] == method]
        progress[key] = dict(cases=rows, errors=[], platform_verified=True, complete=len(rows) == 50)
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    calls = []

    def audit(root):
        calls.append(str(root))
        case = next(c for c in selected if c['archive'] == str(root))
        return dict(archive=str(root), passed=True, context_version='v3',
            evaluation_method=case['method'], native_success=False,
            artifact_manifest_sha256=case['artifact_manifest_sha256'])

    monkeypatch.setattr(subject, 'audit', audit)
    (tmp_path/'follow').mkdir()
    return dict(workflow_path=workflow, approved='approved', output=tmp_path/'follow/audit.json',
        seed_report=tmp_path/'initial.json', seed_pid=123), selected, calls, progress


def test_native_failure_audited_once_and_never_claims_partial_complete(tmp_path, monkeypatch):
    args, cases, calls, _ = fixture(tmp_path, monkeypatch)
    assert subject.step(**args)['status'] == 'waiting'
    result = subject.step(**args)
    assert result['passed_count'] == 1 and len(calls) == 1
    assert not next(iter(result['rows'].values()))['native_success']


def test_100_native_failures_are_complete_not_a_bug_verdict(tmp_path, monkeypatch):
    args, _, calls, _ = fixture(tmp_path, monkeypatch, 100)
    report = subject.step(**args)
    assert report['status'] == 'passed' and report['passed_count'] == 100
    assert len(calls) == 100


def test_active_initial_audit_reserved_then_dead_process_recovered(tmp_path, monkeypatch):
    args, cases, calls, _ = fixture(tmp_path, monkeypatch)
    snapshot = tmp_path/'initial_snapshot.json'
    snapshot.write_text(json.dumps(dict(completed=cases)))
    args['seed_report'].write_text(json.dumps(dict(status='in_progress', rows=[],
        snapshot=str(snapshot), snapshot_sha256=sha256(snapshot))))
    monkeypatch.setattr(subject, 'seed_process_live', lambda *_: True)
    assert subject.step(**args)['passed_count'] == 0 and calls == []
    monkeypatch.setattr(subject, 'seed_process_live', lambda *_: False)
    assert subject.step(**args)['passed_count'] == 1 and len(calls) == 1


def test_import_existing_audit_without_redecoding(tmp_path, monkeypatch):
    args, cases, calls, _ = fixture(tmp_path, monkeypatch)
    row = subject.audit(tmp_path/'archive_0')
    calls.clear()
    args['seed_report'].write_text(json.dumps(dict(status='passed', rows=[row])))
    assert subject.step(**args)['passed_count'] == 1 and calls == []


def test_changed_manifest_blocks_completion(tmp_path, monkeypatch):
    args, _, calls, _ = fixture(tmp_path, monkeypatch)
    subject.step(**args)
    (tmp_path/'archive_0/artifact_manifest.json').write_text('{"changed": true}')
    assert subject.step(**args)['status'] == 'hold_audit_errors'
    assert len(calls) == 1


def test_failed_audit_preserved_without_unbounded_automatic_rechecks(tmp_path, monkeypatch):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    def fail(_):
        raise ValueError('missing observation')
    monkeypatch.setattr(subject, 'audit', fail)
    assert subject.step(**args)['status'] == 'hold_audit_errors'
    assert subject.step(**args)['passed_count'] == 0


def test_reject_duplicate_and_unverified_cases(tmp_path, monkeypatch):
    args, _, _, progress = fixture(tmp_path, monkeypatch)
    plans = subject.load_plans(None, None)
    duplicate = copy.deepcopy(progress)
    duplicate['hybrid']['cases'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        subject.selected_rows(duplicate, plans)
    progress['hybrid']['platform_verified'] = False
    with pytest.raises(ValueError, match='not verified'):
        subject.selected_rows(progress, plans)


def test_real_workflow_hash_is_required(tmp_path):
    path = tmp_path/'workflow.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='Workflow changed'):
        subject.load_plans(path, 'wrong')
