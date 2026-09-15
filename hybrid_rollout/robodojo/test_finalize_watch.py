import hashlib
import json
from pathlib import Path

import pytest

from . import finalize_watch as watch
from .io import sha256, write_json


def fixture(tmp_path, monkeypatch):
    wf = tmp_path/'workflow.json'; wf.write_text('{}')
    root = tmp_path/'finisher'; root.mkdir()
    config = dict(root=str(root), workflow=str(wf), workflow_sha256=sha256(wf),
        source=str(tmp_path/'source'), source_sha256='source', deep_audit=str(tmp_path/'deep.json'),
        output=str(tmp_path/'delivery'))
    rows = [dict(method=method, case_id=str(i), archive=f'/archive/{method}/{i}',
                 artifact_manifest_sha256=f'{method}-{i}', native_success=False)
        for method in ('pi05_plus_gpt', 'gpt_only') for i in range(50)]
    progress = dict(state='complete', hybrid=dict(complete=True), direct=dict(complete=True))
    deep = dict(workflow_sha256=config['workflow_sha256'], status='passed',
        selected_count=100, passed_count=100, errors=[], rows={r['method']+':'+r['case_id']:
            dict(passed=True, archive=r['archive'], selected_manifest_sha256=r['artifact_manifest_sha256'])
            for r in rows})
    write_json(tmp_path/'progress.json', progress); write_json(tmp_path/'deep.json', deep)
    monkeypatch.setattr(watch, 'load_plans', lambda *a: {})
    monkeypatch.setattr(watch, 'selected_rows', lambda *a: rows)
    monkeypatch.setattr(watch, 'verified_source', lambda *a: {})
    return config, rows, progress, deep


def test_ready_includes_all_native_failures_without_score_threshold(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    assert watch.readiness(config)['ready'] is True


@pytest.mark.parametrize('change', ['partial', 'audit_error', 'audit_running', 'wrong_workflow', 'workflow_running'])
def test_waits_for_full_matching_finished_audit(tmp_path, monkeypatch, change):
    config, rows, progress, deep = fixture(tmp_path, monkeypatch)
    if change == 'partial': rows.pop()
    if change == 'audit_error': deep['errors'] = ['bad archive']
    if change == 'audit_running': deep['status'] = 'waiting'
    if change == 'wrong_workflow': deep['workflow_sha256'] = 'wrong'
    if change == 'workflow_running': progress['state'] = 'running'
    write_json(tmp_path/'progress.json', progress); write_json(tmp_path/'deep.json', deep)
    assert watch.readiness(config)['ready'] is False


def test_rejects_audit_of_other_archive(tmp_path, monkeypatch):
    config, _, _, deep = fixture(tmp_path, monkeypatch)
    deep['rows']['gpt_only:0']['archive'] = '/wrong'
    write_json(tmp_path/'deep.json', deep)
    with pytest.raises(ValueError, match='does not cover'): watch.readiness(config)


def test_finalization_snapshots_then_gates_export_and_scores(tmp_path, monkeypatch):
    from . import paired_initial_audit, final_debug_export, score_table
    config, _, progress, _ = fixture(tmp_path, monkeypatch)
    snap = watch.readiness(config)
    write_json(tmp_path/'progress.json', {'state': 'subsequent_refresh'})
    calls = []
    monkeypatch.setattr(watch, 'derived_videos', lambda rows: calls.append('derived'))
    def paired(wf, approved):
        assert json.loads((wf.parent/'progress.json').read_text()) == progress
        assert sha256(wf) == approved
        calls.append('paired'); return dict(status='complete_50_pairs', pairs=50, errors=[])
    monkeypatch.setattr(paired_initial_audit, 'audit_selected', paired)
    def export(wf, approved, deep, paired, output):
        calls.append('videos'); output.mkdir()
        write_json(output/'results.json', {}); write_json(output/'COMPLETE.json', {})
        return dict(videos=100)
    monkeypatch.setattr(final_debug_export, 'export', export)
    def scores(source, target):
        calls.append('scores'); target.mkdir(); write_json(target/'manifest.json', {})
        return dict(rows=11)
    monkeypatch.setattr(score_table, 'export', scores)
    result = watch.finalize(config, snap)
    assert calls == ['derived', 'paired', 'videos', 'scores']
    assert result['state'] == 'complete' and result['model_calls'] == 0
    assert (Path(config['root'])/'FINALIZED.json').exists()
    with pytest.raises(ValueError, match='Previous finalization'): watch.finalize(config, snap)


def test_bad_pair_records_evidence_without_export(tmp_path, monkeypatch):
    from . import paired_initial_audit, final_debug_export
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(watch, 'derived_videos', lambda rows: None)
    monkeypatch.setattr(paired_initial_audit, 'audit_selected', lambda *a:
        dict(status='review_required', pairs=50, errors=['different seed']))
    monkeypatch.setattr(final_debug_export, 'export', lambda *a: pytest.fail('Must not export bad pairs'))
    with pytest.raises(ValueError, match='did not pass'): watch.finalize(config, watch.readiness(config))
    assert (Path(config['root'])/'paired_initial_final.json').exists()
    assert not (Path(config['root'])/'FINALIZED.json').exists()
    assert not Path(config['output']).exists()


def test_existing_delivery_is_preserved(tmp_path, monkeypatch):
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    Path(config['output']).mkdir()
    with pytest.raises(ValueError, match='Preserve existing delivery'):
        watch.finalize(config, watch.readiness(config))
    assert not (Path(config['root'])/'attempt.json').exists()


def test_correct_native_video_does_not_render_again(tmp_path, monkeypatch):
    root = tmp_path/'archive'; (root/'controller/debug_video').mkdir(parents=True)
    write_json(root/'controller/result.json', dict(complete=True, terminated=True, success=False, truncated=False))
    write_json(root/'controller/debug_video/manifest.json', dict(terminal_label_version='native_terminal_labels_v2'))
    monkeypatch.setattr(watch.subprocess, 'run', lambda *a, **kw: pytest.fail('Unnecessary render'))
    watch.derived_videos([dict(archive=str(root))])


def test_idle_renders_only_missing_derived_directory(tmp_path, monkeypatch):
    from . import adjudicated_archive_audit
    root = tmp_path/'archive'; (root/'controller').mkdir(parents=True)
    monkeypatch.setattr(adjudicated_archive_audit, 'adjudicated', lambda row: {})
    calls = []
    monkeypatch.setattr(watch.subprocess, 'run', lambda args, **kwargs: calls.append(args))
    row = dict(archive=str(root), evaluation_failure_reason='simulator_rpc_idle_timeout_900s')
    watch.derived_videos([row])
    assert len(calls) == 1 and calls[0][-1] == str(root/'controller/debug_video_idle_timeout_v1')
    (root/'controller/debug_video_idle_timeout_v1').mkdir()
    watch.derived_videos([row]); assert len(calls) == 1


def test_prepare_freezes_only_finalizer_files_and_preserves_source(tmp_path, monkeypatch):
    parent = tmp_path/'parent'; (parent/'hybrid_rollout/robodojo').mkdir(parents=True)
    original = parent/'hybrid_rollout/robodojo/unchanged.py'; original.write_text('preserved = True\n')
    files = {'hybrid_rollout/robodojo/unchanged.py': sha256(original)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    write_json(parent/'source_manifest.json', dict(files=files, source_sha256=digest))
    wf = tmp_path/'workflow.json'; wf.write_text('{}')
    monkeypatch.setattr(watch, 'load_plans', lambda *a: {'gpt_only': {'shared_root': str(tmp_path)}})
    receipt = watch.prepare(wf, sha256(wf), parent, digest, tmp_path/'watcher',
        tmp_path/'deep.json', tmp_path/'reports/final')
    config = watch.config_read(receipt['config'], receipt['config_sha256'])
    assert config['rollout_mutations'] is False and config['poll_seconds'] == 300
    assert sha256(original) == files['hybrid_rollout/robodojo/unchanged.py']
    assert watch.verified_source(Path(config['source']), config['source_sha256'])['finalization_only'] is True
    assert not Path(config['output']).exists()
    with pytest.raises(ValueError, match='Never overwrite'):
        watch.prepare(wf, sha256(wf), parent, digest, tmp_path/'watcher',
            tmp_path/'deep.json', tmp_path/'reports/final')
    frozen = Path(config['source'])/'hybrid_rollout/robodojo/finalize_watch.py'
    frozen.write_text('tampered')
    with pytest.raises(ValueError, match='Frozen source changed'):
        watch.config_read(receipt['config'], receipt['config_sha256'])


def test_launch_starts_only_own_session_and_prevents_replay(tmp_path, monkeypatch):
    from types import SimpleNamespace
    config, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(watch, 'config_read', lambda *a: config)
    calls = []
    def subprocess_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1 if 'has-session' in args else 0)
    monkeypatch.setattr(watch.subprocess, 'run', subprocess_run)
    path = Path(config['root'])/'config.json'
    watch.launch(path, 'config-sha')
    assert len(calls) == 2 and 'new-session' in calls[1]
    assert 'finalize-100' in calls[1] and 'hybrid_rollout.robodojo.finalize_watch run' in calls[1][-1]
    assert not any(x in calls[1][-1] for x in ['campaign run', 'overnight run', 'reset'])
    assert (Path(config['root'])/'launch.json').exists()
    with pytest.raises(ValueError, match='Launch already attempted'):
        watch.launch(path, 'config-sha')
