import copy
import json

import pytest

from .final_debug_export import verified_video, verify_full_audits, LABEL_VERSION
from . import final_debug_export as subject
from .io import sha256


def fixture(tmp_path, *, early=True, corrected=True):
    root = tmp_path/'episode'; base = root/'controller/debug_video'; base.mkdir(parents=True)
    (base/'debug_rollout.mp4').write_bytes(b'original-video')
    result = dict(complete=True, success=not early, terminated=True, truncated=False, step_id=2)
    (root/'controller/result.json').write_text(json.dumps(result))
    manifest = dict(status='completed', video_sha256=sha256(base/'debug_rollout.mp4'),
        frames=3, fps=25., frame_mapping='exact timeline', episode_result=result,
        duration_seconds=.12, input_sha256={'original': 'hash'})
    (base/'manifest.json').write_text(json.dumps(manifest))
    if corrected:
        revised = root/'controller/debug_video_labels_v2'; revised.mkdir()
        (revised/'debug_rollout.mp4').write_bytes(b'corrected-video')
        (revised/'manifest.json').write_text(json.dumps(dict(manifest,
            video_sha256=sha256(revised/'debug_rollout.mp4'), terminal_label_version=LABEL_VERSION)))
    immutable = [dict(path=str(p.relative_to(root)), sha256=sha256(p))
                 for p in (base/'debug_rollout.mp4', base/'manifest.json', root/'controller/result.json')]
    (root/'artifact_manifest.json').write_text(json.dumps(dict(status='verified', all_immutable_artifacts=immutable)))
    return dict(archive=str(root), artifact_manifest_sha256=sha256(root/'artifact_manifest.json'),
        case_id='case', native_success=not early), root


def test_prefers_correction_preserving_original_hash_and_timeline(tmp_path):
    row, root = fixture(tmp_path)
    before = sha256(root/'artifact_manifest.json')
    result = verified_video(row)
    assert result['corrected'] and 'debug_video_labels_v2' in result['source']
    assert result['sha256'] != result['original_sha256'] and result['frames'] == 3
    assert sha256(root/'artifact_manifest.json') == before


def test_normal_native_success_can_use_original(tmp_path):
    row, _ = fixture(tmp_path, early=False, corrected=False)
    assert not verified_video(row)['corrected']


def test_early_failure_cannot_export_old_incomplete_badge(tmp_path):
    row, _ = fixture(tmp_path, corrected=False)
    with pytest.raises(ValueError, match='label-corrected'):
        verified_video(row)


@pytest.mark.parametrize('field,value', [('frames', 4), ('fps', 24.), ('frame_mapping', 'changed'),
    ('input_sha256', {'different': 'hash'}), ('episode_result', {})])
def test_changed_corrected_timeline_or_inputs_fail(tmp_path, field, value):
    row, root = fixture(tmp_path)
    path = root/'controller/debug_video_labels_v2/manifest.json'
    data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='timing or episode inputs'):
        verified_video(row)


def test_corrupt_correction_no_silent_fallback(tmp_path):
    row, root = fixture(tmp_path)
    (root/'controller/debug_video_labels_v2/debug_rollout.mp4').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Corrected video changed'):
        verified_video(row)


def full_fixture():
    rows = [dict(case_id=str(i), method=m, archive='/'+m+'/'+str(i), artifact_manifest_sha256='hash')
        for m in ('pi05_plus_gpt', 'gpt_only') for i in range(50)]
    deep = dict(passed_count=100, selected_count=100, errors=[], rows={
        r['method']+':'+r['case_id']: dict(passed=True, archive=r['archive'], selected_manifest_sha256='hash') for r in rows})
    pairs = [dict(case_id=str(i), settings_and_robot_text_match=True,
        hybrid_archive='/pi05_plus_gpt/'+str(i), gpt_only_archive='/gpt_only/'+str(i),
        hybrid_artifact_manifest_sha256='hash', gpt_only_artifact_manifest_sha256='hash') for i in range(50)]
    return rows, deep, dict(status='complete_50_pairs', pairs=50, rows=pairs, errors=[])


def test_all_100_audits_required():
    rows, deep, pairs = full_fixture()
    verify_full_audits(rows, deep, pairs)
    with pytest.raises(ValueError, match='exactly fifty'):
        verify_full_audits(rows[:-1], deep, pairs)
    deep['passed_count'] = 99
    with pytest.raises(ValueError, match='deep audit'):
        verify_full_audits(rows, deep, pairs)


def test_audit_identity_and_pair_completeness_not_just_green_count():
    rows, deep, pairs = full_fixture()
    wrong = copy.deepcopy(deep); wrong['rows']['gpt_only:0']['archive'] = '/another'
    with pytest.raises(ValueError, match='does not cover'):
        verify_full_audits(rows, wrong, pairs)
    pairs['rows'][0]['settings_and_robot_text_match'] = False
    with pytest.raises(ValueError, match='Pair audit'):
        verify_full_audits(rows, deep, pairs)


def test_full_export_zip_and_score_files_without_touching_sources(tmp_path, monkeypatch):
    import zipfile
    rows, deep, pairs = full_fixture()
    for row in rows:
        row.update(evaluation_case=dict(case_id=row['case_id']), native_success=False, native_score=0.)
    workflow = tmp_path/'workflow.json'; workflow.write_text('{}')
    (tmp_path/'progress.json').write_text(json.dumps(dict(hybrid={}, direct={})))
    deep_path, pair_path = tmp_path/'deep.json', tmp_path/'pairs.json'
    deep_path.write_text(json.dumps(deep)); pair_path.write_text(json.dumps(pairs))
    video = tmp_path/'source.mp4'; video.write_bytes(b'video-stream')
    original_sha = sha256(video)
    monkeypatch.setattr(subject, 'load_plans', lambda *_: {'a': dict(shared_root=str(tmp_path/'shared'))})
    monkeypatch.setattr(subject, 'selected_rows', lambda *_: rows)
    monkeypatch.setattr(subject, 'comparison', lambda *_: dict(per_task=[dict(task='task', n=5, hybrid_mean_score=0., gpt_only_mean_score=0.)]))
    monkeypatch.setattr(subject, 'verified_video', lambda row: dict(source=str(video), sha256=original_sha, corrected=False))
    output = tmp_path/'export'
    result = subject.export(workflow, 'approved', deep_path, pair_path, output)
    assert result['videos'] == 100 and sha256(video) == original_sha
    with zipfile.ZipFile(result['zip']) as archive:
        assert len(archive.namelist()) == 101 and archive.testzip() is None
        assert len([name for name in archive.namelist() if name.endswith('.mp4')]) == 100
    assert (output/'scores.csv').is_file() and (output/'results.json').is_file()
    before = sha256(output/'COMPLETE.json')
    with pytest.raises(ValueError, match='new export directory'):
        subject.export(workflow, 'approved', deep_path, pair_path, output)
    assert sha256(output/'COMPLETE.json') == before
