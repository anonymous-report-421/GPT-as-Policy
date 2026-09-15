import copy
import json

import numpy as np
import pytest

from . import paired_initial_audit as subject
from .io import sha256


def observations():
    return dict(**{k: np.zeros((2, 3, 3), dtype=np.uint8) for k in subject.CAMERAS},
        states=np.zeros(14, dtype=np.float32), eef_positions=np.zeros((2, 3), dtype=np.float32),
        eef_quaternions_wxyz=np.array([[1, 0, 0, 0]] * 2, dtype=np.float32),
        instruction=np.array('Test task'), remaining_steps=np.array(100))


def fixture(tmp_path, method):
    root = tmp_path/method
    identity = dict(case_id='case0', reset_seed=0, layout_id=0)
    run = {key: 1 for key in subject.RUN_KEYS}
    run.update(context_version='v3', evaluation_case=identity, instruction='Test task',
        evaluation_method=method, pi05_enabled=False, pi05_inference_calls=0, action_space='eef_only')
    values = observations()
    fp = dict(evaluation_case=identity, fields={key: dict(shape=list(v.shape), dtype=str(v.dtype),
        sha256=subject.hashlib.sha256(v.tobytes()).hexdigest()) for key, v in values.items()})
    contents = {'controller/run.json': run,
        'controller/result.json': dict(complete=True),
        'sim/evaluation_outcome.json': dict(complete=True, valid_for_success_rate=True,
            native_success=False, native_score=0, evaluation_case=identity),
        'sim/reset.json': dict(episode_id='episode', step_id=0, metadata=dict(evaluation_case=identity)),
        'sim/initial_observation_fingerprint.json': fp,
        'controller/codex_workspace/worker.json': dict(codex_image_max_edge=480, codex_version='0.153.4'),
        **{key: dict(scene='same') for key in subject.SIM_CONFIGS}}
    for name, data in contents.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    for name in subject.COMMON_SOURCES:
        path = root/'source_snapshot/hybrid_rollout/robodojo'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('same source')
    frame = root/'sim/episode/observations/000000.npz'
    frame.parent.mkdir(parents=True)
    np.savez(frame, **values)
    row = dict(archive=str(root), case_id='case0', evaluation_case=identity, method=method,
        context_version='v3', native_success=False, native_score=0)
    refresh(row)
    return row


def refresh(row):
    root = subject.Path(row['archive'])
    files = [p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    manifest = dict(status='verified', all_immutable_artifacts=[dict(path=str(p.relative_to(root)),
        sha256=sha256(p)) for p in files])
    (root/'artifact_manifest.json').write_text(json.dumps(manifest))
    row['artifact_manifest_sha256'] = sha256(root/'artifact_manifest.json')


def test_rgb_subtraction_uses_signed_numbers_and_reports_differences():
    a, b = observations(), observations()
    b['cam_high'].fill(255)
    result = subject.compare_observations(a, b)
    assert result['robot_text_exact'] and not result['rgb_bitwise_equal']
    assert result['fields']['cam_high']['mae_255'] == 255
    assert result['fields']['cam_high']['max'] == 255


def test_numeric_equality_does_not_claim_bitwise_identity():
    a, b = observations(), observations()
    b['states'][0] = -0.0
    assert np.array_equal(a['states'], b['states'])
    assert not subject.compare_observations(a, b)['robot_text_exact']


@pytest.mark.parametrize('change', ['missing', 'dtype', 'shape', 'nan'])
def test_bad_initial_payload_rejected(change):
    a, b = observations(), observations()
    if change == 'missing':
        del b['states']
    elif change == 'dtype':
        b['states'] = b['states'].astype(np.float64)
    elif change == 'shape':
        b['states'] = np.zeros(13, dtype=np.float32)
    else:
        b['states'][0] = np.nan
    with pytest.raises(ValueError):
        subject.compare_observations(a, b)


def test_native_failures_are_valid_pairs(tmp_path):
    a = fixture(tmp_path, 'pi05_plus_gpt')
    b = fixture(tmp_path, 'gpt_only')
    result = subject.audit_pair(a, b)
    assert result['settings_and_robot_text_match'] and result['rgb_bitwise_equal']
    assert result['shared_setting_differences'] == []


def test_manifest_tampering_and_interrupted_case_rejected(tmp_path):
    a = fixture(tmp_path, 'pi05_plus_gpt')
    b = fixture(tmp_path, 'gpt_only')
    path = subject.Path(b['archive'])/'controller/result.json'
    path.write_text('{"complete":false}')
    with pytest.raises(ValueError, match='Artifact changed'):
        subject.audit_pair(a, b)
    refresh(b)
    with pytest.raises(ValueError, match='Not native complete'):
        subject.audit_pair(a, b)


def test_different_scene_not_silently_accepted(tmp_path):
    a = fixture(tmp_path, 'pi05_plus_gpt')
    b = fixture(tmp_path, 'gpt_only')
    (subject.Path(b['archive'])/'sim/scene_layout.json').write_text('{"scene":"different"}')
    refresh(b)
    result = subject.audit_pair(a, b)
    assert not result['settings_and_robot_text_match']
    assert result['shared_setting_differences'] == ['sim/scene_layout.json']


def test_raw_fingerprint_mismatch_is_caught(tmp_path):
    a = fixture(tmp_path, 'pi05_plus_gpt')
    b = fixture(tmp_path, 'gpt_only')
    values = observations()
    values['states'][0] = 0.1
    np.savez(subject.Path(b['archive'])/'sim/episode/observations/000000.npz', **values)
    refresh(b)
    with pytest.raises(ValueError, match='Fingerprint does not match'):
        subject.audit_pair(a, b)


def test_partial_coverage_never_claims_fifty(tmp_path, monkeypatch):
    a = fixture(tmp_path, 'pi05_plus_gpt')
    b = fixture(tmp_path, 'gpt_only')
    plans = {m: dict(primary_case_ids=['case0'] + [f'case{i}' for i in range(1, 50)])
             for m in ('pi05_plus_gpt', 'gpt_only')}
    monkeypatch.setattr(subject, 'load_plans', lambda *_: plans)
    monkeypatch.setattr(subject, 'expected_identities', lambda _: {'case0': a['evaluation_case']})
    progress = {key: dict(cases=[row], errors=[], platform_verified=True)
                for key, row in (('hybrid', a), ('direct', b))}
    (tmp_path/'progress.json').write_text(json.dumps(progress))
    result = subject.audit_selected(tmp_path/'workflow.json', 'test')
    assert result['pairs'] == 1 and result['target_pairs'] == 50
    assert result['status'] == 'partial' and result['errors'] == []
    monkeypatch.setattr(subject, 'expected_identities', lambda _: {'case0': dict(reset_seed=99)})
    result = subject.audit_selected(tmp_path/'workflow.json', 'test')
    assert result['status'] == 'review_required' and result['pairs'] == 0
    assert result['errors'][0]['reason'] == 'Recorded identity differs from frozen panel'
    bad = copy.deepcopy(progress)
    bad['direct']['cases'].append(b)
    (tmp_path/'progress.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='duplicate'):
        subject.audit_selected(tmp_path/'workflow.json', 'test')
