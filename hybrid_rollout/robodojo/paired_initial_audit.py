"""Read-only paired initial-state/settings audit; never starts or edits rollouts.

RGB differences are reported, not rounded away or treated as identical physics.
The caller must still check the separate full-trajectory/video deep audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_follow import load_plans, selected_rows
from .evaluation import case_identity, read_panel
from .io import require, sha256, write_json
from .campaign import utc

CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
NON_RGB = ('states', 'eef_positions', 'eef_quaternions_wxyz', 'instruction', 'remaining_steps')
RUN_KEYS = ('context_version', 'teacher_model', 'teacher_model_provider',
    'teacher_reasoning_effort', 'task', 'instruction', 'seed', 'evaluation_case',
    'robot_profile', 'action_dim', 'max_episode_steps', 'control_dt',
    'max_decisions', 'require_native_termination', 'no_rollback')
SIM_CONFIGS = ('sim/resolved_config.json', 'sim/scene_layout.json', 'sim/native_evaluation.json')
COMMON_SOURCES = ('prompt_context.py', 'settings.py', 'robodojo_server/validation.py',
    'robodojo_server/kinematics.py', 'robodojo_server/action_edit_kinematics.py')


def compare_observations(left, right):
    require(set(CAMERAS + NON_RGB) <= left.keys() & right.keys(), 'Missing initial observation fields')
    result = dict(robot_text_exact=True, rgb_bitwise_equal=True, fields={})
    for key in CAMERAS + NON_RGB:
        a, b = left[key], right[key]
        require(a.dtype == b.dtype and a.shape == b.shape, 'Initial shape/dtype changed: ' + key)
        if np.issubdtype(a.dtype, np.number):
            require(np.isfinite(a).all() and np.isfinite(b).all(), 'Non-finite initial field: ' + key)
        equal = a.tobytes() == b.tobytes()
        row = dict(shape=list(a.shape), dtype=str(a.dtype), bitwise_equal=bool(equal))
        if key in CAMERAS:
            require(a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, 'Invalid RGB: ' + key)
            delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
            row.update(mae_255=float(delta.mean()), p99=float(np.percentile(delta, 99)),
                max=int(delta.max()), over_8_pct=float((delta > 8).mean() * 100))
            result['rgb_bitwise_equal'] &= bool(equal)
        else:
            result['robot_text_exact'] &= bool(equal)
        result['fields'][key] = row
    return result


def _load_archive(row):
    root = Path(row['archive']).resolve()
    require(sha256(root/'artifact_manifest.json') == row['artifact_manifest_sha256'], 'Manifest changed')
    manifest = json.loads((root/'artifact_manifest.json').read_text())
    idle = row.get('evaluation_failure_reason') is not None
    if idle:
        from .adjudicated_archive_audit import adjudicated
        adjudicated(row)
        require(manifest.get('status') == 'incomplete', 'Original idle outcome must remain incomplete')
    else:
        require(manifest.get('status') == 'verified', 'Archive not verified')
    records = {r['path']: r['sha256'] for r in manifest['all_immutable_artifacts']}

    def checked_path(relative):
        path = (root/relative).resolve()
        require(path.is_relative_to(root), 'Artifact path escapes archive')
        require(sha256(path) == records.get(relative), 'Artifact changed or untracked: ' + relative)
        return path

    def read(relative):
        return json.loads(checked_path(relative).read_text())

    run, reset, outcome = read('controller/run.json'), read('sim/reset.json'), read('sim/evaluation_outcome.json')
    fp = read('sim/initial_observation_fingerprint.json')
    if idle:
        require(outcome.get('complete') is False and outcome.get('native_success') is None
            and outcome.get('native_score') is None, 'Adjudicated native outcome changed')
    else:
        result = read('controller/result.json')
        require(result.get('complete') is True and outcome.get('complete') is True
            and outcome.get('valid_for_success_rate') is True, 'Not native complete')
    require(outcome['native_success'] is row['native_success']
        and outcome['native_score'] == row['native_score'], 'Selected native outcome changed')
    require(run.get('evaluation_method', 'pi05_plus_gpt') == row['method'], 'Wrong method')
    identity = row['evaluation_case']
    require(identity['case_id'] == row['case_id'], 'Wrong case identity')
    require(all(x == identity for x in (run['evaluation_case'], reset['metadata']['evaluation_case'],
        fp['evaluation_case'], outcome['evaluation_case'])), 'Recorded case/seed identities differ')
    require(run['context_version'] == 'v3', 'Not v3')
    require(reset['step_id'] == 0, 'Not an initial observation')
    frame_relative = f"sim/{reset['episode_id']}/observations/000000.npz"
    frame_path = checked_path(frame_relative)
    with np.load(frame_path, allow_pickle=False) as stream:
        observation = {key: stream[key] for key in stream.files}
    for key in CAMERAS + NON_RGB:
        value = observation[key]
        actual = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest())
        require(actual == fp['fields'][key], 'Fingerprint does not match raw NPZ: ' + key)
    worker = read('controller/codex_workspace/worker.json')
    if row['method'] == 'gpt_only':
        require(run.get('pi05_enabled') is False and run.get('pi05_inference_calls') == 0
            and run.get('action_space') == 'eef_only', 'GPT-only method contract changed')
    return dict(root=str(root), run=run, reset=reset, observation=observation,
        source_hashes={name: sha256(checked_path('source_snapshot/hybrid_rollout/robodojo/' + name))
                       for name in COMMON_SOURCES},
        configs={name: read(name) for name in SIM_CONFIGS},
        image_max_edge=worker['codex_image_max_edge'], codex_version=worker['codex_version'],
        initial_npz_sha256=sha256(frame_path), artifact_manifest_sha256=row['artifact_manifest_sha256'])


def audit_pair(left, right):
    require(left['method'] == 'pi05_plus_gpt' and right['method'] == 'gpt_only', 'Wrong paired methods')
    require(left['case_id'] == right['case_id'] and left['evaluation_case'] == right['evaluation_case'],
        'Paired cases/seeds differ')
    a, b = _load_archive(left), _load_archive(right)
    differences = [key for key in RUN_KEYS if a['run'][key] != b['run'][key]]
    differences += [key for key in SIM_CONFIGS if a['configs'][key] != b['configs'][key]]
    differences += ['source:' + key for key in COMMON_SOURCES if a['source_hashes'][key] != b['source_hashes'][key]]
    differences += [key for key in ('image_max_edge', 'codex_version') if a[key] != b[key]]
    obs = compare_observations(a['observation'], b['observation'])
    return dict(case_id=left['case_id'], evaluation_case=left['evaluation_case'],
        hybrid_archive=a['root'], gpt_only_archive=b['root'],
        hybrid_initial_npz_sha256=a['initial_npz_sha256'], gpt_only_initial_npz_sha256=b['initial_npz_sha256'],
        hybrid_artifact_manifest_sha256=a['artifact_manifest_sha256'],
        gpt_only_artifact_manifest_sha256=b['artifact_manifest_sha256'],
        gpt_only_native_complete=right.get('native_complete',True),
        gpt_only_adjudication_sha256=right.get('adjudication_sha256'),
        shared_setting_differences=differences, shared_context_and_constraints_sha256=a['source_hashes'],
        settings_and_robot_text_match=not differences and obs['robot_text_exact'], **obs)


def expected_identities(plan):
    panel = read_panel(plan['eval_manifest'], plan['panel_sha256'])
    return {case['case_id']: case_identity(panel, case) for case in panel['cases']}


def audit_selected(workflow, approved):
    workflow = Path(workflow).resolve()
    plans = load_plans(workflow, approved)
    raw = (workflow.parent/'progress.json').read_bytes()
    progress = json.loads(raw)
    rows = selected_rows(progress, plans)
    expected = {method: expected_identities(plan) for method, plan in plans.items()}
    by_method = {m: {r['case_id']: r for r in rows if r['method'] == m} for m in plans}
    left, right = by_method['pi05_plus_gpt'], by_method['gpt_only']
    pairs, errors = [], []
    for cid in plans['gpt_only']['primary_case_ids']:
        if cid not in left or cid not in right:
            continue
        try:
            require(left[cid]['evaluation_case'] == expected['pi05_plus_gpt'][cid]
                and right[cid]['evaluation_case'] == expected['gpt_only'][cid],
                'Recorded identity differs from frozen panel')
            pair = audit_pair(left[cid], right[cid])
            if not pair['settings_and_robot_text_match']:
                errors.append(dict(case_id=cid, reason='Recorded settings or robot/text mismatch'))
            pairs.append(pair)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(case_id=cid, reason=str(error), error_type=type(error).__name__))
    return dict(schema='robodojo.paired_initial_observation_audit.v2', checked_utc=utc(),
        workflow=str(workflow), workflow_sha256=approved, progress_updated_utc=progress.get('updated_utc'),
        progress_sha256=hashlib.sha256(raw).hexdigest(), target_pairs=50, pairs=len(pairs), rows=pairs,
        errors=errors, status='review_required' if errors else ('complete_50_pairs' if len(pairs) == 50 else 'partial'),
        bitwise_physics_reproducibility_guaranteed=False, original_data_modified=False,
        scope_note='Initial frames, shared task context/constraints, scene config and native settings only; '
            'full action/video integrity is checked by the independent deep audit. '
            'Policy prompts differ in the approved pi05 gate versus direct EEF method instructions.',
        model_calls=0, simulation_runs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--approved-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new report path; preserve previous evidence')
    plans = load_plans(args.workflow.resolve(), args.approved_sha256)
    for plan in plans.values():
        results = Path(plan['shared_root'])/'results'
        require(not args.output.resolve().is_relative_to(results.resolve()), 'Report cannot alter rollout archives')
    report = audit_selected(args.workflow, args.approved_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    if report['errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
