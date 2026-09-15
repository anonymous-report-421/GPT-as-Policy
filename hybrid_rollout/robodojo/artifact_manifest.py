"""Inventory and verify data needed to re-render a rollout without simulation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .io import sha256, write_json


OBSERVATION_KEYS = {
    'cam_high', 'cam_left_wrist', 'cam_right_wrist', 'states',
    'eef_positions', 'eef_quaternions_wxyz', 'instruction', 'remaining_steps',
}
SECRET_NAMES = {'auth.json', 'credentials.json', '.env'}


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def numbered(paths, prefix='', suffix=''):
    values = []
    for path in paths:
        name = path.name
        core = name[len(prefix):len(name)-len(suffix) if suffix else None]
        try:
            values.append(int(core))
        except ValueError:
            return None
    return sorted(values)


def entry(root: Path, path: Path) -> dict:
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
        raise ValueError(f'Artifact symlinks are not allowed: {relative}')
    if path.name in SECRET_NAMES or any(part == 'codex_home' for part in path.parts):
        raise ValueError(f'Secret/runtime state must not enter the archive: {relative}')
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def is_render_input(relative: str) -> bool:
    if relative.startswith('sim/'):
        return True
    if relative == 'codex_config.toml':
        return True
    if relative.startswith('controller/'):
        name = Path(relative).name
        return (name in {'run.json', 'result.json', 'history.json', 'agent_events.jsonl',
                         'rpc_out.jsonl', 'PROMPT.md', 'SKILL.md', 'gate_prompt.md'} or
                name.startswith(('request_', 'response_', 'proposal_', 'execution_', 'edit_')) or
                '/observations/' in relative or '/proposals/' in relative)
    return relative.endswith((
        'robodojo_server/debug_recorder.py', 'robodojo_server/video_panel.py',
        'assets/fonts/NotoSansCJKsc-Regular.otf'))


def build_manifest(run_root: Path) -> dict:
    root = Path(run_root).resolve()
    controller, sim = root/'controller', root/'sim'
    result = read_json(controller/'result.json', {})
    summary = read_json(sim/'summary.json', {})
    reset = read_json(sim/'reset.json', {})
    step_id = result.get('step_id', summary.get('step_id'))
    episode_id = reset.get('episode_id') or result.get('episode_id')
    episode = sim/episode_id if episode_id else None
    observations = sorted((episode/'observations').glob('*.npz')) if episode else []
    actions = sorted(episode.glob('action_*.json')) if episode else []
    observation_indices = numbered(observations, suffix='.npz')
    action_indices = numbered(actions, prefix='action_', suffix='.json')
    observation_keys_complete = bool(observations)
    observation_errors = []
    for path in observations:
        try:
            with np.load(path, allow_pickle=False) as data:
                observation_keys_complete &= OBSERVATION_KEYS.issubset(data.files)
        except Exception as error:
            observation_keys_complete = False
            observation_errors.append({'path': str(path), 'error': type(error).__name__})
    checks = {
        'has_controller_result': bool(result),
        'has_sim_reset': bool(reset),
        'has_sensor_video': (sim/'sensors.mp4').is_file(),
        'has_source_snapshot': (root/'source_snapshot/hybrid_rollout/robodojo/robodojo_server/debug_recorder.py').is_file(),
        'observation_indices_contiguous': (isinstance(step_id, int) and
            observation_indices == list(range(step_id + 1))),
        'action_indices_contiguous': (isinstance(step_id, int) and
            action_indices == list(range(step_id))),
        'observation_payload_keys_complete': observation_keys_complete,
        'video_frame_contract': (isinstance(step_id, int) and
            summary.get('video_frames') == step_id + 1),
    }
    run = read_json(root/'controller/run.json', {})
    if run.get('evaluation_method') == 'gpt_only':
        worker = read_json(root/'controller/codex_workspace/worker.json', {})
        calls = [read_json(p, {}) for p in (root/'controller/codex_workspace').glob('call_*_request.json')]
        checks['gpt_only_no_pi05_service'] = (
            run.get('pi05_enabled') is False and result.get('pi05_enabled') is False
            and result.get('predictions') == 0 and result.get('student_steps') == 0
            and not (root/'policy.pid').exists() and not (root/'policy_identity.json').exists()
            and not list((root/'controller').glob('proposal_*.npz')))
        checks['gpt_only_direct_tool_contract'] = (
            worker.get('evaluation_method') == 'gpt_only'
            and worker.get('tools') == ['robodojo_start', 'robodojo_act']
            and bool(calls) and all(c.get('tool') in ('robodojo_start', 'robodojo_act') for c in calls))
        if run.get('action_space') == 'eef_only':
            responses = [read_json(p, {}) for p in controller.glob('response_*.json')]
            checks['gpt_only_eef_action_contract'] = (
                result.get('gpt_joint_steps') == 0 and bool(responses)
                and all(r.get('mode') == 'eef' and r.get('evaluation_method') == 'gpt_only'
                        and type(r.get('steps')) is int and 1 <= r['steps'] <= 5
                        for r in responses))
    if (root/'evaluation_manifest.json').exists() or (root/'evaluation_case.json').exists():
        from .evaluation import read_panel, case_identity
        try:
            panel = read_panel(root/'evaluation_manifest.json')
            selected = read_json(root/'evaluation_case.json')
            case = selected['case']
            identity = case_identity(panel, case)
            fingerprint = read_json(sim/'initial_observation_fingerprint.json', {})
            outcome = read_json(sim/'evaluation_outcome.json', {})
            checks['paired_evaluation_identity_matches'] = (
                case in panel['cases'] and selected['identity'] == identity and
                reset.get('metadata', {}).get('evaluation_case') == identity and
                result.get('evaluation_case') == identity and
                fingerprint.get('evaluation_case') == identity and
                outcome.get('evaluation_case') == identity)
            checks['frozen_layout_copy_matches'] = sha256(sim/'scene_layout.json') == case['layout']['sha256']
        except (ValueError, OSError, KeyError, TypeError):
            checks['paired_evaluation_identity_matches'] = False
            checks['frozen_layout_copy_matches'] = False
    excluded = {'artifact_manifest.json', 'artifact_manifest.json.tmp',
                'artifact_manifest.log', 'job_exit_status.txt'}
    files = sorted(path for path in root.rglob('*') if path.is_file() and path.name not in excluded)
    inventory = [entry(root, path) for path in files]
    render_inventory = [item for item in inventory if is_render_input(item['path'])]
    verified = all(checks.values()) and bool(render_inventory)
    return {
        'schema': 'hybrid_rollout.robodojo.artifacts.v1',
        'evaluation_method': run.get('evaluation_method', 'pi05_plus_gpt'),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'verified' if verified else 'incomplete',
        'run_root': str(root),
        'episode_id': episode_id,
        'step_id': step_id,
        'checks': checks,
        'raw_observation_count': len(observations),
        'raw_action_count': len(actions),
        'observation_payload_errors': observation_errors,
        'render_contract': {
            'source': 'sim/sensors.mp4 plus lossless per-control observation NPZ files',
            'frame_mapping': 'initial observation 0; observation t is the ACK after action t-1',
            'simulation_required_to_rerender': False,
            'render_input_count': len(render_inventory),
        },
        'render_inputs': render_inventory,
        'all_immutable_artifacts': inventory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    output = args.run_root.resolve()/'artifact_manifest.json'
    manifest = build_manifest(args.run_root)
    write_json(output, manifest)
    if args.require_complete and manifest['status'] != 'verified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
