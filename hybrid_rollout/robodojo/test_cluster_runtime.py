"""Checks for isolated backend, ten sequential workers, and complete archives."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pytest

from .artifact_manifest import build_manifest
from .codex_backend.validate import validate_config
from .job_matrix import TASKS, jobs
from .runtime_manifest import make_manifest
from .settings import BASE_URL, EFFORT, MODEL, PROVIDER


def test_pinned_company_config_and_isolated_entrypoint():
    root = Path(__file__).parent
    config = validate_config(root/'codex_backend/config.toml')
    assert (config['model_provider'], config['model'], config['model_reasoning_effort']) == (
        PROVIDER, MODEL, EFFORT)
    assert config['model_providers'][PROVIDER]['base_url'] == BASE_URL
    entrypoint = (root/'episode_entrypoint.sh').read_text()
    launcher = (root/'run_local.sh').read_text()
    assert 'CODEX_HOME="$ROLLOUT_CODEX_HOME_DIR"' in launcher
    assert '~/.codex' not in entrypoint + launcher
    assert 'OPENAI_API_KEY="$codex_api_key"' in launcher
    assert 'unset api_key OPENAI_API_KEY' in entrypoint
    assert 'rm -rf' not in entrypoint
    assert '/private_runtime/' in entrypoint
    assert config['features']['fast_mode'] is False
    assert config.get('service_tier') in (None, 'default')
    assert not any(value is False for key, value in config['features'].items() if key != 'fast_mode')


@pytest.mark.parametrize('change', ['enable_fast', 'missing_fast', 'fast', 'priority', 'ultrafast'])
def test_fast_mode_and_premium_tiers_are_rejected(tmp_path, change):
    content = (Path(__file__).parent/'codex_backend/config.toml').read_text()
    if change == 'enable_fast':
        content = content.replace('fast_mode = false', 'fast_mode = true')
    elif change == 'missing_fast':
        content = content.replace('fast_mode = false', '')
    else:
        content = f'service_tier = "{change}"\n' + content
    config = tmp_path/'config.toml'
    config.write_text(content)
    with pytest.raises(ValueError):
        validate_config(config)


def test_tool_environment_inherits_directories_not_an_allowlist(tmp_path):
    root = Path(__file__).parent
    codex_dir, python_dir, image_bin = (tmp_path/name for name in ('codex', 'python', 'image'))
    for directory in (codex_dir, python_dir, image_bin):
        directory.mkdir()
    for directory, name in ((codex_dir, 'rg'), (python_dir, 'python'), (image_bin, 'unlisted_tool')):
        program = directory/name
        program.write_text('#!/bin/sh\nexit 0\n')
        program.chmod(0o755)
    env = dict(os.environ, CODEX_BIN=str(codex_dir/'codex'),
               ROBODOJO_PYTHON=str(python_dir/'python'), PATH=f'{image_bin}:/usr/bin:/bin')
    result = subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c',
        'source "$1"; /bin/bash --noprofile --norc -c "command -v rg python unlisted_tool"',
        'test', str(root/'tool_environment.sh')], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(codex_dir/'rg'), str(python_dir/'python'), str(image_bin/'unlisted_tool')]
    assert '&& rg ' not in (root/'run_local.sh').read_text()


@pytest.mark.parametrize('auth_profile', ['galbot', 'codex_a_2', 'codex_a_3', 'codex_b_4', 'codex_b_5'])
def test_bootstrap_forwards_termination_and_preserves_log(tmp_path, auth_profile):
    root = Path(__file__).parent
    fake = tmp_path/'source/hybrid_rollout/robodojo'
    fake.mkdir(parents=True)
    (fake/'cluster_entrypoint.sh').write_text(
        '#!/bin/bash\ntrap \'echo reset_completed; exit 143\' TERM\necho started\n'
        'while :; do sleep 0.1; done\n')
    log = tmp_path/'bootstrap.log'
    env = dict(os.environ, CODE_ROOT=str(tmp_path/'source'), ROLLOUT_EXPERIMENT_ID='test',
        ROBODOJO_TASK='build_tower', ROLLOUT_REPLICA_ID='6', ROLLOUT_COUNT='1',
        ROLLOUT_EVAL_MANIFEST='unused', ROLLOUT_EVAL_MANIFEST_SHA256='unused',
        ROLLOUT_OPENAI_API_KEY_FILE='unused',
        ROLLOUT_RUN_UID=str(os.getuid()), ROLLOUT_RUN_GID=str(os.getgid()),
        ROLLOUT_BOOTSTRAP_LOG=str(log))
    env['ROLLOUT_AUTH_PROFILE'] = auth_profile
    if auth_profile.startswith('codex_'):
        env.pop('ROLLOUT_OPENAI_API_KEY_FILE', None)
    process = subprocess.Popen(['/bin/bash', str(root/'acp_startup.sh')], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic()+10
        while not log.exists() or 'started' not in log.read_text():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        process.terminate()
        process.communicate(timeout=10)
        assert process.returncode == 143
        assert 'reset_completed' in log.read_text()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_frozen_job_matrix_is_10_by_6():
    panel = Path(__file__).parent/'eval_panels/robodojo_panel60_v1.json'
    matrix = list(jobs('panel', panel))
    assert len(TASKS) == 10 and len(matrix) == 10
    assert {row['ROBODOJO_TASK'] for row in matrix} == set(TASKS)
    assert all(sum(row['ROBODOJO_TASK'] == task for row in matrix) == 1 for task in TASKS)
    assert all(row['ROLLOUT_COUNT'] == '6' and row['ROLLOUT_EVAL_MANIFEST'] == str(panel) for row in matrix)
    assert len({row['ROLLOUT_EVAL_MANIFEST_SHA256'] for row in matrix}) == 1


def synthetic_archive(root):
    controller, sim, snapshot = root/'controller', root/'sim', root/'source_snapshot/hybrid_rollout'
    episode = sim/'episode'
    (episode/'observations').mkdir(parents=True)
    controller.mkdir()
    (snapshot/'robodojo/robodojo_server').mkdir(parents=True)
    (snapshot/'assets/fonts').mkdir(parents=True)
    for name in ('debug_recorder.py', 'video_panel.py'):
        (snapshot/'robodojo/robodojo_server'/name).write_text(name)
    (snapshot/'assets/fonts/NotoSansCJKsc-Regular.otf').write_bytes(b'font')
    for tick in range(3):
        np.savez_compressed(episode/'observations'/f'{tick:06d}.npz',
            cam_high=np.zeros((2, 2, 3), np.uint8),
            cam_left_wrist=np.zeros((2, 2, 3), np.uint8),
            cam_right_wrist=np.zeros((2, 2, 3), np.uint8),
            states=np.zeros(14), eef_positions=np.zeros((2, 3)),
            eef_quaternions_wxyz=np.zeros((2, 4)), instruction='test', remaining_steps=3-tick)
    for tick in range(2):
        (episode/f'action_{tick:06d}.json').write_text('{}')
    (sim/'sensors.mp4').write_bytes(b'synthetic-video')
    (sim/'reset.json').write_text(json.dumps({'episode_id': 'episode'}))
    (sim/'summary.json').write_text(json.dumps({'step_id': 2, 'video_frames': 3}))
    (controller/'result.json').write_text(json.dumps({'episode_id': 'episode', 'step_id': 2}))
    (controller/'run.json').write_text('{}')
    (controller/'history.json').write_text('[]')


def test_artifact_manifest_covers_raw_video_inputs(tmp_path):
    synthetic_archive(tmp_path)
    manifest = build_manifest(tmp_path)
    assert manifest['status'] == 'verified'
    assert manifest['raw_observation_count'] == 3 and manifest['raw_action_count'] == 2
    paths = {item['path'] for item in manifest['render_inputs']}
    assert 'sim/sensors.mp4' in paths
    assert 'sim/episode/observations/000002.npz' in paths
    assert 'source_snapshot/hybrid_rollout/robodojo/robodojo_server/video_panel.py' in paths
    assert not manifest['render_contract']['simulation_required_to_rerender']


def test_launch_manifest_never_archives_secret(monkeypatch, tmp_path):
    root = Path(__file__).parent
    monkeypatch.setenv('TASK', 'build_tower')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-appear')
    monkeypatch.setenv('CODEX_MAX_TOTAL_TOKENS', '0')
    monkeypatch.setenv('CODEX_IMAGE_MAX_EDGE', '480')
    manifest = make_manifest(tmp_path, root/'codex_backend/config.toml')
    serialized = json.dumps(manifest)
    assert 'must-not-appear' not in serialized
    assert 'OPENAI_API_KEY' not in manifest['resolved_environment']
    assert manifest['derived_process_environment']['controller']['OPENAI_API_KEY'].endswith(
        'value not archived')
    assert manifest['backend']['credential_value_archived'] is False
    assert manifest['resolved_environment']['CODEX_MAX_TOTAL_TOKENS'] == '0'
    assert manifest['resolved_environment']['CODEX_IMAGE_MAX_EDGE'] == '480'
