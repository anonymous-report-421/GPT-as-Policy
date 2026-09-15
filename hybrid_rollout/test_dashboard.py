"""Synthetic artifact tests; no simulator, model or real RGB inspection."""
import json
from pathlib import Path

import pytest

from .dashboard import Artifacts


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_stopped_run_identity_images_and_read_only(tmp_path):
    run = tmp_path/'ordered_blocks_codex_tools_06'
    controller = run/'controller'
    put(controller/'run.json', dict(teacher_model='gpt-6-astra', teacher_reasoning_effort='high',
                                   control_dt=1/15, max_episode_steps=1350))
    put(controller/'result.json', dict(step_id=735, complete=False, reason='controller_error',
                                      student_steps=555, recovery_steps=180))
    put(controller/'history.json', [dict(decision=0)])
    put(controller/'observations/001/observation.json', dict(step_id=735))
    put(controller/'response_000.json', dict(mode='eef', reason='Synthetic public decision.'))
    image = controller/'observations/001/main_rgb.png'
    image.write_bytes(b'synthetic-png')
    # Partial next observation must not hide the latest complete snapshot.
    put(controller/'observations/002/incomplete.json', {})
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob('*') if p.is_file()}
    artifacts = Artifacts(tmp_path)
    status = artifacts.status()
    assert status['effort'] == 'high' and status['step_id'] == 735
    assert status['sim_seconds'] == 49 and status['state'] == '已停止 / 未完成'
    assert status['decisions'] == 1 and status['observation_tick'] == 735
    assert 'obs=001' in status['images']['main_rgb']
    assert status['decision']['mode'] == 'eef'
    assert artifacts.status() is status  # Shared short cache.
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.rglob('*') if p.is_file()}


def test_no_arbitrary_path_or_symlink_export(tmp_path):
    root = tmp_path/'results'
    run = root/'ordered_blocks_codex_agent_01'
    run.mkdir(parents=True)
    artifacts = Artifacts(root)
    outside = tmp_path/'private.json'
    put(outside, dict(secret='not served'))
    for name in ('../private.json', str(outside), 'missing'):
        with pytest.raises(FileNotFoundError):
            artifacts.resolve_run(name)
    with pytest.raises(FileNotFoundError):
        artifacts.image_path(run, '../../private', 'main_rgb')
    folder = run/'controller/observations/000'
    folder.mkdir(parents=True)
    (folder/'main_rgb.png').symlink_to(outside)
    with pytest.raises(FileNotFoundError):
        artifacts.image_path(run, '000', 'main_rgb')
    assert artifacts.read(outside) is None


def test_waiting_partial_files_and_public_activity_only(tmp_path):
    run = tmp_path/'ordered_blocks_codex_agent_01'
    (run/'controller').mkdir(parents=True)
    (run/'controller/progress.json').write_text('{')
    log = run/'logs/controller.log'
    log.parent.mkdir()
    log.write_text(json.dumps(dict(event='agent_activity', item_type='commandExecution',
        phase='started', command='not exposed', step_id=0))+'\n'+
        json.dumps(dict(event='reasoning', text='not exposed'))+'\n'+'partial')
    report = Artifacts(tmp_path).status()
    assert report['model'] == '未确认' and report['images'] == {}
    assert report['state'] == '等待启动 / 未检测到控制进程'
    assert report['activity']['item_type'] == 'commandExecution'
    assert 'command' not in report['activity'] and 'text' not in report['activity']
