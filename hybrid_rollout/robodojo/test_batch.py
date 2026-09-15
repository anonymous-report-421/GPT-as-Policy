"""Exercise sequential persistence, orphan reset, and fail-stop without model calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .test_evaluation import fake_panel_file


FAKE_EPISODE = '''
import json, os, pathlib, subprocess, sys, shutil
from hybrid_rollout.robodojo.test_cluster_runtime import synthetic_archive
root = pathlib.Path(os.environ['ROLLOUT_ARCHIVE'])
synthetic_archive(root)
(root/'received_case.json').write_text(pathlib.Path(os.environ['ROLLOUT_CASE_FILE']).read_text())
(root/'received_env.json').write_text(json.dumps({key: os.environ[key] for key in
    ('ROBODOJO_TASK', 'ROLLOUT_LAYOUT_ID', 'ROLLOUT_EVAL_SEED')}))
selected = json.loads(pathlib.Path(os.environ['ROLLOUT_CASE_FILE']).read_text())
identity = selected['identity']
shutil.copyfile(os.environ['ROLLOUT_CASE_FILE'], root/'evaluation_case.json')
shutil.copyfile(os.environ['ROLLOUT_EVAL_MANIFEST'], root/'evaluation_manifest.json')
shutil.copyfile(pathlib.Path(os.environ['ROBODOJO_SOURCE'])/selected['case']['layout']['path'], root/'sim/scene_layout.json')
for relative in ('sim/initial_observation_fingerprint.json', 'sim/evaluation_outcome.json', 'controller/result.json'):
    path = root/relative
    value = json.loads(path.read_text()) if path.exists() else {}
    value['evaluation_case'] = identity
    path.write_text(json.dumps(value))
reset = root/'sim/reset.json'
value = json.loads(reset.read_text())
value['metadata'] = {'evaluation_case': identity}
reset.write_text(json.dumps(value))
orphan = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=True)
(root/'fake_orphan.pid').write_text(str(orphan.pid))
if os.environ.get('FAIL_FIRST') == '1':
    raise SystemExit(9)
'''


def execute(tmp_path, fail=False, task='build_tower', replica='6', count='3'):
    panel_path = tmp_path/'evaluation.json'
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        _, panel_path, panel = fake_panel_file(tmp_path)
    env = dict(os.environ, ROLLOUT_SHARED_ROOT=str(tmp_path), ROLLOUT_EXPERIMENT_ID='test',
        ROBODOJO_TASK=task, ROLLOUT_REPLICA_ID=replica, ROLLOUT_COUNT=count,
        ROBODOJO_SOURCE=str(tmp_path/'native'),
        ROLLOUT_EVAL_MANIFEST=str(panel_path), ROLLOUT_EVAL_MANIFEST_SHA256=panel['panel_sha256'],
        ROLLOUT_ATTEMPT='0', FAIL_FIRST='1' if fail else '0',
        ROLLOUT_MAX_SECONDS='0',
        PYTHONPATH=str(Path(__file__).parents[2]))
    # Test-only injected episode command; production uses episode_entrypoint.sh.
    code = ('import os,sys; from hybrid_rollout.robodojo.batch import run_batch; '
            f'raise SystemExit(run_batch(os.environ, episode_command=[sys.executable,"-c",{FAKE_EPISODE!r}]))')
    process = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60)
    batch_root = tmp_path/'results/test/_replicas'/f'replica_{replica}'/'attempt_0'
    return process, json.loads((batch_root/'batch.json').read_text())


def test_sequential_episodes_persist_and_reap_orphans(tmp_path):
    process, batch = execute(tmp_path)
    assert process.returncode == 0, process.stderr + process.stdout
    assert batch['state'] == 'completed' and len(batch['episodes']) == 3
    for row in batch['episodes']:
        assert row['artifact_status'] == 'verified'
        assert row['reset']['all_owned_processes_exited'] and row['reset']['ports_released']
        assert row['reset']['terminated_descendants']
        root = Path(row['archive'])
        pid = int((root/'fake_orphan.pid').read_text())
        assert not Path(f'/proc/{pid}').exists()
        assert (root/'controller/result.json').is_file()
        assert (root/'component_reset.json').is_file()
        assert json.loads((root/'received_case.json').read_text())['identity']['case_id'] == row['case_id']
        passed = json.loads((root/'received_env.json').read_text())
        assert passed['ROLLOUT_EVAL_SEED'] == '0'
        assert passed['ROLLOUT_LAYOUT_ID'] == str(row['index'])
    # The same replica/attempt cannot overwrite existing results.
    again, _ = execute(tmp_path)
    assert again.returncode != 0


def test_infrastructure_failure_keeps_data_and_stops_batch(tmp_path):
    process, batch = execute(tmp_path, fail=True)
    assert process.returncode == 9 and len(batch['episodes']) == 1
    assert batch['state'] == 'failed'
    assert batch['final_reset']['all_owned_processes_exited']
    assert (Path(batch['episodes'][0]['archive'])/'controller/result.json').is_file()


def test_generalization_switches_variant_with_fresh_reset(tmp_path):
    process, batch = execute(tmp_path, task='fold_clothes', replica='8', count='6')
    assert process.returncode == 0, process.stderr + process.stdout
    rows = batch['episodes']
    assert [(r['variant'], r['layout_id']) for r in rows] == [
        (v, i) for v in ('standard', 'random') for i in range(3)]
    assert len({r['archive'] for r in rows}) == 6
    for row in rows:
        assert row['artifact_status'] == 'verified'
        received = json.loads((Path(row['archive'])/'received_env.json').read_text())
        assert received['ROBODOJO_TASK'] == row['runtime_task']
        assert row['reset']['all_owned_processes_exited'] and row['reset']['ports_released']
