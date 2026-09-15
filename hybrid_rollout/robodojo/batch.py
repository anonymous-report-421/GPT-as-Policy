"""Run one task sequentially, persisting each result before a full process reset."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import traceback

from .artifact_manifest import build_manifest
from .io import write_json
from .job_matrix import TASKS
from .evaluation import read_panel, selected_cases, verify_assets, case_identity, archive_path
from .case_ledger import refresh, require_available
from .settings import EFFORT, MODEL, PROVIDER


def utc():
    return datetime.now(timezone.utc).isoformat()


def enable_subreaper():
    # Adopt only this worker's orphan descendants, including Codex's own session.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot isolate rollout descendant cleanup')


def children():
    for _ in range(1000):
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
    path = Path(f'/proc/self/task/{os.getpid()}/children')
    return [int(value) for value in path.read_text().split()]


def reset_descendants(grace=30):
    """Terminate and reap live descendants; never use stored PIDs from another run."""
    signalled = set()
    deadline = time.monotonic() + grace
    while True:
        live = children()
        if not live:
            return {'all_owned_processes_exited': True, 'terminated_descendants': sorted(signalled)}
        sig = signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL
        for pid in live:
            try:
                os.kill(pid, sig)
                signalled.add(pid)
            except ProcessLookupError:
                pass
        if time.monotonic() > deadline + 10:
            return {'all_owned_processes_exited': False, 'remaining_descendants': live,
                    'terminated_descendants': sorted(signalled)}
        time.sleep(0.1)


def ports_released(ports):
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()


def load_if_exists(path):
    return json.loads(path.read_text()) if path.is_file() else None


def run_batch(env, *, episode_command=None):
    env = dict(env)
    from .method import evaluation_method
    method = evaluation_method(env.get('ROLLOUT_EVALUATION_METHOD', 'pi05_plus_gpt'))
    env['ROLLOUT_EVALUATION_METHOD'] = method
    for name in ('ROLLOUT_EXPERIMENT_ID', 'ROBODOJO_TASK', 'ROLLOUT_REPLICA_ID'):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', env.get(name, '')):
            raise ValueError(f'Missing or invalid {name}')
    task = env['ROBODOJO_TASK']
    if task not in TASKS:
        raise ValueError(f'Task is not in the frozen panel: {task}')
    attempt = env.setdefault('ROLLOUT_ATTEMPT', '0')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        raise ValueError('Invalid ROLLOUT_ATTEMPT')
    count = int(env.setdefault('ROLLOUT_COUNT', '1'))
    max_seconds = int(env.setdefault('ROLLOUT_MAX_SECONDS', '7200'))
    if not 1 <= count <= 6 or max_seconds < 0:
        raise ValueError('Require count in 1..6 and non-negative timeout; 0 disables the wall limit')
    if 'ROLLOUT_SEED_BASE' in env:
        raise ValueError('ROLLOUT_SEED_BASE is ambiguous; use a frozen ROLLOUT_EVAL_MANIFEST')
    panel = read_panel(env['ROLLOUT_EVAL_MANIFEST'], env['ROLLOUT_EVAL_MANIFEST_SHA256'])
    explicit_case = env.get('ROLLOUT_CASE_ID')
    if method == 'gpt_only' and episode_command is None:
        from .phase_gate import verify_certificate
        verify_certificate(env.get('ROLLOUT_PHASE1_CERTIFICATE'), panel, explicit_case)
    if episode_command is None and (count != 1 or not explicit_case):
        raise ValueError('New production workers require ROLLOUT_COUNT=1 and an explicit ROLLOUT_CASE_ID')
    cases = selected_cases(panel, task, env['ROLLOUT_REPLICA_ID'], count, explicit_case)
    if episode_command is None:
        verify_assets(panel, env['ROBODOJO_SOURCE'], cases)
    shared = Path(env.setdefault('ROLLOUT_SHARED_ROOT', env.get('RUNTIME_ROOT',
        '/mnt/rollout/robodojo_mixed_control'))).resolve()
    if episode_command is None and (not shared.is_relative_to('/mnt') or shared == Path('/mnt')):
        raise ValueError('Shared output root must be below /mnt')
    experiment, replica = env['ROLLOUT_EXPERIMENT_ID'], env['ROLLOUT_REPLICA_ID']
    batch_root = shared/'results'/experiment/'_replicas'/f'replica_{replica}'/f'attempt_{attempt}'
    rerun_of = json.loads(env.get('ROLLOUT_RERUN_REFERENCE', 'null'))
    if explicit_case:
        require_available(shared, panel, explicit_case, batch_root/'batch.json', rerun_of=rerun_of, method=method)
    batch_root.parent.mkdir(parents=True, exist_ok=True)
    batch_root.mkdir()  # Atomic reservation: repeated scheduler launches cannot overwrite.
    write_json(batch_root/'evaluation_manifest.json', panel)
    ports = [int(env.get('POLICY_PORT', 18830)), int(env.get('SIM_PORT', 19113))]
    command = episode_command or ['bash', str(Path(__file__).with_name('episode_entrypoint.sh'))]
    enable_subreaper()
    from .prompt_context import CONTEXT_VERSION
    manifest = dict(schema='hybrid_rollout.robodojo.batch.v2', state='running',
        evaluation_method=method,
        context_version=CONTEXT_VERSION,
        rerun_of=rerun_of,
        started_utc=utc(), task=task, replica=replica, attempt=attempt, planned_rollouts=count,
        panel_id=panel['panel_id'], panel_sha256=panel['panel_sha256'],
        eval_seed=panel['eval_seed'], model=MODEL, effort=EFFORT, provider=PROVIDER, episodes=[])
    write_json(batch_root/'batch.json', manifest)
    refresh(shared, env['ROLLOUT_EVAL_MANIFEST'], method=method)
    active = None
    exit_code = 0
    def interrupted(signum, frame):
        raise InterruptedError(f'Batch received signal {signum}')
    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, case in enumerate(cases):
            if episode_command is None:
                verify_assets(panel, env['ROBODOJO_SOURCE'], [case])
            archive = archive_path(shared, experiment, case, replica, attempt)
            identity = case_identity(panel, case)
            case_file = batch_root/f'case_{index:03d}.json'
            write_json(case_file, dict(identity=identity, case=case))
            row = dict(index=index, **identity, archive=str(archive), state='starting', started_utc=utc())
            manifest['episodes'].append(row)
            write_json(batch_root/'batch.json', manifest)
            refresh(shared, env['ROLLOUT_EVAL_MANIFEST'], method=method)
            if not ports_released(ports):
                raise RuntimeError('A rollout port is already owned; refusing to start another component')
            child_env = dict(env, ROBODOJO_TASK=case['runtime_task'],
                             ROLLOUT_BASE_TASK=case['task'], ROLLOUT_VARIANT=case['variant'],
                             ROLLOUT_EVAL_SEED=str(case['eval_seed']),
                             ROLLOUT_LAYOUT_ID=str(case['layout_id']), ROLLOUT_INDEX=str(index),
                             ROLLOUT_CASE_FILE=str(case_file), ROLLOUT_ARCHIVE=str(archive))
            print(json.dumps(dict(event='rollout_start', **row)), flush=True)
            with (batch_root/f'rollout_{index:03d}.log').open('x') as log:
                active = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                try:
                    row['exit_code'] = active.wait(timeout=max_seconds or None)
                except subprocess.TimeoutExpired:
                    row['exit_code'] = 124
                    row['error'] = 'Per-rollout wall time limit reached'
                finally:
                    if active.poll() is None:
                        active.terminate()
                        try:
                            active.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            active.kill()
                            active.wait()
                    active = None
            reset = reset_descendants()
            reset['ports_released'] = ports_released(ports)
            reset['finished_utc'] = utc()
            row['reset'] = reset
            write_json(batch_root/f'reset_{index:03d}.json', reset)
            row['result'] = load_if_exists(archive/'controller/result.json')
            row['evaluation_outcome'] = load_if_exists(archive/'sim/evaluation_outcome.json')
            row['token_usage'] = load_if_exists(archive/'controller/token_usage.json')
            # Hash only after every component has exited, including orphaned native tools.
            if archive.is_dir():
                write_json(archive/'component_reset.json', reset)
                artifact = build_manifest(archive)
                write_json(archive/'artifact_manifest.json', artifact)
                row['artifact_status'] = artifact['status']
            okay = (row['exit_code'] == 0 and row.get('artifact_status') == 'verified'
                    and reset['all_owned_processes_exited'] and reset['ports_released'])
            row.update(state='completed' if okay else 'failed', finished_utc=utc())
            write_json(batch_root/'batch.json', manifest)
            refresh(shared, env['ROLLOUT_EVAL_MANIFEST'], method=method)
            print(json.dumps(dict(event='rollout_done', index=index, state=row['state'],
                                  archive=str(archive), reset=reset)), flush=True)
            if not okay:
                exit_code = row['exit_code'] or 5
                break  # Do not spend more tokens or silently retry after infrastructure failure.
    except BaseException as error:
        exit_code = 130 if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 2
        manifest['error'] = traceback.format_exc()
    finally:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=120)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()
        manifest['final_reset'] = reset_descendants()
        manifest['final_reset']['ports_released'] = ports_released(ports)
        if not all(manifest['final_reset'][k] for k in ('all_owned_processes_exited', 'ports_released')):
            exit_code = exit_code or 5
        manifest.update(state='completed' if exit_code == 0 else 'failed',
                        exit_code=exit_code, finished_utc=utc())
        write_json(batch_root/'batch.json', manifest)
        refresh(shared, env['ROLLOUT_EVAL_MANIFEST'], method=method)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(run_batch(os.environ))
