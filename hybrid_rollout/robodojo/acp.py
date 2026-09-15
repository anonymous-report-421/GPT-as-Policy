"""Freeze a shared source snapshot and a secret-free, reviewable ACP submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from .io import sha256, write_json
from .prompt_context import CONTEXT_VERSION
from .job_matrix import TASKS
from .evaluation import read_panel, selected_cases, verify_assets
from .case_ledger import make_rerun_reference, require_available, refresh
from .settings import MODEL, EFFORT, DEFAULT_CODEX_IMAGE_MAX_EDGE
from .codex_backend.profiles import PROFILES, profile, credential_path, validate_credential
from urllib.parse import urlparse


def prepare(args):
    from .method import evaluation_method
    method = evaluation_method(getattr(args, 'evaluation_method', None))
    certificate = getattr(args, 'phase1_certificate', None)
    if method == 'gpt_only' and not certificate:
        raise ValueError('GPT-only preparation requires a phase1 certificate path')
    quota_type = getattr(args, 'quota_type', 'spot')
    if quota_type not in ('spot', 'reserved'):
        raise ValueError('quota_type must explicitly be spot or reserved')
    if args.attempt < 0:
        raise ValueError('Attempt must be non-negative')
    if args.max_decisions < 0 or args.max_seconds < 0:
        raise ValueError('Budget limits must be non-negative; 0 means native termination only')
    if args.codex_image_max_edge < 0:
        raise ValueError('codex_image_max_edge must be non-negative')
    repo = Path(__file__).resolve().parents[2]
    for value in (args.experiment_id, str(args.replica_id)):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('Invalid experiment or replica identifier')
    if not 1 <= args.count <= 6:
        raise ValueError('count must be in 1..6')
    identity = profile(getattr(args, 'auth_profile', 'galbot'))
    https_proxy = getattr(args, 'https_proxy', None)
    https_proxy_file = getattr(args, 'https_proxy_file', None)
    if https_proxy_file:
        from .proxy_config import read_proxy
        if (https_proxy or identity['auth_mode'] != 'chatgpt'
                or not Path(https_proxy_file).resolve().is_relative_to(args.shared_root.resolve()/'private')):
            raise ValueError('Use only a shared private proxy file for managed-account slots')
        read_proxy(https_proxy_file)
    if https_proxy:
        proxy = urlparse(https_proxy)
        if (identity['auth_mode'] != 'chatgpt' or proxy.scheme not in ('http', 'https')
                or not proxy.hostname or proxy.username or proxy.password
                or proxy.path not in ('', '/') or proxy.query or proxy.fragment):
            raise ValueError('Explicit managed-account proxy must be an HTTP(S) URL without credentials')
    credential = (args.credential_file or credential_path(identity['name'], args.shared_root)).resolve()
    if identity['auth_mode'] == 'chatgpt' and credential != credential_path(identity['name'], args.shared_root):
        raise ValueError('Managed account auth must stay in its persistent isolated profile')
    if not getattr(args, 'review_without_credentials', False):
        validate_credential(identity['name'], args.shared_root, credential)
    runtime = args.shared_root.resolve()
    panel = read_panel(args.eval_manifest)
    cases = selected_cases(panel, args.task, args.replica_id, args.count, args.case_id)
    rerun_of = (make_rerun_reference(runtime, panel, args.case_id, args.rerun_v1_archive)
                if getattr(args, 'rerun_v1_archive', None) else None)
    if args.case_id:
        require_available(runtime, panel, args.case_id, rerun_of=rerun_of, method=method)
    verify_assets(panel, runtime/'src/RoboDojo', cases)
    if not runtime.is_relative_to('/mnt/rollout'):
        raise ValueError('New cluster outputs must be below /mnt/rollout')
    if not credential.is_relative_to('/mnt/rollout'):
        raise ValueError('Store the private rollout credential on the requested project share')
    root = runtime/'cluster'/args.experiment_id/f'replica_{args.replica_id}'
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()  # Never rewrite a submitted job's source or command.
    snapshot = root/'source_snapshot'
    shutil.copytree(repo/'hybrid_rollout', snapshot/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    parent_manifest = repo/'source_manifest.json'
    git_head = (json.loads(parent_manifest.read_text())['git_base_commit'] if parent_manifest.exists()
        else subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
    write_json(snapshot/'source_manifest.json', dict(git_base_commit=git_head,
        source_sha256=source_hash, includes_reviewed_worktree_changes=True, files=source))
    write_json(root/'evaluation_manifest.json', panel)
    runtime = str(runtime)
    env = {
        'ROLLOUT_EVALUATION_METHOD': method,
        'ROLLOUT_AUTH_PROFILE': identity['name'],
        'ROLLOUT_GATEWAY_NO_PROXY': urlparse(identity['base_url']).hostname if identity['auth_mode'] == 'api' else '',
        'CODE_ROOT': str(snapshot), 'RUNTIME_ROOT': runtime, 'ROLLOUT_SHARED_ROOT': runtime,
        'ROLLOUT_EXPERIMENT_ID': args.experiment_id, 'ROBODOJO_TASK': args.task,
        'ROLLOUT_REPLICA_ID': str(args.replica_id), 'ROLLOUT_COUNT': str(args.count),
        'ROLLOUT_EVAL_MANIFEST': str(root/'evaluation_manifest.json'),
        'ROLLOUT_EVAL_MANIFEST_SHA256': panel['panel_sha256'], 'ROLLOUT_ATTEMPT': str(args.attempt),
        'ROLLOUT_IMAGE_REF': args.image, 'ROLLOUT_OPENAI_API_KEY_FILE': str(credential),
        'ROLLOUT_RUN_UID': str(os.getuid()), 'ROLLOUT_RUN_GID': str(os.getgid()),
        'ROLLOUT_BOOTSTRAP_LOG': str(root/'bootstrap.log'),
        'ROLLOUT_CODEX_VERSION': '0.153.4', 'CODEX_BIN': str(args.codex.resolve()),
        'ROBODOJO_SOURCE': runtime+'/src/RoboDojo',
        'OPENPI_SOURCE': runtime+'/src/RoboDojo/XPolicyLab/policy/Pi_05/openpi',
        'ROBODOJO_PYTHON': runtime+'/sim-venv/bin/python',
        'OPENPI_PYTHON': '/mnt/rollout/openpi_mailbox_repro/openpi-venv/bin/python',
        'OPENPI_DATA_HOME': '/mnt/rollout/openpi_mailbox_repro/openpi-cache',
        'CHECKPOINT': runtime+'/checkpoints/RoboDojo-sim-arx_x5-joint-0/59999',
        'DAGGER_GRAPHICS_ROOT': '/home/runner/.local/share/robolab-runtime',
        'POLICY_GPU': '0', 'SIM_GPU': '1', 'POLICY_PORT': '18830', 'SIM_PORT': '19113',
        'MAX_DECISIONS': str(args.max_decisions), 'CODEX_MAX_TOTAL_TOKENS': '0',
        'CODEX_IMAGE_MAX_EDGE': str(args.codex_image_max_edge),
        'ROLLOUT_MAX_SECONDS': str(args.max_seconds), 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
    }
    if method == 'gpt_only':
        env['ROLLOUT_PHASE1_CERTIFICATE'] = str(Path(certificate).resolve())
    if identity['auth_mode'] == 'chatgpt':
        env.pop('ROLLOUT_OPENAI_API_KEY_FILE')
        if https_proxy:
            env['ROLLOUT_HTTPS_PROXY'] = https_proxy
        if https_proxy_file:
            env['ROLLOUT_HTTPS_PROXY_FILE'] = str(Path(https_proxy_file).resolve())
    if args.case_id:
        env['ROLLOUT_CASE_ID'] = args.case_id
    if rerun_of:
        env['ROLLOUT_RERUN_REFERENCE'] = json.dumps(rerun_of, sort_keys=True)
    # ACP permits at most 20 --env entries. Keep every resolved public value
    # visible in the inlined startup script, with proper shell quoting.
    script = ('#!/usr/bin/env bash\n' + '\n'.join(
        'export '+key+'='+shlex.quote(value) for key, value in env.items()) + '\n' +
        (snapshot/'hybrid_rollout/robodojo/acp_startup.sh').read_text())
    argv = [str(args.sco), 'acp', 'jobs', 'create', '--workspace-name='+args.workspace,
        '--aec2-name='+args.cluster, '--job-name='+args.experiment_id+'_r'+str(args.replica_id),
        '--training-framework=pt', '--worker-nodes=1', '--worker-spec='+args.worker_spec,
        '--quota-type='+quota_type, '--priority=NORMAL', '--container-image-url='+args.image,
        '--storage-mount=019908a8-2511-7900-aec9-07b367da1fa1:/mnt/home,0198a752-a7c7-7b85-883f-592d22b03062:/mnt/project',
        '--env=CODE_ROOT:'+str(snapshot), '--command='+script]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), task=args.task,
        evaluation_method=method,
        planned_rollouts=args.count, max_decisions=args.max_decisions,
        context_version=CONTEXT_VERSION,
        rerun_of=rerun_of,
        model=MODEL, effort=EFFORT, auth_profile=identity['name'], provider=identity['provider'],
        auth_mode=identity['auth_mode'], base_url=identity['base_url'], workspace=args.workspace, cluster=args.cluster,
        nodes=1, worker_spec=args.worker_spec, requested_gpus=2, quota=quota_type, image=args.image,
        source_snapshot=str(snapshot), git_base_commit=git_head, source_sha256=source_hash,
        evaluation_panel_sha256=panel['panel_sha256'], evaluation_cases=cases,
        environment=env, command=argv,
        batch_result=f'{runtime}/results/{args.experiment_id}/_replicas/replica_{args.replica_id}/attempt_{args.attempt}/batch.json')
    write_json(root/'submission.json', plan)
    print(json.dumps({'submission': str(root/'submission.json'), 'source_sha256': source_hash,
                      'batch_result': plan['batch_result']}, indent=2))


def submit(path, *, cancelled=None):
    plan = json.loads(path.read_text())
    from .method import evaluation_method
    method = evaluation_method(plan.get('evaluation_method', 'pi05_plus_gpt'))
    if plan['environment'].get('ROLLOUT_EVALUATION_METHOD', 'pi05_plus_gpt') != method:
        raise ValueError('Submission method and environment differ')
    # Credentials may have expired or been removed since a plan was prepared.
    identity_name = plan.get('auth_profile', 'galbot')
    if plan['environment'].get('ROLLOUT_HTTPS_PROXY_FILE'):
        from .proxy_config import read_proxy
        read_proxy(plan['environment']['ROLLOUT_HTTPS_PROXY_FILE'])
    validate_credential(identity_name, plan['environment']['ROLLOUT_SHARED_ROOT'],
                        plan['environment'].get('ROLLOUT_OPENAI_API_KEY_FILE'))
    if plan['environment'].get('ROLLOUT_COUNT') != '1' or not plan['environment'].get('ROLLOUT_CASE_ID'):
        raise ValueError('Re-prepare a single explicit case; old multi-episode plans must not be submitted')
    panel = read_panel(plan['environment']['ROLLOUT_EVAL_MANIFEST'], plan['evaluation_panel_sha256'])
    if method == 'gpt_only':
        from .phase_gate import verify_certificate
        verify_certificate(plan['environment'].get('ROLLOUT_PHASE1_CERTIFICATE'), panel,
                           plan['environment']['ROLLOUT_CASE_ID'])
    verify_assets(panel, plan['environment']['ROBODOJO_SOURCE'], plan['evaluation_cases'])
    if plan['environment'].get('ROLLOUT_CASE_ID'):
        rerun_of = plan.get('rerun_of')
        if json.loads(plan['environment'].get('ROLLOUT_RERUN_REFERENCE', 'null')) != rerun_of:
            raise ValueError('Rerun metadata and environment differ')
        require_available(plan['environment']['ROLLOUT_SHARED_ROOT'], panel,
                          plan['environment']['ROLLOUT_CASE_ID'], rerun_of=rerun_of, method=method)
    snapshot = Path(plan['source_snapshot'])
    manifest = json.loads((snapshot/'source_manifest.json').read_text())
    if any(sha256(snapshot/p) != digest for p, digest in manifest['files'].items()):
        raise ValueError('Prepared source snapshot changed; refusing submission')
    from .acp_query import user_jobs
    jobs = user_jobs(plan['command'][0], plan['workspace'])
    totals = {'SPOT': 0, 'RESERVED': 0}
    active = {'RUNNING', 'STARTING', 'CREATING', 'RESTARTING', 'RECOVERING'}
    for job in jobs:
        if job.get('ownership', {}).get('user_name') != 'sujiayi' or job.get('state') not in active:
            continue
        quota = job.get('scheduling', {}).get('quota_type', 'SPOT')
        totals[quota] = totals.get(quota, 0) + sum(
            int(spec.get('replicas', role.get('total_replicas', 1))) *
            int(spec.get('limits', {}).get('nvidia.com/gpu', 0))
            for role in job.get('roles', []) for spec in role.get('resource_spec', []))
    quota_type = plan['quota']
    if quota_type not in ('spot', 'reserved') or '--quota-type='+quota_type not in plan['command']:
        raise ValueError('Submission quota differs from its reviewed metadata')
    projected = dict(totals)
    projected[quota_type.upper()] += plan['requested_gpus']
    if quota_type == 'reserved' and 'h100' in plan['cluster'].lower() and projected['RESERVED'] > 32:
        raise ValueError('H100 reserved use would exceed the 32-GPU per-user cap')
    preflight = dict(created_utc=datetime.now(timezone.utc).isoformat(), current=totals,
                    requested_quota=quota_type, requested_gpus=plan['requested_gpus'],
                    projected=projected, scope='ACP only; all user pages, 100 jobs per page')
    write_json(path.parent/'gpu_preflight.json', preflight)
    print(json.dumps(preflight), flush=True)
    if cancelled is not None and cancelled():
        raise InterruptedError('Dispatch stopped before create; no request sent')
    reservation = path.parent/'submission_started.json'
    with reservation.open('x') as stream:
        json.dump({'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    refresh(plan['environment']['ROLLOUT_SHARED_ROOT'], plan['environment']['ROLLOUT_EVAL_MANIFEST'], method=method)
    if cancelled is not None and cancelled():
        write_json(path.parent/'submission_cancelled.json', dict(reason='Stopped before create', utc=datetime.now(timezone.utc).isoformat()))
        raise InterruptedError('Dispatch stopped before create; reservation retained for audit')
    result = subprocess.run(plan['command'], text=True, capture_output=True, timeout=180)
    write_json(path.parent/'submission_response.json', dict(returncode=result.returncode,
                                                           stdout=result.stdout, stderr=result.stderr))
    refresh(plan['environment']['ROLLOUT_SHARED_ROOT'], plan['environment']['ROLLOUT_EVAL_MANIFEST'], method=method)
    print(result.stdout)
    if result.returncode:
        print(result.stderr)
    raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--experiment-id', required=True)
    p.add_argument('--evaluation-method', choices=('pi05_plus_gpt', 'gpt_only'), default='pi05_plus_gpt')
    p.add_argument('--phase1-certificate', type=Path)
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--replica-id', type=int, default=6)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--attempt', type=int, default=0)
    p.add_argument('--case-id', required=True, help='Exact frozen case; requires count=1, never default back to layout0')
    p.add_argument('--rerun-v1-archive', type=Path,
                   help='Explicit old native v1 failure archive; permits one linked v2 result only')
    p.add_argument('--eval-manifest', type=Path, required=True)
    p.add_argument('--max-decisions', type=int, default=100)
    p.add_argument('--max-seconds', type=int, default=7200, help='Per-episode wall limit; 0 disables')
    p.add_argument('--codex-image-max-edge', type=int,
                   default=int(os.environ.get('CODEX_IMAGE_MAX_EDGE', DEFAULT_CODEX_IMAGE_MAX_EDGE)))
    p.add_argument('--auth-profile', choices=tuple(PROFILES), default='galbot')
    p.add_argument('--credential-file', type=Path, help='API key file override; managed accounts use their isolated profile')
    p.add_argument('--https-proxy', help='Explicit credential-free HTTPS egress proxy for managed accounts only')
    p.add_argument('--https-proxy-file', type=Path, help='Private authenticated proxy URL; only its path is archived')
    p.add_argument('--codex', type=Path, required=True)
    p.add_argument('--sco', type=Path, default='/home/runner/.sco/bin/sco')
    p.add_argument('--workspace', default='galbot-foundation-model')
    p.add_argument('--cluster', default='galbot-foundation-model-4090')
    p.add_argument('--worker-spec', default='n5lp.nn.a80.2')
    p.add_argument('--quota-type', choices=('spot', 'reserved'), default='spot')
    p.add_argument('--image', required=True)
    p.add_argument('--shared-root', type=Path,
        default='/mnt/rollout/robodojo_mixed_control')
    p = sub.add_parser('submit')
    p.add_argument('submission', type=Path)
    args = parser.parse_args()
    prepare(args) if args.action == 'prepare' else submit(args.submission)


if __name__ == '__main__':
    main()
