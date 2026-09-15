"""Unattended, non-overlapping v3 hybrid50 -> v3 EEF-only50 workflow.

Existing hybrid worker snapshots remain unchanged. A frozen operations process
owns refill/retry, quota reserve and one durable B reset authorization across
both phases. Explicit STOPs always win, including after process restarts.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from . import campaign
from .evaluation import read_panel, verify_assets
from .io import sha256, write_json
from .paired_evaluation import select_panel
from .phase_gate import verify_certificate
from .quota_guard import durable
from .two_stage import active_rollouts, dispatcher_unlocked, export_report, release_phase, stop_is_ours


MATCHED_FILES = ('prompt_context.py', 'settings.py', 'evaluation.py', 'run_local.sh',
    'robodojo_server/client.py', 'robodojo_server/validation.py', 'robodojo_server/kinematics.py',
    'robodojo_server/action_edit_kinematics.py', 'robodojo_server/session.py',
    'robodojo_server/server.py', 'robodojo_server/debug_recorder.py',
    'skill/image_preview.py', 'skill/network_recovery.py')


def check_matched_runtime(parent, source):
    old = Path(parent['dispatcher_source'])/'hybrid_rollout/robodojo'
    new = Path(source)/'hybrid_rollout/robodojo'
    hashes = {}
    for relative in MATCHED_FILES:
        if sha256(old/relative) != sha256(new/relative):
            raise ValueError('Unreviewed baseline setting change: '+relative)
        hashes[relative] = sha256(new/relative)
    return hashes


def direct_plan(parent, prefix, directory, source, digest):
    if (parent.get('context_version') != 'v3' or parent.get('evaluation_method') != 'pi05_plus_gpt'
            or len(parent['queue']) != 50 or len(set(parent['primary_case_ids'])) != 50):
        raise ValueError('Require original fifty v3 hybrid cases')
    keys = ('shared_root', 'eval_manifest', 'panel_sha256', 'scope_file', 'scope_file_sha256',
        'slots', 'poll_seconds', 'retry_limit', 'retry_only', 'model', 'effort', 'fast',
        'max_decisions', 'max_seconds', 'gpus_per_replica', 'max_concurrent_gpus',
        'codex', 'sco', 'image', 'workspace', 'cluster', 'worker_spec', 'quota',
        'https_proxy_file', 'preparation_revision', 'quota_protection', 'account_groups',
        'primary_case_ids', 'primary_target', 'dispatch_mode', 'network_recovery')
    plan = {k: copy.deepcopy(parent[k]) for k in keys}
    plan.update(schema=parent['schema'], experiment_prefix=prefix,
        created_utc=campaign.utc(), evaluation_method='gpt_only', context_version='v3',
        action_space='eef_only', source_sha256=digest, operations_source_sha256=digest,
        dispatcher_source=str(source), phase1_certificate=str(directory/'hybrid50_completion.json'),
        queue=[dict(case=copy.deepcopy(e['case']), initial_attempt=0, priority=e.get('priority', 1),
                    evaluation_unit='v3_gpt_only_eef_primary') for e in parent['queue']],
        skipped_completed=[], authorization='User authorized unattended v3 hybrid50 then same-seed '
        'v3 GPT-only EEF50; protect A, exhaust C, at most one additional global B reset.')
    campaign.validate_topology(plan)
    return plan


def prepare(parent_path, prefix):
    parent_path = Path(parent_path).resolve()
    parent = campaign.optional(parent_path)
    panel = read_panel(parent['eval_manifest'], parent['panel_sha256'])
    if sha256(parent['scope_file']) != parent['scope_file_sha256']:
        raise ValueError('Frozen scope changed')
    verify_assets(panel, Path(parent['shared_root'])/'src/RoboDojo', [e['case'] for e in parent['queue']])
    if not prefix.startswith('robodojo_') or not all(c.isalnum() or c == '_' for c in prefix):
        raise ValueError('Invalid unique workflow name')
    root = Path(parent['shared_root'])/'campaigns'/prefix
    root.mkdir(exist_ok=False)
    source = root/'source'
    repo = Path(__file__).resolve().parents[2]
    shutil.copytree(repo/'hybrid_rollout', source/'hybrid_rollout',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    write_json(source/'source_manifest.json', dict(files=files, source_sha256=digest,
        git_base_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        includes_reviewed_worktree_changes=True))
    matched = check_matched_runtime(parent, source)
    direct = direct_plan(parent, prefix, root, source, digest)
    write_json(root/'campaign.json', direct)
    write_json(root/'launch_authorization.json', dict(approved_sha256=sha256(root/'campaign.json'),
        accounts_confirmed=True, scope='User explicitly authorized same-setting v3 EEF-only fifty after hybrid fifty',
        utc=campaign.utc()))
    quota = dict(schema='robodojo.unattended_quota.v1', panel_sha256=parent['panel_sha256'],
        campaigns=[parent['experiment_prefix'], prefix], b_max_resets=1,
        a_hold_remaining=30, a_stop_remaining=20, a_resume_remaining=50,
        reset_budget=str(Path(parent['shared_root'])/'evaluation'/f'{prefix}_b_reset_budget.json'),
        retry_verified_input_guards=True, retry_scope='Native-incomplete only; same cases and seeds; '
        '300s exponential backoff capped at3600s, subject to account budget and manual STOP',
        stop_file=str(root/'STOP'), authorization_utc=campaign.utc())
    write_json(root/'quota_policy.json', quota)
    workflow = dict(schema='robodojo.overnight_v3.v1', created_utc=campaign.utc(),
        phase1_plan=str(parent_path), phase1_sha256=sha256(parent_path),
        phase2_plan=str(root/'campaign.json'), phase2_sha256=sha256(root/'campaign.json'),
        quota_policy=str(root/'quota_policy.json'), quota_sha256=sha256(root/'quota_policy.json'),
        source=str(source), source_sha256=digest, poll_seconds=300,
        phase1_target=50, phase2_target=50, no_v1_v2_reuse=True)
    write_json(root/'workflow.json', workflow)
    write_json(root/'STOP', dict(reason='Prepared only; activate supervisor after offline tests and handoff'))
    write_json(root/'setting_alignment.json', dict(case_bytes_identical=True,
        matched_runtime_file_sha256=matched,
        hybrid_cases=[e['case'] for e in parent['queue']], direct_cases=[e['case'] for e in direct['queue']],
        common_context_file_sha256=sha256(source/'hybrid_rollout/robodojo/prompt_context.py'),
        differences=['Remove pi05 runtime, proposals and proposal-assessment gate',
            'GPT outputs bounded dual-arm EEF targets directly; no joint mode',
            'EEF chunk 1..5 steps; hybrid pi05 chunk can be1..15; no equal decision/token budget'],
        unchanged=['native evaluator and horizons', 'fixture/seed', 'v3 task semantics and arm-return guidance',
            'gpt-6-astra/xhigh, Fast off', '480px teacher previews, original raw observations',
            'EEF/IK constraints and native ACK recording', 'same-thread network Continue20x10seconds']))
    return dict(workflow=str(root/'workflow.json'), sha256=sha256(root/'workflow.json'),
        source_sha256=digest, quota_sha256=workflow['quota_sha256'], phase2_sha256=workflow['phase2_sha256'])


def selection(plan, jobs):
    selected = select_panel(plan['shared_root'], read_panel(plan['eval_manifest'], plan['panel_sha256']),
        plan['primary_case_ids'], plan['evaluation_method'], jobs=jobs, context_version='v3',
        experiment_prefix=plan['experiment_prefix'])
    from .idle_timeout_policy import include_adjudicated
    return include_adjudicated(plan, selected, jobs)


def launch_dispatcher(workflow, phase, directory):
    plan = Path(workflow[f'phase{phase}_plan'])
    argv = [os.sys.executable, '-u', '-m', 'hybrid_rollout.robodojo.campaign', 'run', str(plan),
        '--approved-sha256', workflow[f'phase{phase}_sha256'],
        '--dispatcher-source-sha256', workflow['source_sha256'], '--poll-seconds', '30',
        '--retry-input-guards-once', '--recover-verified-guards',
        '--quota-policy', workflow['quota_policy'], '--quota-policy-sha256', workflow['quota_sha256']]
    with (directory/f'phase{phase}_dispatcher.log').open('a') as log:
        proc = subprocess.Popen(argv, cwd=workflow['source'], stdout=log, stderr=log,
            env=dict(os.environ, PYTHONPATH=workflow['source'], PYTHONDONTWRITEBYTECODE='1'))
    return proc


def cycle(workflow, digest, root):
    if (root/'STOP').exists():
        return dict(state='operator_stopped')
    first_path, second_path = Path(workflow['phase1_plan']), Path(workflow['phase2_plan'])
    for path, key in ((first_path, 'phase1_sha256'), (second_path, 'phase2_sha256'),
                      (Path(workflow['quota_policy']), 'quota_sha256')):
        if sha256(path) != workflow[key]:
            raise ValueError('Workflow input changed')
    first, second = campaign.optional(first_path), campaign.optional(second_path)
    old_stop = first_path.parent/'STOP'
    if old_stop.exists() and not stop_is_ours(old_stop, digest):
        return dict(state='operator_stopped_phase1')
    jobs = campaign.job_list(first)
    hybrid = selection(first, jobs)
    progress = dict(updated_utc=campaign.utc(), hybrid=hybrid, state='running_hybrid50')
    if hybrid['errors']:
        return dict(progress, state='hold_invalid_hybrid_evidence')
    if not hybrid['complete']:
        return dict(progress, phase=1)
    if any(not j.get('display_name', '').startswith(second['experiment_prefix']+'_')
           for j in active_rollouts(jobs)):
        # Native completion precedes container teardown. Never overlap stages.
        return dict(progress, state='waiting_native_complete_containers_exit')
    panel = read_panel(first['eval_manifest'], first['panel_sha256'])
    cert = Path(second['phase1_certificate'])
    if not cert.exists():
        durable(cert, dict(schema='robodojo.hybrid50_completion.v3', panel_sha256=panel['panel_sha256'],
            created_utc=campaign.utc(), selection=hybrid, workflow_sha256=digest))
    verify_certificate(cert, panel)
    if not release_phase(first_path.parent, digest):
        return dict(progress, state='operator_stopped_phase1')
    quota_state = second_path.parent/'quota_guard_state.json'
    if not quota_state.exists():
        before = campaign.optional(first_path.parent/'quota_guard_state.json')
        durable(quota_state, dict(schema='robodojo.pool15_quota.v1',
            panel_sha256=panel['panel_sha256'], stops={}, a_held=before.get('a_held', False),
            inherited_from=str(first_path.parent/'quota_guard_state.json')))
    direct = selection(second, jobs)
    progress.update(direct=direct, state='running_gpt_only_eef50')
    if direct['errors']:
        return dict(progress, state='hold_invalid_direct_evidence')
    if direct['complete']:
        export_report(root, hybrid, direct)
        durable(root/'COMPLETE.json', dict(completed_utc=campaign.utc(), hybrid=50, direct=50,
            hybrid_native_complete=hybrid.get('native_complete_count', hybrid['selected_count']),
            direct_native_complete=direct.get('native_complete_count', direct['selected_count']),
            direct_adjudicated_failures=direct.get('adjudicated_failure_count', 0),
            comparison=str(root/'comparison/results.json')))
        return dict(progress, state='complete')
    return dict(progress, phase=2)


def operations_workflow(workflow, approved_source=None):
    """Keep workflow/worker identities while explicitly upgrading operations."""
    if approved_source is None:
        return workflow
    source = Path(__file__).resolve().parents[2]
    manifest = campaign.optional(source/'source_manifest.json')
    parent = campaign.optional(Path(workflow['source'])/'source_manifest.json')
    allowed = {'hybrid_rollout/robodojo/input_guard_queue.py',
               'hybrid_rollout/robodojo/overnight.py'}
    allowed_revisions = (allowed, allowed | {'hybrid_rollout/robodojo/campaign.py'})
    files, original = manifest.get('files', {}), parent.get('files', {})
    changed = {name for name in files if files[name] != original.get(name)}
    idle_changes = {'hybrid_rollout/robodojo/campaign.py', 'hybrid_rollout/robodojo/overnight.py',
                    'hybrid_rollout/robodojo/paired_evaluation.py', 'hybrid_rollout/robodojo/idle_timeout_policy.py',
                    'hybrid_rollout/robodojo/two_stage.py'}
    idle_override = (changed == idle_changes and not set(original)-set(files)
        and set(files)-set(original) == {'hybrid_rollout/robodojo/idle_timeout_policy.py'}
        and manifest.get('idle_timeout_policy_sha256') == sha256(Path(workflow['phase2_plan']).parent/'rpc_idle_timeout_policy.json')) if changed == idle_changes else False
    if (not approved_source or campaign.source_digest() != approved_source
            or manifest.get('source_sha256') != approved_source
            or manifest.get('parent_source_sha256') != workflow['source_sha256']
            or parent.get('source_sha256') != workflow['source_sha256']
            or not files or (not idle_override and (set(files) != set(original) or changed not in allowed_revisions))
            or any(sha256(Path(workflow['source'])/name) != digest for name, digest in original.items())):
        raise ValueError('Unreviewed operations-only source override')
    return dict(workflow, source=str(source), source_sha256=approved_source)


def run(path, approved, operations_source_sha256=None):
    path = Path(path).resolve()
    if sha256(path) != approved:
        raise ValueError('Workflow hash mismatch')
    workflow = campaign.optional(path)
    root = path.parent
    if workflow.get('schema') != 'robodojo.overnight_v3.v1':
        raise ValueError('Wrong workflow')
    operations = operations_workflow(workflow, operations_source_sha256)
    child = None
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (root/'STOP').exists():
            if sha256(path) != approved or campaign.source_digest() != operations['source_sha256']:
                raise ValueError('Frozen supervisor changed')
            try:
                progress = cycle(workflow, approved, root)
                phase = progress.get('phase')
                if child is not None and child.poll() is not None:
                    progress['last_dispatcher_exit'] = child.returncode
                    child = None
                shared = campaign.optional(workflow['phase1_plan'])['shared_root']
                if phase and child is None and dispatcher_unlocked(shared):
                    directory = Path(workflow[f'phase{phase}_plan']).parent
                    if not (directory/'STOP').exists() and not (root/'STOP').exists():
                        child = launch_dispatcher(operations, phase, root)
                        progress.update(dispatcher_pid=child.pid, launched_phase=phase)
                progress['operations_source_sha256'] = operations['source_sha256']
            except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
                # Read/network/storage failures do not authorize duplicate create.
                progress = dict(updated_utc=campaign.utc(), state='waiting_recheck',
                    error_type=type(error).__name__, error=str(error)[:500])
            durable(root/'progress.json', progress)
            print(json.dumps({k:v for k,v in progress.items() if k not in ('hybrid','direct')}), flush=True)
            if progress['state'] in ('complete', 'operator_stopped', 'operator_stopped_phase1'):
                return
            for _ in range(workflow['poll_seconds']):
                if (root/'STOP').exists():
                    return
                time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('parent_plan', type=Path)
    prep.add_argument('--prefix', required=True)
    start = sub.add_parser('run')
    start.add_argument('workflow', type=Path)
    start.add_argument('--approved-sha256', required=True)
    start.add_argument('--operations-source-sha256', help='Explicitly approved operations-only snapshot')
    args = p.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(args.parent_plan, args.prefix), indent=2))
    else:
        run(args.workflow, args.approved_sha256, args.operations_source_sha256)


if __name__ == '__main__':
    main()
