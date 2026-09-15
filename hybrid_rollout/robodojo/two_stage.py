"""Durable mixed50 -> GPT-only50 workflow; no paid probe and no overlapping stages."""
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
from .case_ledger import build_ledger
from .evaluation import read_panel
from .io import sha256, write_json
from .paired_evaluation import select_panel, comparison
from .phase_gate import verify_certificate


def stage2_plan(parent, panel, prefix, root, source, digest):
    """Copy only shared settings, not any stage1 assignments/results/exemptions."""
    keys = ('shared_root', 'eval_manifest', 'panel_sha256', 'scope_file', 'scope_file_sha256',
        'slots', 'poll_seconds', 'retry_base_seconds', 'retry_max_seconds', 'retry_limit',
        'retry_only', 'model', 'effort', 'fast', 'max_decisions', 'max_seconds',
        'gpus_per_replica', 'max_concurrent_gpus', 'codex', 'sco', 'image', 'workspace',
        'cluster', 'worker_spec', 'quota', 'https_proxy_file')
    plan = {k: copy.deepcopy(parent[k]) for k in keys}
    ids = parent['primary_case_ids']
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError('Need the frozen 50-case scope')
    cases = {c['case_id']: c for c in panel['cases']}
    # Short tasks first exercise the exact production pipeline, not an extra
    # paid smoke case. All slots still claim from one shared queue.
    priority = {'fold_clothes': 0, 'put_bottles_into_dustbin': 1, 'build_tower': 2}
    plan.update(schema='hybrid_rollout.robodojo.campaign.v1', created_utc=campaign.utc(),
        experiment_prefix=prefix, evaluation_method='gpt_only', context_version='v2',
        primary_case_ids=list(ids), primary_target=50, skipped_completed=[],
        dispatch_mode='work_conserving', source_sha256=digest, dispatcher_source=str(source),
        operations_source=str(source), operations_source_sha256=digest,
        phase1_certificate=str(root/'hybrid50_completion.json'),
        queue=[dict(case=cases[cid], initial_attempt=0, evaluation_unit='primary',
                    priority=priority.get(cases[cid]['task'], 3)) for cid in ids],
        initial_submissions=[], routing=dict(slot_task_affinity=False,
            immediate_refill_after_reconciliation=True, release_slots_during_transport_backoff=True,
            rule='First free eligible slot claims next shared case; same seeds on transport retry'),
        authorization='User requested GPT-only all50 after mixed50, same cases/seeds/settings, '
            'no pi05, full artifacts, unattended transport retry, B global at most2 resets, protect A quota.')
    campaign.validate_topology(plan)
    return plan


def prepare(args):
    parent_path = args.phase1_plan.resolve()
    parent = campaign.optional(parent_path)
    if sha256(parent_path) != args.phase1_sha256:
        raise ValueError('Phase1 plan changed')
    panel = read_panel(parent['eval_manifest'], parent['panel_sha256'])
    if sha256(parent['scope_file']) != parent['scope_file_sha256']:
        raise ValueError('Scope changed')
    ids = parent['primary_case_ids']
    ledger = build_ledger(parent['shared_root'], panel)
    exemptions = [r['case_id'] for r in ledger['cases'] if r['case_id'] in ids and any(
        a['state'] == 'completed' and a.get('context_version', 'v1') == 'v1'
        and a['native_success'] is True for a in r['attempts'])]
    if ledger['errors'] or len(exemptions) != 8:
        raise ValueError('Frozen eight v1-success exemptions changed')
    root = Path(parent['shared_root'])/'campaigns'/args.experiment_prefix
    root.mkdir()  # Never overwrite a prepared workflow or running snapshot.
    source = root/'source_gpt_only_v1'
    repo = Path(__file__).resolve().parents[2]
    shutil.copytree(repo/'hybrid_rollout', source/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    base = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(source/'source_manifest.json', dict(files=files, source_sha256=digest,
        git_base_commit=base, includes_reviewed_worktree_changes=True))
    plan = stage2_plan(parent, panel, args.experiment_prefix, root, source, digest)
    write_json(root/'campaign.json', plan)
    quota = campaign.optional(args.phase1_quota_policy)
    quota.update(recovery_source=str(source), recovery_source_sha256=digest)
    write_json(root/'quota_policy.json', quota)
    workflow = dict(schema='robodojo.two_stage_workflow.v1', created_utc=campaign.utc(),
        phase1_plan=str(parent_path), phase1_plan_sha256=sha256(parent_path),
        phase2_plan=str(root/'campaign.json'), phase2_plan_sha256=sha256(root/'campaign.json'),
        phase2_quota_policy=str(root/'quota_policy.json'),
        phase2_quota_policy_sha256=sha256(root/'quota_policy.json'), source=str(source), source_sha256=digest,
        poll_seconds=300, exempt_v1_successes=exemptions, phase1_complete_target=50,
        phase2_complete_target=50, total_account_B_reset_cap=2,
        reset_budget=str(Path(parent['shared_root'])/'evaluation/codex_b_reset_budget.json'),
        certificate=str(root/'hybrid50_completion.json'), all_originals_preserved=True)
    write_json(root/'workflow.json', workflow)
    print(json.dumps(dict(workflow=str(root/'workflow.json'), sha256=sha256(root/'workflow.json'),
                         plan_sha256=sha256(root/'campaign.json'), source_sha256=digest, created_jobs=0), indent=2))


def active_rollouts(jobs):
    return [j for j in jobs if j.get('ownership', {}).get('user_name') == 'sujiayi'
        and j.get('display_name', '').startswith('robodojo_')
        and j.get('state') not in campaign.TERMINAL | {'SUSPENDED'}]


def stop_is_ours(path, digest):
    return path.exists() and campaign.optional(path).get('workflow_sha256') == digest


def release_phase(directory, workflow_sha):
    """After no live rollout remains, request dispatcher exit without touching data."""
    path = directory/'STOP'
    if path.exists():
        return stop_is_ours(path, workflow_sha)
    # Exclusive creation never overwrites a simultaneous operator STOP.
    try:
        with path.open('x') as stream:
            json.dump(dict(reason='Verified evaluation phase complete', workflow_sha256=workflow_sha,
                           utc=campaign.utc()), stream)
            stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        return stop_is_ours(path, workflow_sha)
    return True


def dispatcher_unlocked(shared):
    with (Path(shared)/'evaluation/campaign_dispatch.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def export_report(root, hybrid, direct):
    import csv
    import zipfile
    report = comparison(hybrid, direct)
    target = root/'comparison'
    target.mkdir(exist_ok=True)
    # Snapshot is only produced after both full panels are verified.
    write_json(target/'results.json', report)
    lines = ['# Paired RoboDojo evaluation', '',
        f"Hybrid: {report['hybrid_successes']}/50; GPT-only: {report['gpt_only_successes']}/50.", '',
        '| Task | Hybrid | GPT-only | Hybrid mean score | GPT-only mean score |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['task']} | {r['hybrid_successes']}/5 | {r['gpt_only_successes']}/5 | {r['hybrid_mean_score']} | {r['gpt_only_mean_score']} |" for r in report['per_task']]
    lines += ['', report['caveat'], '', report.get('score_note', ''), '',
              f"GPT-only native-complete: {report.get('direct_native_complete_count', 50)}; "
              f"adjudicated idle failures: {report.get('direct_adjudicated_failure_count', 0)}.",
              '', report['initial_observation_note'], '']
    (target/'README.md').write_text('\n'.join(lines))
    with (target/'paired_cases.csv').open('w', newline='') as stream:
        fields = ['case_id', 'evaluation_case', 'hybrid_context', 'hybrid_success', 'gpt_only_success',
                  'hybrid_score', 'gpt_only_score', 'hybrid_archive', 'gpt_only_archive',
                  'gpt_only_evaluation_success', 'gpt_only_failure_reason', 'gpt_only_native_complete']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        other = {r['case_id']: r for r in direct['cases']}
        for left in hybrid['cases']:
            right = other[left['case_id']]
            writer.writerow(dict(case_id=left['case_id'], evaluation_case=json.dumps(left['evaluation_case'], sort_keys=True),
                hybrid_context=left['context_version'], hybrid_success=left['native_success'],
                gpt_only_success=right['native_success'], hybrid_score=left['native_score'],
                gpt_only_score=right['native_score'], hybrid_archive=left['archive'], gpt_only_archive=right['archive'],
                gpt_only_evaluation_success=right.get('evaluation_success', right['native_success']),
                gpt_only_failure_reason=right.get('evaluation_failure_reason'),
                gpt_only_native_complete=right.get('native_complete', True)))
    videos = target/'debug_videos'; videos.mkdir(exist_ok=True)
    index = []
    for selection in (hybrid, direct):
        for row in selection['cases']:
            original = Path(row['archive'])/'controller/debug_video/debug_rollout.mp4'
            dest = videos/f"{row['method']}__{row['case_id']}__{row['context_version']}.mp4"
            if not original.is_file():
                raise ValueError('Verified case has no debug video')
            if not dest.exists():
                shutil.copy2(original, dest)
            if sha256(original) != sha256(dest):
                raise ValueError('Debug video export hash mismatch')
            index.append(dict(case_id=row['case_id'], method=row['method'], context_version=row['context_version'],
                source=str(original), file=dest.name, sha256=sha256(dest)))
    write_json(videos/'manifest.json', dict(videos=index, original_files_untouched=True))
    archive = target/'debug_videos.zip'
    if not archive.exists():
        temporary = target/'debug_videos.partial.zip'
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as stream:
            for path in sorted(videos.iterdir()):
                stream.write(path, path.name)
        with zipfile.ZipFile(temporary) as stream:
            if stream.testzip() is not None:
                raise ValueError('Debug ZIP failed CRC validation')
        temporary.rename(archive)
    write_json(target/'COMPLETE.json', dict(utc=campaign.utc(), result_sha256=sha256(target/'results.json'),
        videos=100, zip_sha256=sha256(archive)))
    return report


def cycle(config, digest, root, child):
    """One read-only progress check, or authorized stage handoff when ready."""
    first_path, second_path = Path(config['phase1_plan']), Path(config['phase2_plan'])
    if (sha256(first_path) != config['phase1_plan_sha256'] or
            sha256(second_path) != config['phase2_plan_sha256']):
        raise ValueError('Frozen workflow plan changed')
    first, second = campaign.optional(first_path), campaign.optional(second_path)
    if campaign.source_digest() != config['source_sha256']:
        raise ValueError('Workflow source changed')
    panel = read_panel(first['eval_manifest'], first['panel_sha256'])
    if sha256(first['scope_file']) != first['scope_file_sha256']:
        raise ValueError('Frozen scope changed')
    for directory in (first_path.parent, second_path.parent):
        stop = directory/'STOP'
        if stop.exists() and not stop_is_ours(stop, digest):
            return child, dict(state='paused_by_operator', utc=campaign.utc())
    jobs = campaign.job_list(first)
    hybrid = select_panel(first['shared_root'], panel, first['primary_case_ids'], 'pi05_plus_gpt',
                          config['exempt_v1_successes'], jobs=jobs)
    progress = dict(utc=campaign.utc(), state='waiting_for_mixed50', hybrid=hybrid,
                    active_jobs=[j['name'] for j in active_rollouts(jobs)])
    if not hybrid['complete']:
        return child, progress
    certificate = Path(config['certificate'])
    if not certificate.exists():
        if active_rollouts(jobs):
            return child, progress
        write_json(certificate, dict(schema='robodojo.hybrid50_completion.v1', utc=campaign.utc(),
            panel_sha256=panel['panel_sha256'], workflow_sha256=digest, selection=hybrid))
    verify_certificate(certificate, panel)
    if not release_phase(first_path.parent, digest):
        return child, dict(progress, state='paused_by_operator')
    direct = select_panel(first['shared_root'], panel, first['primary_case_ids'], 'gpt_only', jobs=jobs)
    progress.update(state='running_gpt_only50', direct=direct)
    if direct['complete'] and not active_rollouts(jobs):
        release_phase(second_path.parent, digest)
        export_report(root, hybrid, direct)
        return child, dict(progress, state='complete')
    if child is not None and child.poll() is None:
        return child, progress
    # A restarted waiter must not create a duplicate dispatcher. Global file
    # lock plus immutable submission intents make this idempotent.
    if not dispatcher_unlocked(first['shared_root']):
        return child, progress
    other = [j for j in active_rollouts(jobs) if not j['display_name'].startswith(second['experiment_prefix']+'_')]
    if other:
        return child, dict(progress, state='waiting_for_old_containers')
    quota_state = root/'quota_guard_state.json'
    if not quota_state.exists():
        previous = campaign.optional(first_path.parent/'quota_guard_state.json')
        write_json(quota_state, dict(schema='robodojo.quota_guard_state.v1',
            policy_sha256=config['phase2_quota_policy_sha256'], b_reset_attempts=[], stops={},
            a_held=previous.get('a_held', False)))
    for slot in second['slots']:
        original_stop = first_path.parent/f"STOP_SLOT_{slot['id']}"
        inherited_stop = root/original_stop.name
        if original_stop.exists() and not inherited_stop.exists():
            shutil.copy2(original_stop, inherited_stop)
    argv = [os.sys.executable, '-u', '-m', 'hybrid_rollout.robodojo.campaign', 'run', str(second_path),
        '--approved-sha256', config['phase2_plan_sha256'], '--dispatcher-source-sha256', config['source_sha256'],
        '--quota-policy', config['phase2_quota_policy'], '--quota-policy-sha256', config['phase2_quota_policy_sha256']]
    with (root/'dispatcher.log').open('a') as log:
        child = subprocess.Popen(argv, cwd=config['source'], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            env=dict(os.environ, PYTHONPATH=config['source'], PYTHONDONTWRITEBYTECODE='1'))
    write_json(root/'phase2_activation.json', dict(utc=campaign.utc(), argv=argv, pid=child.pid,
        workflow_sha256=digest, certificate_sha256=sha256(certificate)))
    return child, progress


def run(args):
    if sha256(args.workflow) != args.approved_sha256:
        raise ValueError('Workflow review hash mismatch')
    config = campaign.optional(args.workflow); root = args.workflow.resolve().parent
    with (root/'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = None
        while not (root/'STOP_WORKFLOW').exists():
            if sha256(args.workflow) != args.approved_sha256:
                raise ValueError('Workflow changed')
            try:
                child, status = cycle(config, args.approved_sha256, root, child)
            except (OSError, subprocess.SubprocessError) as error:
                status = dict(utc=campaign.utc(), state='network_or_io_retry_next_poll', error_type=type(error).__name__)
            except ValueError as error:
                # Do not submit against incomplete inventory or conflicting
                # identities. Preserve diagnostic; next poll may resolve state.
                status = dict(utc=campaign.utc(), state='validation_hold', error=str(error))
            write_json(root/'workflow_status.json', status)
            print(json.dumps({k: status[k] for k in ('utc', 'state')}), flush=True)
            if status['state'] == 'complete':
                return
            for _ in range(config['poll_seconds']):
                if (root/'STOP_WORKFLOW').exists():
                    return
                time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='action', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--phase1-plan', type=Path, required=True)
    q.add_argument('--phase1-sha256', required=True)
    q.add_argument('--phase1-quota-policy', type=Path, required=True)
    q.add_argument('--experiment-prefix', required=True)
    q = sub.add_parser('run'); q.add_argument('workflow', type=Path); q.add_argument('--approved-sha256', required=True)
    args = p.parse_args(); prepare(args) if args.action == 'prepare' else run(args)


if __name__ == '__main__':
    main()
