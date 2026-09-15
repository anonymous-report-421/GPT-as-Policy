"""Prepare a separate v3 rerun, retaining every earlier result and paired seed.

Preparation is offline and does not authorize dispatch or remove any STOP.
The 60-case identity catalog is retained; the reviewed scope selects exactly 50.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from . import campaign
from .codex_backend.profiles import ACCOUNT_GROUPS, validate_campaign_sessions, POOL15_PROFILES
from .evaluation import canonical_sha256, read_panel, validate_panel, verify_assets, TASKS, GENERALIZATION_TASKS
from .io import sha256, write_json
from .prompt_context import CONTEXT_VERSION


def revised_inputs(panel, scope, panel_id):
    validate_panel(panel)
    if scope.get('source_panel_sha256') != panel['panel_sha256']:
        raise ValueError('Source scope differs from the original panel')
    groups = {t['task']: t['case_ids'] for t in scope['tasks']}
    by_id = {c['case_id']: c for c in panel['cases']}
    selected = [cid for task in TASKS for cid in groups.get(task, [])]
    if set(groups) != set(TASKS) or len(selected) != 50 or len(set(selected)) != 50:
        raise ValueError('Require the same ten tasks with five distinct cases each')
    for task in TASKS:
        cases = [by_id[cid] for cid in groups[task]]
        if len(cases) != 5 or any(c['task'] != task for c in cases):
            raise ValueError('Scope task identity mismatch')
        if task in GENERALIZATION_TASKS and [c['variant'] for c in cases].count('standard') != 2:
            raise ValueError('Preserve two standard and three random cases')
    new = copy.deepcopy(panel)
    new.update(panel_id=panel_id, previous_panel_id=panel['panel_id'],
        previous_panel_sha256=panel['panel_sha256'], context_campaign='v3',
        selection='Same frozen fixtures and 50-case scope as prior evaluation; independent v3 results, no outcome selection')
    new['panel_sha256'] = canonical_sha256({k: v for k, v in new.items() if k != 'panel_sha256'})
    validate_panel(new)
    new_scope = copy.deepcopy(scope)
    new_scope.update(source_panel_sha256=new['panel_sha256'], previous_panel_sha256=panel['panel_sha256'])
    # Round-robin task order; whichever account becomes free claims the next case.
    queue = [dict(case=copy.deepcopy(by_id[groups[t][i]]), initial_attempt=0, priority=1,
                  evaluation_unit='v3_primary') for i in range(5) for t in TASKS]
    return new, new_scope, queue


def prepare(parent_path, prefix):
    if CONTEXT_VERSION != 'v3' or not re.fullmatch(r'robodojo_[A-Za-z0-9_]+', prefix):
        raise ValueError('Require v3 source and an explicit robodojo campaign name')
    parent = campaign.optional(parent_path)
    shared = Path(parent['shared_root']).resolve()
    if not shared.is_relative_to('/mnt/rollout'):
        raise ValueError('Use the requested shared project storage')
    if sha256(parent['scope_file']) != parent['scope_file_sha256']:
        raise ValueError('Original scope changed')
    panel, scope, queue = revised_inputs(read_panel(parent['eval_manifest'], parent['panel_sha256']),
        campaign.optional(parent['scope_file']), prefix+'_fixtures')
    verify_assets(panel, shared/'src/RoboDojo', [e['case'] for e in queue])
    root = shared/'campaigns'/prefix
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'STOP', dict(reason='Prepared only: pending account login, GPU preflight and explicit launch confirmation'))
    write_json(root/'evaluation_manifest.json', panel)
    write_json(root/'scope50.json', scope)
    repo = Path(__file__).resolve().parents[2]
    frozen = root/'source_v3'
    shutil.copytree(repo/'hybrid_rollout', frozen/'hybrid_rollout', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(p.relative_to(frozen)): sha256(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(frozen/'source_manifest.json', dict(git_base_commit=commit, source_sha256=digest,
        includes_reviewed_worktree_changes=True, files=files))
    keys = ('shared_root', 'codex', 'sco', 'image', 'workspace', 'cluster', 'worker_spec', 'quota', 'https_proxy_file')
    plan = {k: parent[k] for k in keys}
    plan.update(schema='hybrid_rollout.robodojo.campaign.v1', preparation_revision='context_v3_pool15',
        created_utc=campaign.utc(), experiment_prefix=prefix, context_version='v3',
        evaluation_method='pi05_plus_gpt', dispatch_mode='work_conserving',
        eval_manifest=str(root/'evaluation_manifest.json'), panel_sha256=panel['panel_sha256'],
        scope_file=str(root/'scope50.json'), scope_file_sha256=sha256(root/'scope50.json'),
        previous_plan=str(Path(parent_path).resolve()), previous_plan_sha256=sha256(parent_path),
        previous_panel_sha256=panel['previous_panel_sha256'],
        source_sha256=digest, dispatcher_source=str(frozen), operations_source_sha256=digest,
        slots=[dict(id=i, auth_profile=p) for i, p in campaign.POOL15_SLOTS.items()],
        account_groups=ACCOUNT_GROUPS, queue=queue, skipped_completed=[], primary_target=50,
        primary_case_ids=[e['case']['case_id'] for e in queue], poll_seconds=300,
        retry_limit=None, retry_only='transient_controller_transport_failure',
        model='gpt-6-astra', effort='xhigh', fast=False, max_decisions=0, max_seconds=0,
        gpus_per_replica=2, max_concurrent_gpus=30,
        quota_protection=dict(a_hold_remaining=30, a_resume_remaining=50, a_stop_remaining=20,
            reset_enabled=False, unknown_quota='hold_new_jobs', b_c_exhausted='hold_until_recovered'),
        authorization='PREPARED ONLY. Requires explicit launch approval after isolated account validation and fresh GPU preflight.',
        reporting='v3 only; retain v1/v2 trajectories and annotations for qualitative clips, never merge their success into v3',
        video_overlays='Existing renderer only; projected correction arrows require separate implementation/validation before final launch packaging')
    campaign.validate_topology(plan)
    write_json(root/'campaign.json', plan)
    write_json(root/'seed_alignment.json', dict(previous_panel_sha256=panel['previous_panel_sha256'],
        panel_sha256=panel['panel_sha256'], same_case_bytes=True, cases=[e['case'] for e in queue]))
    print(json.dumps(dict(plan=str(root/'campaign.json'), sha256=sha256(root/'campaign.json'),
        source_sha256=digest, cases=50, sessions=15, requested_gpus=30, created_jobs=0, stopped=True), indent=2))


def verify_ready(path, approved):
    plan = campaign.optional(path)
    root = Path(path).resolve().parent
    if sha256(path) != approved:
        raise ValueError('Plan changed; explicit review required')
    authorization = campaign.optional(root/'launch_authorization.json')
    if authorization.get('approved_sha256') != approved or authorization.get('accounts_confirmed') is not True:
        raise ValueError('v3 launch is not authorized; finish account onboarding and confirm the exact plan first')
    validate_campaign_sessions(plan['shared_root'], POOL15_PROFILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', type=Path, required=True)
    parser.add_argument('--experiment-prefix', required=True)
    args = parser.parse_args()
    prepare(args.parent_plan, args.experiment_prefix)


if __name__ == '__main__':
    main()
