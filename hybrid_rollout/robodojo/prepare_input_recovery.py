"""Prepare (never activate/create) the reviewed recoverable-input controller.

Copy only an explicit patch set onto the existing frozen operations package.
Keep old workflows, worker snapshots, results, live state and STOPs untouched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from .campaign import optional, utc
from .io import sha256, write_json

REVISION = 'input_recovery_v1'
PATCHES = (
    'hybrid_rollout/robodojo/skill/run.py',
    'hybrid_rollout/robodojo/robodojo_server/video_panel.py',
    'hybrid_rollout/robodojo/robodojo_server/debug_recorder.py',
    'hybrid_rollout/robodojo/test_network_continue.py',
    'hybrid_rollout/robodojo/test_recoverable_tool_input.py',
    'hybrid_rollout/robodojo/test_video_terminal_labels.py',
)
OPS_FILES = {'hybrid_rollout/robodojo/campaign.py',
             'hybrid_rollout/robodojo/input_guard_queue.py',
             'hybrid_rollout/robodojo/overnight.py'}


def verified_source(root, expected):
    manifest = optional(root/'source_manifest.json')
    files = {str(p.relative_to(root)): sha256(p)
             for p in (root/'hybrid_rollout').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if not files or digest != expected or manifest.get('source_sha256') != expected or files != manifest.get('files'):
        raise ValueError('Frozen source changed')
    return manifest


def prepare(workflow_path, approved, operations, operations_sha):
    workflow_path = workflow_path.resolve()
    root = workflow_path.parent
    if sha256(workflow_path) != approved:
        raise ValueError('Unexpected parent workflow')
    workflow = optional(workflow_path)
    plan_path = Path(workflow['phase2_plan'])
    if plan_path.parent != root or sha256(plan_path) != workflow['phase2_sha256']:
        raise ValueError('Unexpected parent plan')
    old = optional(plan_path)
    if (old['evaluation_method'] != 'gpt_only' or old['context_version'] != 'v3'
            or len(old['queue']) != 50 or old['primary_target'] != 50
            or old.get('case_runtime_overrides') or old.get('slot_runtime_overrides')):
        raise ValueError('Require the current unmodified GPT-only fifty-case plan')
    worker = verified_source(Path(workflow['source']), workflow['source_sha256'])
    operations = operations.resolve()
    base = verified_source(operations, operations_sha)
    if (base.get('parent_source_sha256') != worker['source_sha256']
            or set(base['files']) != set(worker['files'])
            or {p for p in base['files'] if base['files'][p] != worker['files'][p]} != OPS_FILES):
        raise ValueError('Unreviewed operations baseline')
    frozen = root/f'source_{REVISION}'
    new_plan = root/f'campaign_{REVISION}.json'
    new_workflow = root/f'workflow_{REVISION}.json'
    if any(p.exists() for p in (frozen, new_plan, new_workflow)):
        raise ValueError('Revision already prepared; never overwrite')
    repo = Path(__file__).resolve().parents[2]
    if not all((repo/p).is_file() for p in PATCHES):
        raise ValueError('Incomplete repair patch')
    shutil.copytree(operations/'hybrid_rollout', frozen/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in PATCHES:
        shutil.copy2(repo/name, frozen/name)
    files = {str(p.relative_to(frozen)): sha256(p) for p in frozen.rglob('*') if p.is_file()}
    changed = {p for p in files if files[p] != base['files'].get(p)}
    if changed != set(PATCHES) or set(base['files']) - set(files):
        raise ValueError('Unexpected repair diff')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    write_json(frozen/'source_manifest.json', dict(base, files=files, source_sha256=digest,
        parent_source_sha256=operations_sha, original_worker_sha256=worker['source_sha256'],
        changed_files=sorted(changed), controller_version='recoverable_tool_input_v1',
        terminal_label_version='native_terminal_labels_v2'))
    plan = copy.deepcopy(old)
    plan.update(dispatcher_source=str(frozen), source_sha256=digest, operations_source_sha256=digest,
        previous_plan=str(plan_path), previous_plan_sha256=sha256(plan_path),
        tool_input_recovery=dict(controller_version='recoverable_tool_input_v1',
            max_recoverable_rejections=None, same_thread=True, validation_unchanged=True,
            active_containers_unchanged=True, apply_to_future_attempts_only=True),
        terminal_label_version='native_terminal_labels_v2')
    if 'controller_error_budget_unchanged' in plan['network_recovery']:
        plan['network_recovery']['controller_error_budget_unchanged'] = False
    write_json(new_plan, plan)
    revised = dict(workflow, phase2_plan=str(new_plan), phase2_sha256=sha256(new_plan),
        source=str(frozen), source_sha256=digest, previous_workflow=str(workflow_path),
        previous_workflow_sha256=approved)
    write_json(new_workflow, revised)
    receipt = dict(prepared_utc=utc(), status='prepared_not_activated', created_jobs=0,
        workflow=str(new_workflow), workflow_sha256=sha256(new_workflow),
        plan=str(new_plan), plan_sha256=sha256(new_plan),
        source=str(frozen), source_sha256=digest, changes_from_operations=sorted(changed),
        inherited_operations_files=sorted(OPS_FILES), active_state_unchanged=True)
    write_json(root/f'prepared_{REVISION}.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('workflow', type=Path)
    p.add_argument('--approved-sha256', required=True)
    p.add_argument('--operations-source', type=Path, required=True)
    p.add_argument('--operations-sha256', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.workflow, a.approved_sha256, a.operations_source, a.operations_sha256), indent=2))
