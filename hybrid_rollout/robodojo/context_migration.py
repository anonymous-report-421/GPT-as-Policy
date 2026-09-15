"""Future-only context cutover; never edit a launched attempt or its retry source."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import campaign
from .case_ledger import build_ledger
from .evaluation import read_panel
from .io import sha256, write_json
# Historical v1 -> v2 migration remains pinned; v3 uses a new independent plan.
CONTEXT_VERSION = 'v2'
from .subscription_migration import _lock

REVISION = 'context_v2_v8'


def revise(plan, state, launched, bindings):
    """Only unlaunched queue entries get a new namespace/source/version."""
    if plan.get('context_revision') or plan.get('case_runtime_overrides'):
        raise ValueError('Require a fresh context cutover, not an implicit second migration')
    if plan.get('slot_runtime_overrides'):
        raise ValueError('Resolve slot-specific sources explicitly before a context cutover')
    new_plan, new_state = copy.deepcopy(plan), copy.deepcopy(state)
    used = {i for s in state['slots'] for i in s.get('assigned', [])}
    pending = {s['index'] for s in state['slots'] if s.get('status') in ('ready', 'submitting', 'submitted', 'running')}
    # Every already-started, nonterminal assignment must have its exact retry source.
    if any(i in launched and str(i) not in bindings for i in pending):
        raise ValueError('Missing immutable runtime binding for a launched case')
    versions = {case: 'v1' for case in plan.get('skipped_completed', [])}
    for i, entry in enumerate(new_plan['queue']):
        version = 'v1' if i in launched else CONTEXT_VERSION
        versions[entry['case']['case_id']] = version
        if version == CONTEXT_VERSION:
            if i in used and i not in pending:
                raise ValueError('Unlaunched case is reserved by a non-pending slot')
            entry['experiment_prefix'] = plan['experiment_prefix']+'_'+REVISION
            # Preserve a prepared-but-unsent attempt number; it has no ACP job.
            for slot in new_state['slots']:
                if slot.get('index') == i and slot.get('status') in ('ready', 'submitting'):
                    slot.update(status='ready', submission=None, job_id=None, platform_state=None)
    new_plan.update(context_revision=REVISION, context_version=CONTEXT_VERSION,
        context_case_versions=versions, case_runtime_overrides=copy.deepcopy(bindings))
    return new_plan, new_state


def prepare(path):
    path = path.resolve(); root = path.parent
    plan = campaign.optional(path)
    target = root/f'campaign_{REVISION}.json'
    if not (root/'STOP').is_file() or target.exists():
        raise ValueError('Require paused dispatcher and fresh revision')
    with _lock(plan):
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != sha256(path):
            raise ValueError('Parent plan/state mismatch')
        panel = read_panel(plan['eval_manifest'], plan['panel_sha256'])
        ledger = build_ledger(plan['shared_root'], panel)
        if ledger['errors']:
            raise ValueError('Resolve ledger errors before cutover')
        rows = {r['case_id']: r for r in ledger['cases']}
        launched = {i for i, e in enumerate(plan['queue']) if rows[e['case']['case_id']]['attempts']}
        bindings = {}
        evidence = []
        for slot in state['slots']:
            i = slot.get('index')
            if i not in launched:
                continue
            sub_path = slot.get('submission')
            if not sub_path:
                sub_path = next((h['submission'] for h in reversed(slot.get('history', []))
                                 if h['index'] == i and h.get('submission')), None)
            if not sub_path:
                raise ValueError('No launched source to preserve')
            sub = campaign.optional(sub_path)
            if sub['evaluation_cases'] != [plan['queue'][i]['case']]:
                raise ValueError('Case identity mismatch; do not infer retry source')
            if sub.get('context_version', 'v1') != 'v1':
                raise ValueError('Only explicit v1-to-v2 migration is supported')
            source = Path(sub['source_snapshot'])
            manifest = campaign.optional(source/'source_manifest.json')
            if (manifest['source_sha256'] != sub['source_sha256'] or
                    any(sha256(source/p) != h for p, h in manifest['files'].items())):
                raise ValueError('Old runtime changed; do not rebind')
            bindings[str(i)] = dict(source_sha256=sub['source_sha256'], dispatcher_source=str(source), context_version='v1')
            evidence.append(dict(index=i, context_version='v1', submission=sub_path,
                submission_sha256=sha256(Path(sub_path)), source_sha256=sub['source_sha256']))
        # Historical completed queue entries remain reserved and are never retried.
        active = {s['index'] for s in state['slots']}
        if any(i not in active and rows[plan['queue'][i]['case']['case_id']]['state'] != 'completed'
               for i in launched):
            raise ValueError('Unassigned historical incomplete case needs explicit source review')
        new_plan, new_state = revise(plan, state, launched, bindings)
        repo = Path(__file__).resolve().parents[2]
        frozen = root/('source_'+REVISION)
        shutil.copytree(repo/'hybrid_rollout', frozen/'hybrid_rollout',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        files = {str(p.relative_to(frozen)): sha256(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        write_json(frozen/'source_manifest.json', dict(git_base_commit=commit, source_sha256=digest,
            includes_reviewed_worktree_changes=True, files=files))
        markers = {p.name: sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()}
        new_plan.update(parent_plan_sha256=sha256(path), source_sha256=digest,
            dispatcher_source=str(frozen), operations_source=str(frozen), operations_source_sha256=digest,
            context_parent_state_sha256=sha256(root/'state.json'), context_stop_sha256=sha256(root/'STOP'),
            context_slot_stops=markers, context_cutover_utc=campaign.utc(), context_preserved_attempts=evidence,
            initial_submissions=[], authorization='User requested concise v2 evaluation context for unlaunched cases only; preserve all launched/completed attempts and seeds.')
        # Previous migration evidence is historical, not new create instructions.
        new_plan['context_parent_initial_submissions'] = new_plan.pop('migration_submissions', [])
        preview_state = copy.deepcopy(new_state)
        for slot in preview_state['slots']:
            if slot['status'] == 'idle':
                campaign.reserve(preview_state, slot, new_plan)
                if slot['status'] == 'ready':
                    new_plan['initial_submissions'].append(str(campaign.ensure_submission(new_plan, slot)))
        new_plan['context_state_payload_sha256'] = hashlib.sha256(json.dumps(new_state, sort_keys=True).encode()).hexdigest()
        write_json(target, new_plan)
        new_state['approved_sha256'] = sha256(target)
        write_json(root/f'state_{REVISION}_prepared.json', new_state)
        write_json(root/f'case_context_versions_{REVISION}.json', dict(
            schema='robodojo.case_context_versions.v1', plan_sha256=sha256(target),
            cases=[dict(case_id=k, context_version=v) for k, v in new_plan['context_case_versions'].items()],
            v2_skill_sha256=sha256(frozen/'hybrid_rollout/robodojo/skill/SKILL.md'),
            existing_results_rewritten=False, old_missing_version_means='v1; source provenance preserved in parent plans'))
        print(json.dumps(dict(plan=str(target), sha256=sha256(target), source_sha256=digest,
            versions=dict(campaign.Counter(new_plan['context_case_versions'].values())), created_jobs=0), indent=2))


def activate(path, approved):
    path = path.resolve(); root = path.parent; plan = campaign.optional(path)
    if sha256(path) != approved or plan.get('context_revision') != REVISION:
        raise ValueError('Unreviewed context plan')
    with _lock(plan):
        backup = root/f'state_before_{REVISION}.json'
        if backup.exists() or sha256(root/'state.json') != plan['context_parent_state_sha256']:
            raise ValueError('Parent state changed or already activated')
        if sha256(root/'STOP') != plan['context_stop_sha256']:
            raise ValueError('Stop intent changed')
        markers = {p.name: sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()}
        if markers != plan['context_slot_stops']:
            raise ValueError('Manual slot stop intent changed')
        state = campaign.optional(root/f'state_{REVISION}_prepared.json')
        payload = dict(state, approved_sha256=plan['parent_plan_sha256'])
        if state['approved_sha256'] != approved or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan['context_state_payload_sha256']:
            raise ValueError('Prepared state changed')
        campaign.validate_topology(plan)
        shutil.copy2(root/'state.json', backup)
        write_json(root/'state.json', state)
        (root/'STOP').rename(root/f'STOP.before_{REVISION}.json')
        # Individual manual STOP markers remain effective, unlike the global migration pause.
        print('Context v2 routing activated; start the new frozen dispatcher once. No jobs created here.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a = sub.add_parser('prepare'); a.add_argument('plan', type=Path)
    a = sub.add_parser('activate'); a.add_argument('plan', type=Path); a.add_argument('--approved-sha256', required=True)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.plan)
    else:
        activate(args.plan, args.approved_sha256)


if __name__ == '__main__':
    main()
