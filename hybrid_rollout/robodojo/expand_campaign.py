"""Prepare/activate the reviewed 50-case scope plus one v2 repeat per v1 failure."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from . import campaign
from .case_ledger import build_ledger, require_available
from .evaluation import read_panel
from .io import sha256, write_json
from .subscription_migration import _lock

REVISION = 'panel50_context_pairs_v9'


def revise(plan, state, panel, scope, ledger):
    """Preserve existing work indices, append extra cases and conditional comparisons."""
    if plan.get('expansion_revision'):
        raise ValueError('Expansion already planned')
    if ledger['errors']:
        raise ValueError('Resolve ledger errors before expansion')
    new, updated = copy.deepcopy(plan), copy.deepcopy(state)
    rows = {r['case_id']: r for r in ledger['cases']}
    by_id = {c['case_id']: c for c in panel['cases']}
    selected = [cid for t in scope['tasks'] for cid in t['case_ids']]
    original = set(plan['skipped_completed']) | {e['case']['case_id'] for e in plan['queue']}
    if len(selected) != 50 or len(set(selected)) != 50 or len(original) != 30 or not original <= set(selected):
        raise ValueError('Expected original30 plus exactly20 fixed new cases')
    new['case_runtime_overrides'] = copy.deepcopy(plan.get('case_runtime_overrides', {}))
    for i, entry in enumerate(new['queue']):
        entry.pop('slot_id', None)
        entry['priority'] = 1
        entry['evaluation_unit'] = 'primary'
        # Preserve every existing runtime, including retries. New operations do
        # not silently replace the source of an already running container.
        new['case_runtime_overrides'].setdefault(str(i), dict(source_sha256=plan['source_sha256'],
            dispatcher_source=plan['dispatcher_source'], context_version='v2'))
    prefix = plan['experiment_prefix']+'_'+REVISION
    extra = []
    for cid in selected:
        if cid in original:
            continue
        if any(a['state'] in ('completed', 'running_or_unreconciled') for a in rows[cid]['attempts']):
            raise ValueError('Additional case unexpectedly completed/active; reconcile without duplicating')
        entry = dict(case=by_id[cid], initial_attempt=1+max(
            (int(a['attempt']) for a in rows[cid]['attempts']), default=-1),
            experiment_prefix=prefix, priority=2, evaluation_unit='primary')
        new['queue'].append(entry)
        extra.append(cid)
    confirmed, conditional = [], []
    for cid, version in plan['context_case_versions'].items():
        if version != 'v1':
            continue
        completed = [a for a in rows[cid]['attempts'] if a['state'] == 'completed']
        if len(completed) > 1:
            raise ValueError('Unexpected duplicate before paired expansion')
        if completed and completed[0]['native_success'] is True:
            continue
        if completed and completed[0].get('context_version', 'v1') != 'v1':
            raise ValueError('Unexpected context at v1 origin')
        entry = dict(case=by_id[cid], initial_attempt=1+max(
            (int(a['attempt']) for a in rows[cid]['attempts']), default=-1),
            experiment_prefix=prefix, priority=0, evaluation_unit='v1_failure_v2_repeat',
            rerun_from_context='v1')
        if completed:
            entry['rerun_v1_archive'] = completed[0]['archive']
            confirmed.append(cid)
        else:
            conditional.append(cid)
        new['queue'].append(entry)
    new.update(expansion_revision=REVISION, dispatch_mode='work_conserving',
        context_version='v2', primary_case_ids=selected, added_case_ids=extra,
        confirmed_v1_failure_reruns=confirmed, conditional_v1_failure_reruns=conditional,
        primary_target=50, minimum_complete_trajectories=50+len(confirmed),
        maximum_complete_trajectories=50+len(confirmed)+len(conditional),
        initial_submissions=[], routing=dict(rule='First available account claims next eligible shared work item',
            priority='v1 failure comparisons, remaining original primary cases, added20 primary cases',
            immediate_refill_after_reconciliation=True, release_slots_during_transport_backoff=True,
            slot_task_affinity=False, max_active_containers=6),
        authorization='User requested 20 remaining primary cases plus exactly one native-complete v2 rerun for each v1 native failure; retain originals, same seeds, shared first-free-slot queue. Native v2 failures are not repeated.')
    for s in updated['slots']:
        if s['status'] == 'drained':
            s['status'] = 'idle'
    return new, updated


def prepare(path):
    path = path.resolve(); root = path.parent
    old = campaign.optional(path)
    target = root/f'campaign_{REVISION}.json'
    if not (root/'STOP').exists() or target.exists():
        raise ValueError('Pause dispatcher first; require fresh expansion revision')
    with _lock(old):
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != sha256(path):
            raise ValueError('Parent state does not match plan')
        panel = read_panel(old['eval_manifest'], old['panel_sha256'])
        scope = campaign.optional(old['scope_file'])
        if sha256(old['scope_file']) != old['scope_file_sha256']:
            raise ValueError('Scope changed')
        ledger = build_ledger(old['shared_root'], panel)
        new, updated = revise(old, state, panel, scope, ledger)
        for cid in new['added_case_ids']:
            require_available(old['shared_root'], panel, cid)
        for binding in new['case_runtime_overrides'].values():
            source = Path(binding['dispatcher_source'])
            manifest = campaign.optional(source/'source_manifest.json')
            if (manifest['source_sha256'] != binding['source_sha256'] or
                    any(sha256(source/p) != digest for p, digest in manifest['files'].items())):
                raise ValueError('Preserved runtime changed')
        repo = Path(__file__).resolve().parents[2]
        frozen = root/f'source_{REVISION}'
        shutil.copytree(repo/'hybrid_rollout', frozen/'hybrid_rollout',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        files = {str(p.relative_to(frozen)): sha256(p) for p in sorted(frozen.rglob('*')) if p.is_file()}
        digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
        write_json(frozen/'source_manifest.json', dict(git_base_commit=commit,source_sha256=digest,
            includes_reviewed_worktree_changes=True,files=files))
        new.update(parent_plan_sha256=sha256(path), source_sha256=digest, dispatcher_source=str(frozen),
            operations_source=str(frozen), operations_source_sha256=digest,
            expansion_parent_state_sha256=sha256(root/'state.json'), expansion_stop_sha256=sha256(root/'STOP'),
            expansion_slot_stops={p.name:sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()},
            expansion_state_payload_sha256=hashlib.sha256(json.dumps(updated,sort_keys=True).encode()).hexdigest(),
            expansion_prepared_utc=campaign.utc())
        write_json(target,new)
        updated['approved_sha256']=sha256(target)
        write_json(root/f'state_{REVISION}_prepared.json',updated)
        write_json(root/f'evaluation_units_{REVISION}.json',dict(plan_sha256=sha256(target),
            primary_case_ids=new['primary_case_ids'], confirmed_reruns=new['confirmed_v1_failure_reruns'],
            conditional_reruns=new['conditional_v1_failure_reruns'], existing_artifacts_modified=False,
            aggregation='Report original v1 and v2 reruns separately; repeats are paired outcomes, not extra independent cases.'))
        print(json.dumps(dict(plan=str(target),sha256=sha256(target),source_sha256=digest,
            extra20=len(new['added_case_ids']),confirmed_reruns=len(new['confirmed_v1_failure_reruns']),
            conditional_reruns=len(new['conditional_v1_failure_reruns']),created_jobs=0),indent=2))


def activate(path,approved):
    path=path.resolve();root=path.parent;plan=campaign.optional(path)
    if sha256(path)!=approved or plan.get('expansion_revision')!=REVISION:
        raise ValueError('Unreviewed expansion')
    with _lock(plan):
        backup=root/f'state_before_{REVISION}.json'
        if backup.exists() or sha256(root/'state.json')!=plan['expansion_parent_state_sha256']:
            raise ValueError('Parent state changed or already activated')
        if sha256(root/'STOP')!=plan['expansion_stop_sha256']:
            raise ValueError('STOP intent changed')
        actual={p.name:sha256(p) for p in root.glob('STOP_SLOT_*') if p.name[-1:].isdigit()}
        if actual!=plan['expansion_slot_stops']:
            raise ValueError('Slot stop intent changed')
        state=campaign.optional(root/f'state_{REVISION}_prepared.json')
        payload=dict(state,approved_sha256=plan['parent_plan_sha256'])
        if state['approved_sha256']!=approved or hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()!=plan['expansion_state_payload_sha256']:
            raise ValueError('Prepared state changed')
        campaign.validate_topology(plan)
        shutil.copy2(root/'state.json',backup)
        write_json(root/'state.json',state)
        (root/'STOP').rename(root/f'STOP.before_{REVISION}.json')
        print('Expansion activated; start the single new frozen dispatcher. No jobs created here.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('plan',type=Path)
    a=sub.add_parser('activate');a.add_argument('plan',type=Path);a.add_argument('--approved-sha256',required=True)
    args=p.parse_args()
    if args.action=='prepare':prepare(args.plan)
    else:activate(args.plan,args.approved_sha256)


if __name__=='__main__':
    main()
