"""Recoverably remove terminal v3 interruptions from the video catalog.

Raw directories move below a hidden directory INSIDE results. Compatibility
symlinks preserve original ledger/artifact paths; the reviewer excludes both
hidden directories and symlinks. Native failures/successes and live jobs stay.
"""
import argparse
from pathlib import Path
import uuid

from . import campaign
from .case_ledger import build_ledger
from .evaluation import read_panel
from .io import sha256, write_json
from .subscription_migration import _lock


def move_recoverably(source, trash):
    source, trash = Path(source), Path(trash)
    if source.is_symlink() or not source.is_dir() or trash.parent != source.parent:
        raise ValueError('Require a real result directory and same-root hidden trash')
    if not trash.name.startswith('.'):
        raise ValueError('Trash must be hidden from the reviewer')
    target = trash/source.name
    if target.exists(): raise ValueError('Never overwrite a retired result')
    link = source.with_name('.retire-link-'+uuid.uuid4().hex)
    link.symlink_to(target, target_is_directory=True)
    try:
        source.rename(target)
        try: link.rename(source)
        except BaseException:
            target.rename(source)
            raise
    finally:
        link.unlink(missing_ok=True)
    return target


def retire(plan_path):
    plan = campaign.optional(plan_path); root = plan_path.resolve().parent
    if plan.get('context_version') != 'v3' or not (root/'STOP').exists():
        raise ValueError('Pause only the v3 dispatcher before retiring terminal attempts')
    shared = Path(plan['shared_root']); results = shared/'results'
    panel = read_panel(plan['eval_manifest'], plan['panel_sha256'])
    with _lock(plan):
        jobs = campaign.job_list(plan)
        ledger = build_ledger(shared, panel)
        if ledger['errors']: raise ValueError('Resolve ledger errors before cleanup')
        rows = []
        for exp in sorted(results.glob(plan['experiment_prefix']+'_c*_a*')):
            if exp.is_symlink() or not exp.is_dir(): continue
            batches = list(exp.glob('_replicas/replica_*/attempt_*/batch.json'))
            if len(batches) != 1: continue
            batch = campaign.optional(batches[0]); episodes = batch.get('episodes', [])
            if batch.get('panel_sha256') != plan['panel_sha256'] or len(episodes) != 1: continue
            archive = Path(episodes[0]['archive'])
            if not archive.is_relative_to(exp): raise ValueError('Unexpected archive path')
            outcome = campaign.optional(archive/'sim/evaluation_outcome.json')
            if outcome.get('complete') is True: continue
            if not (archive/'controller/failure.json').exists(): continue
            if campaign.optional(archive/'controller/run.json').get('context_version') != 'v3': continue
            if (batch.get('state') != 'failed' or not all(batch.get('final_reset', {}).get(k) is True
                    for k in ('ports_released','all_owned_processes_exited'))): continue
            matches = [j for j in jobs if j.get('display_name', '').startswith(exp.name+'_r')]
            if len(matches) != 1 or matches[0]['state'] not in campaign.TERMINAL: continue
            rows.append(dict(source=str(exp), job_id=matches[0]['name'], platform_state=matches[0]['state'],
                archive=str(archive), batch_sha256=sha256(batches[0]),
                failure_sha256=sha256(archive/'controller/failure.json'), case_id=episodes[0]['case_id']))
        trash = results/('.retired_interrupted_v3_'+uuid.uuid4().hex[:12]); trash.mkdir()
        manifest = dict(utc=campaign.utc(), reason='User requested removal of current interrupted v3 trajectories',
            retained_native_complete=True, retained_live_jobs=True, v1_v2_unchanged=True,
            recoverable=True, plan_sha256=sha256(plan_path), status='moving', rows=rows)
        write_json(trash/'retirement_manifest.json', manifest)
        for row in rows:
            row['target'] = str(move_recoverably(Path(row['source']), trash))
            row['moved'] = True
            write_json(trash/'retirement_manifest.json', manifest)
        after = build_ledger(shared, panel)
        if after['errors']: raise ValueError('Unexpected post-retirement ledger errors; preserve manifest')
        for row in rows:
            archive = Path(row['archive'])
            if sha256(archive/'controller/failure.json') != row['failure_sha256']:
                raise ValueError('Failure evidence changed')
        manifest.update(status='complete', ledger_errors=[])
        write_json(trash/'retirement_manifest.json', manifest)
        write_json(root/'interrupted_retirement.json', dict(utc=campaign.utc(), count=len(rows),
            manifest=str(trash/'retirement_manifest.json'), status='complete'))
        print(f'Retired {len(rows)} terminal v3 interruptions; recoverable manifest: {trash}/retirement_manifest.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    retire(p.parse_args().plan)
