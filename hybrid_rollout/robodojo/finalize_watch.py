"""Model-free final delivery after the existing 100-case audit passes.

Separate process/source/output from all rollout daemons. No task submission,
simulation connection, authentication, quota reset, or native-result mutation.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .audit_follow import load_plans, selected_rows
from .campaign import optional, utc
from .io import require, sha256, write_json

PATCHES = ('finalize_watch.py', 'score_table.py')


def verified_source(root, expected):
    """Same digest contract as prepare_input_recovery, without deployment imports.

    Postprocessing snapshots deliberately omit deployment-only preparation code.
    """
    root = Path(root)
    manifest = optional(root/'source_manifest.json')
    files = {str(p.relative_to(root)): sha256(p)
        for p in (root/'hybrid_rollout').rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    require(files and digest == expected and manifest.get('source_sha256') == expected
        and files == manifest.get('files'), 'Frozen source changed')
    return manifest


def prepare(workflow, approved, parent, parent_sha, root, deep, output):
    workflow, parent, root, deep, output = (Path(p).resolve() for p in
        (workflow, parent, root, deep, output))
    plans = load_plans(workflow, approved)
    shared = Path(plans['gpt_only']['shared_root']).resolve()
    require(root.parent == workflow.parent, 'Watcher must have its own campaign subdirectory')
    require(output.parent == shared/'reports', 'Final output must be a new shared reports directory')
    require(not root.exists() and not output.exists(), 'Never overwrite an existing watcher or delivery')
    require(not deep.is_relative_to(shared/'results'), 'Audit report cannot be in original archives')
    base = verified_source(parent, parent_sha)
    frozen = root/'source'
    shutil.copytree(parent/'hybrid_rollout', frozen/'hybrid_rollout',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in PATCHES:
        shutil.copy2(Path(__file__).parent/name, frozen/'hybrid_rollout/robodojo'/name)
    files = {str(p.relative_to(frozen)): sha256(p) for p in frozen.rglob('*') if p.is_file()}
    changed = {p for p in files if files[p] != base['files'].get(p)}
    require(changed == {'hybrid_rollout/robodojo/'+p for p in PATCHES}
        and not set(base['files'])-set(files), 'Unexpected finalizer source diff')
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    write_json(frozen/'source_manifest.json', dict(base, files=files, source_sha256=digest,
        parent_source_sha256=parent_sha, finalization_only=True, workers_unchanged=True))
    config = dict(schema='robodojo.finalize_watch.v1', root=str(root), source=str(frozen),
        source_sha256=digest, workflow=str(workflow), workflow_sha256=approved,
        deep_audit=str(deep), output=str(output), poll_seconds=300,
        rollout_mutations=False, model_calls=0, simulation_runs=0)
    path = root/'config.json'; write_json(path, config)
    return dict(config=str(path), config_sha256=sha256(path), source_sha256=digest)


def config_read(path, digest):
    path = Path(path).resolve()
    require(sha256(path) == digest, 'Finalizer config changed')
    config = optional(path)
    require(config.get('schema') == 'robodojo.finalize_watch.v1'
        and Path(config['root']).resolve() == path.parent, 'Unexpected finalizer config')
    verified_source(Path(config['source']), config['source_sha256'])
    plans = load_plans(Path(config['workflow']), config['workflow_sha256'])
    shared = Path(plans['gpt_only']['shared_root']).resolve()
    require(Path(config['output']).resolve().parent == shared/'reports', 'Unsafe delivery directory')
    require(path.parent.parent == Path(config['workflow']).resolve().parent, 'Unsafe watcher root')
    return config


def readiness(config):
    workflow = Path(config['workflow'])
    plans = load_plans(workflow, config['workflow_sha256'])
    progress = optional(workflow.parent/'progress.json')
    deep = optional(config['deep_audit'])
    rows = selected_rows(progress, plans)
    counts = Counter(r['method'] for r in rows)
    errors = deep.get('errors', [])
    ready = (progress.get('state') == 'complete'
        and all(progress.get(k, {}).get('complete') is True for k in ('hybrid', 'direct'))
        and counts == {'pi05_plus_gpt': 50, 'gpt_only': 50}
        and deep.get('workflow_sha256') == config['workflow_sha256']
        and deep.get('status') == 'passed' and deep.get('selected_count') == 100
        and deep.get('passed_count') == 100 and not errors)
    if ready:
        for row in rows:
            proof = deep.get('rows', {}).get(row['method']+':'+row['case_id'], {})
            require(proof.get('passed') is True and proof.get('archive') == row['archive']
                and proof.get('selected_manifest_sha256') == row['artifact_manifest_sha256'],
                'Deep audit does not cover a selected archive')
    return dict(ready=ready, selected=len(rows), audited=deep.get('passed_count', 0),
        errors=errors, progress=progress, deep=deep, rows=rows)


def derived_videos(rows):
    """Only add missing derived labels; never overwrite an existing rendering."""
    for row in rows:
        root = Path(row['archive'])
        target = None
        if row.get('evaluation_failure_reason'):
            from .adjudicated_archive_audit import adjudicated
            adjudicated(row)  # Exact authorized failure/source evidence, not arbitrary interruptions.
            target = root/'controller/debug_video_idle_timeout_v1'
        else:
            result = optional(root/'controller/result.json')
            meta = optional(root/'controller/debug_video/manifest.json')
            if (result.get('complete') is True and result.get('terminated') is True
                    and result.get('success') is False and not result.get('truncated')
                    and meta.get('terminal_label_version') != 'native_terminal_labels_v2'):
                target = root/'controller/debug_video_labels_v2'
        if target is not None and not target.exists():
            subprocess.run([sys.executable, '-m', 'hybrid_rollout.robodojo.robodojo_server.debug_recorder',
                '--run-root', str(root), '--output', str(target)], check=True, timeout=3600)


def finalize(config, snapshot):
    from .paired_initial_audit import audit_selected
    from .final_debug_export import export as export_debug
    from .score_table import export as export_scores
    require(snapshot['ready'], 'Full 100-case audited result is not ready')
    root = Path(config['root'])
    require(not (root/'attempt.json').exists(), 'Previous finalization attempt requires inspection')
    require(not Path(config['output']).exists(), 'Preserve existing delivery')
    verified_source(Path(config['source']), config['source_sha256'])
    write_json(root/'attempt.json', dict(started_utc=utc(), source_sha256=config['source_sha256']))
    inputs = root/'inputs'; inputs.mkdir()
    # The identical workflow bytes retain their original absolute plan paths.
    # All following stages read this snapshot, not a concurrently refreshed progress file.
    shutil.copy2(config['workflow'], inputs/'workflow.json')
    require(sha256(inputs/'workflow.json') == config['workflow_sha256'], 'Workflow changed while snapshotting')
    write_json(inputs/'progress.json', snapshot['progress'])
    write_json(inputs/'deep.json', snapshot['deep'])
    derived_videos(snapshot['rows'])
    pair_path = root/'paired_initial_final.json'
    paired = audit_selected(inputs/'workflow.json', config['workflow_sha256'])
    write_json(pair_path, paired)
    require(paired['status'] == 'complete_50_pairs' and paired['pairs'] == 50
        and not paired['errors'], 'Full paired initial-state audit did not pass')
    exported = export_debug(inputs/'workflow.json', config['workflow_sha256'],
        inputs/'deep.json', pair_path, Path(config['output']))
    scores = export_scores(Path(config['output'])/'results.json', Path(config['output'])/'score_table')
    result = dict(state='complete', completed_utc=utc(), source_sha256=config['source_sha256'],
        original_workflow=config['workflow'], workflow_sha256=config['workflow_sha256'],
        export=exported, scores=scores, paired_audit_sha256=sha256(pair_path),
        delivery_complete_sha256=sha256(Path(config['output'])/'COMPLETE.json'),
        score_manifest_sha256=sha256(Path(config['output'])/'score_table/manifest.json'),
        model_calls=0, simulation_runs=0, rollout_mutations=False)
    write_json(root/'FINALIZED.json', result)
    return result


def run(path, digest):
    config = config_read(path, digest)
    root = Path(config['root'])
    with (root/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root/'FINALIZED.json').exists(), 'Delivery already finalized; do not replay')
        require(not (root/'attempt.json').exists(), 'Inspect previous finalization attempt before restarting')
        while not (root/'STOP').exists():
            try:
                snap = readiness(config)
                status = dict(updated_utc=utc(), state='waiting',
                    selected=snap['selected'], audited=snap['audited'], errors=snap['errors'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                snap = None
                status = dict(updated_utc=utc(), state='waiting_recheck', error=str(error)[:800])
            if snap is not None and snap['ready']:
                try:
                    status = finalize(config, snap)
                except Exception as error:
                    status = dict(updated_utc=utc(), state='needs_review', error=str(error)[:800])
                    write_json(root/'status.json', status)
                    print(json.dumps(status), flush=True)
                    raise
            write_json(root/'status.json', status)
            print(json.dumps(status), flush=True)
            if status['state'] == 'complete':
                return status
            time.sleep(config['poll_seconds'])


def launch(path, digest):
    config = config_read(path, digest)
    root = Path(config['root'])
    require(not (root/'launch_intent.json').exists(), 'Launch already attempted; inspect before retry')
    tmux = ['tmux', '-L', 'robodojo-v3-overnight']
    require(subprocess.run(tmux+['has-session', '-t', 'finalize-100'], capture_output=True).returncode != 0,
        'A finalizer session already exists')
    args = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH='+config['source'], sys.executable,
        '-u', '-m', 'hybrid_rollout.robodojo.finalize_watch', 'run',
        '--config', str(Path(path).resolve()), '--config-sha256', digest]
    write_json(root/'launch_intent.json', dict(utc=utc(), argv=args, config_sha256=digest))
    command = 'exec '+shlex.join(args)+' >> '+shlex.quote(str(root/'watch.log'))+' 2>&1'
    subprocess.run(tmux+['new-session', '-d', '-s', 'finalize-100', '-c', config['source'], command],
        check=True, timeout=15)
    result = dict(utc=utc(), argv=args, config_sha256=digest, rollout_daemons_unchanged=True)
    write_json(root/'launch.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ('workflow', 'parent-source', 'root', 'deep-audit', 'output'):
        prep.add_argument('--'+name, type=Path, required=True)
    for name in ('approved-sha256', 'parent-source-sha256'):
        prep.add_argument('--'+name, required=True)
    for action in ('run', 'launch'):
        part = sub.add_parser(action)
        part.add_argument('--config', type=Path, required=True)
        part.add_argument('--config-sha256', required=True)
    a = p.parse_args()
    if a.action == 'prepare':
        value = prepare(a.workflow, a.approved_sha256, a.parent_source,
            a.parent_source_sha256, a.root, a.deep_audit, a.output)
    else:
        value = globals()[a.action](a.config, a.config_sha256)
    print(json.dumps(value))
