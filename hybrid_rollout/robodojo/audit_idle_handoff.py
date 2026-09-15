"""Replace only the model-free observer; never stop/reconfigure rollout daemons."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from .campaign import optional, utc
from .io import require, sha256, write_json
from .prepare_input_recovery import verified_source

ROOT=Path('/mnt/rollout/robodojo_mixed_control/campaigns/robodojo_v3_eef50_20260912_01')
WORKFLOW_SHA='e1a9a015bcc376603b4e3887fe112abdcf56e379352cad6dd0adc4b28aeb4ca5'
OPS_SHA='310427fcb878218e52321344fafd71ea1f7ac2d4e7fa750640413a05bacb26ac'
PATCHES=('archive_audit.py','audit_follow.py','adjudicated_archive_audit.py','paired_initial_audit.py',
         'final_debug_export.py','robodojo_server/debug_recorder.py','robodojo_server/video_panel.py')


def prepare(root=ROOT):
    base=root/'operations_rpc_idle_timeout_v1'
    original=verified_source(base,OPS_SHA)
    output=root/'audits/followup_idle_v1'
    require(not output.exists(),'Never overwrite an observer revision')
    source=output/'source'
    shutil.copytree(base/'hybrid_rollout',source/'hybrid_rollout',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    repo=Path(__file__).resolve().parent
    for relative in PATCHES:
        shutil.copy2(repo/relative,source/'hybrid_rollout/robodojo'/relative)
    files={str(p.relative_to(source)):sha256(p) for p in (source/'hybrid_rollout').rglob('*') if p.is_file()}
    changed={p for p in files if files[p]!=original['files'].get(p)}
    require(changed=={'hybrid_rollout/robodojo/'+r for r in PATCHES},'Unexpected observer diff')
    digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    write_json(source/'source_manifest.json',dict(files=files,source_sha256=digest,parent_source_sha256=OPS_SHA,
        idle_timeout_policy_sha256=original['idle_timeout_policy_sha256'],audit_only=True,workers_unchanged=True))
    result=dict(utc=utc(),source=str(source),source_sha256=digest,workflow_sha256=WORKFLOW_SHA,
        old_observer_output=str(root/'audits/followup/audit100.json'),rollout_operations_unchanged=True)
    write_json(output/'prepared.json',result)
    return result


def activate(root=ROOT):
    output=root/'audits/followup_idle_v1';prep=optional(output/'prepared.json')
    require(not (output/'activation_intent.json').exists(),'Activation already attempted; inspect, never replay')
    require(sha256(root/'workflow_input_recovery_v1.json')==WORKFLOW_SHA,'Workflow changed')
    verified_source(Path(prep['source']),prep['source_sha256'])
    idle=optional(root/'audits/idle_c11_20260913.json')
    require(idle.get('passed') is True and idle.get('native_complete') is False,'Independent idle audit must pass first')
    old=Path(prep['old_observer_output']);pid=892252;p=Path('/proc')/str(pid)
    argv=p.joinpath('cmdline').read_bytes().split(b'\0')
    require(b'--output' in argv and str(old).encode() in argv
        and b'-c' in argv and any(b'hybrid_rollout.robodojo.audit_follow' in a for a in argv)
        and str(root/'workflow.json').encode() in argv,'PID is not the exact old audit observer')
    # Do not interrupt an expensive in-flight audit. A subsequent invocation can
    # retry before the durable activation intent exists.
    require(p.joinpath('wchan').read_text().strip() in ('hrtimer_nanosleep','do_nanosleep'),
        'Old observer is auditing; wait for its inter-poll sleep')
    write_json(output/'activation_intent.json',dict(utc=utc(),old_pid=pid,old_argv_sha256=hashlib.sha256(b'\0'.join(argv)).hexdigest(),
        rollout_containers_stopped=0,rollout_daemons_unchanged=True,source_sha256=prep['source_sha256']))
    os.kill(pid,signal.SIGTERM)
    for _ in range(50):
        if not p.exists():break
        time.sleep(.2)
    require(not p.exists(),'Old audit observer has not exited; do not double-start')
    with old.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shutil.copy2(old,output/'previous_audit_report.json')
    previous=optional(output/'previous_audit_report.json')
    imported=[r for r in previous['rows'].values() if r.get('passed') is True]+[idle]
    require(len({r['archive'] for r in imported})==len(imported),'Duplicate import archive')
    for row in imported:
        require(sha256(Path(row['archive'])/'artifact_manifest.json')==row['artifact_manifest_sha256'],
            'Imported audited manifest changed')
    seed=output/'seed_verified.json'
    write_json(seed,dict(status='passed',rows=imported,created_utc=utc(),
        previous_report_sha256=sha256(output/'previous_audit_report.json'),
        separate_idle_report_sha256=sha256(root/'audits/idle_c11_20260913.json')))
    python=str(root.parents[1]/'sim-venv/bin/python')
    args=['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+prep['source'],python,'-u','-m',
        'hybrid_rollout.robodojo.audit_follow','--workflow',str(root/'workflow_input_recovery_v1.json'),
        '--approved-sha256',WORKFLOW_SHA,'--output',str(output/'audit100.json'),
        '--seed-report',str(seed),'--seed-pid','0','--poll-seconds','300']
    tmux=['tmux','-L','robodojo-v3-overnight']
    require(subprocess.run(tmux+['has-session','-t','audit-follow'],capture_output=True).returncode!=0,
        'Old audit tmux session remains; inspect before launch')
    command='exec '+shlex.join(args)+' >> '+shlex.quote(str(output/'observer.log'))+' 2>&1'
    subprocess.run(tmux+['new-session','-d','-s','audit-follow','-c',prep['source'],command],check=True,timeout=15)
    result=dict(utc=utc(),old_observer_retired=pid,source_sha256=prep['source_sha256'],
        imported_native_rows=len(imported)-1,imported_idle_rows=1,argv=args,
        output=str(output/'audit100.json'),rollout_daemons_unchanged=True,rollout_containers_stopped=0)
    write_json(output/'activation.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','activate'])
    print(json.dumps(globals()[p.parse_args().action](),indent=2))
