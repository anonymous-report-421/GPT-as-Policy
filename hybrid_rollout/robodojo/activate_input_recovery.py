"""One-shot handoff of the explicitly confirmed input_recovery_v1 package.

Never calls cluster mutation APIs or signals a container. STOP only releases
the two external daemon locks; submitted jobs retain their source and identity.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil

from . import campaign
from .io import sha256, write_json
from .prepare_input_recovery import verified_source

ROOT = Path('/mnt/rollout/robodojo_mixed_control/campaigns/robodojo_v3_eef50_20260912_01')
REASON = 'approved_input_recovery_v1_handoff'
OLD_WORKFLOW = '97b096d80e34191118a9c0bcac40b79969cf7cf4b19f1292e296ec105a4fd16e'
NEW_WORKFLOW = 'e1a9a015bcc376603b4e3887fe112abdcf56e379352cad6dd0adc4b28aeb4ca5'
OLD_PLAN = '4d374b144d50bed2ce36fe1d97bec55f79f1bf52fe82b0edfe4afea38d90af7e'
NEW_PLAN = 'f83775883343bef355cf66f60c5c3856590d85789b2ad21208d45cd28fa1bb41'
NEW_SOURCE = '909b3ae5629d0790da215d92cb86c68fc3971ddaa1536fb940a188d63c160c0c'
OPS_SOURCE = '80f9b8547f615729a6697192dd18a033dc8d5fe149bd88fb6cfaf94cfa5dfe6c'
QUOTA = '5fc3f4123c7c943be13407cfec77996c430b6664d86b2fdd1cebbd1335f80fad'


def checked_inputs(root):
    for name, digest in [('workflow.json', OLD_WORKFLOW),
            ('workflow_input_recovery_v1.json', NEW_WORKFLOW),
            ('campaign.json', OLD_PLAN), ('campaign_input_recovery_v1.json', NEW_PLAN),
            ('quota_policy.json', QUOTA)]:
        if sha256(root/name) != digest:
            raise ValueError('Frozen input changed: '+name)
    if list(root.glob('STOP_SLOT_*')):
        raise ValueError('Separate operator slot STOP exists')
    old = campaign.optional(root/'workflow.json')
    new = campaign.optional(root/'workflow_input_recovery_v1.json')
    for w in (old, new):
        verified_source(Path(w['source']), w['source_sha256'])
        if sha256(Path(w['phase1_plan'])) != w['phase1_sha256']:
            raise ValueError('Hybrid plan changed')
    verified_source(root/'operations_fair_fill_v1', OPS_SOURCE)
    plan = campaign.optional(root/'campaign_input_recovery_v1.json')
    if plan['queue'] != campaign.optional(root/'campaign.json')['queue']:
        raise ValueError('Case/seed queue changed')
    phase_stop = Path(new['phase1_plan']).parent/'STOP'
    stop = campaign.optional(phase_stop)
    if stop.get('reason') != 'Verified evaluation phase complete' or stop.get('workflow_sha256') != OLD_WORKFLOW:
        raise ValueError('Phase1 marker is not the old completed-phase barrier')
    return plan, new, phase_stop


def preflight(plan):
    jobs = campaign.job_list(plan)
    active = [j for j in jobs if j.get('state') not in campaign.TERMINAL | {'SUSPENDED'}]
    def gpu(job):
        return sum(int(s.get('replicas', r.get('total_replicas', 1))) *
            int(s.get('limits', {}).get('nvidia.com/gpu', 0))
            for r in job.get('roles', []) for s in r.get('resource_spec', []))
    own = [j for j in active if j.get('display_name', '').startswith(plan['experiment_prefix']+'_')]
    total = sum(gpu(j) for j in own)
    if total > 30:
        raise ValueError('Campaign exceeds reviewed 30-GPU cap')
    return dict(utc=campaign.utc(), jobs_read=len(jobs), requested_gpus=0, campaign_gpus=total,
        current={q: sum(gpu(j) for j in active if j.get('scheduling', {}).get('quota_type', '').upper() == q)
                 for q in ('RESERVED', 'SPOT')},
        jobs=[dict(name=j['name'], display_name=j['display_name'], state=j['state'], gpus=gpu(j)) for j in own])


def restore_slots(before, stopped):
    """Restore only our STOP-induced state, not operator stops or old idle jobs."""
    result = copy.deepcopy(stopped)
    prior = {s['id']: s for s in before['slots']}
    if set(prior) != {s['id'] for s in result['slots']}:
        raise ValueError('Slot topology changed')
    restored = []
    for slot in result['slots']:
        if slot['status'] != 'stopped':
            continue
        old = prior[slot['id']]
        keys = ('index', 'attempt', 'submission', 'auth_profile')
        if any(slot.get(k) != old.get(k) for k in keys):
            raise ValueError('Stopped reservation changed; reconcile explicitly')
        if old['status'] not in ('idle', 'ready', 'submitting', 'submitted', 'running', 'drained'):
            raise ValueError('Never release an operator or unexplained stop')
        slot['status'] = old['status']
        restored.append(slot['id'])
    return result, restored


def unsubmitted_drafts(plan):
    shared = Path(plan['shared_root'])
    drafts = []
    for path in (shared/'cluster').glob(plan['experiment_prefix']+'_c*_a*/replica_*/submission.json'):
        directory = path.parent
        if (directory/'submission_started.json').exists():
            continue  # An ambiguous create is also an immutable intent.
        if any((directory/n).exists() for n in ('submission_response.json', 'bootstrap.log',
                'batch.json', 'scheduler_reconciliation.json')) or (shared/'results'/directory.parent.name).exists():
            raise ValueError('No intent but execution evidence exists: '+str(directory))
        submission = campaign.optional(path)
        if submission['source_sha256'] == NEW_SOURCE:
            continue
        if submission['source_sha256'] != campaign.optional(ROOT/'campaign.json')['source_sha256']:
            raise ValueError('Unreviewed draft source')
        drafts.append(directory)
    return drafts


def pause(root):
    plan, workflow, phase_stop = checked_inputs(root)
    handoff = root/'handoff/input_recovery_v1'
    if handoff.exists() or (root/'STOP').exists():
        raise ValueError('Handoff or STOP already exists; never replay')
    fresh = preflight(plan)
    state = campaign.optional(root/'state.json')
    if state['approved_sha256'] != OLD_PLAN:
        raise ValueError('Unexpected live plan')
    for slot in state['slots']:
        if slot['status'] in ('submitting', 'submitted'):
            sub = slot.get('submission')
            if not sub or campaign.optional(Path(sub).with_name('submission_response.json')).get('returncode') != 0:
                raise ValueError('Wait for the in-flight create response before pausing')
    handoff.mkdir(parents=True)
    doc = Path(__file__).resolve().parents[2]/'docs/operations/robodojo_input_recovery_v1_confirmation.md'
    receipt = dict(utc=campaign.utc(), user_confirmation='确认 继续吧', confirmation_document=str(doc),
        confirmation_document_sha256=sha256(doc), workflow_sha256=NEW_WORKFLOW, plan_sha256=NEW_PLAN,
        source_sha256=NEW_SOURCE, containers_unchanged=True, phase1_marker_sha256=sha256(phase_stop),
        helper_sha256=sha256(Path(__file__)))
    write_json(handoff/'authorization.json', receipt)
    write_json(handoff/'fresh_preflight_pause.json', fresh)
    write_json(handoff/'before_state.json', state)
    for name in ('quota_guard_state.json', 'launch_authorization.json'):
        shutil.copy2(root/name, handoff/('before_'+name))
    shutil.copy2(Path(__file__), handoff/'activation_helper.py')
    stop = dict(reason=REASON, utc=campaign.utc(), workflow_sha256=OLD_WORKFLOW,
        target_workflow_sha256=NEW_WORKFLOW, authorization_sha256=sha256(handoff/'authorization.json'))
    with (root/'STOP').open('x') as stream:
        json.dump(stop, stream); stream.flush(); os.fsync(stream.fileno())
    return dict(paused_external_dispatch=True, containers_stopped=0, preflight=fresh, receipt=receipt)


def activate(root):
    plan, workflow, phase_stop = checked_inputs(root)
    handoff = root/'handoff/input_recovery_v1'
    receipt = campaign.optional(handoff/'authorization.json')
    stop = campaign.optional(root/'STOP')
    if (stop.get('reason') != REASON or stop.get('target_workflow_sha256') != NEW_WORKFLOW
            or stop.get('authorization_sha256') != sha256(handoff/'authorization.json')
            or receipt.get('source_sha256') != NEW_SOURCE):
        raise ValueError('Not this authorized handoff STOP')
    if (handoff/'activation_intent.json').exists():
        raise ValueError('Activation already attempted; inspect durable intent, never replay')
    with (root/'supervisor.lock').open('a') as sup, (Path(plan['shared_root'])/'evaluation/campaign_dispatch.lock').open('a') as disp:
        fcntl.flock(sup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(disp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != OLD_PLAN:
            raise ValueError('Unexpected stopped state plan')
        revised, restored = restore_slots(campaign.optional(handoff/'before_state.json'), state)
        drafts = unsubmitted_drafts(plan)
        fresh = preflight(plan)
        # Recheck after the external read; operator STOP always wins.
        checked_inputs(root)
        if campaign.optional(root/'STOP') != stop or campaign.optional(root/'state.json') != state:
            raise ValueError('Live state or stop changed under handoff')
        if sha256(phase_stop) != receipt['phase1_marker_sha256']:
            raise ValueError('Phase1 marker changed')
        write_json(handoff/'stopped_state.json', state)
        shutil.copy2(phase_stop, handoff/'phase1_STOP_original')
        shutil.copy2(root/'launch_authorization.json', handoff/'stopped_launch_authorization.json')
        write_json(handoff/'fresh_preflight_activate.json', fresh)
        intent = dict(utc=campaign.utc(), source_sha256=NEW_SOURCE, plan_sha256=NEW_PLAN,
            workflow_sha256=NEW_WORKFLOW, restored_slots=restored,
            retired_drafts=[str(p) for p in drafts], containers_stopped=0,
            previous_state_sha256=sha256(root/'state.json'))
        write_json(handoff/'activation_intent.json', intent)
        for directory in drafts:
            target = handoff/'unsubmitted_drafts'/directory.parent.name/directory.name
            target.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(target)
            for slot in revised['slots']:
                if slot.get('submission') == str(directory/'submission.json'):
                    if slot['status'] not in ('ready', 'submitting', 'submitted'):
                        raise ValueError('Unexpected draft slot state')
                    slot.update(status='ready', submission=None, job_id=None, platform_state=None)
        revised.update(approved_sha256=NEW_PLAN, updated_utc=campaign.utc())
        write_json(root/'state.json', revised)
        auth = campaign.optional(root/'launch_authorization.json')
        auth.update(approved_sha256=NEW_PLAN, accounts_confirmed=True,
            input_recovery_authorization=str(handoff/'authorization.json'), updated_utc=campaign.utc())
        write_json(root/'launch_authorization.json', auth)
        barrier = campaign.optional(phase_stop)
        barrier.update(workflow_sha256=NEW_WORKFLOW, previous_workflow_sha256=OLD_WORKFLOW,
            handoff_authorization=str(handoff/'authorization.json'))
        write_json(phase_stop, barrier)
        write_json(handoff/'activation.json', dict(intent, status='activated',
            state_sha256=sha256(root/'state.json'), quota_policy_sha256=QUOTA))
        (root/'STOP').rename(handoff/'STOP_released')
        return dict(activated=True, **intent, preflight=fresh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pause', 'activate'))
    print(json.dumps(globals()[parser.parse_args().action](ROOT), ensure_ascii=False, indent=2))
