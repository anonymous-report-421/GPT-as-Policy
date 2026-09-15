"""Operator-only release of terminal input-guard cases; never retry their actions."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from . import campaign
from .io import sha256, write_json
from .subscription_migration import _lock

REASON = 'user_requested_restore15_defer_input_guard_analysis'


def release_state(state, targets, started_paths):
    result = copy.deepcopy(state)
    deferred = result.setdefault('deferred_cases', {})
    for slot in result['slots']:
        if slot['id'] in targets:
            if (slot['status'] != 'pause_unknown_failure'
                    or slot.get('job_id') != targets[slot['id']]
                    or str(slot['index']) in deferred
                    or any(r['index'] == slot['index'] for r in result.get('pending_retries', []))):
                raise ValueError('Require an unqueued, paused, exact terminal guard case')
            if slot['index'] not in slot.get('assigned', []):
                raise ValueError('Deferred case must remain reserved in the original 50-case plan')
            deferred[str(slot['index'])] = dict(status='awaiting_operator_analysis',
                reason=REASON, original_slot=copy.deepcopy(slot), native_outcome=None)
            slot.update(status='idle', submission=None, job_id=None, platform_state=None,
                        next_retry_at=0, last_action='deferred_input_guard_case')
        elif slot['status'] == 'stopped':
            path = slot.get('submission')
            slot['status'] = ('submitted' if path in started_paths else 'ready') if path else 'idle'
    if set(targets) != {s['id'] for s in state['slots'] if s['id'] in targets}:
        raise ValueError('Unknown target slot')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('pause', 'release'))
    p.add_argument('plan', type=Path)
    p.add_argument('--approved-sha256', required=True)
    args = p.parse_args(); root = args.plan.resolve().parent
    plan = campaign.optional(args.plan)
    if (sha256(args.plan) != args.approved_sha256 or plan.get('context_version') != 'v3'
            or len(plan['slots']) != 15 or plan['primary_target'] != 50):
        raise ValueError('Require the reviewed v3 15-slot / 50-case plan')
    if args.action == 'pause':
        if (root/'STOP').exists():
            raise ValueError('Do not overwrite another stop intent')
        write_json(root/'STOP', dict(reason=REASON, utc=campaign.utc()))
        return
    with _lock(plan):
        if campaign.optional(root/'STOP').get('reason') != REASON:
            raise ValueError('Require this operator pause and an exited dispatcher')
        state = campaign.optional(root/'state.json')
        if state['approved_sha256'] != args.approved_sha256 or list(root.glob('STOP_SLOT_*')):
            raise ValueError('Plan changed or explicit slot stop exists')
        jobs = {j['name']: j for j in campaign.job_list(plan)}
        targets = {}
        for slot in state['slots']:
            if slot['status'] != 'pause_unknown_failure':
                continue
            sub = campaign.optional(slot['submission']); batch = campaign.optional(sub['batch_result'])
            episodes = batch.get('episodes', [])
            if len(episodes) != 1:
                continue
            archive = Path(episodes[0]['archive'])
            failure = campaign.optional(archive/'controller/failure.json')
            if 'Five rejected tool calls' not in failure.get('error', ''):
                continue
            if (jobs[slot['job_id']]['state'] not in campaign.TERMINAL
                    or not all(batch.get('final_reset', {}).get(k) is True for k in
                               ('ports_released', 'all_owned_processes_exited'))
                    or campaign.optional(archive/'sim/evaluation_outcome.json').get('complete') is True):
                raise ValueError('Guard case still active, unclean, or natively complete')
            targets[slot['id']] = slot['job_id']
        if not targets:
            raise ValueError('No verified terminal input-guard slot to release')
        started = {s['submission'] for s in state['slots'] if s.get('submission')
                   and Path(s['submission']).with_name('submission_started.json').exists()}
        revised = release_state(state, targets, started)
        revised['updated_utc'] = campaign.utc()
        backup = root/'before_guard_slot_release'; backup.mkdir()
        shutil.copy2(root/'state.json', backup/'state.json')
        write_json(root/'state.json', revised)
        write_json(root/'guard_slot_release.json', dict(utc=campaign.utc(), targets=targets,
            reason=REASON, plan_sha256=args.approved_sha256, previous_state=str(backup/'state.json'),
            deferred_cases=list(revised['deferred_cases']), raw_results_unchanged=True,
            model_prompt_limits_unchanged=True, created_jobs=0))
        (root/'STOP').rename(backup/'STOP_released')
        print(json.dumps(dict(released_slots=targets, deferred_cases=list(revised['deferred_cases']))))


if __name__ == '__main__':
    main()
