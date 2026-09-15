"""Recover only proven terminal, native-incomplete input-guard attempts.

No changes to controller prompts, safety limits, old artifacts or running jobs.
Retries back off and are constrained by the account guard and explicit STOPs.
"""
import copy
from pathlib import Path
import time

from . import campaign
from .input_guard_queue import release_verified_guards


def release_deferred(plan, state, directory, jobs):
    directory = Path(directory)
    if (directory/'STOP').exists():
        return
    # A timed-out create can become visible later. Reconcile that one exact job
    # instead of freezing forever or issuing a second create.
    for slot in state['slots']:
        if slot['status'] != 'pause_ambiguous_submission' or not slot.get('submission'):
            continue
        if (directory/f"STOP_SLOT_{slot['id']}").exists():
            continue
        sub = campaign.optional(slot['submission'])
        env = sub.get('environment', {})
        name = env.get('ROLLOUT_EXPERIMENT_ID', '')+'_r'+env.get('ROLLOUT_REPLICA_ID', '')
        matches = [j for j in jobs if j.get('display_name') == name
                   and j.get('ownership', {}).get('user_name') == 'sujiayi']
        if len(matches) == 1:
            slot.update(status='submitted', job_id=matches[0]['name'],
                        last_action='previous_ambiguous_create_now_visible')
    deferred = state.setdefault('deferred_cases', {})
    for key, record in list(deferred.items()):
        slot = copy.deepcopy(record['original_slot'])
        index = slot['index']
        if (str(index) != key or (directory/f"STOP_SLOT_{slot['id']}").exists()
                or any(r['index'] == index for r in state.get('pending_retries', []))
                or any(s.get('index') == index and s['status'] in
                       {'ready', 'running', 'submitted', 'submitting'} for s in state['slots'])):
            continue
        # A newer prepare/create intent is never silently ignored, even if its
        # response was lost. Let the dispatcher/operator reconcile it first.
        newer = list((Path(plan['shared_root'])/'cluster').glob(
            f"{plan['experiment_prefix']}_c{index}_a*/replica_*/submission.json"))
        if any(int(p.parents[1].name.rsplit('_a', 1)[1]) > slot['attempt'] for p in newer):
            continue
        probe = dict(slots=[slot], input_guard_retry_counts={})
        # The strict existing validator verifies platform owner, all case/seed
        # fields, incomplete native outcome, exact guard error, archive and reset.
        slot['history'] = []
        release_verified_guards(plan, probe, directory, jobs)
        retries = probe.get('pending_retries', [])
        if len(retries) != 1:
            continue
        counts = state.setdefault('unattended_guard_retry_counts', {})
        count = counts.get(key, 0) + 1
        counts[key] = count
        retry = retries[0]
        retry.update(next_retry_at=time.time()+campaign.retry_delay(count),
                     reason='unattended_verified_incomplete_same_case_seed')
        state.setdefault('pending_retries', []).append(retry)
        state.setdefault('unattended_recovery_history', []).append(dict(
            utc=campaign.utc(), index=index, old=record, next_attempt=retry['attempt'],
            same_case_seed=True, original_artifacts_preserved=True))
        del deferred[key]
