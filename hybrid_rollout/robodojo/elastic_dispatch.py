"""Shared work queue, immediate refill, and backoff without idle account slots.

Case identity is independent of slot identity. Native v2 outcomes never cause
further reruns; only explicitly planned v1-failure comparisons may repeat a case.
"""
import time

from . import campaign
from .case_ledger import build_ledger
from .evaluation import read_panel
from .io import write_json

ACTIVE = {'ready', 'submitted', 'submitting', 'running'}


def reserve(state, slot, plan):
    used = {i for s in state['slots'] for i in s.get('assigned', [])}
    skipped = state.setdefault('skipped_indices', {})
    busy = {plan['queue'][s['index']]['case']['case_id'] for s in state['slots']
            if s is not slot and s.get('status') in ACTIVE}
    retries = state.setdefault('pending_retries', [])
    candidates = [(r['index'], r) for r in retries if r['next_retry_at'] <= time.time()]
    candidates += [(i, None) for i, e in sorted(enumerate(plan['queue']),
                   key=lambda item: (item[1].get('priority', 1), item[0]))
                   if i not in used and str(i) not in skipped]
    ledger = None
    for i, retry in candidates:
        entry = plan['queue'][i]
        cid = entry['case']['case_id']
        if cid in busy:
            continue
        origin = entry.get('rerun_v1_archive')
        if origin or entry.get('rerun_from_context') == 'v1':
            if ledger is None:
                ledger = build_ledger(plan['shared_root'], read_panel(plan['eval_manifest']))
                if ledger['errors']:
                    raise ValueError('Cannot schedule context comparisons with ledger errors')
            row = next(r for r in ledger['cases'] if r['case_id'] == cid)
            old = [a for a in row['attempts'] if (not origin or a.get('archive') == origin)
                   and a['state'] == 'completed' and a.get('context_version', 'v1') == 'v1']
            if len(old) != 1:
                continue  # Still running or unverified: never head-of-line block other work.
            if old[0]['native_success'] is True:
                skipped[str(i)] = 'v1_native_success_no_rerun'
                continue
            origin = old[0]['archive']
        if retry:
            retries.remove(retry)
            values = {k: retry[k] for k in ('attempt', 'retry_number', 'next_retry_at')}
        else:
            values = dict(attempt=entry['initial_attempt'], retry_number=0, next_retry_at=0)
            if origin:
                values['attempt'] = max(values['attempt'], 1 + max(
                    (int(a['attempt']) for a in row['attempts']), default=-1))
        if i not in slot.setdefault('assigned', []):
            slot['assigned'].append(i)
        slot.update(index=i, status='ready', submission=None, job_id=None,
                    platform_state=None, **values)
        slot.pop('rerun_v1_archive', None)
        if origin:
            slot['rerun_v1_archive'] = origin
        return
    unclaimed = any(i not in used and str(i) not in skipped for i in range(len(plan['queue'])))
    slot['status'] = 'idle' if unclaimed or retries else 'drained'


def tick(plan, state, directory, jobs, *, retry_input_guards_once=False, max_new_submissions=None):
    # First reconcile ALL old containers, so a newly idle earlier slot can take
    # work released by a later slot during this same poll.
    old = {s['id'] for s in state['slots'] if s['status'] in ('submitted', 'submitting', 'running')}
    campaign.tick(plan, state, directory, jobs, only_slots=old)
    if retry_input_guards_once:
        from .input_guard_queue import release_verified_guards
        release_verified_guards(plan, state, directory, jobs)
    for slot in state['slots']:
        if slot['status'] == 'ready' and slot.get('next_retry_at', 0) > time.time():
            state.setdefault('pending_retries', []).append(dict(index=slot['index'],
                attempt=slot['attempt'], retry_number=slot['retry_number'], next_retry_at=slot['next_retry_at']))
            slot.update(status='idle', submission=None, job_id=None, platform_state=None)
    # Previously drained slots can help with newly released retries.
    if state.get('pending_retries'):
        for slot in state['slots']:
            if slot['status'] == 'drained':
                slot['status'] = 'idle'
    state['updated_utc'] = campaign.utc()
    write_json(directory/'state.json', state)
    fill = {s['id'] for s in state['slots'] if s['status'] in ('idle', 'ready')}
    campaign.tick(plan, state, directory, jobs, only_slots=fill, max_new_submissions=max_new_submissions)
    # Do NOT reconcile freshly submitted jobs against the pre-submit job list.
