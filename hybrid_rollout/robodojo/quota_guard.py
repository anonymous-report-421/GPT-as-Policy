"""Model-free quota protection for the sole campaign dispatcher.

Reset intents are fsynced before sending. Ambiguous responses reuse the SAME
idempotency key forever; at most two logical redemptions are ever allocated.
Neither auth stores nor historical rollout artifacts are modified.
"""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from . import campaign
from .evaluation import archive_path, case_identity, read_panel
from .io import sha256
from .quota_rpc import account_rpc, codex_limits
from .codex_backend.profiles import validate_credential


def durable(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def maybe_reset(rpc, limits, journal, save, *, max_resets=2):
    """Only B callers; a pending attempt counts against the cap until resolved."""
    attempts = journal.setdefault('b_reset_attempts', [])
    if max_resets not in (1, 2) or len(attempts) > max_resets:
        raise ValueError('Reset journal exceeds user authorization')
    pending = next((a for a in attempts if a['status'] == 'pending'), None)
    if pending is None:
        if (limits is None or limits['remaining_percent'] > 0
                or not limits.get('reset_credits_available') or len(attempts) >= max_resets):
            return limits
        pending = dict(idempotency_key=str(uuid.uuid4()), status='pending',
                       requested_utc=campaign.utc())
        attempts.append(pending)
        save()  # Must succeed before any redemption RPC.
    result = rpc.request('account/rateLimitResetCredit/consume',
                         dict(idempotencyKey=pending['idempotency_key']))
    outcome = result.get('outcome')
    if outcome not in ('reset', 'alreadyRedeemed', 'nothingToReset', 'noCredit'):
        raise RuntimeError('Unknown reset result; retain pending idempotency key')
    pending.update(status='resolved', outcome=outcome, completed_utc=campaign.utc())
    save()
    # Even nothingToReset/noCredit consumes one local authorization attempt.
    # This conservative accounting can never exceed two actual redemptions.
    return codex_limits(rpc.request('account/rateLimits/read'))


class QuotaGuard:
    def __init__(self, policy_path, approved, plan):
        self.path = Path(policy_path)
        if not approved or sha256(self.path) != approved:
            raise ValueError('Quota policy hash mismatch')
        self.approved = approved
        self.policy = json.loads(self.path.read_text())
        self.plan = plan
        p = self.policy
        if (p['schema'] != 'robodojo.quota_guard.v1' or p['b_max_resets'] != 2
                or not 0 < p['a_stop_remaining'] < p['a_hold_remaining'] < p['a_resume_remaining'] <= 100
                or p['account_a_profile'] != 'codex_a_2' or p['account_b_profile'] != 'codex_b_4'):
            raise ValueError('Unexpected quota guard policy')
        # Only account-interrupted retries adopt the approved recovery snapshot
        # for this method. Never cross from GPT-only back to a hybrid runtime.
        recovery = Path(p['recovery_source'])
        manifest = campaign.optional(recovery/'source_manifest.json')
        if manifest.get('source_sha256') != p['recovery_source_sha256']:
            raise ValueError('Quota recovery source mismatch')

    def read_account(self, profile):
        return account_rpc(profile, self.plan['shared_root'], self.policy['codex'],
                           self.plan['https_proxy_file'])

    def reset_b(self, rpc, value):
        # Global across both experiment phases and all B slots, not per campaign.
        root = Path(self.plan['shared_root'])/'evaluation'
        path = root/'codex_b_reset_budget.json'
        tokens = json.loads(validate_credential(self.policy['account_b_profile'],
            self.plan['shared_root']).read_text())['tokens']
        identity = hashlib.sha256(tokens['account_id'].encode()).hexdigest()
        with (root/'codex_b_reset_budget.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = campaign.optional(path) or dict(schema='robodojo.account_reset_budget.v1',
                account_sha256=identity, max_resets=2, b_reset_attempts=[])
            if budget.get('account_sha256') != identity or budget.get('max_resets') != 2:
                raise ValueError('Global B reset budget identity mismatch')
            value = maybe_reset(rpc, value, budget, lambda: durable(path, budget))
            return value, budget['b_reset_attempts']

    def tick(self, state, directory, jobs):
        if sha256(self.path) != self.approved:
            raise ValueError('Quota policy changed during execution')
        path = directory/'quota_guard_state.json'
        journal = campaign.optional(path) or dict(schema='robodojo.quota_guard_state.v1',
            policy_sha256=self.approved, b_reset_attempts=[], stops={})
        if journal['policy_sha256'] != self.approved:
            raise ValueError('Existing quota journal belongs to another policy')
        for index in journal.get('recovery_indices', []):
            self.adopt_recovery_source(index)
        def save():
            journal['updated_utc'] = campaign.utc()
            durable(path, journal)
        limits = {}
        for group, key in (('a', 'account_a_profile'), ('b', 'account_b_profile')):
            if (directory/'STOP').exists():
                return
            try:
                with self.read_account(self.policy[key]) as rpc:
                    value = codex_limits(rpc.request('account/rateLimits/read'))
                    if group == 'b' and not (directory/'STOP').exists():
                        value, attempts = self.reset_b(rpc, value)
                        journal['b_reset_attempts'] = attempts
                if value is None:
                    raise ValueError('Main Codex quota unavailable')
                limits[group] = value
                journal[group] = dict(checked_utc=campaign.utc(), limits=value, query_ok=True)
            except (RuntimeError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as error:
                # No stale quota inference; do not reset/restart on failed reads.
                journal.setdefault(group, {}).update(query_ok=False,
                    last_error_type=type(error).__name__, last_error_utc=campaign.utc())
                limits[group] = None
            save()
        a = limits['a']
        held = journal.get('a_held', False)
        if a is not None:
            if a['remaining_percent'] <= self.policy['a_hold_remaining']:
                held = True
            elif a['remaining_percent'] >= self.policy['a_resume_remaining']:
                held = False
        journal['a_held'] = held
        blocked = [s['auth_profile'] for s in state['slots'] if (
            s['auth_profile'].startswith('codex_a') and (held or a is None)) or (
            s['auth_profile'].startswith('codex_b') and
            (limits['b'] is None or limits['b']['remaining_percent'] <= 0))]
        state['quota_guard'] = dict(blocked_profiles=blocked, checked_utc=campaign.utc(),
                                   policy_sha256=self.approved)
        for slot in state['slots']:
            # Explicit user STOP is never undone by account recovery.
            if (directory/'STOP').exists() or (directory/f"STOP_SLOT_{slot['id']}").exists():
                continue
            is_a = slot['auth_profile'].startswith('codex_a')
            critical = is_a and a is not None and a['remaining_percent'] <= self.policy['a_stop_remaining']
            if critical and slot['status'] in ('running', 'submitted', 'submitting'):
                self.stop_for_reserve(slot, journal, jobs, save)
            self.reconcile_guard_stop(slot, state, journal, jobs, save)
            if slot['status'] == 'pause_quota':
                # Only a verified terminal quota failure is eligible; never resume
                # native failures, auth failures, manual stops or ambiguous creates.
                slot.update(status='ready', submission=None, attempt=slot['attempt'] + 1,
                    retry_number=slot.get('retry_number', 0) + 1, next_retry_at=time.time() + 1)
        # Release not-yet-created work from held accounts to available accounts.
        for slot in state['slots']:
            if slot['status'] == 'ready' and slot['auth_profile'] in blocked:
                sub = slot.get('submission')
                if sub and Path(sub).with_name('submission_started.json').exists():
                    continue
                prepared = campaign.submission_path(self.plan, slot)
                next_attempt = slot['attempt'] + int(prepared.exists())
                state.setdefault('pending_retries', []).append(dict(index=slot['index'],
                    attempt=next_attempt, retry_number=slot.get('retry_number', 0), next_retry_at=0))
                slot.update(status='idle', submission=None, job_id=None, platform_state=None)
        save()
        durable(directory/'state.json', state)

    def adopt_recovery_source(self, index):
        self.plan.setdefault('case_runtime_overrides', {})[str(index)] = dict(
            dispatcher_source=self.policy['recovery_source'],
            source_sha256=self.policy['recovery_source_sha256'], context_version='v2')

    def matched_job(self, slot, jobs):
        if not slot.get('submission'):
            return None
        sub = campaign.optional(slot['submission'])
        env = sub.get('environment', {})
        name = env.get('ROLLOUT_EXPERIMENT_ID', '') + '_r' + env.get('ROLLOUT_REPLICA_ID', '')
        found = [j for j in jobs if j.get('display_name') == name
                 and j.get('ownership', {}).get('user_name') == 'sujiayi']
        return found[0] if len(found) == 1 else None

    def stop_for_reserve(self, slot, journal, jobs, save):
        job = self.matched_job(slot, jobs)
        if job is None or job['state'] in campaign.TERMINAL | campaign.MANUAL_STOP:
            return
        key = job['name']
        if key not in journal['stops']:
            journal['stops'][key] = dict(slot_id=slot['id'], index=slot['index'],
                attempt=slot['attempt'], submission=slot['submission'], reason='preserve_A_scheduler_quota',
                utc=campaign.utc(), state='intent')
            save()
        argv = [self.plan['sco'], 'acp', 'jobs', 'stop',
                '--workspace-name=' + self.plan['workspace'], key]
        try:
            result = subprocess.run(argv, capture_output=True, timeout=60)
            journal['stops'][key]['stop_returncode'] = result.returncode
        except (OSError, subprocess.SubprocessError) as error:
            journal['stops'][key]['stop_error_type'] = type(error).__name__
        save()  # Job state, not this return code, determines whether requeue is safe.

    def reconcile_guard_stop(self, slot, state, journal, jobs, save):
        job = self.matched_job(slot, jobs)
        if job is None:
            return
        record = journal['stops'].get(job['name'])
        if not record or record['state'] == 'reconciled':
            return
        if job['state'] not in campaign.TERMINAL | {'SUSPENDED'}:
            return
        if (record['index'], record['attempt'], record['submission']) != (
                slot['index'], slot['attempt'], slot['submission']):
            raise ValueError('Quota stop identity mismatch')
        case = self.plan['queue'][slot['index']]['case']
        sub = campaign.optional(slot['submission'])
        archive = archive_path(Path(self.plan['shared_root']),
            sub['environment']['ROLLOUT_EXPERIMENT_ID'], case, case['replica_id'], slot['attempt'])
        outcome = campaign.optional(archive/'sim/evaluation_outcome.json')
        result = campaign.optional(archive/'controller/result.json')
        if outcome.get('complete') or result.get('complete'):
            # Completion can race with stop: never repeat a native terminal case.
            batch = campaign.optional(sub['batch_result'])
            batch = dict(batch, exit_code=0)
            action = campaign.assess(batch, archive, 'SUCCEEDED',
                case_identity(read_panel(self.plan['eval_manifest']), case))
            slot['status'] = 'idle' if action == 'completed' else 'pause_terminal_artifacts'
        else:
            # Sidecar, not a rewrite of the stopped episode/batch. The explicitly
            # pinned recovery snapshot understands this infrastructure category.
            reconciliation = Path(slot['submission']).with_name('scheduler_reconciliation.json')
            if not reconciliation.exists():
                durable(reconciliation, dict(schema='hybrid_rollout.robodojo.scheduler_reconciliation.v1',
                    checked_utc=campaign.utc(), job_id=job['name'], platform_state=job['state'],
                    batch_path=sub['batch_result'], case_id=case['case_id'],
                    action='user_authorized_quota_recovery', old_container_terminal=True,
                    reason='Preserve A quota for scheduler; same case/seed, new attempt, old data retained'))
            indices = journal.setdefault('recovery_indices', [])
            if slot['index'] not in indices:
                indices.append(slot['index'])
                save()
            self.adopt_recovery_source(slot['index'])
            state.setdefault('pending_retries', []).append(dict(index=slot['index'],
                attempt=slot['attempt'] + 1, retry_number=slot.get('retry_number', 0) + 1, next_retry_at=0))
            slot['status'] = 'idle'
        slot.setdefault('history', []).append(dict(index=slot['index'], attempt=slot['attempt'],
            job_id=job['name'], submission=slot['submission'], action='quota_guard_stop_reconciled',
            finished_utc=campaign.utc()))
        slot.update(submission=None, job_id=None, platform_state=None)
        record.update(state='reconciled', reconciled_utc=campaign.utc())
        # Atomic state first; replay must not allocate a duplicate retry after crash.
        # A cleared submission prevents another match even if journal write fails.
        durable(self.path.parent/'state.json', state)
        save()
