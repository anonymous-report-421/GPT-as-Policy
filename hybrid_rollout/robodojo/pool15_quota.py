"""Model-free quota checks for A/B/C; no automatic reset credit consumption.

New work stops on an unknown quota. A retains a scheduler reserve. Only jobs
stopped by this guard are automatically recovered, keeping the same case/seed.
"""
import subprocess
import time
import fcntl
import hashlib
import json
from pathlib import Path

from . import campaign
from .codex_backend.profiles import ACCOUNT_GROUPS, validate_credential
from .quota_guard import QuotaGuard, durable, maybe_reset
from .quota_rpc import codex_limits
from .io import sha256


class Pool15QuotaGuard(QuotaGuard):
    def __init__(self, plan, policy_path=None, approved=None):
        campaign.validate_topology(plan)
        policy = plan['quota_protection']
        if (plan.get('preparation_revision') != 'context_v3_pool15'
                or policy.get('reset_enabled') is not False
                or not 0 < policy['a_stop_remaining'] < policy['a_hold_remaining']
                < policy['a_resume_remaining'] <= 100):
            raise ValueError('Unreviewed pool15 quota policy')
        self.plan = plan
        self.path = Path(plan['eval_manifest']).parent/'campaign.json'
        self.policy = dict(policy, codex=plan['codex'],
            recovery_source=plan['dispatcher_source'],
            recovery_source_sha256=plan['source_sha256'])
        self.external_path = Path(policy_path) if policy_path else None
        self.external_sha = approved
        self.external = {}
        if self.external_path:
            if not approved or sha256(self.external_path) != approved:
                raise ValueError('Unreviewed unattended quota policy')
            extra = campaign.optional(self.external_path)
            budget = Path(extra.get('reset_budget', '')).resolve()
            if (extra.get('schema') != 'robodojo.unattended_quota.v1'
                    or extra.get('b_max_resets') != 1
                    or extra.get('panel_sha256') != plan['panel_sha256']
                    or plan['experiment_prefix'] not in extra.get('campaigns', [])
                    or not budget.is_relative_to(Path(plan['shared_root'])/'evaluation')
                    or (extra.get('a_hold_remaining'), extra.get('a_stop_remaining'),
                        extra.get('a_resume_remaining')) != (30, 20, 50)):
                raise ValueError('Unattended quota authorization mismatch')
            self.external = extra

    def adopt_recovery_source(self, index):
        self.plan.setdefault('case_runtime_overrides', {})[str(index)] = dict(
            dispatcher_source=self.policy['recovery_source'],
            source_sha256=self.policy['recovery_source_sha256'], context_version='v3')

    def reset_b(self, rpc, value, profile=None):
        if not self.external:
            raise RuntimeError('Reset credits require the new explicit unattended authorization')
        path = Path(self.external['reset_budget'])
        tokens = json.loads(validate_credential(profile, self.plan['shared_root']).read_text())['tokens']
        identity = hashlib.sha256(tokens['account_id'].encode()).hexdigest()
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            budget = campaign.optional(path) or dict(schema='robodojo.account_reset_budget.v2',
                account_sha256=identity, max_resets=1, authorization_sha256=self.external_sha,
                b_reset_attempts=[])
            if (budget.get('account_sha256') != identity or budget.get('max_resets') != 1
                    or budget.get('authorization_sha256') != self.external_sha):
                raise ValueError('B reset authorization/account changed')
            value = maybe_reset(rpc, value, budget, lambda: durable(path, budget), max_resets=1)
            return value, budget['b_reset_attempts']

    def tick(self, state, directory, jobs):
        self.path = directory/'campaign.json'
        if self.external_path and sha256(self.external_path) != self.external_sha:
            raise ValueError('Unattended quota policy changed')
        path = directory/'quota_guard_state.json'
        journal = campaign.optional(path) or dict(schema='robodojo.pool15_quota.v1',
            panel_sha256=self.plan['panel_sha256'], stops={})
        if (journal.get('schema') != 'robodojo.pool15_quota.v1'
                or journal.get('panel_sha256') != self.plan['panel_sha256']):
            raise ValueError('Quota journal belongs to another experiment')
        for index in journal.get('recovery_indices', []):
            self.adopt_recovery_source(index)

        def save():
            journal['updated_utc'] = campaign.utc()
            durable(path, journal)

        limits = {}
        for group, profiles in ACCOUNT_GROUPS.items():
            value, checked_profile, errors = None, None, []
            # A stale access token in one isolated session must not masquerade
            # as account exhaustion. Reads use access tokens, never refresh one
            # of the active rollout sessions from this auxiliary process.
            for profile in reversed(profiles):
                if (directory/'STOP').exists():
                    return
                try:
                    with self.read_account(profile) as rpc:
                        value = codex_limits(rpc.request('account/rateLimits/read'))
                        if group == 'b' and self.external and not (directory/'STOP').exists():
                            value, attempts = self.reset_b(rpc, value, profile)
                            journal['b_reset_attempts'] = attempts
                    if value is None:
                        raise ValueError('Main Codex quota unavailable')
                    checked_profile = profile
                    break
                except (RuntimeError, TimeoutError, OSError, ValueError, subprocess.SubprocessError) as error:
                    value = None
                    errors.append(dict(profile=profile, error_type=type(error).__name__))
            limits[group] = value
            journal[group] = dict(checked_utc=campaign.utc(), limits=value,
                query_ok=value is not None, checked_profile=checked_profile, errors=errors)
            save()

        a = limits['a']
        held = journal.get('a_held', False)
        if a is not None:
            if a['remaining_percent'] <= self.policy['a_hold_remaining']:
                held = True
            elif a['remaining_percent'] >= self.policy['a_resume_remaining']:
                held = False
        journal['a_held'] = held
        blocked = [p for g, names in ACCOUNT_GROUPS.items() for p in names
            if limits[g] is None or (held if g == 'a' else limits[g]['remaining_percent'] <= 0)]
        disabled = journal.setdefault('disabled_auth_profiles', [])
        if self.external:
            for slot in state['slots']:
                if slot['status'] != 'pause_auth' or (directory/f"STOP_SLOT_{slot['id']}").exists():
                    continue
                job = self.matched_job(slot, jobs)
                if job is None or job['state'] not in campaign.TERMINAL - campaign.MANUAL_STOP:
                    continue
                from .evaluation import archive_path, case_identity, read_panel
                sub = campaign.optional(slot['submission'])
                case = self.plan['queue'][slot['index']]['case']
                archive = archive_path(Path(self.plan['shared_root']),
                    sub['environment']['ROLLOUT_EXPERIMENT_ID'], case, case['replica_id'], slot['attempt'])
                if campaign.assess(campaign.optional(sub['batch_result']), archive, job['state'],
                        case_identity(read_panel(self.plan['eval_manifest']), case)) != 'pause_auth':
                    continue
                if slot['auth_profile'] not in disabled:
                    disabled.append(slot['auth_profile'])
                # The failed session is not refreshed/copied by this monitor.
                # Let a healthy isolated session claim the same unfinished case.
                slot.update(status='pause_quota', last_action='isolated_auth_session_disabled')
            blocked += [p for p in disabled if p not in blocked]
        state['quota_guard'] = dict(blocked_profiles=blocked, checked_utc=campaign.utc(),
            reset_enabled=bool(self.external), policy_sha256=self.external_sha)
        for slot in state['slots']:
            if (directory/'STOP').exists() or (directory/f"STOP_SLOT_{slot['id']}").exists():
                continue
            if (slot['auth_profile'] in ACCOUNT_GROUPS['a'] and a is not None
                    and a['remaining_percent'] <= self.policy['a_stop_remaining']
                    and slot['status'] in ('running', 'submitted', 'submitting')):
                self.stop_for_reserve(slot, journal, jobs, save)
            self.reconcile_guard_stop(slot, state, journal, jobs, save)
            if slot['status'] == 'pause_quota':
                # The ordinary dispatcher only assigns this state after the
                # old container is terminal and its quota error is verified.
                slot.update(status='ready', submission=None, attempt=slot['attempt'] + 1,
                    retry_number=slot.get('retry_number', 0) + 1, next_retry_at=time.time() + 1)
            if slot['status'] == 'ready' and slot['auth_profile'] in blocked:
                sub = slot.get('submission')
                if sub and Path(sub).with_name('submission_started.json').exists():
                    continue
                prepared = campaign.submission_path(self.plan, slot)
                state.setdefault('pending_retries', []).append(dict(index=slot['index'],
                    attempt=slot['attempt'] + int(prepared.exists()),
                    retry_number=slot.get('retry_number', 0), next_retry_at=0))
                slot.update(status='idle', submission=None, job_id=None, platform_state=None)
        save()
        durable(directory/'state.json', state)
