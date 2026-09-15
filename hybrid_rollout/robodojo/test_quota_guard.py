import copy
from contextlib import contextmanager
from pathlib import Path

import pytest

from . import campaign
from .io import sha256, write_json
from .quota_guard import QuotaGuard, maybe_reset
from .quota_rpc import codex_limits
from .test_campaign import dispatcher_fixture
from .case_ledger import require_available
from .evaluation import case_identity


@pytest.mark.parametrize('rpc_failure', [False, True])
def test_private_cleanup_race_does_not_hide_rpc_outcome(tmp_path, monkeypatch, rpc_failure):
    import io
    from types import SimpleNamespace
    from . import quota_rpc
    auth = tmp_path/'test_auth.json'
    write_json(auth, dict(tokens=dict(access_token='synthetic', account_id='synthetic')))
    monkeypatch.setattr(quota_rpc, 'validate_credential', lambda *a: auth)
    monkeypatch.setattr(quota_rpc, 'clean_login_env', lambda *a: {})
    @contextmanager
    def scratch(**kwargs):
        assert kwargs['ignore_cleanup_errors'] is True
        yield str(tmp_path)
        # Actual TemporaryDirectory suppresses cleanup-only ENOTEMPTY when
        # ignore_cleanup_errors=True; it must never suppress an RPC failure.
    monkeypatch.setattr(quota_rpc.tempfile, 'TemporaryDirectory', scratch)
    proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), poll=lambda:0)
    monkeypatch.setattr(quota_rpc.subprocess,'Popen',lambda *a,**k:proc)
    class Admin:
        def __init__(self,*a):pass
        def send(self,*a):pass
        def request(self,method,*a):
            if rpc_failure and method=='account/rateLimits/read':
                raise RuntimeError('Real RPC failed')
            return {'valid':True}
    monkeypatch.setattr(quota_rpc,'AccountRPC',Admin)
    def read():
        with quota_rpc.account_rpc('codex_b_4',tmp_path,'codex','proxy') as rpc:
            value=rpc.request('account/rateLimits/read')
        return value
    if rpc_failure:
        with pytest.raises(RuntimeError,match='Real RPC failed'):read()
    else:
        assert read()=={'valid':True}


def limits(remaining=0, credits=3):
    return dict(remaining_percent=remaining, reset_credits_available=credits)


class RPC:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.keys = []

    def request(self, method, params=None):
        if method == 'account/rateLimits/read':
            return dict(rateLimits=dict(limitId='codex', primary=dict(usedPercent=0)))
        self.keys.append(params['idempotencyKey'])
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return dict(outcome=value)


def test_reset_ambiguous_reuses_key_and_two_cap():
    rpc = RPC([TimeoutError(), 'alreadyRedeemed', 'reset'])
    journal = {}
    saved = []
    def save():
        saved.append(copy.deepcopy(journal))
    with pytest.raises(TimeoutError):
        maybe_reset(rpc, limits(), journal, save)
    assert saved[0]['b_reset_attempts'][0]['status'] == 'pending'
    maybe_reset(rpc, limits(80), journal, save)  # Resolve even if quota already reset.
    assert rpc.keys[0] == rpc.keys[1]
    maybe_reset(rpc, limits(), journal, save)
    assert rpc.keys[2] != rpc.keys[1]
    maybe_reset(rpc, limits(), journal, save)
    assert len(rpc.keys) == 3 and len(journal['b_reset_attempts']) == 2


@pytest.mark.parametrize('value', [limits(1), limits(0, 0), None])
def test_no_early_reset(value):
    rpc = RPC([])
    maybe_reset(rpc, value, {}, lambda: None)
    assert not rpc.keys


def test_no_rpc_when_intent_cannot_be_saved():
    rpc = RPC(['reset'])
    def fail():
        raise OSError('disk full')
    with pytest.raises(OSError):
        maybe_reset(rpc, limits(), {}, fail)
    assert not rpc.keys


def test_wrong_bucket_not_used():
    assert codex_limits(dict(rateLimits=dict(limitId='codex_bengalfox',
        primary=dict(usedPercent=0)))) is None
    actual = codex_limits(dict(rateLimitsByLimitId={'codex': dict(
        primary=dict(usedPercent=40), secondary=dict(usedPercent=75))}))
    assert actual['remaining_percent'] == 25


def guard(tmp_path, plan):
    path = tmp_path/'quota_policy.json'
    write_json(tmp_path/'source_manifest.json', dict(source_sha256='test'))
    write_json(path, dict(schema='robodojo.quota_guard.v1', b_max_resets=2,
        a_stop_remaining=20, a_hold_remaining=30, a_resume_remaining=50,
        account_a_profile='codex_a_2', account_b_profile='codex_b_4', codex='/unused',
        recovery_source=str(tmp_path), recovery_source_sha256='test'))
    return QuotaGuard(path, sha256(path), plan)


def set_reads(monkeypatch, obj, a, b):
    monkeypatch.setattr(obj, 'reset_b', lambda rpc, value: (value, []))
    @contextmanager
    def read(profile):
        class Probe:
            def request(self, method):
                percent = a if profile.startswith('codex_a') else b
                if percent is None:
                    raise TimeoutError()
                return dict(rateLimits=dict(limitId='codex', primary=dict(usedPercent=100-percent)))
        yield Probe()
    monkeypatch.setattr(obj, 'read_account', read)


def test_a_hysteresis_and_unavailable_read_blocks_new_work(tmp_path, monkeypatch):
    obj = guard(tmp_path, {})
    state = dict(slots=[dict(id=7, auth_profile='codex_a_2', status='idle')])
    for remaining, blocked in [(44, False), (29, True), (44, True), (55, False), (None, True)]:
        set_reads(monkeypatch, obj, remaining, 25)
        obj.tick(state, tmp_path, [])
        assert bool(state['quota_guard']['blocked_profiles']) == blocked


def test_paused_quota_releases_work_without_reset_or_rewriting_old_results(tmp_path, monkeypatch):
    obj = guard(tmp_path, {})
    state = dict(slots=[dict(id=7, auth_profile='codex_a_2', status='pause_quota',
        index=3, attempt=4, retry_number=2, submission=None)])
    set_reads(monkeypatch, obj, 10, 25)
    monkeypatch.setattr(campaign, 'submission_path', lambda *a: tmp_path/'missing.json')
    obj.tick(state, tmp_path, [])
    assert state['slots'][0]['status'] == 'idle'
    assert state['pending_retries'] == [dict(index=3, attempt=5, retry_number=3, next_retry_at=0)]


def test_explicit_stop_never_rearmed(tmp_path, monkeypatch):
    obj = guard(tmp_path, {})
    state = dict(slots=[dict(id=7, auth_profile='codex_a_2', status='pause_quota')])
    write_json(tmp_path/'STOP_SLOT_7', {})
    set_reads(monkeypatch, obj, 90, 90)
    obj.tick(state, tmp_path, [])
    assert state['slots'][0]['status'] == 'pause_quota'


def test_blocked_account_does_not_reserve(tmp_path, monkeypatch):
    plan, state, *_ = dispatcher_fixture(tmp_path, monkeypatch)
    slot = state['slots'][0]
    slot['status'] = 'idle'
    state['quota_guard'] = dict(blocked_profiles=[slot['auth_profile']])
    def fail(*args):
        raise AssertionError('blocked account cannot reserve or create')
    monkeypatch.setattr(campaign, 'reserve', fail)
    campaign.tick(plan, state, tmp_path, [])
    assert slot['status'] == 'idle'


def test_stopped_container_requeues_once_with_old_bytes_intact(tmp_path, monkeypatch):
    plan, state, job, panel, sub, batch, archive = dispatcher_fixture(tmp_path, monkeypatch)
    obj = guard(tmp_path, plan)
    slot = state['slots'][0]
    slot['auth_profile'] = 'codex_a_2'
    batch.parent.mkdir(parents=True)
    write_json(batch, dict(state='running', panel_sha256=panel['panel_sha256'], attempt='0',
        episodes=[dict(case_identity(panel, plan['queue'][0]['case']), state='starting', archive=str(archive))]))
    before = batch.read_bytes()
    journal = dict(stops={job['name']: dict(state='intent', index=0, attempt=0, submission=str(sub))})
    job['state'] = 'RUNNING'
    obj.reconcile_guard_stop(slot, state, journal, [job], lambda: None)
    assert not state.get('pending_retries')
    job['state'] = 'SUSPENDED'
    obj.reconcile_guard_stop(slot, state, journal, [job], lambda: None)
    assert state['pending_retries'][0]['attempt'] == 1
    assert batch.read_bytes() == before
    require_available(tmp_path, panel, plan['queue'][0]['case']['case_id'])
    obj.reconcile_guard_stop(slot, state, journal, [job], lambda: None)
    assert len(state['pending_retries']) == 1
    assert plan['case_runtime_overrides']['0']['context_version'] == 'v2'


def test_native_complete_race_does_not_requeue(tmp_path, monkeypatch):
    plan, state, job, panel, sub, batch, archive = dispatcher_fixture(tmp_path, monkeypatch)
    obj = guard(tmp_path, plan)
    slot = state['slots'][0]
    write_json(archive/'controller/result.json', dict(complete=True))
    journal = dict(stops={job['name']: dict(state='intent', index=0, attempt=0, submission=str(sub))})
    job['state'] = 'SUSPENDED'
    obj.reconcile_guard_stop(slot, state, journal, [job], lambda: None)
    assert not state.get('pending_retries')
    assert slot['status'] == 'pause_terminal_artifacts'
