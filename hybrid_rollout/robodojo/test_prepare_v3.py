"""Offline v3 planning, same fixture identities and three-account isolation."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from . import campaign
from .case_ledger import build_ledger, require_available
from .codex_backend import profiles
from .evaluation import TASKS, verify_assets
from .io import write_json, sha256
from .pool15_quota import Pool15QuotaGuard
from .prepare_v3 import revised_inputs, verify_ready
from .test_case_ledger import attempt
from .test_evaluation import fake_panel_file


def scope_for(panel):
    scope = json.loads(Path(__file__).with_name('eval_panels').joinpath('robodojo_panel50_scope_v2.json').read_text())
    scope['source_panel_sha256'] = panel['panel_sha256']
    return scope


def test_v3_same_fixtures_but_old_success_never_completes_new_experiment(tmp_path):
    source, _, old = fake_panel_file(tmp_path)
    scope = scope_for(old)
    before = copy.deepcopy((old, scope))
    new, new_scope, queue = revised_inputs(old, scope, 'new_v3')
    assert (old, scope) == before
    assert new['cases'] == old['cases']
    assert new['panel_sha256'] != old['panel_sha256']
    assert new_scope['source_panel_sha256'] == new['panel_sha256']
    assert [e['case']['task'] for e in queue[:10]] == list(TASKS)
    assert len(queue) == 50 and len({e['case']['case_id'] for e in queue}) == 50
    assert all('slot_id' not in e for e in queue)
    verify_assets(new, source, [e['case'] for e in queue])
    attempt(tmp_path, old, queue[0]['case'], success=True)
    old_files = {str(p):p.read_bytes() for p in (tmp_path/'results').rglob('*') if p.is_file()}
    require_available(tmp_path, new, queue[0]['case']['case_id'])
    assert not any(r['state'] == 'completed' for r in build_ledger(tmp_path, new)['cases'])
    assert old_files == {str(p):p.read_bytes() for p in (tmp_path/'results').rglob('*') if p.is_file()}


def test_plan_rejects_missing_case_or_unapproved_launch(tmp_path):
    _, _, old = fake_panel_file(tmp_path)
    scope = scope_for(old); scope['tasks'][0]['case_ids'].pop()
    with pytest.raises(ValueError): revised_inputs(old, scope, 'v3')
    path = tmp_path/'campaign.json'; write_json(path, {})
    with pytest.raises(ValueError, match='not authorized'): verify_ready(path, sha256(path))


def pool_plan(tmp_path):
    return dict(preparation_revision='context_v3_pool15', context_version='v3',
        panel_sha256='test', eval_manifest=str(tmp_path/'evaluation.json'),
        slots=[dict(id=i, auth_profile=p) for i, p in campaign.POOL15_SLOTS.items()],
        max_concurrent_gpus=30, dispatch_mode='work_conserving', codex='/unused',
        dispatcher_source=str(tmp_path), source_sha256='frozen-v3',
        quota_protection=dict(a_stop_remaining=20, a_hold_remaining=30,
                              a_resume_remaining=50, reset_enabled=False))


def test_topology_15_requires_v3_dynamic_and_explicit_gpu_budget(tmp_path):
    p = pool_plan(tmp_path)
    assert set(campaign.validate_topology(p)) == set(profiles.POOL15_PROFILES)
    for edit in (dict(context_version='v2'), dict(max_concurrent_gpus=14), dict(dispatch_mode='pinned')):
        with pytest.raises(ValueError): campaign.validate_topology(dict(p, **edit))


def mock_accounts(obj, monkeypatch, values):
    calls = []
    @contextmanager
    def read(name):
        class RPC:
            def request(self, method):
                calls.append((name, method))
                assert method == 'account/rateLimits/read'  # Never consume reset/model tokens.
                remaining = values[name.split('_')[1]]
                if remaining is None: raise TimeoutError()
                return dict(rateLimits=dict(limitId='codex', primary=dict(usedPercent=100-remaining)))
        yield RPC()
    monkeypatch.setattr(obj, 'read_account', read)
    return calls


def test_three_account_quota_no_reset_and_unknown_fail_closed(tmp_path, monkeypatch):
    obj = Pool15QuotaGuard(pool_plan(tmp_path))
    state = dict(slots=[dict(id=i, auth_profile=p, status='idle') for i,p in campaign.POOL15_SLOTS.items()])
    for a, expect_hold in ((29,True), (40,True), (50,False)):
        calls = mock_accounts(obj, monkeypatch, dict(a=a,b=0,c=None))
        obj.tick(state, tmp_path, [])
        blocked = set(state['quota_guard']['blocked_profiles'])
        assert all((n in blocked) == expect_hold for n in profiles.ACCOUNT_GROUPS['a'])
        assert set(profiles.ACCOUNT_GROUPS['b']+profiles.ACCOUNT_GROUPS['c']) <= blocked
        assert calls
    with pytest.raises(RuntimeError): obj.reset_b(None,None)
    obj.adopt_recovery_source(1)
    assert obj.plan['case_runtime_overrides']['1']['context_version'] == 'v3'
    write_json(tmp_path/'STOP', {})
    calls = mock_accounts(obj, monkeypatch, dict(a=90,b=90,c=90))
    obj.tick(state, tmp_path, [])
    assert not calls


def test_quota_moves_unstarted_work_to_other_account_and_keeps_manual_stop(tmp_path, monkeypatch):
    obj = Pool15QuotaGuard(pool_plan(tmp_path))
    slots = [dict(id=i, auth_profile=p, status='idle') for i,p in campaign.POOL15_SLOTS.items()]
    slots[0].update(status='pause_quota', index=3, attempt=1)
    slots[1].update(status='pause_quota', index=4, attempt=1)
    write_json(tmp_path/'STOP_SLOT_1', {})
    monkeypatch.setattr(campaign, 'submission_path', lambda *a: tmp_path/'missing.json')
    mock_accounts(obj, monkeypatch, dict(a=10,b=90,c=90))
    state = dict(slots=slots)
    obj.tick(state, tmp_path, [])
    assert slots[0]['status'] == 'idle' and state['pending_retries'][0]['index'] == 3
    assert state['pending_retries'][0]['attempt'] == 2
    assert slots[1]['status'] == 'pause_quota'
