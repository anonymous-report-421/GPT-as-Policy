import copy
import json
from pathlib import Path

import pytest

from . import campaign, two_stage
from .evaluation import case_identity
from .io import sha256, write_json
from .paired_evaluation import comparison
from .phase_gate import verify_certificate
from .test_evaluation import fake_panel_file


def certificate(tmp_path):
    _, _, panel = fake_panel_file(tmp_path)
    chosen = [c for c in panel['cases'] if c['rollout_index'] != 2
              ]  # Ten tasks, five cases each; test identity fixture only.
    rows = []
    for i, case in enumerate(chosen):
        archive = tmp_path/'artifacts'/str(i)
        (archive/'controller').mkdir(parents=True)
        (archive/'sim').mkdir()
        expected = case_identity(panel, case)
        write_json(archive/'controller/result.json', dict(complete=True, evaluation_case=expected))
        write_json(archive/'sim/evaluation_outcome.json', dict(complete=True, valid_for_success_rate=True,
            evaluation_case=expected, native_success=i < 8))
        write_json(archive/'artifact_manifest.json', dict(status='verified'))
        rows.append(dict(case_id=case['case_id'], task=case['task'], evaluation_case=expected,
            native_success=i < 8, archive=str(archive), reused_v1_success=i < 8,
            context_version='v1' if i < 8 else 'v2',
            result_sha256=sha256(archive/'controller/result.json'),
            outcome_sha256=sha256(archive/'sim/evaluation_outcome.json'),
            artifact_manifest_sha256=sha256(archive/'artifact_manifest.json')))
    selection = dict(method='pi05_plus_gpt', complete=True, platform_verified=True,
                     cases=rows, native_successes=8)
    document = dict(schema='robodojo.hybrid50_completion.v1', panel_sha256=panel['panel_sha256'], selection=selection)
    path = tmp_path/'certificate.json'; write_json(path, document)
    return path, panel, document


def test_phase_gate_all50_native_success_or_failure(tmp_path):
    path, panel, doc = certificate(tmp_path)
    assert verify_certificate(path, panel) == doc
    assert sum(r['native_success'] for r in doc['selection']['cases']) == 8


@pytest.mark.parametrize('change', ['missing', 'only49', 'duplicate', 'unverified', 'seed',
                                  'result_changed', 'v1_failure', 'nonexempt_v1'])
def test_phase_gate_rejects_invalid_or_incomplete(tmp_path, change):
    path, panel, doc = certificate(tmp_path)
    rows = doc['selection']['cases']
    if change == 'missing':
        path = tmp_path/'notyet.json'
    elif change == 'only49':
        rows.pop()
    elif change == 'duplicate':
        rows[1] = rows[0]
    elif change == 'unverified':
        doc['selection']['platform_verified'] = False
    elif change == 'seed':
        rows[0]['evaluation_case']['reset_seed'] = 123
    elif change == 'result_changed':
        write_json(Path(rows[0]['archive'])/'controller/result.json', dict(complete=False))
    elif change == 'v1_failure':
        rows[9].update(reused_v1_success=True, context_version='v1')
    elif change == 'nonexempt_v1':
        rows[9]['context_version'] = 'v1'
    if change != 'missing':
        write_json(path, doc)
    with pytest.raises(ValueError):
        verify_certificate(path, panel)


def test_paired_summary_checks_seed_and_labels_v1(tmp_path):
    _, _, doc = certificate(tmp_path)
    hybrid = doc['selection']; direct = copy.deepcopy(hybrid)
    direct['method'] = 'gpt_only'
    report = comparison(hybrid, direct)
    assert report['hybrid_success_rate'] == 8/50
    assert report['reused_v1_success_count'] == 8 and len(report['per_task']) == 10
    direct['cases'][0]['evaluation_case']['reset_seed'] = 345
    with pytest.raises(ValueError, match='aligned'):
        comparison(hybrid, direct)


def test_stage2_has_fresh_shared_queue_without_hybrid_bindings(tmp_path):
    _, panel, doc = certificate(tmp_path)
    parent = dict(shared_root=str(tmp_path), eval_manifest='p', panel_sha256=panel['panel_sha256'],
        scope_file='s', scope_file_sha256='s', slots=[dict(id=i, auth_profile=p) for i,p in campaign.B5_A2_SLOTS.items()],
        poll_seconds=300, retry_base_seconds=300, retry_max_seconds=3600, retry_limit=None, retry_only='transport',
        model='gpt-6-astra', effort='xhigh', fast=False, max_decisions=0, max_seconds=0,
        gpus_per_replica=2, max_concurrent_gpus=12, codex='codex', sco='sco', image='image', workspace='w',
        cluster='galbot-vla', worker_spec='n5lp.nn.a80.2', quota='reserved', https_proxy_file='private',
        primary_case_ids=[r['case_id'] for r in doc['selection']['cases']],
        case_runtime_overrides={'0': {'dispatcher_source':'old-hybrid'}}, skipped_completed=['old'],
        context_case_versions={'old':'v1'})
    plan = two_stage.stage2_plan(parent, panel, 'robodojo_direct', tmp_path, tmp_path/'source', 'hash')
    assert len(plan['queue']) == 50 and plan['skipped_completed'] == []
    assert plan['evaluation_method'] == 'gpt_only' and plan['context_version'] == 'v2'
    assert 'case_runtime_overrides' not in plan and 'context_case_versions' not in plan
    assert all('slot_id' not in e and e['initial_attempt'] == 0 for e in plan['queue'])
    assert {e['case']['case_id'] for e in plan['queue']} == set(parent['primary_case_ids'])
    assert plan['model'] == parent['model'] and plan['slots'] == parent['slots']


def test_operator_stop_is_never_replaced(tmp_path):
    write_json(tmp_path/'STOP', dict(reason='operator'))
    before = sha256(tmp_path/'STOP')
    assert not two_stage.release_phase(tmp_path, 'workflow')
    assert sha256(tmp_path/'STOP') == before


def test_no_stage2_until_all_mixed_complete(tmp_path, monkeypatch):
    first = tmp_path/'first/plan.json'; second = tmp_path/'second/plan.json'
    first.parent.mkdir(); second.parent.mkdir()
    scope = tmp_path/'scope.json'; write_json(scope,{})
    plan = dict(shared_root=str(tmp_path), eval_manifest='unused', panel_sha256='p', scope_file=str(scope),
                scope_file_sha256=sha256(scope), primary_case_ids=[])
    write_json(first, plan); write_json(second, plan)
    config = dict(phase1_plan=str(first), phase2_plan=str(second), phase1_plan_sha256=sha256(first),
        phase2_plan_sha256=sha256(second), source_sha256='source', exempt_v1_successes=[])
    monkeypatch.setattr(campaign,'source_digest',lambda:'source')
    monkeypatch.setattr(two_stage,'read_panel',lambda *a:{})
    monkeypatch.setattr(campaign,'job_list',lambda *a:[])
    monkeypatch.setattr(two_stage,'select_panel',lambda *a,**k:dict(complete=False, selected_count=49))
    monkeypatch.setattr(two_stage.subprocess,'Popen',lambda *a,**k:pytest.fail('Premature GPT-only launch'))
    _, status = two_stage.cycle(config,'wf',tmp_path,None)
    assert status['state'] == 'waiting_for_mixed50'


def test_blocked_idle_slot_can_drain_after_queue_exhausted(tmp_path):
    state = dict(slots=[dict(id=7, auth_profile='codex_a_2', status='idle', assigned=[0])],
                 quota_guard=dict(blocked_profiles=['codex_a_2']))
    plan = dict(dispatch_mode='work_conserving', queue=[dict(case={})])
    campaign.tick(plan,state,tmp_path,[])
    assert state['slots'][0]['status'] == 'drained'


def test_completed_stage_handoff_preserves_A_hold_and_is_singleton(tmp_path, monkeypatch):
    cert, panel, doc = certificate(tmp_path)
    first = tmp_path/'first/plan.json'; root = tmp_path/'second'; second = root/'plan.json'
    first.parent.mkdir(); root.mkdir()
    scope = tmp_path/'scope.json'; write_json(scope,{})
    plan = dict(shared_root=str(tmp_path), eval_manifest='unused', panel_sha256=panel['panel_sha256'],
        scope_file=str(scope), scope_file_sha256=sha256(scope),
        primary_case_ids=[r['case_id'] for r in doc['selection']['cases']], experiment_prefix='robodojo_direct', slots=[])
    write_json(first,plan); write_json(second,plan)
    write_json(first.parent/'quota_guard_state.json',dict(a_held=True))
    config = dict(phase1_plan=str(first), phase2_plan=str(second), phase1_plan_sha256=sha256(first),
        phase2_plan_sha256=sha256(second), source_sha256='source', source=str(tmp_path),
        exempt_v1_successes=[], certificate=str(cert), phase2_quota_policy='policy', phase2_quota_policy_sha256='q')
    monkeypatch.setattr(campaign,'source_digest',lambda:'source')
    monkeypatch.setattr(two_stage,'read_panel',lambda *a:panel)
    monkeypatch.setattr(campaign,'job_list',lambda *a:[])
    monkeypatch.setattr(two_stage,'select_panel',lambda *a,**k:
        doc['selection'] if a[3]=='pi05_plus_gpt' else dict(complete=False,selected_count=0))
    monkeypatch.setattr(two_stage,'dispatcher_unlocked',lambda *a:True)
    calls=[]
    class Child:
        pid=123
        def poll(self):return None
    def create(*a,**k):calls.append(a);return Child()
    monkeypatch.setattr(two_stage.subprocess,'Popen',create)
    child,status=two_stage.cycle(config,'wf',root,None)
    assert status['state']=='running_gpt_only50' and len(calls)==1
    assert campaign.optional(root/'quota_guard_state.json')['a_held'] is True
    assert two_stage.stop_is_ours(first.parent/'STOP','wf')
    two_stage.cycle(config,'wf',root,child)
    assert len(calls)==1
