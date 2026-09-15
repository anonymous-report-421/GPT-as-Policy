"""Pure migration/state guards; no ACP, credentials, model or simulator calls."""
import copy
import hashlib
import json

import pytest

from . import campaign
from .io import sha256, write_json
from .subscription_migration import activate, revised_state, REVISION
from . import subscription_migration


def source_state():
    slots = [dict(id=i, auth_profile=n, status='running' if i > 2 else 'pause_unknown_failure',
                  index=i, attempt=2, assigned=[i], submission=f'old/{i}', job_id=f'job-{i}')
             for i,n in enumerate(campaign.SLOTS)]
    return (dict(slots=[dict(id=i, auth_profile=n) for i,n in enumerate(campaign.SLOTS)],
                 queue=[dict(case=dict(case_id=f'case-{i}', replica_id=i%6), initial_attempt=0,
                             slot_id=i%7) for i in range(28)], max_concurrent_gpus=14),
            dict(slots=slots, approved_sha256='old'))


@pytest.mark.parametrize('galbot_completed', [True, False])
@pytest.mark.parametrize('topology', ['a3_b3', 'b5_a1', 'b5_a2'])
def test_retired_pending_queue_kept_and_live_official_slots_preserved(galbot_completed, topology):
    plan, state = source_state()
    original = copy.deepcopy((plan,state))
    completed = {1} if galbot_completed else set()
    new, result = revised_state(plan, state, completed, dict.fromkeys(range(28), 3), topology)
    assert (plan,state) == original
    _, target = subscription_migration.migration_spec(topology)
    assert campaign.validate_topology(new) == tuple(target.values())
    assert new['max_concurrent_gpus'] == 12
    retired_ids = (1, 6) if topology == 'b5_a2' else (1,)
    assert result['retired_slots'] == [state['slots'][i] for i in retired_ids]
    for slot in result['slots']:
        if slot['id'] not in (0, 2, 7):
            assert slot == state['slots'][slot['id']]
        else:
            assert slot['auth_profile'] == target[slot['id']]
            assert slot['status'] == 'ready' and slot['attempt'] == 3
            source_id = 6 if slot['id'] == 7 else slot['id']
            assert slot['index'] == state['slots'][source_id]['index']
            assert slot['job_id'] is None and slot['submission'] is None
            assert slot['history'][-1]['auth_profile'] == state['slots'][source_id]['auth_profile']
    # Populate all idle slots until exhausted: no case omitted or duplicated,
    # including all formerly Koozhan-routed entries and the retired Galbot case.
    while True:
        changed = False
        for slot in result['slots']:
            campaign.reserve(result, slot, new)
            if slot['status'] != 'drained': changed = True
        if not changed: break
    assigned = [i for slot in result['slots'] for i in slot['assigned']]
    assert len(assigned) == len(set(assigned)) == 28
    assert set(assigned) == set(range(28))
    assert [e['case'] for e in new['queue']] == [e['case'] for e in plan['queue']]
    if not galbot_completed: assert new['queue'][1]['initial_attempt'] == 3


def test_completed_target_cannot_be_restarted_by_backend_migration():
    plan, state = source_state()
    for index in (0,2):
        with pytest.raises(ValueError, match='Never rerun completed'):
            revised_state(plan, state, {index}, dict.fromkeys(range(28),3))
    with pytest.raises(ValueError, match='Never rerun completed'):
        revised_state(plan, state, {6}, dict.fromkeys(range(28),3), 'b5_a2')


def test_five_b_expansion_preserves_paused_a_and_does_not_enable_new_a():
    plan, state = source_state()
    state['slots'][6]['status'] = 'pause_unknown_failure'
    new, result = revised_state(plan, state, {1}, dict.fromkeys(range(28), 3), 'b5_a1')
    assert result['slots'][-1] == state['slots'][6]
    profiles = [s['auth_profile'] for s in result['slots']]
    assert sum(n.startswith('codex_b') for n in profiles) == 5
    assert 'codex_a_2' not in profiles and 'codex_a_3' not in profiles
    assert new['max_concurrent_gpus'] == 12  # Five B streams; original A remains paused.
    with pytest.raises(ValueError, match='Unknown subscription topology'):
        revised_state(plan, state, set(), dict.fromkeys(range(28), 3), 'unknown')


def test_topology_rejects_duplicate_slots_and_unreviewed_concurrency():
    plan, _ = source_state()
    plan['slots'].append(dict(id=0, auth_profile='codex_a'))
    with pytest.raises(ValueError): campaign.validate_topology(plan)
    plan['slots'] = [dict(id=i, auth_profile=n) for i,n in campaign.SUBSCRIPTION_SLOTS.items()]
    plan['slots'].append(dict(id=1, auth_profile='galbot'))
    with pytest.raises(ValueError): campaign.validate_topology(plan)


def test_activate_rejects_changed_stop_without_mutation(tmp_path):
    root = tmp_path/'campaign'; root.mkdir()
    (tmp_path/'evaluation').mkdir()
    write_json(root/'STOP', dict(reason='reviewed'))
    write_json(root/'STOP_SLOT_2', dict(reason='provider retired'))
    write_json(root/'state.json', dict(unchanged=True))
    plan = dict(shared_root=str(tmp_path), migration_revision=REVISION,
                migration_parent_state_sha256=sha256(root/'state.json'),
                migration_global_stop_sha256=sha256(root/'STOP'),
                migration_stop_hashes={'STOP_SLOT_2': sha256(root/'STOP_SLOT_2')})
    path = root/'plan.json'; write_json(path, plan)
    original = (root/'state.json').read_bytes()
    write_json(root/'STOP_SLOT_0', dict(reason='new operator stop'))
    with pytest.raises(ValueError, match='Manual slot stop intent changed'):
        activate(path, sha256(path))
    assert (root/'state.json').read_bytes() == original
    assert not (root/f'state_before_{REVISION}.json').exists()


@pytest.mark.parametrize('topology', ['a3_b3', 'b5_a1', 'b5_a2'])
def test_activate_uses_reviewed_sessions_preserves_pause_and_archives_stops(tmp_path, monkeypatch, topology):
    root = tmp_path/'campaign'; root.mkdir()
    (tmp_path/'evaluation').mkdir()
    plan, original = source_state()
    original['slots'][6]['status'] = 'pause_unknown_failure'
    new, state = revised_state(plan, original, {1}, dict.fromkeys(range(28), 3), topology)
    revision, target = subscription_migration.migration_spec(topology)
    write_json(root/'state.json', original)
    for name in ('STOP', 'STOP_SLOT_1', 'STOP_SLOT_2'):
        write_json(root/name, dict(reason='reviewed-'+name))
    new.update(shared_root=str(tmp_path), migration_revision=revision, migration_topology=topology,
               parent_plan_sha256=original['approved_sha256'],
               migration_parent_state_sha256=sha256(root/'state.json'),
               migration_global_stop_sha256=sha256(root/'STOP'),
               migration_stop_hashes={name: sha256(root/name) for name in ('STOP_SLOT_1', 'STOP_SLOT_2')},
               migration_state_payload_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest())
    path = root/'plan.json'; write_json(path, new)
    state['approved_sha256'] = sha256(path)
    write_json(root/f'state_{revision}_prepared.json', state)
    checked = []
    monkeypatch.setattr(subscription_migration, 'validate_campaign_sessions',
                        lambda shared, names: checked.append(names))
    activate(path, sha256(path))
    assert checked == [tuple(target.values())]
    assert json.loads((root/'state.json').read_text()) == state
    assert json.loads((root/f'state_before_{revision}.json').read_text()) == original
    if topology == 'b5_a2':
        assert state['retired_slots'][-1] == original['slots'][6]
        assert state['slots'][-1]['id'] == 7
        assert state['slots'][-1]['auth_profile'] == 'codex_a_2'
        assert state['slots'][-1]['index'] == original['slots'][6]['index']
        assert state['slots'][-1]['attempt'] == 3
        assert 6 not in [s['id'] for s in state['slots']]
    else:
        assert state['slots'][-1] == original['slots'][6]
    for name in ('STOP', 'STOP_SLOT_1', 'STOP_SLOT_2'):
        assert not (root/name).exists()
        assert (root/f'{name}.before_{revision}.json').exists()
