import copy

import pytest

from .release_guard_slots import release_state


def sample():
    return dict(slots=[
        dict(id=0, status='running', index=0, assigned=[0], job_id='live'),
        dict(id=10, status='pause_unknown_failure', index=10, assigned=[10], job_id='failed'),
        dict(id=11, status='stopped', index=11, assigned=[11], submission='started'),
        dict(id=12, status='stopped', index=12, assigned=[12], submission='prepared'),
        dict(id=13, status='pause_terminal_artifacts', index=13, assigned=[13]),
    ], pending_retries=[dict(index=2, attempt=1)])


def test_defer_keeps_case_reserved_and_does_not_replay_or_drop_evidence():
    state = sample(); old = copy.deepcopy(state)
    new = release_state(state, {10: 'failed'}, {'started'})
    assert state == old
    assert new['slots'][0] == old['slots'][0]
    assert new['slots'][1]['status'] == 'idle' and new['slots'][1]['assigned'] == [10]
    assert new['deferred_cases']['10']['original_slot'] == old['slots'][1]
    assert new['slots'][2]['status'] == 'submitted'
    assert new['slots'][3]['status'] == 'ready'
    assert new['slots'][4] == old['slots'][4]
    assert new['pending_retries'] == old['pending_retries']


@pytest.mark.parametrize('status', ['running', 'completed', 'stopped', 'idle'])
def test_reject_nonpaused_target(status):
    state = sample(); state['slots'][1]['status'] = status
    with pytest.raises(ValueError): release_state(state, {10: 'failed'}, set())


def test_reject_duplicate_or_wrong_job():
    state = sample()
    with pytest.raises(ValueError): release_state(state, {10: 'wrong'}, set())
    state['pending_retries'].append(dict(index=10))
    with pytest.raises(ValueError): release_state(state, {10: 'failed'}, set())
