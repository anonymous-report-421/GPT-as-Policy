import copy

import pytest

from .activate_input_recovery import restore_slots


@pytest.mark.parametrize('status', ['idle', 'ready', 'submitting', 'submitted', 'running', 'drained'])
def test_stop_restores_exact_prior_status_including_idle_with_old_job(status):
    before = dict(slots=[dict(id=0, status=status, index=2, attempt=4,
        submission='/old/submission.json', job_id='old-job', auth_profile='codex_a', assigned=[1, 2])],
        pending_retries=[dict(index=9, attempt=3)], quota_guard=dict(blocked_profiles=['codex_a']))
    stopped = copy.deepcopy(before)
    stopped['slots'][0]['status'] = 'stopped'
    result, restored = restore_slots(before, stopped)
    assert result == before and restored == [0]
    assert stopped['slots'][0]['status'] == 'stopped'


def test_ready_claim_without_submission_is_not_lost():
    before = dict(slots=[dict(id=14, index=25, attempt=4, status='ready', submission=None)])
    result, _ = restore_slots(before, dict(slots=[dict(before['slots'][0], status='stopped')]))
    assert result == before


@pytest.mark.parametrize('field,value', [('index', 4), ('attempt', 8), ('submission', '/different'), ('auth_profile', 'codex_c')])
def test_changed_reservation_fails_closed(field, value):
    before = dict(slots=[dict(id=0, index=1, attempt=2, status='running', submission='sub', auth_profile='codex_b')])
    stopped = copy.deepcopy(before)
    stopped['slots'][0].update(status='stopped', **{field: value})
    with pytest.raises(ValueError, match='reservation changed'):
        restore_slots(before, stopped)


def test_operator_stop_not_released():
    state = dict(slots=[dict(id=0, status='stopped')])
    with pytest.raises(ValueError, match='operator'):
        restore_slots(state, state)


def test_new_completion_while_waiting_is_preserved():
    before = dict(slots=[dict(id=0, status='running', index=1)])
    after = dict(slots=[dict(id=0, status='idle', index=1, history=['completed'])])
    assert restore_slots(before, after) == (after, [])
