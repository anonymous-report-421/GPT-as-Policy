import copy

import pytest

from .repair_io_campaign import repaired_state


def sample_state():
    return dict(slots=[
        dict(id=0, index=12, attempt=1, status='running', submission='live', assigned=[0,12]),
        dict(id=7, index=7, attempt=0, retry_number=0, status='pause_unknown_failure',
             submission='failed-io', job_id='pt-io', platform_state='FAILED', assigned=[7]),
        dict(id=10, index=10, attempt=0, status='pause_unknown_failure', submission='bad-input', assigned=[10]),
        dict(id=11, index=11, attempt=0, status='stopped', submission='completed', assigned=[11]),
        dict(id=12, index=15, attempt=1, status='stopped', submission=None, assigned=[15]),
    ], pending_retries=[dict(index=15, attempt=1, retry_number=1, next_retry_at=100)])


def test_only_startup_io_failure_requeued_originals_untouched():
    state = sample_state(); before = copy.deepcopy(state)
    new = repaired_state(state, retry_index=7, started_paths={'live','completed'})
    assert state == before
    assert new['slots'][0] == state['slots'][0]
    assert new['slots'][2] == state['slots'][2]  # Do not loosen/retry model guard failures.
    assert new['slots'][1]['status'] == 'ready'
    assert new['slots'][1]['attempt'] == 1
    assert new['slots'][1]['assigned'] == [7]
    assert new['slots'][3]['status'] == 'submitted'  # Reconcile, not a second create.
    assert new['slots'][4]['status'] == 'idle'
    assert new['pending_retries'][0] == state['pending_retries'][0]
    assert len(new['pending_retries']) == 1  # Prebuilt B3 retry cannot migrate auth.


def test_do_not_adopt_ambiguous_unsubmitted_source():
    with pytest.raises(ValueError, match='unsubmitted'):
        repaired_state(sample_state(), retry_index=7, started_paths=set())


def test_do_not_duplicate_repair():
    state = sample_state(); state['pending_retries'].append(dict(index=7))
    with pytest.raises(ValueError, match='already queued'):
        repaired_state(state, retry_index=7, started_paths={'completed'})


@pytest.mark.parametrize('status', ['running', 'completed', 'stopped', 'idle'])
def test_never_repair_active_completed_or_unrelated_status(status):
    state=sample_state(); state['slots'][1]['status']=status
    with pytest.raises(ValueError, match='exactly one paused'):
        repaired_state(state, retry_index=7, started_paths={'completed'})
