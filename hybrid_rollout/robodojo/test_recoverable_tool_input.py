import json

import pytest

from .io import InputError
from .skill import run
from .test_network_continue import Rollout, call, completed, worker


@pytest.mark.parametrize('method', ['pi05_plus_gpt', 'gpt_only'])
def test_more_than_five_rejections_then_valid_action_same_episode(tmp_path, monkeypatch, method):
    class CorrectingRollout(Rollout):
        request = dict(request_id='current', current_eef={})

        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, **kw):
            self.calls += 1
            if self.calls <= 12:
                raise InputError('Response must match the current observation request_id')
            return super().execute(**kw)

        act = execute

    events = [call('t0', i) for i in range(13)] + [completed('t0', 'completed', None)]
    if method == 'gpt_only':
        for e in events[:-1]:
            e['params']['tool'] = 'robodojo_act'
    w = worker(tmp_path, monkeypatch, events)
    w.method = method
    rollout = CorrectingRollout()
    w.run(rollout)
    assert rollout.actions == 1 and rollout.tick == 46
    assert w.transport.turns == 1
    assert [result['success'] for _, result in w.transport.replies] == [False]*12 + [True]
    assert len(list(tmp_path.glob('call_*_request.json'))) == 13
    assert len(list(tmp_path.glob('call_*_result.json'))) == 13
    for i in range(12):
        result = json.loads((tmp_path/f'call_{i:04d}_result.json').read_text())
        assert result['no_execution'] is True and result['retryable'] is True
        assert result['rejected_calls'] == i+1 and result['step_id'] == 45
        assert result['request_id'] == 'current'
        assert result['controller_version'] == run.CONTROLLER_VERSION
        assert result['next_call'] == rollout.next_call()


@pytest.mark.parametrize('arguments', ['{broken json', '[]', [], None])
def test_bad_argument_container_returns_error_before_handler(tmp_path, monkeypatch, arguments):
    invalid = call('t0', 0)
    invalid['params']['arguments'] = arguments
    w = worker(tmp_path, monkeypatch, [invalid, call('t0', 1), completed('t0', 'completed', None)])
    rollout = Rollout()
    w.run(rollout)
    assert rollout.actions == 1 and w.transport.turns == 1
    packet = json.loads((tmp_path/'call_0000_result.json').read_text())
    assert packet['no_execution'] is True and 'JSON object' in packet['error']


@pytest.mark.parametrize('error', [RuntimeError('uncertain action ACK'),
    ValueError('unexpected runtime corruption'), TypeError('internal implementation error')])
def test_non_input_error_still_fatal_never_claims_no_execution(tmp_path, monkeypatch, error):
    class BrokenRollout(Rollout):
        def execute(self, **kw):
            self.actions += 1
            raise error
    w = worker(tmp_path, monkeypatch, [call('t0')])
    rollout = BrokenRollout()
    with pytest.raises(type(error), match=str(error)):
        w.run(rollout)
    assert rollout.actions == 1
    assert w.transport.replies == [] and not list(tmp_path.glob('call_*_result.json'))


def test_reply_failure_preserves_rejection_without_retrying_handler(tmp_path, monkeypatch):
    w = worker(tmp_path, monkeypatch, [call('t0')])
    def broken_reply(*args):
        raise RuntimeError('reply transport closed')
    w.transport.reply = broken_reply
    rollout = Rollout(reject=True)
    with pytest.raises(RuntimeError, match='transport closed'):
        w.run(rollout)
    assert rollout.actions == 0
    assert json.loads((tmp_path/'call_0000_result.json').read_text())['no_execution']


def test_feedback_explains_euclidean_bound_without_mutating_action():
    rollout = Rollout()
    rollout.request = dict(request_id='fresh', current_eef={
        arm: dict(position=[0, 0, 0]) for arm in ('left', 'right')})
    args = dict(response=dict(mode='eef', target={
        arm: dict(position=[.04, .04, 0]) for arm in ('left', 'right')}))
    original = json.dumps(args)
    packet = run.rejected_input(InputError('Recovery EEF target exceeds 5 cm'), rollout, args, 30)
    assert packet['eef_translation']['distance_m']['left'] == pytest.approx(0.05656854)
    assert packet['eef_translation']['max_distance_m'] == .05
    assert packet['request_id'] == 'fresh' and json.dumps(args) == original
    assert rollout.actions == 0
