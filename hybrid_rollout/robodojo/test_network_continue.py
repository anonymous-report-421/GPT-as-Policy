import copy
import json

import pytest

from .io import InputError
from .skill import run
from .skill.network_recovery import closed_network_turn, is_network_error

NET = {'message': 'stream disconnected before completion: network error: error decoding response body'}


def event(method, turn_id='t0', **fields):
    return dict(method=method, params=dict(threadId='thread', turnId=turn_id, **fields))


def completed(turn='t0', status='failed', error=NET):
    return event('turn/completed', turn, turn=dict(id=turn, status=status, error=error))


class Transport:
    def __init__(self, events, snapshot=None):
        self.events = list(events); self.requests = []; self.replies = []; self.turns = 0
        self.snapshot = snapshot

    def request(self, method, params, timeout):
        self.requests.append((method, params))
        if method == 'thread/read': return self.snapshot
        assert method == 'turn/start'
        identity = f't{self.turns}'; self.turns += 1
        return dict(turn=dict(id=identity))

    def next_message(self, timeout):
        assert self.events, 'Unexpected wait or duplicate continue'
        e = self.events.pop(0)
        if isinstance(e, BaseException): raise e
        return e

    def reply(self, identifier, result): self.replies.append((identifier, result))


class Rollout:
    phase = 'execute'; tick = 45; counters = {}; observation_path = '/evidence/observation.json'

    def __init__(self, reject=False): self.actions = 0; self.reject = reject
    def next_call(self): return dict(tool='robodojo_execute', proposal_path='/evidence/actions.npz')
    def start(self, **kw): raise AssertionError('No simulator reset')
    def infer(self, **kw): raise AssertionError('No repeated inference')
    def execute(self, **kw):
        if self.reject: raise InputError('Recovery EEF target exceeds 5 cm')
        self.actions += 1; self.tick += 1; self.phase = 'done'
        return dict(step_id=self.tick, rollout_finished=True)


def call(turn='t1', identity=9):
    e = event('item/tool/call', turn, callId=str(identity), tool='robodojo_execute', arguments={})
    return dict(e, id=identity)


def worker(tmp_path, monkeypatch, events, snapshot=None):
    monkeypatch.setattr(run, 'NETWORK_CONTINUE_DELAYS', (0,) * 20)
    w = object.__new__(run.CodexPolicy)
    w.workspace = tmp_path; w.agent_workspace = tmp_path; w.thread_id = 'thread'
    w.method = 'pi05_plus_gpt'; w.timeout = 1; w.image_max_edge = 480
    w.transport = Transport(events, snapshot)
    return w


def test_same_thread_continue_after_final_network_error_no_replay(tmp_path, monkeypatch):
    w = worker(tmp_path, monkeypatch, [event('error', error=NET, willRetry=False),
        completed(), event('error', error=NET, willRetry=False), call(),
        completed('t1', 'completed', None)])
    rollout = Rollout(); w.run(rollout)
    assert rollout.actions == 1 and rollout.tick == 46
    turns = [p for m,p in w.transport.requests if m == 'turn/start']
    assert len(turns) == 2 and all(p['threadId'] == 'thread' for p in turns)
    assert all(p['model'] == run.MODEL and p['effort'] == run.EFFORT for p in turns)
    assert turns[1]['input'][0]['text'].startswith('Continue the same rollout.')
    assert '/evidence/observation.json' in turns[1]['input'][0]['text']
    assert json.loads((tmp_path/'network_continue_0000.json').read_text())['physical_actions_replayed'] == 0
    assert len(list(tmp_path.glob('turn_*_completed.json'))) == 2


def test_internal_retry_that_recovers_is_not_interrupted(tmp_path, monkeypatch):
    w=worker(tmp_path,monkeypatch,[event('error',error=NET,willRetry=True),call('t0'),completed('t0','completed',None)])
    w.run(Rollout()); assert w.transport.turns == 1


@pytest.mark.parametrize('error', [dict(message='Usage limit reached'),
    dict(message='Authentication failed'), dict(message='Recovery EEF target exceeds 5 cm'),
    dict(message='Selected model is at capacity',codexErrorInfo='serverOverloaded')])
def test_nonnetwork_failure_does_not_trigger_continue(tmp_path,monkeypatch,error):
    w=worker(tmp_path,monkeypatch,[event('error',error=error,willRetry=False)])
    with pytest.raises(RuntimeError): w.run(Rollout())
    assert w.transport.turns == 1


def test_operator_interruption_not_woken(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[event('error',error=NET,willRetry=True),completed(status='interrupted',error=None)])
    with pytest.raises(RuntimeError): w.run(Rollout())
    assert w.transport.turns == 1


@pytest.mark.parametrize('state', ['idle','active','systemError'])
def test_timeout_probe_only_wakes_confirmed_idle_failed_turn(tmp_path,monkeypatch,state):
    snapshot=dict(thread=dict(id='thread',status=dict(type=state),turns=[dict(id='t0',status='failed',error=NET)]))
    events=[event('error',error=NET,willRetry=False),TimeoutError('missing terminal notification')]
    if state=='idle':events += [call(),completed('t1','completed',None)]
    w=worker(tmp_path,monkeypatch,events,snapshot)
    if state=='idle':w.run(Rollout()); assert w.transport.turns==2
    else:
        with pytest.raises(TimeoutError):w.run(Rollout())
        assert w.transport.turns==1


def test_twenty_consecutive_wakeups_then_existing_failure_path(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[completed(f't{i}') for i in range(21)])
    with pytest.raises(RuntimeError,match='exhausted'):w.run(Rollout())
    assert w.transport.turns==21  # Original turn plus exactly twenty continues.
    assert len(list(tmp_path.glob('network_continue_*.json')))==20


def test_native_done_never_uses_an_extra_model_call(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[call('t0'),event('error',error=NET,willRetry=False)])
    rollout=Rollout();w.run(rollout)
    assert rollout.actions==1 and w.transport.turns==1


def test_rejections_do_not_restart_or_consume_network_continue_budget(tmp_path,monkeypatch):
    w=worker(tmp_path,monkeypatch,[call('t0',i) for i in range(12)] +
        [completed('t0')] + [completed(f't{i}') for i in range(1,21)])
    rollout=Rollout(reject=True)
    with pytest.raises(RuntimeError,match='network continue exhausted'):w.run(rollout)
    assert rollout.actions==0 and w.transport.turns==21
    assert len(w.transport.replies)==12
    assert len(list(tmp_path.glob('call_*_result.json')))==12


def test_network_classifier_and_snapshot_identity():
    assert is_network_error(dict(message='Error running remote compact task: stream disconnected before completion'))
    assert not is_network_error(dict(message='network error: quota exhausted'))
    assert closed_network_turn(dict(thread=dict(id='another')), 'thread','t0') is None
