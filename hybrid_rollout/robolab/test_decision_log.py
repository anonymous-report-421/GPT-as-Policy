"""Decision logging tests: no model, simulator, or GPU required."""
import json

import pytest

from .io import write_json
from .robolab_server.decision_log import mirror_available
from .robolab_server.server import EditEnvironmentSession
from .test_contract import invoke_next, make, response


def test_simulator_records_full_decision_without_stepping(tmp_path, capsys):
    session = object.__new__(EditEnvironmentSession)
    session.episode_id, session.step_id = 'episode', 0
    session.episode_dir, session.poisoned = tmp_path, False
    value = response(dict(request_id='req', step_id=0))
    value['reason'] = '沿用策略；只执行前两步。'
    args = dict(episode_id='episode', step_id=0, decision=0, prediction_id='prediction', response=value)
    receipt = session.dispatch('record_codex_decision', args)
    saved = json.loads((tmp_path/'codex_decisions/000.json').read_text())
    assert saved['response'] == value and saved['reason'] == value['reason']
    assert saved['prefix_only'] and not saved['codex_override']
    assert saved['numeric_action_changed'] is None
    assert receipt['physical_steps'] == session.step_id == 0
    assert json.loads(capsys.readouterr().out)['record_path'] == receipt['record_path']
    with pytest.raises(ValueError, match='duplicate'):
        session.dispatch('record_codex_decision', args)
    with pytest.raises(ValueError, match='Stale'):
        session.dispatch('record_codex_decision', dict(args, decision=1, step_id=1))
    assert not (tmp_path/'codex_decisions/001.json').exists()


def test_receipt_failure_prevents_actions(tmp_path):
    tools = make(tmp_path)
    invoke_next(tools)
    invoke_next(tools)
    original = tools.sim.request
    def fail(op, **args):
        if op == 'record_codex_decision':
            raise OSError('disk full')
        return original(op, **args)
    tools.sim.request = fail
    with pytest.raises(OSError, match='disk full'):
        invoke_next(tools, response=response(tools.request))
    assert tools.tick == 0 and 'chunk_step' not in tools.sim.calls


def test_mirror_is_explicit_and_does_not_repeat(tmp_path, capsys):
    request = dict(request_id='req', step_id=5, episode_id='episode', prediction_id='p')
    value = response(request, mode='eef')
    write_json(tmp_path/'request_000.json', request)
    write_json(tmp_path/'response_000.json', value)
    before = (tmp_path/'response_000.json').read_bytes()
    seen = set()
    mirror_available(tmp_path, seen)
    mirror_available(tmp_path, seen)
    event = json.loads(capsys.readouterr().out)
    assert event['origin'] == 'controller_record_mirror'
    assert 'not_server_receipt' in event['status']
    assert event['codex_override'] and not event['prefix_only']
    assert (tmp_path/'response_000.json').read_bytes() == before


def test_stop_is_recorded_before_finish(tmp_path):
    tools = make(tmp_path)
    invoke_next(tools)
    invoke_next(tools)
    value = response(tools.request, mode='stop')
    value['steps'] = 0
    invoke_next(tools, response=value)
    assert tools.sim.decisions[0]['response'] == value
    assert tools.sim.calls.index('record_codex_decision') < tools.sim.calls.index('finish_pilot')
    assert 'chunk_step' not in tools.sim.calls
