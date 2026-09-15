"""Focused CPU checks; never contact a model, GPU, or real simulator."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .io import InputError
from .robolab_server.client import RolloutTools
from .robolab_server.gate_assessment import GATE_INSTRUCTION
from .skill.run import CodexPolicy, content_items


def response(request, mode='student'):
    return dict(request_id=request['request_id'], mode=mode, steps=2, reason='Synthetic test evidence',
        target=dict(position=[0, 0, .01], quaternion_wxyz=[1, 0, 0, 0], gripper_closed=False),
        edit=dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='keep'),
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting='approach', remaining=['stack']),
            current_subgoal='approach', execution_status='not_started' if request['step_id'] == 0 else 'progressing',
            execution_evidence='test images', expected_next_intent='approach', predicted_next_intent='approach',
            intent_status='aligned', intent_evidence='test FK'))


class Student:
    metadata = dict(checkpoint='/test/checkpoint', checkpoint_sha256='a'*64)
    def __init__(self):
        self.calls = 0
    def infer(self, obs, output):
        actions = np.zeros((15, 8), np.float32)
        np.savez_compressed(output, actions=actions, raw_actions=actions,
                            **{k: obs[k] for k in ('main_images', 'wrist_images', 'states')})
        self.calls += 1
        return actions, dict(prediction_id=str(self.calls), inference_index=self.calls-1, inference_seconds=0)


class Sim:
    def __init__(self, *args, **kwargs):
        self.tick = 0
        self.calls = []
        self.decisions = []
    def request(self, op, **args):
        self.calls.append(op)
        if op == 'metadata':
            return dict(task='blocks', instruction='stack four blocks', max_episode_steps=2, control_dt=.1)
        if op == 'reset':
            return dict(episode_id='episode', step_id=0)
        if op == 'teacher_observation':
            return dict(instruction='stack four blocks', remaining_steps=max(0, 2-self.tick),
                states=np.zeros(8, np.float32), ee_pos=np.zeros(3), ee_quat_wxyz=np.array([1,0,0,0]),
                main_images=np.zeros((4,4,3), np.uint8), wrist_images=np.zeros((4,4,3), np.uint8),
                main_rgb=np.zeros((4,4,3), np.uint8), wrist_rgb=np.zeros((4,4,3), np.uint8))
        if op == 'fk_preview':
            return dict(trajectory=[dict(position=[0,0,0], quaternion_wxyz=[1,0,0,0], gripper_closed=False)]*15,
                        measured_fk_check=dict(passed=True))
        if op == 'eef_joint_target':
            return dict(action=np.zeros(8), diagnostics=dict(valid=True))
        if op == 'record_codex_decision':
            self.decisions.append(args)
            return dict(record_path='/test/sim/codex_decisions/000.json', physical_steps=0)
        if op == 'chunk_step':
            steps = []
            for action in args['actions']:
                self.tick += 1
                steps.append(dict(valid=True, terminated=False, truncated=self.tick >= 2, success=False,
                                  executed_action=action, obs=dict(states=np.zeros(8))))
            return dict(step_id=self.tick, steps=steps)
        if op == 'finish_pilot':
            return dict(episode_id='episode', step_id=self.tick, success=False,
                        terminated=False, truncated=self.tick >= 2, reason=args['reason'])
        return {}
    def close(self):
        pass


def make(tmp_path):
    output = tmp_path/'controller'
    output.mkdir()
    return RolloutTools(output, 'blocks', Student(), rpc_factory=Sim)


def invoke_next(tools, **extra):
    arguments = tools.next_call().copy()
    name = arguments.pop('tool')
    arguments.update(extra)
    return dict(robolab_start=tools.start, pi05_infer=tools.infer, robolab_execute=tools.execute)[name](**arguments)


def test_blocking_sequence_and_timeout_persistence(tmp_path):
    tools = make(tmp_path)
    before = invoke_next(tools)
    assert tools.student.calls == 0 and tools.tick == 0
    proposal = invoke_next(tools)
    assert tools.student.calls == 1 and tools.tick == 0
    after = invoke_next(tools, response=response(tools.request))
    assert after['result']['truncated'] and after['result']['complete']
    assert after['rollout_finished'] and after['next_call'] is None
    assert Path(before['observation_path']).is_file() and Path(proposal['proposal_path']).is_file()
    assert (tools.output/'execution_000.npz').is_file()
    assert json.loads((tools.output/'result.json').read_text())['student_steps'] == 2
    assert tools.sim.decisions[0]['response'] == response(tools.request)
    assert tools.sim.calls.index('record_codex_decision') < tools.sim.calls.index('chunk_step')
    assert tools.history[0]['simulator_decision_receipt']['physical_steps'] == 0
    assert len(content_items(before, images=True)) == 3
    with pytest.raises(InputError):
        tools.execute(proposal_path=proposal['proposal_path'], response=response(tools.request), output_dir='/tmp/replay')


def test_no_execution_without_fresh_infer_and_invalid_inputs_do_not_mutate(tmp_path):
    tools = make(tmp_path)
    invoke_next(tools)
    with pytest.raises(InputError):
        tools.execute(output_dir='/tmp/invalid', response={})
    args = tools.next_call().copy()
    args.pop('tool')
    args['observation_path'] = '/tmp/stale.json'
    with pytest.raises(InputError):
        tools.infer(**args)
    assert tools.student.calls == 0 and tools.tick == 0
    invoke_next(tools)
    with pytest.raises(InputError):
        invoke_next(tools, response=response(tools.request, mode='eef'))
    assert 'chunk_step' not in tools.sim.calls
    assert 'record_codex_decision' not in tools.sim.calls
    decision = response(tools.request, mode='eef')
    decision['assessment']['intent_status'] = 'misaligned'
    decision['target']['position'] = [0, 0, 1]
    with pytest.raises(InputError):
        invoke_next(tools, response=decision)
    assert 'eef_joint_target' not in tools.sim.calls
    decision['target']['position'] = [0, 0, .01]
    invoke_next(tools, response=decision)
    assert tools.sim.calls.count('eef_joint_target') == 2
    assert tools.counters['recovery_steps'] == 2


def test_independent_imports_and_exact_copied_gate():
    root = Path(__file__).parent
    assert (root/'skill/gate_prompt.md').read_text() == GATE_INSTRUCTION
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('rlinf_robolab'), path
            if isinstance(node, ast.Import):
                assert not any(n.name.startswith('rlinf_robolab') for n in node.names), path


class Transport:
    def __init__(self, argv, workspace):
        self.process = SimpleNamespace(pid=123)
        self.stage = 0
        self.packet = None
        self.replies = []
    def request(self, method, params, timeout):
        if method == 'thread/start':
            assert len(params['dynamicTools']) == 3
            assert GATE_INSTRUCTION in params['developerInstructions']
            self.thread_params = params
            return dict(thread=dict(id='thread'), model='gpt-6-astra', reasoningEffort='xhigh')
        if method == 'turn/start':
            assert params['model'] == 'gpt-6-astra' and params['effort'] == 'xhigh'
            self.first = json.loads(params['input'][0]['text'].split('First call: ')[1])
            return dict(turn=dict(id='turn'))
        return {}
    def notify(self, *args):
        pass
    def next_message(self, timeout):
        if self.stage == 3:
            return dict(method='turn/completed', params=dict(threadId='thread', turn=dict(id='turn', status='completed')))
        arguments = dict(self.first if self.stage == 0 else self.packet['next_call'])
        name = arguments.pop('tool')
        if name == 'robolab_execute':
            arguments['response'] = response(self.packet)
        self.stage += 1
        return dict(id=self.stage, method='item/tool/call', params=dict(
            threadId='thread', turnId='turn', callId=str(self.stage), tool=name, arguments=arguments))
    def reply(self, identifier, result):
        assert result['success']
        self.replies.append(result)
        self.packet = json.loads(result['contentItems'][0]['text'])
    def close(self):
        pass


def test_codex_dispatches_all_three_tools_with_images(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    worker = CodexPolicy(tmp_path/'workspace', 'codex', transport_factory=Transport)
    tools = make(tmp_path)
    worker.run(tools)
    assert tools.phase == 'done' and tools.student.calls == 1
    assert [len(r['contentItems']) for r in worker.transport.replies] == [3, 1, 3]
    assert (tmp_path/'workspace/call_0002_result.json').exists()


def test_retryable_model_transport_notice_does_not_repeat_tools(tmp_path, monkeypatch):
    from .skill import run
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='codex-fixture'))
    class RetryTransport(Transport):
        sent_retry = False
        def next_message(self, timeout):
            if not self.sent_retry:
                self.sent_retry = True
                return dict(method='error', params=dict(willRetry=True, error=dict(message='Reconnecting')))
            return super().next_message(timeout)
    worker = CodexPolicy(tmp_path/'workspace', 'codex', transport_factory=RetryTransport)
    tools = make(tmp_path)
    worker.run(tools)
    assert tools.student.calls == 1 and tools.sim.calls.count('chunk_step') == 1
