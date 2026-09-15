"""CPU contract checks for the independent RoboDojo pipeline; no model/physics."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from .io import InputError
from .pi05_server.client import Pi05Client
from .pi05_server.checkpoint import ROBODOJO_CONFIG
from .robodojo_server.client import RoboDojoTools, validate_dual_response
from .skill.schema import response_schema
from .skill.workspace import prepare_workspace


def test_identity_is_shared_and_original_gate_is_preserved():
    from . import settings
    from .. import settings as required
    root = Path(__file__).parent
    assert (required.MODEL, required.EFFORT) == ('gpt-6-astra', 'xhigh')
    assert (settings.PROVIDER, settings.MODEL, settings.EFFORT) == ('LiteLLM', 'gpt-6-astra', 'xhigh')
    assert settings.BASE_URL == 'https://gateway.example.invalid'
    assert (root/'skill/gate_prompt.md').read_bytes() == (root.parent/'robolab/skill/gate_prompt.md').read_bytes()

POSE = dict(position=[0., 0., 0.], quaternion_wxyz=[1., 0., 0., 0.], gripper_closed=False)


def response(request, mode='student'):
    return dict(request_id=request['request_id'], mode=mode, steps=2, reason='Synthetic contract test.',
        target={arm: copy.deepcopy(POSE) for arm in ('left', 'right')},
        edit={arm: dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='keep')
              for arm in ('left', 'right')},
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting='Arrange digits', remaining=['Arrange']),
            current_subgoal='Arrange digits', execution_status='not_started', execution_evidence='Synthetic image.',
            expected_next_intent='Arrange', predicted_next_intent='Arrange',
            intent_status='aligned' if mode == 'student' else 'misaligned', intent_evidence='Synthetic FK.'))


class Student:
    metadata = dict(checkpoint='/test/checkpoint', checkpoint_sha256='a'*64,
        config=ROBODOJO_CONFIG, action_horizon=50, action_dim=14)

    def infer(self, obs, output):
        actions = np.zeros((50, 14), np.float32)
        actions[:, [6, 13]] = .3
        np.savez_compressed(output, actions=actions, raw_actions=actions)
        return actions, dict(prediction_id='prediction', inference_index=0, inference_seconds=0)


class Sim:
    def __init__(self, *args, **kwargs):
        self.tick, self.calls = 0, []

    def request(self, op, **args):
        self.calls.append(op)
        if op == 'metadata':
            return dict(task='arrange_largest_number', instruction='Arrange the digits.',
                        max_episode_steps=2, control_dt=.04)
        if op == 'reset':
            return dict(episode_id='episode', step_id=0)
        if op == 'teacher_observation':
            return dict(instruction='Arrange the digits.', remaining_steps=2-self.tick,
                states=np.zeros(14, np.float32), eef_positions=np.zeros((2, 3)),
                eef_quaternions_wxyz=np.array([[1, 0, 0, 0]]*2),
                **{k: np.zeros((8, 8, 3), np.uint8) for k in RoboDojoTools.camera_keys})
        if op == 'fk_preview':
            return dict(trajectory=[{arm: dict(POSE, gripper_opening=.3) for arm in ('left', 'right')}
                                    for _ in range(50)], measured_fk_check=dict(passed=True))
        if op == 'eef_joint_target':
            return dict(action=np.zeros(14), diagnostics=dict(valid=True))
        if op == 'record_codex_decision':
            return dict(physical_steps=0, record_path='/test/decision.json')
        if op == 'chunk_step':
            rows = []
            for action in args['actions']:
                self.tick += 1
                rows.append(dict(valid=True, executed_action=action, obs=dict(states=np.zeros(14)),
                    terminated=False, truncated=self.tick >= 2, success=False))
            return dict(step_id=self.tick, steps=rows)
        if op == 'finish_pilot':
            return dict(step_id=self.tick, success=False, terminated=False, truncated=self.tick >= 2)
        return {}


def invoke(tools, **extra):
    args = tools.next_call().copy()
    name = args.pop('tool'); args.update(extra)
    return dict(robodojo_start=tools.start, pi05_infer=tools.infer, robodojo_execute=tools.execute)[name](**args)


@pytest.mark.parametrize('limit', [0, 1])
def test_zero_decision_limit_waits_for_native_terminal(tmp_path, limit):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim, max_decisions=limit)
    invoke(tools)
    invoke(tools)
    decision = response(tools.request)
    decision['steps'] = 1
    invoke(tools, response=decision)
    if limit == 1:
        assert tools.phase == 'done'
        assert not json.loads((tmp_path/'result.json').read_text())['complete']
    else:
        assert tools.phase == 'infer' and not (tmp_path/'result.json').exists()
        invoke(tools)
        decision = response(tools.request)
        decision['steps'] = 1
        decision['assessment']['execution_status'] = 'progressing'
        invoke(tools, response=decision)
        assert tools.phase == 'done'
        assert json.loads((tmp_path/'result.json').read_text())['complete']


class BenchmarkSim(Sim):
    def request(self, op, **args):
        result = super().request(op, **args)
        if op == 'reset':
            result['metadata'] = dict(evaluation_case=dict(case_id='synthetic_fixed_case'))
        return result


def test_benchmark_rejects_model_stop_without_execution_or_consuming_proposal(tmp_path):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(),
                         rpc_factory=BenchmarkSim, max_decisions=0)
    observation = invoke(tools)
    proposal = invoke(tools)
    assert observation['require_native_termination'] is True
    assert proposal['require_native_termination'] is True
    assert proposal['remaining_steps'] == 2
    assert json.loads((tmp_path/'run.json').read_text())['require_native_termination'] is True
    stop = response(tools.request, 'stop')
    stop['steps'] = 0
    before = list(tools.sim.calls)
    with pytest.raises(InputError, match='requires native termination'):
        invoke(tools, response=stop)
    assert tools.sim.calls == before
    assert tools.tick == 0 and tools.phase == 'execute' and not tools.history
    assert not (tmp_path/'response_000.json').exists()
    assert not (tmp_path/'result.json').exists()
    assert not (tmp_path/'observations/001').exists()
    assert tools.request['request_id'] == proposal['request_id']
    # The same, still-unexecuted proposal can be corrected without a reset or
    # extra inference. The real native timeout, even a failure, ends the case.
    result = invoke(tools, response=response(tools.request))
    assert result['rollout_finished'] and result['result']['complete']
    assert result['result']['truncated'] and not result['result']['success']
    assert tools.counters['predictions'] == 1
    assert tools.sim.calls.count('reset') == 1


def test_nonbenchmark_model_stop_and_benchmark_operator_abort_remain_incomplete(tmp_path):
    (tmp_path/'standalone').mkdir()
    standalone = RoboDojoTools(tmp_path/'standalone', 'arrange_largest_number', Student(), rpc_factory=Sim)
    invoke(standalone)
    invoke(standalone)
    stop = response(standalone.request, 'stop')
    stop['steps'] = 0
    result = invoke(standalone, response=stop)
    assert result['rollout_finished'] and not result['result']['complete']
    assert 'chunk_step' not in standalone.sim.calls
    for reason in ('operator_stop', 'controller_error', 'physics_error'):
        (tmp_path/reason).mkdir()
        tools = RoboDojoTools(tmp_path/reason, 'arrange_largest_number', Student(),
                             rpc_factory=BenchmarkSim, max_decisions=0)
        invoke(tools)
        result = tools.finish(reason)
        assert tools.phase == 'done' and not result['complete']
        assert 'chunk_step' not in tools.sim.calls


@pytest.mark.parametrize('mode', ['student', 'eef', 'edit'])
def test_dual_recorded_sequence(tmp_path, mode):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim)
    obs = invoke(tools)
    assert len(obs['images']) == 3 and set(obs['current_eef']) == {'left', 'right'}
    proposal = invoke(tools)
    assert len(proposal['student_eef_trajectory']) == 50
    decision = response(tools.request, mode)
    if mode == 'edit':
        assert tools.correction_targets(decision)[0]['left']['gripper_opening'] == .3
    result = invoke(tools, response=decision)
    assert result['rollout_finished'] and result['result']['truncated']
    assert tools.sim.calls.index('record_codex_decision') < tools.sim.calls.index('chunk_step')
    assert tools.history[0]['discarded_student_steps'] == (48 if mode == 'student' else 50)
    with np.load(tmp_path/'execution_000.npz') as data:
        assert data['actions'].shape == (2, 14)
    with pytest.raises(InputError):
        tools.execute(**dict(output_dir=str(tmp_path/'replay'), response=decision))


def test_both_arm_bounds_and_gate(tmp_path):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim)
    invoke(tools); invoke(tools)
    decision = response(tools.request, 'eef')
    validate_dual_response(decision, tools.request)
    decision['target']['right']['position'][0] = .06
    with pytest.raises(ValueError, match='5 cm'):
        validate_dual_response(decision, tools.request)
    decision = response(tools.request, 'eef')
    decision['assessment']['intent_status'] = 'uncertain'
    with pytest.raises(ValueError):
        validate_dual_response(decision, tools.request)
    decision = response(tools.request, 'student')
    decision['reason'] = '必须拒绝非英文公开说明'
    with pytest.raises(ValueError, match='English'):
        validate_dual_response(decision, tools.request)


def test_pi05_native_input_and_continuous_gripper(tmp_path):
    class Policy:
        def __init__(self, *args):
            self._ws = SimpleNamespace(ping_timeout=20, close=lambda: None)
        def get_server_metadata(self):
            return dict(Student.metadata, backend='OpenPI/JAX', inferences=0)
        def infer(self, obs):
            assert obs['state'].shape == (14,)
            assert set(obs['images']) == set(RoboDojoTools.camera_keys)
            assert obs['images']['cam_high'].shape == (3, 8, 8)
            actions = np.zeros((50, 14), np.float32)
            actions[:, 6], actions[:, 13] = .3, 1.2
            return dict(actions=actions, policy_identity=dict(checkpoint_sha256='a'*64, inference_index=0))
    client = Pi05Client(1, '/test/checkpoint', client_cls=Policy)
    obs = Sim().request('teacher_observation')
    actions, _ = client.infer(obs, tmp_path/'proposal.npz')
    assert np.allclose(actions[:, 6], .3) and np.all(actions[:, 13] == 1.)
    client.close()


def test_independent_workspace_and_schema(tmp_path):
    root = Path(__file__).parent
    agent = prepare_workspace(tmp_path, root/'skill')
    skill = agent/'.agents/skills/robodojo-hybrid-rollout'
    assert (skill/'SKILL.md').is_file()
    assert not (agent/'.agents/skills/robolab-hybrid-rollout').exists()
    assert 'scratch' in json.loads((agent/'workspace.json').read_text())
    assert 'every public explanation' in (agent/'AGENTS.md').read_text()
    for field in ('edit', 'target'):
        assert response_schema()['properties'][field]['required'] == ['left', 'right']
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(not n.name.startswith(('robodojo_rollout', 'rlinf_robolab')) for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('robodojo_rollout', 'rlinf_robolab'))
