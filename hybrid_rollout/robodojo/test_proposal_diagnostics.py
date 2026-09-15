"""Offline numerical and additive-payload checks; no models or physics."""
import copy
import json

import numpy as np
import pytest

from .robodojo_server.client import RoboDojoTools, validate_dual_response
from .robodojo_server.proposal_diagnostics import action_diagnostics
from .skill.run import content_items
from .test_contract import Sim, Student, invoke, response


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_matches_pilot_arithmetic_without_rounding_or_mutation(dtype):
    actions = np.random.default_rng(17).normal(size=(50, 14)).astype(dtype)
    original = actions.copy()
    actions.flags.writeable = False
    result = action_diagnostics(actions)
    assert result['status'] == 'computed' and result['finite'] is True
    assert result['shape'] == [50, 14]
    assert result['opening_range_arm_order'] == ['left', 'right']
    assert result['opening_range'] == [
        [float(actions[:, i].min()), float(actions[:, i].max())] for i in (6, 13)]
    assert result['max_joint_step_rad'] == float(np.abs(np.diff(
        actions[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], axis=0)).max())
    assert actions.tobytes() == original.tobytes()
    json.dumps(result, allow_nan=False)


def test_grippers_are_not_joint_angles_and_no_step_from_measured_state():
    actions = np.full((50, 14), 100., dtype=np.float32)
    actions[:, 6] = np.linspace(0., 1., 50)
    actions[:, 13] = np.linspace(1., 0., 50)
    result = action_diagnostics(actions)
    assert result['opening_range'] == [[0., 1.], [0., 1.]]
    assert result['max_joint_step_rad'] == 0.
    assert 'excludes measured state' in result['joint_step_scope']
    assert 'not a safety' in result['scope']
    assert not {'safe', 'success', 'aligned', 'mode', 'steps'} & result.keys()


@pytest.mark.parametrize('actions', [
    np.zeros((0, 14)), np.zeros((1, 14)), np.zeros((50, 13)), np.zeros(14),
    np.zeros((50, 14), dtype=np.int64),
    np.full((50, 14), np.nan), np.full((50, 14), np.inf), np.array([['text']]),
])
def test_invalid_diagnostic_is_unavailable_not_a_zero_or_success(actions):
    result = action_diagnostics(actions)
    assert result['status'] == 'unavailable'
    assert result['opening_range'] is None and result['max_joint_step_rad'] is None
    json.dumps(result, allow_nan=False)


def test_overflow_does_not_emit_nonfinite_json():
    actions = np.zeros((50, 14), np.float32)
    actions[0, 0] = np.finfo(np.float32).max
    actions[1, 0] = -np.finfo(np.float32).max
    result = action_diagnostics(actions)
    assert result['finite'] is True and result['status'] == 'unavailable'
    json.dumps(result, allow_nan=False)


def test_diagnostics_are_additive_and_full_observation_fk_actions_are_unchanged(tmp_path):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim)
    observation = invoke(tools)
    image_items = content_items(observation, images=True)[1:]
    proposal = invoke(tools)
    saved_request = json.loads((tmp_path/'request_000.json').read_text())
    saved_proposal = json.loads((tmp_path/'proposals/000/proposal.json').read_text())
    assert proposal == saved_proposal
    assert proposal['action_diagnostics'] == saved_request['action_diagnostics']
    assert proposal['action_diagnostics'] == action_diagnostics(tools.actions)
    assert proposal['student_eef_trajectory'] == tools.trajectory()
    assert len(proposal['student_eef_trajectory']) == 50
    assert proposal['current_state'] == observation['current_state']
    assert proposal['current_eef'] == observation['current_eef']
    assert proposal['images'] == observation['images']
    assert content_items(observation, images=True)[1:] == image_items
    assert tools.sim.calls == ['metadata', 'reset', 'begin_combination',
                               'teacher_observation', 'fk_preview']
    with np.load(tmp_path/'proposals/000/actions.npz') as saved:
        np.testing.assert_array_equal(tools.actions, saved['actions'])
        np.testing.assert_array_equal(saved['raw_actions'], saved['actions'])
    assert tools.phase == 'execute' and tools.tick == 0


@pytest.mark.parametrize('mode', ['student', 'eef', 'edit', 'stop'])
def test_diagnostics_do_not_change_gate_validation(tmp_path, mode):
    tools = RoboDojoTools(tmp_path, 'arrange_largest_number', Student(), rpc_factory=Sim)
    invoke(tools)
    invoke(tools)
    with_diagnostic = tools.request
    without_diagnostic = {k: v for k, v in tools.request.items() if k != 'action_diagnostics'}
    decision = response(with_diagnostic, mode)
    for request in (with_diagnostic, without_diagnostic):
        validate_dual_response(decision, request)
        stale = copy.deepcopy(decision)
        stale['request_id'] = 'stale'
        with pytest.raises(ValueError, match='stale'):
            validate_dual_response(stale, request)
