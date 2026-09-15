"""Regression for the missing native run_eval preamble; no model or physics."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from .robodojo_server.session import RoboDojoSession


class NativeLikeEnv:
    num_envs = 1
    step_lim = 1050
    obs_manager = SimpleNamespace(collect_freq=25)

    def __init__(self, register=True):
        self.register = register
        self.events = []

    def reset(self, seed):
        self.events.append('reset')
        self.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]])
        self.take_action_cnt, self.end_flag, self.success = [0], [False], [True]

    def run_reward(self):
        self.events.append('run_reward')
        if self.register:
            self.reward_manager.check_list[0].append(['unfinished native tower condition'])

    def get_score(self):
        self.events.append('get_score')

    def take_action(self, command):
        self.events.append('take_action')
        self.take_action_cnt[0] += 1
        # Same empty-list terminal behavior as native RewardManager.get_reward.
        self.end_flag[0] = not self.reward_manager.check_list[0]


def test_reset_registers_native_conditions_before_first_action(tmp_path, monkeypatch):
    from .robodojo_server import session as module
    env = NativeLikeEnv()
    session = RoboDojoSession(env, tmp_path, 'build_tower')
    monkeypatch.setattr(module, 'DualKinematics', lambda env: SimpleNamespace(check=lambda: {'passed': True}))
    def observe(record=False):
        session.obs = {'states': np.zeros(14, dtype=np.float32), 'instruction': 'Build a tower.'}
        return session.obs
    monkeypatch.setattr(session, '_observe', observe)
    session.reset(seed=0, source='student', policy_version='test')
    result = session.chunk_step(np.zeros((1, 14), dtype=np.float32))
    assert env.events == ['reset', 'run_reward', 'get_score', 'take_action']
    assert result['step_id'] == 1
    assert not result['steps'][0]['terminated'] and not result['steps'][0]['success']
    assert (tmp_path/'native_evaluation.json').is_file()


def test_empty_native_conditions_fail_before_any_action(tmp_path):
    env = NativeLikeEnv(register=False)
    session = RoboDojoSession(env, tmp_path, 'build_tower')
    with pytest.raises(ValueError, match='conditions are empty'):
        session.reset(seed=0, source='student', policy_version='test')
    assert env.events == ['reset', 'run_reward', 'get_score']
    assert env.take_action_cnt == [0] and session.episode_id is None


def test_support_arm_preamble_precedes_first_policy_action():
    from .robodojo_server.session import register_native_evaluation
    env = NativeLikeEnv()
    env.interact = True
    env.get_running_env_idx_list = lambda: [0]
    env.query_support_arm_traj = lambda env_idx: env.events.append(f'support_{env_idx}')
    env.reset(seed=[0])
    result = register_native_evaluation(env)
    assert env.events == ['reset', 'run_reward', 'get_score', 'support_0']
    assert result['initial_support_arm_query_envs'] == [0]


def test_different_reset_layout_rejected_before_physics(tmp_path):
    env = NativeLikeEnv()
    session = RoboDojoSession(env, tmp_path, 'build_tower', evaluation_identity={'layout_id': 3})
    with pytest.raises(ValueError, match='frozen layout_id'):
        session.reset(seed=0, source='student', policy_version='test')
    assert not env.events


def test_native_score_invalid_layout_and_budget_are_distinct(tmp_path):
    env = NativeLikeEnv()
    env.reward_manager = SimpleNamespace(get_score=lambda: [40.0])
    env.unstable_envs = set()
    session = RoboDojoSession(env, tmp_path, 'build_tower')
    session.episode_id = 'test'
    session.poisoned = False
    session.truncated = True
    session.step_id = 1050
    session._write_summary('terminal')
    read = lambda: json.loads((tmp_path/'evaluation_outcome.json').read_text())
    assert read()['native_score'] == .4 and read()['valid_for_success_rate']
    env.unstable_envs.add(0)
    session._write_summary('terminal')
    assert read()['status'] == 'invalid_native_layout' and read()['native_score'] is None
    env.unstable_envs.clear()
    session.truncated = False
    session.finish_reason = 'decision_budget'
    session._write_summary('server_close')
    assert read()['status'] == 'budget_censored' and not read()['valid_for_success_rate']
