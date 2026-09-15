"""Observation-to-action GPT baseline, without a pi05 client or proposals."""
import uuid

import numpy as np

from ..io import InputError, write_json
from ..prompt_context import CONTEXT_VERSION
from ..settings import MODEL, EFFORT, PROVIDER
from .client import RoboDojoTools, ARMS
from .decision_log import decision_event, emit
from .validation import validate_response, validate_public_language


def validate_action(response, request):
    if not isinstance(response, dict) or response.get('request_id') != request['request_id']:
        raise ValueError('Response must match the current observation request_id')
    if not isinstance(response.get('reason'), str) or not response['reason'].strip():
        raise ValueError('Brief visible evidence and action purpose are required')
    validate_public_language(response)
    mode = response.get('mode')
    if mode == 'eef':
        if response.get('actions') not in (None, []):
            raise ValueError('Direct joint actions are not available in EEF-only evaluation')
        if set(response.get('target', {})) != set(ARMS):
            raise ValueError('EEF action requires explicit left and right targets')
        for arm in ARMS:
            validate_response(dict(response, target=response['target'][arm]),
                dict(request, current_eef=request['current_eef'][arm]))
    else:
        raise ValueError('GPT-only supports bounded EEF actions only; no joint/student/edit/stop')


class GPTOnlyTools(RoboDojoTools):
    def __init__(self, output, task, **kwargs):
        super().__init__(output, task, None, **kwargs)
        self.source = 'gpt_eef'
        self.counters.update(gpt_joint_steps=0, gpt_eef_steps=0)

    def next_call(self):
        if self.phase == 'start':
            return super().next_call()
        if self.phase == 'act':
            return dict(tool='robodojo_act', observation_path=str(self.observation_path),
                output_dir=str(self.output/'observations'/f'{len(self.history)+1:03d}'),
                response='Choose bounded dual-arm EEF action from current observation')
        return None

    def infer(self, **arguments):
        raise InputError('pi05 is unavailable in the GPT-only evaluation')

    def execute(self, **arguments):
        raise InputError('Use robodojo_act; no pi05 proposal exists')

    def _observation(self, directory, *, result=None):
        packet = super()._observation(directory, result=result)
        packet['evaluation_method'] = 'gpt_only'
        if self.phase != 'done':
            self.request = dict(packet, request_id=uuid.uuid4().hex,
                decision=len(self.history), prediction_id=None, gate_policy='gpt_only')
            packet.update(request_id=self.request['request_id'],
                request_path=str(self.output/f'request_{len(self.history):03d}.json'))
            write_json(self.output/f'request_{len(self.history):03d}.json', self.request)
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check('robodojo_start', arguments)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched task differs from requested task')
        reset = self.sim.request('reset', seed=self.seed, source='gpt_eef', policy_version='gpt_only')
        if reset.get('instruction'):
            self.meta['instruction'] = reset['instruction']
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        self.require_native_termination = bool(reset.get('metadata', {}).get('evaluation_case'))
        self._rpc('begin_combination', teacher_model=MODEL, teacher_model_provider=PROVIDER,
            context_version=CONTEXT_VERSION, evaluation_method='gpt_only',
            prompt_sha256=self.prompt_sha256, student_policy_version=None, student_policy_sha256=None)
        self._rpc('switch_control_source', source='gpt_eef', reason='GPT-only EEF observation-to-action baseline')
        self.run = dict(schema='robodojo_rollout.run.v1', evaluation_method='gpt_only',
            teacher='codex_tools', context_version=CONTEXT_VERSION, teacher_model=MODEL,
            teacher_model_provider=PROVIDER, teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'), robot_profile=self.profile,
            checkpoint=None, student_backend=None, student_identity_sha256=None,
            student_server_metadata=None, pi05_enabled=False, pi05_inference_calls=0,
            action_horizon=5, action_dim=14, action_space='eef_only', gate_policy='gpt_only', no_rollback=True,
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
            require_native_termination=self.require_native_termination, training_steps=0,
            training_bundle_ready=False)
        write_json(self.output/'run.json', self.run)
        write_json(self.output/'history.json', self.history)
        self.phase = 'act'
        return self._observation(directory)

    def act(self, **arguments):
        directory = self._check('robodojo_act', arguments)
        response = arguments.get('response')
        try:
            validate_action(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        decision = len(self.history)
        response = dict(response, evaluation_method='gpt_only')
        write_json(self.output/f'response_{decision:03d}.json', response)
        receipt = self._rpc('record_codex_decision', decision=decision, prediction_id=None, response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick, decision=decision,
            prediction_id=None, response=response), simulator_receipt=receipt))
        wanted = 'gpt_eef'
        if wanted != self.source:
            self._rpc('switch_control_source', source=wanted, reason=response['reason'])
            self.source = wanted
        start = self.tick
        trace = []
        for _ in range(response['steps']):
            proposal = self.target_proposal(response['target'])
            result = self._rpc('chunk_step', actions=np.asarray([proposal['action']], np.float32),
                               teacher_request_id=self.request['request_id'])
            for row in result['steps']:
                row['edited_target'], row['ik_diagnostics'] = response['target'], proposal['diagnostics']
            trace.extend(result['steps'])
            self.tick = result['step_id']
            if any(r['terminated'] or r['truncated'] for r in result['steps']):
                break
        write_json(self.output/f'edit_{decision:03d}.json', [dict(target=r['edited_target'],
            diagnostics=r['ik_diagnostics'], valid=r['valid'],
            executed_action=np.asarray(r['executed_action']).tolist()) for r in trace])
        valid = [r for r in trace if r['valid']]
        self.counters['gpt_eef_steps'] += len(valid)
        executed = np.asarray([r['executed_action'] for r in valid], np.float32).reshape(-1, 14)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
            states=np.asarray([r['obs']['states'] for r in valid], np.float32).reshape(-1, 14))
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        self.history.append(dict(decision=decision, start_tick=start, end_tick=self.tick,
            response=response, evaluation_method='gpt_only', simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail=response['mode'], prediction_id=None,
            executed_gripper_closed=self.gripper_history(executed), terminal=done,
            native_success=bool(valid and valid[-1]['success'])))
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick,
            evaluation_method='gpt_only', **self.counters))
        exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        self.phase = 'act'
        final = self.finish('terminal' if done else 'decision_budget') if done or exhausted else None
        return self._observation(directory, result=final)

    def finish(self, reason):
        result = super().finish(reason)
        result.update(evaluation_method='gpt_only', pi05_enabled=False, pi05_inference_calls=0)
        write_json(self.output/'result.json', result)
        return result
