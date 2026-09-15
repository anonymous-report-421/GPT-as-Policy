"""File-backed blocking rollout API, using the copied native recorder/EEF contract.

One owner holds the simulator connection. Codex, not this class, schedules every
start -> infer -> execute cycle. No inference, retry or correction runs implicitly.
"""
import hashlib
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from ..io import InputError, write_json
from ..settings import MODEL, EFFORT, PROVIDER
from .protocol import RPCClient
from .gate_assessment import GATE_INSTRUCTION, previous_observation, validate_assessment
from .validation import angles, validate_response
from .action_edit_kinematics import edited_targets
from .debug_recorder import DebugVideoRecorder
from .decision_log import decision_event, emit
from .proposal_diagnostics import action_diagnostics


ARMS = ('left', 'right')


def validate_dual_response(response, request):
    if response.get('mode') in ('eef', 'edit'):
        field = 'target' if response['mode'] == 'eef' else 'edit'
        if set(response.get(field, {})) != set(ARMS):
            raise ValueError(f'{field} requires explicit left and right arms')
        for arm in ARMS:
            local = dict(response, **{field: response[field][arm]})
            local_request = dict(request, current_eef=request['current_eef'][arm],
                student_eef_trajectory=[p[arm] for p in request['student_eef_trajectory']])
            validate_response(local, local_request)
    else:
        validate_response(response, request)


class RoboDojoTools:
    profile = 'robodojo'
    camera_keys = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    observation_keys = (*camera_keys, 'states', 'eef_positions', 'eef_quaternions_wxyz')
    validate = staticmethod(validate_dual_response)

    def current_eef(self):
        return {arm: angles(dict(position=self.obs['eef_positions'][i].tolist(),
            quaternion_wxyz=self.obs['eef_quaternions_wxyz'][i].tolist(),
            frame='environment_origin', link='link6',
            gripper_opening_command=float(self.obs['states'][i*7+6])))
            for i, arm in enumerate(ARMS)}

    def trajectory(self):
        return [dict(index=i, **{arm: angles(p[arm]) for arm in ARMS})
                for i, p in enumerate(self.fk['trajectory'])]

    def correction_targets(self, response):
        if response['mode'] == 'eef':
            return [response['target']]*response['steps']
        targets = {}
        for arm in ARMS:
            trajectory = [p[arm] for p in self.fk['trajectory']]
            targets[arm] = edited_targets(trajectory, response['steps'], response['edit'][arm])
            if response['edit'][arm]['gripper'] == 'keep':
                for target, original in zip(targets[arm], trajectory):
                    target['gripper_opening'] = original['gripper_opening']
        return [{arm: targets[arm][i] for arm in ARMS} for i in range(response['steps'])]

    def target_proposal(self, target):
        return self._rpc('eef_joint_target', targets=target)

    def gripper_history(self, executed):
        return [{arm: bool(row[i*7+6] < .5) for i, arm in enumerate(ARMS)} for row in executed]

    def __init__(self, output, task, student, *, sim_port=19113, seed=0,
                 max_decisions=180, prompt_sha256='', rpc_factory=RPCClient):
        self.output = Path(output).resolve()
        self.task, self.student = task, student
        self.sim_port, self.seed = sim_port, seed
        if max_decisions < 0:
            raise ValueError('max_decisions must be non-negative; 0 uses native termination only')
        self.max_decisions, self.prompt_sha256 = max_decisions, prompt_sha256
        self.rpc_factory = rpc_factory
        self.sim = None
        self.episode = None
        self.require_native_termination = False
        self.tick = 0
        self.phase = 'start'
        self.history = []
        self.preceding = None
        self.counters = dict(student_steps=0, edited_steps=0, recovery_steps=0, predictions=0)
        self.source = 'student'
        self.started = time.monotonic()
        self.recorder = DebugVideoRecorder(self.output)

    def next_call(self):
        index = len(self.history)
        if self.phase == 'start':
            return dict(tool='robodojo_start', task=self.task,
                        output_dir=str(self.output/'observations'/'000'))
        if self.phase == 'infer':
            return dict(tool='pi05_infer', observation_path=str(self.observation_path),
                        output_dir=str(self.output/'proposals'/f'{index:03d}'))
        if self.phase == 'execute':
            return dict(tool='robodojo_execute', proposal_path=str(self.proposal_path),
                        output_dir=str(self.output/'observations'/f'{index+1:03d}'),
                        response=('Supply your gate assessment and student/edit/eef decision; '
                                  'this benchmark waits for native termination'
                                  if self.require_native_termination else
                                  'Supply your gate assessment and student/edit/eef/stop decision'))
        return None

    def _check(self, name, arguments):
        expected = self.next_call()
        if expected is None or name != expected['tool']:
            raise InputError(f'Wrong order or completed rollout; next call: {expected}')
        for key, value in expected.items():
            if key not in ('tool', 'response') and arguments.get(key) != value:
                raise InputError(f'{key} must equal {value!r}')
        destination = Path(arguments['output_dir'])
        if destination.exists() or not destination.resolve().is_relative_to(self.output):
            raise InputError('Output must be a fresh directory under this rollout root')
        return destination

    def _rpc(self, op, **kwargs):
        return self.sim.request(op, episode_id=self.episode, step_id=self.tick, **kwargs)

    def _observation(self, directory, *, result=None):
        directory.mkdir(parents=True)
        self.obs = self._rpc('teacher_observation')
        np.savez_compressed(directory/'observation.npz', **{
            k: self.obs[k] for k in self.observation_keys})
        images = []
        for camera in self.camera_keys:
            path = directory/f'{camera}.png'
            Image.fromarray(self.obs[camera]).save(path)
            images.append(dict(camera=camera, path=str(path)))
        self.observation_path = directory/'observation.json'
        packet = dict(episode_id=self.episode, step_id=self.tick, task=self.task,
            max_episode_steps=self.meta['max_episode_steps'],
            instruction=self.obs['instruction'], remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            current_state=self.obs['states'].tolist(), images=images,
            current_eef=self.current_eef(), robot_profile=self.profile,
            observation_path=str(self.observation_path), arrays_path=str(directory/'observation.npz'),
            history_path=str(self.output/'history.json'),
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), result=result, rollout_finished=self.phase == 'done',
            next_call=self.next_call())
        self.observation_packet = packet
        if self.tick == 0:
            from ..prompt_context import task_context
            packet['task_context'] = task_context(self.task)
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check('robodojo_start', arguments)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched RoboDojo task differs from requested task')
        identity = self.student.metadata['checkpoint_sha256']
        config = self.student.metadata['config']
        version = f'OpenPI-JAX/{config}/{identity[:16]}'
        reset = self.sim.request('reset', seed=self.seed, source='student', policy_version=version)
        if reset.get('instruction'):
            self.meta['instruction'] = reset['instruction']
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        # A frozen benchmark case must use its native horizon, not a model's
        # judgement that further attempts are unlikely to help. Operator and
        # infrastructure aborts still use finish() and remain incomplete.
        self.require_native_termination = bool(reset.get('metadata', {}).get('evaluation_case'))
        from ..prompt_context import CONTEXT_VERSION
        self._rpc('begin_combination', teacher_model=MODEL, teacher_model_provider=PROVIDER,
                  context_version=CONTEXT_VERSION,
                  prompt_sha256=self.prompt_sha256,
                  student_policy_version=version, student_policy_sha256=identity)
        self.run = dict(schema='robodojo_rollout.run.v1', teacher='codex_tools',
            context_version=CONTEXT_VERSION,
            teacher_model=MODEL, teacher_model_provider=PROVIDER,
            teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            evaluation_case=reset.get('metadata', {}).get('evaluation_case'),
            config=config, robot_profile=self.profile, checkpoint=self.student.metadata['checkpoint'],
            action_horizon=self.student.metadata.get('action_horizon', 15),
            action_dim=self.student.metadata.get('action_dim', 8),
            student_backend='OpenPI/JAX', student_identity_sha256=identity,
            student_server_metadata=self.student.metadata, teacher_off=False, split=None,
            gate_policy='failure-or-intent', no_rollback=True,
            gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
            require_native_termination=self.require_native_termination,
            training_steps=0, training_bundle_ready=False)
        write_json(self.output/'run.json', self.run)
        write_json(self.output/'history.json', self.history)
        self.phase = 'infer'
        return self._observation(directory)

    def infer(self, **arguments):
        directory = self._check('pi05_infer', arguments)
        directory.mkdir(parents=True)
        decision = len(self.history)
        self.proposal_path = directory/'actions.npz'
        self.actions, self.prediction = self.student.infer(self.obs, self.proposal_path)
        # Preserve the colleague's downstream audit file layout without copying arrays.
        (self.output/f'proposal_{decision:03d}.npz').hardlink_to(self.proposal_path)
        self.counters['predictions'] += 1
        self.fk = self._rpc('fk_preview', actions=self.actions)
        self.request = dict(request_id=uuid.uuid4().hex, episode_id=self.episode,
            step_id=self.tick, decision=decision, gate_policy='failure-or-intent',
            gate_instruction=GATE_INSTRUCTION, task=self.task, instruction=self.obs['instruction'],
            sim_time_s=self.tick*self.meta['control_dt'],
            max_episode_steps=self.meta['max_episode_steps'],
            remaining_steps=self.obs['remaining_steps'],
            require_native_termination=self.require_native_termination,
            images=self.observation_packet['images'],
            current_state=self.observation_packet['current_state'],
            current_eef=self.observation_packet['current_eef'],
            student_eef_trajectory=self.trajectory(), robot_profile=self.profile,
            action_diagnostics=action_diagnostics(self.actions),
            fk_check=self.fk['measured_fk_check'],
            previous_observation=previous_observation(self.preceding, self.history[-1]) if self.history else None,
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), **self.prediction)
        write_json(self.output/f'request_{decision:03d}.json', self.request)
        self.phase = 'execute'
        # The unchanged gate prompt is already in the persistent Codex workspace.
        packet = {k: v for k, v in self.request.items() if k != 'gate_instruction'}
        packet.update(proposal_path=str(self.proposal_path), rollout_finished=False,
                      request_path=str(self.output/f'request_{decision:03d}.json'), next_call=self.next_call())
        write_json(directory/'proposal.json', packet)
        return packet

    def execute(self, **arguments):
        directory = self._check('robodojo_execute', arguments)
        response = arguments.get('response')
        try:
            self.validate(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        if response['mode'] == 'stop' and self.require_native_termination:
            raise InputError(
                'This benchmark requires native termination; model stop is not allowed. '
                f"There are {self.obs['remaining_steps']} native control steps remaining. "
                'No action or stop was executed. Reassess the current observation and '
                'fresh proposal, then submit a student/edit/eef decision under the unchanged '
                'gate. Failed grasp attempts alone are not an episode termination. '
                'Do not reset or fabricate progress; operator/transport/physics aborts '
                'are handled separately by the runner.')
        decision = len(self.history)
        write_json(self.output/f'response_{decision:03d}.json', response)
        # RoboDojo persists every decision (including student/stop), not just
        # source transitions. A failed receipt aborts before physical execution.
        receipt = self._rpc('record_codex_decision', decision=decision,
                            prediction_id=self.prediction['prediction_id'], response=response)
        emit(dict(decision_event(episode_id=self.episode, step_id=self.tick,
            decision=decision, prediction_id=self.prediction['prediction_id'], response=response),
            simulator_receipt=receipt))
        if response['mode'] == 'stop':
            return self._observation(directory, result=self.finish('model_stop'))
        trigger = validate_assessment(response, self.request)
        wanted = 'student' if response['mode'] == 'student' else 'gpt_eef'
        if wanted != self.source:
            self._rpc('switch_control_source', source=wanted, reason=response['reason'])
            self.source = wanted
        trace = []
        start_tick = self.tick
        if response['mode'] == 'student':
            result = self._rpc('chunk_step', actions=self.actions[:response['steps']],
                               student_prediction_id=self.prediction['prediction_id'])
            trace = result['steps']
            self.tick = result['step_id']
        else:
            targets = self.correction_targets(response)
            for target in targets:
                proposal = self.target_proposal(target)
                result = self._rpc('chunk_step', actions=np.asarray([proposal['action']], np.float32),
                                   teacher_request_id=self.request['request_id'])
                for row in result['steps']:
                    row['edited_target'], row['ik_diagnostics'] = target, proposal['diagnostics']
                trace.extend(result['steps'])
                self.tick = result['step_id']
                if any(r['terminated'] or r['truncated'] for r in result['steps']):
                    break
        valid = [r for r in trace if r['valid']]
        if response['mode'] != 'student':
            write_json(self.output/f'edit_{decision:03d}.json', [dict(
                target=r['edited_target'], diagnostics=r['ik_diagnostics'], valid=r['valid'],
                executed_action=np.asarray(r['executed_action']).tolist()) for r in trace])
        countkey = dict(student='student_steps', edit='edited_steps', eef='recovery_steps')[response['mode']]
        self.counters[countkey] += len(valid)
        executed = np.asarray([r['executed_action'] for r in valid], np.float32)
        np.savez_compressed(self.output/f'execution_{decision:03d}.npz', actions=executed,
                            states=np.asarray([r['obs']['states'] for r in valid], np.float32))
        done = bool(valid and (valid[-1]['terminated'] or valid[-1]['truncated']))
        record = dict(decision=decision, start_tick=start_tick, end_tick=self.tick, response=response,
            simulator_decision_receipt=receipt,
            executed_steps=len(valid), source=self.source, source_detail=response['mode'],
            prediction_id=self.prediction['prediction_id'],
            selected_student_steps=response['steps'] if response['mode'] == 'student' else 0,
            discarded_student_steps=len(self.actions)-len(valid) if response['mode'] == 'student' else len(self.actions),
            executed_gripper_closed=self.gripper_history(executed),
            assessment=response['assessment'], takeover_trigger=trigger, terminal=done,
            native_success=bool(valid and valid[-1]['success']))
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick, **self.counters))
        self.phase = 'infer'
        budget_exhausted = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        final = self.finish('terminal' if done else 'decision_budget') if done or budget_exhausted else None
        return self._observation(directory, result=final)

    def finish(self, reason):
        final = self._rpc('finish_pilot', reason=reason)
        self.phase = 'done'
        from ..prompt_context import CONTEXT_VERSION
        final.update(self.counters, decisions=len(self.history), context_version=CONTEXT_VERSION,
                     complete=bool(final['terminated'] or final['truncated']),
                     wall_seconds=time.monotonic()-self.started, training_bundle_ready=False,
                     trajectory_path=str(self.output.parent/'sim'),
                     history_path=str(self.output/'history.json'),
                     debug_video_path=str(self.recorder.output/'debug_rollout.mp4'),
                     debug_video_manifest=str(self.recorder.output/'manifest.json'))
        write_json(self.output/'result.json', final)
        return final

    def close(self):
        if self.sim is not None:
            self.sim.close()
        # finish_pilot has finalized sensors.mp4; this runs after the Codex turn
        # so its public final message is available too. No extra physics steps.
        if self.episode is not None:
            self.recorder.finalize()
