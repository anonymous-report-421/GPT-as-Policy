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
from ..settings import MODEL, EFFORT
from .protocol import RPCClient
from .gate_assessment import GATE_INSTRUCTION, previous_observation, validate_assessment
from .validation import angles, validate_response
from .action_edit_kinematics import edited_targets
from .debug_recorder import DebugVideoRecorder
from .decision_log import decision_event, emit


class RolloutTools:
    def __init__(self, output, task, student, *, sim_port=19093, seed=0,
                 max_decisions=180, prompt_sha256='', rpc_factory=RPCClient):
        self.output = Path(output).resolve()
        self.task, self.student = task, student
        self.sim_port, self.seed = sim_port, seed
        self.max_decisions, self.prompt_sha256 = max_decisions, prompt_sha256
        self.rpc_factory = rpc_factory
        self.sim = None
        self.episode = None
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
            return dict(tool='robolab_start', task=self.task,
                        output_dir=str(self.output/'observations'/'000'))
        if self.phase == 'infer':
            return dict(tool='pi05_infer', observation_path=str(self.observation_path),
                        output_dir=str(self.output/'proposals'/f'{index:03d}'))
        if self.phase == 'execute':
            return dict(tool='robolab_execute', proposal_path=str(self.proposal_path),
                        output_dir=str(self.output/'observations'/f'{index+1:03d}'),
                        response='Supply your gate assessment and student/edit/eef/stop decision')
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
            k: self.obs[k] for k in ('main_images', 'wrist_images', 'states', 'ee_pos', 'ee_quat_wxyz')})
        images = []
        for camera in ('main_rgb', 'wrist_rgb'):
            path = directory/f'{camera}.png'
            Image.fromarray(self.obs[camera]).save(path)
            images.append(dict(camera=camera, path=str(path)))
        self.observation_path = directory/'observation.json'
        packet = dict(episode_id=self.episode, step_id=self.tick, task=self.task,
            instruction=self.obs['instruction'], remaining_steps=self.obs['remaining_steps'],
            current_state=self.obs['states'].tolist(), images=images,
            current_eef=angles(dict(position=self.obs['ee_pos'].tolist(),
                                   quaternion_wxyz=self.obs['ee_quat_wxyz'].tolist())),
            observation_path=str(self.observation_path), arrays_path=str(directory/'observation.npz'),
            history_path=str(self.output/'history.json'),
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), result=result, rollout_finished=self.phase == 'done',
            next_call=self.next_call())
        self.observation_packet = packet
        write_json(self.observation_path, packet)
        return packet

    def start(self, **arguments):
        directory = self._check('robolab_start', arguments)
        self.sim = self.rpc_factory('127.0.0.1', self.sim_port, timeout=600)
        self.meta = self.sim.request('metadata')
        if self.meta['task'] != self.task:
            raise RuntimeError('Launched RoboLab task differs from requested task')
        identity = self.student.metadata['checkpoint_sha256']
        version = f'OpenPI-JAX/pi05_droid_jointpos_polaris/{identity[:16]}'
        reset = self.sim.request('reset', seed=self.seed, source='student', policy_version=version)
        self.episode, self.tick = reset['episode_id'], reset['step_id']
        self._rpc('begin_combination', teacher_model=MODEL,
                  prompt_sha256=self.prompt_sha256,
                  student_policy_version=version, student_policy_sha256=identity)
        self.run = dict(schema='hybrid_rollout.run.v1', teacher='codex_tools',
            teacher_model=MODEL, teacher_reasoning_effort=EFFORT,
            task=self.task, instruction=self.meta['instruction'], seed=self.seed,
            config='pi05_droid_jointpos_polaris', checkpoint=self.student.metadata['checkpoint'],
            student_backend='OpenPI/JAX', student_identity_sha256=identity,
            student_server_metadata=self.student.metadata, teacher_off=False, split=None,
            gate_policy='failure-or-intent', no_rollback=True,
            gate_instruction_sha256=hashlib.sha256(GATE_INSTRUCTION.encode()).hexdigest(),
            teacher_prompt_sha256=self.prompt_sha256, initial_state_hash=reset.get('initial_state_hash'),
            reset_metadata=reset.get('metadata'), max_episode_steps=self.meta['max_episode_steps'],
            control_dt=self.meta['control_dt'], max_decisions=self.max_decisions,
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
            images=self.observation_packet['images'],
            current_state=self.observation_packet['current_state'],
            current_eef=self.observation_packet['current_eef'],
            student_eef_trajectory=[angles(p) for p in self.fk['trajectory']],
            fk_check=self.fk['measured_fk_check'],
            previous_observation=previous_observation(self.preceding, self.history[-1]) if self.history else None,
            previous_result=self.history[-1] if self.history else None,
            counters=dict(self.counters), **self.prediction)
        write_json(self.output/f'request_{decision:03d}.json', self.request)
        self.phase = 'execute'
        # The unchanged gate prompt is already in the persistent Codex workspace.
        packet = {k: v for k, v in self.request.items() if k != 'gate_instruction'}
        packet.update(proposal_path=str(self.proposal_path),
                      request_path=str(self.output/f'request_{decision:03d}.json'), next_call=self.next_call())
        write_json(directory/'proposal.json', packet)
        return packet

    def execute(self, **arguments):
        directory = self._check('robolab_execute', arguments)
        response = arguments.get('response')
        try:
            validate_response(response, self.request)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise InputError(str(error)) from error
        decision = len(self.history)
        write_json(self.output/f'response_{decision:03d}.json', response)
        # RoboLab persists every decision (including student/stop), not just
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
            targets = (edited_targets(self.fk['trajectory'], response['steps'], response['edit'])
                       if response['mode'] == 'edit' else [response['target']]*response['steps'])
            for target in targets:
                proposal = self._rpc('eef_joint_target', position=target['position'],
                    quaternion_wxyz=target['quaternion_wxyz'], gripper_closed=target['gripper_closed'])
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
            discarded_student_steps=15-len(valid) if response['mode'] == 'student' else 15,
            executed_gripper_closed=[bool(row[7] > .5) for row in executed],
            assessment=response['assessment'], takeover_trigger=trigger, terminal=done,
            native_success=bool(valid and valid[-1]['success']))
        self.history.append(record)
        self.preceding = self.request
        write_json(self.output/'history.json', self.history)
        write_json(self.output/'progress.json', dict(episode_id=self.episode, step_id=self.tick, **self.counters))
        self.phase = 'infer'
        final = self.finish('terminal' if done else 'decision_budget') if done or len(self.history) >= self.max_decisions else None
        return self._observation(directory, result=final)

    def finish(self, reason):
        final = self._rpc('finish_pilot', reason=reason)
        self.phase = 'done'
        final.update(self.counters, decisions=len(self.history),
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
