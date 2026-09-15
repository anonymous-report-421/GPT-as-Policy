"""Isolated action-edit pilot server, using the existing DROID RPC/recorder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np

from .session import EnvironmentSession, serve
from .action_edit_kinematics import RobotFK, pose, transform
from .decision_log import record_decision


class EditEnvironmentSession(EnvironmentSession):
    student_execution_lengths = tuple(range(1, 16))

    def __init__(self, *args, **kwargs):
        self.video_writer = None
        self.video_frames = 0
        self.fk = None
        self.fk_checks = []
        super().__init__(*args, **kwargs)
        self.metadata['policy_identity_source'] = 'external_openpi_pilot_recorded_separately_not_native_RLinf_audit'
        self.metadata['supports_fk_preview'] = True
        self.metadata['supports_codex_decision_recording'] = True
        self.metadata['student_execution_lengths'] = list(self.student_execution_lengths)

    def extract_obs(self, raw):
        result = super().extract_obs(raw)
        from PIL import Image
        images = []
        for name in ('over_shoulder_left_camera', 'wrist_cam'):
            rgb = raw['image_obs'][name][0].detach().cpu().numpy()[..., :3]
            images.append(np.asarray(Image.fromarray(rgb).resize((640, 360))))
        if self.video_writer is None:
            import imageio.v2 as imageio
            self.video_writer = imageio.get_writer(str(self.output/'sensors.mp4'), fps=1/self.env.step_dt,
                codec='libx264', pixelformat='yuv420p', macro_block_size=2,
                output_params=['-movflags', '+faststart'])
        self.video_writer.append_data(np.concatenate(images, axis=1))
        self.video_frames += 1
        return result

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        import omni.usd
        self.fk = RobotFK.from_stage(omni.usd.get_context().get_stage())
        (self.output/'robot_fk_chain.json').write_text(json.dumps(self.fk.chain, indent=2))
        self._check_fk()
        return result

    def _check_fk(self):
        from scipy.spatial.transform import Rotation
        p = self.raw_obs['proprio_obs']
        observed = transform(p['ee_pos'][0].detach().cpu().numpy(), p['ee_quat'][0].detach().cpu().numpy())
        predicted = self.fk.matrix(self.obs['states'][:7])
        distance = float(np.linalg.norm(predicted[:3, 3]-observed[:3, 3]))
        angle = float(Rotation.from_matrix(predicted[:3, :3] @ observed[:3, :3].T).magnitude())
        check = dict(step_id=self.step_id, position_error_m=distance, rotation_error_rad=angle,
                     physical_steps=0, passed=distance < .002 and angle < .01)
        self.fk_checks.append(check)
        (self.output/'fk_validation.json').write_text(json.dumps(self.fk_checks, indent=2))
        if not check['passed']:
            self.poisoned = True
            raise ValueError(f'FK does not match native measured flange: {check}')
        return check

    def teacher_observation(self, *args, **kwargs):
        result = super().teacher_observation(*args, **kwargs)
        from openpi_client.image_tools import resize_with_pad
        for key, name in [('main_rgb', 'over_shoulder_left_camera'), ('wrist_rgb', 'wrist_cam')]:
            result[key] = resize_with_pad(self.raw_obs['image_obs'][name][0].detach().cpu().numpy()[..., :3], 504, 896)
        return result

    def dispatch(self, op, args):
        if op == 'reset':
            if self.episode_id is not None or args.get('initial_state_id') is not None:
                raise ValueError('New version allows one fresh episode; no restore or rollback')
        if op not in ('metadata', 'reset', 'chunk_step', 'teacher_observation',
                      'eef_joint_target', 'begin_combination', 'switch_control_source',
                      'fk_preview', 'finish_pilot', 'record_codex_decision'):
            raise ValueError('Operation is not part of the no-rollback rollout service')
        if op == 'record_codex_decision':
            return record_decision(self, args)
        if op == 'fk_preview':
            self._check_identity(args['episode_id'], args['step_id'])
            check = self._check_fk()
            return dict(episode_id=self.episode_id, step_id=self.step_id, physical_steps=0,
                frame='base_link_in_robot_root', units='meters_and_wxyz', measured_fk_check=check,
                trajectory=self.fk.trajectory(args['actions']),
                interpretation='kinematic_target_preview_not_physics_or_object_future')
        if op == 'finish_pilot':
            self._check_identity(args['episode_id'], args['step_id'])
            self._write_summary('terminal' if self.terminated or self.truncated else 'operator_finished')
            if self.video_writer:
                self.video_writer.close(); self.video_writer = None
            report = dict(episode_id=self.episode_id, step_id=self.step_id, success=self.success,
                terminated=self.terminated, truncated=self.truncated, video_frames=self.video_frames,
                control_dt=self.env.step_dt, reason=args['reason'])
            (self.output/'pilot_finished.json').write_text(json.dumps(report, indent=2))
            return report
        return super().dispatch(op, args)

    def chunk_step(self, *args, **kwargs):
        result = super().chunk_step(*args, **kwargs)
        if (self.terminated or self.truncated) and self.video_writer is not None:
            # Native chunks/summary are already persisted by the copied recorder.
            # Finalize video even if the controller disconnects before finish_pilot.
            self.video_writer.close()
            self.video_writer = None
        return result

    def _write_summary(self, reason):
        super()._write_summary(reason)
        if reason in ('controller_disconnect', 'simulator_error', 'recorder_error') and self.video_writer is not None:
            # Preserve a decodable partial video on an interrupted episode too.
            self.video_writer.close()
            self.video_writer = None


def main():
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='Registered RoboLab task class name')
    parser.add_argument('--port', type=int, default=19093)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, help='Optional cap; otherwise native task horizon')
    parser.add_argument('--randomize-background', action='store_true')
    parser.add_argument('--background-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instruction-type', choices=('default', 'specific'), default='default')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(); args.headless = True; args.enable_cameras = True
    args.output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = session = None
    try:
        import robolab.constants as constants
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        constants.RECORD_IMAGE_DATA = False
        constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        constants.set_output_dir(str(args.output/'native'))
        auto_register_droid_envs(task=args.task, randomize_background=args.randomize_background,
                                 background_seed=args.background_seed)
        names = get_envs(task=args.task)
        if len(names) != 1:
            raise ValueError(f'Expected exactly one registration: {names}')
        env, cfg = create_env(names[0], num_envs=1, policy='pi05', seed=args.seed,
                              instruction_type=args.instruction_type)
        session = EditEnvironmentSession(env, cfg, args.output, args.max_steps,
            'OpenPI-JAX/pi05_droid_jointpos_polaris/local', True, 'rgb_proprio')
        session.metadata.update(task=args.task, robot_adapter='droid_panda_robotiq',
            randomize_background=args.randomize_background,
            background_seed=args.background_seed,
            seed=args.seed, instruction_type=args.instruction_type,
            implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')})
        serve(session, args.port)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if session and session.video_writer:
            session.video_writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
