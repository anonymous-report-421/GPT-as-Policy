"""One RoboLab environment per endpoint, stepping only on explicit requests."""
import argparse
import hashlib
import json
import re
import socket
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

from .protocol import VERSION, receive_packet, send_packet


class EnvironmentSession:
    student_execution_lengths = (15,)

    def __init__(self, env, cfg, output, max_steps=None, policy_version="unassigned", record_scene_state=False,
                 teacher_observation_mode="rgb_proprio"):
        if teacher_observation_mode != "rgb_proprio":
            raise ValueError("Unknown teacher observation mode")
        self.teacher_observation_mode = teacher_observation_mode
        self.env, self.cfg = env, cfg
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_steps = min(int(env.max_episode_length), max_steps or int(env.max_episode_length))
        self.max_steps_cap = self.max_steps
        self.policy_version = policy_version
        self.server_default_policy_version = policy_version
        self.episode_id = None
        self.step_id = 0
        self.poisoned = False
        self.obs = None
        self.terminated = self.truncated = self.success = False
        self.episode_return = 0.0
        self.record_scene_state = bool(record_scene_state)
        self.branch = None
        self.combination = None
        self.teacher_request_id = ""
        self.raw_obs = None
        self.metadata = dict(protocol_version=VERSION, task=cfg.task_name if hasattr(cfg, "task_name") else "",
            instruction=cfg.instruction, native_max_episode_steps=int(env.max_episode_length),
            max_episode_steps=self.max_steps, control_dt=float(env.step_dt), action_dim=8,
            action_semantics="absolute_arm_joint_positions_7_and_binary_gripper_gt_0.5",
            observation_semantics="RGB_uint8_224x224_pad; joint_position_7_and_gripper_position",
            success_criterion="termination_manager.get_term('success')",
            reset_protocol="reset_eval_state; reset(seed); reset()",
            policy_version=policy_version, server_default_policy_version=policy_version,
            policy_identity_source="caller_declared_validated_by_native_model_audit",
            scene_recording=self.record_scene_state,
            supports_live_takeover=True, supports_replay=False,
            supports_online_combination=True,
            supports_eef_joint_target=True, teacher_observation_mode=teacher_observation_mode)

    def _check_identity(self, episode_id, step_id):
        if self.poisoned or self.episode_id is None:
            raise RuntimeError("Reset required after simulator error or new connection")
        if episode_id != self.episode_id or type(step_id) is not int or step_id != self.step_id:
            raise ValueError("Stale episode_id or step_id")

    def _capture_state(self, kind="post_step"):
        if not self.record_scene_state:
            return None
        from .scene_recording import capture_scene_state, save_snapshot
        snapshot = capture_scene_state(self.env, episode_id=self.episode_id,
            step_id=self.step_id, kind=kind, metadata=dict(task=self.metadata["task"],
            control_dt=self.metadata["control_dt"], max_episode_steps=self.max_steps,
            success=self.success, terminated=self.terminated, truncated=self.truncated,
            **{key: self.metadata.get(key) for key in (
                "reset_contract", "fixture_id", "split", "initial_layout", "initial_state_id")}))
        relative = "initial_scene.json" if kind == "initial" else f"scene_{self.step_id:06d}.json"
        record = save_snapshot(self.episode_dir / relative, snapshot)
        if kind == "initial":
            self.initial_state_hash = record["state_sha256"]
        return snapshot

    def extract_obs(self, raw):
        from openpi_client.image_tools import resize_with_pad

        def array(value):
            return value.detach().cpu().numpy().copy()
        proprio = raw["proprio_obs"]
        state = np.concatenate((array(proprio["arm_joint_pos"][0]),
                                array(proprio["gripper_pos"][0]).reshape(-1))).astype(np.float32)
        main = resize_with_pad(array(raw["image_obs"]["over_shoulder_left_camera"][0]), 224, 224)
        wrist = resize_with_pad(array(raw["image_obs"]["wrist_cam"][0]), 224, 224)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Invalid proprioception")
        for value in (main, wrist):
            if value.dtype != np.uint8 or value.shape != (224, 224, 3):
                raise ValueError("Expected RGB uint8 image from native camera")
        return dict(main_images=main, wrist_images=wrist, states=state,
                    task_description=self.cfg.instruction)

    def _write_summary(self, reason):
        if self.episode_id is None:
            return
        summary = dict(self.metadata, episode_id=self.episode_id, seed=self.seed,
            source=self.source, control_steps=self.step_id, success=self.success,
            terminated=self.terminated, truncated=self.truncated,
            episode_return=self.episode_return, end_reason=reason,
            wall_seconds=time.monotonic() - self.started,
            initial_state_hash=getattr(self, "initial_state_hash", None),
            branch_id=self.branch["branch_id"] if self.branch else "",
            combination_id=self.combination["combination_id"] if self.combination else "",
            control_epoch=self.combination["control_epoch"] if self.combination else None,
            complete=bool(self.terminated or self.truncated))
        (self.episode_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if self.branch:
            manifest = dict(self.branch, success=self.success,
                complete=bool(self.terminated or self.truncated),
                summary={"path": "summary.json", "sha256": self._sha(self.episode_dir / "summary.json")})
            (self.episode_dir / "correction_manifest.json").write_text(json.dumps(manifest, indent=2))
        if self.combination:
            manifest = dict(self.combination, source=self.source, step_id=self.step_id,
                success=self.success, terminated=self.terminated, truncated=self.truncated,
                complete=bool(self.terminated or self.truncated), end_reason=reason,
                summary={"path": "summary.json", "sha256": self._sha(self.episode_dir / "summary.json")})
            (self.episode_dir / "combination_manifest.json").write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def reset(self, seed, source="student", max_steps=None, initial_state_id=None,
              fixture_id=None, split=None, initial_layout=None, policy_version=None):
        from robolab.core.environments.runtime import end_episode
        from .scene_recording import (RESET_CONTRACT, canonicalize_droid_targets,
            randomize_banana_initial_layout, refresh_initial_observation)
        if source not in ("student", "diagnostic_hold", "replay_prefix"):
            raise ValueError("Use begin_correction to enter a teacher source")
        requested_policy = self.server_default_policy_version if policy_version is None else policy_version
        if (not isinstance(requested_policy, str) or not requested_policy.strip()
                or len(requested_policy) > 256 or any(ord(char) < 32 for char in requested_policy)):
            raise ValueError("policy_version must be a nonempty, bounded single-line string")
        if fixture_id is not None and (not isinstance(fixture_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", fixture_id)):
            raise ValueError("fixture_id must be a short, path-free identifier")
        if split not in (None, "debug", "train", "dev", "test") or ((fixture_id is None) != (split is None)):
            raise ValueError("A fixture requires both fixture_id and a debug/train/dev/test split")
        if initial_layout not in (None, "default", "banana_uniform_10cm"):
            raise ValueError("Unknown initial layout")
        initial = None
        if initial_state_id is not None:
            if not self.record_scene_state or not re.fullmatch(r"[a-f0-9]{32}", initial_state_id):
                raise ValueError("A recorded initial episode ID is required")
            from .scene_recording import load_snapshot
            initial = load_snapshot(self.output / initial_state_id / "initial_scene.json")
            if (initial["episode_id"] != initial_state_id or initial["step_id"] != 0
                    or initial["kind"] != "initial"):
                raise ValueError("Initial scene snapshot identity does not match its episode/path")
            metadata = initial["metadata"]
            if metadata["task"] != self.metadata["task"] or metadata["control_dt"] != self.metadata["control_dt"]:
                raise ValueError("Recorded task or control frequency differs")
            for key, requested in (("fixture_id", fixture_id), ("split", split), ("initial_layout", initial_layout)):
                if requested is not None and metadata.get(key) != requested:
                    raise ValueError(f"Recorded fixture {key} does not match request")
            fixture_id, split = metadata.get("fixture_id"), metadata.get("split")
            initial_layout = metadata.get("initial_layout", "legacy_recorded")
        else:
            initial_layout = initial_layout or "default"
            if initial_layout == "banana_uniform_10cm" and self.metadata["task"] != "BananaInBowlTask":
                raise ValueError("Banana randomization is restricted to BananaInBowlTask")
            if fixture_id is not None and not self.record_scene_state:
                raise ValueError("Fixtures require scene-state recording")
        limit = self.max_steps_cap if max_steps is None else int(max_steps)
        if not 1 <= limit <= self.max_steps_cap:
            raise ValueError("Episode limit must be positive and no greater than the configured native cap")
        if self.episode_id is not None:
            self._write_summary("terminal" if self.terminated or self.truncated else "controller_reset")
            end_episode(self.env)
        self.poisoned = True
        self.env.reset_eval_state()
        self.env.reset(seed=int(seed))
        raw, _ = self.env.reset()
        if initial is not None:
            from .scene_recording import restore_initial_state
            raw, _ = restore_initial_state(self.env, initial, seed=int(seed))
            target_reset = {"position_target": "recorded_initial_targets", "physical_steps": 0}
            layout_reset = None
        else:
            layout_reset = randomize_banana_initial_layout(self.env) if initial_layout == "banana_uniform_10cm" else None
            target_reset = canonicalize_droid_targets(self.env)
        raw, camera_refresh = refresh_initial_observation(self.env)
        self.raw_obs = raw
        self.obs = self.extract_obs(raw)
        self.episode_id = uuid.uuid4().hex
        self.step_id, self.chunk_id = 0, 0
        self.seed, self.source = int(seed), source
        self.policy_version = requested_policy
        self.max_steps = limit
        self.metadata = dict(self.metadata)
        self.metadata["policy_version"] = requested_policy
        self.metadata["max_episode_steps"] = limit
        self.metadata["reset_protocol"] = (
            "reset_eval_state; reset(seed); reset(); reset_eval_state; reset(seed); "
            "reset_to(recorded_initial,seed=None); restore_controller_targets; refresh_cameras_no_physics"
            if initial is not None else "reset_eval_state; reset(seed); reset(); "
            "optional_layout_event; canonicalize_droid_targets; refresh_cameras_no_physics")
        self.metadata["reset_mode"] = "recorded_initial" if initial is not None else "fresh"
        self.metadata.update(reset_contract=RESET_CONTRACT, fixture_id=fixture_id, split=split,
            initial_layout=initial_layout, initial_state_id=initial_state_id or self.episode_id,
            target_reset=target_reset, camera_refresh=camera_refresh, layout_reset=layout_reset)
        self.terminated = self.truncated = self.success = False
        self.episode_return = 0.0
        self.branch = None
        self.combination = None
        self.teacher_request_id = ""
        self.initial_state_hash = None
        self.replay_provenance = None
        self.started = time.monotonic()
        self.episode_dir = self.output / self.episode_id
        self.episode_dir.mkdir()
        np.savez_compressed(self.episode_dir / "initial_observation.npz", **self.obs)
        self._capture_state("initial")
        self._write_summary("active")
        self.poisoned = False
        print(json.dumps(dict(event="reset", episode_id=self.episode_id, seed=self.seed)), flush=True)
        return dict(obs=self.obs, episode_id=self.episode_id, step_id=0,
            initial_state_hash=self.initial_state_hash, metadata=self.metadata)

    def begin_combination(self, episode_id, step_id, teacher_model, prompt_sha256,
                          student_policy_version, student_policy_sha256,
                          teacher_variant="gpt_eef"):
        """Declare a sequential student/EEF episode without a legacy correction branch.

        Policy identifiers are caller declarations: the controller must separately
        archive the native prediction service's model audit and request receipts.
        This operation performs no physics and initially leaves student control.
        """
        self._check_identity(episode_id, step_id)
        if (self.step_id != 0 or self.source != "student" or self.branch or self.combination
                or self.terminated or self.truncated):
            raise ValueError("Combination must begin once at a live student reset at tick zero")
        if teacher_variant != "gpt_eef":
            raise ValueError("Combination currently supports only the explicit gpt_eef teacher")
        for name, value in (("teacher_model", teacher_model),
                            ("student_policy_version", student_policy_version)):
            if (not isinstance(value, str) or not value.strip() or len(value) > 256
                    or any(ord(c) < 32 for c in value)):
                raise ValueError(f"Invalid {name}")
        for name, value in (("prompt_sha256", prompt_sha256),
                            ("student_policy_sha256", student_policy_sha256)):
            if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError(f"Invalid {name}")
        if student_policy_version != self.policy_version:
            raise ValueError("Combination student policy differs from reset policy identity")
        if not self.record_scene_state or not self.initial_state_hash:
            raise ValueError("Combination requires initial and physical-state recording")
        self.combination = dict(schema="rlinf_robolab.online_combination.v1",
            combination_id=uuid.uuid4().hex, episode_id=self.episode_id,
            initial_state_hash=self.initial_state_hash, teacher_variant=teacher_variant,
            teacher_model=teacher_model, prompt_sha256=prompt_sha256,
            student_policy_version=student_policy_version, student_policy_sha256=student_policy_sha256,
            policy_identity_source="caller_declared_requires_native_service_audit",
            initial_observation={"path": "initial_observation.npz",
                "sha256": self._sha(self.episode_dir / "initial_observation.npz")},
            max_episode_steps=self.max_steps, control_epoch=0, rewind_enabled=False,
            transitions=[dict(control_epoch=0, step_id=0, previous_source=None,
                source="student", reason="begin_combination")], chunks=[], student_prediction_ids=[])
        self.teacher_request_id = ""
        self._persist_combination()
        return self._combination_ack(changed=True)

    def _combination_ack(self, changed):
        return dict(schema="rlinf_robolab.combination_control_ack.v1",
            episode_id=self.episode_id, step_id=self.step_id, source=self.source,
            combination_id=self.combination["combination_id"],
            control_epoch=self.combination["control_epoch"], physical_steps=0, changed=changed)

    def _persist_combination(self):
        try:
            self._write_summary("active")
        except Exception:
            self.poisoned = True
            raise

    def switch_control_source(self, episode_id, step_id, source, reason):
        self._check_identity(episode_id, step_id)
        if not self.combination or self.branch or self.terminated or self.truncated:
            raise ValueError("Source switching requires a live combination episode")
        if source not in ("student", "gpt_eef"):
            raise ValueError("Combination source must be student or gpt_eef")
        if (not isinstance(reason, str) or not reason.strip() or len(reason) > 1024
                or any(ord(c) < 32 for c in reason)):
            raise ValueError("Source switching requires a bounded single-line reason")
        if source == self.source:
            return self._combination_ack(changed=False)
        previous = self.source
        self.source = source
        self.teacher_request_id = ""
        self.combination["control_epoch"] += 1
        self.combination["transitions"].append(dict(control_epoch=self.combination["control_epoch"],
            step_id=self.step_id, previous_source=previous, source=source, reason=reason))
        self._persist_combination()
        return self._combination_ack(changed=True)

    def teacher_observation(self, episode_id, step_id):
        self._check_identity(episode_id, step_id)
        # This whitelist excludes object ground truth and future trajectory data.
        def array(value):
            return value.detach().cpu().numpy().copy()
        proprio = self.raw_obs["proprio_obs"]
        result = dict(episode_id=self.episode_id, step_id=self.step_id,
            remaining_steps=self.max_steps-self.step_id, instruction=self.cfg.instruction,
            states=self.obs["states"], main_images=self.obs["main_images"], wrist_images=self.obs["wrist_images"],
            ee_pos=array(proprio["ee_pos"][0]), ee_quat_wxyz=array(proprio["ee_quat"][0]),
            eef_pos=array(proprio["eef_pos"][0]), eef_quat_wxyz=array(proprio["eef_quat"][0]),
            frame="robot_root", ee_frame="base_link_in_robot_root", eef_frame="eef_frame_in_robot_root",
            position_units="meters", teacher_inputs="RGB_proprio_calibration")
        if self.teacher_observation_mode == "rgb_proprio":
            result["teacher_inputs"] = "RGB_proprio_no_object_GT"
            return result
        raise ValueError('Only RGB/proprio observations are supported')

    def eef_joint_target(self, episode_id, step_id, position=None, quaternion_wxyz=None,
                         gripper_closed=None, damping=0.05, max_joint_delta_rad=0.05,
                         max_translation_step_m=0.02, max_rotation_step_rad=0.1,
                         joint_limit_margin_rad=0.0):
        """Propose one jointpos action; no physics, camera refresh or clock change.

        Executing this proposal still requires a separate one-step chunk_step
        with the same identity. The teacher must re-observe before proposing
        the next action. This does not switch the native action controller.
        """
        self._check_identity(episode_id, step_id)
        legacy_eef = (self.branch and self.source in ("gpt_eef", "diagnostic_teacher", "diagnostic_eef")
            and self.branch.get("teacher_variant") == self.source)
        combination_eef = self.combination and self.source == "gpt_eef"
        if self.terminated or self.truncated or not (legacy_eef or combination_eef):
            raise ValueError("EEF proposal requires a live EEF or diagnostic correction branch")
        from .eef_controller import build_eef_proposal, EEFControlError
        # These are experiment limits, not model-selectable escalation knobs.
        for key, value, cap in (("max_joint_delta_rad", max_joint_delta_rad, 0.05),
                               ("max_translation_step_m", max_translation_step_m, 0.02),
                               ("max_rotation_step_rad", max_rotation_step_rad, 0.1)):
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= cap:
                raise EEFControlError(f"{key} must be positive and at most {cap}")
        if type(damping) not in (int, float) or damping != 0.05:
            raise EEFControlError("This EEF RPC uses the fixed experiment damping 0.05")
        if type(joint_limit_margin_rad) not in (int, float) or joint_limit_margin_rad != 0.0:
            raise EEFControlError("This EEF RPC uses native soft limits with margin 0.0")
        result = build_eef_proposal(self.env, position, quaternion_wxyz, gripper_closed,
            damping=damping, max_joint_delta_rad=max_joint_delta_rad,
            max_translation_step_m=max_translation_step_m, max_rotation_step_rad=max_rotation_step_rad,
            joint_limit_margin_rad=joint_limit_margin_rad)
        return dict(result, episode_id=self.episode_id, step_id=self.step_id)

    def chunk_step(self, actions, episode_id, step_id, teacher_request_id=None,
                   student_prediction_id=None):
        import torch
        actions = np.asarray(actions, dtype=np.float32)
        self._check_identity(episode_id, step_id)
        if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= len(actions) <= 64 or not np.isfinite(actions).all():
            raise ValueError("Actions must be finite [H,8], 1 <= H <= 64")
        if self.combination and not (self.terminated or self.truncated):
            if self.source == "student":
                if (not isinstance(student_prediction_id, str) or not student_prediction_id.strip()
                        or len(student_prediction_id) > 256
                        or any(ord(c) < 32 for c in student_prediction_id)
                        or teacher_request_id is not None):
                    raise ValueError("Combination student actions require only their native prediction ID")
                if student_prediction_id in self.combination["student_prediction_ids"]:
                    raise ValueError("A student prediction cannot be physically executed twice")
                self.teacher_request_id = ""
            elif student_prediction_id is not None:
                raise ValueError("Teacher actions cannot carry a student prediction ID")
            if self.source == "gpt_eef" and len(actions) != 1:
                raise ValueError("Combination EEF execution requires one action per physical ACK")
            if self.source == "student" and len(actions) not in self.student_execution_lengths:
                raise ValueError("Combination student execution requires one fresh H15 prediction")
        if (self.branch or (self.combination and self.source == "gpt_eef")) and not (self.terminated or self.truncated):
            if not isinstance(teacher_request_id, str) or not teacher_request_id:
                raise ValueError("Teacher execution must identify its model request")
            self.teacher_request_id = teacher_request_id
        executed = actions.copy()
        executed[:, -1] = (executed[:, -1] > 0.5).astype(np.float32)
        steps, records = [], []
        for action, proposed in zip(executed, actions):
            valid = not (self.terminated or self.truncated)
            reward = 0.0
            if valid:
                pre_obs = self.obs
                try:
                    raw, rew, term, trunc, _ = self.env.step(torch.from_numpy(action[None]).to(self.env.device))
                    self.step_id += 1
                    self.terminated = bool(term[0].item())
                    native_truncated = bool(trunc[0].item())
                    if (self.terminated or native_truncated) and not bool(self.env._frozen_envs[0].item()):
                        raise RuntimeError("Native early auto-reset artifact; terminal observation unavailable")
                    self.truncated = native_truncated or (self.step_id >= self.max_steps and not self.terminated)
                    self.success = bool(self.env.termination_manager.get_term("success")[0].item())
                    self.raw_obs = raw
                    self.obs = self.extract_obs(raw)
                    reward = float(rew[0].item())
                    if not np.isfinite(reward):
                        raise ValueError("Nonfinite native reward")
                    self.episode_return += reward
                    self._capture_state()
                except Exception:
                    self.poisoned = True
                    self._write_summary("simulator_error")
                    raise
                records.append(dict(pre_obs=pre_obs, proposed=proposed, action=action.copy(),
                                    reward=reward, terminated=self.terminated, truncated=self.truncated,
                                    success=self.success, step_id=self.step_id - 1))
            steps.append(dict(obs=self.obs, reward=reward, terminated=self.terminated,
                truncated=self.truncated, valid=valid, success=self.success,
                episode_id=self.episode_id, step_id=self.step_id,
                executed_action=action.copy() if valid else np.zeros(8, dtype=np.float32),
                action_source=self.source if valid else "padding"))
        if records:
            try:
                # Each file is one contiguous, single-source chunk. No padded labels are saved.
                payload = {key: np.stack([r["pre_obs"][key] for r in records])
                           for key in ("main_images", "wrist_images", "states")}
                for key in ("proposed", "action", "reward", "terminated", "truncated", "success", "step_id"):
                    payload[key] = np.array([r[key] for r in records])
                payload.update({"final_" + k: self.obs[k] for k in ("main_images", "wrist_images", "states")})
                payload.update(episode_id=self.episode_id, source=self.source,
                               policy_version=self.policy_version, instruction=self.cfg.instruction,
                               branch_id=self.branch["branch_id"] if self.branch else "",
                               teacher_request_id=self.teacher_request_id,
                               initial_state_hash=self.initial_state_hash or "")
                if self.combination:
                    payload.update(combination_id=self.combination["combination_id"],
                        control_epoch=self.combination["control_epoch"],
                        student_prediction_id=student_prediction_id or "",
                        student_policy_sha256=self.combination["student_policy_sha256"])
                path = self.episode_dir / f"chunk_{self.chunk_id:05d}.npz"
                np.savez_compressed(path, **payload)
                if self.branch:
                    self.branch["chunks"].append({"path": path.name, "sha256": self._sha(path)})
                if self.combination:
                    if self.source == "student":
                        self.combination["student_prediction_ids"].append(student_prediction_id)
                    self.combination["chunks"].append(dict(path=path.name, sha256=self._sha(path),
                        source=self.source, control_epoch=self.combination["control_epoch"],
                        step_start=records[0]["step_id"], step_end_exclusive=self.step_id,
                        valid_controls=len(records), teacher_request_id=self.teacher_request_id,
                        student_prediction_id=student_prediction_id or ""))
                self.chunk_id += 1
                self._write_summary("terminal" if self.terminated or self.truncated else "active")
                print(json.dumps(dict(event="chunk", episode_id=self.episode_id, step_id=self.step_id,
                    valid_steps=len(records), success=self.success, terminated=self.terminated,
                    truncated=self.truncated)), flush=True)
            except Exception:
                # Physics already advanced: never continue an episode with an uncertain log.
                self.poisoned = True
                try:
                    self._write_summary("recorder_error")
                except Exception:
                    pass  # Preserve the original persistence error when storage remains unavailable.
                raise
        return dict(steps=steps, episode_id=self.episode_id, step_id=self.step_id)

    def dispatch(self, op, args):
        if op == "metadata":
            return self.metadata
        if op == "reset":
            return self.reset(**args)
        if op == "chunk_step":
            return self.chunk_step(**args)
        if op == "teacher_observation":
            return self.teacher_observation(**args)
        if op == "eef_joint_target":
            return self.eef_joint_target(**args)
        if op == "begin_combination":
            return self.begin_combination(**args)
        if op == "switch_control_source":
            return self.switch_control_source(**args)
        raise ValueError("Unknown environment operation")


def serve(session, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        print(json.dumps(dict(event="ready", port=port, metadata=session.metadata)), flush=True)
        while True:
            connection, _ = listener.accept()
            session.poisoned = True  # A new controller must explicitly reset.
            with connection:
                connection.settimeout(600)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    while True:
                        request = receive_packet(connection)
                        response = dict(version=VERSION, request_id=request.get("request_id"))
                        try:
                            if request.get("version") != VERSION:
                                raise ValueError("Unsupported protocol version")
                            result = session.dispatch(request["op"], request.get("args", {}))
                            response.update(ok=True, result=result)
                        except Exception as exc:
                            if request.get("op") in ("reset", "chunk_step", "begin_correction", "replay_to"):
                                session.poisoned = True
                            traceback.print_exc()
                            response.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                        send_packet(connection, response)
                except (EOFError, ConnectionError, TimeoutError):
                    session._write_summary("terminal" if session.terminated or session.truncated else "controller_disconnect")
