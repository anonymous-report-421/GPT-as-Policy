"""Physical scene records for initial-state restore and validated action replay.

These records are NOT complete intermediate simulator checkpoints. In particular,
they omit RNG streams, pending policy actions, task/manager histories and PhysX
contact/solver internals. Restore the initial state, then replay executed actions.
Core serialization/comparison uses only the standard library. Torch is imported
only by the simulator restore function.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA = "rlinf_robolab.scene_snapshot.v1"
SEQUENCE_SCHEMA = "rlinf_robolab.scene_sequence.v1"
FRAME = "root_position_env_local__quaternion_world_wxyz__velocity_world"
OMITTED_STATE = (
    "python_numpy_torch_and_simulator_rng_streams",
    "pending_policy_action_chunk_and_cursor",
    "observation_action_event_command_manager_history",
    "robolab_predicate_and_subtask_history",
    "sensor_history_and_contact_buffers",
    "physx_contact_solver_and_sleep_internals",
    "runtime_material_randomization_and_static_scene_assets",
)
DEFAULT_TOLERANCES = {
    "position_m": 1e-3,
    "orientation_rad": 1e-2,
    "linear_velocity_m_s": 1e-2,
    "angular_velocity_rad_s": 1e-2,
    "joint_position_rad": 1e-3,
    "joint_velocity_rad_s": 1e-2,
    "joint_position_target_rad": 1e-3,
    "joint_velocity_target_rad_s": 1e-2,
    "joint_effort_target_native": 1e-2,
}
_TARGET_SETTERS = {
    "joint_pos_target": "set_joint_position_target",
    "joint_vel_target": "set_joint_velocity_target",
    "joint_effort_target": "set_joint_effort_target",
}


class SceneRecordError(ValueError):
    """A state record is invalid, incomplete or incompatible with this scene."""


def _plain(value: Any) -> Any:
    """Copy tensors/arrays to JSON primitives without importing their libraries."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise SceneRecordError("State/metadata keys must be nonempty strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SceneRecordError(f"Nonfinite or unsupported record value: {type(value).__name__}")


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _vector(value: Any, length: int | None, path: str) -> list[float]:
    if not isinstance(value, list) or (length is not None and len(value) != length) or not value:
        raise SceneRecordError(f"{path}: wrong vector shape")
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           or not math.isfinite(item) for item in value):
        raise SceneRecordError(f"{path}: expected finite real values")
    return [float(item) for item in value]


def _row(value: Any, length: int | None, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 1:
        raise SceneRecordError(f"{path}: expected exactly one environment row")
    return _vector(value[0], length, path)


def _quaternion(values: list[float]) -> list[float]:
    # Preserve the left-to-right float arithmetic used by existing Python 3.11
    # records. Python 3.12's improved sum() changes some normalized components
    # by one ULP, which changes the canonical physical-state SHA256.
    squared_norm = 0.0
    for value in values:
        squared_norm += value * value
    norm = math.sqrt(squared_norm)
    if norm < 1e-12 or not math.isfinite(norm):
        raise SceneRecordError("Zero or invalid quaternion")
    return [value / norm for value in values]


def _validate_scene(state: dict) -> None:
    families = {"articulation", "rigid_object", "deformable_object"}
    if not isinstance(state, dict) or set(state) != families:
        raise SceneRecordError(f"scene_state must contain exactly {sorted(families)}")
    for family, assets in state.items():
        if not isinstance(assets, dict):
            raise SceneRecordError(f"{family}: expected an asset mapping")
        for name, fields in assets.items():
            path = f"{family}/{name}"
            if not isinstance(name, str) or not name or "/" in name:
                raise SceneRecordError("Scene entity names must be nonempty and cannot contain '/'")
            expected = ({"nodal_position", "nodal_velocity"} if family == "deformable_object"
                        else {"root_pose", "root_velocity"})
            if family == "articulation":
                expected |= {"joint_position", "joint_velocity"}
            if not isinstance(fields, dict) or set(fields) != expected:
                raise SceneRecordError(f"{path}: missing or unexpected state fields")
            if family == "deformable_object":
                counts = []
                for key, value in fields.items():
                    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list) or not value[0]:
                        raise SceneRecordError(f"{path}/{key}: expected [1,nodes,3]")
                    for node in value[0]:
                        _vector(node, 3, path + "/" + key)
                    counts.append(len(value[0]))
                if counts[0] != counts[1]:
                    raise SceneRecordError(f"{path}: nodal position/velocity lengths differ")
            else:
                pose = _row(fields["root_pose"], 7, path + "/root_pose")
                _quaternion(pose[3:])
                _row(fields["root_velocity"], 6, path + "/root_velocity")
                if family == "articulation":
                    positions = _row(fields["joint_position"], None, path + "/joint_position")
                    _row(fields["joint_velocity"], len(positions), path + "/joint_velocity")


def _physical_payload(snapshot: dict) -> dict:
    """Physical identity excludes episode IDs/clocks and normalizes q/-q."""
    state = copy.deepcopy(snapshot["scene_state"])
    for family in ("articulation", "rigid_object"):
        for fields in state[family].values():
            quat = _quaternion(fields["root_pose"][0][3:])
            first = next((value for value in quat if abs(value) > 1e-15), 1.0)
            if first < 0:
                quat = [-value for value in quat]
            fields["root_pose"][0][3:] = [0.0 if value == 0 else value for value in quat]
    return {"frame": snapshot["frame"], "scene_state": state,
            "controller_targets": snapshot["controller_targets"],
            "joint_names": snapshot["joint_names"]}


def snapshot_from_state(
    scene_state: Mapping[str, Any], *, episode_id: str, step_id: int,
    kind: str = "post_step", controller_targets: Mapping | None = None,
    joint_names: Mapping | None = None, clocks: Mapping | None = None,
    metadata: Mapping | None = None,
) -> dict:
    """Build a single-env snapshot from native get_state-shaped arrays or lists."""
    snapshot = _plain(dict(
        schema=SCHEMA, scope="physical_scene_and_targets_not_complete_checkpoint",
        frame=FRAME, kind=kind, episode_id=episode_id, step_id=step_id,
        scene_state=scene_state, controller_targets=controller_targets or {},
        joint_names=joint_names or {}, clocks=clocks or {}, metadata=metadata or {},
        omitted_state=list(OMITTED_STATE),
    ))
    validate_snapshot(snapshot, verify_hash=False)
    snapshot["state_sha256"] = hashlib.sha256(_bytes(_physical_payload(snapshot))).hexdigest()
    return snapshot


def validate_snapshot(snapshot: dict, *, verify_hash: bool = True) -> None:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA or snapshot.get("frame") != FRAME:
        raise SceneRecordError("Unsupported scene snapshot schema or coordinate frame")
    if snapshot.get("scope") != "physical_scene_and_targets_not_complete_checkpoint":
        raise SceneRecordError("Snapshot must explicitly declare its limited restoration scope")
    if not isinstance(snapshot.get("episode_id"), str) or not snapshot["episode_id"]:
        raise SceneRecordError("episode_id must be a nonempty string")
    step = snapshot.get("step_id")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise SceneRecordError("step_id must be a nonnegative integer")
    if snapshot.get("kind") not in ("initial", "post_step"):
        raise SceneRecordError("kind must be initial or post_step")
    if (snapshot["kind"] == "initial") != (step == 0):
        raise SceneRecordError("Only the canonical initial snapshot may have step_id=0")
    _plain(snapshot)
    _validate_scene(snapshot.get("scene_state"))
    for field in ("controller_targets", "joint_names", "clocks", "metadata"):
        if not isinstance(snapshot.get(field), dict):
            raise SceneRecordError(f"{field} must be a mapping")
    if (snapshot["kind"] == "initial" and "episode_length_buf" in snapshot["clocks"]
            and snapshot["clocks"]["episode_length_buf"] != [0]):
        raise SceneRecordError("Canonical initial state must have episode_length_buf=0")
    for name, fields in snapshot["controller_targets"].items():
        if name not in snapshot["scene_state"]["articulation"]:
            raise SceneRecordError(f"Controller target for unknown articulation {name}")
        dof = len(snapshot["scene_state"]["articulation"][name]["joint_position"][0])
        if not isinstance(fields, dict) or not set(fields).issubset(_TARGET_SETTERS):
            raise SceneRecordError(f"Unknown controller target for {name}")
        for key, value in fields.items():
            _row(value, dof, f"controller_targets/{name}/{key}")
    for name, names in snapshot["joint_names"].items():
        if name not in snapshot["scene_state"]["articulation"]:
            raise SceneRecordError(f"Joint names for unknown articulation {name}")
        dof = len(snapshot["scene_state"]["articulation"][name]["joint_position"][0])
        if (not isinstance(names, list) or len(names) != dof
                or any(not isinstance(x, str) or not x for x in names)
                or len(set(names)) != dof):
            raise SceneRecordError(f"Invalid joint name ordering for {name}")
    if snapshot.get("omitted_state") != list(OMITTED_STATE):
        raise SceneRecordError("Snapshot omission declaration changed")
    if verify_hash:
        expected = hashlib.sha256(_bytes(_physical_payload(snapshot))).hexdigest()
        if snapshot.get("state_sha256") != expected:
            raise SceneRecordError("Physical state SHA256 mismatch")


def capture_scene_state(env: Any, *, episode_id: str, step_id: int,
                        kind: str = "post_step", metadata: Mapping | None = None) -> dict:
    """Capture after the final reset or one valid env.step(); never on padding."""
    if int(env.num_envs) != 1:
        raise SceneRecordError("Scene recording currently supports one environment per process")
    state = _plain(env.scene.get_state(is_relative=True))
    targets, names, clocks = {}, {}, {}
    for name, articulation in env.scene.articulations.items():
        targets[name] = {key: _plain(getattr(articulation.data, key))
                         for key in _TARGET_SETTERS if hasattr(articulation.data, key)}
        if hasattr(articulation, "joint_names"):
            names[name] = _plain(articulation.joint_names)
    for key in ("episode_length_buf", "_sim_step_counter", "common_step_counter"):
        if hasattr(env, key):
            clocks[key] = _plain(getattr(env, key))
    if kind == "initial" and "episode_length_buf" in clocks and clocks["episode_length_buf"] != [0]:
        raise SceneRecordError("Canonical initial state must be captured at episode_length_buf=0")
    meta = dict(metadata or {})
    if hasattr(env.scene, "env_origins"):
        meta["captured_env_origins"] = _plain(env.scene.env_origins)
    if hasattr(env, "step_dt"):
        meta["control_dt"] = float(env.step_dt)
    return snapshot_from_state(state, episode_id=episode_id, step_id=step_id, kind=kind,
                               controller_targets=targets, joint_names=names,
                               clocks=clocks, metadata=meta)


def _save_exclusive(path: str | Path, value: dict) -> dict:
    path = Path(path)
    payload = _bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL also refuses symlinks. A partial file from a killed writer is never
    # silently replaced; its invalid JSON/hash makes recovery explicit.
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def save_snapshot(path: str | Path, snapshot: dict) -> dict:
    validate_snapshot(snapshot)
    return {**_save_exclusive(path, snapshot), "state_sha256": snapshot["state_sha256"]}


def _read(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise SceneRecordError("State record exceeds the 128 MiB read limit")
    try:
        return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
            SceneRecordError(f"Nonfinite JSON value: {value}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SceneRecordError("Invalid or incomplete scene record") from error


def load_snapshot(path: str | Path) -> dict:
    snapshot = _read(path)
    validate_snapshot(snapshot)
    return snapshot


def save_state_sequence(path: str | Path, snapshots: Sequence[dict]) -> dict:
    """Persist one contiguous chunk of real post-step snapshots, without padding."""
    records = list(snapshots)
    _validate_sequence(records)
    return _save_exclusive(path, {"schema": SEQUENCE_SCHEMA, "snapshots": records})


def _validate_sequence(records: Any) -> None:
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise SceneRecordError("A state sequence must contain 1 to 64 valid control steps")
    for record in records:
        validate_snapshot(record)
        if record["kind"] != "post_step":
            raise SceneRecordError("State sequences contain post_step records; save initial state separately")
    first = records[0]
    if any(record["episode_id"] != first["episode_id"] or record["step_id"] != first["step_id"] + i
           for i, record in enumerate(records)):
        raise SceneRecordError("State sequence crosses an episode boundary or has discontinuous step IDs")


def load_state_sequence(path: str | Path) -> list[dict]:
    value = _read(path)
    if not isinstance(value, dict) or value.get("schema") != SEQUENCE_SCHEMA:
        raise SceneRecordError("Unsupported state sequence schema")
    records = value.get("snapshots")
    _validate_sequence(records)
    return records


def _flatten(tree: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten(value, path))
        else:
            result[path] = value
    return result


def _norm_difference(first: list, second: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(first, second)))


def compare_snapshots(reference: dict, actual: dict, tolerances: Mapping | None = None) -> dict:
    """Compare physical states/targets; episode IDs, clocks and RNG are not compared.

    Root translation/velocity errors use vector L2 norms, joints maximum absolute
    error, and rotations the shortest quaternion angle in radians (q == -q).
    """
    validate_snapshot(reference)
    validate_snapshot(actual)
    limits = dict(DEFAULT_TOLERANCES)
    if tolerances:
        if set(tolerances) - set(limits):
            raise SceneRecordError("Unknown comparison tolerance")
        limits.update(tolerances)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in limits.values()):
        raise SceneRecordError("Comparison tolerances must be finite and nonnegative")
    expected = _flatten({"scene_state": reference["scene_state"], "controller_targets": reference["controller_targets"]})
    observed = _flatten({"scene_state": actual["scene_state"], "controller_targets": actual["controller_targets"]})
    missing, unexpected = sorted(set(expected) - set(observed)), sorted(set(observed) - set(expected))
    metrics, mismatches = [], []
    if reference["joint_names"] != actual["joint_names"]:
        mismatches.append("joint_names/order")

    def add(path, metric, error):
        metrics.append(dict(path=path, metric=metric, error=error,
                            tolerance=limits[metric], passed=error <= limits[metric]))

    for path in sorted(set(expected) & set(observed)):
        first, second = expected[path][0], observed[path][0]
        if len(first) != len(second):
            mismatches.append(path + ": shape")
            continue
        key = path.rsplit("/", 1)[-1]
        if key == "root_pose":
            add(path, "position_m", _norm_difference(first[:3], second[:3]))
            one, two = _quaternion(first[3:]), _quaternion(second[3:])
            if sum(x * y for x, y in zip(one, two)) < 0:
                two = [-value for value in two]
            # atan2 avoids acos cancellation near zero and gives exactly zero
            # for equal normalized quaternions, including q versus -q.
            difference = _norm_difference(one, two)
            total = math.sqrt(sum((x + y) ** 2 for x, y in zip(one, two)))
            add(path, "orientation_rad", 4 * math.atan2(difference, total))
        elif key == "root_velocity":
            add(path, "linear_velocity_m_s", _norm_difference(first[:3], second[:3]))
            add(path, "angular_velocity_rad_s", _norm_difference(first[3:], second[3:]))
        elif key in ("nodal_position", "nodal_velocity"):
            add(path, "position_m" if key == "nodal_position" else "linear_velocity_m_s",
                max(_norm_difference(one, two) for one, two in zip(first, second)))
        else:
            metric = {"joint_position": "joint_position_rad", "joint_velocity": "joint_velocity_rad_s",
                      "joint_pos_target": "joint_position_target_rad", "joint_vel_target": "joint_velocity_target_rad_s",
                      "joint_effort_target": "joint_effort_target_native"}[key]
            add(path, metric, max(abs(x - y) for x, y in zip(first, second)))
    return dict(passed=not (missing or unexpected or mismatches) and all(item["passed"] for item in metrics),
                reference_state_sha256=reference["state_sha256"], actual_state_sha256=actual["state_sha256"],
                missing=missing, unexpected=unexpected, mismatches=mismatches, metrics=metrics,
                compared_scope="physical_state_and_controller_targets_only",
                complete_checkpoint_validated=False)


def _state_to_torch(state: Any, device: Any) -> Any:
    import torch
    if isinstance(state, dict):
        return {key: _state_to_torch(value, device) for key, value in state.items()}
    return torch.as_tensor(state, dtype=torch.float32, device=device)


RESET_CONTRACT = "droid_canonical_targets_camera_refresh.v2"


def canonicalize_droid_targets(env: Any) -> dict:
    """Clear only the DROID robot's residual PD/effort targets, without stepping.

    Fresh episode only. Recorded initial-state restore must retain its recorded
    targets instead. Passive mimic joints receive measured position targets in
    the robot buffer; no actuator or unrelated scene articulation is changed.
    """
    if int(env.num_envs) != 1 or "robot" not in env.scene.articulations:
        raise SceneRecordError("Canonical reset requires one DROID robot")
    robot = env.scene.articulations["robot"]
    names = list(robot.joint_names)
    controlled = [f"panda_joint{i}" for i in range(1, 8)] + ["finger_joint"]
    actuated = set()
    for actuator in robot.actuators.values():
        indices = actuator.joint_indices
        indices = list(range(len(names)))[indices] if isinstance(indices, slice) else _plain(indices)
        actuated.update(names[int(index)] for index in indices)
    if actuated != set(controlled):
        raise SceneRecordError(f"Unexpected DROID actuator joint mapping: {sorted(actuated)}")
    positions = _plain(robot.data.joint_pos)
    _row(positions, len(names), "robot/joint_pos")
    zero = [[0.0] * len(names)]
    robot.set_joint_position_target(_state_to_torch(positions, env.device))
    robot.set_joint_velocity_target(_state_to_torch(zero, env.device))
    robot.set_joint_effort_target(_state_to_torch(zero, env.device))
    env.scene.write_data_to_sim()
    for field, expected in (("joint_pos_target", positions),
                            ("joint_vel_target", zero), ("joint_effort_target", zero)):
        if _plain(getattr(robot.data, field)) != expected:
            raise SceneRecordError(f"Canonical DROID target write did not persist: {field}")
    return {"robot": "robot", "actuated_joint_names": controlled,
            "robot_target_buffer_dofs": len(names),
            "position_target": "measured_joint_position", "velocity_target": 0.0,
            "effort_target": 0.0, "physical_steps": 0}


def refresh_initial_observation(env: Any, *, render_frames: int = 3):
    """Flush rendered sensors after reset without advancing physical time.

    IsaacLab Camera.reset marks its cached output outdated. update(0, True)
    alone does not do that when update_period > 0. Render first, then reset and
    reread cameras, and compute observations without another history update.
    """
    if type(render_frames) is not int or not 1 <= render_frames <= 8:
        raise SceneRecordError("Reset render count must be an integer in [1, 8]")
    before = capture_scene_state(env, episode_id="reset_refresh", step_id=0, kind="initial")
    cameras = {name: sensor for name, sensor in env.scene.sensors.items()
               if "rgb" in getattr(sensor.cfg, "data_types", ())}
    required = {"over_shoulder_left_camera", "wrist_cam"}
    if not required.issubset(cameras):
        raise SceneRecordError("Reset refresh requires native main and wrist RGB cameras")
    env.scene.write_data_to_sim()
    env.sim.forward()
    for _ in range(render_frames):
        env.sim.render()
    for camera in cameras.values():
        camera.reset()
        camera.update(0.0, force_recompute=True)
    observations = env.observation_manager.compute(update_history=False)
    env.obs_buf = observations
    after = capture_scene_state(env, episode_id="reset_refresh", step_id=0, kind="initial")
    if before["clocks"] != after["clocks"] or before["state_sha256"] != after["state_sha256"]:
        raise SceneRecordError("Camera refresh changed physical state or simulation clocks")
    report = {"render_frames": render_frames, "cameras": sorted(cameras),
              "physical_steps": 0, "clocks_unchanged": True,
              "physical_state_unchanged": True,
              "sensor_reset_then_update_dt": 0.0, "observation_update_history": False}
    return observations, report


def randomize_banana_initial_layout(env: Any) -> dict:
    """Reuse the upstream Banana 10 cm reset event; no environment step."""
    import torch
    from robolab.core.events.reset_pose import reset_pose_uniform, _get_object_radius
    if int(env.num_envs) != 1 or not {"banana", "bowl", "table"}.issubset(env.scene.rigid_objects):
        raise SceneRecordError("Banana layout randomization requires the BananaInBowl scene")
    reset_pose_uniform(env, torch.arange(1, dtype=torch.int64, device=env.device),
        pose_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
        velocity_range={}, asset_cfg=["banana", "bowl"],
        reset_to_default_otherwise=True, use_collision_check=True)
    # Upstream can return its last colliding sample after exhausting retries,
    # and its radius helper returns zero if geometry lookup fails. Check the
    # final candidate explicitly before the fixture generator admits it.
    states = _plain(env.scene.get_state(is_relative=True))["rigid_object"]
    radii = {name: float(_get_object_radius(env.scene.rigid_objects[name], env_id=0))
             for name in ("banana", "bowl")}
    failures = []
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii.values()):
        failures.append("bounding_radius_unavailable")
    for name in ("banana", "bowl"):
        current = states[name]["root_pose"][0][:3]
        default = _plain(env.scene.rigid_objects[name].data.default_root_state)[0][:3]
        if any(abs(current[index] - default[index]) > 0.100001 for index in (0, 1)):
            failures.append(name + "_outside_xy_range")
        if abs(current[2] - default[2]) > 1e-6:
            failures.append(name + "_changed_z")
    banana = states["banana"]["root_pose"][0]
    bowl = states["bowl"]["root_pose"][0]
    distance = math.hypot(banana[0] - bowl[0], banana[1] - bowl[1])
    if distance < radii["banana"] + radii["bowl"] + 0.01:
        failures.append("upstream_xy_bounding_circles_overlap")
    return {"upstream": "robolab.core.events.reset_pose.reset_pose_uniform",
            "task_variant": "BananaInBowlUniformInitPose10cmTask",
            "assets": ["banana", "bowl"], "xy_range_m": [-0.1, 0.1],
            "collision_check": True, "physical_steps": 0,
            "legality": {"passed": not failures, "failures": failures,
                "scope": "configured_xy_range_and_upstream_xy_bounding_circles",
                "collision_margin_m": 0.01, "upstream_max_sampling_retries": 100,
                "bounding_radii_m": {name: radius if math.isfinite(radius) else None
                                     for name, radius in radii.items()},
                "does_not_validate_full_mesh_contacts_or_task_success": True}}


def restore_initial_state(env: Any, snapshot: dict, *, seed: int | None = None):
    """Wrap upstream env.reset_to for canonical initial state only.

    Call between episodes. The caller must finalize/archive the previous episode
    before this mutation. This clears RoboLab freeze state, performs a native
    reset, then resets to recorded scene state and restores controller targets.
    Like RoboLab's restore_recorded_initial_state this uses env.reset_to with
    env-relative get_state data. Unlike that HDF5 convenience function, missing
    assets or fields fail instead of silently falling back to the default scene.
    The returned comparison checks this call's physical state only; it does not
    establish that future PhysX replay is deterministic or that RNG was restored.
    Replay executed actions to reach a later tick; intermediate records are refused.
    """
    validate_snapshot(snapshot)
    if snapshot["kind"] != "initial" or snapshot["step_id"] != 0:
        raise SceneRecordError("Only canonical initial-state restoration is supported")
    if int(env.num_envs) != 1 or snapshot["scene_state"]["deformable_object"]:
        raise SceneRecordError("Restore currently supports one rigid/articulated RoboLab environment")
    current = capture_scene_state(env, episode_id="restore_compatibility_check", step_id=1)
    structure = compare_snapshots(snapshot, current)
    if structure["missing"] or structure["unexpected"] or structure["mismatches"]:
        details = {key: structure[key] for key in ("missing", "unexpected", "mismatches")}
        raise SceneRecordError("Snapshot and live scene assets, joint order or target fields differ: "
                               + json.dumps(details, sort_keys=True))
    if not hasattr(env, "reset_eval_state"):
        raise SceneRecordError("Expected a RoboLab environment with reset_eval_state")
    scene = copy.deepcopy(snapshot["scene_state"])
    for family in ("articulation", "rigid_object"):
        for fields in scene[family].values():
            fields["root_pose"][0][3:] = _quaternion(fields["root_pose"][0][3:])
    # Resolve tensors before mutating the simulator so conversion failures are safe.
    scene = _state_to_torch(scene, env.device)
    targets = _state_to_torch(snapshot["controller_targets"], env.device)
    env.reset_eval_state()
    env.reset(seed=seed)
    observations, extras = env.reset_to(scene, env_ids=None, seed=None, is_relative=True)
    for name, fields in targets.items():
        articulation = env.scene.articulations[name]
        for field, values in fields.items():
            getattr(articulation, _TARGET_SETTERS[field])(values)
    env.scene.write_data_to_sim()
    restored = capture_scene_state(env, episode_id=snapshot["episode_id"], step_id=0, kind="initial")
    validation = compare_snapshots(snapshot, restored)
    if not validation["passed"]:
        details = {key: validation[key] for key in ("missing", "unexpected", "mismatches")}
        details["failed_metrics"] = [item for item in validation["metrics"] if not item["passed"]]
        raise SceneRecordError("Initial-state restore failed physical-state/target comparison: "
                               + json.dumps(details, sort_keys=True))
    return observations, {**extras, "scene_restore": validation}
