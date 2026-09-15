"""Checkpoint identity/loading contract, copied and scoped to inference."""
import dataclasses
import hashlib
import json
from pathlib import Path
from ..io import require, sha256

ROBODOJO_CONFIG = 'pi05_base_aloha_full_sim_arx-x5_seed_0'


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = {str(p.relative_to(path)): sha256(p) for folder in ['params', 'assets']
             for p in sorted((path / folder).rglob('*')) if p.is_file()}
    require(any(k.startswith('params/') for k in files), 'Checkpoint has no params')
    asset = 'arx_x5_sim'
    require(f'assets/{asset}/norm_stats.json' in files, f'{asset} normalizer missing')
    return dict(checkpoint=str(path), config=ROBODOJO_CONFIG, backend='OpenPI/JAX',
                checkpoint_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                files_sha256=files, robot_profile='robodojo', action_horizon=50,
            action_dim=14, action_space='left_joints6_opening1_right_joints6_opening1',
            gripper_execution_threshold=None, gripper_semantics='continuous_0_closed_1_open',
            internal_action_space='delta_arm_joints_relative_to_current_state',
            cameras=['cam_high', 'cam_left_wrist', 'cam_right_wrist'])


def data_contract(base_checkpoint):
    from openpi.training import config, checkpoints
    base_checkpoint = Path(base_checkpoint).resolve()
    asset = 'arx_x5_sim'
    cfg = config.get_config(ROBODOJO_CONFIG)
    factory = dataclasses.replace(cfg.data, assets=config.AssetsConfig(
        assets_dir=str(base_checkpoint / 'assets'), asset_id=asset))
    cfg = dataclasses.replace(cfg, data=factory)
    dc = factory.create(cfg.assets_dirs, cfg.model)
    norms = checkpoints.load_norm_stats(base_checkpoint / 'assets', asset)
    return cfg, dataclasses.replace(dc, norm_stats=norms)
