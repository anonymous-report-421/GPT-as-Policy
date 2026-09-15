"""Checkpoint identity/loading contract, copied and scoped to inference."""
import dataclasses
import hashlib
import json
from pathlib import Path
from ..io import require, sha256

BASE_CONFIG = 'pi05_droid_jointpos_polaris'
NATIVE_OPENPI_CONFIG = 'pi05_droid_jointpos'


def checkpoint_identity(path):
    path = Path(path).resolve()
    files = {str(p.relative_to(path)): sha256(p) for folder in ['params', 'assets']
             for p in sorted((path / folder).rglob('*')) if p.is_file()}
    require(any(k.startswith('params/') for k in files), 'Checkpoint has no params')
    require('assets/droid/norm_stats.json' in files, 'DROID normalizer missing')
    return dict(checkpoint=str(path), config=BASE_CONFIG, backend='OpenPI/JAX',
                checkpoint_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                files_sha256=files, action_space='absolute_joint_position_and_gripper_closure',
                gripper_execution_threshold=0.5, action_horizon=15, action_dim=8,
                internal_action_space='delta_first_7_relative_to_current_state')


def data_contract(base_checkpoint):
    from openpi.training import config, checkpoints
    base_checkpoint = Path(base_checkpoint).resolve()
    cfg = config.get_config(NATIVE_OPENPI_CONFIG)
    factory = dataclasses.replace(cfg.data, assets=config.AssetsConfig(
        assets_dir=str(base_checkpoint / 'assets'), asset_id='droid'))
    cfg = dataclasses.replace(cfg, data=factory)
    dc = factory.create(cfg.assets_dirs, cfg.model)
    norms = checkpoints.load_norm_stats(base_checkpoint / 'assets', 'droid')
    return cfg, dataclasses.replace(dc, norm_stats=norms)
