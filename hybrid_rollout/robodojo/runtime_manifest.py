"""Write a secret-free record of every environment value used by a rollout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform

from .codex_backend.validate import validate_config
from .io import sha256, write_json
from .settings import AUTH_PROFILE, AUTH_MODE, BASE_URL, EFFORT, MODEL, PROVIDER, WIRE_API


PUBLIC_ENVIRONMENT = (
    'ROLLOUT_EVALUATION_METHOD', 'ROLLOUT_PHASE1_CERTIFICATE',
    'ROLLOUT_AUTH_PROFILE', 'ROLLOUT_CODEX_STATE_DIR', 'ROLLOUT_GATEWAY_NO_PROXY',
    'CODE_ROOT', 'RUNTIME_ROOT', 'RESULTS_ROOT', 'ROBODOJO_SOURCE',
    'OPENPI_SOURCE', 'OPENPI_PYTHON', 'ROBODOJO_PYTHON', 'CODEX_BIN',
    'RUN_ID', 'TASK', 'SEED', 'POLICY_PORT', 'SIM_PORT', 'CHECKPOINT',
    'POLICY_GPU', 'SIM_GPU', 'SIM_EXTRA_PYTHONPATH', 'TASK_ROOT',
    'OPENPI_DATA_HOME', 'MAX_DECISIONS', 'DEBUG_VIDEO_FONT',
    'DAGGER_NVIDIA_RUNTIME', 'DAGGER_SYSROOT', 'LD_LIBRARY_PATH',
    'ROLLOUT_SHARED_ROOT', 'ROLLOUT_EXPERIMENT_ID', 'ROLLOUT_REPLICA_ID',
    'ROLLOUT_ATTEMPT', 'ROLLOUT_IMAGE_REF', 'ROLLOUT_SCHEDULER_JOB_ID',
    'ROLLOUT_CODEX_CONFIG', 'ROLLOUT_CREDENTIAL_SOURCE',
    'ROLLOUT_HTTPS_PROXY',
    'ROLLOUT_HTTPS_PROXY_FILE',
    'ROLLOUT_COUNT', 'ROLLOUT_CASE_ID', 'ROLLOUT_SEED_BASE', 'ROLLOUT_INDEX', 'ROLLOUT_MAX_SECONDS',
    'ROLLOUT_CODEX_VERSION', 'CODEX_MAX_TOTAL_TOKENS', 'PYTHONDONTWRITEBYTECODE',
    'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCH_EXTENSIONS_DIR', 'MPLCONFIGDIR',
    'ROLLOUT_RUN_UID', 'ROLLOUT_RUN_GID', 'DAGGER_GRAPHICS_ROOT',
    'PATH', 'TMPDIR', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_STATE_HOME',
    'ROLLOUT_CODEX_HOME_DIR', 'ROLLOUT_BOOTSTRAP_LOG', 'CODEX_IMAGE_MAX_EDGE',
    'EVAL_SEED', 'ROLLOUT_EVAL_MANIFEST', 'ROLLOUT_EVAL_MANIFEST_SHA256',
    'ROLLOUT_EVAL_SEED', 'ROLLOUT_LAYOUT_ID', 'ROLLOUT_CASE_FILE', 'ROLLOUT_ARCHIVE',
    'ROLLOUT_BASE_TASK', 'ROLLOUT_VARIANT', 'ROBODOJO_RUN_ID',
)


def make_manifest(archive: Path, config_path: Path) -> dict:
    from .method import evaluation_method
    method = evaluation_method()
    validate_config(config_path)
    values = {name: os.environ[name] for name in PUBLIC_ENVIRONMENT if name in os.environ}
    forbidden = [name for name in values if set(name.split('_')) & {'KEY', 'TOKEN', 'SECRET', 'PASSWORD'}]
    if forbidden:
        raise ValueError(f'Secret-like environment names cannot be archived: {forbidden}')
    snapshot = str(Path(archive).resolve()/'source_snapshot')
    openpi = values.get('OPENPI_SOURCE', '')
    robodojo = values.get('ROBODOJO_SOURCE', '')
    derived = {
        'policy': {
            'CUDA_VISIBLE_DEVICES': values.get('POLICY_GPU', '0'),
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
            'PYTHONPATH': f'{snapshot}:{openpi}/src:{openpi}/packages/openpi-client/src',
        },
        'simulator': {
            'CUDA_VISIBLE_DEVICES': values.get('SIM_GPU', '1'),
            'PYTHONPATH': (f'{snapshot}:{robodojo}:{robodojo}/XPolicyLab:'
                           f'{robodojo}/third_party/curobo' +
                           (f':{values["SIM_EXTRA_PYTHONPATH"]}' if values.get('SIM_EXTRA_PYTHONPATH') else '')),
            'DAGGER_NVIDIA_RUNTIME': values.get('DAGGER_NVIDIA_RUNTIME', 'derived from container home and NVIDIA driver'),
            'DAGGER_SYSROOT': values.get('DAGGER_SYSROOT', 'derived from container home'),
            'LD_LIBRARY_PATH': 'prepended by robodojo_server/runtime.sh',
            'VK_ICD_FILENAMES': 'derived by robodojo_server/runtime.sh',
            '__EGL_VENDOR_LIBRARY_FILENAMES': 'derived by robodojo_server/runtime.sh',
            'XDG_RUNTIME_DIR': f'{values.get("TASK_ROOT", values.get("RUNTIME_ROOT", ""))}/runtime/xdg',
            'OMNI_KIT_ACCEPT_EULA': 'Y',
        },
        'controller': {
            'CODEX_HOME': values.get('ROLLOUT_CODEX_HOME_DIR', 'per-episode private shared directory'),
            'OPENAI_API_KEY': ('absent; managed ChatGPT file auth' if AUTH_MODE == 'chatgpt'
                               else 'present only in the Codex controller process; value not archived'),
            'NO_PROXY': '127.0.0.1,localhost,' + values.get('ROLLOUT_GATEWAY_NO_PROXY', 'gateway.example.invalid'),
            'no_proxy': '127.0.0.1,localhost,' + values.get('ROLLOUT_GATEWAY_NO_PROXY', 'gateway.example.invalid'),
            'PYTHONPATH': f'{snapshot}:{openpi}/packages/openpi-client/src',
        },
    }
    if method == 'gpt_only':
        derived['policy'] = dict(enabled=False, checkpoint_loaded=False, inference_calls=0)
        derived['controller']['PYTHONPATH'] = snapshot
    return {
        'schema': 'hybrid_rollout.robodojo.launch.v1',
        'evaluation_method': method,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'archive': str(Path(archive).resolve()),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'backend': {
            'auth_profile': AUTH_PROFILE, 'auth_mode': AUTH_MODE,
            'provider': PROVIDER, 'model': MODEL, 'reasoning_effort': EFFORT,
            'base_url': BASE_URL, 'wire_api': WIRE_API,
            'credential_value_archived': False,
        },
        'resolved_environment': values,
        'derived_process_environment': derived,
        'codex_config_sha256': sha256(config_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, make_manifest(args.archive, args.config))


if __name__ == '__main__':
    main()
