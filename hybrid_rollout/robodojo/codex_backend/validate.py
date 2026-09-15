"""Fail closed when the rollout Codex config diverges from the pinned backend."""
from __future__ import annotations

import argparse
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 environments used by some simulators.
    import tomli as tomllib

from ..settings import EFFORT, MODEL, WIRE_API
from .profiles import profile, shared_root


def validate_config(path: Path, profile_name=None) -> dict:
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text())
    identity = profile(profile_name)
    provider_name = identity['provider']
    expected = {
        'model_provider': provider_name,
        'model': MODEL,
        'model_reasoning_effort': EFFORT,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{key} must be {value!r}, got {config.get(key)!r}')
    provider = config.get('model_providers', {}).get(provider_name, {})
    provider_expected = {
        'base_url': identity['base_url'],
        'env_key': 'OPENAI_API_KEY',
        'wire_api': WIRE_API,
    }
    if identity['auth_mode'] == 'api':
        for key, value in provider_expected.items():
            if provider.get(key) != value:
                raise ValueError(f'model_providers.{provider_name}.{key} must be {value!r}')
        if provider.get('experimental_bearer_token') or provider.get('requires_openai_auth'):
            raise ValueError('The rollout gateway must use only its environment key')
    elif (config.get('model_providers') or config.get('chatgpt_base_url')
            or config.get('cli_auth_credentials_store') != 'file'
            or config.get('forced_login_method') != 'chatgpt'):
        raise ValueError('Managed accounts require built-in OpenAI provider and isolated file auth')
    features = config.get('features', {})
    expected_features = {'codex_hooks': True, 'hooks': True, 'fast_mode': False}
    if any(features.get(key) is not value for key, value in expected_features.items()):
        raise ValueError('Rollout Codex feature isolation differs from the pinned configuration')
    if config.get('service_tier') not in (None, 'default'):
        raise ValueError('Rollout Codex Fast/Priority service tiers are prohibited')
    if any(value is False for key, value in features.items() if key != 'fast_mode'):
        raise ValueError('Backend isolation must not disable agent tool capabilities')
    if config.get('analytics', {}).get('enabled') is not False:
        raise ValueError('Rollout Codex analytics must remain disabled')
    return config


def validate_home_config(expected_path, actual_path, profile_name=None, shared=None):
    """Compare effective settings, permitting only Codex's rollout trust records.

    Persistent managed homes acquire [projects] entries on thread creation.
    They are not backend changes. Do not rewrite the account config or ignore
    arbitrary extra settings; model, provider, permissions and features must
    still match the freshly generated, validated episode config exactly.
    """
    expected = validate_config(expected_path, profile_name)
    actual = validate_config(actual_path, profile_name)
    if profile(profile_name)['auth_mode'] == 'chatgpt':
        projects = actual.pop('projects', {})
        if not isinstance(projects, dict):
            raise ValueError('Invalid persistent Codex project metadata')
        root = shared_root(shared)/'results'
        for name, settings in projects.items():
            path = Path(name)
            if (not path.is_absolute() or not path.resolve().is_relative_to(root)
                    or path.parts[-3:] != ('controller', 'codex_workspace', 'agent')
                    or settings != {'trust_level': 'trusted'}):
                raise ValueError('Unexpected persistent Codex project metadata')
    if actual != expected:
        raise ValueError('Isolated Codex effective settings differ from the validated rollout config')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--home-config', type=Path)
    parser.add_argument('--shared-root', type=Path)
    args = parser.parse_args()
    if args.home_config:
        validate_home_config(args.config, args.home_config, shared=args.shared_root)
    else:
        validate_config(args.config)


if __name__ == '__main__':
    main()
