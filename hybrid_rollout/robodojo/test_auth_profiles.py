"""Offline identity/credential/lease tests: no real account or model requests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from .codex_backend.profiles import (PROFILES, account_home, account_lock,
    clean_login_env, config_text, credential_path, initialize, profile, validate_credential)
from .codex_backend.validate import validate_config, validate_home_config
from .skill.run import agent_config


def test_login_group_resumes_missing_sessions_without_overwriting(tmp_path, monkeypatch):
    from .codex_backend import profiles as p
    def fake_login(name, *args):
        home = p.initialize(name, tmp_path)
        auth = home/'auth.json'
        auth.write_text(json.dumps(dict(auth_mode='chatgpt', tokens=dict(
            access_token='synthetic', id_token='synthetic', refresh_token=name,
            account_id=name.split('_')[1]))))
        auth.chmod(0o600)
    fake_login('codex_a', None)
    before = p.credential_path('codex_a', tmp_path).read_bytes()
    calls = []
    def login(name, *args):
        calls.append(name); fake_login(name)
    monkeypatch.setattr(p, 'login_session', login)
    for group in ('a','b','c'): p.login_group(group, '/unused', tmp_path)
    assert len(calls) == 14 and 'codex_a' not in calls
    assert p.credential_path('codex_a', tmp_path).read_bytes() == before
    p.validate_campaign_sessions(tmp_path, p.POOL15_PROFILES)
    for group in ('a','b','c'): p.login_group(group, '/unused', tmp_path)
    assert len(calls) == 14  # Rerunning onboarding is safe and idempotent.
    path = p.credential_path('codex_c_5', tmp_path)
    data = json.loads(path.read_text()); data['tokens']['account_id'] = 'b'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): p.validate_campaign_sessions(tmp_path, p.POOL15_PROFILES)


@pytest.mark.parametrize('name', PROFILES)
def test_profile_config_and_isolation(tmp_path, name):
    home = initialize(name, tmp_path)
    config = validate_config(home/'config.toml', name)
    assert config['model'] == 'gpt-6-astra'
    assert config['model_reasoning_effort'] == 'xhigh'
    assert config['features']['fast_mode'] is False
    assert not (home/'auth.json').exists()
    assert home.stat().st_mode & 0o077 == 0
    assert initialize(name, tmp_path) == home
    for other in PROFILES:
        if profile(other)['provider'] != profile(name)['provider']:
            with pytest.raises(ValueError):
                validate_config(home/'config.toml', other)


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError, match='no automatic fallback'):
        profile('other')


def test_shell_launchers_accept_all_fifteen_sessions_and_no_unknown_names():
    from .codex_backend.profiles import POOL15_PROFILES
    pattern = 'codex_[abc]|codex_[abc]_[2-5]'
    for name in ('acp_startup.sh', 'cluster_entrypoint.sh', 'episode_entrypoint.sh', 'run_local.sh'):
        source = Path(__file__).with_name(name).read_text()
        assert pattern+')' in source
    for name in (*POOL15_PROFILES, 'codex_d', 'codex_a_6', 'codex_b_1', 'codex_c_injected'):
        result = subprocess.run(['bash', '-c', 'case "$1" in '+pattern+') exit 0;; *) exit 1;; esac', 'test', name])
        assert (result.returncode == 0) == (name in POOL15_PROFILES)


def test_accounts_have_separate_credentials_and_exclusive_leases(tmp_path):
    for name in ('codex_a', 'codex_b'):
        home = initialize(name, tmp_path)
        auth = home/'auth.json'
        auth.write_text(json.dumps({'auth_mode': 'chatgpt', 'tokens': {
            'access_token': 'fake-access', 'id_token': 'fake-id', 'refresh_token': 'fake-refresh'}}))
        auth.chmod(0o600)
        assert validate_credential(name, tmp_path) == auth
    assert credential_path('codex_a', tmp_path) != credential_path('codex_b', tmp_path)
    with account_lock('codex_a', tmp_path):
        with pytest.raises(ValueError, match='in use'):
            with account_lock('codex_a', tmp_path):
                pass
        with account_lock('codex_b', tmp_path):
            pass


def test_refresh_is_preserved_and_invalid_credentials_rejected(tmp_path):
    home = initialize('codex_a', tmp_path)
    auth = home/'auth.json'
    auth.write_text('{"auth_mode":"chatgptAuthTokens","tokens":{}}')
    auth.chmod(0o600)
    with pytest.raises(ValueError, match='managed ChatGPT'):
        validate_credential('codex_a', tmp_path)
    initialize('codex_a', tmp_path)
    assert 'chatgptAuthTokens' in auth.read_text()  # init never replaces auth
    auth.chmod(0o644)
    with pytest.raises(ValueError, match='0600'):
        validate_credential('codex_a', tmp_path)


def test_login_environment_does_not_modify_developer_home(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEX_HOME', '/developer/unchanged')
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://wrong.example')
    env = clean_login_env(tmp_path)
    assert env['CODEX_HOME'] == str(tmp_path)
    assert 'OPENAI_API_KEY' not in env and 'OPENAI_BASE_URL' not in env
    assert os.environ['CODEX_HOME'] == '/developer/unchanged'


def test_login_proxy_is_private_and_child_only(monkeypatch, tmp_path):
    proxy_file = tmp_path/'proxy.url'
    proxy_file.write_text('http://test-user:test-pass@proxy.invalid:3128')
    proxy_file.chmod(0o600)
    monkeypatch.setenv('HTTPS_PROXY', 'http://old.invalid:80')
    env = clean_login_env(tmp_path/'account', proxy_file)
    assert env['HTTPS_PROXY'] == proxy_file.read_text()
    assert env['https_proxy'] == env['HTTPS_PROXY']
    assert os.environ['HTTPS_PROXY'] == 'http://old.invalid:80'


def test_all_staged_sessions_require_matching_accounts_and_independent_logins(tmp_path):
    from .codex_backend.profiles import CHATGPT_PROFILES, validate_campaign_sessions
    for name in CHATGPT_PROFILES:
        home = initialize(name, tmp_path)
        auth = home/'auth.json'
        auth.write_text(json.dumps(dict(auth_mode='chatgpt', tokens=dict(
            access_token='fake', id_token='fake', refresh_token='fake-'+name,
            account_id=name.split('_')[1].upper()))))
        auth.chmod(0o600)
    validate_campaign_sessions(tmp_path, CHATGPT_PROFILES)
    auth = credential_path('codex_a_3', tmp_path)
    data = json.loads(auth.read_text())
    data['tokens']['account_id'] = 'B'
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='same intended account'):
        validate_campaign_sessions(tmp_path, CHATGPT_PROFILES)
    # Old topology does not require or consume the newly staged A sessions.
    validate_campaign_sessions(tmp_path)


def test_five_b_sessions_do_not_require_unused_a_logins(tmp_path):
    from .codex_backend.profiles import FIVE_B_PROFILES, validate_campaign_sessions
    for name in FIVE_B_PROFILES:
        home = initialize(name, tmp_path)
        auth = home/'auth.json'
        auth.write_text(json.dumps(dict(auth_mode='chatgpt', tokens=dict(
            access_token='fake', id_token='fake', refresh_token='fake-'+name,
            account_id='B'))))
        auth.chmod(0o600)
    validate_campaign_sessions(tmp_path, FIVE_B_PROFILES)
    auth = credential_path('codex_b_5', tmp_path)
    data = json.loads(auth.read_text())
    data['tokens']['refresh_token'] = 'fake-codex_b_4'
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='independent logins'):
        validate_campaign_sessions(tmp_path, FIVE_B_PROFILES)
    data['tokens']['refresh_token'] = 'fake-codex_b_5'
    data['tokens']['account_id'] = 'wrong-account'
    auth.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='same intended account'):
        validate_campaign_sessions(tmp_path, FIVE_B_PROFILES)


def test_each_episode_keeps_separate_state(monkeypatch, tmp_path):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path/'persistent-account'))
    monkeypatch.setenv('ROLLOUT_CODEX_STATE_DIR', str(tmp_path/'episode-private'))
    cfg = agent_config(tmp_path/'audit', tmp_path/'agent')
    assert cfg['sqlite_home'] == str(tmp_path/'episode-private/runtime_db')
    assert cfg['log_dir'] == str(tmp_path/'episode-private/runtime_logs')


@pytest.mark.parametrize('name', PROFILES)
def test_process_settings_select_profile(name):
    result = subprocess.run([sys.executable, '-c',
        'from hybrid_rollout.robodojo.settings import AUTH_PROFILE,PROVIDER; print(AUTH_PROFILE,PROVIDER)'],
        env=dict(os.environ, ROLLOUT_AUTH_PROFILE=name), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == name+' '+profile(name)['provider']


def test_no_persistent_config_overwrite(tmp_path):
    home = initialize('codex_a', tmp_path)
    (home/'config.toml').write_text('model="different"')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        initialize('codex_a', tmp_path)


def test_managed_home_accepts_only_rollout_trust_metadata_without_writing(tmp_path):
    expected = tmp_path/'expected.toml'
    actual = tmp_path/'actual.toml'
    expected.write_text(config_text('codex_a'))
    agent = tmp_path/'results/old-attempt/controller/codex_workspace/agent'
    text = config_text('codex_a') + f'\n[projects."{agent}"]\ntrust_level = "trusted"\n'
    actual.write_text(text)
    validate_home_config(expected, actual, 'codex_a', tmp_path)
    assert actual.read_text() == text
    actual.write_text(text.replace('xhigh', 'high'))
    with pytest.raises(ValueError):
        validate_home_config(expected, actual, 'codex_a', tmp_path)


@pytest.mark.parametrize('extra', [
    '\n[projects."/tmp/arbitrary"]\ntrust_level="trusted"\n',
    '\n[permissions.extra]\nextends=":danger-full-access"\n',
    '\n[model_providers.other]\nbase_url="https://example.invalid"\n',
    '\n[shell_environment_policy]\ninherit="none"\n',
])
def test_home_settings_drift_remains_rejected(tmp_path, extra):
    expected = tmp_path/'expected.toml'
    actual = tmp_path/'actual.toml'
    expected.write_text(config_text('codex_b'))
    actual.write_text(config_text('codex_b')+extra)
    with pytest.raises(ValueError):
        validate_home_config(expected, actual, 'codex_b', tmp_path)


def test_api_home_does_not_ignore_project_overrides(tmp_path):
    expected = tmp_path/'expected.toml'; actual = tmp_path/'actual.toml'
    expected.write_text(config_text('galbot'))
    actual.write_text(config_text('galbot')+'\n[projects."/tmp"]\ntrust_level="trusted"\n')
    with pytest.raises(ValueError):
        validate_home_config(expected, actual, 'galbot', tmp_path)
