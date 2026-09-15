"""Explicit rollout identities; never read or write the developer's Codex home.

Managed ChatGPT credentials stay in one persistent private home per account.
One serialized job stream owns each independently logged-in home; per-rollout DB, notes and threads
remain separate. No automatic provider/account fallback or token refresh API.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

DEFAULT_SHARED = '/mnt/rollout/robodojo_mixed_control'
ORCAROUTER_BASE_URL = 'https://api.orcarouter.ai/v1'
PROFILES = {
    'galbot': ('api', 'LiteLLM', 'https://gateway.example.invalid'),
    'koozhan': ('api', 'Koozhan', 'https://relay.example.invalid/v1'),
    # OrcaRouter is a first-class provider with two explicit authentication
    # choices that share one inference identity: a pasted API key, and a PKCE
    # sign-in that issues the same kind of key. Their credentials live in the
    # private rollout store as private/orcarouter.key and
    # private/orcarouter_oauth.key. See codex_backend/orcarouter/.
    'orcarouter': ('api', 'orcarouter', ORCAROUTER_BASE_URL),
    'orcarouter_oauth': ('api', 'orcarouter_oauth', ORCAROUTER_BASE_URL),
    'codex_a': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_a_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_b_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_2': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_3': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_4': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
    'codex_c_5': ('chatgpt', 'openai', 'https://chatgpt.com/backend-api/codex'),
}
CHATGPT_PROFILES = tuple(k for k, v in PROFILES.items() if v[0] == 'chatgpt')
LEGACY_CAMPAIGN_PROFILES = ('codex_a', 'codex_b', 'codex_b_2', 'codex_b_3')
FIVE_B_PROFILES = ('codex_b', 'codex_b_2', 'codex_b_3', 'codex_b_4', 'codex_b_5')
ACCOUNT_GROUPS = {g: tuple('codex_'+g+(f'_{i}' if i > 1 else '') for i in range(1, 6))
                  for g in ('a', 'b', 'c')}
POOL15_PROFILES = tuple(n for names in ACCOUNT_GROUPS.values() for n in names)


def profile(name=None):
    name = name or os.environ.get('ROLLOUT_AUTH_PROFILE', 'galbot')
    if name not in PROFILES:
        raise ValueError('Unknown rollout auth profile; no automatic fallback')
    mode, provider, url = PROFILES[name]
    return dict(name=name, auth_mode=mode, provider=provider, base_url=url)


def shared_root(shared=None):
    return Path(shared or os.environ.get('ROLLOUT_SHARED_ROOT', DEFAULT_SHARED)).resolve()


def account_home(name, shared=None):
    p = profile(name)
    if p['auth_mode'] != 'chatgpt':
        raise ValueError('An API profile has no managed ChatGPT home')
    return shared_root(shared)/'private/auth_profiles'/name/'codex_home'


def credential_path(name, shared=None):
    if profile(name)['auth_mode'] == 'chatgpt':
        return account_home(name, shared)/'auth.json'
    return shared_root(shared)/'private'/f'{name}.key'


def config_text(name=None):
    p = profile(name)
    text = (f'model_provider = "{p["provider"]}"\nmodel = "gpt-6-astra"\n'
            'model_reasoning_effort = "xhigh"\nmodel_reasoning_summary = "auto"\n')
    if p['auth_mode'] == 'chatgpt':
        text += 'cli_auth_credentials_store = "file"\nforced_login_method = "chatgpt"\n'
    text += '\n[analytics]\nenabled = false\n'
    if p['auth_mode'] == 'api':
        text += (f'\n[model_providers.{p["provider"]}]\nname = "{p["provider"]}"\n'
                 f'base_url = "{p["base_url"]}"\nenv_key = "OPENAI_API_KEY"\nwire_api = "responses"\n')
    return text + '\n[features]\nfast_mode = false\ncodex_hooks = true\nhooks = true\n'


def private_dir(path):
    if path.is_symlink():
        raise ValueError('Private auth directories cannot be symlinks')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError(f'Private auth directory must be owned by this user and mode 0700: {path}')


def initialize(name, shared=None):
    p = profile(name)
    root = shared_root(shared)/'private/auth_profiles'
    private_dir(root)
    private_dir(root/name)
    home = root/name/'codex_home'
    private_dir(home)
    config = home/'config.toml'
    text = config_text(name)
    if config.exists():
        if config.is_symlink() or config.read_text() != text:
            raise ValueError('Existing profile config differs; refusing to overwrite it')
    else:
        fd = os.open(config, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
    return home


def validate_credential(name, shared=None, path=None):
    p = profile(name)
    path = Path(path) if path else credential_path(name, shared)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Credential missing or symlink; initialize/login for {name}')
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError('Credential must be user-owned and mode 0600 or stricter')
    if p['auth_mode'] == 'api':
        value = path.read_text().strip()
        if not value.startswith('sk-') or any(c.isspace() for c in value):
            raise ValueError('Invalid API credential format (value withheld)')
    else:
        try:
            data = json.loads(path.read_text())
        except ValueError:
            raise ValueError('Invalid managed auth JSON (contents withheld)') from None
        tokens = data.get('tokens') or {}
        if data.get('auth_mode') != 'chatgpt' or not all(
                tokens.get(k) for k in ('access_token', 'refresh_token', 'id_token')):
            raise ValueError('Use a fresh managed ChatGPT login, not an API key or external-token cache')
    return path


@contextmanager
def account_lock(name, shared=None):
    home = account_home(name, shared)
    private_dir(home.parent)
    fd = os.open(home.parent/'account.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'{name} is in use; do not log in or run another job with this account') from None
        yield fd
    finally:
        os.close(fd)


def clean_login_env(home, proxy_file=None):
    env = dict(os.environ, CODEX_HOME=str(home))
    for key in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'CODEX_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_PROJECT_ID'):
        env.pop(key, None)
    if proxy_file is not None:
        from ..proxy_config import read_proxy
        proxy = read_proxy(proxy_file)
        for key in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy'):
            env[key] = proxy
        for key in ('ALL_PROXY', 'all_proxy'):
            env.pop(key, None)
    return env


def validate_campaign_sessions(shared=None, names=LEGACY_CAMPAIGN_PROFILES):
    """Reject copied refresh sessions; never print IDs/tokens or rewrite them."""
    sessions, accounts = [], {}
    if not names or len(set(names)) != len(names) or any(n not in CHATGPT_PROFILES for n in names):
        raise ValueError('Select distinct managed session profiles explicitly')
    for name in names:
        path = validate_credential(name, shared)
        tokens = json.loads(path.read_text())['tokens']
        sessions.append(hashlib.sha256(tokens['refresh_token'].encode()).digest())
        accounts[name] = tokens.get('account_id')
    if len(set(sessions)) != len(sessions):
        raise ValueError('Concurrent slots require independent logins, not copied refresh tokens')
    if not all(accounts.values()):
        raise ValueError('Managed account identity unavailable; verify logins before dispatch')
    groups = [{value for name, value in accounts.items() if name == prefix or name.startswith(prefix+'_')}
              for prefix in ('codex_a', 'codex_b', 'codex_c')]
    if any(len(group) > 1 for group in groups):
        raise ValueError('Each account group must log into the same intended account')
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
        raise ValueError('Separate account groups must use different accounts')


def login_session(name, codex, shared=None, proxy_file=None, browser=False):
    home = initialize(name, shared)
    with account_lock(name, shared):
        if (home/'auth.json').exists():
            raise ValueError('Account already has credentials; refusing silent login replacement')
        os.umask(0o077)
        command = [str(Path(codex).resolve()), 'login']
        if not browser:
            command.append('--device-auth')
        result = subprocess.run(command, env=clean_login_env(home, proxy_file))
        if result.returncode:
            raise SystemExit(result.returncode)
        validate_credential(name, shared)
        print(f'{name}: isolated managed login saved; developer Codex unchanged.')


def login_group(group, codex, shared=None, proxy_file=None, browser=False):
    """Interactive, resumable onboarding; never replace existing credentials."""
    present = []
    for name in POOL15_PROFILES:
        if credential_path(name, shared).exists():
            validate_credential(name, shared)
            present.append(name)
    if present:
        validate_campaign_sessions(shared, tuple(present))
    for name in ACCOUNT_GROUPS[group]:
        if name in present:
            print(f'{name}: saved login retained (network validity not checked).')
            continue
        print(f'\nAuthorize {name} using account {group.upper()}. Do not paste login codes or tokens into chat.', flush=True)
        login_session(name, codex, shared, proxy_file, browser)
        present.append(name)
        validate_campaign_sessions(shared, tuple(present))
    print(f'Account {group.upper()}: five independent sessions verified locally; no rollout launched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-root', type=Path)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('init')
    sub.add_parser('status')
    sub.add_parser('check-pool15')
    p = sub.add_parser('login-group')
    p.add_argument('group', choices=ACCOUNT_GROUPS)
    p.add_argument('--codex', required=True, type=Path)
    p.add_argument('--browser', action='store_true')
    p.add_argument('--https-proxy-file', type=Path)
    p = sub.add_parser('login')
    p.add_argument('name', choices=CHATGPT_PROFILES)
    p.add_argument('--codex', required=True, type=Path)
    p.add_argument('--browser', action='store_true', help='Use browser callback instead of device code')
    p.add_argument('--https-proxy-file', type=Path,
                   help='Read a private authenticated proxy only into the login child environment')
    p = sub.add_parser('config')
    p.add_argument('--output', required=True, type=Path)
    # OrcaRouter is a first-class provider with two explicit authentication
    # choices. Both are reachable from here so neither is a hidden path:
    #   orcarouter-set-key   -> OrcaRouter - API   (paste an sk-orca-… key)
    #   orcarouter-connect   -> OrcaRouter - Auth  (OAuth 2.0 + PKCE sign-in)
    sub.add_parser('orcarouter-status', help='Show the OrcaRouter credential status')
    p = sub.add_parser('orcarouter-set-key', help='OrcaRouter - API: store an sk-orca-… key')
    p.add_argument('--key-file', type=Path,
                   help='Read the key from a private file instead of the ORCA_KEY environment')
    p = sub.add_parser('orcarouter-clear', help='Remove a stored OrcaRouter credential')
    p.add_argument('--auth', choices=('api', 'oauth'), required=True)
    p = sub.add_parser('orcarouter-connect', help='OrcaRouter - Auth: sign in with PKCE')
    p.add_argument('--flow', choices=('loopback', 'oob', 'device'), default='oob',
                   help='loopback receives the redirect; oob prints a code to paste; '
                        'device polls (optional extra, never a substitute for PKCE)')
    p = sub.add_parser('orcarouter-models', help='List the live OrcaRouter model catalog')
    p.add_argument('--capability', default='chat',
                   choices=('chat', 'embedding', 'image', 'video', 'rerank'))
    p.add_argument('--modality', choices=('image', 'audio', 'video'),
                   help='Require this input modality (multimodal entry points)')
    p = sub.add_parser('orcarouter-console', help='Serve the local OrcaRouter provider console')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8770)
    p = sub.add_parser('lease')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'init':
        for name in PROFILES:
            initialize(name, args.shared_root)
        print('Isolated profiles initialized; existing credentials were not overwritten.')
    elif args.action == 'status':
        for name in PROFILES:
            try:
                validate_credential(name, args.shared_root)
                state = 'credential_present_not_network_verified'
            except ValueError as error:
                state = str(error)
            print(json.dumps(dict(profile(name), credential_file=str(credential_path(name, args.shared_root)), status=state)))
    elif args.action == 'config':
        # Runtime-only destination, created fresh; never overwrite profile/global config.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(config_text())
    elif args.action == 'lease':
        name = profile()['name']
        validate_credential(name, args.shared_root)
        with account_lock(name, args.shared_root) as fd:
            os.set_inheritable(fd, True)
            env = dict(os.environ, ROLLOUT_ACCOUNT_LOCK_FD=str(fd))
            command = args.command
            if command and command[0] == '--':
                command = command[1:]
            if not command:
                raise ValueError('A leased command is required')
            os.execvpe(command[0], command, env)
    elif args.action == 'login-group':
        login_group(args.group, args.codex, args.shared_root, args.https_proxy_file, args.browser)
    elif args.action == 'check-pool15':
        validate_campaign_sessions(args.shared_root, POOL15_PROFILES)
        print('Three different accounts; five independent sessions each. Local check only; no rollout launched.')
    elif args.action.startswith('orcarouter-'):
        orcarouter_command(args)
    else:
        login_session(args.name, args.codex, args.shared_root, args.https_proxy_file, args.browser)


def orcarouter_command(args):
    """OrcaRouter CLI entry points; no credential value is ever printed."""
    from .orcarouter import catalog, console, origins, provider
    from .orcarouter.credentials import CredentialError, CredentialStore
    store = CredentialStore(shared=args.shared_root)
    if args.action == 'orcarouter-status':
        print(json.dumps(dict(
            inference_base=origins.api_v1(), auth_base=origins.auth_base(),
            providers=[provider.describe(provider.API_PROVIDER, store),
                       provider.describe(provider.OAUTH_PROVIDER, store)],
            credential_value_exposed=False), indent=1))
    elif args.action == 'orcarouter-set-key':
        if args.key_file:
            path = Path(args.key_file)
            if path.is_symlink() or not path.is_file():
                raise ValueError('Key file missing or symlink')
            if path.stat().st_mode & 0o077:
                raise ValueError('Key file must be mode 0600 or stricter')
            key = path.read_text().strip()
        else:
            key = os.environ.get('ORCA_KEY', '')
        credential = provider.adapter_for(provider.API_PROVIDER, store).acquire(key)
        print(f'OrcaRouter - API: stored {credential.masked}; manage keys at '
              f'{provider.KEY_DASHBOARD_URL}')
    elif args.action == 'orcarouter-clear':
        source = provider.provider_source(
            provider.API_PROVIDER if args.auth == 'api' else provider.OAUTH_PROVIDER)
        removed = store.clear(source)
        print(f'OrcaRouter - {args.auth}: ' + ('removed the stored credential.'
                                               if removed else 'nothing was stored.'))
    elif args.action == 'orcarouter-connect':
        provider.adapter_for(provider.OAUTH_PROVIDER, store).acquire(flow=args.flow)
        print('OrcaRouter - Auth: signed in; the key is stored and reused until revoked.')
    elif args.action == 'orcarouter-models':
        required = (args.modality,) if args.modality else ()
        credential = store.load_any()
        if credential is None:
            raise CredentialError('No OrcaRouter credential is configured; run '
                                  'orcarouter-set-key or orcarouter-connect first')
        result = catalog.catalog(credential.reveal(), args.capability, required)
        print(json.dumps(dict(capability=args.capability,
                              required_modalities=list(required),
                              source=result['source'], degraded=result['degraded'],
                              catalog_url=catalog.discovery_url(args.capability),
                              count=len(result['models']),
                              models=[model['id'] for model in result['models']],
                              reason=result.get('reason')), indent=1))
    elif args.action == 'orcarouter-console':
        server = console.serve(store, args.host, args.port)
        print(json.dumps(dict(event='orcarouter_console',
                              url=f'http://{args.host}:{server.server_port}/',
                              inference_base=origins.api_v1(),
                              credential_value_exposed=False)))
        server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
