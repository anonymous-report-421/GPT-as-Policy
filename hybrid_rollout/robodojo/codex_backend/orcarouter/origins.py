"""Resolve OrcaRouter origins without ever deriving one from the other.

Authentication lives on the web origin and inference lives on the relay origin.
They are separate public hosts and neither may be produced by rewriting the
other's hostname or by appending a path to it. Each is resolved independently
from its own override, then from the shared self-hosted base, then from the
documented public default.
"""
from __future__ import annotations

import os
import urllib.parse

DEFAULT_AUTH_BASE = 'https://www.orcarouter.ai'
DEFAULT_API_BASE = 'https://api.orcarouter.ai'

# Fixed protocol paths. The relay serves inference under /v1; authentication is
# a web-origin API under /api/v1/auth. These are not interchangeable.
AUTHORIZE_PATH = '/auth'
EXCHANGE_PATH = '/api/v1/auth/keys'
DEVICE_CODE_PATH = '/api/v1/auth/device/code'
DEVICE_TOKEN_PATH = '/api/v1/auth/device/token'
API_VERSION_PREFIX = '/v1'
MODELS_PATH = API_VERSION_PREFIX + '/models'
CHAT_COMPLETIONS_PATH = API_VERSION_PREFIX + '/chat/completions'

LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')


class OriginError(ValueError):
    """A configured origin is unusable; the message never contains the value."""


def _clean(base, label, env):
    try:
        parsed = urllib.parse.urlsplit(base)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError
        # Credentials in the URL would leak into logs and proxy history.
        if parsed.username or parsed.password:
            raise ValueError
        # An origin with a path silently double-prefixes the fixed protocol
        # paths below (e.g. ``/v1`` + ``/v1/models``), so it is refused here
        # rather than tolerated and mis-routed later.
        if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
            raise ValueError
        if any(character.isspace() for character in base):
            raise ValueError
        if parsed.scheme == 'http' and parsed.hostname not in LOOPBACK_HOSTS:
            # Plaintext transport to a remote host would expose the exchanged
            # API key. Loopback stays usable for local self-hosted development.
            raise ValueError
    except ValueError:
        raise OriginError(f'{label} is not a usable origin (value withheld)') from None
    port = f':{parsed.port}' if parsed.port else ''
    return f'{parsed.scheme}://{parsed.hostname}{port}'


def _lookup(env, *names):
    for name in names:
        value = (env or os.environ).get(name)
        if value:
            return value.strip(), name
    return None, None


def auth_base(env=None):
    """Return the web origin that serves the consent screen and code exchange."""
    value, name = _lookup(env, 'ORCA_AUTH_BASE_URL')
    if value is None:
        value, name = _lookup(env, 'ORCA_BASE_URL')
        label = 'ORCA_BASE_URL (auth)'
    else:
        label = 'ORCA_AUTH_BASE_URL'
    if value is None:
        return DEFAULT_AUTH_BASE
    return _clean(value, label, env)


def api_base(env=None):
    """Return the relay origin that serves inference and model discovery."""
    value, name = _lookup(env, 'ORCA_API_BASE_URL')
    if value is None:
        value, name = _lookup(env, 'ORCA_BASE_URL')
        label = 'ORCA_BASE_URL (api)'
    else:
        label = 'ORCA_API_BASE_URL'
    if value is None:
        return DEFAULT_API_BASE
    return _clean(value, label, env)


def api_v1(env=None):
    """Return the OpenAI-compatible inference base, including the version prefix."""
    return api_base(env) + API_VERSION_PREFIX


def url(origin, path, query=None):
    """Join an origin with one of the fixed protocol paths above."""
    target = origin.rstrip('/') + path
    if query:
        target = target + '?' + urllib.parse.urlencode(query)
    return target
