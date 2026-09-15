"""First-class OrcaRouter provider entries for the rollout backend.

OrcaRouter appears in the profile registry as two explicit authentication
choices that share one inference identity:

======  ==========================  ==========================================
ID      Label                       Credential source
======  ==========================  ==========================================
``orcarouter``        ``OrcaRouter - API``    user pastes an ``sk-orca-…`` key
``orcarouter_oauth``  ``OrcaRouter - Auth``   PKCE sign-in issues an ``sk-orca-…`` key
======  ==========================  ==========================================

Keeping them separate matters for support, logout and reauthentication: a single
button that sometimes asks for a key and sometimes opens a browser cannot report
which credential is broken. Both entries resolve to the same ``base_url``,
``env_key``, ``wire_api`` and model namespace, so nothing downstream needs to
know which one the user picked.
"""
from __future__ import annotations

from . import catalog, origins
from .credentials import Credential, CredentialError, CredentialStore

API_PROVIDER = 'orcarouter'
OAUTH_PROVIDER = 'orcarouter_oauth'
API_LABEL = 'OrcaRouter - API'
OAUTH_LABEL = 'OrcaRouter - Auth'
DEFAULT_MODEL = 'orcarouter/auto'
ENV_KEY = 'OPENAI_API_KEY'
WIRE_API = 'responses'
KEY_DASHBOARD_URL = 'https://www.orcarouter.ai/console'
REVOKE_URL = 'https://www.orcarouter.ai/console/authorized-apps'

PROVIDER_SOURCES = {API_PROVIDER: 'api_key', OAUTH_PROVIDER: 'oauth_pkce'}
SOURCE_PROVIDERS = {source: name for name, source in PROVIDER_SOURCES.items()}
AUTH_CHOICES = ((API_PROVIDER, API_LABEL, 'api_key'), (OAUTH_PROVIDER, OAUTH_LABEL, 'oauth_pkce'))


class ProviderError(ValueError):
    """A provider id or credential is unusable; messages never carry a secret."""


def is_orcarouter(provider_name):
    return provider_name in PROVIDER_SOURCES


def provider_source(provider_name):
    try:
        return PROVIDER_SOURCES[provider_name]
    except KeyError:
        raise ProviderError(
            f'Unknown OrcaRouter authentication choice: {provider_name!r}') from None


def provider_name(source):
    try:
        return SOURCE_PROVIDERS[source]
    except KeyError:
        raise ProviderError(f'Unknown credential source: {source!r}') from None


def label(provider_name):
    return OAUTH_LABEL if provider_name == OAUTH_PROVIDER else API_LABEL


def provider_block(provider_name, env=None):
    """The ``[model_providers.<name>]`` body for a Codex config file.

    Both authentication choices emit the same block: the credential is what
    differs, not the routing.
    """
    is_orcarouter(provider_name) or _unknown(provider_name)
    return dict(name='OrcaRouter', base_url=origins.api_v1(env),
                env_key=ENV_KEY, wire_api=WIRE_API)


def _unknown(provider_name):
    raise ProviderError(f'Not an OrcaRouter provider: {provider_name!r}')


def config_lines(provider_name, env=None):
    block = provider_block(provider_name, env)
    return [f'[model_providers.{provider_name}]',
            'name = "OrcaRouter"',
            f'base_url = "{block["base_url"]}"',
            f'env_key = "{block["env_key"]}"',
            f'wire_api = "{block["wire_api"]}"']


def load_credential(provider_name, store=None):
    """Read the credential belonging to one authentication choice.

    Returns ``None`` when nothing is stored, so a caller can distinguish
    "not configured yet" from "configured but rejected".
    """
    store = store or CredentialStore()
    return store.load(provider_source(provider_name))


def describe(provider_name, store=None):
    """Non-secret status for the provider console and ``status`` output."""
    store = store or CredentialStore()
    source = provider_source(provider_name)
    credential = store.load(source)
    return dict(
        id=provider_name, label=label(provider_name), source=source,
        base_url=origins.api_v1(),
        configured=credential is not None,
        secret_masked=credential.masked if credential else None,
        account=credential.account_id if credential else None,
        scope=credential.scope if credential else None,
        generation=credential.generation if credential else None,
        needs_reauth=store.needs_reauth(source),
        key_dashboard_url=KEY_DASHBOARD_URL,
    )


def adapter_for(provider_name, store=None):
    """Return the credential adapter for an authentication choice."""
    source = provider_source(provider_name)
    if source == 'api_key':
        return ApiKeyAdapter(store)
    return PkceAdapter(store, provider_name)


class CredentialAdapter:
    """The small seam both authentication choices implement.

    :meth:`acquire` is the only operation the rest of the program calls, and it
    always yields the same :class:`~.credentials.Credential`. Provider requests,
    model discovery and every inference entry point therefore stay independent of
    how the key was obtained.
    """

    source = None
    provider = None

    def __init__(self, store=None):
        self.store = store or CredentialStore()

    def acquire(self, **kwargs):  # pragma: no cover - interface definition
        raise NotImplementedError

    def current(self):
        return self.store.load(self.source)

    def clear(self):
        return self.store.clear(self.source)

    def handle_unauthorized(self, credential):
        """Terminal classification for a relay ``401``. Never refreshes."""
        from .credentials import terminal_401
        return terminal_401(credential, self.store)


class ApiKeyAdapter(CredentialAdapter):
    """Existing-key users: paste an ``sk-orca-…`` credential."""

    source = 'api_key'
    provider = API_PROVIDER

    def acquire(self, key, account_id='local', scope='api'):
        # Catch obvious paste mistakes at entry rather than storing them.
        text = self.validate(key)
        credential = Credential(text, 'api_key', account_id, scope, 0)
        return self.store.save(credential)

    def validate(self, key):
        from .credentials import validate_key_text
        return validate_key_text(key)


class PkceAdapter(CredentialAdapter):
    """Account users: sign in and receive a durable OrcaRouter key.

    The key is stored exactly where the project already keeps rollout secrets and
    is reused on later runs. OrcaRouter keys are durable, not refreshable, so
    there is no refresh grant here and re-authorizing on every launch is wrong —
    the consent endpoint allows ten issued keys per user per 24 hours.
    """

    source = 'oauth_pkce'
    provider = OAUTH_PROVIDER

    def __init__(self, store=None, provider_name=OAUTH_PROVIDER, auth_base=None, env=None):
        super().__init__(store)
        self.provider = provider_name
        self.auth_base = auth_base or origins.auth_base(env)

    def acquire_loopback(self, **kwargs):
        from .pkce import connect_loopback
        return connect_loopback(self.auth_base, self.store, **kwargs)

    def acquire_oob(self, **kwargs):
        from .pkce import connect_oob
        return connect_oob(self.auth_base, self.store, **kwargs)

    def acquire_device(self, app_name='GPT as Policy rollout', **kwargs):
        """Optional Flow C. It never replaces PKCE for the shipped surfaces."""
        from .pkce import device_poll, device_start
        start = device_start(self.auth_base, app_name)
        credential = device_poll(self.auth_base, start['device_code'], **kwargs)
        return self.store.save(credential)

    def acquire(self, flow='loopback', **kwargs):
        if flow == 'loopback':
            return self.acquire_loopback(**kwargs)
        if flow == 'oob':
            return self.acquire_oob(**kwargs)
        if flow == 'device':
            return self.acquire_device(**kwargs)
        raise ProviderError(f'Unknown authorization flow: {flow!r}')


def discover_models(provider_name, store=None, capability='chat', required_modalities=(),
                    timeout=None, env=None):
    """Run model discovery through whichever credential the user configured."""
    credential = load_credential(provider_name, store)
    if credential is None:
        raise CredentialError(
            f'{label(provider_name)} has no stored credential; connect it first')
    if store is not None and store.needs_reauth(credential.source, credential.account_id):
        raise CredentialError(
            f'{label(provider_name)} needs reauthentication; sign in again to issue a new key')
    kwargs = {} if timeout is None else {'timeout': timeout}
    return catalog.catalog(credential.reveal(), capability, required_modalities, env=env, **kwargs)
