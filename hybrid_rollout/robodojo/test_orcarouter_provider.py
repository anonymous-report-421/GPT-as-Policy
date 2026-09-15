"""Offline OrcaRouter provider tests: fake servers only, no real account calls.

Covers the credential seam (both adapters), PKCE verifier/challenge/state and
both flows end to end through the real adapter, the origin split, terminal
``401`` handling, login-cancel release, and model discovery with per-capability
filtering. Live tests run only when ``ORCAROUTER_API_KEY`` is present.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .codex_backend.orcarouter import catalog as catalog_module
from .codex_backend.orcarouter import origins, pkce, provider
from .codex_backend.orcarouter.credentials import (Credential, CredentialError, CredentialStore,
    looks_like_key, mask, terminal_401, validate_key_text)
from .codex_backend.profiles import initialize, private_dir

FAKE_KEY = 'sk-orca-' + 'A' * 32
SECOND_KEY = 'sk-orca-' + 'B' * 32


# --------------------------------------------------------------------------
# Fake authorization / API servers
# --------------------------------------------------------------------------

def make_handler(routes, records):
    """Build a request handler bound to one server's routes and record list.

    Routing state is per instance rather than per class, so two fake origins can
    be alive at once without one clobbering the other's routes.
    """

    class Recorder(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _store(self):
            length = int(self.headers.get('Content-Length') or 0)
            raw = self.rfile.read(length) if length else b''
            parsed = urllib.parse.urlsplit(self.path)
            records.append(dict(
                method=self.command, path=parsed.path,
                query=urllib.parse.parse_qs(parsed.query), raw_query=parsed.query,
                body=raw.decode('utf-8', 'replace'), headers=dict(self.headers)))
            return parsed, raw

        def _respond(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            parsed, raw = self._store()
            handler = routes.get(('POST', parsed.path))
            if handler is None:
                self._respond(404, dict(error='not_found'))
                return
            status, payload = handler(json.loads(raw or b'{}') if raw else {})
            self._respond(status, payload)

        def do_GET(self):
            parsed, _ = self._store()
            handler = routes.get(('GET', parsed.path))
            if handler is None:
                self._respond(404, dict(error='not_found'))
                return
            status, payload = handler(urllib.parse.parse_qs(parsed.query))
            self._respond(status, payload)

    return Recorder


class FakeServer:
    """A loopback HTTP origin for tests; never contacted by production code."""

    def __init__(self, routes):
        self.records = []
        self._routes = dict(routes)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self._routes, self.records))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={'poll_interval': 0.05}, daemon=True)
        self.thread.start()

    @property
    def base(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    def clear(self):
        self.records.clear()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def exchange_routes(scope='api', status=200, key=FAKE_KEY, user_id='4242'):
    def exchange(payload):
        if status != 200:
            return status, dict(error='invalid_grant', error_description='denied')
        return 200, dict(key=key, user_id=user_id, scope=scope)
    return {('POST', origins.EXCHANGE_PATH): exchange}


def model_record(identifier, **extra):
    record = dict(id=identifier, object='model', owned_by='test',
                  supported_endpoint_types=['openai', 'openai-response'])
    record.update(extra)
    return record


CHAT_ONLY = model_record('acme/text-only', architecture=dict(input_modalities=['text'],
                                                             output_modalities=['text']))
CHAT_VISION = model_record('acme/vision-chat', context_length=200000,
                           architecture=dict(input_modalities=['text', 'image'],
                                             output_modalities=['text']))
EMBEDDING = model_record('acme/embed-small', supported_endpoint_types=['embeddings'])
IMAGE_GEN = model_record('acme/draw', supported_endpoint_types=['image-generation'])
VIDEO_GEN = model_record('acme/movie', supported_endpoint_types=['openai-video'])
RERANK = model_record('acme/rank', supported_endpoint_types=['jina-rerank'])


def catalog_fixture():
    return dict(data=[CHAT_ONLY, CHAT_VISION, EMBEDDING, IMAGE_GEN, VIDEO_GEN, RERANK])


@pytest.fixture
def store(tmp_path):
    directory = tmp_path/'private'
    private_dir(directory)
    return CredentialStore(directory=directory)


@pytest.fixture(autouse=True)
def clean_origin_env(monkeypatch):
    for name in ('ORCA_BASE_URL', 'ORCA_AUTH_BASE_URL', 'ORCA_API_BASE_URL'):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# Provider registration
# --------------------------------------------------------------------------

def test_orcarouter_is_a_first_class_provider_with_two_named_choices():
    assert provider.is_orcarouter(provider.API_PROVIDER)
    assert provider.is_orcarouter(provider.OAUTH_PROVIDER)
    assert provider.API_LABEL == 'OrcaRouter - API'
    assert provider.OAUTH_LABEL == 'OrcaRouter - Auth'
    assert [row[0] for row in provider.AUTH_CHOICES] == [provider.API_PROVIDER,
                                                         provider.OAUTH_PROVIDER]
    # Separate ids, but one inference identity.
    api = provider.provider_block(provider.API_PROVIDER)
    oauth = provider.provider_block(provider.OAUTH_PROVIDER)
    assert api == oauth
    assert api['base_url'] == 'https://api.orcarouter.ai/v1'
    assert api['env_key'] == 'OPENAI_API_KEY' and api['wire_api'] == 'responses'
    assert provider.config_lines(provider.API_PROVIDER)[0] == '[model_providers.orcarouter]'
    assert provider.config_lines(provider.OAUTH_PROVIDER)[0] == '[model_providers.orcarouter_oauth]'


def test_unknown_orcarouter_provider_fails_closed():
    with pytest.raises(provider.ProviderError):
        provider.provider_block('orcarouter_typo')
    with pytest.raises(provider.ProviderError):
        provider.provider_source('openrouter')


def test_both_adapters_produce_the_same_credential_type(store, monkeypatch):
    """The downstream contract is one Credential, whatever the acquisition path."""
    monkeypatch.setenv('ORCA_AUTH_BASE_URL', 'https://www.orcarouter.ai')
    key_credential = provider.adapter_for(provider.API_PROVIDER, store).acquire(FAKE_KEY)
    assert isinstance(key_credential, Credential) and key_credential.source == 'api_key'

    with FakeServer(exchange_routes()) as server:
        oauth = provider.adapter_for(provider.OAUTH_PROVIDER, store,
                                     ).__class__(store, provider.OAUTH_PROVIDER, server.base)
        attempt = pkce.PkceAttempt()

        def flow(_prompt):
            return 'code-from-browser'
        credential = pkce.connect_oob(server.base, store, read_code=flow)
    assert isinstance(credential, Credential) and credential.source == 'oauth_pkce'

    # Nothing that consumes a credential inspects where it came from.
    for credential in (key_credential, credential):
        assert credential.reveal().startswith('sk-orca-')
        assert 'source' not in json.dumps(dict(model='orcarouter/auto',
                                               base_url=provider.provider_block('orcarouter')['base_url'],
                                               api_key=credential.reveal()), default=str)


def test_describe_reports_both_choices_without_exposing_secrets(store):
    store.save(Credential(FAKE_KEY, 'api_key'))
    described = provider.describe(provider.API_PROVIDER, store)
    assert described['configured'] is True
    assert described['secret_masked'] == mask(FAKE_KEY)
    assert FAKE_KEY not in json.dumps(described)
    other = provider.describe(provider.OAUTH_PROVIDER, store)
    assert other['configured'] is False and other['secret_masked'] is None
    assert described['label'] != other['label']


# --------------------------------------------------------------------------
# API key adapter
# --------------------------------------------------------------------------

def test_api_key_save_load_clear_and_mask(store):
    credential = provider.adapter_for(provider.API_PROVIDER, store).acquire(FAKE_KEY)
    assert credential.generation == 1
    assert store.load('api_key').reveal() == FAKE_KEY
    path = store.path_for('api_key')
    assert path.stat().st_mode & 0o077 == 0
    assert mask(FAKE_KEY) == FAKE_KEY[:8] + '…' + FAKE_KEY[-4:]
    assert FAKE_KEY not in mask(FAKE_KEY)
    assert store.clear('api_key') == ['api_key']
    assert store.load('api_key') is None
    assert store.clear('api_key') == []


def test_api_key_rotates_generation_and_clear_is_per_source(store):
    adapter = provider.adapter_for(provider.API_PROVIDER, store)
    adapter.acquire(FAKE_KEY)
    second = adapter.acquire(SECOND_KEY)
    assert second.generation == 2
    provider.adapter_for(provider.OAUTH_PROVIDER, store)
    store.save(Credential(SECOND_KEY, 'oauth_pkce'))
    store.clear('api_key')
    assert store.load('api_key') is None
    assert store.load('oauth_pkce') is not None


def test_invalid_api_key_shapes_are_rejected_without_echoing_them(store):
    adapter = provider.adapter_for(provider.API_PROVIDER, store)
    for bad in ('', '   ', 'sk-openai-abcdefgh', 'sk-orca-short', 'sk-orca-aaaa bbbb cccc dddd'):
        with pytest.raises(CredentialError) as error:
            adapter.acquire(bad)
        assert bad.strip() not in str(error.value) or not bad.strip()
    assert looks_like_key(FAKE_KEY) and not looks_like_key('sk-orca-')
    with pytest.raises(CredentialError):
        validate_key_text('x' * 5000)


def test_stored_credential_rejects_symlink_and_loose_permissions(store, tmp_path):
    store.save(Credential(FAKE_KEY, 'api_key'))
    path = store.path_for('api_key')
    path.chmod(0o644)
    with pytest.raises(CredentialError, match='0600'):
        store.load('api_key')
    path.chmod(0o600)
    target = tmp_path/'elsewhere.key'
    target.write_text(FAKE_KEY)
    path.unlink()
    path.symlink_to(target)
    assert store.load('api_key') is None


# --------------------------------------------------------------------------
# PKCE mechanics
# --------------------------------------------------------------------------

def test_verifier_challenge_and_state_are_fresh_and_s256():
    first, second = pkce.PkceAttempt(), pkce.PkceAttempt()
    assert first.verifier != second.verifier and first.state != second.state
    for attempt in (first, second):
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(attempt.verifier.encode('ascii')).digest()).decode().rstrip('=')
        assert attempt.challenge == expected
        assert '=' not in attempt.challenge
        assert len(attempt.verifier) == 43
    assert first.challenge != pkce.challenge_for('plain')


def test_verifier_never_appears_in_repr_str_or_authorize_url():
    attempt = pkce.PkceAttempt()
    assert attempt.verifier not in repr(attempt)
    assert attempt.verifier not in str(attempt)
    target = pkce.authorize_url('https://www.orcarouter.ai', attempt, 'http://127.0.0.1:1/cb')
    assert attempt.verifier not in target
    assert 'code_challenge_method=S256' in target
    assert 'code_challenge=' + attempt.challenge in urllib.parse.unquote(target)
    assert 'callback_url=http%3A%2F%2F127.0.0.1%3A1%2Fcb' in target
    assert target.startswith('https://www.orcarouter.ai/auth?')


def test_authorize_url_uses_the_auth_origin_only():
    attempt = pkce.PkceAttempt()
    target = pkce.authorize_url(origins.auth_base(), attempt, 'oob')
    parsed = urllib.parse.urlsplit(target)
    assert parsed.netloc == 'www.orcarouter.ai' and parsed.path == '/auth'
    assert 'api.orcarouter.ai' not in target


def test_exchange_posts_to_the_auth_origin_with_the_documented_body():
    with FakeServer(exchange_routes()) as server:
        credential = pkce.exchange_code(server.base, 'auth-code', 'the-verifier')
        assert credential.reveal() == FAKE_KEY
        (record,) = server.records
        assert record['method'] == 'POST'
        assert record['path'] == '/api/v1/auth/keys'
        body = json.loads(record['body'])
        assert body['code'] == 'auth-code'
        assert body['code_verifier'] == 'the-verifier'
        assert body['code_challenge_method'] == 'S256'
        # The verifier is in the body, never in the URL an intermediary logs.
        assert 'the-verifier' not in record['raw_query']
        assert record['query'] == {}


def test_exchange_never_targets_the_relay_versioned_auth_path():
    with FakeServer(exchange_routes()) as server:
        pkce.exchange_code(server.base, 'auth-code', 'verifier')
        assert [record['path'] for record in server.records] == ['/api/v1/auth/keys']
        assert not any(record['path'].startswith('/v1/auth') for record in server.records)


# --------------------------------------------------------------------------
# Flow A and Flow B end to end through the real adapter
# --------------------------------------------------------------------------

def test_flow_a_loopback_end_to_end_persists_through_adapter(store):
    with FakeServer(exchange_routes()) as server:
        adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, server.base)
        seen = {}

        def deliver(target):
            """Stand in for the browser: hit the loopback callback with the code."""
            seen['url'] = target
            parsed = urllib.parse.urlsplit(target)
            query = urllib.parse.parse_qs(parsed.query)
            callback = query['callback_url'][0]
            state = query['state'][0]
            response = urllib.request.urlopen(
                callback + '?code=one-time-code&state=' + urllib.parse.quote(state), timeout=10)
            assert response.status == 200
            seen['page'] = response.read().decode()

        credential = adapter.acquire_loopback(open_browser=deliver)
        assert credential.source == 'oauth_pkce'
        assert credential.reveal() == FAKE_KEY
        assert credential.account_id == '4242'
        assert 'close this tab' in seen['page']
        assert urllib.parse.urlsplit(seen['url']).netloc == urllib.parse.urlsplit(server.base).netloc
        # Persisted in the project's existing private store, mode 0600.
        stored = store.path_for('oauth_pkce')
        assert stored.read_text().strip() == FAKE_KEY
        assert stored.stat().st_mode & 0o077 == 0
        # Re-running reuses the stored key rather than minting another one.
        assert store.load('oauth_pkce').reveal() == FAKE_KEY


def test_flow_a_state_mismatch_is_rejected_and_stores_nothing(store):
    with FakeServer(exchange_routes()) as server:
        adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, server.base)

        def attacker(target):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
            urllib.request.urlopen(query['callback_url'][0] + '?code=stolen&state=wrong',
                                   timeout=10).read()

        with pytest.raises(pkce.PkceError) as error:
            adapter.acquire_loopback(open_browser=attacker)
        assert error.value.category == 'state'
        assert store.load('oauth_pkce') is None
        assert not any(record['path'] == '/api/v1/auth/keys' for record in server.records)


def test_flow_a_denial_is_reported_and_stores_nothing(store):
    with FakeServer(exchange_routes()) as server:
        adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, server.base)

        def deny(target):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)
            urllib.request.urlopen(
                query['callback_url'][0] + '?error=access_denied&state='
                + urllib.parse.quote(query['state'][0]), timeout=10).read()

        with pytest.raises(pkce.PkceError) as error:
            adapter.acquire_loopback(open_browser=deny)
        assert error.value.category == 'denied'
        assert store.load('oauth_pkce') is None


def test_flow_b_out_of_band_end_to_end(store):
    with FakeServer(exchange_routes()) as server:
        adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, server.base)
        seen = {}
        credential = adapter.acquire_oob(on_url=lambda url: seen.update(url=url),
                                        read_code=lambda prompt: 'pasted-code')
        assert credential.reveal() == FAKE_KEY
        assert 'callback_url=oob' in seen['url']
        # Flow B mandates S256: the code is handled by a person.
        assert 'code_challenge_method=S256' in seen['url']
        (record,) = [r for r in server.records if r['path'] == '/api/v1/auth/keys']
        assert json.loads(record['body'])['code'] == 'pasted-code'


def test_flow_b_empty_code_cancels_without_exchange(store):
    with FakeServer(exchange_routes()) as server:
        adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, server.base)
        with pytest.raises(pkce.PkceCancelled):
            adapter.acquire_oob(read_code=lambda prompt: '   ')
        assert store.load('oauth_pkce') is None
        assert server.records == []


def test_oauth_login_lock_is_released_on_cancel_and_a_second_login_can_start(store):
    """The back-forward-cache shape: cancel must free the listener, no remount."""
    adapter = provider.PkceAdapter(store, provider.OAUTH_PROVIDER, 'https://www.orcarouter.ai')
    session = pkce.PkceSession('https://www.orcarouter.ai')
    target = session.begin()
    assert session.port > 0
    port = session.port
    assert session.wait_for_code.__self__ is session
    session.cancel()
    assert session.cancelled is True
    with pytest.raises(pkce.PkceCancelled):
        session.wait_for_code()
    # The port is free again, so a second attempt starts without remounting.
    second = pkce.PkceSession('https://www.orcarouter.ai')
    assert second.begin().startswith('https://www.orcarouter.ai/auth?')
    assert second.attempt.verifier != session.attempt.verifier
    assert second.id != session.id
    second.cancel()


def test_timeout_ends_waiting_without_hanging(store):
    session = pkce.PkceSession('https://www.orcarouter.ai')
    session.begin()
    try:
        with pytest.raises(pkce.PkceError) as error:
            session._receiver.wait(timeout=0.3)
        assert error.value.category == 'timeout'
    finally:
        session.cancel()


def test_cancelled_session_never_publishes_a_late_credential(store):
    session = pkce.PkceSession('https://www.orcarouter.ai')
    session.begin()
    session.cancel()
    with pytest.raises(pkce.PkceCancelled):
        session.finish(Credential(FAKE_KEY, 'oauth_pkce'))
    assert store.load('oauth_pkce') is None


# --------------------------------------------------------------------------
# Error semantics
# --------------------------------------------------------------------------

@pytest.mark.parametrize('status,category', [
    (400, 'protocol'), (403, 'rejected'), (429, 'rate_limited'), (500, 'server'),
])
def test_exchange_error_statuses_map_to_actionable_categories(status, category):
    with FakeServer(exchange_routes(status=status)) as server:
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'code', 'verifier')
        assert error.value.category == category
        assert 'verifier' not in str(error.value)
        assert 'sk-orca' not in str(error.value)


def test_reused_or_expired_code_is_reported_as_rejected():
    calls = {'count': 0}

    def once(payload):
        calls['count'] += 1
        if calls['count'] > 1:
            return 403, dict(error='invalid_grant')
        return 200, dict(key=FAKE_KEY, user_id='1', scope='api')

    with FakeServer({('POST', origins.EXCHANGE_PATH): once}) as server:
        assert pkce.exchange_code(server.base, 'code', 'verifier').reveal() == FAKE_KEY
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'code', 'verifier')
        assert error.value.category == 'rejected'


def test_scope_downgrade_is_refused_not_assumed(store):
    """A narrower grant than requested must be reported, never silently accepted."""
    with FakeServer(exchange_routes(scope='api')) as server:
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'code', 'verifier', requested_scope_='connector')
        assert error.value.category == 'scope'
        assert 'connector' in str(error.value)
        # A refused grant must not be persisted either.
        assert store.load('oauth_pkce') is None


def test_missing_scope_in_response_is_refused():
    with FakeServer({('POST', origins.EXCHANGE_PATH):
                     lambda payload: (200, dict(key=FAKE_KEY, user_id='1'))}) as server:
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'code', 'verifier')
        assert error.value.category == 'scope'


def test_malformed_key_in_response_is_refused_without_echoing_it():
    with FakeServer({('POST', origins.EXCHANGE_PATH):
                     lambda payload: (200, dict(key='not-a-key', user_id='1', scope='api'))}) as server:
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'code', 'verifier')
        assert error.value.category == 'protocol'
        assert 'not-a-key' not in str(error.value)


def test_network_failure_is_reported_without_credential_material():
    with pytest.raises(pkce.PkceError) as error:
        pkce.exchange_code('http://127.0.0.1:1', 'code', 'verifier', timeout=2)
    assert error.value.category == 'network' and error.value.retryable is True
    assert 'verifier' not in str(error.value)


def test_unsupported_scope_is_refused_before_any_request():
    with pytest.raises(pkce.PkceError, match='Unsupported scope'):
        pkce.PkceSession('https://www.orcarouter.ai', scope='admin')


def test_device_grant_branches_on_error_only():
    responses = [dict(error='authorization_pending'), dict(error='slow_down'),
                 dict(key=FAKE_KEY, user_id='1', scope='api')]
    steps = []

    def token(payload):
        steps.append(payload['grant_type'])
        return 200, responses.pop(0)

    with FakeServer({('POST', origins.DEVICE_TOKEN_PATH): token}) as server:
        credential = pkce.device_poll(server.base, 'device', interval=0.01, sleep=lambda s: None)
    assert credential.reveal() == FAKE_KEY
    assert steps == [pkce.DEVICE_GRANT_TYPE] * 3

    for error_name, category in (('access_denied', 'denied'), ('expired_token', 'timeout'),
                                 ('unsupported_grant_type', 'protocol')):
        with FakeServer({('POST', origins.DEVICE_TOKEN_PATH):
                         lambda payload, e=error_name: (400, dict(error=e))}) as server:
            with pytest.raises(pkce.PkceError) as error:
                pkce.device_poll(server.base, 'device', interval=0.01, sleep=lambda s: None)
            assert error.value.category == category


# --------------------------------------------------------------------------
# Origins and network policy
# --------------------------------------------------------------------------

def test_default_origins_are_distinct_and_match_the_public_hosts():
    assert origins.auth_base() == 'https://www.orcarouter.ai'
    assert origins.api_base() == 'https://api.orcarouter.ai'
    assert origins.api_v1() == 'https://api.orcarouter.ai/v1'
    assert origins.url(origins.auth_base(), origins.EXCHANGE_PATH) == \
        'https://www.orcarouter.ai/api/v1/auth/keys'
    assert origins.MODELS_PATH == '/v1/models'


def test_the_wrong_auth_path_on_the_relay_is_never_produced():
    """``/v1/auth/keys`` on the relay 404s; it must not be derivable here.

    The invariant is that auth is always built from the auth origin and never
    from the API origin. Composing the exchange path onto the API base produces
    a different URL, which is exactly the mistake this pins down.
    """
    documented = origins.url(origins.auth_base(), origins.EXCHANGE_PATH)
    assert documented == 'https://www.orcarouter.ai/api/v1/auth/keys'
    # The two public origins are genuinely different hosts.
    assert origins.auth_base() != origins.api_base()
    assert urllib.parse.urlsplit(origins.auth_base()).netloc == 'www.orcarouter.ai'
    assert urllib.parse.urlsplit(origins.api_v1()).netloc == 'api.orcarouter.ai'
    # Building it from the inference base yields the documented 404 instead.
    mistaken = origins.api_v1() + origins.EXCHANGE_PATH
    assert mistaken != documented
    assert mistaken == 'https://api.orcarouter.ai/v1/api/v1/auth/keys'
    # No helper derives one public origin from the other.
    for env in ({}, {'ORCA_BASE_URL': 'https://shared.internal'}):
        assert origins.url(origins.auth_base(env), origins.EXCHANGE_PATH).startswith(
            origins.auth_base(env))
        assert origins.api_v1(env).endswith('/v1')


def test_explicit_overrides_win_over_the_shared_base(monkeypatch):
    monkeypatch.setenv('ORCA_BASE_URL', 'https://shared.internal')
    assert origins.auth_base() == 'https://shared.internal'
    assert origins.api_base() == 'https://shared.internal'
    monkeypatch.setenv('ORCA_AUTH_BASE_URL', 'https://auth.internal')
    monkeypatch.setenv('ORCA_API_BASE_URL', 'https://api.internal')
    assert origins.auth_base() == 'https://auth.internal'
    assert origins.api_base() == 'https://api.internal'
    assert origins.api_v1() == 'https://api.internal/v1'


@pytest.mark.parametrize('value', [
    'http://relay.example.com', 'ftp://example.com', 'https://user:pw@example.com',
    'https://example.com/v1', 'https://example.com#frag', 'https://example.com?x=1', 'not a url',
])
def test_remote_origins_require_https_and_a_clean_root(value, monkeypatch):
    monkeypatch.setenv('ORCA_BASE_URL', value)
    with pytest.raises(origins.OriginError) as error:
        origins.auth_base()
    assert value not in str(error.value)  # the value is withheld


def test_loopback_http_is_permitted_for_local_development(monkeypatch):
    monkeypatch.setenv('ORCA_BASE_URL', 'http://127.0.0.1:9700')
    assert origins.auth_base() == 'http://127.0.0.1:9700'
    assert origins.api_v1() == 'http://127.0.0.1:9700/v1'
    monkeypatch.setenv('ORCA_BASE_URL', 'http://localhost:9700/')
    assert origins.api_base() == 'http://localhost:9700'


# --------------------------------------------------------------------------
# Model catalog and capabilities
# --------------------------------------------------------------------------

def test_catalog_url_and_request_use_the_api_origin_with_bearer_auth():
    with FakeServer({('GET', '/v1/models'):
                     lambda query: (200, catalog_fixture())}) as server:
        result = catalog_module.fetch_catalog(FAKE_KEY, 'chat', env={'ORCA_API_BASE_URL': server.base})
        assert result['source'] == 'live' and result['degraded'] is False
        (record,) = server.records
        assert record['path'] == '/v1/models'
        assert record['query']['capability'] == ['chat']
        assert record['headers']['Authorization'] == f'Bearer {FAKE_KEY}'


def test_discovery_never_touches_the_auth_origin(store):
    with FakeServer({('GET', '/v1/models'): lambda query: (200, catalog_fixture())}) as server:
        store.save(Credential(FAKE_KEY, 'api_key'))
        provider.discover_models(provider.API_PROVIDER, store,
                                 env={'ORCA_API_BASE_URL': server.base,
                                      'ORCA_AUTH_BASE_URL': 'https://www.orcarouter.ai'})
        assert all('auth/keys' not in record['path'] for record in server.records)
        assert server.records[0]['path'] == '/v1/models'


def test_capability_filters_select_only_compatible_models():
    models = catalog_module.parse_catalog(catalog_fixture(), 'chat')
    ids = {model['id'] for model in models}
    assert ids == {'acme/text-only', 'acme/vision-chat'}

    assert {m['id'] for m in catalog_module.parse_catalog(catalog_fixture(), 'embedding')} == \
        {'acme/embed-small'}
    assert {m['id'] for m in catalog_module.parse_catalog(catalog_fixture(), 'image')} == \
        {'acme/draw'}
    assert {m['id'] for m in catalog_module.parse_catalog(catalog_fixture(), 'video')} == \
        {'acme/movie'}
    assert {m['id'] for m in catalog_module.parse_catalog(catalog_fixture(), 'rerank')} == \
        {'acme/rank'}


def test_multimodal_chat_requires_a_declared_image_modality():
    """Fail closed: an undeclared modality must not appear in the image picker."""
    models = catalog_module.parse_catalog(catalog_fixture(), 'chat', required_modalities=('image',))
    assert {model['id'] for model in models} == {'acme/vision-chat'}
    # A model with no architecture block at all is excluded from multimodal use.
    undeclared = model_record('acme/unknown-modalities')
    assert catalog_module.filter_models(
        [catalog_module.parse_model(undeclared)], 'chat', required_modalities=('image',)) == []


def test_media_generation_families_never_appear_in_the_text_picker():
    text_only = catalog_module.parse_catalog(catalog_fixture(), 'chat')
    for model in text_only:
        assert not set(model['endpoint_types']) & set(catalog_module.NON_TEXT_MARKERS)
    # A record advertising only a media endpoint is not a chat target.
    assert catalog_module.filter_models([catalog_module.parse_model(IMAGE_GEN)], 'chat') == []


def test_discovery_is_authoritative_and_never_merges_the_seed():
    live = dict(data=[CHAT_ONLY])
    with FakeServer({('GET', '/v1/models'): lambda query: (200, live)}) as server:
        result = catalog_module.catalog(FAKE_KEY, 'chat', env={'ORCA_API_BASE_URL': server.base})
    assert result['source'] == 'live' and result['degraded'] is False
    assert [model['id'] for model in result['models']] == ['acme/text-only']
    seed_ids = {model['id'] for model in catalog_module.seed_models()}
    assert not seed_ids & {model['id'] for model in result['models']}


def test_outage_keeps_a_verified_seed_with_metadata_intact():
    with FakeServer({}) as server:
        result = catalog_module.catalog(FAKE_KEY, 'chat', env={'ORCA_API_BASE_URL': server.base})
    assert result['source'] == 'verified_seed' and result['degraded'] is True
    assert result['reason']
    by_id = {model['id']: model for model in result['models']}
    assert set(by_id) == {model['id'] for model in catalog_module.seed_models()}
    # Reasoning ladders and modalities survive the fallback.
    assert by_id['openai/gpt-5.5']['reasoning_efforts'] == ['low', 'medium', 'high', 'xhigh']
    assert 'image' in by_id['openai/gpt-5.5']['input_modalities']
    assert 'image' in by_id['anthropic/claude-opus-4.8']['input_modalities']
    assert by_id['deepseek/deepseek-v4-pro']['input_modalities'] == ['text']
    assert by_id['openai/gpt-5.5']['context_length'] == 400000


def test_metadata_gap_is_filled_from_verified_metadata_not_erased():
    payload = dict(data=[model_record('openai/gpt-5.5')])
    (model,) = catalog_module.parse_catalog(payload, 'chat')
    assert model['reasoning_efforts'] == ['low', 'medium', 'high', 'xhigh']
    assert 'image' in model['input_modalities']


def test_non_chat_capabilities_never_fall_back_to_the_chat_seed():
    with FakeServer({}) as server:
        result = catalog_module.catalog(FAKE_KEY, 'embedding',
                                        env={'ORCA_API_BASE_URL': server.base})
    assert result['models'] == [] and result['degraded'] is True
    assert result['source'] == 'unavailable'


def test_empty_catalog_result_is_an_answer_not_an_outage():
    with FakeServer({('GET', '/v1/models'): lambda query: (200, dict(data=[CHAT_ONLY]))}) as server:
        result = catalog_module.catalog(FAKE_KEY, 'embedding',
                                        env={'ORCA_API_BASE_URL': server.base})
    assert result['models'] == [] and result['degraded'] is False and result['source'] == 'live'


def test_catalog_rejects_credential_failure_and_bounds_the_response():
    with FakeServer({('GET', '/v1/models'): lambda query: (401, dict(error='bad_key'))}) as server:
        with pytest.raises(catalog_module.CatalogError) as error:
            catalog_module.fetch_catalog(FAKE_KEY, 'chat', env={'ORCA_API_BASE_URL': server.base})
        assert 'rejected the stored key' in str(error.value)
        assert 'acme' not in str(error.value)


def test_malformed_catalog_records_are_dropped_not_displayed():
    payload = dict(data=[CHAT_ONLY, 'not-a-dict', dict(id=''), dict(id='has space'),
                         dict(id='x' * 500), dict(id='ok/model', supported_endpoint_types='nope')])
    models = catalog_module.parse_catalog(payload, 'chat')
    assert [model['id'] for model in models] == ['acme/text-only', 'ok/model']
    assert models[1]['input_modalities'] == ['text']


def test_selection_is_cleared_when_it_stops_being_compatible():
    models = catalog_module.parse_catalog(catalog_fixture(), 'chat')
    assert catalog_module.reconcile_selection('acme/vision-chat', models) == ('acme/vision-chat', False)
    assert catalog_module.reconcile_selection('acme/draw', models) == (None, True)
    assert catalog_module.reconcile_selection(None, models) == (None, False)
    # Switching to an image attachment invalidates a text-only selection.
    vision_only = catalog_module.filter_models(models, 'chat', ('image',))
    assert catalog_module.reconcile_selection('acme/text-only', vision_only) == (None, True)
    assert catalog_module.reconcile_selection('acme/vision-chat', vision_only) == \
        ('acme/vision-chat', False)


def test_is_compatible_matches_the_filtered_options():
    models = catalog_module.parse_catalog(catalog_fixture(), 'chat')
    text_only = next(model for model in models if model['id'] == 'acme/text-only')
    vision = next(model for model in models if model['id'] == 'acme/vision-chat')
    assert catalog_module.is_compatible(text_only, 'chat')
    assert not catalog_module.is_compatible(text_only, 'chat', ('image',))
    assert catalog_module.is_compatible(vision, 'chat', ('image',))
    assert not catalog_module.is_compatible(text_only, 'image')


def test_unknown_capability_fails_closed():
    with pytest.raises(catalog_module.CatalogError):
        catalog_module.filter_models([], 'telepathy')


# --------------------------------------------------------------------------
# Terminal 401 / generation safety
# --------------------------------------------------------------------------

def test_401_marks_the_exact_account_for_reauth_without_refresh_or_deletion(store):
    adapter = provider.adapter_for(provider.OAUTH_PROVIDER, store)
    credential = store.save(Credential(FAKE_KEY, 'oauth_pkce', 'acct-1'))
    outcome = adapter.handle_unauthorized(credential)
    assert outcome['action'] == 'reauthenticate'
    assert outcome['refresh_attempted'] is False
    assert store.needs_reauth('oauth_pkce', 'acct-1') is True
    # The stored secret survives, so a transient failure is not an account loss.
    assert store.load('oauth_pkce').reveal() == FAKE_KEY


def test_stale_401_cannot_poison_a_newer_credential(store):
    adapter = provider.adapter_for(provider.OAUTH_PROVIDER, store)
    old = store.save(Credential(FAKE_KEY, 'oauth_pkce', 'acct-1'))
    new = store.save(Credential(SECOND_KEY, 'oauth_pkce', 'acct-1'))
    assert new.generation == old.generation + 1
    # A late failure carrying the old generation must not mark the new one.
    assert store.mark_needs_reauth(old) is False
    assert store.needs_reauth('oauth_pkce', 'acct-1') is False
    assert store.load('oauth_pkce').reveal() == SECOND_KEY
    # The current generation is still allowed to transition.
    assert store.mark_needs_reauth(new) is True
    assert store.needs_reauth('oauth_pkce', 'acct-1') is True


def test_reauth_transition_does_not_touch_the_other_auth_choice(store):
    provider.adapter_for(provider.API_PROVIDER, store).acquire(FAKE_KEY)
    oauth = store.save(Credential(SECOND_KEY, 'oauth_pkce', 'acct-9'))
    provider.adapter_for(provider.OAUTH_PROVIDER, store).handle_unauthorized(oauth)
    assert store.needs_reauth('oauth_pkce', 'acct-9') is True
    assert store.needs_reauth('api_key') is False
    assert store.load('api_key').reveal() == FAKE_KEY


def test_terminal_401_helper_keeps_the_secret_and_reports_no_refresh(store):
    credential = store.save(Credential(FAKE_KEY, 'api_key'))
    outcome = terminal_401(credential, store)
    assert outcome['refresh_attempted'] is False
    assert outcome['generation'] == credential.generation
    assert store.load('api_key') is not None


def test_discovery_refuses_a_credential_marked_for_reauthentication(store):
    adapter = provider.adapter_for(provider.OAUTH_PROVIDER, store)
    credential = store.save(Credential(FAKE_KEY, 'oauth_pkce'))
    adapter.handle_unauthorized(credential)
    with pytest.raises(CredentialError, match='reauthentication'):
        provider.discover_models(provider.OAUTH_PROVIDER, store)
    assert store.load('oauth_pkce') is not None


def test_discovery_without_a_credential_asks_for_one(store):
    with pytest.raises(CredentialError, match='connect it first'):
        provider.discover_models(provider.API_PROVIDER, store)


def test_credential_value_absent_from_repr_errors_and_status(store):
    credential = store.save(Credential(FAKE_KEY, 'api_key', 'acct-1'))
    assert FAKE_KEY not in repr(credential)
    assert FAKE_KEY not in str(credential)
    assert FAKE_KEY not in json.dumps(provider.describe(provider.API_PROVIDER, store))
    for bad in ('sk-openai-nope', ''):
        try:
            provider.adapter_for(provider.API_PROVIDER, store).acquire(bad)
        except CredentialError as error:
            assert bad not in str(error) or not bad
    assert '***' in repr(credential)


# --------------------------------------------------------------------------
# The repo's own backend config integration
# --------------------------------------------------------------------------

def test_orcarouter_provider_block_validates_against_the_repo_validator(tmp_path):
    """The generated config must satisfy the repository's own fail-closed validator."""
    from .codex_backend.profiles import config_text
    text = config_text('galbot')
    assert '[model_providers.LiteLLM]' in text
    lines = '\n'.join(provider.config_lines(provider.API_PROVIDER)) + '\n'
    assert 'base_url = "https://api.orcarouter.ai/v1"' in lines
    assert 'wire_api = "responses"' in lines
    assert 'env_key = "OPENAI_API_KEY"' in lines


def test_other_providers_keep_their_existing_behaviour():
    """Adding OrcaRouter must not alter any existing profile."""
    from .codex_backend.profiles import PROFILES, config_text
    before = {name: config_text(name) for name in PROFILES}
    assert before['galbot'].startswith('model_provider = "LiteLLM"')
    assert 'orcarouter' not in before['galbot']
    assert 'model_providers.LiteLLM' in before['galbot']
    assert 'base_url = "https://gateway.example.invalid"' in before['galbot']
    for name in ('codex_a', 'codex_b_3', 'koozhan'):
        assert 'orcarouter' not in before[name]


def test_existing_suite_still_passes_for_auth_profiles():
    result = subprocess.run([sys.executable, '-m', 'pytest',
                             'hybrid_rollout/robodojo/test_auth_profiles.py', '-q', '--no-header'],
                            cwd=str(Path(__file__).resolve().parents[2]),
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------
# Network policy: every request goes to the origin that owns it
# --------------------------------------------------------------------------

def test_every_auth_request_targets_the_auth_origin_and_no_other_host():
    """Observed destinations, not asserted constants.

    A fake auth origin stands in for ``www.orcarouter.ai`` and a second fake
    stands in for the inference origin. Every request the auth fake receives
    must be an auth path, and the inference fake must stay untouched.
    """
    device_route = {('POST', origins.DEVICE_CODE_PATH):
                    lambda p: (200, dict(device_code='dev', user_code='ABCD-EFGH',
                                         interval=1, expires_in=60))}
    with FakeServer({**exchange_routes(), **device_route}) as auth_server, \
            FakeServer({('GET', '/v1/models'): lambda q: (200, catalog_fixture())}) as api_server:
        pkce.exchange_code(auth_server.base, 'code', 'verifier')
        pkce.device_start(auth_server.base)
        paths = [record['path'] for record in auth_server.records]
        assert paths == ['/api/v1/auth/keys', '/api/v1/auth/device/code']
        assert api_server.records == []


def test_inference_and_discovery_only_target_the_api_origin():
    """The relay origin receives inference and catalog calls; auth stays away."""
    with FakeServer({('GET', '/v1/models'): lambda q: (200, catalog_fixture())}) as api_server, \
            FakeServer(exchange_routes()) as auth_server:
        result = catalog_module.fetch_catalog(FAKE_KEY, 'chat',
                                              env={'ORCA_API_BASE_URL': api_server.base})
        assert result['source'] == 'live'
        assert [record['path'] for record in api_server.records] == ['/v1/models']
        assert auth_server.records == []
        # The inference path carries the version prefix exactly once.
        assert not any(record['path'].startswith('/api/v1/auth')
                       for record in api_server.records)


def test_describe_and_snapshot_never_contain_the_key(store):
    credential = store.save(Credential(FAKE_KEY, 'api_key'))
    described = provider.describe(provider.API_PROVIDER, store)
    assert FAKE_KEY not in json.dumps(described)
    assert FAKE_KEY not in repr(credential) and FAKE_KEY not in str(credential)
    assert described['secret_masked'] == mask(FAKE_KEY)


def test_auth_origin_is_never_derived_from_the_api_origin(monkeypatch):
    """Setting only the API override must not move authentication."""
    monkeypatch.setenv('ORCA_API_BASE_URL', 'https://relay.internal')
    assert origins.api_v1() == 'https://relay.internal/v1'
    assert origins.auth_base() == 'https://www.orcarouter.ai'
    monkeypatch.setenv('ORCA_AUTH_BASE_URL', 'https://signin.internal')
    assert origins.auth_base() == 'https://signin.internal'
    assert origins.api_base() == 'https://relay.internal'
    # A shared self-hosted base moves both only when neither override is set.
    monkeypatch.delenv('ORCA_API_BASE_URL')
    monkeypatch.delenv('ORCA_AUTH_BASE_URL')
    monkeypatch.setenv('ORCA_BASE_URL', 'https://one.internal')
    assert origins.auth_base() == origins.api_base() == 'https://one.internal'


def test_remote_plaintext_api_origin_is_refused(monkeypatch):
    monkeypatch.setenv('ORCA_AUTH_BASE_URL', 'https://www.orcarouter.ai')
    monkeypatch.setenv('ORCA_API_BASE_URL', 'http://relay.remote')
    with pytest.raises(origins.OriginError):
        origins.api_base()
    # Loopback over http stays available for a local self-hosted relay.
    monkeypatch.setenv('ORCA_API_BASE_URL', 'http://127.0.0.1:9999')
    assert origins.api_v1() == 'http://127.0.0.1:9999/v1'
    assert origins.url(origins.auth_base(), origins.EXCHANGE_PATH) == \
        'https://www.orcarouter.ai/api/v1/auth/keys'


def test_key_and_verifier_never_appear_in_urls_errors_or_logs():
    """Verifier and key must not reach a URL, an error message, or a log line."""
    with FakeServer(exchange_routes()) as server:
        credential = pkce.exchange_code(server.base, 'the-code', 'the-verifier')
    for record in server.records:
        # The verifier is in the JSON body, never in the query a proxy logs.
        assert 'the-verifier' not in record['raw_query']
        assert 'the-code' not in record['raw_query']
        assert record['query'] == {}
        assert credential.reveal() not in record['raw_query']

    with FakeServer(exchange_routes(status=403)) as server:
        with pytest.raises(pkce.PkceError) as error:
            pkce.exchange_code(server.base, 'the-code', 'the-verifier')
        assert 'the-verifier' not in str(error.value)
        assert 'the-code' not in str(error.value)

    # Not in a repr, and not in the status payload the browser receives.
    assert 'the-verifier' not in repr(pkce.PkceAttempt())
    assert credential.reveal() not in repr(credential)


def test_device_start_and_poll_use_the_documented_paths():
    def device_code(payload):
        return 200, dict(device_code='dev', user_code='ABCD-EFGH', interval=1, expires_in=60,
                         verification_uri='https://www.orcarouter.ai/device',
                         verification_uri_complete='https://www.orcarouter.ai/device?code=ABCD-EFGH')

    with FakeServer({('POST', origins.DEVICE_CODE_PATH): device_code,
                     ('POST', origins.DEVICE_TOKEN_PATH):
                         lambda p: (200, dict(key=FAKE_KEY, user_id='7', scope='api'))}) as server:
        start = pkce.device_start(server.base)
        assert start['verification_uri_complete'] == 'https://www.orcarouter.ai/device?code=ABCD-EFGH'
        credential = pkce.device_poll(server.base, start['device_code'], interval=0.01,
                                      sleep=lambda s: None)
        assert credential.reveal() == FAKE_KEY
    paths = [record['path'] for record in server.records]
    assert paths == ['/api/v1/auth/device/code', '/api/v1/auth/device/token']


def test_discovery_and_inference_helpers_only_use_versioned_relay_paths():
    assert origins.MODELS_PATH == '/v1/models'
    assert origins.CHAT_COMPLETIONS_PATH == '/v1/chat/completions'
    # The version prefix belongs to the path constant, not to the origin: the
    # origin is a bare host and must never carry /v1 itself.
    assert not origins.api_base().endswith('/v1')
    for path in (origins.MODELS_PATH, origins.CHAT_COMPLETIONS_PATH):
        assert origins.url(origins.api_base(), path) == 'https://api.orcarouter.ai' + path
    assert origins.url(origins.api_base(), origins.MODELS_PATH) == \
        'https://api.orcarouter.ai/v1/models'
    # A configured base that already carries /v1 is refused, not doubled.
    with pytest.raises(origins.OriginError):
        origins.api_base({'ORCA_API_BASE_URL': 'https://api.orcarouter.ai/v1'})


# --------------------------------------------------------------------------
# Provider console (server side)
# --------------------------------------------------------------------------

def console_client(store, env=None):
    """Start the real console on loopback and return a small JSON client."""
    from .codex_backend.orcarouter import console
    server = console.serve(store, '127.0.0.1', 0, env=env)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05},
                              daemon=True)
    thread.start()

    class Client:
        base = f'http://127.0.0.1:{server.server_port}'

        def request(self, method, path, body=None, token=None, headers=None):
            data = json.dumps(body).encode() if body is not None else None
            request = urllib.request.Request(self.base + path, data=data, method=method)
            request.add_header('Content-Type', 'application/json')
            if token is not None:
                request.add_header('X-Review-Token', token)
            for key, value in (headers or {}).items():
                request.add_header(key, value)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return response.status, json.loads(response.read() or b'{}')
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read() or b'{}')

        def token(self):
            page = urllib.request.urlopen(self.base + '/', timeout=30).read().decode()
            return page.split('const csrf=')[1].split(';')[0].strip().strip('"')

        def page(self):
            return urllib.request.urlopen(self.base + '/', timeout=30).read().decode()

        def close(self):
            server.orca_manager.cancel()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    return Client()


def test_console_serves_both_auth_choices_and_never_the_key(store):
    store.save(Credential(FAKE_KEY, 'api_key'))
    client = console_client(store)
    try:
        page = client.page()
        # Both authentication choices are present as first-class controls.
        assert 'OrcaRouter - API' in page
        assert 'OrcaRouter - Auth' in page
        assert 'id="apiKey"' in page and 'id="connect"' in page and 'id="clearKey"' in page
        assert 'Connect with OrcaRouter' in page
        # The page itself never embeds the credential.
        assert FAKE_KEY not in page
        status, body = client.request('GET', '/api/status')
        assert status == 200
        assert body['api_key']['secret_masked'] == mask(FAKE_KEY)
        assert body['oauth']['configured'] is False
        assert body['inference_base'] == 'https://api.orcarouter.ai/v1'
        assert body['auth_base'] == 'https://www.orcarouter.ai'
        assert FAKE_KEY not in json.dumps(body)
    finally:
        client.close()


def test_console_models_endpoint_keeps_the_key_server_side(store):
    store.save(Credential(FAKE_KEY, 'api_key'))
    with FakeServer({('GET', '/v1/models'): lambda q: (200, catalog_fixture())}) as api_server:
        client = console_client(store, env={'ORCA_API_BASE_URL': api_server.base})
        try:
            status, body = client.request('GET', '/api/models?capability=chat')
            assert status == 200 and body['source'] == 'live'
            assert {model['id'] for model in body['models']} == {'acme/text-only', 'acme/vision-chat'}
            # Only minimal metadata crosses to the browser.
            for model in body['models']:
                assert set(model) == {'id', 'name', 'owned_by', 'context_length', 'input_modalities'}
            assert FAKE_KEY not in json.dumps(body)
            # The catalog call was made server side, with the stored key.
            assert api_server.records[0]['headers']['Authorization'] == f'Bearer {FAKE_KEY}'
            # And the browser never received it.
            page = client.page()
            assert FAKE_KEY not in page
        finally:
            client.close()


def test_console_multimodal_filter_is_applied_server_side(store):
    store.save(Credential(FAKE_KEY, 'api_key'))
    with FakeServer({('GET', '/v1/models'): lambda q: (200, catalog_fixture())}) as api_server:
        client = console_client(store, env={'ORCA_API_BASE_URL': api_server.base})
        try:
            _, text = client.request('GET', '/api/models?capability=chat')
            _, image = client.request('GET', '/api/models?capability=chat&modality=image')
            assert {m['id'] for m in image['models']} == {'acme/vision-chat'}
            assert {m['id'] for m in image['models']} < {m['id'] for m in text['models']}
            assert image['required_modalities'] == ['image']
        finally:
            client.close()


def test_console_asks_for_a_credential_before_listing_models(store):
    client = console_client(store)
    try:
        status, body = client.request('GET', '/api/models?capability=chat')
        assert status == 200 and body['models'] == []
        assert 'No OrcaRouter credential' in body['error']
    finally:
        client.close()


def test_console_writes_require_the_page_token_and_a_trusted_host(store):
    client = console_client(store)
    try:
        status, body = client.request('POST', '/api/key', {'key': FAKE_KEY}, token='wrong')
        assert status == 403 and 'write token' in body['error']
        status, body = client.request('POST', '/api/key', {'key': FAKE_KEY})
        assert status == 403
        # A rebound Host header is refused even with the right token.
        status, body = client.request('POST', '/api/key', {'key': FAKE_KEY},
                                      token=client.token(), headers={'Host': 'evil.example'})
        assert status == 403
        # Nothing was stored by any of the rejected attempts.
        assert store.load('api_key') is None
    finally:
        client.close()


def test_console_saves_and_clears_the_api_key_without_echoing_it(store):
    client = console_client(store)
    try:
        status, body = client.request('POST', '/api/key', {'key': FAKE_KEY}, token=client.token())
        assert status == 200
        assert FAKE_KEY not in json.dumps(body)
        assert body['api_key']['secret_masked'] == mask(FAKE_KEY)
        assert body['api_key']['configured'] is True
        assert store.load('api_key').reveal() == FAKE_KEY

        status, body = client.request('POST', '/api/key', {'key': 'not-an-orca-key'},
                                      token=client.token())
        assert status == 400 and 'sk-orca-' in body['error']
        assert 'not-an-orca-key' not in body['error']

        status, body = client.request('POST', '/api/key/clear', {}, token=client.token())
        assert status == 200 and body['api_key']['configured'] is False
        assert store.load('api_key') is None
    finally:
        client.close()


def test_console_login_lock_is_released_by_cancel_and_pagehide(store):
    """Both the Cancel button and the pagehide unload path free the login lock."""
    client = console_client(store)
    try:
        token = client.token()
        # Cancel path.
        status, first = client.request('POST', '/api/connect/start', {}, token=token)
        assert status == 200 and first['authorize_url'].startswith('https://www.orcarouter.ai/auth?')
        assert 'client_secret' not in first['authorize_url']
        assert first['authorize_url'].count('code_challenge_method=S256') == 1
        status, polled = client.request(
            'GET', f"/api/connect/poll?session={first['session']}")
        assert status == 200 and polled['state'] == 'pending'
        status, _ = client.request('POST', '/api/connect/cancel', {'session': first['session']},
                                   token=token)
        assert status == 200
        _, after = client.request('GET', f"/api/connect/poll?session={first['session']}")
        assert after['state'] == 'none'  # server lock released

        # A second attempt starts without any remount.
        status, second = client.request('POST', '/api/connect/start', {}, token=token)
        assert status == 200 and second['session'] != first['session']
        # The pagehide path cancels via the same endpoint.
        status, _ = client.request('POST', '/api/connect/cancel', {'session': second['session']},
                                   token=token)
        assert status == 200
        _, third = client.request('GET', f"/api/connect/poll?session={second['session']}")
        assert third['state'] == 'none'
    finally:
        client.close()


def test_console_stale_poll_of_an_abandoned_attempt_reports_nothing(store):
    client = console_client(store)
    try:
        token = client.token()
        _, first = client.request('POST', '/api/connect/start', {}, token=token)
        _, second = client.request('POST', '/api/connect/start', {}, token=token)
        assert first['session'] != second['session']
        # A late poll from the superseded attempt must not publish its state.
        _, stale = client.request('GET', f"/api/connect/poll?session={first['session']}")
        assert stale['state'] == 'none'
        client.request('POST', '/api/connect/cancel', {'session': second['session']}, token=token)
    finally:
        client.close()


def test_console_connect_url_targets_the_auth_origin_not_the_relay(store):
    client = console_client(store)
    try:
        _, started = client.request('POST', '/api/connect/start', {}, token=client.token())
        target = started['authorize_url']
        parsed = urllib.parse.urlsplit(target)
        assert parsed.netloc == 'www.orcarouter.ai' and parsed.path == '/auth'
        params = urllib.parse.parse_qs(parsed.query)
        assert params['code_challenge_method'] == ['S256']
        assert params['callback_url'][0].startswith('http://127.0.0.1:')
        assert 'api.orcarouter.ai' not in target
        client.request('POST', '/api/connect/cancel', {}, token=client.token())
    finally:
        client.close()


# --------------------------------------------------------------------------
# Live checks (real OrcaRouter account; only when a key is present)
# --------------------------------------------------------------------------

LIVE_KEY = os.environ.get('ORCAROUTER_API_KEY')

live = pytest.mark.skipif(not LIVE_KEY, reason='ORCAROUTER_API_KEY not configured')


@live
def test_live_model_catalog_through_the_provider_path(tmp_path):
    """Real discovery through the implemented provider/model-discovery path."""
    directory = tmp_path/'live-private'
    private_dir(directory)
    store = CredentialStore(directory=directory)
    provider.adapter_for(provider.API_PROVIDER, store).acquire(LIVE_KEY)
    result = provider.discover_models(provider.API_PROVIDER, store, 'chat')
    assert result['source'] == 'live', result.get('reason')
    assert result['degraded'] is False
    assert len(result['models']) > 1
    for model in result['models']:
        assert '/' in model['id']  # vendor/model namespace preserved verbatim
        assert set(model['endpoint_types']) & set(catalog_module.TEXT_ENDPOINT_TYPES)
        assert not set(model['endpoint_types']) & set(catalog_module.NON_TEXT_MARKERS)


@live
def test_live_multimodal_dropdown_contains_only_declared_image_models(tmp_path):
    directory = tmp_path/'live-vision-private'
    private_dir(directory)
    store = CredentialStore(directory=directory)
    provider.adapter_for(provider.API_PROVIDER, store).acquire(LIVE_KEY)
    result = provider.discover_models(provider.API_PROVIDER, store, 'chat',
                                      required_modalities=('image',))
    assert result['source'] == 'live'
    for model in result['models']:
        assert 'image' in model['input_modalities']


@live
def test_live_inference_through_the_configured_provider(tmp_path):
    """A real completion through the implemented provider/credential path.

    A workspace catalog lists every model the workspace can route, while an
    individual key may still be entitled to a subset; the relay answers 403
    ``model_access_denied`` for the rest. The check therefore walks the live
    catalog (bounded) and requires at least one model to actually complete.
    """
    store_directory = tmp_path/'live-infer-private'
    private_dir(store_directory)
    store = CredentialStore(directory=store_directory)
    credential = provider.adapter_for(provider.API_PROVIDER, store).acquire(LIVE_KEY)
    result = provider.discover_models(provider.API_PROVIDER, store, 'chat')
    assert result['source'] == 'live'

    attempts, served = [], None
    for model in result['models'][:12]:
        request = urllib.request.Request(
            origins.url(origins.api_base(), origins.CHAT_COMPLETIONS_PATH),
            data=json.dumps(dict(
                model=model['id'], max_tokens=64,
                messages=[dict(role='user', content='Reply with the single word: ok')])).encode(),
            headers={'Authorization': f'Bearer {credential.reveal()}',
                     'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
            # A reasoning model can spend a small token budget entirely on
            # reasoning and return empty content; the envelope is what proves
            # the request was routed and answered.
            choices = payload.get('choices')
            assert isinstance(choices, list) and choices, model['id']
            assert 'message' in choices[0], model['id']
            served = model['id']
            attempts.append((model['id'], 'ok'))
            break
        except urllib.error.HTTPError as error:
            # A per-key entitlement refusal is a catalog/inference-scope fact,
            # not a wiring failure; anything else is a real failure.
            assert error.code in (403, 404, 429), f'{model["id"]}: HTTP {error.code}'
            attempts.append((model['id'], f'HTTP {error.code}'))
    assert served, f'no live model completed; attempts={attempts}'
    formatted = next((entry for entry in attempts if entry[1] == 'ok'), None)
    print(f'live inference served by {formatted[0]}; attempts={attempts}')
    # The origin that served inference is the relay, never the auth host.
    assert origins.api_base() == 'https://api.orcarouter.ai'


@live
def test_live_auth_origin_differs_from_inference_origin():
    assert origins.auth_base() == 'https://www.orcarouter.ai'
    assert origins.api_v1() == 'https://api.orcarouter.ai/v1'
    # The relay's versioned auth path is a documented 404 and must stay unused.
    assert origins.url(origins.auth_base(), origins.EXCHANGE_PATH) == \
        'https://www.orcarouter.ai/api/v1/auth/keys'
