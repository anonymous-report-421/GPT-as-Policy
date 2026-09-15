"""Cancellable OAuth 2.0 + PKCE connect flows for OrcaRouter.

Implemented here:

* **Flow A** — loopback redirect (``callback_url=http://127.0.0.1:<port>/cb``).
  Used by the local provider console, which is itself a loopback HTTP service
  and can therefore receive the redirect.
* **Flow B** — out-of-band code (``callback_url=oob``). Used by the CLI, where a
  rollout install address differs on every deployment and no predictable
  callback can be registered.
* **Flow C** — device grant (RFC 8628). Optional extra capability; it never
  substitutes for PKCE and is not required by either shipped surface.

Every attempt mints a fresh verifier and state from the operating system's
cryptographic RNG. The challenge is ``base64url(sha256(verifier))`` with no
padding and the method is **always** ``S256`` — including Flow A, because the
consent screen lets a user ask for a displayed code, and a displayed code must be
redeemable only by the process holding the verifier.

The verifier never leaves this process before the exchange: it is never placed in
a URL, a log record, an exception, or a telemetry event, and :class:`PkceAttempt`
reprs without it.

Every authorization attempt belongs to a *generation*. :meth:`PkceSession.begin`
mints a new generation and cancels whatever came before, so a late poll or a
late exchange from an abandoned attempt can never publish its result over a
newer one, and a cancelled attempt always releases its listener instead of
leaving the caller waiting.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import origins
from .credentials import Credential, CredentialError, validate_key_text

CODE_CHALLENGE_METHOD = 'S256'
DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'
DEFAULT_SCOPE = 'api'
BROWSER_PATH = '/cb'
MAX_RESPONSE_BYTES = 65536
DEFAULT_TIMEOUT = 30.0

#: Authorization codes are single use with a 10 minute TTL; the local wait stays
#: just below that so a slow user gets a clear timeout instead of a 403.
AUTHORIZATION_TIMEOUT = 540.0

_PAGE = ('<!doctype html><html lang="en"><meta charset="utf-8">'
         '<title>OrcaRouter</title><body style="font:16px system-ui;padding:2rem">'
         '<h3>OrcaRouter connected</h3>'
         '<p>You can close this tab and return to the terminal.</p>'
         '</body></html>').encode()


class PkceError(RuntimeError):
    """A connect attempt ended without a credential.

    Messages are written to be shown to a user, so they never embed an auth
    code, a verifier, a key, or a raw response body.
    """

    def __init__(self, category, message, *, retryable=False):
        super().__init__(message)
        self.category = category
        self.retryable = retryable


class PkceCancelled(PkceError):
    """The attempt was cancelled locally; nothing was stored."""

    def __init__(self, message='Authorization was cancelled. Nothing was stored.'):
        super().__init__('cancelled', message)


def b64url(raw):
    return base64.urlsafe_b64encode(raw).decode('ascii').rstrip('=')


def challenge_for(verifier):
    """``base64url(sha256(verifier))`` with no padding."""
    return b64url(hashlib.sha256(verifier.encode('ascii')).digest())


class PkceAttempt:
    """One authorization attempt: fresh verifier, fresh challenge, fresh state."""

    __slots__ = ('verifier', 'challenge', 'state')

    def __init__(self, entropy=None):
        source = entropy or secrets.token_bytes
        self.verifier = b64url(source(32))
        self.challenge = challenge_for(self.verifier)
        self.state = b64url(source(16))

    def __repr__(self):
        # The verifier must never be printable.
        return f'PkceAttempt(challenge={self.challenge!r}, state={self.state!r}, verifier=***)'

    __str__ = __repr__


def authorize_url(auth_base, attempt, callback_url, app_name='GPT as Policy rollout',
                  scope=DEFAULT_SCOPE):
    """Build the consent URL. The verifier is deliberately absent from it."""
    return origins.url(auth_base, origins.AUTHORIZE_PATH, {
        'callback_url': callback_url,
        'code_challenge': attempt.challenge,
        'code_challenge_method': CODE_CHALLENGE_METHOD,
        'state': attempt.state,
        'app_name': app_name,
        'scope': scope,
    })


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _post_json(target, payload, timeout=DEFAULT_TIMEOUT):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        target, data=body, method='POST',
        headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except urllib.error.HTTPError as error:
        # An error body may echo credentials, so only the status is propagated.
        raw = error.read(MAX_RESPONSE_BYTES + 1)
        status = error.code
    except (urllib.error.URLError, OSError, ValueError):
        raise PkceError('network',
                        'Could not reach the OrcaRouter authorization service. '
                        'Check your network or proxy settings and try again.',
                        retryable=True) from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise PkceError('protocol', 'Authorization response was unreasonably large.')
    try:
        data = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeDecodeError):
        data = None
    return status, data if isinstance(data, dict) else {}


def requested_scope(scope):
    """Validate a requested scope before it is sent.

    The consent endpoint refuses anything outside ``api``/``connector``; failing
    locally gives a clearer message and avoids a pointless round trip.
    """
    if scope not in ('api', 'connector'):
        raise PkceError('protocol', f'Unsupported scope {scope!r}; use "api" or "connector".')
    return scope


def credential_from_response(data, requested):
    """Turn an exchange/device payload into a credential, checking the *granted* scope.

    The response reports what was granted, not what was asked for. A client that
    requested ``connector`` and reads ``api`` was approved by someone whose
    workspace role does not permit the wider grant, so the narrower grant is
    refused here rather than assumed sufficient.
    """
    try:
        key = data.get('key')
        if not isinstance(key, str):
            raise CredentialError('no key')
        validate_key_text(key)
    except CredentialError:
        raise PkceError('protocol',
                        'The authorization response did not contain a usable key. '
                        'Start the sign-in again.') from None
    granted = data.get('scope')
    if isinstance(granted, str) and granted != requested:
        # Covers both downgrades (connector -> api) and an echo of a wider grant
        # than was requested.
        raise PkceError('scope',
                        f'Authorization was granted scope {granted!r}, but {requested!r} is '
                        f'required for this client. Ask a workspace owner to grant '
                        f'{requested!r}, or use an API key instead.')
    if not isinstance(granted, str) or granted not in ('api', 'connector'):
        raise PkceError('scope',
                        'Authorization returned no usable scope. Start the sign-in again.')
    return Credential(key, 'oauth_pkce', str(data.get('user_id') or 'local'), granted, 0)


def exchange_code(auth_base, code, verifier, requested_scope_=DEFAULT_SCOPE,
                  timeout=DEFAULT_TIMEOUT):
    """Redeem an auth code for a durable OrcaRouter API key.

    JSON is sent rather than form encoding so the verifier never sits in a
    URL-encoded body that an intermediary might log.
    """
    if not code or not verifier:
        raise PkceError('protocol', 'The authorization code or verifier was missing.')
    status, data = _post_json(origins.url(auth_base, origins.EXCHANGE_PATH), {
        'code': code,
        'code_verifier': verifier,
        'code_challenge_method': CODE_CHALLENGE_METHOD,
    }, timeout=timeout)
    if status == 200:
        return credential_from_response(data, requested_scope_)
    if status == 400:
        raise PkceError('protocol',
                        'The authorization service rejected the PKCE method. '
                        'Start the sign-in again.')
    if status == 403:
        raise PkceError('rejected',
                        'That authorization code is unknown, expired, already used, or does '
                        'not match this session. Sign in again to get a new one.')
    if status == 429:
        raise PkceError('rate_limited',
                        'Too many OrcaRouter keys were issued for this account recently. '
                        'Keep using the stored key, or wait before signing in again.')
    if status >= 500:
        raise PkceError('server',
                        f'The OrcaRouter authorization service is unavailable (HTTP {status}). '
                        'Try again shortly.', retryable=True)
    raise PkceError('protocol', f'Unexpected authorization response (HTTP {status}).')


# --------------------------------------------------------------------------
# Flow A — loopback redirect
# --------------------------------------------------------------------------

class LoopbackReceiver:
    """Single-use loopback listener that receives the authorization redirect."""

    def __init__(self, path=BROWSER_PATH, host='127.0.0.1'):
        self.path = path
        self.host = host
        self.state = None
        self._result = {}
        self._event = threading.Event()
        self._server = None
        self._thread = None

    def start(self, state):
        self.state = state
        receiver = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # Query strings carry the one-time code; never log them.

            def do_GET(self):
                parsed = urllib.parse.urlsplit(self.path)
                if parsed.path != receiver.path:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(_PAGE)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(_PAGE)
                receiver._deliver(urllib.parse.parse_qs(parsed.query))

        self._server = http.server.HTTPServer((self.host, 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        self._thread.start()
        return self.port

    def _deliver(self, query):
        arg = lambda key: query.get(key, [''])[0]
        # Compare state before touching the code. The listener is reachable by
        # any page that can send a request to loopback, so state is the only
        # thing standing between it and somebody else's code.
        received = arg('state')
        if not isinstance(self.state, str) or not secrets.compare_digest(received, self.state):
            self._result.update(error='state',
                                detail='Authorization state did not match this session.')
        elif arg('error'):
            self._result.update(error='denied', detail=arg('error'))
        elif arg('code'):
            self._result.update(code=arg('code'))
        else:
            self._result.update(error='missing', detail='No authorization code was returned.')
        self._event.set()

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def redirect_uri(self):
        return f'http://{self.host}:{self.port}{self.path}'

    def wait(self, timeout=AUTHORIZATION_TIMEOUT, cancelled=None):
        """Block until the redirect arrives, the attempt is cancelled, or time runs out.

        ``cancelled`` is an optional callable polled while waiting so a cancel
        from another thread ends the wait promptly instead of after the full
        authorization timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                raise PkceCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PkceError('timeout',
                                'Timed out waiting for the browser to return an authorization '
                                'code. Start the sign-in again.')
            if self._event.wait(min(0.25, remaining)):
                break
        if 'code' in self._result:
            return self._result['code']
        if self._result.get('error') == 'state':
            raise PkceError('state', self._result['detail'])
        if self._result.get('error') == 'cancelled':
            raise PkceCancelled()
        raise PkceError('denied',
                        f'Authorization was declined ({self._result.get("detail", "denied")}). '
                        'Nothing was stored; your existing credential is unchanged.')

    def cancel(self):
        """Release the listener without waiting; safe to call more than once."""
        self._result.setdefault('error', 'cancelled')
        self._result.setdefault('detail', 'cancelled')
        self._event.set()
        self.close()

    def close(self):
        server, self._server = self._server, None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except OSError:
                pass


class PkceSession:
    """A generation-guarded authorization attempt that can always be cancelled.

    The session holds one :class:`LoopbackReceiver` and one attempt for its
    lifetime. :meth:`cancel` is safe from any thread and from a signal handler,
    and is the single path that releases the listener, so a closed dialog, a
    ``pagehide`` or a timeout cannot leave a socket or a wait behind.
    """

    def __init__(self, auth_base, app_name='GPT as Policy rollout', scope=DEFAULT_SCOPE,
                 timeout=DEFAULT_TIMEOUT, receiver=None, open_browser=None):
        self.auth_base = auth_base
        self.app_name = app_name
        self.scope = requested_scope(scope)
        self.timeout = timeout
        self.attempt = PkceAttempt()
        self.id = b64url(secrets.token_bytes(8))
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._receiver = receiver or LoopbackReceiver()
        self._open_browser = open_browser
        self._port = None
        self._started = False
        self._finished = False

    # -- lifecycle --------------------------------------------------------
    @property
    def cancelled(self):
        return self._cancelled.is_set()

    @property
    def finished(self):
        return self._finished

    @property
    def port(self):
        """The bound loopback port, or ``None`` before :meth:`begin`."""
        return self._port

    @property
    def redirect_uri(self):
        return self._receiver.redirect_uri

    def begin(self):
        """Start listening and return the consent URL to show or open."""
        with self._lock:
            if self._cancelled.is_set():
                raise PkceCancelled()
            if self._started:
                raise PkceError('protocol', 'This authorization session already started.')
            self._port = self._receiver.start(self.attempt.state)
            self._started = True
        target = authorize_url(self.auth_base, self.attempt, self._receiver.redirect_uri,
                               self.app_name, self.scope)
        return target

    def open_browser(self, target):
        if self._open_browser and not self._cancelled.is_set():
            self._open_browser(target)

    def cancel(self, reason='cancelled'):
        """Mark the attempt abandoned and release its listener.

        Callable repeatedly and from any thread. A cancel that lands while the
        exchange is in flight prevents the credential from being persisted.
        """
        self._cancelled.set()
        self._receiver.cancel()

    def release(self):
        """Close the listener without marking the attempt cancelled.

        Used on the success path, where the redirect has already arrived and the
        port must be freed before the code is redeemed.
        """
        self._receiver.cancel()

    def finish(self, credential):
        """Publish a credential unless the attempt was cancelled meanwhile."""
        with self._lock:
            self._finished = True
            if self._cancelled.is_set():
                raise PkceCancelled()
            return credential

    def wait_for_code(self):
        return self._receiver.wait(cancelled=self._cancelled.is_set)

    def __repr__(self):
        return f'PkceSession(id={self.id!r}, port={self._port!r}, cancelled={self.cancelled})'


def connect_loopback(auth_base, store, *, app_name='GPT as Policy rollout',
                     scope=DEFAULT_SCOPE, timeout=DEFAULT_TIMEOUT,
                     open_browser=None, receiver=None, session=None,
                     on_url=None):
    """Flow A end to end: listen, open the browser, redeem, persist."""
    session = session or PkceSession(auth_base, app_name, scope, timeout, receiver, open_browser)
    try:
        target = session.begin()
        if on_url:
            on_url(target)
        session.open_browser(target)
        code = session.wait_for_code()
    finally:
        session.release()
    credential = exchange_code(auth_base, code, session.attempt.verifier, session.scope, timeout)
    session.finish(credential)
    return store.save(credential)


def connect_oob(auth_base, store, *, app_name='GPT as Policy rollout',
                scope=DEFAULT_SCOPE, timeout=DEFAULT_TIMEOUT, on_url=None,
                read_code=input):
    """Flow B end to end: show the URL, read the pasted code, redeem, persist."""
    attempt = PkceAttempt()
    app = requested_scope(scope)
    target = authorize_url(auth_base, attempt, 'oob', app_name, app)
    if on_url:
        on_url(target)
    code = (read_code('Authorization code: ') or '').strip()
    if not code:
        raise PkceCancelled('No authorization code entered. Nothing was stored.')
    credential = exchange_code(auth_base, code, attempt.verifier, app, timeout)
    return store.save(credential)


# --------------------------------------------------------------------------
# Flow C — device grant (optional extra)
# --------------------------------------------------------------------------

def device_start(auth_base, app_name='GPT as Policy rollout', scope=DEFAULT_SCOPE,
                 timeout=DEFAULT_TIMEOUT):
    status, data = _post_json(origins.url(auth_base, origins.DEVICE_CODE_PATH),
                              {'app_name': app_name, 'scope': requested_scope(scope)},
                              timeout=timeout)
    if status != 200 or not data.get('device_code'):
        raise PkceError('protocol', f'Could not start a device authorization (HTTP {status}).')
    return data


def device_poll(auth_base, device_code, *, interval=5.0, expires_in=600.0,
                timeout=DEFAULT_TIMEOUT, sleep=time.sleep, clock=time.monotonic,
                scope=DEFAULT_SCOPE, cancelled=None):
    """Poll the device endpoint, branching on ``error`` and nothing else."""
    app = requested_scope(scope)
    deadline = clock() + min(float(expires_in), AUTHORIZATION_TIMEOUT)
    wait = max(float(interval), 1.0)
    while clock() < deadline:
        sleep(wait)
        if cancelled is not None and cancelled():
            raise PkceCancelled()
        status, data = _post_json(origins.url(auth_base, origins.DEVICE_TOKEN_PATH), {
            'device_code': device_code,
            'grant_type': DEVICE_GRANT_TYPE,
        }, timeout=timeout)
        error = data.get('error')
        if status == 200 and not error:
            return credential_from_response(data, app)
        if error == 'authorization_pending':
            continue
        if error == 'slow_down':
            wait += 5.0
            continue
        if error == 'access_denied':
            raise PkceError('denied', 'The device authorization was declined.')
        if error == 'expired_token':
            raise PkceError('timeout', 'The device authorization window closed. Start again.')
        if error == 'unsupported_grant_type':
            raise PkceError('protocol', 'The device grant type was rejected.')
        raise PkceError('protocol', f'Device authorization failed (HTTP {status}).')
    raise PkceError('timeout', 'Device authorization timed out. Start again.')
