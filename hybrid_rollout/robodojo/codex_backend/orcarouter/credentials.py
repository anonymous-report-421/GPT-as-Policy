"""One credential interface with two adapters: pasted API key and PKCE login.

Both acquisition paths end here and produce the same :class:`Credential`, so the
provider block, the model catalog and every inference consumer stay unaware of
where a key came from. Secrets never appear in a ``repr``, an exception message,
a log line or a telemetry payload; the value is exposed only through
``reveal()`` and only at the point of use.

Storage reuses the existing rollout private store
(``<shared_root>/private``, user-owned, mode 0700, no symlinks). No second
secret store is introduced.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time

from ..profiles import private_dir, shared_root

KEY_PREFIX = 'sk-orca-'
API_KEY_FILENAME = 'orcarouter.key'
OAUTH_KEY_FILENAME = 'orcarouter_oauth.key'
ACCOUNTS_FILENAME = 'orcarouter_accounts.json'
MAX_KEY_BYTES = 4096

# OrcaRouter issues opaque keys of the form ``sk-orca-<random>``. This is a
# lightweight shape check that catches obvious paste mistakes (a wrong provider's
# key, a truncated copy, an embedded newline). It is *not* proof that a
# credential is valid: the only real test is a request the provider accepts, so
# no paid inference call is made merely to make a settings form show "valid".
_KEY_RE = re.compile(r'^sk-orca-[A-Za-z0-9_\-]{8,}$')

SOURCES = ('api_key', 'oauth_pkce')


class CredentialError(ValueError):
    """A credential or its storage is unusable; messages never echo the value."""


class Credential:
    """An OrcaRouter API key plus the provenance needed for lifecycle decisions.

    ``generation`` increments every time a credential is written for an account.
    A late ``401`` from an old request must never mark a newer credential as
    broken, so reauthentication state is recorded per (account, generation).
    """

    __slots__ = ('_value', 'source', 'account_id', 'scope', 'generation')

    def __init__(self, value, source, account_id='local', scope='api', generation=0):
        if not isinstance(value, str) or not value.strip():
            raise CredentialError('Credential is empty (value withheld)')
        if source not in SOURCES:
            raise CredentialError('Unknown credential source')
        self._value = value.strip()
        self.source = source
        self.account_id = str(account_id or 'local')
        self.scope = scope or 'api'
        self.generation = int(generation)

    def reveal(self):
        """Return the secret. Call only where the key is placed into a request."""
        return self._value

    @property
    def masked(self):
        return mask(self._value)

    def __repr__(self):
        return (f'Credential(source={self.source!r}, account_id={self.account_id!r}, '
                f'scope={self.scope!r}, generation={self.generation}, value=***)')

    __str__ = __repr__

    def __eq__(self, other):
        return (isinstance(other, Credential) and self._value == other._value
                and self.source == other.source and self.generation == other.generation)

    def __hash__(self):
        return hash((self._value, self.source, self.generation))


def mask(value):
    """Render a credential for display without disclosing it."""
    if not value:
        return ''
    if len(value) <= 12:
        return 'sk-orca-…'
    return f'{value[:8]}…{value[-4:]}'


def looks_like_key(value):
    return bool(_KEY_RE.match((value or '').strip()))


def validate_key_text(value):
    """Lightweight format check for a pasted key. Returns the stripped value."""
    text = (value or '').strip()
    if not text:
        raise CredentialError('No API key provided (value withheld)')
    if len(text) > MAX_KEY_BYTES or any(character.isspace() for character in text):
        raise CredentialError('API key must be a single token (value withheld)')
    if not text.startswith(KEY_PREFIX):
        raise CredentialError(f'OrcaRouter API keys start with {KEY_PREFIX!r} (value withheld)')
    if not looks_like_key(text):
        raise CredentialError(
            'That looks like a truncated OrcaRouter key; paste the whole key '
            '(value withheld)')
    return text


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def store_dir(shared=None):
    """Reuse the rollout private store; it is the only secret store available."""
    return shared_root(shared)/'private'


def _write_secret(path, value):
    """Create or replace a secret atomically with owner-only permissions."""
    private_dir(path.parent)
    temporary = path.parent/f'.{path.name}.{os.getpid()}.tmp'
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(value)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_secret(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise CredentialError('Stored credential must be user-owned and mode 0600 or stricter')
    value = path.read_text().strip()
    return value or None


class CredentialStore:
    """Persists one OrcaRouter credential per acquisition path, plus account state.

    Both adapters write the same credential shape; the filename differs only so
    that "which path did the user choose" stays visible for support and logout.
    """

    def __init__(self, shared=None, directory=None):
        self.directory = Path(directory) if directory else store_dir(shared)

    # -- paths ------------------------------------------------------------
    def path_for(self, source):
        if source == 'oauth_pkce':
            return self.directory/OAUTH_KEY_FILENAME
        if source == 'api_key':
            return self.directory/API_KEY_FILENAME
        raise CredentialError('Unknown credential source')

    @property
    def accounts_path(self):
        return self.directory/ACCOUNTS_FILENAME

    # -- accounts ---------------------------------------------------------
    def _accounts(self):
        try:
            path = self.accounts_path
            if not path.is_file() or path.is_symlink():
                return {}
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            # A corrupt state file must not become an account loss: fall back to
            # "nothing known" rather than deleting the stored secret.
            return {}

    def generation(self, source, account_id='local'):
        record = self._accounts().get(f'{source}:{account_id}') or {}
        return int(record.get('generation', 0))

    def mark(self, source, account_id, *, scope='api', needs_reauth=False, generation=None):
        accounts = self._accounts()
        key = f'{source}:{account_id}'
        previous = accounts.get(key) or {}
        next_generation = int(previous.get('generation', 0)) + 1 if generation is None else int(generation)
        accounts[key] = dict(account=account_id, source=source, scope=scope,
                             generation=next_generation, needs_reauth=bool(needs_reauth),
                             updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        _write_secret(self.accounts_path, json.dumps(accounts, indent=1, sort_keys=True))
        return accounts[key]

    def state(self, source, account_id='local'):
        return self._accounts().get(f'{source}:{account_id}') or {}

    def needs_reauth(self, source, account_id='local'):
        return bool(self.state(source, account_id).get('needs_reauth'))

    # -- credentials ------------------------------------------------------
    def save(self, credential, account_id=None):
        """Persist a credential for the adapter that produced it."""
        account = account_id or credential.account_id
        _write_secret(self.path_for(credential.source), credential.reveal() + '\n')
        record = self.mark(credential.source, account, scope=credential.scope, needs_reauth=False)
        return Credential(credential.reveal(), credential.source, account,
                          credential.scope, record['generation'])

    def load(self, source, account_id=None):
        """Read a stored credential, or return ``None`` when nothing is stored."""
        value = _read_secret(self.path_for(source))
        if value is None:
            return None
        account = account_id or 'local'
        record = self.state(source, account)
        return Credential(value, source, account, record.get('scope', 'api'),
                          int(record.get('generation', 0)))

    def load_any(self):
        """Prefer a stored OAuth credential, then a stored API key."""
        for source in ('oauth_pkce', 'api_key'):
            credential = self.load(source)
            if credential is not None:
                return credential
        return None

    def clear(self, source=None):
        """Delete stored secrets. Returns the sources that were removed."""
        removed = []
        for candidate in (SOURCES if source is None else (source,)):
            path = self.path_for(candidate)
            if path.exists() or path.is_symlink():
                path.unlink()
                removed.append(candidate)
        if source is None:
            accounts = self.accounts_path
            if accounts.exists():
                accounts.unlink()
        return removed

    def mark_needs_reauth(self, credential):
        """Terminal classification for a rejected request.

        Applies to the exact account and the exact generation that made the
        rejected request. A credential that has since been replaced is left
        untouched, so a stale failure cannot poison a fresh login.
        """
        record = self.state(credential.source, credential.account_id)
        if int(record.get('generation', -1)) != credential.generation:
            return False
        self.mark(credential.source, credential.account_id, scope=credential.scope,
                  needs_reauth=True, generation=credential.generation)
        return True


def terminal_401(credential, store):
    """Handle a relay ``401``: mark reauth, never refresh, never delete the secret.

    OrcaRouter returns a durable key, not an access/refresh pair, so there is no
    refresh grant to run. The stored secret is deliberately kept until a
    replacement login succeeds.
    """
    store.mark_needs_reauth(credential)
    return dict(action='reauthenticate', source=credential.source,
                account=credential.account_id, generation=credential.generation,
                refresh_attempted=False,
                message='OrcaRouter rejected this key. Sign in again to issue a new one; '
                        'the existing key stays valid until you do.')
