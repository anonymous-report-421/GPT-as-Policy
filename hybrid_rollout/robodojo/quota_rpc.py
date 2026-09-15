"""Account administration only: no model turns or managed refresh-token writes.

Use a disposable CLI home and external access-token auth. Never log raw RPCs,
credentials or stderr. The independently logged-in rollout remains auth owner.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

from .codex_backend.profiles import validate_credential, clean_login_env, CHATGPT_PROFILES


class AccountRPC:
    def __init__(self, process):
        self.process = process
        self.incoming = queue.Queue()
        self.sequence = 0
        def read():
            for line in process.stdout:
                try:
                    self.incoming.put(json.loads(line))
                except ValueError:
                    pass
            self.incoming.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()

    def request(self, method, params=None, timeout=45):
        self.sequence += 1
        identifier = self.sequence
        self.send(dict(id=identifier, method=method, params=params or {}))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Account administration RPC timed out')
            try:
                message = self.incoming.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError('Account administration RPC timed out') from None
            if message is None:
                raise RuntimeError('Account administration CLI exited')
            if message.get('id') == identifier and 'method' not in message:
                if 'error' in message:
                    # Error bodies can include request details: never disclose them.
                    raise RuntimeError('Account administration RPC rejected: ' + method)
                return message.get('result', {})
            if 'id' in message and 'method' in message:
                self.send(dict(id=message['id'], error=dict(code=-32601,
                    message='No token refresh or model tools available in quota monitor')))


@contextmanager
def account_rpc(profile, shared, codex, proxy_file):
    if profile not in CHATGPT_PROFILES:
        raise ValueError('Only managed subscription accounts are allowed')
    tokens = json.loads(validate_credential(profile, shared).read_text())['tokens']
    # Do not copy auth.json or pass a refresh token to this auxiliary CLI.
    private = Path(shared)/'private'
    # Shared-storage hook cleanup can race after a successful RPC. Preserve
    # private scratch leftovers instead of losing a valid quota/reset response.
    with tempfile.TemporaryDirectory(prefix='quota_cli_', dir=private,
                                     ignore_cleanup_errors=True) as directory:
        env = clean_login_env(Path(directory), proxy_file)
        env['RUST_LOG'] = 'off'
        proc = subprocess.Popen([str(codex), 'app-server', '--stdio',
            '-c', 'model_provider="openai"', '-c', 'features.fast_mode=false',
            '-c', 'cli_auth_credentials_store="ephemeral"',
            '-c', 'analytics.enabled=false'], cwd=directory, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
        try:
            rpc = AccountRPC(proc)
            rpc.request('initialize', dict(clientInfo=dict(name='robodojo-quota-monitor',
                version='1'), capabilities=dict(experimentalApi=True)))
            rpc.send(dict(method='initialized', params={}))
            rpc.request('account/login/start', dict(type='chatgptAuthTokens',
                accessToken=tokens['access_token'], chatgptAccountId=tokens['account_id']))
            yield rpc
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            for stream in (proc.stdin, proc.stdout):
                stream.close()


def codex_limits(result):
    """Do not mistake unrelated Spark quota for the GPT-6 subscription bucket."""
    bucket = (result.get('rateLimitsByLimitId') or {}).get('codex')
    fallback = result.get('rateLimits') or {}
    if bucket is None and fallback.get('limitId') in (None, 'codex'):
        bucket = fallback
    if not bucket:
        return None
    windows = [bucket[k] for k in ('primary', 'secondary')
               if isinstance(bucket.get(k), dict)
               and isinstance(bucket[k].get('usedPercent'), (float, int))]
    if not windows:
        return None
    return dict(remaining_percent=min(100 - w['usedPercent'] for w in windows),
        windows=windows, reached_type=bucket.get('rateLimitReachedType'),
        reset_credits_available=(result.get('rateLimitResetCredits') or {}).get('availableCount'))
