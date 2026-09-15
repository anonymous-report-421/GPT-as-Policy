"""Private company-proxy storage and redacted connectivity checks; no model calls."""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


def validate_url(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or not parsed.port
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment
                or any(c.isspace() for c in value)):
            raise ValueError
    except ValueError:
        raise ValueError('Invalid proxy URL (contents withheld)') from None
    return parsed


def read_proxy(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Private proxy file missing or symlink')
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError('Private proxy file must be user-owned and mode0600 or stricter')
    value = path.read_text().strip()
    validate_url(value)
    return value


def import_zshrc(source, destination):
    # Parse commented literals only; never source or execute the user's shell rc.
    candidates = set()
    for line in Path(source).read_text().splitlines():
        if not line.lstrip().startswith('#') or 'proxy' not in line.lower():
            continue
        for value in re.findall(r'https?://[^\s\x22\x27;]+', line):
            parsed = validate_url(value)
            if parsed.username and parsed.password:
                candidates.add(value)
    if len(candidates) != 1:
        raise ValueError('Need exactly one distinct authenticated proxy in commented proxy lines')
    value = candidates.pop()
    path = Path(destination)
    if path.exists():
        if read_proxy(path) != value:
            raise ValueError('Existing private proxy differs; refusing to overwrite')
    else:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(value+'\n')
    print('Private company proxy saved; credentials withheld.')


def probe(value):
    validate_url(value)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': value}))
    try:
        with opener.open(urllib.request.Request(
                'https://chatgpt.com/backend-api/codex/models', method='HEAD'), timeout=20) as response:
            code = response.status
    except urllib.error.HTTPError as error:
        code = error.code
    except Exception as error:
        # Exceptions can embed credential-bearing URLs; never print them.
        category = 'proxy_authentication_required' if '407' in str(error) else 'connection_failed'
        raise RuntimeError(category+' (connection details withheld)') from None
    if code >= 500 or code == 407:
        raise RuntimeError('Proxy/server unavailable; HTTP '+str(code))
    print(json.dumps(dict(event='subscription_https_reachable', status=code,
                         model_credentials_sent=False, model_requests=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    a=sub.add_parser('import-zshrc'); a.add_argument('--source', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a=sub.add_parser('validate'); a.add_argument('--file', type=Path, required=True)
    a=sub.add_parser('probe'); a.add_argument('--file', type=Path)
    args=p.parse_args()
    if args.action=='import-zshrc':
        import_zshrc(args.source,args.output)
    elif args.action=='validate':
        read_proxy(args.file)
    else:
        probe(read_proxy(args.file) if args.file else os.environ['HTTPS_PROXY'])


if __name__=='__main__':
    main()
