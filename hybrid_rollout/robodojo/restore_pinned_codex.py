"""Recover the approved 0.153.4 runtime from its verified local npm cache.

One-off dependency repair, not imported by workers. Prepare does not change the
missing compatibility entrypoint; activate requires a successful no-model tool
preflight and never overwrites an existing path. No auth or model requests.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

SHARED = Path('/mnt/rollout/robodojo_mixed_control')
INTEGRITY = 'x1EcwBlY3AObM1VTUHNM2AzAJQsyreGdagpF+qFiYi/Oa30VBktvvG0C6tLtCzqW6hjZNWkGZQWmeVk7MuJKWg=='
DIGEST = base64.b64decode(INTEGRITY).hex()
CACHE = Path('/home/runner/.npm/_cacache/content-v2/sha512') / DIGEST[:2] / DIGEST[2:4] / DIGEST[4:]
DEST = SHARED / 'tools/codex-0.153.4-linux-x64-recovered-20260913'
ALIAS = Path('/home/runner/.vscode-server/extensions/openai.chatgpt-26.903.71938-linux-x64/bin/linux-x86_64')
VENDOR = Path('vendor/x86_64-unknown-linux-musl')
FILES = {
    'package.json', 'README.md', str(VENDOR / 'codex-package.json'),
    str(VENDOR / 'bin/codex'), str(VENDOR / 'bin/codex-code-mode-host'),
    str(VENDOR / 'codex-path/rg'), str(VENDOR / 'codex-resources/bwrap'),
    str(VENDOR / 'codex-resources/zsh/bin/zsh'),
}


def digest(path, algorithm='sha256'):
    value = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare():
    if os.path.lexists(DEST) or os.path.lexists(ALIAS):
        raise ValueError('Recovery target or compatibility entry already exists')
    if digest(CACHE, 'sha512') != DIGEST:
        raise ValueError('Cached npm archive integrity mismatch')
    with tarfile.open(CACHE, 'r:gz') as archive:
        members = archive.getmembers()
        if {m.name for m in members} != {'package/' + n for n in FILES} or len(members) != len(FILES):
            raise ValueError('Unexpected archive members')
        if not all(m.isfile() and not (m.mode & 0o7000) for m in members):
            raise ValueError('Unsafe archive entry type or mode')
        package = json.load(archive.extractfile('package/package.json'))
        if package['name'] != '@openai/codex' or package['version'] != '0.153.4-linux-x64':
            raise ValueError('Cached package release differs from approved CLI')
        DEST.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='codex-0.153.4-staging-', dir=DEST.parent))
        for member in members:
            target = stage / member.name.removeprefix('package/')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    (stage / VENDOR / 'bin/rg').symlink_to('../codex-path/rg')
    probe = Path(tempfile.mkdtemp(prefix='codex-version-probe-', dir='/tmp'))
    env = dict(os.environ, CODEX_HOME=str(probe))
    result = subprocess.run([str(stage / VENDOR / 'bin/codex'), '--version'],
                            env=env, capture_output=True, text=True, timeout=15)
    if result.returncode or result.stdout.strip() != 'codex-cli 0.153.4':
        raise ValueError('Recovered CLI version check failed')
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_cache=str(CACHE), source_integrity='sha512-' + INTEGRITY,
        package_name=package['name'], package_version=package['version'],
        cli_version=result.stdout.strip(), model_calls=0, auth_files_modified=False,
        compatibility_alias=str(ALIAS), binary_dir=str(DEST / VENDOR / 'bin'),
        files={name: digest(stage / name) for name in sorted(FILES)})
    write_new(stage / 'recovery_manifest.json', receipt)
    stage.rename(DEST)
    print(json.dumps(dict(prepared=str(DEST), version=receipt['cli_version'],
                         binary_sha256=receipt['files'][str(VENDOR / 'bin/codex')], activated=False)))


def activate(preflight):
    receipt = json.loads((DEST / 'recovery_manifest.json').read_text())
    if set(receipt['files']) != FILES or receipt['source_integrity'] != 'sha512-' + INTEGRITY:
        raise ValueError('Recovery manifest differs')
    for name, expected in receipt['files'].items():
        if digest(DEST / name) != expected:
            raise ValueError('Recovered package changed')
    preflight = Path(preflight).resolve()
    if not preflight.is_relative_to(DEST) or preflight.is_symlink():
        raise ValueError('Preflight must belong to this recovery')
    check = json.loads(preflight.read_text())
    if check.get('model_turns') != 0 or check.get('simulator_connections') != 0 or check.get('command', {}).get('exitCode') != 0:
        raise ValueError('No successful no-model tool preflight')
    if os.path.lexists(ALIAS) or ALIAS.parent.is_symlink() or ALIAS.parent.parent.is_symlink():
        raise ValueError('Refusing to replace or traverse an existing compatibility entry')
    ALIAS.parent.mkdir(parents=True, exist_ok=True)
    ALIAS.symlink_to(DEST / VENDOR / 'bin', target_is_directory=True)
    write_new(DEST / 'activation.json', dict(activated_utc=datetime.now(timezone.utc).isoformat(),
        alias=str(ALIAS), target=str(ALIAS.resolve()), preflight=str(preflight),
        preflight_sha256=digest(preflight), existing_paths_overwritten=False,
        auth_files_modified=False, frozen_plan_modified=False, model_calls=0))
    print(json.dumps(dict(activated=True, alias=str(ALIAS), target=str(ALIAS.resolve()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'activate'])
    parser.add_argument('--preflight')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.preflight:
        activate(args.preflight)
    else:
        parser.error('activate requires --preflight')
