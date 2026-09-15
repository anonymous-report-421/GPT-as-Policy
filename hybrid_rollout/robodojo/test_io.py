"""Atomic publication must not rely on cross-container advisory locks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from . import io


def test_concurrent_json_writers_have_independent_temporary_files(tmp_path, monkeypatch):
    target = tmp_path/'cases.json'
    workers = 15
    ready = threading.Barrier(workers)
    replaced = []
    replace = io.os.replace

    def simultaneous_replace(source, destination):
        # All writers finish writing before anyone publishes: deterministically
        # exposes a shared .tmp path even when a local flock would serialize it.
        ready.wait(timeout=10)
        replaced.append((Path(source), json.loads(Path(source).read_text())))
        replace(source, destination)

    monkeypatch.setattr(io.os, 'replace', simultaneous_replace)
    payloads = [dict(writer=i, data=str(i)*1000) for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda payload: io.write_json(target, payload), payloads))
    assert len({p for p, _ in replaced}) == workers
    assert sorted(v['writer'] for _, v in replaced) == list(range(workers))
    assert json.loads(target.read_text()) in payloads
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize('failure', ['serialization', 'fsync', 'replace'])
def test_failed_json_write_preserves_old_target_and_cleans_own_temp(tmp_path, monkeypatch, failure):
    target = tmp_path/'cases.json'
    target.write_text('{"old": true}')
    unrelated = tmp_path/'unrelated.tmp'
    unrelated.write_text('not ours')

    def fail(*args):
        raise OSError('injected I/O failure')

    value = {'new': float('nan')} if failure == 'serialization' else {'new': True}
    if failure != 'serialization':
        monkeypatch.setattr(io.os, failure, fail)
    with pytest.raises((ValueError, OSError)):
        io.write_json(target, value)
    assert json.loads(target.read_text()) == {'old': True}
    assert unrelated.read_text() == 'not ours'
    assert set(tmp_path.iterdir()) == {target, unrelated}


def test_json_unicode_and_new_file_permissions_match_normal_open(tmp_path):
    reference = tmp_path/'reference.json'
    reference.write_text('{}')
    target = tmp_path/'result.json'
    io.write_json(target, {'comment': '归位'})
    assert json.loads(target.read_text()) == {'comment': '归位'}
    assert target.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777
