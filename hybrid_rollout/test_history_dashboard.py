"""Read-only catalog and HTTP streaming tests, entirely synthetic."""
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from .history_dashboard import Catalog, handler
from .test_dashboard import put


def test_multiple_roots_cameras_numeric_order_and_invalid_result(tmp_path):
    roots = {name: tmp_path/name for name in ('robolab', 'robodojo')}
    for root in roots.values():
        put(root/'same/controller/run.json', {})
    run = roots['robodojo']/'same'
    put(run/'controller/result.json', {'complete': True, 'success': True})
    for number in ('999', '1000'):
        folder = run/'controller/observations'/number
        put(folder/'observation.json', {'step_id': int(number)})
        (folder/'cam_high.png').write_bytes(b'image')
    ledger = tmp_path/'ledger.json'
    put(ledger, {'records': [{'run_id': 'same', 'status': 'invalid_native_result', 'result': 'test audit'}]})
    catalog = Catalog(roots, ledger)
    assert set(catalog.runs()) == {'robolab/same', 'robodojo/same'}
    status = catalog.status('robodojo/same')
    assert status['observation_tick'] == 1000
    assert 'robodojo%2Fsame' in status['images']['cam_high']
    assert status['state'] == '无效终局（不计任务成功）'
    assert status['result']['success'] is True  # Raw evidence unchanged.
    for key in ('../ledger.json', 'robodojo/../ledger.json', 'missing/same'):
        with pytest.raises(FileNotFoundError):
            catalog.resolve(key)
    private = tmp_path/'private.mp4'
    private.write_bytes(b'private')
    folder = run/'controller/debug_video'
    folder.mkdir()
    (folder/'debug_rollout.mp4').symlink_to(private)
    assert catalog.videos(run) == []


def test_discovers_nested_cluster_archive(tmp_path):
    root = tmp_path/'robodojo'
    run = root/'panel/build_tower/seed_1000/replica_0/attempt_0'
    put(run/'controller/result.json', {'step_id': 0, 'complete': False})
    catalog = Catalog({'robodojo': root})
    key = 'robodojo/panel/build_tower/seed_1000/replica_0/attempt_0'
    assert key in catalog.runs()
    assert catalog.status(key)['run_id'] == 'panel/build_tower/seed_1000/replica_0/attempt_0'


def test_http_ranges_and_no_raw_rpc_or_arbitrary_files(tmp_path):
    run = tmp_path/'run'
    put(run/'controller/run.json', {})
    video = run/'controller/debug_video_ui_v3/debug_rollout.mp4'
    video.parent.mkdir()
    video.write_bytes(b'0123456789')
    catalog = Catalog({'test': tmp_path})
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(catalog))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    url = '/video?run=test%2Frun&file=controller%2Fdebug_video_ui_v3%2Fdebug_rollout.mp4'
    try:
        client.request('GET', '/api/runs')
        response = client.getresponse()
        assert response.status == 200 and json.loads(response.read()) == ['test/run']
        for byte_range, expected in [('bytes=2-5', b'2345'), ('bytes=-3', b'789'), ('bytes=8-', b'89')]:
            client.request('GET', url, headers={'Range': byte_range})
            response = client.getresponse()
            assert response.status == 206 and response.read() == expected
        client.request('GET', url, headers={'Range': 'bytes=99-'})
        response = client.getresponse()
        assert response.status == 416
        response.read()
        for path in ('/controller/codex_workspace/rpc_in.jsonl', '/video?run=test/run&file=../private', '/api/status?run=../private'):
            client.request('GET', path)
            response = client.getresponse()
            assert response.status == 404
            response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
