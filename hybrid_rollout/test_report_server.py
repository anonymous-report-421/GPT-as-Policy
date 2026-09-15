from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import zipfile

from hybrid_rollout.report_server import report_handler


def test_browser_zip_download_range_and_file_boundary(tmp_path):
    root = tmp_path / 'report'
    root.mkdir()
    (root / 'index.html').write_text('<html>Report</html>')
    archive = tmp_path / 'offline.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('index.html', '<html>Offline</html>')
    body = archive.read_bytes()
    (tmp_path / 'private.txt').write_text('not served')
    server = ThreadingHTTPServer(('127.0.0.1', 0), report_handler(root, download_file=archive))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    try:
        path = '/download/gpt-dagger-report.zip'
        client.request('GET', path)
        response = client.getresponse()
        assert response.status == 200
        assert response.getheader('Content-Type') == 'application/zip'
        assert 'attachment;' in response.getheader('Content-Disposition')
        assert response.read() == body
        client.request('HEAD', path)
        response = client.getresponse()
        assert response.status == 200 and int(response.getheader('Content-Length')) == len(body)
        assert response.read() == b''
        client.request('GET', path, headers={'Range': 'bytes=0-9'})
        response = client.getresponse()
        assert response.status == 206 and response.read() == body[:10]
        assert 'attachment;' in response.getheader('Content-Disposition')
        for bad in ['/download/private.txt', '/download/../private.txt', '/../private.txt']:
            client.request('GET', bad)
            response = client.getresponse()
            assert response.status == 404
            response.read()
        client.request('GET', '/')
        response = client.getresponse()
        assert response.getheader('Content-Type') == 'text/html'
        assert "media-src 'self';" in response.getheader('Content-Security-Policy')
        assert response.getheader('Content-Disposition') is None
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_release_video_csp_only_expands_media_hosts(tmp_path):
    (tmp_path/'index.html').write_text('<html>Report</html>')
    (tmp_path/'robolab-gallery-manifest.json').write_text(json.dumps({'video_hosting':'github-release'}))
    server = ThreadingHTTPServer(('127.0.0.1', 0), report_handler(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection('127.0.0.1', server.server_port, timeout=5)
    try:
        client.request('GET', '/')
        response = client.getresponse()
        policy = response.getheader('Content-Security-Policy')
        assert "media-src 'self' https://github.com https://release-assets.githubusercontent.com;" in policy
        assert "script-src 'self' 'unsafe-inline';" in policy
        assert "frame-ancestors 'none';" in policy
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()
