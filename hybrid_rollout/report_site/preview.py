"""Read-only static preview with byte ranges for accurate video seeking."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


def byte_range(value: str | None, size: int):
    if value is None:
        return 0, size - 1, 200
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
    if not match or not any(match.groups()) or size == 0:
        raise ValueError('Invalid range')
    first, last = match.groups()
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    else:
        if int(last) == 0:
            raise ValueError('Empty suffix range')
        start, end = max(0, size - int(last)), size - 1
    if start > end or start >= size:
        raise ValueError('Unsatisfiable range')
    return start, end, 206


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, directory: Path, **kwargs):
        self.root = directory.resolve()
        super().__init__(*args, **kwargs)

    def do_HEAD(self):
        self.send_file(head=True)

    def do_GET(self):
        self.send_file(head=False)

    def send_file(self, head):
        try:
            url_path = unquote(urlsplit(self.path).path)
            target = (self.root / (url_path.lstrip('/') or 'index.html')).resolve()
            if not target.is_relative_to(self.root) or not target.is_file():
                self.send_error(404)
                return
            with target.open('rb') as stream:
                size = target.stat().st_size
                try:
                    start, end, status = byte_range(self.headers.get('Range'), size)
                except ValueError:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                length = max(0, end - start + 1)
                self.send_response(status)
                self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(length))
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Content-Type-Options', 'nosniff')
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                if head:
                    return
                stream.seek(start)
                while length:
                    chunk = stream.read(min(length, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    length -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel one range when seeking to another part of the file.
            pass
        except (OSError, ValueError):
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--bind', choices=('127.0.0.1', '0.0.0.0'), default='127.0.0.1',
                        help='Use 0.0.0.0 only when LAN exposure is explicitly authorized.')
    args = parser.parse_args()
    if not (args.directory / 'index.html').is_file():
        parser.error('The directory must contain the report index.html')
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=args.directory))
    print(f'Report preview: http://{args.bind}:{args.port}/ (read-only, byte ranges enabled)', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
