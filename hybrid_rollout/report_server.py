"""Serve only the static report directory, with seekable video and a review link."""
import argparse
import json
import mimetypes
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from .annotation_dashboard import contained_file, handler


def report_handler(root, review_port=8767, download_file=None):
    root = Path(root).resolve()
    download_file = Path(download_file).resolve() if download_file else None
    base = handler(None, None)
    media_sources = "'self'"
    gallery_manifest = root / 'robolab-gallery-manifest.json'
    if gallery_manifest.is_file() and json.loads(gallery_manifest.read_text()).get('video_hosting') == 'github-release':
        # Only video sources expand; scripts, frames and file access stay restricted.
        media_sources += ' https://github.com https://release-assets.githubusercontent.com'

    class ReportHandler(base):
        def headers_out(self, status, kind, size, extra=None):
            # The upstream static build embeds its scripts, styles and fonts.
            policy = f"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self' data:; media-src {media_sources}; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
            extra = dict(extra or {})
            if urlsplit(self.path).path == '/download/gpt-dagger-report.zip' and status in (200, 206):
                extra['Content-Disposition'] = 'attachment; filename="gpt-dagger-report.zip"'
            super().headers_out(status, kind, size, {
                'Content-Security-Policy': policy, **extra})

        def do_GET(self):
            if not self.trusted_host():
                self.respond(403, {'error': 'Use the server IP or a localhost forward'})
                return
            request = urlsplit(self.path)
            if request.path == '/download/gpt-dagger-report.zip':
                if not download_file or not download_file.is_file():
                    self.respond(404, {'error': 'Offline report is not configured'})
                    return
                try:
                    self.stream_video(download_file, 'application/zip')
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if request.path == '/gallery':
                host = urlsplit('http://' + self.headers['Host']).hostname
                host = f'[{host}]' if ':' in host else host
                target = urlunsplit(('http', f'{host}:{review_port}', '/', request.query, ''))
                self.headers_out(302, 'text/plain', 0, {'Location': target})
                return
            file = root / (unquote(request.path).lstrip('/') or 'index.html')
            if not contained_file(root, file):
                self.respond(404, {'error': 'Not exposed'})
                return
            try:
                self.stream_video(file, mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            self.respond(405, {'error': 'This report is read-only'})

    return ReportHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--review-port', type=int, default=8767)
    parser.add_argument('--download-file', type=Path, help='One explicit offline ZIP, served as a browser download')
    args = parser.parse_args()
    if not (args.root / 'index.html').is_file():
        parser.error('Report root must contain index.html')
    if args.download_file and not args.download_file.is_file():
        parser.error('Download ZIP must already exist')
    server = ThreadingHTTPServer((args.host, args.port), report_handler(args.root, args.review_port, args.download_file))
    server.daemon_threads = True
    server.serve_forever()


if __name__ == '__main__':
    main()
