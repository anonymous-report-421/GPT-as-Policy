"""Local artifact/video review server. Writes only separate, versioned annotations.

No simulator, policy, Codex, scheduler, or credential imports. Loopback by default;
explicit all-interface binding supports trusted LAN access. Python standard library only.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit


DEFAULT_BASE = Path('/mnt/rollout/robodojo_mixed_control')
UI_ROOT = Path(__file__).with_name('annotation_ui')
ID_RE = re.compile(r'[0-9a-f]{24}')
STATES = {'success': '原生成功', 'failure': '原生失败', 'invalid': '无效场景',
          'interrupted': '中断/未完成', 'unverified': '历史终局/待核验',
          'unfinished': '未收尾/状态待核验'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, limit=4_000_000):
    try:
        with path.open('rb') as stream:
            data = stream.read(limit + 1)
        return json.loads(data) if len(data) <= limit else None
    except (OSError, ValueError):
        return None


def contained_file(root, path):
    """Reject symlinks (including intermediate directories), not just traversal."""
    try:
        rel = path.relative_to(root)
        if any((root/Path(*rel.parts[:n])).is_symlink() for n in range(1, len(rel.parts)+1)):
            return False
        return path.resolve().is_relative_to(root) and path.is_file()
    except (OSError, ValueError):
        return False


def number(value, default=None):
    return value if type(value) in (int, float) and math.isfinite(value) else default


def mapping(value):
    return value if isinstance(value, dict) else {}


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.lock = threading.RLock()
        self.paths = {}
        self.rows = []
        self.scanned_at = 0

    def read(self, path, limit=4_000_000):
        return read_json(path, limit) if contained_file(self.root, path) else None

    def scan(self, force=False):
        with self.lock:
            if not force and time.monotonic()-self.scanned_at < 20:
                return self.rows
            rows, paths = [], {}
            # Stop at each archive: never traverse thousands of NPZs/observations.
            for folder, dirs, _ in os.walk(self.root, followlinks=False):
                parent = Path(folder)
                dirs[:] = [d for d in dirs if not (parent/d).is_symlink() and not d.startswith('.')]
                if 'controller' not in dirs and not ('logs' in dirs and parent.name.startswith('attempt_')):
                    if len(parent.relative_to(self.root).parts) > 10:
                        dirs[:] = []
                    continue
                relative = parent.relative_to(self.root).as_posix()
                key = hashlib.sha256(relative.encode()).hexdigest()[:24]
                if key in paths:
                    raise RuntimeError('Duplicate archive identity')
                paths[key] = parent
                rows.append(self.describe(key, parent))
                dirs[:] = []
            self.paths = paths
            self.rows = sorted(rows, key=lambda r: (r['updated_at'], r['path']), reverse=True)
            self.scanned_at = time.monotonic()
            return self.rows

    def resolve(self, key):
        if not ID_RE.fullmatch(key):
            raise FileNotFoundError('Unknown rollout')
        with self.lock:
            path = self.paths.get(key)
        if path is None:
            self.scan()
            with self.lock:
                path = self.paths.get(key)
        if path is None or path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise FileNotFoundError('Unknown rollout')
        # Verify every parent after lookup as directories may have changed.
        rel = path.relative_to(self.root)
        if any((self.root/Path(*rel.parts[:n])).is_symlink() for n in range(1, len(rel.parts)+1)):
            raise FileNotFoundError('Linked archive is not exposed')
        return path

    def videos(self, run):
        rows = []
        for path in sorted(run.glob('controller/debug_video*/debug_rollout.mp4'), reverse=True):
            if not contained_file(self.root, path) or path.stat().st_size == 0:
                continue
            manifest = mapping(self.read(path.parent/'manifest.json'))
            if manifest.get('status') not in ('complete', 'completed', 'verified') and not self.read(run/'controller/result.json'):
                continue  # Do not stream a video currently being finalized.
            rel = path.relative_to(run).as_posix()
            key = hashlib.sha256(rel.encode()).hexdigest()[:24]
            fps = number(manifest.get('fps'))
            frames = number(manifest.get('frames'))
            duration = number(manifest.get('duration_seconds'))
            frame_mapping = str(manifest.get('frame_mapping', ''))
            step_mapping = (fps is not None and fps > 0 and 'initial frame 0' in frame_mapping
                            and 'action t-1' in frame_mapping and manifest.get('reading_pause_frames', 0) == 0
                            and manifest.get('playback_speed', 1) == 1)
            rows.append(dict(key=key, name=path.parent.name, file=rel, bytes=path.stat().st_size,
                             fps=fps, frames=frames, duration=duration, step_mapping=step_mapping,
                             sha256=manifest.get('video_sha256'), frame_mapping=frame_mapping))
        return rows

    def describe(self, key, run):
        meta = mapping(self.read(run/'controller/run.json'))
        result = mapping(self.read(run/'controller/result.json'))
        outcome = mapping(self.read(run/'sim/evaluation_outcome.json'))
        progress = mapping(self.read(run/'controller/progress.json'))
        identity = mapping(outcome.get('evaluation_case') or result.get('evaluation_case') or meta.get('evaluation_case'))
        if outcome.get('status') == 'invalid_native_layout':
            state = 'invalid'
        elif outcome.get('complete') is True and outcome.get('valid_for_success_rate') is True and type(outcome.get('native_success')) is bool:
            state = 'success' if outcome['native_success'] else 'failure'
        elif result.get('complete'):
            state = 'unverified'
        elif result or contained_file(self.root, run/'controller/failure.json'):
            state = 'interrupted'
        else:
            state = 'unfinished'
        updated = max((p.stat().st_mtime for p in (run/'controller/result.json', run/'controller/progress.json',
                      run/'sim/evaluation_outcome.json', run/'controller/run.json') if contained_file(self.root, p)), default=run.stat().st_mtime)
        relative = run.relative_to(self.root).as_posix()
        match = re.search(r'(?:^|/)attempt_(\d+)(?:/|$)', relative)
        seeds = {k: identity.get(k, meta.get('seed') if k == 'eval_seed' else None)
                 for k in ('eval_seed', 'layout_id', 'reset_seed', 'simulator_initial_seed', 'policy_rng_seed')}
        return dict(id=key, path=relative, experiment=relative.split('/')[0],
                    task=identity.get('task') or meta.get('task') or 'unknown',
                    variant=identity.get('variant', 'unknown'), case_id=identity.get('case_id'),
                    context_version=result.get('context_version') or meta.get('context_version') or 'unknown',
                    method=meta.get('evaluation_method') or meta.get('method') or ('pi05_plus_gpt' if meta.get('student_backend') else 'unknown'),
                    attempt=int(match[1]) if match else None, seeds=seeds, status=state,
                    native_score=number(outcome.get('native_score')) if state in ('success', 'failure') else None,
                    step=progress.get('step_id', result.get('step_id')), max_steps=meta.get('max_episode_steps'),
                    student_steps=result.get('student_steps', progress.get('student_steps')),
                    edited_steps=result.get('edited_steps', progress.get('edited_steps')),
                    recovery_steps=result.get('recovery_steps', progress.get('recovery_steps')),
                    model=meta.get('teacher_model'), effort=meta.get('teacher_reasoning_effort'),
                    instruction=meta.get('instruction', ''), updated_at=updated,
                    videos=self.videos(run), outcome_status=outcome.get('status'))

    def detail(self, key):
        run = self.resolve(key)
        row = self.describe(key, run)
        history = self.read(run/'controller/history.json', 12_000_000) or []
        decisions = []
        if isinstance(history, list):
            for item in history[:10000]:
                if not isinstance(item, dict):
                    continue
                response = mapping(item.get('response'))
                decisions.append(dict(index=item.get('decision'), start_step=item.get('start_tick'),
                                      end_step=item.get('end_tick'), mode=response.get('mode'),
                                      reason=response.get('reason', '')))
        row['decisions'] = decisions
        row['absolute_path'] = str(run)
        return row

    def video_path(self, key, video):
        run = self.resolve(key)
        row = next((r for r in self.videos(run) if r['key'] == video), None)
        if row is None:
            raise FileNotFoundError('Unknown debug video')
        return run/row['file']


class Conflict(Exception):
    pass


class AnnotationStore:
    def __init__(self, root, results_root):
        self.root = Path(root).resolve()
        results = Path(results_root).resolve()
        if self.root.is_relative_to(results) or results.is_relative_to(self.root):
            raise ValueError('Annotations must be separate from the results tree')
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()

    @staticmethod
    def empty(key):
        return dict(run_id=key, revision=0, reviewed=False, selected=False, tags=[], comment='', clips=[])

    def directory(self, key):
        if not ID_RE.fullmatch(key):
            raise ValueError('Invalid rollout identity')
        folder = self.root/key
        if folder.is_symlink():
            raise ValueError('Linked annotation directory rejected')
        return folder

    def revisions(self, key):
        folder = self.directory(key)
        return sorted(p for p in folder.glob('*.json') if re.fullmatch(r'\d{8}\.json', p.name))

    def get(self, key):
        files = self.revisions(key)
        if not files:
            return self.empty(key)
        if not contained_file(self.root, files[-1]):
            raise ValueError('Unsafe annotation revision')
        data = read_json(files[-1], 300_000)
        if not isinstance(data, dict) or data.get('run_id') != key:
            raise ValueError('Unreadable annotation revision; refusing to overwrite')
        return data

    def history(self, key):
        paths = self.revisions(key)[-100:]
        if not all(contained_file(self.root, p) for p in paths):
            raise ValueError('Unsafe annotation revision')
        return [read_json(p, 300_000) for p in reversed(paths)]

    @staticmethod
    def validate(data, videos):
        if not isinstance(data, dict) or set(data)-{'revision', 'reviewed', 'selected', 'tags', 'comment', 'clips'}:
            raise ValueError('Unknown annotation fields')
        if type(data.get('revision')) is not int or data['revision'] < 0:
            raise ValueError('An integer base revision is required')
        for key in ('reviewed', 'selected'):
            if type(data.get(key)) is not bool:
                raise ValueError(f'{key} must be boolean')
        def text(value, max_length):
            if not isinstance(value, str) or len(value) > max_length:
                raise ValueError('Text is missing or too long')
            return value
        def tags(value):
            if not isinstance(value, list) or len(value) > 30 or not all(isinstance(v, str) for v in value):
                raise ValueError('At most 30 tags')
            return list(dict.fromkeys(text(v, 80).strip() for v in value if isinstance(v, str) and v.strip()))
        clean = dict(reviewed=data['reviewed'], selected=data['selected'],
                     tags=tags(data.get('tags')), comment=text(data.get('comment'), 20000), clips=[])
        clips = data.get('clips')
        if not isinstance(clips, list) or len(clips) > 100:
            raise ValueError('At most 100 clips')
        for clip in clips:
            if not isinstance(clip, dict) or set(clip)-{'video_key', 'start', 'end', 'comment', 'tags'}:
                raise ValueError('Invalid clip fields')
            video = next((v for v in videos if v['key'] == clip.get('video_key')), None)
            start, end = number(clip.get('start')), number(clip.get('end'))
            if video is None or start is None or end is None or not 0 <= start <= end <= 1_000_000:
                raise ValueError('Invalid clip time/video')
            if video['duration'] is not None and end > video['duration'] + 0.05:
                raise ValueError('Clip exceeds video duration')
            item = dict(video_key=video['key'], start=start, end=end, comment=text(clip.get('comment'), 10000),
                        tags=tags(clip.get('tags', [])), video_file=video['file'], video_sha256=video['sha256'])
            if video['step_mapping']:
                maximum = video['frames']-1 if video['frames'] else float('inf')
                item.update(start_step=int(min(math.floor(start*video['fps']+1e-6), maximum)),
                            end_step=int(min(math.floor(end*video['fps']+1e-6), maximum)))
            clean['clips'].append(item)
        return clean

    def save(self, key, data, row):
        clean = self.validate(data, row['videos'])
        with self.lock:
            folder = self.directory(key)
            folder.mkdir(mode=0o700, exist_ok=True)
            lockpath = folder/'.lock'
            if lockpath.is_symlink():
                raise ValueError('Unsafe lock path')
            with lockpath.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                old = self.get(key)
                if data['revision'] != old['revision']:
                    raise Conflict('Another tab saved a newer revision. Reload before saving.')
                record = dict(clean, schema='rollout.annotation.v1', run_id=key,
                              revision=old['revision']+1, updated_utc=utc(),
                              archive=str(row['absolute_path']), case_id=row['case_id'],
                              context_version=row['context_version'], method=row['method'], seeds=row['seeds'])
                target = folder/f"{record['revision']:08d}.json"
                if target.exists():
                    raise Conflict('Revision already exists')
                # Immutable revisions are the source of truth; no fragile latest pointer.
                fd, name = tempfile.mkstemp(prefix='.pending-', dir=folder)
                try:
                    with os.fdopen(fd, 'w') as stream:
                        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
                        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                    os.replace(name, target)
                    directory_fd = os.open(folder, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if os.path.exists(name):
                        os.unlink(name)
                return record


def export(catalog, annotations, selected_only, kind, version=''):
    rows = []
    for r in catalog.scan():
        if version and r['context_version'] != version:
            continue
        a = annotations.get(r['id'])
        if selected_only and not a['selected']:
            continue
        rows.append(dict(r, annotation=a))
    if kind == 'json':
        return json.dumps(dict(exported_utc=utc(), source=str(catalog.root),
                               unit='individual attempt; not a merged success-rate table', rows=rows),
                          ensure_ascii=False, indent=2).encode(), 'application/json'
    if kind == 'csv':
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['run_id', 'task', 'variant', 'case_id', 'context_version', 'method', 'attempt',
                         'status', 'native_score', 'seeds', 'selected', 'reviewed', 'tags', 'comment', 'clips', 'archive'])
        def cell(value):
            value = str(value)
            # Prevent formula execution when opening user comments in Excel.
            return "'"+value if value.lstrip().startswith(('=', '+', '-', '@')) else value
        for r in rows:
            a = r['annotation']
            writer.writerow([cell(v) for v in [r['id'], r['task'], r['variant'], r['case_id'], r['context_version'],
                r['method'], r['attempt'], r['status'], r['native_score'], json.dumps(r['seeds']), a['selected'],
                a['reviewed'], ', '.join(a['tags']), a['comment'], json.dumps(a['clips'], ensure_ascii=False), str(catalog.root/r['path'])]])
        return ('\ufeff'+stream.getvalue()).encode(), 'text/csv; charset=utf-8'
    if kind != 'md':
        raise ValueError('Unknown export format')
    lines = ['# Rollout behavior review', '', f'Exported: {utc()}', '',
             'Individual attempts; annotations do not override native evaluation results.', '']
    for r in rows:
        a = r['annotation']
        lines += [f"## {r['task']} · {r['context_version']} · {r['variant']} · attempt {r['attempt']}", '',
                  f"Case: {r['case_id']} | Native: {r['status']} | Score: {r['native_score']}",
                  f"Seeds: {json.dumps(r['seeds'])}", f"Archive: {catalog.root/r['path']}",
                  f"Tags: {', '.join(a['tags'])}", '', a['comment'], '']
        for clip in a['clips']:
            lines += [f"- {clip['start']:.3f}–{clip['end']:.3f}s · {clip['video_file']}: {clip['comment']}"]
        lines.append('')
    return '\n'.join(lines).encode(), 'text/markdown; charset=utf-8'


def handler(catalog, annotations):
    csrf = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Never log query contents, comments, or paths to authentication files.
            print(json.dumps(dict(utc=utc(), method=self.command, route=urlsplit(self.path).path,
                                  message=format % args if format.startswith('code') else 'request')), flush=True)

        def headers_out(self, status, kind, size, extra=None):
            self.send_response(status)
            headers = {'Content-Type': kind, 'Content-Length': str(size), 'Cache-Control': 'no-store',
                       'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
                       'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"}
            for k, v in dict(headers, **(extra or {})).items():
                self.send_header(k, v)
            self.end_headers()

        def respond(self, status, data):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.headers_out(status, 'application/json; charset=utf-8', len(body))
            if self.command != 'HEAD':
                self.wfile.write(body)

        def trusted_host(self):
            try:
                parsed = urlsplit('http://'+self.headers.get('Host', ''))
                # A wildcard listener still knows the connection's actual local IP.
                # Accept that IP, not arbitrary domains (DNS rebinding protection).
                local_ip = self.connection.getsockname()[0]
                parsed.port  # Reject malformed ports too.
                return (not parsed.username and not parsed.password and not parsed.path
                        and not parsed.query and not parsed.fragment
                        and parsed.hostname in ('127.0.0.1', 'localhost', '::1', local_ip))
            except ValueError:
                return False

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self.trusted_host():
                self.respond(403, dict(error='Use the server IP address or a localhost port forward'))
                return
            parsed = urlsplit(self.path); query = parse_qs(parsed.query)
            arg = lambda key, default='': query.get(key, [default])[0]
            try:
                if parsed.path == '/api/runs':
                    rows = [dict(r, annotation=annotations.get(r['id'])) for r in catalog.scan()]
                    self.respond(200, dict(rows=rows, states=STATES, csrf=csrf,
                                          annotation_root=str(annotations.root), source=str(catalog.root)))
                elif parsed.path == '/api/run':
                    self.respond(200, dict(catalog.detail(arg('id')), annotation=annotations.get(arg('id'))))
                elif parsed.path == '/api/revisions':
                    catalog.resolve(arg('id'))
                    self.respond(200, dict(revisions=annotations.history(arg('id'))))
                elif parsed.path == '/video':
                    self.stream_video(catalog.video_path(arg('id'), arg('video')))
                elif parsed.path == '/api/export':
                    kind = arg('format', 'json')
                    body, mime = export(catalog, annotations, arg('selected', '1') == '1', kind, arg('version'))
                    self.headers_out(200, mime, len(body), {'Content-Disposition': f'attachment; filename="rollout_review.{kind}"'})
                    if self.command != 'HEAD': self.wfile.write(body)
                elif parsed.path in ('/', '/app.js', '/style.css'):
                    name, mime = {'/': ('index.html', 'text/html; charset=utf-8'),
                                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                                  '/style.css': ('style.css', 'text/css; charset=utf-8')}[parsed.path]
                    body = (UI_ROOT/name).read_bytes()
                    self.headers_out(200, mime, len(body))
                    if self.command != 'HEAD': self.wfile.write(body)
                else:
                    self.respond(404, dict(error='Not exposed'))
            except (BrokenPipeError, ConnectionResetError):
                pass
            except FileNotFoundError:
                self.respond(404, dict(error='Unknown or unavailable artifact'))
            except ValueError as error:
                self.respond(400, dict(error=str(error)))
            except OSError:
                self.respond(503, dict(error='Shared storage unavailable; retry without changing annotations'))

        def stream_video(self, path, mime='video/mp4'):
            with path.open('rb') as stream:
                size = os.fstat(stream.fileno()).st_size
                start, end, status = 0, size-1, 200
                value = self.headers.get('Range')
                if value:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
                    if not match or not any(match.groups()):
                        self.headers_out(416, mime, 0, {'Content-Range': f'bytes */{size}'})
                        return
                    first, last = match.groups()
                    start = int(first) if first else max(0, size-int(last))
                    end = min(size-1, int(last)) if first and last else size-1
                    if start > end or start >= size:
                        self.headers_out(416, mime, 0, {'Content-Range': f'bytes */{size}'})
                        return
                    status = 206
                extra = {'Accept-Ranges': 'bytes'}
                if status == 206: extra['Content-Range'] = f'bytes {start}-{end}/{size}'
                self.headers_out(status, mime, end-start+1, extra)
                if self.command == 'HEAD': return
                stream.seek(start); remaining = end-start+1
                while remaining > 0:
                    chunk = stream.read(min(256*1024, remaining))
                    if not chunk: break
                    self.wfile.write(chunk); remaining -= len(chunk)

        def do_POST(self):
            if not self.trusted_host() or not secrets.compare_digest(self.headers.get('X-Review-Token', ''), csrf):
                self.respond(403, dict(error='Invalid write token; reload this local dashboard'))
                return
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                self.respond(403, dict(error='Cross-origin writes are not allowed'))
                return
            if urlsplit(self.path).path != '/api/annotation':
                self.respond(404, dict(error='Not exposed')); return
            try:
                size = int(self.headers.get('Content-Length', '-1'))
                if not 0 <= size <= 256_000 or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Expected JSON body up to 256 KB')
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict) or set(data) != {'id', 'annotation'}:
                    raise ValueError('Expected rollout id and annotation')
                row = catalog.detail(data['id'])
                saved = annotations.save(data['id'], data['annotation'], row)
                self.respond(200, saved)
            except Conflict as error:
                self.respond(409, dict(error=str(error)))
            except (ValueError, TypeError, KeyError) as error:
                self.respond(400, dict(error=str(error)))
            except FileNotFoundError:
                self.respond(404, dict(error='Unknown rollout'))
            except OSError:
                self.respond(503, dict(error='Save unavailable; keep draft and reload to verify latest revision'))

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=DEFAULT_BASE/'results')
    parser.add_argument('--annotation-root', type=Path, default=DEFAULT_BASE/'annotations/video_review_v1')
    parser.add_argument('--host', choices=('127.0.0.1', '0.0.0.0'), default='127.0.0.1',
                        help='0.0.0.0 allows LAN access; no authentication, trusted network only')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    catalog = Catalog(args.results_root)
    annotations = AnnotationStore(args.annotation_root, catalog.root)
    catalog.scan()
    server = ThreadingHTTPServer((args.host, args.port), handler(catalog, annotations))
    server.daemon_threads = True
    print(json.dumps(dict(utc=utc(), url=f'http://{args.host}:{server.server_port}',
                          bind_host=args.host, authenticated=False,
                          results=str(catalog.root), annotations=str(annotations.root),
                          initial_attempts=len(catalog.rows)), ensure_ascii=False), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
