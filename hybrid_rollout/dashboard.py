"""Read-only, loopback-only artifact dashboard; never connects to policy/simulator."""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CAMERAS = ('main_rgb', 'wrist_rgb', 'cam_high', 'cam_left_wrist', 'cam_right_wrist')

PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboLab rollout</title><style>
body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1100px;margin:24px auto;padding:0 16px}
select,button{font:inherit;padding:6px} .images{display:flex;gap:12px}.images figure{margin:0;width:50%}
img{width:100%;background:#1f2937}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1f2937;padding:16px}
small{color:#9ca3af}progress{width:100%}#error{color:#fca5a5}</style>
<h2>RoboLab · 只读进度</h2><select id="runs"><option value="">自动显示最新实验</option></select>
<button id="refresh">刷新</button><p id="error"></p><p id="status">读取中…</p>
<small>每 5 秒读取已有记录；隐藏页面暂停刷新。不调用 Codex，不推进仿真。进度条是步数预算，不是任务完成度。</small>
<progress id="progress" value="0" max="1"></progress><p id="counts"></p>
<div class="images"><figure><figcaption>主视角</figcaption><img id="main_rgb" alt="尚无观测"></figure>
<figure><figcaption>腕部视角</figcaption><img id="wrist_rgb" alt="尚无观测"></figure></div>
<p id="age"></p><h3>最近决策（可能尚未执行）</h3><pre id="decision">暂无</pre>
<h3>最近活动 / 结果</h3><pre id="activity">暂无</pre>
<script>
const el=id=>document.getElementById(id);let busy=false;
async function refresh(){if(busy||document.hidden)return;busy=true;
try{const list=await fetch('/api/runs',{cache:'no-store'}).then(r=>r.json());
const chosen=el('runs').value;el('runs').replaceChildren(new Option('自动显示最新实验',''));
for(const name of list)el('runs').add(new Option(name,name));el('runs').value=chosen;
const response=await fetch('/api/status?run='+encodeURIComponent(chosen),{cache:'no-store'});
if(!response.ok)throw Error('读取状态失败：HTTP '+response.status);const s=await response.json();
el('error').textContent='';el('status').textContent=s.run_id+' · '+s.state+' · '+s.model+' / '+s.effort;
el('progress').max=s.max_steps||1;el('progress').value=s.step_id||0;
el('counts').textContent=`实际控制 ${s.step_id??'—'} / ${s.max_steps??'—'}；仿真 ${s.sim_seconds??'—'} 秒；已执行决策 ${s.decisions}；student ${s.student_steps} / edit ${s.edited_steps} / EEF ${s.recovery_steps}`;
el('age').textContent=`观测 tick：${s.observation_tick??'—'}；观测距今 ${s.observation_age_seconds??'—'} 秒。阻塞等待期间画面不变属正常现象。`;
for(const camera of ['main_rgb','wrist_rgb']){const img=el(camera),url=s.images[camera];
if(url&&img.getAttribute('src')!==url)img.src=url;else if(!url)img.removeAttribute('src');}
el('decision').textContent=JSON.stringify(s.decision,null,2)||'暂无';
el('activity').textContent=JSON.stringify({activity:s.activity,result:s.result,debug_video:s.debug_video},null,2);
}catch(e){el('error').textContent=String(e)}finally{busy=false}}
el('runs').onchange=refresh;el('refresh').onclick=refresh;document.addEventListener('visibilitychange',refresh);
setInterval(refresh,5000);refresh();</script></html>'''


class Artifacts:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        self.lock = threading.Lock()

    def runs(self):
        return sorted((p.name for p in self.root.glob('ordered_blocks_codex_*')
                       if p.is_dir() and not p.is_symlink()), reverse=True)

    def resolve_run(self, name):
        names = self.runs()
        if not name:
            # Directory creation time, not lexicographic tools vs agent naming.
            name = max(names, key=lambda n: (self.root/n).stat().st_mtime) if names else ''
        if name not in names:
            raise FileNotFoundError('Unknown run')
        return self.root/name

    def read(self, path, limit=2_000_000):
        path = path.resolve()
        if not path.is_relative_to(self.root):
            return None
        try:
            with path.open('rb') as stream:
                data = stream.read(limit + 1)
            return json.loads(data) if len(data) <= limit else None
        except (OSError, ValueError):
            return None  # A partial write is not a rollout failure.

    def latest_activity(self, run):
        path = run/'logs/controller.log'
        if not path.resolve().is_relative_to(self.root):
            return None
        try:
            with path.open('rb') as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size-32768))
                lines = stream.read(32768).splitlines()
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get('event') in (
                        'tool_start', 'tool_done', 'agent_activity', 'codex_decision', 'codex_transport_retry'):
                    return {k: row[k] for k in ('event', 'tool', 'phase', 'step_id', 'item_type', 'status', 'text') if k in row}
        except OSError:
            pass
        return None

    def image_path(self, run, observation, camera):
        if not re.fullmatch(r'\d{3,6}', observation) or camera not in CAMERAS:
            raise FileNotFoundError('Unknown image')
        path = (run/'controller/observations'/observation/(camera+'.png')).resolve()
        if not path.is_relative_to(run.resolve()) or not path.is_file():
            raise FileNotFoundError('Unknown image')
        return path

    def status(self, name=''):
        with self.lock:
            return self._status(name)

    def _status(self, name):
        run = self.resolve_run(name)
        run_id = run.relative_to(self.root).as_posix()
        now = time.monotonic()
        previous = self.cache.get(run_id)
        if previous and now-previous[0] < 2:
            return previous[1]
        controller = run/'controller'
        metadata = self.read(controller/'run.json') or {}
        worker = self.read(controller/'codex_workspace/worker.json') or {}
        progress = self.read(controller/'progress.json') or {}
        result = self.read(controller/'result.json')
        history = self.read(controller/'history.json') or []
        failure = (controller/'failure.json').exists()
        # A PID alone is insufficient: require the command to name this run.
        alive = False
        try:
            pid = int((run/'controller.pid').read_text())
            alive = str(controller).encode() in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        except (OSError, ValueError):
            pass
        state = ('已保存终局' if result and result.get('complete') else
                 '已停止 / 未完成' if result or failure or (run/'job_exit_status.txt').exists() else
                 '运行中（不代表成功）' if alive else '等待启动 / 未检测到控制进程')
        observations = sorted((p for p in (controller/'observations').glob('*')
                               if re.fullmatch(r'\d{3,6}', p.name)),
                              key=lambda p: int(p.name), reverse=True)
        obs, folder = {}, None
        for candidate in observations:
            obs = self.read(candidate/'observation.json')
            if isinstance(obs, dict):
                folder = candidate
                break
        obs = obs or {}
        responses = sorted((p for p in controller.glob('response_*.json')
                            if re.fullmatch(r'response_\d{3,6}', p.stem)),
                           key=lambda p: int(p.stem.split('_')[1]), reverse=True)
        decision = self.read(responses[0]) if responses else None
        images = {}
        if folder:
            for camera in CAMERAS:
                try:
                    image = self.image_path(run, folder.name, camera)
                    images[camera] = (f'/image?run={run_id}&obs={folder.name}&camera={camera}'
                                      f'&v={image.stat().st_mtime_ns}')
                except OSError:
                    pass
        counts = result or progress
        tick = counts.get('step_id', obs.get('step_id', 0))
        report = dict(run_id=run_id, state=state, model=worker.get('model', metadata.get('teacher_model', '未确认')),
            effort=worker.get('reasoning_effort', metadata.get('teacher_reasoning_effort', '未确认')),
            step_id=tick, max_steps=metadata.get('max_episode_steps'),
            sim_seconds=round(tick*metadata['control_dt'], 2) if 'control_dt' in metadata else None,
            decisions=len(history), student_steps=counts.get('student_steps', 0),
            edited_steps=counts.get('edited_steps', 0), recovery_steps=counts.get('recovery_steps', 0),
            observation_tick=obs.get('step_id'), images=images,
            observation_age_seconds=round(time.time()-(folder/'observation.json').stat().st_mtime) if folder else None,
            decision=decision, activity=self.latest_activity(run),
            result={k: result.get(k) for k in ('complete', 'success', 'terminated', 'truncated', 'reason')} if result else None,
            debug_video=self.read(controller/'debug_video/manifest.json'))
        self.cache = {run_id: (now, report)}
        return report


def handler(artifacts):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            arg = lambda key: query.get(key, [''])[0]
            try:
                if parsed.path == '/':
                    body, content_type = PAGE.encode(), 'text/html; charset=utf-8'
                elif parsed.path == '/api/runs':
                    body, content_type = json.dumps(artifacts.runs()).encode(), 'application/json'
                elif parsed.path == '/api/status':
                    body, content_type = json.dumps(artifacts.status(arg('run')), ensure_ascii=False).encode(), 'application/json'
                elif parsed.path == '/image':
                    path = artifacts.image_path(artifacts.resolve_run(arg('run')), arg('obs'), arg('camera'))
                    etag = f'"{path.stat().st_mtime_ns}"'
                    if self.headers.get('If-None-Match') == etag:
                        self.send_response(304)
                        self.end_headers()
                        return
                    with path.open('rb') as stream:
                        body = stream.read(8_000_001)
                    if len(body) > 8_000_000:
                        raise FileNotFoundError('Image exceeds limit')
                    content_type = 'image/png'
                else:
                    raise FileNotFoundError('Not exposed')
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'private, max-age=300' if parsed.path == '/image' else 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'")
                if parsed.path == '/image':
                    self.send_header('ETag', etag)
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError):
                self.send_error(404, 'Artifact unavailable')
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Artifacts(args.results_root)))
    print(f'Read-only rollout dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
