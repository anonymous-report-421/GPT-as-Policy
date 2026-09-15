"""Read-only historical artifact browser. No model/simulator imports or connections."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

from .dashboard import Artifacts, CAMERAS


class HistoricalArtifacts(Artifacts):
    def runs(self):
        runs = []
        for marker in self.root.rglob('controller'):
            run = marker.parent
            relative = run.relative_to(self.root)
            linked = any((self.root/Path(*relative.parts[:index])).is_symlink()
                         for index in range(1, len(relative.parts) + 1))
            if marker.is_dir() and not linked:
                runs.append(relative.as_posix())
        for marker in self.root.rglob('logs'):
            run = marker.parent
            name = run.relative_to(self.root).as_posix()
            if name not in runs and marker.is_dir() and not run.is_symlink():
                runs.append(name)
        return sorted(runs, reverse=True)

    def image_path(self, run, observation, camera):
        try:
            return super().image_path(run, observation, camera)
        except FileNotFoundError:
            if not re.fullmatch(r'\d{3,6}', observation) or camera not in CAMERAS:
                raise
            path = (run/'controller/images'/f'{observation}_{camera}.png').resolve()
            if not path.is_relative_to(run.resolve()) or not path.is_file():
                raise FileNotFoundError('Unknown image')
            return path


class Catalog:
    def __init__(self, roots, ledger=None):
        self.roots = {name: HistoricalArtifacts(path) for name, path in roots.items()}
        self.ledger = Path(ledger) if ledger else None

    def runs(self):
        entries = [(f'{label}/{name}', (art.root/name).stat().st_mtime)
                   for label, art in self.roots.items() for name in art.runs()]
        return [key for key, _ in sorted(entries, key=lambda pair: pair[1], reverse=True)]

    def resolve(self, key):
        if not key:
            key = next(iter(self.runs()), '')
        label, _, name = key.partition('/')
        if label not in self.roots or not name:
            raise FileNotFoundError('Unknown run')
        art = self.roots[label]
        return key, art, art.resolve_run(name)

    def annotation(self, name):
        if self.ledger:
            try:
                with self.ledger.open('rb') as stream:
                    data = json.loads(stream.read(2_000_000))
                return next((r for r in data.get('records', []) if r.get('run_id') == name), {})
            except (OSError, ValueError, AttributeError):
                pass
        return {}

    def videos(self, run):
        # Only finished known video artifacts; never expose arbitrary paths from JSON.
        paths = sorted(run.glob('controller/debug_video*/debug_rollout.mp4'), reverse=True)
        if (run/'controller/result.json').exists() or (run/'job_exit_status.txt').exists():
            paths += [run/'sim/sensors.mp4']
        return [str(p.relative_to(run)) for p in paths
                if p.is_file() and p.resolve().is_relative_to(run.resolve())]

    def video_path(self, key, video):
        _, _, run = self.resolve(key)
        if video not in self.videos(run):
            raise FileNotFoundError('Unknown video')
        return run/video

    def status(self, key):
        key, art, run = self.resolve(key)
        run_name = run.relative_to(art.root).as_posix()
        report = dict(art.status(run_name))
        report['key'] = key
        legacy_images = sorted((run/'controller/images').glob('*_main_rgb.png'), reverse=True)
        if not report['images'] and legacy_images:
            observation = legacy_images[0].name.split('_')[0]
            report['images'] = {}
            for camera in CAMERAS:
                try:
                    image = art.image_path(run, observation, camera)
                    report['images'][camera] = '/image?'+urlencode({
                        'obs': observation, 'camera': camera, 'v': image.stat().st_mtime_ns})
                except OSError:
                    pass
        report['images'] = {camera: '/image?'+urlencode({
            **{k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}, 'run': key})
            for camera, url in report['images'].items()}
        report['videos'] = [{'name': name, 'url': '/video?'+urlencode({'run': key, 'file': name})}
                            for name in self.videos(run)]
        annotation = self.annotation(run.name)
        report['audit_status'] = annotation.get('status')
        report['audit_note'] = annotation.get('result') if annotation.get('status') != 'in_progress' else None
        if annotation.get('status') == 'invalid_native_result':
            report['state'] = '无效终局（不计任务成功）'
        # Old mailbox archives may lack the new observation/result schema.
        report['legacy_layout'] = not (run/'controller/observations').is_dir()
        return report


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rollout 历史与进度</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{font:15px system-ui,sans-serif;background:#0c1220;color:#e4eaf5;margin:0 auto;max-width:1440px;padding:28px}
h1{font-size:26px;margin:0 0 8px}h2{font-size:17px;margin:0 0 14px}small,.muted{color:#9aabc5;line-height:1.6}
.bar{display:flex;gap:12px;align-items:center;margin:22px 0;flex-wrap:wrap}select{flex:1;min-width:230px}select,button{font:inherit;background:#19253a;color:inherit;border:1px solid #354660;border-radius:8px;padding:10px;cursor:pointer}
.card{background:#141f32;border:1px solid #24334b;border-radius:12px;padding:18px;margin:16px 0}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.value{display:block;font-size:24px;margin-top:5px}
#state{color:#8dd7ff}progress{width:100%;height:12px;accent-color:#58b6ef;margin:18px 0 4px}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}figure{margin:0}figcaption{color:#9aabc5;margin:8px 0}img{width:100%;border-radius:8px;background:#080e19}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}#reason{white-space:pre-wrap;line-height:1.8}#mode{display:inline-block;padding:6px 12px;border-radius:6px;background:#29435e;margin-bottom:12px}#mode.edit{background:#704419;color:#ffe0a0}#error,#audit{color:#ffc98e}a{color:#8ed5ff}video{width:100%;max-height:720px;background:#000}details{margin-top:16px}@media(max-width:700px){body{padding:14px}.stats{grid-template-columns:repeat(2,1fr)}}
</style><h1>Rollout 历史与进度</h1><small>只读已有文件 · 每 5 秒刷新 · 隐藏页面暂停 · 不调用模型或仿真 · 不改变 Codex 上下文</small>
<div class="bar"><select id="runs"><option value="">自动显示最新实验</option></select><button id="refresh">刷新</button><small id="total"></small></div>
<p id="error"></p><div class="card"><h2 id="title">读取中…</h2><p id="state"></p><small id="identity"></small>
<progress id="budget" value="0" max="1"></progress><small>控制步数预算使用量，并非任务完成度</small>
<div class="stats"><div><small>实际控制 / 上限</small><span class="value" id="steps">—</span></div><div><small>已执行决策</small><span class="value" id="decisions">—</span></div><div><small>Student / Edit / EEF</small><span class="value" id="controls">—</span></div><div><small>仿真时间（非等待时间）</small><span class="value" id="sim">—</span></div></div><p id="audit"></p></div>
<div class="images" id="images"></div><p class="muted" id="age"></p>
<div class="card"><h2>最近公开决策 · 可能尚未执行</h2><div id="mode"></div><div id="reason">暂无记录</div><details><summary>完整决策字段</summary><pre id="decision"></pre></details></div>
<div class="card"><h2>活动与结果</h2><p id="activity"></p><pre id="result"></pre></div>
<div class="card"><h2>已有视频</h2><small>只在点击播放时读取；不在线渲染视频。旧实验没有新格式观测时，可优先查看视频。</small><div id="videos"></div><video id="player" controls preload="none" hidden></video></div>
<script>
const el=id=>document.getElementById(id),labels={main_rgb:'主视角',wrist_rgb:'腕部视角',cam_high:'主视角',cam_left_wrist:'左腕视角',cam_right_wrist:'右腕视角'};
let busy=false,current='',videoKey='';el('runs').value='';
async function get(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);return r.json()}
async function refresh(){if(busy||document.hidden)return;busy=true;try{
const list=await get('/api/runs'),chosen=el('runs').value;el('runs').replaceChildren(new Option('自动显示最新实验',''));
for(const key of list)el('runs').add(new Option(key,key));el('runs').value=list.includes(chosen)?chosen:'';el('total').textContent=list.length+' 个实验';
const s=await get('/api/status?run='+encodeURIComponent(el('runs').value));el('error').textContent='';
el('title').textContent=s.key;el('state').textContent=s.state;el('identity').textContent=s.model+' / '+s.effort;
el('steps').textContent=(s.step_id??'—')+' / '+(s.max_steps??'—');el('decisions').textContent=s.decisions;
el('controls').textContent=[s.student_steps,s.edited_steps,s.recovery_steps].join(' / ');el('sim').textContent=(s.sim_seconds??'—')+' s';
el('budget').max=s.max_steps||1;el('budget').value=s.step_id||0;el('audit').textContent=s.audit_note||'';
if(current!==s.key){el('images').replaceChildren();el('player').pause();el('player').removeAttribute('src');el('player').hidden=true;current=s.key}
for(const camera of Object.keys(labels)){const url=s.images[camera];let fig=el('camera-'+camera);if(!url){if(fig)fig.remove();continue}if(!fig){fig=document.createElement('figure');fig.id='camera-'+camera;const caption=document.createElement('figcaption');caption.textContent=labels[camera];fig.append(caption,document.createElement('img'));el('images').append(fig)}const img=fig.querySelector('img');if(img.getAttribute('src')!==url)img.src=url}
el('age').textContent=s.legacy_layout?'历史格式：部分实时字段不可用；保留原记录，未重新执行。':`观测步 ${s.observation_tick??'—'} · 距今 ${s.observation_age_seconds??'—'} 秒。Codex 阻塞决策期间画面不变是正常现象。`;
const d=s.decision||{},modified=['eef','edit'].includes(d.mode);el('mode').textContent=modified?'GPT 修正 · '+d.mode:d.mode==='student'?'沿用 pi05 动作':d.mode||'尚无决策';el('mode').className=modified?'edit':'';
el('reason').textContent=d.reason||'暂无公开说明';el('decision').textContent=JSON.stringify(d,null,2);
const a=s.activity||{};el('activity').textContent=[a.event,a.tool||a.item_type,a.phase||a.status,a.step_id!=null?'步 '+a.step_id:''].filter(Boolean).join(' · ')||'暂无活动记录';
el('result').textContent=s.result?JSON.stringify(s.result,null,2):'尚无终局（不等于成功或失败）';
const signature=s.key+JSON.stringify(s.videos);if(signature!==videoKey){el('videos').replaceChildren();for(const v of s.videos){const p=document.createElement('p'),b=document.createElement('button'),link=document.createElement('a');b.textContent='播放 '+v.name;b.onclick=()=>{el('player').src=v.url;el('player').hidden=false;el('player').play().catch(()=>{})};link.textContent=' 新窗口';link.href=v.url;link.target='_blank';link.rel='noopener';p.append(b,link);el('videos').append(p)}if(!s.videos.length)el('videos').textContent='尚无视频文件；通常在结束后导出。';videoKey=signature}
}catch(e){el('error').textContent='暂时无法读取：'+e}finally{busy=false}}
el('runs').onchange=refresh;el('refresh').onclick=refresh;document.addEventListener('visibilitychange',refresh);setInterval(refresh,5000);refresh();
</script></html>'''


def handler(catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_artifact_headers(self, status, kind, length, extra=None):
            self.send_response(status)
            for key, value in {'Content-Type': kind, 'Content-Length': str(length),
                               'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                               'Content-Security-Policy': "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'",
                               **(extra or {})}.items():
                self.send_header(key, value)
            self.end_headers()

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            arg = lambda key: query.get(key, [''])[0]
            try:
                if parsed.path == '/video':
                    path = catalog.video_path(arg('run'), arg('file'))
                    with path.open('rb') as stream:
                        size = stream.seek(0, 2)
                        start, end, status = 0, size-1, 200
                        extra = {'Accept-Ranges': 'bytes'}
                        byte_range = self.headers.get('Range')
                        if byte_range:
                            match = re.fullmatch(r'bytes=(\d*)-(\d*)', byte_range)
                            if not match or not any(match.groups()):
                                self.headers_out(416, size)
                                return
                            first, last = match.groups()
                            start = int(first) if first else max(0, size-int(last))
                            end = min(size-1, int(last)) if first and last else size-1
                            if start > end or start >= size:
                                self.headers_out(416, size)
                                return
                            status = 206
                            extra['Content-Range'] = f'bytes {start}-{end}/{size}'
                        self.send_artifact_headers(status, 'video/mp4', end-start+1, extra)
                        stream.seek(start)
                        remaining = end-start+1
                        while remaining > 0:
                            data = stream.read(min(256*1024, remaining))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                    return
                if parsed.path == '/':
                    body, kind = PAGE.encode(), 'text/html; charset=utf-8'
                elif parsed.path == '/api/runs':
                    body, kind = json.dumps(catalog.runs()).encode(), 'application/json'
                elif parsed.path == '/api/status':
                    body, kind = json.dumps(catalog.status(arg('run')), ensure_ascii=False).encode(), 'application/json'
                elif parsed.path == '/image':
                    _, art, run = catalog.resolve(arg('run'))
                    path = art.image_path(run, arg('obs'), arg('camera'))
                    with path.open('rb') as stream:
                        body = stream.read(8_000_001)
                    if len(body) > 8_000_000:
                        raise FileNotFoundError('Image exceeds limit')
                    kind = 'image/png'
                else:
                    raise FileNotFoundError('Not exposed')
                self.send_artifact_headers(200, kind, len(body))
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (OSError, ValueError):
                self.send_error(404, 'Artifact unavailable')

        def headers_out(self, status, size):
            self.send_artifact_headers(status, 'text/plain', 0, {'Content-Range': f'bytes */{size}'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    roots = {}
    for entry in args.root:
        label, separator, path = entry.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in roots or not Path(path).is_dir():
            parser.error('Each root must be a unique LABEL=existing-directory')
        roots[label] = Path(path)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Catalog(roots, args.ledger)))
    print(f'Read-only history dashboard: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == '__main__':
    main()
