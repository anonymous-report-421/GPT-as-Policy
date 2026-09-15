"""Build an offline, portable report from frozen results and latest annotations.

Only copies debug videos / makes exact-time derived clips. No credentials,
private reasoning, RPC logs, simulator imports, or original archive writes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHARED = Path('/mnt/rollout/robodojo_mixed_control')
DELIVERY = 'reports/robodojo_v3_hybrid_vs_gpt_only_100_20260913'
OFFICIAL = 'reports/robodojo_v3_score_comparison_20260913/sources/index-BQSAx3cX.js'
SOURCE_SHA = 'a200ce90aa19ee13ab83794f6b7885190dbfb3c87f0ebfe628156d4f8c7207d6'
OFFICIAL_SHA = '364aadb9777d369084348b4eca462d77f7f86712f02dbb1eba63b430eb4eea20'
MODEL_KEYS = {'DM0.5': 'OpenDM05', 'GalaxeaVLA (G0.5)': 'G05',
              'Xiaomi-Robotics-1': 'Xiaomi_Robotics_1', 'OpenWAM-α': 'OpenWAM-α',
              'Meituan-Robotics-0': 'Meituan_Robotics_0', 'Hy-Embodied-0.5-VLA': 'hy_vla',
              'Spatial Forcing': 'Pi_05_SF', 'Pi-05': 'Pi_05',
              'InternVLA-A1.5': 'InternVLA_A1_5', 'StarVLA-PI_v3': 'starVLA-PI_v3'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def checked_under(path, root):
    p = Path(path).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Source outside permitted archive root: {p}')
    return p


def parse_leaderboard(script):
    # Team is optional! Omitting that case silently drops Pi-05 from the ranking.
    pattern = r'\{model:"([^"]+)"(?:,team:"([^"]*)")?,generalizationStd:o\((.*?)average:o\(([\d.]+),([\d.]+)\)'
    rows = [dict(name=m[1], team=m[2] or '', overall_score=float(m[4]))
            for m in re.finditer(pattern, script)]
    if len(rows) < 10 or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Could not unambiguously parse the simulation leaderboard')
    top = sorted(rows, key=lambda r: (-r['overall_score'], r['name']))[:10]
    decoder = json.JSONDecoder()
    for r in top:
        key = MODEL_KEYS[r['name']]
        marker = json.dumps(key, ensure_ascii=False) + ':{'
        if script.count(marker) != 1:
            raise ValueError(f'Ambiguous task data for {r["name"]}')
        r['tasks'] = decoder.raw_decode('{' + script.split(marker, 1)[1])[0]
    return top


def public_case(row, video, archive):
    run = read(archive/'controller/run.json')
    history = read(archive/'controller/history.json')
    executed = [h for h in history if h.get('executed_steps', 0) > 0]
    if sum(h['executed_steps'] for h in executed) != row['control_steps']:
        raise ValueError('History does not cover the selected control steps')
    n_correct = sum(h['executed_steps'] for h in executed
                    if h['response']['mode'] in ('eef', 'edit')) if row['method'] == 'pi05_plus_gpt' else 0
    usage = read(archive/'controller/token_usage.json').get('usage', {}).get('total', {})
    case_id = row['case_id']
    short = 'mix' if row['method'] == 'pi05_plus_gpt' else 'gpt'
    name = short + '__' + case_id
    outcome = read(archive/'sim/evaluation_outcome.json')
    success = video['evaluation_success']
    status = 'success' if success else ('idle' if not video['native_complete'] else 'failure')
    seeds = row['evaluation_case']
    return dict(id=name, case_id=case_id, method=short, task=row['task'],
                variant=row['variant'], version='v3',
                seeds={k: seeds[k] for k in ('eval_seed', 'layout_id', 'reset_seed',
                       'simulator_initial_seed', 'policy_rng_seed')},
                status=status, success=success, score=row['native_score'],
                native_complete=video['native_complete'], steps=row['control_steps'],
                limit=row['shared_settings']['max_episode_steps'],
                dt=run['control_dt'], fps=video['fps'], frames=video['frames'],
                duration=video['frames']/video['fps'],
                instruction=run['instruction'], corrected_steps=n_correct,
                chunks=len(executed), tokens=usage,
                video=f'media/rollouts/{name}.mp4', poster=f'media/posters/{name}.jpg',
                result_sha256=row.get('outcome_sha256'),
                controller_version=video.get('controller_version', 'unrecorded'),
                video_ui='existing_debug_placeholder',
                native_outcome={k: outcome.get(k) for k in ('native_success', 'native_score', 'complete')})


def ffmpeg(args):
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                    '-threads', '1', *args], check=True, capture_output=True)


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=width,height,nb_frames,r_frame_rate',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)['streams'][0]


def materialize(job):
    kind, source, target, meta = job
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if kind == 'copy':
        shutil.copyfile(source, target)
        if sha(target) != meta['sha256']:
            raise ValueError(f'Video copy hash mismatch: {target.name}')
    elif kind == 'clip':
        # Decode and re-encode: stream-copy seeks are not annotation-exact.
        ffmpeg(['-ss', str(meta['start_frame']/meta['fps']), '-i', str(source),
                '-frames:v', str(meta['frames']), '-an', '-c:v', 'libx264',
                '-threads', '1', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(target)])
    if kind in ('copy', 'clip'):
        info = probe(target)
        if int(info['nb_frames']) != meta['frames']:
            raise ValueError(f'Frame count mismatch: {target.name}')
        poster = target.parents[1]/'posters'/(target.stem+'.jpg')
        poster.parent.mkdir(exist_ok=True)
        ffmpeg(['-i', str(target), '-frames:v', '1', '-vf', 'scale=768:-2',
                '-q:v', '3', str(poster)])
    elif kind == 'initial':
        ffmpeg(['-i', str(source), '-frames:v', '1', '-vf', 'crop=640:360:0:0',
                '-q:v', '2', str(target)])
    return target.name


def build(output, shared=SHARED, workers=3):
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; previous report builds are preserved')
    comparison_path = shared/DELIVERY/'results.json'
    official_path = shared/OFFICIAL
    if sha(comparison_path) != SOURCE_SHA or sha(official_path) != OFFICIAL_SHA:
        raise ValueError('Frozen report input changed; review before regenerating')
    content = read(HERE/'content.json')
    comparison = read(comparison_path)
    videos = read(shared/DELIVERY/'debug_videos/manifest.json')['videos']
    lookup = {(v['method'], v['case_id']): v for v in videos}
    cases, jobs, provenance = [], [], []
    by_archive = {}
    output.mkdir(parents=True)
    for method in ('hybrid', 'direct'):
        rows = comparison[method]['cases']
        if len(rows) != 50 or len({r['case_id'] for r in rows}) != 50:
            raise ValueError('Expected fifty unique selected cases per method')
        for row in rows:
            archive = checked_under(row['archive'], shared/'results')
            v = lookup[row['method'], row['case_id']]
            source = checked_under(shared/DELIVERY/'debug_videos'/v['file'], shared/DELIVERY)
            if sha(source) != v['sha256']:
                raise ValueError('Debug delivery hash mismatch')
            case = public_case(row, v, archive)
            cases.append(case); by_archive[str(archive)] = case
            jobs.append(('copy', source, output/case['video'], v))
            provenance.append(dict(id=case['id'], source_archive_relative=str(archive.relative_to(shared/'results')),
                                   source_video_sha256=v['sha256'], frames=v['frames'], fps=v['fps']))
    top = parse_leaderboard(official_path.read_text())
    task_rows = []
    for task, labels in content['tasks'].items():
        mix = [r for r in cases if r['task'] == task and r['method'] == 'mix']
        direct = [r for r in cases if r['task'] == task and r['method'] == 'gpt']
        if len(mix) != 5 or len(direct) != 5:
            raise ValueError('Task panel mismatch')
        weights = Counter(r['variant'] for r in mix)
        scores, rates = {}, {}
        for model in top:
            scores[model['name']] = rates[model['name']] = 0.
            for variant, n in weights.items():
                key = task + ('_random' if variant == 'random' else '')
                record = model['tasks'][key]
                for field, target in (('score', scores), ('successRate', rates)):
                    value = float(record[field])
                    if not math.isfinite(value) or not 0 <= value <= 100:
                        raise ValueError('Invalid official score')
                    target[model['name']] += value * n/5
        for label, rows in [('mix', mix), ('gpt', direct)]:
            present = [r['score']*100 for r in rows if r['score'] is not None]
            scores[label] = statistics.mean(present) if present else None
            rates[label] = sum(r['success'] for r in rows)*20
        first = mix[0]
        image = f'media/tasks/{task}.jpg'
        source = Path(next(r['archive'] for r in comparison['hybrid']['cases'] if r['case_id'] == first['case_id']))/'sim/sensors.mp4'
        jobs.append(('initial', source, output/image, {}))
        task_rows.append(dict(id=task, label=labels[0], category=labels[1], initial=labels[2],
                              goal=labels[3], image=image, scores=scores, rates=rates,
                              gpt_score_n=sum(r['score'] is not None for r in direct),
                              variants=dict(weights), limit=first['limit']))
    leaderboard = []
    for model in top:
        leaderboard.append(dict(name=model['name'], display='π0.5' if model['name']=='Pi-05' else model['name'],
            origin='official', overall_score=model['overall_score'],
            score=statistics.mean(t['scores'][model['name']] for t in task_rows),
            sr=statistics.mean(t['rates'][model['name']] for t in task_rows), score_n=None))
    totals = {}
    for method in ('mix', 'gpt'):
        rows = [r for r in cases if r['method']==method]
        tokens = Counter()
        for r in rows:
            tokens.update({k:v for k,v in r['tokens'].items() if isinstance(v, (int,float))})
        totals[method] = dict(steps=sum(r['steps'] for r in rows),
            corrected_steps=sum(r['corrected_steps'] for r in rows),
            chunks=sum(r['chunks'] for r in rows), tokens=dict(tokens),
            score_n=sum(r['score'] is not None for r in rows), successes=sum(r['success'] for r in rows))
        leaderboard.append(dict(name=method, display='π0.5 + GPT' if method=='mix' else 'GPT-only',
            origin='local', score=statistics.mean(r['score']*100 for r in rows if r['score'] is not None),
            sr=sum(r['success'] for r in rows)*2, score_n=totals[method]['score_n']))
    # Publish only public comments and IDs, never the annotation store or RPC traces.
    annotations = []
    clips = []
    clip_jobs = []
    featured = {(f['run_id'], f['clip']): f for f in content['featured']}
    for folder in sorted((shared/'annotations/video_review_v1').iterdir()):
        revisions = sorted(folder.glob('[0-9]*.json'))
        if not revisions:
            continue
        annotation = read(revisions[-1])
        archive = checked_under(annotation['archive'], shared/'results')
        run = read(archive/'controller/run.json')
        matched = by_archive.get(str(archive))
        if matched:
            matched['comment'] = annotation['comment']
        annotations.append(dict(id=annotation['run_id'], version=annotation['context_version'],
                                method=annotation['method'], case_id=annotation['case_id'],
                                comment=annotation['comment'], revision=annotation['revision']))
        for i, clip in enumerate(annotation['clips']):
            source = checked_under(archive/clip['video_file'], archive)
            if not source.is_file() or sha(source) != clip['video_sha256']:
                raise ValueError('Annotated video changed or disappeared')
            meta = read(source.parent/'manifest.json')
            fps = float(meta['fps'])
            # Half-open frame range; retain original mapping for later UI replacement.
            start = max(0, math.floor(clip['start']*fps))
            end = min(int(meta['frames']), math.ceil(clip['end']*fps))
            if end <= start:
                raise ValueError('Empty annotated clip')
            key = f'clip__{annotation["run_id"]}__{i}'
            item = dict(id=key, run_id=annotation['run_id'],
                case_id=annotation['case_id'], task=annotation['case_id'].split('__')[0],
                method='mix' if annotation['method']=='pi05_plus_gpt' else 'gpt',
                version=annotation['context_version'], instruction=run['instruction'],
                comment=clip['comment'].strip(), tags=clip['tags'],
                video=f'media/clips/{key}.mp4', poster=f'media/posters/{key}.jpg',
                start=start/fps, end=end/fps, start_frame=start, end_frame_exclusive=end,
                fps=fps, frames=end-start, duration=(end-start)/fps,
                original_annotation_start=clip['start'], original_annotation_end=clip['end'],
                source_video_sha256=clip['video_sha256'],
                video_ui='existing_debug_placeholder', paired_gallery_id=matched['id'] if matched else None)
            if (annotation['run_id'], i) in featured:
                item.update({k:v for k,v in featured[annotation['run_id'], i].items() if k not in ('run_id', 'clip')})
            clips.append(item)
            clip_jobs.append(('clip', source, output/item['video'], dict(start_frame=start, frames=end-start, fps=fps)))
    if set(featured) != {(c['run_id'], int(c['id'].rsplit('__',1)[1])) for c in clips if 'section' in c}:
        raise ValueError('A featured annotation is missing')
    data = dict(schema='embodied_policy.static_report.v1', language='zh-CN', date='2026-09-13',
                content=content, cases=cases, tasks=task_rows, models=leaderboard,
                top_four=[r['name'] for r in top[:4]], totals=totals, clips=clips,
                annotations=annotations, references=content['references'],
                source_sha256=SOURCE_SHA, official_sha256=OFFICIAL_SHA,
                video_note='旧版 debug video 占位；视频内任务标题与方向箭头待后续 UI 重渲染。',
                community_references_status=content.get('community_references_status','pending'), robolab_status='pending')
    for filename in ('gallery.html', 'gallery.css', 'gallery.js'):
        shutil.copyfile(HERE/'web'/filename, output/filename)
    (output/'data.js').write_text('window.REPORT_DATA = ' + json.dumps(data, ensure_ascii=False).replace('<','\\u003c') + ';\n')
    dump(output/'data.json', data)
    write_app_snapshot(data)
    dump(output/'provenance.json', dict(results_sha256=SOURCE_SHA, official_sha256=OFFICIAL_SHA,
        videos=provenance, old_data_untouched=True, reran_rollouts=False,
        clip_frame_policy='floor(start*fps) to ceil(end*fps), end exclusive; at most one frame boundary expansion',
        video_ui='existing_debug_placeholder', model_calls=0))
    with (output/'scores.csv').open('w', newline='') as f:
        fields=['task', 'model', 'score_100', 'success_rate_percent', 'score_n', 'origin']
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for t in task_rows:
            for m in leaderboard:
                writer.writerow(dict(task=t['id'], model=m['display'], score_100=t['scores'][m['name']],
                    success_rate_percent=t['rates'][m['name']],
                    score_n=(5 if m['name']=='mix' else t['gpt_score_n']) if m['origin']=='local' else '',
                    origin=m['origin']))
    print(json.dumps(dict(stage='data_ready', cases=len(cases), clips=len(clips), output=str(output))), flush=True)
    all_jobs=jobs+clip_jobs
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, name in enumerate(pool.map(materialize, all_jobs),1):
            if i%10==0 or i==len(all_jobs):
                print(json.dumps(dict(stage='media', done=i, total=len(all_jobs), file=name)), flush=True)
    dump(output/'BUILD_COMPLETE.json', dict(status='complete', cases=len(cases), clips=len(clips),
         source_sha256=SOURCE_SHA, data_sha256=sha(output/'data.json'),
         debug_placeholders=True, community_references=data['community_references_status'], robolab='pending'))
    print('BUILD COMPLETE', output, flush=True)


def write_app_snapshot(data):
    """Bind the report's editable narrative to immutable, inspectable evidence."""
    path = HERE/'app/src/data.json'
    identity = read(path)['id']
    source = dict(label='RoboDojo v3 · frozen evaluation and public leaderboard',
                  files=['data.json', 'provenance.json', 'scores.csv'],
                  results_sha256=SOURCE_SHA, official_sha256=OFFICIAL_SHA,
                  tables=[dict(name='RoboDojo official scores', href='https://robodojo-benchmark.com/')])
    queries = {}
    for name, rows in [('models', data['models']), ('tasks', data['tasks']),
                       ('cases', data['cases']), ('clips', data['clips'])]:
        queries[name] = dict(label=name, rows=rows, source=source)
    queries['intervention'] = dict(label='Executed control steps, selected 50 hybrid cases',
        rows=[dict(action='π0.5', steps=data['totals']['mix']['steps']-data['totals']['mix']['corrected_steps']),
              dict(action='GPT 修正', steps=data['totals']['mix']['corrected_steps'])], source=source)
    queries['costs'] = dict(label='Usage of selected attempts only',
        rows=[dict(method=k, **v) for k,v in data['totals'].items()], source=source)
    dump(path, dict(id=identity, title='GPT 6 Astra 具身策略评测', surface='report', status='draft',
        buildStatus='creating', generatedAt=datetime.now(timezone.utc).isoformat(), filters=[],
        report=dict(asOf='2026-09-13'), queries=queries,
        reportData={k:v for k,v in data.items() if k not in ('models','cases','tasks','clips')}))


def refresh_copy(output):
    """Refresh authored copy and gallery assets without re-copying videos."""
    data = read(output/'data.json')
    if data['source_sha256'] != SOURCE_SHA or data['official_sha256'] != OFFICIAL_SHA:
        raise ValueError('Not this frozen report')
    content = read(HERE/'content.json')
    if content['tasks'] != data['content']['tasks'] or content['featured'] != data['content']['featured']:
        raise ValueError('Task or clip selection changed; build a new artifact')
    data.update(content=content, references=content['references'],
                community_references_status=content.get('community_references_status','pending'))
    dump(output/'data.json',data)
    (output/'data.js').write_text('window.REPORT_DATA = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n')
    for f in ('gallery.html','gallery.css','gallery.js'):
        shutil.copyfile(HERE/'web'/f,output/f)
    write_app_snapshot(data)
    complete = output/'BUILD_COMPLETE.json'
    if complete.exists():
        old=read(complete);old.update(data_sha256=sha(output/'data.json'),community_references=data['community_references_status']);dump(complete,old)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shared', type=Path, default=SHARED)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--refresh-copy', action='store_true')
    args=parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('Use 1–4 offline media workers')
    if args.refresh_copy:
        refresh_copy(args.output.resolve())
    else:
        build(args.output.resolve(), args.shared.resolve(), args.workers)
