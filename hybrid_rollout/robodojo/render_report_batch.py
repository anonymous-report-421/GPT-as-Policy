"""Resumable, model-free report-video re-rendering into a fresh media tree.

Read-only inputs: frozen case results, existing report clip ranges, original
annotations and archived observations. Never overwrite archived data/videos.
Only media-replacements.json's files are intended for publication; the ledger,
render manifests and source snapshots are internal reproducibility evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import uuid

from .io import sha256, write_json


REPO=Path(__file__).resolve().parents[2]
SHARED=Path('/mnt/rollout/robodojo_mixed_control')
DEFAULT_PYTHON=SHARED/'sim-venv/bin/python'
ROBODOJO_SOURCE=SHARED/'src/RoboDojo'
SCHEMA='robodojo.report_debug_batch.v2'
RESULTS_SHA256='a200ce90aa19ee13ab83794f6b7885190dbfb3c87f0ebfe628156d4f8c7207d6'
LEASE_FD=None


def read(path): return json.loads(Path(path).read_text())
def now(): return datetime.now(timezone.utc).isoformat()
def checked(path,root):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError(f'Path outside permitted root: {path}')
    return path


def relative_media(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0]!='media':
        raise ValueError(f'Invalid relative public media path: {value}')
    return path.as_posix()


def verify_global_inputs(plan):
    """Fail closed if any frozen selection, annotation or result changed."""
    inputs=plan.get('input_sha256')
    if not inputs:raise ValueError('Frozen plan has no global source hashes')
    for path,digest in inputs.items():
        if not Path(path).is_file() or sha256(path)!=digest:
            raise ValueError(f'Frozen global input changed: {path}')
    return len(inputs)


def record_event(root,event,**fields):
    row=dict(time=now(),event=event,**fields)
    with (root/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps(row,ensure_ascii=False),flush=True)


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-threads','1','-select_streams','v:0',
        '-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration',
        '-of','json',str(path)],check=True,capture_output=True,text=True)
    stream=json.loads(result.stdout)['streams'][0]
    numerator,denominator=map(int,stream['r_frame_rate'].split('/'))
    return dict(width=int(stream['width']),height=int(stream['height']),frames=int(stream['nb_read_frames']),
        fps=numerator/denominator,duration=float(stream['duration']))


def prepare(output,site,snapshot,selection,shared,approved_sample):
    if (output/'plan.json').exists():
        raise FileExistsError('Plan already exists; use --phase run to resume the frozen plan')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a fresh batch output directory')
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    raw=read(site/'data.json'); app=read(snapshot); chosen=read(selection)
    if len(raw['cases'])!=100 or len(raw['clips'])!=43 or len(chosen)!=21:
        raise ValueError('Unexpected reviewed case/clip counts')
    if {r['id'] for r in raw['cases']}!={r['id'] for r in app['queries']['cases']['rows']}:
        raise ValueError('Case identities differ between report and authoring snapshot')
    selected={row['id'] for row in chosen}
    clips=raw['clips']+app['queries']['featured_case']['rows']
    if len(clips)!=44 or not selected<={r['id'] for r in clips}:
        raise ValueError('Narrative selection does not match existing clip evidence')
    provenance={row['id']:row for row in read(site/'provenance.json')['videos']}
    delivery=shared/'reports/robodojo_v3_hybrid_vs_gpt_only_100_20260913'
    exported=read(delivery/'debug_videos/manifest.json')['videos']
    export_map={('mix' if x['method']=='pi05_plus_gpt' else 'gpt')+'__'+x['case_id']:x for x in exported}
    if sha256(delivery/'results.json')!=RESULTS_SHA256:
        raise ValueError('Frozen final evaluation results have changed')
    comparison=read(delivery/'results.json')
    native_rows={('mix' if x['method']=='pi05_plus_gpt' else 'gpt')+'__'+x['case_id']:x
        for group in ('hybrid','direct') for x in comparison[group]['cases']}
    annotations={a['id']:a for a in raw['annotations']}
    inputs={str(site/'data.json'):sha256(site/'data.json'),str(site/'provenance.json'):sha256(site/'provenance.json'),
        str(snapshot):sha256(snapshot),str(selection):sha256(selection),
        str(delivery/'results.json'):sha256(delivery/'results.json')}
    runs={};media=[]

    def add_run(archive,source,source_sha,frames,fps):
        archive=checked(archive,shared/'results');source=checked(source,shared)
        if sha256(source)!=source_sha: raise ValueError(f'Original video SHA mismatch: {source}')
        key=hashlib.sha256(str(archive).encode()).hexdigest()[:20]
        if key not in runs:
            run=read(archive/'controller/run.json');result_path=archive/'controller/result.json'
            adjudicated=(archive/'evaluation_adjudication.json').is_file()
            adjudication=None
            if adjudicated:
                from .idle_timeout_policy import verified_adjudication
                adjudication=verified_adjudication(archive)
            if not result_path.is_file() and not adjudicated:
                raise FileNotFoundError(f'Unadjudicated source lacks controller result: {archive}')
            result=read(result_path) if result_path.is_file() else {}
            required=[archive/'sim/sensors.mp4',archive/'sim/resolved_config.json',archive/'robodojo_head.txt']
            if not all(p.is_file() for p in required): raise FileNotFoundError(f'Missing replay inputs: {archive}')
            runs[key]=dict(id=key,archive=str(archive),frames=frames,fps=fps,
                method='gpt' if run.get('evaluation_method')=='gpt_only' else 'mix',
                instruction=run['instruction'],original_result=result,original_videos=[],
                inputs={str(p):sha256(p) for p in [*required,archive/'controller/run.json',
                    archive/'controller/history.json',result_path,archive/'sim/evaluation_outcome.json'] if p.is_file()},
                original_result_missing=not result_path.is_file(),adjudicated_idle=adjudicated,
                adjudicated_control_steps=adjudication['control_steps'] if adjudication else None)
            if runs[key]['adjudicated_idle']:
                p=archive/'evaluation_adjudication.json';runs[key]['inputs'][str(p)]=sha256(p)
        elif runs[key]['frames']!=frames or runs[key]['fps']!=fps:
            raise ValueError('Same archive has inconsistent source-video time geometry')
        item=dict(path=str(source),sha256=source_sha)
        if item not in runs[key]['original_videos']:runs[key]['original_videos'].append(item)
        return key

    for row in raw['cases']:
        p=provenance[row['id']];v=export_map[row['id']]
        archive=shared/'results'/p['source_archive_relative']
        if str(archive)!=native_rows[row['id']]['archive']:
            raise ValueError('Report archive differs from final selected evaluation case')
        source=delivery/'debug_videos'/v['file']
        key=add_run(archive,source,v['sha256'],row['frames'],row['fps'])
        for field in ('success','native_complete','score'):
            expected=v[{'success':'evaluation_success','native_complete':'native_complete','score':'native_score'}[field]]
            if row[field]!=expected:raise ValueError(f'Report result changed: {row["id"]}/{field}')
        media.append(dict(id=row['id'],kind='rollout',run=key,video=row['video'],poster=row['poster'],
            frames=row['frames'],fps=row['fps'],expected_evaluation_success=row['success'],
            expected_native_complete=row['native_complete'],expected_score=row['score'],
            original_site_sha256=sha256(site/row['video'])))
    for row in clips:
        if 'run_id' in row:
            a=annotations[row['run_id']]
            annotation_path=shared/'annotations/video_review_v1'/row['run_id']/f'{a["revision"]:08d}.json'
            annotation=read(annotation_path);inputs[str(annotation_path)]=sha256(annotation_path)
            item=annotation['clips'][int(row['id'].rsplit('__',1)[1])]
            archive=Path(annotation['archive']);source=checked(archive/item['video_file'],archive)
            if item['video_sha256']!=row['source_video_sha256']:
                raise ValueError('Reviewed clip provenance differs from its frozen annotation revision')
        else:
            if row['id']!='imitate_sorting_sequence_a5_frames_493_1309':
                raise ValueError('Unknown historical narrative-only source')
            archive=(shared/'results/robodojo_pool7_vla_reserved_20260911_01_stable_c1_a5/'
                'imitate_sorting_sequence/standard/eval_seed_0/layout_0/replica_2/attempt_5')
            source=archive/'controller/debug_video/debug_rollout.mp4'
        original=read(source.parent/'manifest.json')
        key=add_run(archive,source,row['source_video_sha256'],original['frames'],float(original['fps']))
        start,end=row['start_frame'],row['end_frame_exclusive']
        if not (0<=start<end<=runs[key]['frames']) or end-start!=row['frames']:
            raise ValueError('Invalid frozen clip frame range')
        media.append(dict(id=row['id'],kind='clip',run=key,video=row['video'],poster=row['poster'],
            frames=row['frames'],fps=row['fps'],start_frame=start,end_frame_exclusive=end,
            narrative=row['id'] in selected,original_site_sha256=sha256(site/row['video'])))
    for row in media:
        row['video']=relative_media(row['video']);row['poster']=relative_media(row['poster'])
    if len({row['id'] for row in media})!=144 or len({r['video'] for r in media})!=144:
        raise ValueError('Expected exactly 144 unique media identities')
    # Freeze only model-free modules, not credentials, controller logs or a whole
    # environment. The old approved preview may be reused as an explicit input.
    files=['hybrid_rollout/__init__.py','hybrid_rollout/robodojo/__init__.py',
        'hybrid_rollout/robodojo/io.py','hybrid_rollout/robodojo/idle_timeout_policy.py',
        'hybrid_rollout/robodojo/robodojo_server/__init__.py',
        *['hybrid_rollout/robodojo/robodojo_server/'+name for name in
          ('debug_recorder.py','video_panel.py','video_panel_paper.py','video_projection.py',
           'kinematics.py','action_edit_kinematics.py')],
        'hybrid_rollout/robodojo/render_report_batch.py']
    source_hashes={}
    for relative in files:
        source=REPO/relative;target=output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        source_hashes[relative]=sha256(target)
    from .robodojo_server.debug_recorder import load_font
    _,font=load_font(22);font=Path(font)
    target=output/'source/hybrid_rollout/assets/fonts'/font.name
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(font,target)
    source_hashes[str(target.relative_to(output/'source'))]=sha256(target)
    sample=None
    if approved_sample:
        manifest=read(approved_sample/'manifest.json')
        if manifest['status']!='completed' or manifest['layout']!='paper_prompt_arrows_sample_v1':
            raise ValueError('Unrecognized approved sample')
        sample=dict(output=str(approved_sample.resolve()),manifest_sha256=sha256(approved_sample/'manifest.json'),
            video_sha256=manifest['video_sha256'],archive=str(Path(manifest['source_video']).parent.parent))
        if sha256(approved_sample/'debug_rollout.mp4')!=sample['video_sha256']:raise ValueError('Sample changed')
    write_json(output/'source-manifest.json',source_hashes)
    plan=dict(schema=SCHEMA,created=now(),site=str(site),runs=list(runs.values()),media=media,
        input_sha256=inputs,renderer_source_sha256=source_hashes,font=str(target),approved_sample=sample,
        robodojo_source=str(shared/'src/RoboDojo'),
        expected=dict(rollouts=100,gallery_clips=43,narrative_clips=21,distinct_clip_files=44,media_files=288),
        native_metrics_unchanged=True,model_calls=0,simulation_connections=0,old_data_mutated=False)
    write_json(output/'plan.json',plan)
    write_json(output/'ledger.json',dict(schema=SCHEMA,plan_sha256=sha256(output/'plan.json'),
        created=now(),status='prepared',runs={key:dict(status='pending',attempts=[]) for key in runs},media={}))
    record_event(output,'prepared',unique_source_runs=len(runs),rollouts=100,clips=44,
        source_frames=sum(r['frames'] for r in runs.values()),plan_sha256=sha256(output/'plan.json'))
    return plan


def validate_render(root,run):
    manifest=read(root/'manifest.json')
    if manifest['status']!='completed' or manifest['layout']!='paper_prompt_arrows_sample_v1':
        raise ValueError('Incomplete or wrong presentation')
    video=root/'debug_rollout.mp4';info=probe(video)
    if (info['frames'],info['fps'],info['width'],info['height'])!=(run['frames'],run['fps'],1280,636):
        raise ValueError(f'Rendered frame geometry mismatch: {info}')
    if sha256(video)!=manifest['video_sha256']:raise ValueError('Rendered video SHA mismatch')
    if manifest.get('task_prompt')!=run['instruction']:raise ValueError('Task prompt changed')
    result=manifest['episode_result'];original=run['original_result']
    for key in ('step_id','complete','success','terminated','truncated'):
        if result.get(key)!=original.get(key):
            authorized_idle=(run['adjudicated_idle'] and (
                (key in ('complete','success') and result.get(key) is False) or
                (key=='step_id' and result.get(key)==run.get('adjudicated_control_steps'))))
            if not authorized_idle:
                raise ValueError(f'Renderer altered original outcome: {key}')
    if run['adjudicated_idle'] and result.get('evaluation_failure_reason')!='simulator_rpc_idle_timeout_900s':
        raise ValueError('Idle-timeout failure label missing')
    from .robodojo_server.video_panel import terminal_badge
    label=terminal_badge(result)
    if run['adjudicated_idle'] and label!='Failed · RPC idle 900s':raise ValueError('Incorrect terminal badge')
    segments=read(root/'decisions.json');commands=read(root/'projected_commands.json')
    by_decision={s['decision']:s for s in segments}
    for command in commands:
        segment=by_decision[command['decision']]
        if (run['method']!='mix' or not segment['codex_override'] or
                not segment['start_tick']<command['tick']<=segment['end_tick']):
            raise ValueError('Arrow not attributable to an executed hybrid correction ACK')
        if command['drawn'] and sum(x*x for x in command['commanded_translation_m'])<1e-6:
            raise ValueError('Arrow drawn for a zero/rotation-only translation')
    if run['method']=='gpt' and (commands or manifest['arrow_frames']!=0):
        raise ValueError('Direct policy must not receive hybrid correction arrows')
    if manifest['arrow_frames']!=len({r['tick'] for r in commands if r['drawn']}):
        raise ValueError('Arrow frame summary differs from recorded per-frame projection')
    for item in run['original_videos']:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Original source video changed')
    for path,digest in run['inputs'].items():
        if sha256(path)!=digest:raise ValueError(f'Original evidence changed: {path}')
    if run.get('original_result_missing') and (Path(run['archive'])/'controller/result.json').exists():
        raise ValueError('An originally missing idle result was created; re-review evidence')
    return dict(output=str(root),video=str(video),sha256=manifest['video_sha256'],
        manifest_sha256=sha256(root/'manifest.json'),bytes=video.stat().st_size,**info,
        arrow_frames=manifest['arrow_frames'],arrow_records=len(commands),
        native_result_unchanged=True,original_source_sha_verified=True,
        terminal_badge=label,idle_label_verified=run['adjudicated_idle'],
        source_renderer_sha256=manifest.get('presentation_source_sha256',{}))


def render_one(output,plan,run,attempt,python,force_new=False):
    started=time.monotonic();sample=plan.get('approved_sample')
    if not force_new and sample and sample['archive']==run['archive']:
        target=Path(sample['output'])
        if sha256(target/'manifest.json')!=sample['manifest_sha256']:raise ValueError('Approved sample manifest changed')
        result=validate_render(target,run);result['reused_approved_preview']=True
    else:
        target=output/'renders'/run['id']/f'attempt_{attempt}'
        log=output/'logs'/f'{run["id"]}_a{attempt}.log';log.parent.mkdir(exist_ok=True)
        command=[str(python),'-m','hybrid_rollout.robodojo.robodojo_server.debug_recorder',
            '--run-root',run['archive'],'--output',str(target),'--presentation','paper',
            '--robodojo-source',plan['robodojo_source'],'--usd-python','/usr/bin/python3',
            '--encoder-threads','1','--font',plan['font']]
        env=dict(os.environ,PYTHONPATH=str(output/'source'),OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with log.open('x') as stream:
            # Inherit the batch lease, so an orphaned renderer still blocks a
            # second dispatcher even across PID namespaces. Never trust a
            # historical numeric PID as ownership proof by itself.
            process=subprocess.Popen(command,cwd=output/'source',env=env,stdout=stream,stderr=subprocess.STDOUT,
                pass_fds=(() if LEASE_FD is None else (LEASE_FD,)))
            process_path=output/'logs'/f'{run["id"]}_a{attempt}.process.json'
            write_json(process_path,dict(pid=process.pid,argv=command,started=now(),status='running'))
            returncode=process.wait()
            write_json(process_path,dict(pid=process.pid,argv=command,ended=now(),status='exited',returncode=returncode))
        if returncode:raise RuntimeError(f'Renderer exited {returncode}; see {log}')
        result=validate_render(target,run)
    result['wall_seconds']=round(time.monotonic()-started,3)
    return result


def export_one(output,row,render):
    started=time.monotonic();video=checked(output/relative_media(row['video']),output)
    poster=checked(output/relative_media(row['poster']),output)
    receipt_path=output/'media-receipts'/f'{row["id"]}.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        if receipt['source_render_sha256']!=render['sha256']:
            raise ValueError('Completed media came from a different rendered source')
        if not all(sha256(output/item['source'])==item['sha256'] for item in receipt['files']):
            raise ValueError('Completed media hash changed; do not overwrite')
        return receipt
    if video.exists() or poster.exists():
        raise FileExistsError('Unreceipted media exists; preserve it for review instead of overwriting')
    video.parent.mkdir(parents=True,exist_ok=True);poster.parent.mkdir(parents=True,exist_ok=True)
    source=Path(render['video'])
    temporary_video=video.with_name(video.stem+f'.partial_{uuid.uuid4().hex}.mp4')
    if not video.exists():
        if row['kind']=='rollout':
            shutil.copyfile(source,temporary_video)
            if sha256(temporary_video)!=render['sha256']:raise ValueError('Rendered rollout copy differs')
        else:
            subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-threads','1',
                '-i',str(source),'-vf',f'trim=start_frame={row["start_frame"]}:end_frame={row["end_frame_exclusive"]},setpts=PTS-STARTPTS',
                '-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(temporary_video)],check=True,capture_output=True)
    info=probe(temporary_video)
    if (info['frames'],info['fps'],info['width'],info['height'])!=(row['frames'],row['fps'],1280,636):
        raise ValueError(f'Clip media geometry changed: {row["id"]}')
    if not poster.exists():
        temporary_poster=poster.with_name(poster.stem+f'.partial_{uuid.uuid4().hex}.jpg')
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n','-threads','1',
            '-i',str(temporary_video),'-frames:v','1','-vf','scale=768:-2','-q:v','3',str(temporary_poster)],check=True,capture_output=True)
    if temporary_video.stat().st_size>=100*1024*1024:raise ValueError('Single media exceeds GitHub file limit')
    os.replace(temporary_video,video);os.replace(temporary_poster,poster)
    files=[dict(path=relative,source=relative,sha256=sha256(path),bytes=path.stat().st_size)
        for relative,path in ((row['video'],video),(row['poster'],poster))]
    receipt=dict(id=row['id'],kind=row['kind'],files=files,**info,source_render_sha256=render['sha256'],
        start_frame=row.get('start_frame'),end_frame_exclusive=row.get('end_frame_exclusive'),
        wall_seconds=round(time.monotonic()-started,3))
    receipt_path.parent.mkdir(exist_ok=True);write_json(receipt_path,receipt)
    return receipt


def run_batch(output,workers,python):
    plan=read(output/'plan.json');ledger=read(output/'ledger.json')
    if plan['schema']!=SCHEMA or sha256(output/'plan.json')!=ledger['plan_sha256']:
        raise ValueError('Frozen plan changed')
    record_event(output,'global_inputs_verified',files=verify_global_inputs(plan),phase='before_render')
    for relative,digest in plan['renderer_source_sha256'].items():
        if sha256(output/'source'/relative)!=digest:raise ValueError('Frozen renderer source changed')
    driver_sha=sha256(__file__);driver_path=output/'drivers'/f'{driver_sha}.py'
    driver_path.parent.mkdir(exist_ok=True)
    if not driver_path.exists():shutil.copyfile(__file__,driver_path)
    ledger.update(status='running',pid=os.getpid(),workers=workers,last_started=now(),driver_sha256=driver_sha)
    write_json(output/'ledger.json',ledger)
    pending=[]
    for run in plan['runs']:
        record=ledger['runs'][run['id']]
        if record['status']=='completed':
            if sha256(record['result']['video'])!=record['result']['sha256']:
                raise ValueError('Previously completed render has changed; do not overwrite it')
            continue
        if record['status']=='running':
            last=record['attempts'][-1]['attempt']
            process_path=output/'logs'/f'{run["id"]}_a{last}.process.json'
            if process_path.exists():
                process=read(process_path);cmdline=Path('/proc')/str(process['pid'])/'cmdline'
                if cmdline.exists() and cmdline.read_bytes().split(b'\0')[:-1]==[x.encode() for x in process['argv']]:
                    raise RuntimeError('An earlier renderer still owns an unfinished output; do not duplicate it')
            record['status']='interrupted';record['attempts'][-1]['status']='interrupted'
        pending.append(run)
    # Article videos first, then the remaining gallery. Work is claimed by the
    # next free CPU worker, not permanently bound to particular trajectory IDs.
    narrative={m['run'] for m in plan['media'] if m.get('narrative')}
    pending.sort(key=lambda r:(r['id'] not in narrative,r['frames'],r['id']))
    start=time.monotonic();active={};last_ping=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            while pending and len(active)<workers:
                run=pending.pop(0);record=ledger['runs'][run['id']];attempt=len(record['attempts'])
                record['status']='running';record['attempts'].append(dict(attempt=attempt,status='running',started=now()))
                active[pool.submit(render_one,output,plan,run,attempt,python)]=run
                record_event(output,'render_started',id=run['id'],frames=run['frames'],method=run['method'],attempt=attempt)
                write_json(output/'ledger.json',ledger)
            completed,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
            for future in completed:
                run=active.pop(future);record=ledger['runs'][run['id']]
                try:
                    result=future.result();record.update(status='completed',result=result)
                    record['attempts'][-1].update(status='completed',ended=now())
                    record_event(output,'render_completed',id=run['id'],frames=result['frames'],
                        arrow_frames=result['arrow_frames'],seconds=result['wall_seconds'])
                except Exception:
                    record.update(status='failed',error=traceback.format_exc());record['attempts'][-1].update(status='failed',ended=now())
                    record_event(output,'render_failed',id=run['id'],error=record['error'])
                write_json(output/'ledger.json',ledger)
            if time.monotonic()-last_ping>=30:
                last_ping=time.monotonic();counts={state:sum(r['status']==state for r in ledger['runs'].values())
                    for state in ('completed','running','pending','failed')}
                write_json(output/'status.json',dict(time=now(),phase='rendering',counts=counts,
                    active_ids=[run['id'] for run in active.values()],elapsed_seconds=round(time.monotonic()-start)))
                record_event(output,'progress',**counts)
    failed=[key for key,value in ledger['runs'].items() if value['status']!='completed']
    if failed:
        ledger.update(status='needs_review',failed_runs=failed);write_json(output/'ledger.json',ledger)
        raise RuntimeError(f'{len(failed)} source render(s) need review; completed outputs preserved')
    record_event(output,'all_source_renders_complete',count=len(plan['runs']))
    record_event(output,'global_inputs_verified',files=verify_global_inputs(plan),phase='before_export')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={}
        for row in plan['media']:
            existing=ledger['media'].get(row['id'])
            if existing and all(sha256(output/item['source'])==item['sha256'] for item in existing['files']):continue
            futures[pool.submit(export_one,output,row,ledger['runs'][row['run']]['result'])]=row
        while futures:
            completed,_=wait(futures,timeout=5,return_when=FIRST_COMPLETED)
            for future in completed:
                row=futures.pop(future);ledger['media'][row['id']]=future.result()
                write_json(output/'ledger.json',ledger)
                record_event(output,'media_completed',id=row['id'],kind=row['kind'],frames=row['frames'])
            write_json(output/'status.json',dict(time=now(),phase='exporting',completed=len(ledger['media']),total=len(plan['media'])))
    files=sorted([item for row in ledger['media'].values() for item in row['files']],key=lambda item:item['path'])
    if len(files)!=288 or len({item['path'] for item in files})!=288:raise ValueError('Public media set is incomplete/duplicated')
    if not all(sha256(output/item['source'])==item['sha256'] for item in files):raise ValueError('Final public media SHA mismatch')
    record_event(output,'global_inputs_verified',files=verify_global_inputs(plan),phase='before_handoff')
    mapping=dict(schema=SCHEMA,complete=True,created=now(),plan_sha256=ledger['plan_sha256'],
        files=files,counts=plan['expected'],total_bytes=sum(item['bytes'] for item in files),
        source_paths_relative_to_manifest=True,
        presentation='paper_prompt_arrows_sample_v1',model_calls=0,simulation_connections=0,
        original_data_unchanged=True,frame_ranges_unchanged=True,
        notes='Prompt header; compact panel; hybrid-only recorded command-direction arrows. Direct and idle-failure labels retained.')
    write_json(output/'media-replacements.json',mapping)
    ledger.update(status='completed',completed=now(),media_replacements_sha256=sha256(output/'media-replacements.json'))
    write_json(output/'ledger.json',ledger)
    write_json(output/'status.json',dict(status='completed',time=now(),source_runs=len(plan['runs']),media_files=len(files),total_bytes=mapping['total_bytes']))
    record_event(output,'batch_completed',source_runs=len(plan['runs']),media_files=len(files),total_bytes=mapping['total_bytes'])


def run_sample(output,case_id,python):
    """One fresh production renderer and its associated clips, never the batch."""
    plan=read(output/'plan.json');ledger=read(output/'ledger.json')
    if sha256(output/'plan.json')!=ledger['plan_sha256']:raise ValueError('Frozen plan changed')
    verify_global_inputs(plan)
    for relative,digest in plan['renderer_source_sha256'].items():
        if sha256(output/'source'/relative)!=digest:raise ValueError('Frozen source changed')
    row=next((x for x in plan['media'] if x['kind']=='rollout' and x['id']==case_id),None)
    if row is None:raise ValueError('Sample must select exactly one existing formal case ID')
    run=next(x for x in plan['runs'] if x['id']==row['run']);record=ledger['runs'][run['id']]
    if record['status']=='completed':raise FileExistsError('Sample is already completed; inspect it instead of rerunning')
    if record['attempts']:raise RuntimeError('Existing sample attempt needs explicit review')
    record.update(status='running',attempts=[dict(attempt=0,status='running',started=now())])
    ledger.update(status='sample_running',sample_case=case_id,pid=os.getpid())
    write_json(output/'ledger.json',ledger);record_event(output,'sample_started',id=run['id'],case=case_id,frames=run['frames'])
    try:
        result=render_one(output,plan,run,0,python,force_new=True)
        record.update(status='completed',result=result);record['attempts'][0].update(status='completed',ended=now())
        write_json(output/'ledger.json',ledger)
        relevant=[m for m in plan['media'] if m['run']==run['id']]
        for item in relevant:
            exported=export_one(output,item,result);ledger['media'][item['id']]=exported
            write_json(output/'ledger.json',ledger)
        files=[file for item in relevant for file in ledger['media'][item['id']]['files']]
        verify_global_inputs(plan)
        # The handoff is safe to inspect/copy: no private absolute paths, run
        # logs, account slots, prompts from Codex internals, or annotations.
        mapping=dict(schema=SCHEMA,complete=False,sample_complete=True,files=files,
            source_paths_relative_to_manifest=True,presentation='paper_prompt_arrows_sample_v1',
            sample_case=case_id,frames=result['frames'],fps=result['fps'],
            dimensions=[result['width'],result['height']],terminal_badge=result['terminal_badge'],
            arrow_frames=result['arrow_frames'],total_bytes=sum(f['bytes'] for f in files),
            original_data_unchanged=True,model_calls=0,simulation_connections=0)
        write_json(output/'sample-replacements.json',mapping)
        ledger.update(status='sample_validated_awaiting_batch',sample_validation=dict(case=case_id,**result))
        write_json(output/'ledger.json',ledger)
        record_event(output,'sample_validated',id=run['id'],case=case_id,seconds=result['wall_seconds'],
            frames=result['frames'],bytes=result['bytes'],arrow_frames=result['arrow_frames'],batch_started=False)
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
        ledger.update(status='sample_needs_review');write_json(output/'ledger.json',ledger)
        record_event(output,'sample_failed',error=record['error']);raise


def main():
    global LEASE_FD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--site',type=Path,default=REPO/'runtime/report_site_20260913_v3')
    parser.add_argument('--snapshot',type=Path,default=REPO/'hybrid_rollout/report_site/app/src/data.json')
    parser.add_argument('--selection',type=Path,default=REPO/'hybrid_rollout/report_site/app/src/content/report/video-selection.json')
    parser.add_argument('--shared',type=Path,default=SHARED)
    parser.add_argument('--approved-sample',type=Path,default=REPO/'runtime/report_debug_preview_20260913/arrange_largest_number_sample')
    parser.add_argument('--python',type=Path,default=DEFAULT_PYTHON)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--phase',choices=('prepare','sample','run'),default='prepare')
    parser.add_argument('--sample-case',default='mix__arrange_largest_number__standard__g0__l0')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(REPO/'runtime') or output==REPO/'runtime':
        parser.error('Only a dedicated directory under this repository runtime is writable')
    if not 1<=args.workers<=4:parser.error('Use 1..4 CPU workers; each video codec is single-threaded')
    if args.phase=='prepare':
        prepare(output,args.site.resolve(),args.snapshot.resolve(),args.selection.resolve(),args.shared.resolve(),args.approved_sample)
    if args.phase in ('run','sample'):
        with (output/'batch.lock').open('a+') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:parser.error('A batch process already owns this output directory')
            LEASE_FD=lock.fileno()
            if args.phase=='sample':run_sample(output,args.sample_case,args.python)
            else:run_batch(output,args.workers,args.python)


if __name__=='__main__':main()
