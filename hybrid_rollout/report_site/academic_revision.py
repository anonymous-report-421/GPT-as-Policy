"""Add the user-selected, frame-exact teaser without changing the gallery or scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .build import HERE, SHARED, checked_under, dump, probe, read, sha

SOURCE = ('results/robodojo_pool7_vla_reserved_20260911_01_stable_c1_a5/'
          'imitate_sorting_sequence/standard/eval_seed_0/layout_0/replica_2/attempt_5')
SOURCE_SHA = '9d123eed4ae3ea54e64ea244da2a36cf5f62d03b54a84fd057476b834c2f9836'
CLIP_ID = 'imitate_sorting_sequence_a5_frames_493_1309'


def revise(site: Path):
    site = site.resolve()
    source_dir = checked_under(SHARED / SOURCE, SHARED / 'results')
    source = source_dir / 'controller/debug_video/debug_rollout.mp4'
    if sha(source) != SOURCE_SHA:
        raise ValueError('The requested source video has changed')
    info = probe(source)
    if info['r_frame_rate'] != '25/1' or int(info['nb_frames']) != 1601:
        raise ValueError('Unexpected source frame geometry')
    run = read(source_dir / 'controller/run.json')
    outcome = read(source_dir / 'sim/evaluation_outcome.json')
    # Preserve all delivery files and gallery inputs before updating only index.html.
    backup = site.parent / 'report_before_academic_20260913'
    backup.mkdir(exist_ok=True)
    for src in (site / 'index.html', HERE / 'app/src/data.json'):
        dst = backup / ('index.html' if src.name == 'index.html' else 'app_snapshot.json')
        if not dst.exists():
            shutil.copyfile(src, dst)
    untouched = ['gallery.html', 'gallery.css', 'gallery.js', 'data.js', 'data.json', 'scores.csv']
    before = {f: sha(site / f) for f in untouched}
    video = site / f'media/clips/{CLIP_ID}.mp4'
    poster = site / f'media/posters/{CLIP_ID}.jpg'
    if not video.exists():
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                        '-i', str(source), '-vf', 'trim=start_frame=493:end_frame=1310,setpts=PTS-STARTPTS',
                        '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    actual = probe(video)
    if (int(actual['nb_frames']), actual['r_frame_rate'], actual['width'], actual['height']) != (817, '25/1', 1280, 680):
        raise ValueError('Derived clip failed its frame-exact check')
    if not poster.exists():
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                        '-i', str(video), '-frames:v', '1', '-q:v', '2', str(poster)], check=True)
    hero = dict(id=CLIP_ID, task='imitate_sorting_sequence', method='mix',
                version=run.get('context_version') or '历史版本（未记录 context 版本）',
                instruction=run['instruction'], title='基于反馈的动作调整与非抓取操作',
                start_frame=493, end_frame_inclusive=1309, end_frame_exclusive=1310,
                frame_index_base=0, frames=817, fps=25, start=493/25, end=1310/25,
                duration=817/25, video=video.relative_to(site).as_posix(),
                poster=poster.relative_to(site).as_posix(), source_video_sha256=SOURCE_SHA,
                video_sha256=sha(video), native_complete=outcome['complete'],
                native_success=outcome['native_success'], native_score=outcome['native_score'],
                included_in_v3_statistics=False, seeds=run['evaluation_case'],
                video_ui='existing_debug_placeholder',
                caption='在模仿排序任务中，混合策略围绕手机的搬运反复调整接近方式，'
                        '使用拨动等非抓取接触改变物体状态，并在后续阶段切换至手表操作。'
                        '该片段展示了依据反馈调整动作与继续任务序列的局部行为；'
                        '完整轨迹最终未成功（原生 Score 0.15），不计入本文 v3 结果。')
    dump(site / 'hero_clip.json', hero)
    snapshot_path = HERE / 'app/src/data.json'
    snapshot = read(snapshot_path)
    snapshot['buildStatus'] = 'updating'
    snapshot['queries']['featured_case'] = dict(label='User-selected historical behavior excerpt', rows=[hero],
        source=dict(label='Requested debug-video frames 493–1309 (inclusive, zero-based)',
                    files=['hero_clip.json'], video_sha256=SOURCE_SHA,
                    notes=['Historical qualitative example only; excluded from all v3 scores.',
                           '817 frames at 25 FPS; original trajectory complete but unsuccessful.']))
    dump(snapshot_path, snapshot)
    after = {f: sha(site / f) for f in untouched}
    if before != after:
        raise ValueError('Gallery or evaluation data changed during teaser preparation')
    dump(site.parent / 'report_academic_revision_receipt.json',
         dict(status='media_verified', hero=hero, unchanged=after,
              original_video_unchanged=sha(source) == SOURCE_SHA,
              previous_html=str(backup / 'index.html'), model_calls=0))
    print(json.dumps(dict(frames=817, duration=817/25, source_was_in_body=False,
                          gallery_unchanged=before == after, video=str(video))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('site', type=Path)
    revise(parser.parse_args().site)
