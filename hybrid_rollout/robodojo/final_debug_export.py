"""Export the full audited 50+50 panel, preferring verified label corrections.

Offline post-processing only: no scheduler/model/simulator operations and no
writes in original archives. Old default exports are preserved separately.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from .audit_follow import load_plans, selected_rows
from .campaign import optional, utc
from .io import require, sha256, write_json
from .paired_evaluation import comparison

LABEL_VERSION = 'native_terminal_labels_v2'
TIMELINE_KEYS = ('frames', 'fps', 'frame_mapping', 'reading_pause_frames',
                'duration_seconds', 'playback_speed', 'episode_result', 'input_sha256')


def verified_video(row):
    root = Path(row['archive']).resolve()
    require(sha256(root/'artifact_manifest.json') == row['artifact_manifest_sha256'], 'Archive manifest changed')
    if row.get('evaluation_failure_reason'):
        return verified_idle_video(row)
    artifacts = optional(root/'artifact_manifest.json')
    require(artifacts.get('status') == 'verified', 'Archive is not verified')
    recorded = {r['path']: r['sha256'] for r in artifacts['all_immutable_artifacts']}

    def original(relative):
        path = (root/relative).resolve()
        require(path.is_relative_to(root), 'Original artifact escaped archive')
        require(sha256(path) == recorded.get(relative), 'Original artifact changed: '+relative)
        return path

    base_video = original('controller/debug_video/debug_rollout.mp4')
    base_manifest = original('controller/debug_video/manifest.json')
    base = optional(base_manifest)
    result = optional(original('controller/result.json'))
    require(result.get('complete') is True and result.get('success') is row['native_success'], 'Not a native-complete selected result')
    require(base.get('status') == 'completed' and base.get('video_sha256') == sha256(base_video), 'Original video incomplete or changed')
    target = root/'controller/debug_video_labels_v2'
    video, manifest, meta = base_video, base_manifest, base
    if target.exists():
        # A broken correction is not silently replaced by a mislabeled original.
        video, manifest = target/'debug_rollout.mp4', target/'manifest.json'
        require(video.resolve().is_relative_to(root) and manifest.resolve().is_relative_to(root), 'Correction escaped archive')
        meta = optional(manifest)
        require(meta.get('status') == 'completed' and meta.get('terminal_label_version') == LABEL_VERSION,
                'Label correction incomplete or unrecognized')
        require(all(meta.get(k) == base.get(k) for k in TIMELINE_KEYS), 'Correction changed annotation timing or episode inputs')
        require(meta.get('video_sha256') == sha256(video), 'Corrected video changed')
    early_failure = result.get('terminated') is True and result.get('success') is False and not result.get('truncated')
    require(not early_failure or meta.get('terminal_label_version') == LABEL_VERSION,
            'Native early failure still needs a label-corrected derived video: '+row['case_id'])
    require(meta['frames'] == result['step_id'] + 1 and meta['fps'] > 0,
            'Video does not preserve every observation frame')
    return dict(source=str(video), sha256=sha256(video), source_manifest=str(manifest),
        source_manifest_sha256=sha256(manifest), original_source=str(base_video),
        original_sha256=sha256(base_video), corrected=video != base_video,
        terminal_label_version=meta.get('terminal_label_version', 'legacy'),
        frames=meta['frames'], fps=meta['fps'], frame_mapping=meta['frame_mapping'])


def verified_idle_video(row):
    from .adjudicated_archive_audit import adjudicated
    from .idle_timeout_policy import SIDECAR
    root=Path(row['archive']).resolve()
    evidence=adjudicated(row)
    artifacts=optional(root/'artifact_manifest.json')
    require(artifacts.get('status')=='incomplete','Idle archive must retain original incomplete status')
    recorded={r['path']:r['sha256'] for r in artifacts['all_immutable_artifacts']}
    base_video=root/'controller/debug_video/debug_rollout.mp4'
    base_manifest=root/'controller/debug_video/manifest.json'
    for path in (base_video,base_manifest):
        require(sha256(path)==recorded.get(str(path.relative_to(root))),'Original idle video changed')
    base=optional(base_manifest)
    target=root/'controller/debug_video_idle_timeout_v1'
    video,manifest=target/'debug_rollout.mp4',target/'manifest.json'
    require(video.resolve().is_relative_to(root) and manifest.resolve().is_relative_to(root),
        'Idle label correction escaped archive')
    meta=optional(manifest)
    require(base.get('status')=='completed' and base.get('video_sha256')==sha256(base_video),
        'Original idle video incomplete or changed')
    require(meta.get('status')=='completed' and meta.get('evaluation_adjudication')==evidence,
        'Idle failure needs its independently labeled debug video')
    keys=set(TIMELINE_KEYS)-{'episode_result','input_sha256'}
    require(all(meta.get(k)==base.get(k) for k in keys),'Idle correction changed frame timing')
    require(meta.get('input_sha256')==dict(base.get('input_sha256',{}),
        **{str(root/SIDECAR):row['adjudication_sha256']}), 'Idle correction changed original rendering inputs')
    result=meta.get('episode_result',{})
    require(result.get('complete') is False and result.get('native_complete') is False
        and result.get('step_id')==row['control_steps']
        and result.get('evaluation_failure_reason')==row['evaluation_failure_reason']
        and meta.get('frames')==row['control_steps']+1,'Idle video fabricated native completion or frames')
    require(meta.get('video_sha256')==sha256(video),'Idle corrected video changed')
    process=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames','-of','json',str(video)],
        capture_output=True,text=True,check=True,timeout=180)
    require(int(json.loads(process.stdout)['streams'][0]['nb_read_frames'])==meta['frames'],
        'Idle corrected video does not decode to the recorded frame count')
    return dict(source=str(video),sha256=sha256(video),source_manifest=str(manifest),
        source_manifest_sha256=sha256(manifest),original_source=str(base_video),original_sha256=sha256(base_video),
        corrected=True,terminal_label_version='rpc_idle_timeout_v1',frames=meta['frames'],fps=meta['fps'],
        frame_mapping=meta['frame_mapping'],adjudication_sha256=row['adjudication_sha256'])


def verify_full_audits(rows, deep, paired):
    counts = Counter(r['method'] for r in rows)
    keys = {r['method']+':'+r['case_id'] for r in rows}
    require(counts == {'pi05_plus_gpt': 50, 'gpt_only': 50} and len(keys) == 100,
            'Require exactly fifty unique complete cases per method')
    require(deep.get('passed_count') == 100 and deep.get('selected_count') == 100 and not deep.get('errors'),
            'Full 100-trajectory deep audit is not finished')
    for row in rows:
        audit = deep.get('rows', {}).get(row['method']+':'+row['case_id'], {})
        require(audit.get('passed') is True and audit.get('archive') == row['archive']
            and audit.get('selected_manifest_sha256') == row['artifact_manifest_sha256'],
            'Deep audit does not cover selected trajectory')
        if row.get('evaluation_failure_reason'):
            require(audit.get('native_complete') is False
                and audit.get('adjudication_sha256')==row.get('adjudication_sha256')
                and audit.get('native_success') is None and audit.get('evaluation_success') is False,
                'Missing separate deep audit of adjudicated failure')
    require(paired.get('status') == 'complete_50_pairs' and paired.get('pairs') == 50 and not paired.get('errors'),
            'Full fifty-pair initial-setting audit is not finished')
    pairs = {r['case_id']: r for r in paired.get('rows', [])}
    require(len(pairs) == 50 and len(paired.get('rows', [])) == 50, 'Duplicate or missing pair audit')
    for row in rows:
        audit = pairs.get(row['case_id'], {})
        side = 'hybrid' if row['method'] == 'pi05_plus_gpt' else 'gpt_only'
        require(audit.get('settings_and_robot_text_match') is True
            and audit.get(side+'_archive') == row['archive']
            and audit.get(side+'_artifact_manifest_sha256') == row['artifact_manifest_sha256'],
            'Pair audit does not cover selected trajectory')
        if row.get('evaluation_failure_reason'):
            require(audit.get('gpt_only_native_complete') is False
                and audit.get('gpt_only_adjudication_sha256')==row.get('adjudication_sha256'),
                'Pair audit does not cover the adjudicated failure')


def export(workflow, approved, deep_path, paired_path, output):
    workflow = Path(workflow).resolve()
    plans = load_plans(workflow, approved)
    progress = optional(workflow.parent/'progress.json')
    rows = selected_rows(progress, plans)
    verify_full_audits(rows, optional(deep_path), optional(paired_path))
    report = comparison(progress['hybrid'], progress['direct'])
    output = Path(output).resolve()
    for plan in plans.values():
        require(not output.is_relative_to(Path(plan['shared_root'])/'results'), 'Export cannot alter original archives')
    require(not output.exists(), 'Use a new export directory; preserve existing downloads')
    # Validate every source before creating a partial output directory.
    videos = []
    for row in rows:
        name = f"{row['method']}__{row['case_id']}__v3.mp4"
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', name) is not None, 'Unsafe output filename')
        worker = optional(Path(row['archive'])/'controller/codex_workspace/worker.json')
        videos.append(dict(case_id=row['case_id'], method=row['method'], context_version='v3',
            evaluation_case=row['evaluation_case'], archive=row['archive'], file=name,
            native_success=row['native_success'], native_score=row['native_score'],
            native_complete=row.get('native_complete',True),
            evaluation_success=row.get('evaluation_success',row['native_success']),
            evaluation_failure_reason=row.get('evaluation_failure_reason'),
            controller_version=worker.get('controller_version', 'legacy_unversioned'),
            artifact_manifest_sha256=row['artifact_manifest_sha256'], **verified_video(row)))
    output.mkdir(parents=True)
    directory = output/'debug_videos'; directory.mkdir()
    for item in videos:
        destination = directory/item['file']
        shutil.copy2(item['source'], destination)
        require(sha256(destination) == item['sha256'], 'Exported video changed')
    metadata = dict(created_utc=utc(), workflow_sha256=approved, videos=videos,
        deep_audit=str(deep_path), deep_audit_sha256=sha256(deep_path),
        paired_audit=str(paired_path), paired_audit_sha256=sha256(paired_path),
        original_files_untouched=True, annotations_time_mapping_unchanged=True,
        controller_caveat='Controller versions differ across attempts; inspect per-video metadata. '
        'The recoverable-input fix applies only to attempts launched after its approved cutover.')
    write_json(directory/'manifest.json', metadata)
    write_json(output/'results.json', report)
    with (output/'scores.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['per_task'][0]))
        writer.writeheader(); writer.writerows(report['per_task'])
    archive = output/'debug_videos.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as stream:
        for path in sorted(directory.iterdir()):
            stream.write(path, path.name)
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None, 'Debug ZIP failed CRC')
        require(set(stream.namelist()) == {v['file'] for v in videos} | {'manifest.json'}, 'ZIP video membership mismatch')
    write_json(output/'COMPLETE.json', dict(utc=utc(), videos=100, corrected=sum(v['corrected'] for v in videos),
        zip_sha256=sha256(archive), result_sha256=sha256(output/'results.json'),
        score_csv_sha256=sha256(output/'scores.csv'), manifest_sha256=sha256(directory/'manifest.json')))
    return dict(output=str(output), videos=100, corrected=sum(v['corrected'] for v in videos), zip=str(archive))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow', type=Path, required=True)
    parser.add_argument('--approved-sha256', required=True)
    parser.add_argument('--deep-audit', type=Path, required=True)
    parser.add_argument('--paired-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.workflow, args.approved_sha256, args.deep_audit, args.paired_audit, args.output)))
