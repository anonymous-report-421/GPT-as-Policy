"""Build the same report app, including only its selected narrative clips.

The authoring snapshot, gallery, raw annotations and original media are untouched.
The temporary source build preserves the app identity and protected-runtime checks.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

def selected_video_ids():
    selection = json.loads((HERE/'app/src/content/report/video-selection.json').read_text())
    ids = [row['id'] for row in selection]
    if len(ids) != len(set(ids)):
        raise ValueError('A narrative clip may only appear once')
    return set(ids)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def public_snapshot(original, include_gallery=False):
    data = deepcopy(original)
    # The withdrawn overall-reference query is not part of the selected-task report.
    data['queries'].pop('robolab_opening_models', None)
    data['publication'] = {'galleryAvailable': include_gallery, 'mediaScope': 'report-and-gallery' if include_gallery else 'article-clips-only'}
    data['buildStatus'] = 'complete'
    data['reportData'].pop('annotations', None)
    data['reportData'].pop('content', None)
    data['reportData'].pop('community_references_status', None)
    private = {'version','controller_version','video_ui','comment','tags','run_id',
               'original_annotation_start','original_annotation_end'}
    selected = selected_video_ids()
    if not include_gallery:
        data['queries']['clips']['rows'] = [r for r in data['queries']['clips']['rows'] if r['id'] in selected]
    found = {r['id'] for r in data['queries']['clips']['rows'] + data['queries']['featured_case']['rows']}
    if not selected.issubset(found) or (not include_gallery and found != selected):
        raise ValueError(f'Narrative selection does not match reviewed media: {selected ^ found}')
    for name, query in data['queries'].items():
        if not name.startswith('robolab_'):
            query['source']['label'] = 'RoboDojo · frozen evaluation and public leaderboard'
        for row in query['rows']:
            for key in private:
                row.pop(key, None)
            if name == 'cases' and not include_gallery:
                for field in ('video', 'poster', 'video_url', 'poster_url'):
                    row.pop(field, None)
            if name == 'featured_case':
                if 'included_in_v3_statistics' in row:
                    row['included_in_quantitative_statistics'] = row.pop('included_in_v3_statistics')
                if 'included_in_quantitative_statistics' not in row:
                    raise ValueError('Historical clip is missing its quantitative inclusion flag')
                row.pop('caption', None)
                row.get('seeds', {}).pop('panel_id', None)
    return data

def public_gallery(data):
    """Only reviewed public collections; never embed author annotations."""
    return {name:deepcopy(data['queries'][name]['rows']) for name in ('models','tasks','cases','clips')}

def version_gallery_assets(output):
    """Invalidate cached gallery code/data together with changed media."""
    page=output/'gallery.html'
    content=page.read_text()
    for name in ('gallery.css','gallery.js','data.js'):
        original='"'+name+'"'
        if content.count(original)!=1:
            raise ValueError('Expected one gallery asset reference: '+name)
        content=content.replace(original,'"'+name+'?v='+sha(output/name)[:16]+'"')
    page.write_text(content)

def load_media_replacements(manifest_path, required):
    """Validate a completed offline bundle; sources stay under its directory."""
    manifest_path=Path(manifest_path).resolve()
    metadata=json.loads(manifest_path.read_text())
    if metadata.get('complete') is not True:
        raise ValueError('Offline video rendering has not completed')
    replacements={}
    for row in metadata['files']:
        relative=Path(row['path']); source_relative=Path(row['source'])
        for value in (relative,source_relative):
            if value.is_absolute() or '..' in value.parts or not value.parts or value.parts[0]!='media':
                raise ValueError('Replacement paths must be relative media paths')
        if row['path'] in replacements:
            raise ValueError('Duplicate replacement paths')
        source=(manifest_path.parent/source_relative).resolve()
        if not source.is_relative_to(manifest_path.parent) or not source.is_file():
            raise ValueError('Rendered source escapes the media bundle or is missing')
        if source.stat().st_size!=row['bytes'] or sha(source)!=row['sha256']:
            raise ValueError('Rendered media integrity error: '+row['path'])
        replacements[row['path']]={**row,'resolved_source':source}
    if not set(required).issubset(replacements):
        raise ValueError('Missing new media: '+str(sorted(set(required)-set(replacements))))
    return replacements

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--site',type=Path,default=REPO/'runtime/report_site_20260913_v3')
    parser.add_argument('--output',type=Path,default=REPO/'report_web')
    parser.add_argument('--node',type=Path,required=True)
    build_mode=parser.add_mutually_exclusive_group(required=True)
    build_mode.add_argument('--data-app',type=Path)
    build_mode.add_argument('--local-source-build',action='store_true',help='Explicitly use the copied app verification and Vite build when the external plugin is unavailable')
    parser.add_argument('--build-project',type=Path,help='Reuse an existing temporary build of this same report and its installed dependencies')
    parser.add_argument('--include-gallery',action='store_true',help='Publish all 100 evaluation videos and the reviewed gallery clips')
    parser.add_argument('--robolab-gallery',type=Path,help='Verified static RoboLab gallery bundle exported from the selected recorded trials')
    parser.add_argument('--robolab-video-base-url',help='Verified GitHub Release download prefix; keep large originals outside the Pages artifact')
    parser.add_argument('--media-replacements',type=Path,help='Completed offline render manifest; all included videos and posters must be covered')
    parser.add_argument('--supplementary',type=Path,default=REPO/'report_web',help='Reviewed RoboLab result exports from the collaborator')
    args=parser.parse_args()
    if args.robolab_video_base_url and not args.robolab_gallery:
        parser.error('--robolab-video-base-url requires --robolab-gallery')
    site=args.site.resolve(); output=args.output.resolve()
    if output in {REPO,site} or site in output.parents:
        raise ValueError('Use a separate article-only output directory')
    if output.exists() and not (output/'media-manifest.json').exists():
        raise ValueError('Refusing to overwrite an unrelated output directory')
    original=json.loads((HERE/'app/src/data.json').read_text())
    data=public_snapshot(original,args.include_gallery)
    rows=data['queries']['clips']['rows']+data['queries']['featured_case']['rows']
    if args.include_gallery:
        rows+=data['queries']['cases']['rows']
        assert len(rows)==144
    else:
        assert len(rows)==len(selected_video_ids())
    assert len(data['queries']['cases']['rows'])==100
    for key in ['models','tasks','intervention','costs']:
        assert data['queries'][key]['rows']==original['queries'][key]['rows']
    # No raw comments, credentials or local mount paths.
    encoded=json.dumps(data,ensure_ascii=False)
    assert not re.search(r'"comment"|"annotations"|/mnt/|sk-(?:dz-)?[A-Za-z0-9]{16,}',encoded)
    replacements={}
    if args.media_replacements:
        required={r[k] for r in rows for k in ('video','poster')}
        replacements=load_media_replacements(args.media_replacements,required)
        data['reportData']['video_note']='Debug videos include the task prompt and projected GPT correction directions.'
        for row in rows:
            row.pop('source_video_sha256',None)
            row.pop('video_sha256',None)
            row['video_sha256']=replacements[row['video']]['sha256']
            # Stable case/media paths remain unchanged; browser URLs identify
            # the rendered bytes so an existing viewer cannot reuse an old clip.
            row['video_url']=row['video']+'?v='+row['video_sha256'][:16]
            row['poster_url']=row['poster']+'?v='+replacements[row['poster']]['sha256'][:16]
    output.mkdir(parents=True,exist_ok=True)
    manifest=[]
    asset_paths=sorted({r[k] for r in rows for k in ['video','poster']} |
                       {t['image'] for t in data['queries']['tasks']['rows']})
    for relative in asset_paths:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'Invalid relative asset path: {relative}')
        source=(site/relative).resolve()
        replacement=replacements.get(relative)
        if replacement:
            source=replacement['resolved_source']
            if not source.is_file() or sha(source)!=replacement['sha256'] or source.stat().st_size!=replacement['bytes']:
                raise ValueError(f'Rendered media integrity error: {relative}')
        elif not source.is_relative_to(site) or not source.is_file():
            raise ValueError(f'Invalid or missing asset: {relative}')
        target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert sha(source)==sha(target)
        manifest.append({'path':relative,'bytes':target.stat().st_size,'sha256':sha(target)})
    shutil.copyfile(site/'scores.csv',output/'scores.csv')
    supplementary_files=[]
    if 'robolab_models' in data['queries']:
        for filename in ('robolab.json','robolab-baselines.json','robolab-leaderboard.csv','robolab-scores.csv'):
            source=args.supplementary/filename
            if source.resolve() != (output/filename).resolve():
                shutil.copyfile(source,output/filename)
            supplementary_files.append({'path':filename,'sha256':sha(source)})
    licenses=output/'licenses';licenses.mkdir(exist_ok=True)
    for filename in ['NotoSerif-LICENSE.txt','SourceSerif-LICENSE.txt']:
        shutil.copyfile(HERE/'app/src/content/assets'/filename,licenses/filename)
    dump(output/'data.json',data)
    if args.include_gallery:
        gallery=public_gallery(data)
        gallery['translations']=json.loads((HERE/'app/src/content/report/locale.en.json').read_text())['tasks']
        (output/'data.js').write_text('window.REPORT_DATA = '+json.dumps(gallery,ensure_ascii=False).replace('<','\\u003c')+';\n')
        for filename in ('gallery.html','gallery.css','gallery.js'):
            shutil.copyfile(HERE/'web'/filename,output/filename)
        version_gallery_assets(output)
        font=output/'media/fonts/report-cjk.woff'
        font.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/'app/src/content/assets/report-song.woff',font)
    robolab_manifest = None
    if args.robolab_gallery:
        from .robolab_gallery import attach_gallery
        robolab_manifest = attach_gallery(args.robolab_gallery, output, data,
                                         video_base_url=args.robolab_video_base_url)
        dump(output/'data.json', data)
    publication=json.loads((HERE/'app/src/content/report/publication.json').read_text())
    (output/'citation.bib').write_text(publication['bibtex']+'\n')
    scope=data['publication']['mediaScope']
    dump(output/'media-manifest.json',{'scope':scope,'videos':len(rows),
        'video_bytes':sum(r['bytes'] for r in manifest if r['path'].endswith('.mp4')),
        'files':manifest})
    dump(output/'provenance.json',{'scope':scope,
        'results_sha256':data['reportData']['source_sha256'],
        'official_sha256':data['reportData']['official_sha256'],
        'official_source':'https://robodojo-benchmark.com/',
        'score_missingness':'GPT 6 Astra（direct） mean 37.8125 uses 48 scored episodes; success rate uses all 50.',
        'official_scope':'Official results are reweighted for the selected ten tasks and scene mix, not paired seed reruns.',
        'media_manifest':'media-manifest.json','raw_annotations_included':False,
        'full_rollout_videos_included':args.include_gallery,'original_media_unchanged':True,
        'robolab_supplementary':supplementary_files,
        'presentation':'paper' if replacements else 'original-debug',
        'correction_arrows':'Projected commanded EEF direction on executed hybrid correction steps; not an achieved-motion guarantee.' if replacements else None})
    # Separate build directory, not a separate renderer, static HTML fork or patched bundle.
    if args.build_project:
        app=args.build_project.resolve()
        if app == (HERE/'app').resolve() or not (app/'src/data.json').is_file():
            raise ValueError('Use an existing separate temporary build of the same report')
        if json.loads((app/'src/data.json').read_text())['id'] != original['id']:
            raise ValueError('Cannot overwrite a different report')
        if not (app/'node_modules').is_dir() or (app/'node_modules').is_symlink():
            raise ValueError('Build dependencies must be contained in the build project')
        shutil.copytree(HERE/'app',app,dirs_exist_ok=True,ignore=shutil.ignore_patterns('node_modules','.npm-cache','dist','.data-app-offline'))
    else:
        app=Path(tempfile.mkdtemp(prefix='robodojo-article-build-'))/'app'
        shutil.copytree(HERE/'app',app,ignore=shutil.ignore_patterns('node_modules','.npm-cache','dist','.data-app-offline'))
        # Copy existing contained dependencies; never fetch/install or relax checks.
        shutil.copytree(HERE/'app/node_modules',app/'node_modules')
    dump(app/'src/data.json',data)
    build_env=dict(os.environ)
    for key in ['CODEX_SESSION_ID','CODEX_THREAD_ID']:
        build_env.pop(key,None)
    if args.local_source_build:
        # The app's own source-build contract; no renderer substitution,
        # installation, integrity bypass or automatic fallback.
        subprocess.run([str(args.node),'scripts/verify-protected-runtime.mjs'],cwd=app,check=True,env=build_env)
        subprocess.run([str(args.node),'node_modules/vite/bin/vite.js','build'],cwd=app,check=True,env=build_env)
    else:
        subprocess.run([str(args.node),str(args.data_app),'build','--project-dir',str(app),'--source'],check=True,env=build_env)
    shutil.copyfile(app/'dist/index.html',output/'index.html')
    if not args.include_gallery:
        assert not list(output.glob('media/rollouts/*'))
    files=[p for p in output.rglob('*') if p.is_file() and p.relative_to(output).parts[0] not in {'downloads','previews'} and p.name != 'build-manifest.json']
    assert all(p.stat().st_size<100*1024*1024 for p in files)
    result={'output':str(output),'html_sha256':sha(output/'index.html'),
        'files':len(files)+1,'total_bytes':sum(p.stat().st_size for p in files),
        'video_bytes':sum(r['bytes'] for r in manifest if r['path'].endswith('.mp4')),
        'build_directory':str(app)}
    if robolab_manifest:
        result['robolab_videos'] = robolab_manifest['videos']
        result['video_bytes'] += sum(r['bytes'] for r in robolab_manifest['files'] if r['path'].endswith('.mp4'))
        result['external_video_bytes'] = sum(r['bytes'] for r in robolab_manifest.get('external_files', []))
    base_bytes=result['total_bytes']
    # Count this manifest as part of the package, not the previous manifest's size.
    while True:
        manifest_value={k:v for k,v in result.items() if k not in {'output','build_directory'}}
        total=base_bytes+len((json.dumps(manifest_value,ensure_ascii=False,indent=2)+'\n').encode())
        if total==result['total_bytes']:
            break
        result['total_bytes']=total
    dump(output/'build-manifest.json',manifest_value)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    main()
