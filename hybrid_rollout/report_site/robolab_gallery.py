"""Export a static RoboLab gallery from the selected, already recorded trials.

The private catalog supplies explicit local files. Only whitelisted public fields
and copied MP4s/posters enter the output; no controller or model is imported.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote, urlsplit

HERE = Path(__file__).resolve().parent
METHODS = ('astra_pi05', 'pure_astra', 'pi05_only')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def selected_rows(catalog, summary):
    """Require the exact first-five panel, retaining failures and final retries."""
    tasks = {t['task']: t for t in summary['tasks']}
    rows = [r for r in catalog['rows'] if r['method'] in METHODS and r['task'] in tasks
            and r.get('scope') == ('historical baseline' if r['method']=='pi05_only' else 'primary')
            and 0 <= r['trial'] < 5]
    slots = [(r['method'], r['task'], r['trial']) for r in rows]
    expected = {(m, task, trial) for m in METHODS for task in tasks for trial in range(5)}
    if len(slots) != len(set(slots)) or set(slots) != expected:
        raise ValueError('Expected exactly one final recording for every first-five slot')
    if any(r['status'] not in {'success', 'failure'} for r in rows):
        raise ValueError('Incomplete trials cannot enter the completed gallery')
    for task, reference in tasks.items():
        for method in METHODS:
            successes = sum(r['status'] == 'success' for r in rows
                            if r['task'] == task and r['method'] == method)
            if successes != reference['successes'][method]:
                raise ValueError(f'Outcome mismatch: {task}/{method}')
    order = {task: i for i, task in enumerate(tasks)}
    return sorted(rows, key=lambda r: (METHODS.index(r['method']), order[r['task']], r['trial']))


def export_gallery(catalog, summary, output, media_roots):
    output = Path(output).resolve()
    roots = [Path(p).resolve() for p in media_roots]
    rows = selected_rows(catalog, summary)
    sources = []
    for row in rows:
        if not row['videos']:
            raise ValueError(f"Missing rendered recording: {row['id']}")
        video = row['videos'][0]
        source = Path(video['file']).resolve()
        if not any(source.is_relative_to(root) for root in roots) or not source.is_file():
            raise ValueError('Video is missing or outside the explicit media roots')
        digest = sha(source)
        if video.get('sha256') and video['sha256'] != digest:
            raise ValueError('Video hash changed')
        if source.stat().st_size != video['bytes']:
            raise ValueError('Video size changed')
        sources.append((row, video, source, digest))
    if output.exists():
        raise ValueError('Use a new output directory; existing recordings stay untouched')
    output.mkdir(parents=True)
    labels = json.loads((HERE/'app/src/content/report/robolab-locales.json').read_text())
    data = dict(benchmark='RoboLab', methods=[{'id': m, 'label': next(
        x['label'] for x in summary['methods'] if x['id'] == m)} for m in METHODS],
        tasks=[{'id': t['task'], 'label': labels[t['task']]['zh']} for t in summary['tasks']],
        translations={t['task']: {'label': t['label']} for t in summary['tasks']},
        cases=[], clips=[], models=[], paired=False)
    files = []
    for row, video, source, digest in sources:
        key = f"robolab__{row['method']}__{row['task']}__{row['trial']:03d}"
        target = output/'media/robolab'/f'{key}.mp4'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != digest:
            raise ValueError('Copied video hash mismatch')
        poster = target.with_suffix('.jpg')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(target),
                        '-frames:v', '1', '-vf', 'scale=640:-2', '-threads', '1', str(poster)], check=True)
        for path in (target, poster):
            files.append(dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size,
                              sha256=sha(path)))
        data['cases'].append(dict(id=key, method=row['method'], task=row['task'],
            trial=row['trial'], status=row['status'], instruction=row['instruction'],
            steps=row['step'], limit=row.get('max_steps'), duration=video['duration'],
            fps=video['fps'], frames=video['frames'],
            video=target.relative_to(output).as_posix(), poster=poster.relative_to(output).as_posix(),
            video_sha256=digest, video_url=target.relative_to(output).as_posix()+'?v='+digest[:16]))
    dump(output/'robolab-gallery.json', data)
    dump(output/'robolab-gallery-manifest.json', dict(schema='robolab.gallery.v1',
        videos=len(rows), tasks=len(data['tasks']), trials_per_task=5, files=files,
        successes=dict(Counter(r['method'] for r in rows if r['status']=='success')),
        source_summary_sha256=hashlib.sha256(json.dumps(summary,sort_keys=True).encode()).hexdigest(),
        original_media_unchanged=True, raw_annotations_included=False))
    return data


def release_video_base(value):
    """Allow a public GitHub release prefix, never credentials or queries."""
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or parsed.netloc != 'github.com'
            or parsed.query or parsed.fragment
            or not re.fullmatch(r'/[\w.-]+/[\w.-]+/releases/download/[\w.-]+/', parsed.path)):
        raise ValueError('Expected an HTTPS GitHub release download URL ending in /')
    return value


def attach_gallery(bundle, output, data, robodojo_url=None, video_base_url=None):
    """Attach reviewed files to an existing report without exposing a raw folder."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = json.loads((bundle/'robolab-gallery-manifest.json').read_text())
    gallery = json.loads((bundle/'robolab-gallery.json').read_text())
    if video_base_url:
        video_base_url = release_video_base(video_base_url)
    if manifest['videos'] != 150 or len(gallery['cases']) != 150:
        raise ValueError('Expected the completed 150-trial gallery')
    external_files = []
    for row in manifest['files']:
        name = Path(row['path']); source = (bundle/name).resolve()
        if (name.is_absolute() or '..' in name.parts or name.parts[:2] != ('media', 'robolab')
                or name.suffix not in {'.mp4', '.jpg'} or not source.is_relative_to(bundle)):
            raise ValueError('Invalid gallery media path')
        if source.stat().st_size != row['bytes'] or sha(source) != row['sha256']:
            raise ValueError('Gallery media integrity error')
        if video_base_url and name.suffix == '.mp4':
            external_files.append({**row, 'url': video_base_url + quote(name.name)})
            continue
        target = output/name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(output):
            raise ValueError('Gallery destination escapes the report')
        if source != target.resolve():
            shutil.copyfile(source, target)
    if video_base_url:
        hosted = {row['path']: row for row in external_files}
        if len(hosted) != 150 or len(external_files) != 150:
            raise ValueError('Expected 150 distinct release video assets')
        for row in gallery['cases']:
            asset = hosted.get(row['video'])
            if not asset or asset['sha256'] != row['video_sha256']:
                raise ValueError('Gallery video does not match the release manifest')
            row['video_url'] = asset['url']
        manifest['files'] = [row for row in manifest['files'] if row['path'] not in hosted]
        manifest['external_files'] = external_files
        manifest['video_hosting'] = 'github-release'
    has_dojo = data['publication'].get('galleryAvailable', False)
    galleries = []
    if has_dojo or robodojo_url:
        galleries.append(dict(benchmark='RoboDojo', href='gallery.html' if has_dojo else robodojo_url))
    galleries.append(dict(benchmark='RoboLab', href='robolab-gallery.html'))
    gallery['galleries'] = galleries
    encoded = json.dumps(gallery, ensure_ascii=False).replace('<', '\\u003c')
    (output/'robolab-gallery-data.js').write_text('window.REPORT_DATA = '+encoded+';\n')
    dump(output/'robolab-gallery.json', gallery)
    dump(output/'robolab-gallery-manifest.json', manifest)
    for name in ('gallery.js', 'gallery.css'):
        shutil.copyfile(HERE/'web'/name, output/name)
    page = (HERE/'web/gallery.html').read_text().replace('src="data.js"', 'src="robolab-gallery-data.js"')
    for name in ('gallery.js', 'gallery.css', 'robolab-gallery-data.js'):
        page = page.replace('"'+name+'"', '"'+name+'?v='+sha(output/name)[:16]+'"')
    (output/'robolab-gallery.html').write_text(page)
    font = output/'media/fonts/report-cjk.woff'; font.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE/'app/src/content/assets/report-song.woff', font)
    data['publication']['robolabGalleryAvailable'] = True
    if has_dojo:
        from .package_article import public_gallery, version_gallery_assets
        dojo = public_gallery(data)
        dojo['translations'] = json.loads((HERE/'app/src/content/report/locale.en.json').read_text())['tasks']
        dojo['galleries'] = galleries
        (output/'data.js').write_text('window.REPORT_DATA = '+json.dumps(dojo,ensure_ascii=False).replace('<','\\u003c')+';\n')
        shutil.copyfile(HERE/'web/gallery.html', output/'gallery.html')
        version_gallery_assets(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--media-root', type=Path, action='append', required=True)
    args = parser.parse_args()
    data = export_gallery(json.loads(args.catalog.read_text()), json.loads(args.summary.read_text()),
                          args.output, args.media_root)
    print(json.dumps(dict(videos=len(data['cases']), tasks=len(data['tasks']), output=str(args.output))))


if __name__ == '__main__':
    main()
