"""Guard completed trial selection and gallery media boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .robolab_gallery import METHODS, selected_rows, attach_gallery, release_video_base


class RoboLabGalleryTests(unittest.TestCase):
    def setUp(self):
        self.summary = {'tasks': [{'task': 'BlocksInBinTask',
                                  'successes': {m: 3 for m in METHODS}}]}
        self.catalog = {'rows': [dict(method=m, task='BlocksInBinTask', trial=i,
            scope='historical baseline' if m=='pi05_only' else 'primary',
            status='success' if i<3 else 'failure') for m in METHODS for i in range(5)]}

    def test_first_five_preserve_failures_and_ignore_extras(self):
        self.catalog['rows'].append({**self.catalog['rows'][0], 'trial': 5})
        rows = selected_rows(self.catalog, self.summary)
        self.assertEqual(len(rows), 15)
        self.assertEqual(sum(r['status']=='failure' for r in rows), 6)
        self.assertEqual([r['trial'] for r in rows[:5]], list(range(5)))

    def test_missing_or_duplicate_final_slots_rejected(self):
        for rows in (self.catalog['rows'][:-1], self.catalog['rows']+[self.catalog['rows'][0]]):
            with self.assertRaisesRegex(ValueError, 'one final recording'):
                selected_rows({'rows': rows}, self.summary)

    def test_incomplete_and_changed_outcomes_rejected(self):
        for status, error in [('incomplete', 'Incomplete'), ('failure', 'Outcome mismatch')]:
            catalog = deepcopy(self.catalog)
            catalog['rows'][0]['status'] = status
            with self.assertRaisesRegex(ValueError, error):
                selected_rows(catalog, self.summary)

    def test_attachment_rejects_escaping_media_paths(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'robolab-gallery.json').write_text(json.dumps({'cases':[{}]*150}))
            (root/'robolab-gallery-manifest.json').write_text(json.dumps({
                'videos':150,'files':[{'path':'../private.mp4'}]}))
            with self.assertRaisesRegex(ValueError, 'Invalid gallery media path'):
                attach_gallery(root, root/'site', {'publication':{}})

    def test_release_prefix_rejects_credentials_queries_and_other_hosts(self):
        base = 'https://github.com/example/report/releases/download/recordings/'
        self.assertEqual(release_video_base(base), base)
        for value in (base+'?token=test', base+'#fragment', base.replace('https:', 'http:'),
                      base.replace('github.com', 'user:password@github.com'),
                      base.replace('github.com', 'github.com.invalid'), base.rstrip('/')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                release_video_base(value)

    def test_release_attachment_preserves_hashes_outcomes_and_local_option(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)/'bundle'
            media = root/'media/robolab'
            media.mkdir(parents=True)
            files, cases = [], []
            for index in range(150):
                paths = {}
                for suffix in ('.mp4', '.jpg'):
                    path = media/f'trial_{index:03d}{suffix}'
                    path.write_bytes(f'original fixture {index}{suffix}'.encode())
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    files.append(dict(path=path.relative_to(root).as_posix(),
                                      bytes=path.stat().st_size, sha256=digest))
                    paths[suffix] = files[-1]
                cases.append(dict(id=str(index), status='success' if index%2 else 'failure',
                                  video=paths['.mp4']['path'], poster=paths['.jpg']['path'],
                                  video_sha256=paths['.mp4']['sha256']))
            manifest = dict(videos=150, files=files, original_media_unchanged=True)
            (root/'robolab-gallery.json').write_text(json.dumps(dict(cases=cases)))
            (root/'robolab-gallery-manifest.json').write_text(json.dumps(manifest))
            base = 'https://github.com/example/report/releases/download/recordings/'
            target = Path(folder)/'hosted'
            data = {'publication': {}}
            attached = attach_gallery(root, target, data, video_base_url=base)
            self.assertTrue(data['publication']['robolabGalleryAvailable'])
            self.assertEqual(len(attached['external_files']), 150)
            self.assertEqual(len(attached['files']), 150)
            self.assertEqual(len(list(target.rglob('*.mp4'))), 0)
            self.assertEqual(len(list(target.rglob('*.jpg'))), 150)
            output = json.loads((target/'robolab-gallery.json').read_text())
            for before, after in zip(cases, output['cases']):
                self.assertEqual(after['video_url'], base+Path(before['video']).name)
                self.assertEqual({k:v for k,v in after.items() if k!='video_url'}, before)
            self.assertEqual(json.loads((root/'robolab-gallery-manifest.json').read_text()), manifest)
            local = Path(folder)/'local'
            attached_local = attach_gallery(root, local, {'publication': {}})
            self.assertNotIn('external_files', attached_local)
            self.assertEqual(len(list(local.rglob('*.mp4'))), 150)
            (media/'trial_000.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                attach_gallery(root, Path(folder)/'bad', {'publication': {}}, video_base_url=base)


if __name__ == '__main__':
    unittest.main()
