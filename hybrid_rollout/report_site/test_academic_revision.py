"""Model-free regression checks for the separate, historical homepage excerpt."""
import json
from pathlib import Path
import unittest

from .academic_revision import SOURCE, SOURCE_SHA
from .build import SHARED, probe, sha

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / 'runtime/report_site_20260913_v3'


class AcademicRevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SITE / 'hero_clip.json').exists():
            raise unittest.SkipTest('Prepare the academic report excerpt first')
        cls.hero = json.loads((SITE / 'hero_clip.json').read_text())
        cls.receipt = json.loads((ROOT / 'runtime/report_academic_revision_receipt.json').read_text())

    def test_exact_frames_and_original_unchanged(self):
        h = self.hero
        self.assertEqual((h['frame_index_base'], h['start_frame'], h['end_frame_inclusive']), (0, 493, 1309))
        self.assertEqual(h['end_frame_exclusive'] - h['start_frame'], 817)
        self.assertEqual(h['duration'], 32.68)
        video = SITE / h['video']
        self.assertEqual(sha(video), h['video_sha256'])
        meta = probe(video)
        self.assertEqual((int(meta['nb_frames']), meta['r_frame_rate'], meta['width'], meta['height']),
                         (817, '25/1', 1280, 680))
        self.assertEqual(sha(SHARED / SOURCE / 'controller/debug_video/debug_rollout.mp4'), SOURCE_SHA)

    def test_historical_result_not_added_to_v3(self):
        self.assertTrue(self.hero['native_complete'])
        self.assertFalse(self.hero['native_success'])
        self.assertEqual(self.hero['native_score'], 0.15)
        self.assertFalse(self.hero['included_in_v3_statistics'])
        data = json.loads((SITE / 'data.json').read_text())
        self.assertEqual(len(data['cases']), 100)
        self.assertNotIn(self.hero['id'], [r['id'] for r in data['clips']])

    def test_original_evidence_and_queries_preserved(self):
        # The user subsequently requested a gallery redesign. Its original
        # presentation stays in the pre-redesign backup; evidence stays live.
        revised_ui = {'gallery.html', 'gallery.css', 'gallery.js', 'media/report-cjk.woff2'}
        for filename, digest in self.receipt['unchanged'].items():
            with self.subTest(file=filename):
                if filename in revised_ui:
                    preserved = ROOT / 'runtime/report_before_fresh_20260913' / filename
                    self.assertEqual(sha(preserved), digest)
                else:
                    self.assertEqual(sha(SITE / filename), digest)
        before = json.loads((ROOT / 'runtime/report_before_academic_20260913/app_snapshot.json').read_text())
        current = json.loads((Path(__file__).parent / 'app/src/data.json').read_text())
        for key, query in before['queries'].items():
            if key != 'featured_case':
                with self.subTest(query=key):
                    self.assertEqual(current['queries'][key], query)


if __name__ == '__main__':
    unittest.main()
