"""Model-free checks on the report's frozen calculations and public payload."""
import json
from pathlib import Path
import unittest
from .build import parse_leaderboard, SHARED, OFFICIAL, SOURCE_SHA, OFFICIAL_SHA, sha

SITE = Path(__file__).resolve().parents[2]/'runtime/report_site_20260913_v3'

class ReportDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SITE/'data.json').exists():
            raise unittest.SkipTest('Build the report data first')
        cls.data=json.loads((SITE/'data.json').read_text())

    def test_frozen_sources(self):
        self.assertEqual(self.data['source_sha256'],SOURCE_SHA)
        self.assertEqual(sha(SHARED/OFFICIAL),OFFICIAL_SHA)

    def test_top10_includes_pi05_once(self):
        rows=parse_leaderboard((SHARED/OFFICIAL).read_text())
        self.assertEqual(len(rows),10)
        self.assertEqual(sum(r['name']=='Pi-05' for r in rows),1)
        self.assertEqual([r['name'] for r in rows[:4]],['DM0.5','GalaxeaVLA (G0.5)','Xiaomi-Robotics-1','OpenWAM-α'])

    def test_complete_panel_and_seed_pairing(self):
        rows=self.data['cases'];self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        for t in self.data['tasks']:
            for method in ('mix','gpt'):
                self.assertEqual(sum(r['task']==t['id'] and r['method']==method for r in rows),5)
        for m in (r for r in rows if r['method']=='mix'):
            g=next(r for r in rows if r['method']=='gpt' and r['case_id']==m['case_id'])
            self.assertEqual(m['seeds'],g['seeds']);self.assertEqual(m['instruction'],g['instruction'])
            self.assertEqual(m['limit'],g['limit'])

    def test_scores_and_missingness(self):
        models={r['name']:r for r in self.data['models']}
        self.assertAlmostEqual(models['mix']['score'],62.6)
        self.assertAlmostEqual(models['gpt']['score'],37.8125)
        self.assertAlmostEqual(models['Pi-05']['score'],24.43)
        self.assertEqual(models['mix']['sr'],48)
        self.assertEqual(models['gpt']['sr'],26)
        missing=[r for r in self.data['cases'] if r['score'] is None]
        self.assertEqual(len(missing),2)
        self.assertTrue(all(r['method']=='gpt' and r['status']=='idle' and not r['native_complete'] for r in missing))

    def test_step_share_not_chunk_share(self):
        m=self.data['totals']['mix']
        self.assertEqual(m['steps'],42750);self.assertEqual(m['corrected_steps'],6174)
        self.assertEqual(round(100*m['corrected_steps']/m['steps'],1),14.4)
        self.assertEqual(sum(r['steps'] for r in self.data['cases'] if r['method']=='mix'),m['steps'])

    def test_public_payload_and_clips(self):
        forbidden={'archive','rpc_out','auth','access_token','refresh_token','reasoning_text'}
        def walk(x):
            if isinstance(x,dict):
                self.assertFalse(forbidden.intersection(x))
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        walk(self.data)
        self.assertEqual(len(self.data['clips']),43)
        self.assertEqual(sum('section' in c for c in self.data['clips']),12)
        for c in self.data['clips']:
            self.assertGreater(c['frames'],0)
            self.assertEqual(c['end_frame_exclusive']-c['start_frame'],c['frames'])
            self.assertEqual(c['video_ui'],'existing_debug_placeholder')

if __name__=='__main__':unittest.main()
