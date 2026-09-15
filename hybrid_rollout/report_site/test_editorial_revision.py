"""Offline checks for the requested report restructuring and evidence selection."""
import json
from collections import Counter
from pathlib import Path
import hashlib
import unittest

from .package_article import public_snapshot

HERE = Path(__file__).resolve().parent
CONTENT = HERE/'app/src/content/report'

class EditorialRevisionTests(unittest.TestCase):
    def test_selection_is_unique_complete_and_bounded(self):
        selected=json.loads((CONTENT/'video-selection.json').read_text())
        self.assertEqual(len(selected),21)
        self.assertEqual(len({r['id'] for r in selected}),21)
        self.assertEqual([r['number'] for r in selected],list(range(1,22)))
        self.assertEqual(Counter(r['section'] for r in selected),{
            'grounding':4,'contact':4,'reasoning':4,'failures':4,'gpt-only':3,'direct-control':2})
        self.assertTrue(any(r['id']=='clip__6ba3a538a3405206181aa6c6__0' for r in selected))

    def test_public_snapshot_keeps_exact_quantitative_queries(self):
        original=json.loads((HERE/'app/src/data.json').read_text())
        public=public_snapshot(original)
        for query in ('models','tasks','costs','intervention'):
            self.assertEqual(public['queries'][query]['rows'],original['queries'][query]['rows'])
        self.assertEqual(len(public['queries']['cases']['rows']),100)
        self.assertEqual(sum(r['score'] is None for r in public['queries']['cases']['rows']),2)
        self.assertEqual(len(public['queries']['clips']['rows']),20)
        self.assertEqual(len(public['queries']['featured_case']['rows']),1)
        self.assertNotIn('annotations',public['reportData'])
        self.assertTrue(all('comment' not in r for r in public['queries']['clips']['rows']))

    def test_full_skill_sources_have_exact_hashes(self):
        docs=json.loads((CONTENT/'skill-documents.json').read_text())
        self.assertEqual(len(docs),7)
        for doc in docs:
            path=HERE.parents[1]/doc['source_name']
            self.assertEqual(path.read_text(),doc['body'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),doc['sha256'])

    def test_prose_retains_findings_without_removed_process_notes(self):
        copy=json.loads((CONTENT/'article-copy.json').read_text())
        self.assertIn('14.4%',copy['hero-intro'])
        self.assertIn('6、2、1、1',copy['paired-settings'])
        self.assertIn('$p_t$',copy['notation'])
        self.assertNotIn('$s_t$',copy['notation'])
        self.assertIn(r'\mathbf{x}',copy['notation'])
        for key in ['hero-intro','paired-settings','chart-scope','table-footnote','limits-copy','conclusion-copy']:
            for text in ['Fast','DAgger','48 条有分','2 条','完整性与中断','版本差异']:
                self.assertNotIn(text,copy[key])

if __name__=='__main__':
    unittest.main()
