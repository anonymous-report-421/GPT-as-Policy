"""Publication boundaries and bilingual RoboLab supplement checks."""
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from .package_article import public_snapshot, public_gallery, load_media_replacements, sha, version_gallery_assets

HERE=Path(__file__).resolve().parent

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.data=json.loads((HERE/'app/src/data.json').read_text())
        public_evidence=HERE.parents[1]/'public_results/data.json'
        if public_evidence.is_file():
            self.data=json.loads(public_evidence.read_text())

    def test_public_export_is_idempotent(self):
        first=public_snapshot(self.data)
        self.assertEqual(public_snapshot(first),first)

    def test_article_only_drops_versioned_full_video_links(self):
        full=public_snapshot(self.data,True)
        for row in full['queries']['cases']['rows']:
            row['video_url']=row['video']+'?v=synthetic'
            row['poster_url']=row['poster']+'?v=synthetic'
        article=public_snapshot(full)
        for row in article['queries']['cases']['rows']:
            self.assertFalse({'video','poster','video_url','poster_url'} & row.keys())

    def test_gallery_assets_have_content_versions(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            page='<link href="gallery.css"><script src="data.js"></script><script src="gallery.js"></script>'
            (root/'gallery.html').write_text(page)
            for name in ('gallery.css','gallery.js','data.js'):
                (root/name).write_text('synthetic '+name)
            version_gallery_assets(root)
            rendered=(root/'gallery.html').read_text()
            for name in ('gallery.css','gallery.js','data.js'):
                self.assertIn(name+'?v='+sha(root/name)[:16],rendered)

    def test_full_gallery_is_public_and_complete(self):
        public=public_snapshot(self.data,True)
        gallery=public_gallery(public)
        self.assertEqual(len(gallery['cases']),100)
        self.assertEqual(len(gallery['clips']),43)
        for method in ('mix','gpt'):
            rows=[r for r in gallery['cases'] if r['method']==method]
            self.assertEqual(len(rows),50)
            self.assertEqual(len({r['case_id'] for r in rows}),50)
        self.assertTrue(all(r.get('video') for r in gallery['cases']))
        self.assertNotIn('annotations',public['reportData'])
        for row in gallery['cases']+gallery['clips']:
            self.assertFalse({'version','controller_version','run_id','comment','tags'}&row.keys())

    def test_robolab_rates_and_translations(self):
        tasks=self.data['queries']['robolab_tasks']['rows']
        models=self.data['queries']['robolab_models']['rows']
        self.assertEqual(len(tasks),10)
        expected={'pure_astra':49,'astra_pi05':46,'pi05_only':18,'cosmos_nano_policy':18,'dreamzero':17}
        for model in models:
            self.assertEqual(model['successes'],expected[model['id']])
            self.assertEqual(model['episodes'],50)
            self.assertEqual(model['sr'],model['successes']*2)
            self.assertEqual(sum(t['successes'][model['id']] for t in tasks),model['successes'])
        public=public_snapshot(self.data)
        self.assertEqual(public['queries']['robolab_models'],self.data['queries']['robolab_models'])
        labels=json.loads((HERE/'app/src/content/report/robolab-locales.json').read_text())
        self.assertEqual(set(labels),{r['task'] for r in tasks})
        copies=[json.loads((HERE/'app/src/content/report'/name).read_text()) for name in ('article-copy.json','article-copy.en.json')]
        for key in ('robolab-results-summary','robolab-figure-caption'):
            for token in ('49/50','46/50','18/50'):
                self.assertTrue(all(token in c[key] for c in copies))
        self.assertNotIn('RoboLab',copies[0]['hero-intro'])
        self.assertNotIn('RoboLab',copies[1]['hero-intro'])

    def test_opening_robolab_reuses_reviewed_three_baselines(self):
        content=HERE/'app/src/content/report'
        component=(content/'LeadingRoboLabRanking.jsx').read_text()
        self.assertIn("reviewedRows('robolab_models')",component)
        self.assertIn('queryId="robolab_models"',component)
        self.assertIn('height={models.length * 40 + 28}',component)
        self.assertNotIn('官方总榜',component)
        self.assertIn('valueDomain={[0,100]}',component)
        self.assertIn("t('RoboLab（zero-shot）')",component)
        page=(content/'ReportContent.jsx').read_text()
        self.assertLess(page.index('<LeadingRanking models={models} />'),page.index('<LeadingRoboLabRanking />'))
        self.assertLess(page.index('<LeadingRoboLabRanking />'),page.index('id="abstract"'))
        baselines=json.loads((HERE.parents[1]/'report_web/robolab-baselines.json').read_text())
        self.assertFalse(baselines['initial_states_paired'])
        models={r['id']:r for r in self.data['queries']['robolab_models']['rows']}
        tasks={r['task']:r for r in self.data['queries']['robolab_tasks']['rows']}
        self.assertEqual({m['method'] for m in baselines['methods']},{'pi05_only','cosmos_nano_policy','dreamzero'})
        for method in baselines['methods']:
            self.assertEqual({t['task'] for t in method['tasks']},set(tasks))
            total=0
            for task in method['tasks']:
                trials=task['trials']
                keys=[(r['run'],r['episode'],r['env_id']) for r in trials]
                self.assertEqual(len(trials),5)
                self.assertEqual(keys,sorted(set(keys)))
                successes=sum(r['success'] for r in trials)
                self.assertEqual(successes,tasks[task['task']]['successes'][method['method']])
                total+=successes
            self.assertEqual(total,models[method['method']]['successes'])
            self.assertEqual(method['source_sha256'],models[method['method']]['source_sha256'])

    def test_opening_robodojo_score_only_and_body_retains_success_rate(self):
        content=HERE/'app/src/content/report'
        lead=(content/'LeadingRanking.jsx').read_text()
        chart=(content/'ModelChart.jsx').read_text()
        self.assertIn('models={models} icons large',lead)
        self.assertNotIn(' paired ',lead)
        self.assertIn('valueDomain={[0,80]}',lead)
        self.assertIn("y:metric==='score'?'Score':rateLabel",chart)
        self.assertIn('id="panel-sr-ranking" models={models} metric="sr"',(content/'ReportContent.jsx').read_text())
        self.assertIn('stackable:false',chart)
        self.assertIn('Score:r.score,[rateLabel]:r.sr',chart)
        self.assertIn('sourceRows={models}',chart)
        self.assertNotIn(' paired ',(content/'LeadingRoboLabRanking.jsx').read_text())
        for model in self.data['queries']['models']['rows']:
            self.assertTrue(0<=model['score']<=80)
            self.assertTrue(0<=model['sr']<=80)

    def test_robolab_opening_cites_existing_benchmark_reference(self):
        refs=self.data['reportData']['references']
        papers=[r for r in refs if r['id']=='robolab']
        self.assertEqual(len(papers),1)
        self.assertEqual(papers[0]['url'],'https://arxiv.org/abs/2604.09860')
        self.assertEqual(papers[0]['title'],'RoboLab: A High-Fidelity Simulation Benchmark for Analysis of Task Generalist Policies')
        self.assertIn('Yang',papers[0]['author'])
        component=(HERE/'app/src/content/report/LeadingRoboLabRanking.jsx').read_text()
        self.assertIn("findIndex(row=>row.id==='robolab')",component)
        self.assertIn('reference(snapshot.reportData.references[paperIndex])',component)
        self.assertIn('sourcePreviews={paperPreview}',component)
        self.assertIn('paperIndex+1',component)

    def test_official_overall_references_are_separate_and_reproducible(self):
        from .robolab_opening import add_overall_references
        reference=json.loads((HERE/'app/src/content/report/robolab-official-overall.json').read_text())
        updated=add_overall_references(self.data,reference)
        for key,value in self.data['queries'].items():
            if key!='robolab_opening_models':
                self.assertEqual(updated['queries'][key],value,key)
        self.assertEqual(updated['reportData'],self.data['reportData'])
        self.assertEqual(add_overall_references(updated,reference),updated)
        overview=updated['queries']['robolab_opening_models']
        self.assertEqual(overview['rows'][:5],self.data['queries']['robolab_models']['rows'])
        official=overview['rows'][5:]
        self.assertEqual([r['sr'] for r in official],[22.9,15.5,7.2,5.0])
        self.assertEqual([r['successes'] for r in official],[275,186,87,60])
        self.assertTrue(all(r['episodes']==1200 and r['task_count']==120 and r['instruction_variant']=='default' and r['reference_label']=='†' for r in official))
        published=public_snapshot(updated)
        self.assertNotIn('robolab_opening_models',published['queries'])
        self.assertEqual(published['queries']['robolab_models'],self.data['queries']['robolab_models'])
        self.assertEqual(updated['queries']['robolab_opening_models'],overview)

    def test_replacements_resolve_relative_to_manifest(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'media').mkdir();video=root/'media/example.mp4'
            video.write_bytes(b'synthetic video fixture')
            row=dict(path='media/example.mp4',source='media/example.mp4',bytes=video.stat().st_size,sha256=sha(video))
            manifest=root/'replacements.json'
            manifest.write_text(json.dumps(dict(complete=True,files=[row])))
            parsed=load_media_replacements(manifest,{row['path']})
            self.assertEqual(parsed[row['path']]['resolved_source'],video)
            with self.assertRaisesRegex(ValueError,'Missing new media'):
                load_media_replacements(manifest,{'media/missing.jpg'})
            for bad in ('../outside.mp4','/tmp/outside.mp4','secrets/token'):
                manifest.write_text(json.dumps(dict(complete=True,files=[{**row,'source':bad}])))
                with self.assertRaisesRegex(ValueError,'relative media paths'):
                    load_media_replacements(manifest,{row['path']})
            manifest.write_text(json.dumps(dict(complete=True,files=[row,row])))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                load_media_replacements(manifest,{row['path']})
            manifest.write_text(json.dumps(dict(complete=False,files=[row])))
            with self.assertRaisesRegex(ValueError,'not completed'):
                load_media_replacements(manifest,{row['path']})
            manifest.write_text(json.dumps(dict(complete=True,files=[row])))
            video.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'integrity'):
                load_media_replacements(manifest,{row['path']})

if __name__=='__main__':unittest.main()
