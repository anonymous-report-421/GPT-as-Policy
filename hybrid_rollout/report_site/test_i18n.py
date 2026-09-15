"""Translation coverage and unchanged-evidence checks; no model or simulator calls."""
import json
import re
import ast
import hashlib
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONTENT=HERE/'app/src/content/report'

class ReportLanguageTests(unittest.TestCase):
    def setUp(self):
        self.zh=json.loads((CONTENT/'article-copy.json').read_text())
        self.en=json.loads((CONTENT/'article-copy.en.json').read_text())
        self.locale=json.loads((CONTENT/'locale.en.json').read_text())

    def test_all_blocks_links_equations_and_empty_blocks_match(self):
        self.assertEqual(self.zh.keys(),self.en.keys())
        for key,text in self.zh.items():
            translated=self.en[key]
            self.assertEqual(bool(text),bool(translated),key)
            self.assertFalse(re.search(r'[\u4e00-\u9fff]',translated),key)
            for pattern in [r'https://[^)]+',r'\$[^$]+\$']:
                self.assertEqual(re.findall(pattern,text),re.findall(pattern,translated),key)

    def test_important_numeric_claims_match(self):
        for key,tokens in {
            'hero-intro':['14.4%','85.6%','48%','62.60','26%','37.81','15.67%','24.43','44.8%'],
            'intervention-copy':['50','42,750','36,576','6,174','85.6%','14.4%'],
            'cost-heading':['48%','26%','624.8M','1.13B','44.8%','2.05'],
            'physical-time-correction':['50','0.04'],
            'method-copy':['14','50','1–15','1–5'],
            'paired-settings':['0–72%'],
        }.items():
            for token in tokens:
                self.assertIn(token,self.zh[key],key)
                self.assertIn(token,self.en[key],key)

    def test_localized_collections_are_complete(self):
        snapshot=json.loads((HERE/'app/src/data.json').read_text())
        selected=json.loads((CONTENT/'video-selection.json').read_text())
        docs=json.loads((CONTENT/'skill-documents.json').read_text())
        self.assertEqual(set(self.locale['tasks']),{r['id'] for r in snapshot['queries']['tasks']['rows']})
        self.assertEqual(set(self.locale['captions']),{r['id'] for r in selected})
        self.assertEqual(set(self.locale['references']),{r['id'] for r in snapshot['reportData']['references']})
        for row in self.locale['tasks'].values():
            self.assertEqual(set(row),{'label','category','initial','goal'})
        for doc in docs:
            self.assertIn(doc['title'],self.locale['ui'])
            self.assertFalse(re.search(r'[\u4e00-\u9fff]',doc['body']))
        translations=[list(self.locale['ui'].values()),self.locale['tasks'],self.locale['captions'],self.locale['references'],self.locale['infrastructure']]
        self.assertFalse(re.search(r'[\u4e00-\u9fff]',json.dumps(translations,ensure_ascii=False)))

    def test_chinese_copy_selection_and_skill_sources_unchanged(self):
        # Frozen reviewed inputs; also verifiable in the history-free public release.
        expected = {
            # User-approved copy, including the TL;DR details merged into the abstract (2026-09-14).
            'article-copy.json':'c9d3f553b76a2e6665c4b842d37b187407cfe4abe664a61d231616fed74fb504',
            'reference-notes.json':'fa1594ad412907a4f8e72f9820d3e5beaa37cd031bdea686f68d573cf876b8eb',
            'video-selection.json':'f44dfd7820fa7fba87e12b588bc842109ae2736162cb957f0fc75bf632d3120a',
            'skill-documents.json':'21f98feb61e6166fd22dbfb8b75f80df0ef6808c67868405a76fc6807a8e022b',
        }
        for name,digest in expected.items():
            contents=(CONTENT/name).read_bytes()
            self.assertEqual(hashlib.sha256(contents).hexdigest(),digest,name)

    def test_notation_keeps_definitions_without_redundant_comparison(self):
        self.assertNotIn('使用不同记号',self.zh['notation'])
        self.assertNotIn('is distinct from',self.en['notation'])
        for copy in (self.zh,self.en):
            self.assertIn('$p_t$',copy['notation'])
            self.assertIn(r'$(\mathbf{x},R,g)$',copy['notation'])

    def test_robolab_protocol_paragraph_removed_bilingually(self):
        self.assertEqual(self.zh['robolab-protocol'],'')
        self.assertEqual(self.en['robolab-protocol'],'')
        self.assertNotIn('<Prose id="robolab-protocol" />',(CONTENT/'RoboLabResults.jsx').read_text())

    def test_no_visible_usage_heading_and_all_literals_translated(self):
        source=(CONTENT/'ReportContent.jsx').read_text()
        self.assertNotIn('用量口径 · 各 50 个选定 attempt',source)
        self.assertRegex(source,r'id="cost-table"[^>]+showHeading=\{false\}')
        for file in CONTENT.glob('*.jsx'):
            for literal in re.findall(r"\bt\('([^']+)'\)",file.read_text()):
                literal=ast.literal_eval("'"+literal+"'")
                self.assertIn(literal,self.locale['ui'],file.name+': '+literal)

    def test_publication_link_and_bibtex(self):
        publication=json.loads((CONTENT/'publication.json').read_text())
        self.assertEqual(publication['repository'],'https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy')
        for text in ['Su, Jiayi and Zheng, Yixin and Yan, Mi and Yi, Li and Zhang, Zhizheng and Wang, He','2026',publication['repository']]:
            self.assertIn(text,publication['bibtex'])
        self.assertIn('<Citation />',(CONTENT/'ReportContent.jsx').read_text())

    def test_bilingual_policy_title_and_citation_match(self):
        en='GPT 6 Astra as an Embodied Policy'
        zh='GPT 6 Astra 作为具身策略'
        self.assertEqual(self.en['hero-title'].splitlines()[0],'# '+en)
        self.assertEqual(self.zh['hero-title'].splitlines()[0],'# '+zh)
        for title in (en,zh):
            self.assertIn(title,(CONTENT/'Language.jsx').read_text())
        bib=json.loads((CONTENT/'publication.json').read_text())['bibtex']
        self.assertIn(en,bib.replace('{','').replace('}',''))
        snapshot=json.loads((HERE/'app/src/data.json').read_text())
        self.assertEqual(snapshot['title'],en)

    def test_takeaway_bilingual_scope_and_author_placement(self):
        copy=json.loads((CONTENT/'takeaway-copy.json').read_text())
        self.assertEqual(set(copy),{'zh','en'})
        for text in copy.values():
            self.assertTrue(text.startswith('## '))
            self.assertNotIn('\n',text)
            self.assertNotIn('%',text)
            self.assertFalse(re.search(r'\bGPT\b(?! 6 Astra)',text))
        self.assertIn('仅靠推理并不能保证及时响应，也不能保证连贯、稳健的运动轨迹',copy['zh'])
        self.assertIn('习得的动作能力与推理相结合',copy['zh'])
        self.assertIn('延迟仍是尚未解决的挑战',copy['zh'])
        self.assertEqual(copy['en'],
            '## GPT 6 Astra demonstrates strong System 2 reasoning for embodied tasks, '
            'but reasoning alone does not guarantee timely responses or coherent, robust trajectories. '
            r'Pairing Astra with $\pi_{0.5}$ improves performance by combining System 2 deliberation '
            'with System 1 sensorimotor skills—highlighting the value of learned action alongside reasoning, '
            'while latency remains an unresolved challenge.')
        for text in copy.values():
            self.assertIn('System 2',text)
            self.assertIn('System 1',text)
            self.assertIn(r'$\pi_{0.5}$',text)
        self.assertLess(len(copy['zh']),250)
        self.assertLess(len(copy['en'].split()),90)
        for language,abstract in [('zh',self.zh['hero-intro']),('en',self.en['hero-intro'])]:
            self.assertEqual(abstract.count('\n\n'),3,language)
            self.assertIn('RoboDojo',abstract)
            self.assertIn(r'$\pi_{0.5}$',abstract)
            self.assertIn('44.8%',abstract)
            self.assertNotIn('这些结果体现了具身 Policy 的重要价值',abstract)
            self.assertNotIn('These results highlight the value',abstract)
        self.assertIn('逆运动学',self.zh['hero-intro'])
        self.assertIn('inverse kinematics',self.en['hero-intro'])
        self.assertIn('延迟仍是实时部署尚未解决的一大难题',self.zh['hero-intro'])
        self.assertIn('latency remains a hugely unresolved issue to real-time deployment',self.en['hero-intro'])
        self.assertIn('物理过程没有暂停键',self.zh['hero-intro'])
        self.assertIn('real-world physics has no pause button',self.en['hero-intro'])
        self.assertEqual(hashlib.sha256((CONTENT/'article-copy.en.json').read_bytes()).hexdigest(),
            '38c2d48136e716976cb5b6a9be6fdfbfd055362a4a9b81245981140974afb2e9')
        self.assertFalse(re.search(r'[\u4e00-\u9fff]',copy['en']))
        source=(CONTENT/'ReportContent.jsx').read_text()
        self.assertLess(source.index('className="rr-authors"'),source.index('className="rr-tldr"'))
        self.assertLess(source.index('className="rr-tldr"'),source.index('className="rr-project-links"'))
        self.assertIn('<Prose id="hero-tldr">{takeawayCopy[language]}</Prose>',source)

    def test_authors_order_equal_contribution_and_profile_links(self):
        authors=json.loads((CONTENT/'authors.json').read_text())
        self.assertEqual([a['name'] for a in authors['authors']],['Jiayi Su','Yixin Zheng','Mi Yan','Li Yi','Zhizheng Zhang','He Wang'])
        self.assertEqual([a.get('equalContribution',False) for a in authors['authors']],[True,True,False,False,False,False])
        self.assertEqual([a['name'] for a in authors['authors'] if a.get('correspondingAuthor')], ['Li Yi', 'Zhizheng Zhang', 'He Wang'])
        self.assertEqual(authors['affiliations'],[])
        self.assertTrue(all('email' not in a for a in authors['authors']))
        self.assertEqual([a['url'] for a in authors['authors']], [
            'https://shuyumo2003.github.io/',
            'https://steveouo.github.io/',
            'https://miyandoris.github.io/',
            'https://ericyi.github.io/',
            'https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en',
            'https://hughw19.github.io/',
        ])

    def test_user_supplied_galbot_logo_preserved(self):
        logo=(CONTENT.parent/'assets/galbot-logo.png').read_bytes()
        self.assertEqual(hashlib.sha256(logo).hexdigest(),
            '760f9afbd645287eae72e67665b8e290f137ed7032c4171be071aa8e559fa9ba')
        source=(CONTENT/'ReportContent.jsx').read_text()
        self.assertIn("import galbotWordmark from '../assets/galbot-logo.png';",source)
        self.assertIn('src={galbotWordmark} width="1412" height="446" alt="Galbot"',source)

    def test_every_video_takeaway_uses_recorded_policy_and_reviewed_caption(self):
        data=json.loads((HERE/'app/src/data.json').read_text())
        rows={r['id']:r for q in ('clips','featured_case') for r in data['queries'][q]['rows']}
        selected=json.loads((CONTENT/'video-selection.json').read_text())
        self.assertEqual(len(selected),21)
        self.assertEqual([rows[r['id']]['method'] for r in selected],['mix']*16+['gpt']*5)
        for row in selected:
            self.assertTrue(row['caption'].strip())
            self.assertTrue(self.locale['captions'][row['id']].strip())
        video=(CONTENT/'VideoCard.jsx').read_text()
        self.assertIn('data-method={clip.method}',video)
        self.assertEqual(video.count('<VideoContext clip={clip} />'),2)
        self.assertIn("t('行为分析：')",video)
        self.assertNotIn('片段看点',video)
        self.assertEqual(self.locale['ui']['行为分析：'],'Clip takeaway: ')

    def test_direct_display_name_is_consistent(self):
        for name in ('article-copy.json','article-copy.en.json','Math.jsx'):
            text=(CONTENT/name).read_text()
            self.assertIn('GPT 6 Astra（direct）',text)
            self.assertNotIn('GPT 6 Astra Direct',text)
        gallery=(HERE/'web/gallery.js').read_text()
        self.assertIn("['gpt','pure_astra'].includes(row.id)?'GPT 6 Astra（direct）'",gallery)

    def test_direct_comparison_distinguishes_plan_diversity_from_robustness(self):
        self.assertIn('方案的多样性并不等于执行的鲁棒性',self.zh['gpt-only-heading'])
        self.assertIn('diversity of plans is not the same as robustness of execution',self.en['gpt-only-heading'])
        self.assertIn('成熟动作与物体交互先验',self.zh['gpt-only-heading'])
        self.assertIn('action and object-interaction priors',self.en['gpt-only-heading'])

if __name__=='__main__':
    unittest.main()
