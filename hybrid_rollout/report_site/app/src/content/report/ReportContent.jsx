import React, { useState } from 'react';
import { DataComponent, RichNarrative, useDataApp } from '../../data-app-public.jsx';
import snapshot from '../../data.json';
import './report.css';
import {ReportLanguage, LanguageSwitch, useReportLanguage} from './Language.jsx';
import {VideoCard} from './VideoCard.jsx';
import {LeadingRanking} from './LeadingRanking.jsx';
import {LeadingRoboLabRanking} from './LeadingRoboLabRanking.jsx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import {MathText, ModelName, PI_LATEX, GPT_NAME, modelLabel} from './Math.jsx';
import authors from './authors.json';
import {ModelChart} from './ModelChart.jsx';
import videoSelection from './video-selection.json';
import {SkillDetails} from './SkillDetails.jsx';
import referenceNotes from './reference-notes.json';
import publication from './publication.json';
import {Citation} from './Citation.jsx';
import {RoboLabResults} from './RoboLabResults.jsx';
import takeawayCopy from './takeaway-copy.json';
import galbotWordmark from '../assets/galbot-logo.png';

const D = snapshot.reportData;
const galleryAvailable = snapshot.publication?.galleryAvailable !== false;
const robolabGalleryAvailable = snapshot.publication?.robolabGalleryAvailable === true;
const galleryIds = new Set([...snapshot.queries.cases.rows, ...snapshot.queries.clips.rows].map(r => r.id));
const fmt = (x, n = 2) => x == null ? 'N/A' : Number(x).toFixed(n);
const num = x => Number(x).toLocaleString('en-US');
const name = modelLabel;
const heatStyle = value => {
  if(value==null) return {};
  const x=Number(value)/100, rgb=[247+(33-247)*x,250+(91-250)*x,255+(177-255)*x].map(Math.round);
  const linear=rgb.map(c=>c/255).map(c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4);
  const luminance=linear[0]*.2126+linear[1]*.7152+linear[2]*.0722;
  return {backgroundColor:`rgb(${rgb.join(',')})`,color:1.05/(luminance+.05)>=4.5?'#fff':'#000'};
};
function Prose({ id, children }) {
  const {copy,reference,language,narrativeId}=useReportLanguage();
  const previews=Object.fromEntries(D.references.map(original=>{
    const r=reference(original);
    return [r.url,{title:r.title,summary:language==='zh'?(referenceNotes[r.id]||r.summary):r.summary,source:r.author,approvedForReport:true}];
  }));
  const value = (copy[id] ?? children ?? '').replaceAll('π0.5', () => '$' + PI_LATEX + '$');
  const numbered = id.startsWith('reference-') ? value : value.replace(
    /\[([^\]]+)\]\((https:\/\/[^)]+)\)/g, (link, label, url) => {
      const index = D.references.findIndex(r => r.url === url);
      return index < 0 ? link : label + ' [\\[' + (index + 1) + '\\]](' + url + ')';
    });
  return <div className="rr-prose"><RichNarrative key={narrativeId(id)} id={narrativeId(id)} value={numbered} sourcePreviews={previews} remarkPlugins={[remarkMath]} rehypePlugins={[[rehypeKatex,{trust:false,strict:'error'}]]} /></div>;
}
function ClipPlayer({ clip, selected }) {
  const {t,caption}=useReportLanguage();
  const target = [clip.paired_gallery_id, clip.id].find(id => id && galleryIds.has(id));
  const title = caption(selected);
  return <div className="rr-clip"><VideoCard clip={{...clip, title}} />
    <div className="rr-video-caption"><Prose id={`clip-${clip.id}-caption`}>{`${t('视频')} ${selected.number}. ${title}`}</Prose></div>
    {galleryAvailable && target && <a className="rr-link" href={`gallery.html?id=${target}`}>{t('在视频库查看 ↗')}</a>}
  </div>;
}

function ClipGroup({ section, clips }) {
  const {t}=useReportLanguage();
  const selected = videoSelection.filter(c => c.section === section && c.id !== 'imitate_sorting_sequence_a5_frames_493_1309');
  const rows = selected.map(c => clips.find(row => row.id === c.id)).filter(Boolean);
  return <DataComponent id={`clips-${section}`} queryId="clips" title={t('行为片段')} showHeading={false} sourceRows={rows} displayRows={rows}>
    <div className="rr-clip-grid" data-reviewed-rows>{rows.map(c => <ClipPlayer key={c.id} clip={c} selected={selected.find(s=>s.id===c.id)} />)}</div>
  </DataComponent>;
}

export function ReportContent() {
  return <ReportLanguage><ReportArticle /></ReportLanguage>;
}
function ReportArticle() {
  const {language,t,task,reference}=useReportLanguage();
  const { reviewedRows } = useDataApp();
  const models = reviewedRows('models'), tasks = reviewedRows('tasks'), clips = reviewedRows('clips');
  const intervention = reviewedRows('intervention'), costs = reviewedRows('costs');
  const hero = reviewedRows('featured_case')[0];
  const [tableMetric, setTableMetric] = useState('score');
  const tableModels = [...D.top_four, 'Pi-05', 'mix', 'gpt'];
  const costRows = costs.map(r => ({ method: name(r.method), steps: r.steps, chunks: r.chunks,
    meanSeconds: r.steps * 0.04 / 50, totalTokens: r.tokens.totalTokens,
    cachedInput: r.tokens.cachedInputTokens, uncachedInput: r.tokens.inputTokens - r.tokens.cachedInputTokens,
    outputTokens: r.tokens.outputTokens }));
  const tableRows = tasks.map(row => ({ task: task(row).label, ...Object.fromEntries(tableModels.map(m => [name(m), (tableMetric==='score'?row.scores:row.rates)[m]])) }));
  tableRows.push({ task: t('总体'), ...Object.fromEntries(tableModels.map(m => [name(m), models.find(x => x.name === m)[tableMetric]])) });
  return <article className="rr-report" lang={language==='zh'?'zh-CN':'en'}>
    <header className="rr-hero">
      <LanguageSwitch />
      <Prose id="hero-title" />
      <div className="rr-authors" aria-label={t('作者信息')}>
        {authors.authors.length ? authors.authors.map((a,i)=>{const href=a.url||(a.email?`mailto:${a.email}`:null);return <span className="rr-author" key={a.name||i}>{href?<a className="rr-author-name" href={href} target={a.url?'_blank':undefined} rel={a.url?'noopener noreferrer':undefined}>{a.name}</a>:<span className="rr-author-name">{a.name}</span>}{a.equalContribution&&<sup aria-describedby="equal-contribution-note">*</sup>}{a.correspondingAuthor&&<sup aria-describedby="corresponding-author-note">†</sup>}{a.affiliation&&<sup>{a.affiliation}</sup>}</span>;}) : <span className="rr-author-placeholder">{t('作者姓名')}</span>}
      </div>
      {authors.authors.some(a=>a.equalContribution||a.correspondingAuthor)&&<div className="rr-author-note">
        {authors.authors.some(a=>a.equalContribution)&&<span id="equal-contribution-note">* {language==='zh'?'同等贡献':'Equal contribution'}</span>}
        {authors.authors.some(a=>a.correspondingAuthor)&&<span id="corresponding-author-note">† {language==='zh'?'通讯作者':'Corresponding authors'}</span>}
      </div>}
      {authors.affiliations.length>0&&<div className="rr-affiliations">{authors.affiliations.map((a,i)=><span key={i}>{a}</span>)}</div>}
      <div className="rr-institution-brand"><img src={galbotWordmark} width="1412" height="446" alt="Galbot" /></div>
      <blockquote className="rr-tldr" aria-label="TL;DR">
        <span className="rr-tldr-label">TL;DR</span>
        <Prose id="hero-tldr">{takeawayCopy[language]}</Prose>
      </blockquote>
      <div className="rr-project-links"><a className="rr-repository-link" href={publication.repository} target="_blank" rel="noopener noreferrer" aria-label={t('在 GitHub 查看代码')}>GitHub</a>{galleryAvailable&&<a href={`gallery.html?lang=${language}`}>{t('RoboDojo 视频库')}</a>}{robolabGalleryAvailable&&<a href={`robolab-gallery.html?lang=${language}`}>{t('RoboLab 视频库')}</a>}<a href="scores.csv">{t('评测数据')}</a><a href="#references">{t('参考文献')}</a></div>
    </header>
    <LeadingRanking models={models} />
    <LeadingRoboLabRanking />
    <section id="abstract" className="rr-section rr-abstract"><Prose id="hero-intro" /></section>

    <section id="motivation" className="rr-section">
      <Prose id="motivation-copy" />
      <Prose id="related-work" />
    </section>

    <section id="method" className="rr-section">
      <Prose id="method-copy" />
      <Prose id="notation" />
      <div className="rr-methods">
        <div className="rr-method"><h3><ModelName model="mix" /></h3><div className="rr-flow">
          <div>{t('图像')} <MathText>{'o_t'}</MathText> · {t('本体状态')} <MathText>{'p_t'}</MathText> · {t('指令')} <MathText>{String.raw`\ell`}</MathText></div><b aria-hidden="true">↓</b>
          <div><MathText>{String.raw`A_t^{\pi}=\pi_{0.5}(o_t,p_t,\ell)`}</MathText><br/><small>{t('关节空间候选动作段')}</small></div><b aria-hidden="true">↓</b>
          <div className="rr-node-accent">{GPT_NAME} {t('审核')}<br/><MathText>{String.raw`(o_t,p_t,\ell,h_t,A_t^{\pi})`}</MathText></div>
          <div className="rr-choice" aria-label={t('二选一，不同时执行')}><span>{t('二选一 · OR')}</span><i aria-hidden="true"/><div className="rr-branches"><div>{t('沿用')} <MathText>{String.raw`A_t^{\pi}`}</MathText><br/><small>{t('执行 1–15 步')}</small></div><div>{t('EEF 修正')} <MathText>{String.raw`A_t^{\mathrm{Astra}}`}</MathText><br/><small>{t('执行 1–5 步')}</small></div></div><i className="rr-choice-join" aria-hidden="true"/></div><b aria-hidden="true">↓</b>
          <div>{t('执行')} <MathText>{'k_t'}</MathText> {t('步')} → <MathText>{'o_{t+k_t},p_{t+k_t}'}</MathText> ↺</div>
        </div></div>
        <div className="rr-method"><h3><ModelName model="gpt" /></h3><div className="rr-flow">
          <div>{t('图像')} <MathText>{'o_t'}</MathText> · {t('本体状态')} <MathText>{'p_t'}</MathText> · {t('指令')} <MathText>{String.raw`\ell`}</MathText></div><b aria-hidden="true">↓</b>
          <div>{t('执行历史')} <MathText>{'h_t'}</MathText><br/><small>{t('工具 · 持久笔记')}</small></div><b aria-hidden="true">↓</b>
          <div className="rr-node-accent"><MathText>{String.raw`A_t^{\mathrm{Astra}}=\operatorname{GPT\ 6\ Astra}(o_t,p_t,\ell,h_t)`}</MathText></div>
          <div className="rr-branches"><div>{t('双臂 EEF 目标')} <MathText>{String.raw`(\mathbf{x},R,g)`}</MathText><br/><small>{t('位置 · 姿态 · 夹爪；执行 1–5 步')}</small></div></div><b aria-hidden="true">↓</b>
          <div>{t('执行')} <MathText>{'k_t'}</MathText> {t('步')} → <MathText>{'o_{t+k_t},p_{t+k_t}'}</MathText> ↺</div>
        </div></div>
      </div>
      <Prose id="method-caption" />
      <SkillDetails />
      <Prose id="paired-settings" />
      <details className="rr-details"><summary>{t('10 个任务：初态与期望终态')}</summary>
        <DataComponent id="task-catalog" queryId="tasks" title={t('任务说明')} sourceRows={tasks} displayRows={tasks.map(task)}>
          <div className="rr-task-grid" data-reviewed-rows>{tasks.map(row => {const view=task(row);return <div key={row.id} className="rr-task"><img loading="lazy" src={row.image} alt={`${view.label}${language==='en'?' ':''}${t('初态示例')}`} /><span>{view.category} · {t('最多')} {row.limit} {t('步')}</span><h3>{view.label}</h3><p>{view.initial}</p><p><b>{t('目标：')}</b>{view.goal}</p>{galleryAvailable&&<a href={`gallery.html?id=mix__${snapshot.queries.cases.rows.find(c => c.task === row.id && c.method === 'mix').case_id}`}>{t('查看完整过程 ↗')}</a>}</div>;})}</div>
        </DataComponent>
      </details>
    </section>

    <section id="results" className="rr-section">
      <Prose id="results-heading" />
      <Prose id="results-summary" />
      <div className="rr-paired-metrics">
        <ModelChart id="panel-score-ranking" models={models}/>
        <ModelChart id="panel-sr-ranking" models={models} metric="sr"/>
      </div>
      <Prose id="chart-scope" />
      <Prose id="official-scope" />
      <div className="rr-chart-switch" role="group" aria-label={t('表一指标')}><button aria-pressed={tableMetric==='score'} onClick={()=>setTableMetric('score')}>{t('Score 热力表')}</button><button aria-pressed={tableMetric==='sr'} onClick={()=>setTableMetric('sr')}>{t('成功率热力表')}</button></div>
      <DataComponent id="task-score-table" queryId="tasks" queryIds={['tasks', 'models']}
        sourceRowsByQuery={{ tasks, models }} displayRows={tableRows} title={t(tableMetric==='score'?'逐任务 Score（0–100）':'逐任务成功率（%）')}>
        <div className="rr-heat-legend"><span>{tableMetric==='score'?'Score':t('成功率 (%)')}</span>{[0,20,40,60,80,100].map(v=><span key={v} style={heatStyle(v)}>{v}</span>)}</div>
        <div className="rr-table-scroll"><table className="rr-table rr-heat-table" data-reviewed-rows><thead><tr><th>{t('任务')}</th>{tableModels.map(m => <th key={m}><ModelName model={m} /></th>)}</tr></thead>
          <tbody>{tasks.map(row => <tr key={row.id}><th scope="row">{task(row).label}</th>{tableModels.map(m => {const value=(tableMetric==='score'?row.scores:row.rates)[m];return <td key={m} style={heatStyle(value)} data-value={value}>{fmt(value)}{tableMetric==='sr'?'%':''}</td>;})}</tr>)}</tbody>
          <tfoot><tr><th>{t('总体')}</th>{tableModels.map(m => {const value=models.find(x=>x.name===m)[tableMetric];return <td key={m} style={heatStyle(value)}>{fmt(value)}{tableMetric==='sr'?'%':''}</td>;})}</tr></tfoot>
        </table></div>
      </DataComponent>
      <Prose id="table-footnote" />
    </section>

    <RoboLabResults Prose={Prose}/>

    <section id="findings" className="rr-section">
      <DataComponent id="executed-step-share" queryId="intervention" title={t('混合策略 · 实际执行控制步')} showHeading={false} sourceRows={intervention} displayRows={intervention}><Prose id="intervention-copy" /></DataComponent>
      <div className="rr-video-note"><Prose id="video-note" /></div>
      <Prose id="grounding-heading" />
      <ClipGroup section="grounding" clips={clips} />
      <Prose id="contact-heading" />
      <DataComponent id="featured-historical-case" queryId="featured_case" title={t('反馈驱动的非抓取操作')} sourceRows={[hero]} displayRows={[hero]} showHeading={false}>
        <div className="rr-clip-grid"><ClipPlayer clip={hero} selected={videoSelection.find(s=>s.id===hero.id)} /></div>
      </DataComponent>
      <ClipGroup section="contact" clips={clips} />
      <Prose id="reasoning-heading" />
      <ClipGroup section="reasoning" clips={clips} />
    </section>

    <section id="failure" className="rr-section">
      <Prose id="failure-heading" />
      <ClipGroup section="failures" clips={clips} />
    </section>

    <section id="direct" className="rr-section">
      <Prose id="gpt-only-heading" />
      <ClipGroup section="gpt-only" clips={clips} />
      <Prose id="direct-control-heading" />
      <ClipGroup section="direct-control" clips={clips} />
      <div id="cost" className="rr-cost-comparison">
      <Prose id="cost-heading" />
      <DataComponent id="cost-table" queryId="costs" title={t('执行与用量')} showHeading={false} sourceRows={costs} displayRows={costRows}>
        <div className="rr-table-scroll"><table className="rr-table" data-reviewed-rows><thead><tr><th>{t('指标')}</th><th><ModelName model="mix" /></th><th><ModelName model="gpt" /></th></tr></thead><tbody>
          {[['实际控制步', 'steps', 0], ['已执行动作段', 'chunks', 0], ['平均物理时长（秒）', 'meanSeconds', 2], ['总 token（含缓存输入）', 'totalTokens', 0], ['其中：缓存输入 token', 'cachedInput', 0], ['未缓存输入 token', 'uncachedInput', 0], ['输出 token', 'outputTokens', 0]].map(([label, key, decimals]) => <tr key={key}><th scope="row">{t(label)}</th>{costRows.map(r => <td key={r.method}>{decimals ? fmt(r[key], decimals) : num(r[key])}</td>)}</tr>)}
        </tbody></table></div>
      </DataComponent>
      <Prose id="physical-time-correction" />
      </div>
    </section>

    <section id="limits" className="rr-section"><Prose id="limits-copy" /></section>

    <section id="conclusion" className="rr-section"><Prose id="conclusion-copy" /></section>

    <section id="references" className="rr-section">
      <Prose id="reference-heading" />
      <ol className="rr-reference-list">{D.references.map(original => {const r=reference(original);return <li key={r.id} id={`ref-${r.id}`}><Prose id={`reference-${r.id}`}>{`${r.author}. [${r.title}](${r.url}). ${r.type}.`}</Prose></li>;})}</ol>
      <div className="rr-closing"><Prose id="closing-copy" />{galleryAvailable&&<a href={`gallery.html?lang=${language}`}>{t('RoboDojo 视频库')}</a>}{robolabGalleryAvailable&&<a href={`robolab-gallery.html?lang=${language}`}>{t('RoboLab 视频库')}</a>}<a href="provenance.json">{t('来源与校验记录')}</a></div>
    </section>
    <Citation />
  </article>;
}
