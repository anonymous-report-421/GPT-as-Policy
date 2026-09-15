import React from 'react';
import {RichNarrative} from '../../data-app-public.jsx';
import {ModelChart} from './ModelChart.jsx';
import robodojoLogo from '../assets/robodojo-logo.png';
import {useReportLanguage} from './Language.jsx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import {PI_LATEX} from './Math.jsx';
export function LeadingRanking({models}) {
  const {t,narrativeId}=useReportLanguage();
  return <section className="rr-opening-figure" aria-label={t('十任务平均 Score')}>
    <div className="rr-opening-head">
      <a href="https://robodojo-benchmark.com/" aria-label="RoboDojo Benchmark"><img className="rr-benchmark-logo" src={robodojoLogo} alt="RoboDojo" /></a>
      <div><RichNarrative key={narrativeId('opening-score-heading')} id={narrativeId('opening-score-heading')} value={t('## 十任务平均 Score')}/><span>{t('RoboDojo-Sim · 10 个任务')}</span></div>
    </div>
    <ModelChart id="opening-success-ranking" models={models} icons large showHeading={false} valueDomain={[0,80]}/>
    <div className="rr-caption"><RichNarrative key={narrativeId('opening-success-caption')} id={narrativeId('opening-success-caption')}
      value={t('**图 1. 所选十任务的平均 Score。** 十个任务按 π0.5 的公开成功率分层选取，偏向低成功率任务。官方模型成绩按相同任务子集及标准/随机场景配比重新汇总，因此与官网全任务总榜的数值不同。').replaceAll('π0.5', () => '$' + PI_LATEX + '$')}
      remarkPlugins={[remarkMath]} rehypePlugins={[[rehypeKatex,{trust:false,strict:'error'}]]} /></div>
  </section>;
}
