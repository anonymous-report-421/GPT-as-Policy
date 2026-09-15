import React from 'react';
import {RichNarrative, useDataApp} from '../../data-app-public.jsx';
import {ModelChart} from './ModelChart.jsx';
import {useReportLanguage} from './Language.jsx';
import snapshot from '../../data.json';

export function LeadingRoboLabRanking() {
  const {reviewedRows}=useDataApp();
  const {t,reference,narrativeId}=useReportLanguage();
  const models=reviewedRows('robolab_models');
  const paperIndex=snapshot.reportData.references.findIndex(row=>row.id==='robolab');
  const paper=reference(snapshot.reportData.references[paperIndex]);
  const paperPreview={[paper.url]:{title:paper.title,summary:paper.summary,source:paper.author,approvedForReport:true}};
  const caption=t('**RoboLab · 所选十任务的平均成功率。**')+` [\\[${paperIndex+1}\\]](${paper.url})`;
  return <section id="opening-robolab" className="rr-opening-figure rr-opening-robolab" aria-label={t('RoboLab 十任务成功率')}>
    <div className="rr-opening-head">
      <a className="rr-robolab-wordmark" href="https://research.nvidia.com/labs/srl/projects/robolab/" aria-label="RoboLab zero-shot benchmark">{t('RoboLab（zero-shot）')}</a>
      <div><RichNarrative key={narrativeId('opening-robolab-heading')} id={narrativeId('opening-robolab-heading')} value={t('## 十任务平均成功率')}/><span>{t('单臂 Franka · 每策略 10 任务 × 5 次')}</span></div>
    </div>
    <ModelChart id="opening-robolab-ranking" queryId="robolab_models" models={models}
      metric="sr" sortBy="sr" icons large height={models.length * 40 + 28} showHeading={false} valueDomain={[0,100]}/>
    <div className="rr-caption"><RichNarrative key={narrativeId('opening-robolab-caption')} id={narrativeId('opening-robolab-caption')}
      value={caption} sourcePreviews={paperPreview} /></div>
  </section>;
}
