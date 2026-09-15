import React from 'react';
import {DataComponent, useDataApp} from '../../data-app-public.jsx';
import {ModelChart} from './ModelChart.jsx';
import {ModelName} from './Math.jsx';
import {useReportLanguage} from './Language.jsx';
import labels from './robolab-locales.json';

export function RoboLabResults({Prose}) {
  const {reviewedRows} = useDataApp();
  const {language,t} = useReportLanguage();
  const models = [...reviewedRows('robolab_models')].sort((a,b)=>b.sr-a.sr);
  const tasks = reviewedRows('robolab_tasks');
  return <section id="robolab-results" className="rr-section rr-robolab">
    <Prose id="robolab-intro" />
    <figure className="rr-opening-figure" id="robolab-result-figure" aria-label={t('RoboLab 十任务成功率')}>
      <div className="rr-opening-head">
        <a className="rr-robolab-wordmark" href="https://arxiv.org/abs/2604.09860">RoboLab</a>
        <div><h2>{t('十任务成功率')}</h2><span>{t('单臂 Franka · 每策略 10 任务 × 5 次')}</span></div>
      </div>
      <ModelChart id="robolab-success-ranking" queryId="robolab_models" models={models}
        metric="sr" sortBy="sr" icons large height={300} showHeading={false}/>
      <figcaption className="rr-caption"><Prose id="robolab-figure-caption" /></figcaption>
    </figure>
    <div className="rr-robolab-task-results" id="robolab-task-details">
      <DataComponent id="robolab-task-table" queryId="robolab_tasks" queryIds={['robolab_tasks','robolab_models']}
        title={t('表 2. RoboLab 逐任务成功率（每任务 5 次）')} sourceRowsByQuery={{robolab_tasks:tasks,robolab_models:models}} displayRows={tasks}>
        <div className="rr-table-scroll"><table className="rr-table" data-reviewed-rows>
          <thead><tr><th>{t('任务')}</th>{models.map(m=><th key={m.id}><ModelName model={m.name}/></th>)}</tr></thead>
          <tbody>{tasks.map(row=><tr key={row.task}><th scope="row">{language==='zh'?labels[row.task].zh:row.label}</th>{models.map(m=><td key={m.id}>{20*row.successes[m.id]}% ({row.successes[m.id]}/5)</td>)}</tr>)}</tbody>
          <tfoot><tr><th>{t('总体')}</th>{models.map(m=><td key={m.id}>{m.sr}% ({m.successes}/{m.episodes})</td>)}</tr></tfoot>
        </table></div>
      </DataComponent>
    </div>
    <Prose id="robolab-results-summary" />
    <Prose id="robolab-analysis" />
    <div className="rr-robolab-links">
      <a href="robolab-leaderboard.csv">{t('总体结果 CSV')}</a>
      <a href="robolab-scores.csv">{t('逐任务 CSV')}</a>
      <a href="robolab-baselines.json">{t('Baseline 选取记录')}</a>
    </div>
    <Prose id="robolab-scope-note" />
  </section>;
}
