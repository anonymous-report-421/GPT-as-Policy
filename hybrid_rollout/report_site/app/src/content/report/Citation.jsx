import React from 'react';
import {RichNarrative} from '../../data-app-public.jsx';
import {useReportLanguage} from './Language.jsx';
import publication from './publication.json';

export function Citation(){
  const {t,narrativeId}=useReportLanguage();
  return <section id="citation" className="rr-section rr-citation">
    <RichNarrative key={narrativeId('citation-intro')} id={narrativeId('citation-intro')}
      value={t('## 引用本文\n\n如果本文或代码对您的研究有所帮助，欢迎使用以下 BibTeX 引用。')}/>
    <RichNarrative id="citation-bibtex" value={'```bibtex\n'+publication.bibtex+'\n```'}/>
    <a className="rr-citation-download" href="citation.bib" download>{t('下载 BibTeX')}</a>
  </section>;
}
