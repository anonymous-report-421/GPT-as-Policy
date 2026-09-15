import React, {createContext, useContext, useEffect, useState} from 'react';
import english from './locale.en.json';
import copyZh from './article-copy.json';
import copyEn from './article-copy.en.json';

const LanguageContext=createContext(null);
const required=(collection,key)=>{
  if(!Object.hasOwn(collection,key))throw new Error(`Missing report translation: ${key}`);
  return collection[key];
};
// Language changes presentation only. Query IDs, reviewed rows, math, and media stay intact.
// No preference is inferred from the browser: a plain URL always opens in English.
export function ReportLanguage({children}){
  const [language,setLanguage]=useState(()=>new URLSearchParams(window.location.search).get('lang')==='zh'?'zh':'en');
  const changeLanguage=next=>{
    if(!['en','zh'].includes(next))return;
    setLanguage(next);
    const url=new URL(window.location.href);url.searchParams.set('lang',next);
    window.history.replaceState(window.history.state,'',url);
  };
  useEffect(()=>{
    document.documentElement.lang=language==='zh'?'zh-CN':'en';
    document.title=language==='zh'?'GPT 6 Astra 作为具身策略':'GPT 6 Astra as an Embodied Policy';
  },[language]);
  const value={language,changeLanguage,
    t:zh=>language==='zh'?zh:required(english.ui,zh),
    copy:language==='zh'?copyZh:copyEn,
    task:row=>language==='zh'?row:{...row,...required(english.tasks,row.id)},
    caption:selected=>language==='zh'?selected.caption:required(english.captions,selected.id),
    reference:row=>language==='zh'?row:{...row,...required(english.references,row.id)},
    infrastructure:english.infrastructure,
    // Preserve existing Chinese saved edits; English edits have their own stable namespace.
    narrativeId:id=>language==='zh'?id:`${id}--en`,
  };
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}
export function useReportLanguage(){
  const context=useContext(LanguageContext);
  if(!context)throw new Error('Report language provider is missing');
  return context;
}
export function LanguageSwitch(){
  const {language,changeLanguage}=useReportLanguage();
  return <button type="button" className="rr-language-switch" onClick={()=>changeLanguage(language==='en'?'zh':'en')}
    lang={language==='en'?'zh-CN':'en'} aria-label={language==='en'?'Switch to Chinese':'切换为英文'}
    title={language==='en'?'Switch to Chinese':'切换为英文'}>
    <svg aria-hidden="true" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c5 5 5 13 0 18-5-5-5-13 0-18Z"/></svg>
    {language==='en'?'中文':'English'}
  </button>;
}
