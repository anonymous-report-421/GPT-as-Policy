import React from 'react';
import {EvidenceChart} from '../../data-app-public.jsx';
import {ModelName, MathText, PI_LATEX, modelLabel} from './Math.jsx';
import logos from './model-logos.json';
import {useReportLanguage} from './Language.jsx';

const shortNames = {'GalaxeaVLA (G0.5)':'Galaxea G0.5','Xiaomi-Robotics-1':'Xiaomi R1',
  'Meituan-Robotics-0':'Meituan R0','Hy-Embodied-0.5-VLA':'Hy-VLA 0.5',
  'InternVLA-A1.5':'InternVLA A1.5','StarVLA-PI_v3':'StarVLA-PI'};
const plainName = r => (shortNames[r.name] || modelLabel(r.name)) + (r.reference_label || '');

// This is the actual category-axis tick: icons and math share the bar's y coordinate.
// The shared renderer continues to own marks, tooltips, editing, export and source inspection.
export function ModelAxisTick({x,y,payload,width,side,rows,icons=false,score=false}) {
  const row=rows.find(r=>r.label===payload.value);
  if(!row)return <text x={x} y={y} dy=".35em" textAnchor="end">{payload.value}</text>;
  const keys=row.name==='mix'?['Pi-05','gpt']:[row.logo_key||row.name];
  return <g className="rr-model-axis-tick" data-model={row.name}>
    <foreignObject x={side==='right'?x+8:x-width} y={y-19} width={width-12} height={38}>
      <div xmlns="http://www.w3.org/1999/xhtml" className={`rr-axis-label${icons?' has-icons':''}${row.name==='mix'?' is-hybrid':''}`}>
        {icons&&<span className="rr-axis-icons">{keys.filter(key=>logos[key]).map(key=><img key={key} src={logos[key].src} alt="" />)}</span>}
        <span className="rr-axis-model">{shortNames[row.name]||<ModelName model={row.name} />}{row.reference_label&&<sup className="rr-scope-marker">{row.reference_label}</sup>}</span>
      </div>
    </foreignObject>
  </g>;
}

export function ModelChart({id,models,metric='score',icons=false,large=false,showHeading=true,queryId='models',sortBy='score',height,valueDomain=[0,100]}) {
  const {t}=useReportLanguage();
  const rateLabel=t('成功率 (%)');
  // Paired lower panels share order and units; the leading chart also ranks by Score.
  const rows=[...models].sort((a,b)=>b[sortBy]-a[sortBy]).map(r=>({...r,label:plainName(r),Score:r.score,[rateLabel]:r.sr}));
  const colors=Object.fromEntries(rows.map(r=>[r.label,r.name==='mix'?'#3478f6':r.name==='gpt'?'#56b5a4':r.name==='Pi-05'?'#e9a150':'#a7bddc']));
  const width=icons?240:182;
  return <EvidenceChart id={id} queryId={queryId} title={t(metric==='score'?'平均 Score（0–100）':'成功率（%）')}
    showHeading={showHeading} rows={rows} sourceRows={models} height={height??(large?496:470)}
    spec={{type:'horizontalBar',x:'label',y:metric==='score'?'Score':rateLabel,valueDomain,stackable:false,showValues:true,valueDecimals:2,valueLabelColor:'#355477',valueLabelFontSize:large?15:12,colors,markRadius:4,markStartRadius:0}}
    chartOptions={{categoryAxisMinWidth:width,renderCategoryTick:p=><ModelAxisTick {...p} rows={rows} icons={icons} score={metric==='score'}/>}}
    renderPlot={plot=><div className={`rr-model-plot-scroll${large?' is-large':''}`}><div className="rr-model-plot">{plot}</div></div>}/>
}

export function InterventionAxisTick({x,y,payload,width}) {
  return <foreignObject x={x-width} y={y-18} width={width-10} height={36}>
    <div xmlns="http://www.w3.org/1999/xhtml" className="rr-axis-label">{payload.value==='π0.5'?<MathText>{PI_LATEX}</MathText>:payload.value}</div>
  </foreignObject>;
}
