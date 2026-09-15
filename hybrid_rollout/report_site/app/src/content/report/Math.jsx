import React from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

export const PI_LATEX = String.raw`\pi_{0.5}`;
export const GPT_NAME = 'GPT 6 Astra';
export const GPT_DIRECT_NAME = 'GPT 6 Astra（direct）';
export const modelLabel = model => model === 'mix' ? `π₀.₅ + ${GPT_NAME}`
  : model === 'gpt' ? GPT_DIRECT_NAME : model === 'Pi-05' ? 'π₀.₅'
  : model === 'Pi-0' ? 'π₀' : model === 'Pi-0-FAST' ? 'π₀-FAST'
  : model === 'StarVLA-PI_v3' ? 'StarVLA-PI' : model;
export function MathText({children, display=false}) {
  return <span className={display ? 'rr-math-display' : 'rr-math'}
    dangerouslySetInnerHTML={{__html:katex.renderToString(String(children), {
      displayMode:display, throwOnError:true, trust:false,
    })}} />;
}
export function ModelName({model}) {
  if (model === 'Pi-05') return <MathText>{PI_LATEX}</MathText>;
  if (model === 'Pi-0') return <MathText>{String.raw`\pi_{0}`}</MathText>;
  if (model === 'Pi-0-FAST') return <MathText>{String.raw`\pi_{0}\text{-FAST}`}</MathText>;
  if (model === 'mix') return <><MathText>{PI_LATEX}</MathText> + {GPT_NAME}</>;
  return modelLabel(model);
}
