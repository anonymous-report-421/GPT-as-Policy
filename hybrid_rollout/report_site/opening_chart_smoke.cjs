const {chromium}=require(process.env.REPORT_PLAYWRIGHT||'/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const base=process.env.REPORT_URL||'http://127.0.0.1:8771/';
const out=process.env.REPORT_QA||'/tmp/report-opening-chart-qa';
const data=JSON.parse(fs.readFileSync('hybrid_rollout/report_site/app/src/data.json','utf8'));
const ordered=[...data.queries.models.rows].sort((a,b)=>b.score-a.score);
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  for(const language of ['en','zh']){
   await page.goto(new URL('?lang='+language,base).href,{waitUntil:'domcontentloaded'});
   const chart=page.locator('[data-component-id="opening-success-ranking"]');
   await chart.waitFor();await chart.scrollIntoViewIfNeeded();await page.evaluate(()=>document.fonts.ready);
   const title=language==='zh'?'GPT 6 Astra 作为具身策略':'GPT 6 Astra as an Embodied Policy';
   assert.equal(await page.locator('.rr-hero h1').innerText(),title);
   assert.equal(await page.title(),title);
   assert(!/使用不同记号|is distinct from the proprioceptive-state notation/.test(await page.locator('.rr-report').innerText()));
   assert(!/本节报告我们在 RoboLab 中部署|Evaluation protocol\./.test(await page.locator('#robolab-results').innerText()));
   const videoLabels=await page.locator('.rr-video-card .rr-video-takeaway-label').allTextContents();
   assert.equal(videoLabels.length,21);
   assert(videoLabels.every(t=>t.trim()===(language==='zh'?'行为分析：':'Clip takeaway:')));
   await page.waitForFunction(()=>document.querySelectorAll('[data-component-id="opening-success-ranking"] .recharts-bar-rectangle').length===12);
   assert.equal(await chart.locator('.recharts-bar').count(),1);
   const labels=await chart.locator('.recharts-label-list text').allTextContents();
   assert.deepEqual(labels.map(Number),ordered.map(r=>Number(r.score.toFixed(2))));
   assert(labels.every(t=>!t.includes('%')));
   assert.equal(await chart.locator('.chart-legend-button').count(),0);
   const rates=await page.locator('[data-component-id="panel-sr-ranking"] .recharts-label-list text').allTextContents();
   assert.deepEqual(rates.map(t=>Number(t.replace('%',''))),ordered.map(r=>Number(r.sr.toFixed(2))));
   assert(rates.every(t=>t.includes('%')));
   assert.equal(await chart.locator('.rr-axis-icons img').count(),13);
   assert.equal(await chart.locator('.katex').count(),2);
   await chart.locator('.recharts-bar-rectangle').first().hover();
   const tooltip=chart.locator('.recharts-tooltip-wrapper');
   await tooltip.waitFor({state:'visible'});
   assert((await tooltip.innerText()).includes('62.6'));
   assert(!(await tooltip.innerText()).includes('48%'));
   await page.mouse.move(2,2);
   const paper=data.reportData.references.find(r=>r.id==='robolab');
   const number=data.reportData.references.findIndex(r=>r.id==='robolab')+1;
   const citation=page.locator('#opening-robolab .rr-caption a');
   assert.equal(await page.locator('#opening-robolab .rr-robolab-wordmark').innerText(),language==='zh'?'RoboLab（zero-shot）':'RoboLab (zero-shot)');
   assert.equal(await citation.innerText(),`[${number}]`);
   assert.equal(await citation.getAttribute('href'),paper.url);
   assert.equal(await page.locator('#ref-robolab').count(),1);
   await citation.hover();
   const preview=page.locator('.source-preview-card:visible .source-preview-title');
   await preview.waitFor({state:'visible'});assert.equal(await preview.innerText(),paper.title);
   await page.mouse.move(2,2);
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:1100});await chart.scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
    const labTitle=page.locator('#opening-robolab .rr-robolab-wordmark');
    assert(await labTitle.evaluate(x=>x.scrollWidth<=x.clientWidth+1));
    assert.equal(await chart.locator('.recharts-bar-rectangle').count(),12);
    await page.locator('.rr-report>.rr-opening-figure').first().screenshot({path:path.join(out,`score-${language}-${width}.png`)});
    if(width===390){
     const scroller=chart.locator('.rr-model-plot-scroll');
     await scroller.evaluate(x=>{x.scrollLeft=x.scrollWidth;});
     assert(await scroller.evaluate(x=>x.scrollLeft>0),'Wide mobile charts must remain horizontally scrollable');
     for(const label of await chart.locator('.recharts-label-list text').all()){
      await label.scrollIntoViewIfNeeded();
      assert(await label.evaluate(x=>{
       const r=x.getBoundingClientRect(),clip=x.closest('.rr-model-plot-scroll').getBoundingClientRect();
       return r.x>=clip.x-1&&r.right<=clip.right+1;
      }),'Each value label must be reachable by horizontal scrolling');
     }
     await page.locator('.rr-report>.rr-opening-figure').first().screenshot({path:path.join(out,`score-${language}-390-values.png`)});
     await scroller.evaluate(x=>{x.scrollLeft=0;});
    }
   }
   await page.setViewportSize({width:1440,height:1100});
  }
  assert.deepEqual(errors,[]);
  const result={status:'passed',models:12,bars:12,openingMetric:'score',successRateInBody:true,axis:[0,80],exactValues:true,tooltip:true,robolabCitation:true,requestedParagraphsAbsent:true,desktop:true,mobile:true,languages:['en','zh'],pageErrors:errors};
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
