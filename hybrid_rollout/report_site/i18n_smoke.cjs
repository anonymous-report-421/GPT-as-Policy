const {chromium}=require('/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const base=process.env.REPORT_URL||'http://127.0.0.1:8768/';
const out=path.resolve(process.env.REPORT_QA||'runtime/report_i18n_qa');
fs.mkdirSync(out,{recursive:true});
const baselineFile=path.join(out,'chinese-baseline.json');
async function measure(page){
 return page.evaluate(()=>{
  const selectors=['.rr-hero','.rr-opening-figure','#abstract','#motivation','#method','#results','#findings','#failure','#direct','.rr-heat-table','.rr-methods',...Array.from({length:21},(_,i)=>`.rr-clip:nth-of-type(${i+1})`)];
  return Object.fromEntries(selectors.map(s=>{const e=document.querySelector(s);if(!e)return [s,null];const r=e.getBoundingClientRect(),c=getComputedStyle(e);return [s,{x:r.x,y:r.y+scrollY,width:r.width,height:r.height,font:c.fontSize,lineHeight:c.lineHeight}];}));
 });
}
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('.rr-report').waitFor();await page.evaluate(()=>document.fonts.ready);
  await page.waitForFunction(()=>document.querySelectorAll('[data-component-id="opening-success-ranking"] .recharts-bar-rectangle').length===12);
  if(process.env.CAPTURE_BASELINE==='1'){
   const baseline={desktop:await measure(page),prose:await page.locator('.rr-prose').allTextContents(),css:fs.readFileSync('hybrid_rollout/report_site/app/src/content/report/report.css','utf8')};
   await page.screenshot({path:path.join(out,'before-zh-desktop.png')});
   await page.setViewportSize({width:390,height:844});await page.waitForTimeout(250);
   baseline.mobile=await measure(page);await page.screenshot({path:path.join(out,'before-zh-mobile.png')});
   fs.writeFileSync(baselineFile,JSON.stringify(baseline,null,2));
   console.log('Captured original Chinese text, typography and layout.');
   return;
  }
  const baseline=JSON.parse(fs.readFileSync(baselineFile,'utf8'));
  const data=JSON.parse(fs.readFileSync('report_web/data.json','utf8'));
  const locales=JSON.parse(fs.readFileSync('hybrid_rollout/report_site/app/src/content/report/locale.en.json','utf8'));
  const videos=await page.locator('.rr-video-card video').evaluateAll(xs=>xs.map(x=>x.getAttribute('src')));
  const numericCells=()=>page.locator('.rr-heat-table tbody td').evaluateAll(xs=>xs.map(x=>Number(x.dataset.value)));
  const costs=await page.locator('#cost tbody td').allTextContents();
  const sourceIds=await page.locator('[data-component-id]').evaluateAll(xs=>xs.map(x=>x.dataset.componentId));
  const math=await page.locator('#method .katex-mathml annotation').allTextContents();
  const originalScoreCells=await numericCells();
  async function noChinese(){
   const unexpected=await page.locator('.rr-report').evaluate(root=>{
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),bad=[];
    let n;
    while((n=walker.nextNode())){
     if(n.parentElement.closest('.rr-language-switch'))continue;
     const range=document.createRange();range.selectNodeContents(n);
     if(range.getClientRects().length && /[\u4e00-\u9fff]/.test(n.textContent))bad.push(n.textContent);
    }
    return bad;
   });
   assert.deepEqual(unexpected,[],'Untranslated visible English-page text');
  }
  async function validateCharts(){
   for(const id of ['opening-success-ranking','panel-score-ranking','panel-sr-ranking']){
    const root=page.locator(`[data-component-id="${id}"]`);
    assert.equal(await root.locator('.recharts-bar-rectangle').count(),12);
    const metric=id==='panel-sr-ranking'?'sr':'score';
    const expected=[...data.queries.models.rows].sort((a,b)=>b.score-a.score).map(r=>Number(r[metric].toFixed(2)));
    const labels=await root.locator('.recharts-label-list text').allTextContents();
    assert.deepEqual(labels.map(x=>Number(x.replace('%',''))),expected);
    const ticks=await root.locator('.recharts-xAxis-tick-labels text').allTextContents();
    assert.equal(Math.max(...ticks.map(x=>Number(x.replace('%','')))),id==='opening-success-ranking'?80:100);
   }
  }
  async function englishLayout(){
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'Page overflow');
   const overlap=await page.evaluate(()=>{
    const a=document.querySelector('.rr-language-switch').getBoundingClientRect(),b=document.querySelector('.rr-hero h1').getBoundingClientRect();
    return a.bottom>b.top&&a.left<b.right&&a.right>b.left;
   });assert(!overlap,'Language button overlaps title');
   const text=await page.locator('#abstract .rr-prose').boundingBox();
   for(const r of await page.locator('.rr-clip-grid,.rr-table-scroll').evaluateAll(xs=>xs.map(x=>{const r=x.getBoundingClientRect();return {x:r.x,width:r.width};}))){
    assert(Math.abs(r.x-text.x)<2&&Math.abs(r.width-text.width)<2,'Media/table width differs from prose');
   }
   assert.equal(await page.locator('.katex-error').count(),0);
  }
  async function compareChineseLayout(size){
   const actual=await measure(page),expected=baseline[size];
   for(const [selector,want] of Object.entries(expected)){
    if(!want)continue;
    const got=actual[selector];
    assert(got,selector);
    for(const field of ['x','y','width','height']){
     if(selector==='#direct'&&field==='height'){
      // This section intentionally shrinks when the user-requested usage heading is removed.
      assert(got.height<want.height&&want.height-got.height<100);
     }else assert(Math.abs(got[field]-want[field])<2,`${size}: Chinese ${selector}.${field} changed: ${want[field]} -> ${got[field]}`);
    }
    assert.equal(got.font,want.font);assert.equal(got.lineHeight,want.lineHeight);
   }
  }
  assert.equal(await page.locator('.rr-report').getAttribute('lang'),'en');
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  assert((await page.title()).includes('GPT 6 Astra'));
  await noChinese();await validateCharts();await englishLayout();
  assert.equal(await page.locator('#cost h3').count(),1,'Only the section heading should remain');
  assert(!(await page.locator('.rr-report').innerText()).includes('用量口径'));
  assert.deepEqual(await page.locator('.rr-heat-table tbody th').allTextContents(),data.queries.tasks.rows.map(r=>locales.tasks[r.id].label));
  assert.deepEqual(await page.locator('.rr-video-caption').allTextContents().then(xs=>xs.map(x=>Number(x.match(/Video (\d+)/)[1]))),Array.from({length:21},(_,i)=>i+1));
  await page.screenshot({path:path.join(out,'en-desktop.png')});
  await page.locator('.rr-methods').screenshot({path:path.join(out,'en-methods.png')});
  await page.locator('.rr-heat-table').screenshot({path:path.join(out,'en-heatmap.png')});
  await page.locator('#cost').screenshot({path:path.join(out,'en-cost.png')});
  // Read every disclosure, including the seven unmodified English Skill sources.
  await page.locator('.rr-report details').evaluateAll(xs=>xs.forEach(x=>x.open=true));
  await noChinese();assert.equal(await page.locator('.rr-task').count(),10);
  assert.equal(await page.locator('.rr-skill-source').count(),7);
  await page.locator('.rr-skill-source').first().screenshot({path:path.join(out,'en-skill.png')});
  await page.locator('.rr-task-grid').screenshot({path:path.join(out,'en-tasks.png')});
  await page.locator('.rr-report details').evaluateAll(xs=>xs.forEach(x=>x.open=false));
  // Source preview titles remain exact; the reviewed explanatory preview is translated.
  const reference=page.locator('#motivation a[href="https://github.com/zjwzcx/Awesome-Astra-Embodied-AI"]');
  await reference.hover();
  await page.waitForTimeout(400);
  assert((await page.locator('body').innerText()).includes('collects community explorations'));
  await page.mouse.move(0,0);
  const sample=page.locator('.rr-video-card').first();
  await sample.click();
  const dialog=page.locator('.rr-video-dialog[open]');
  assert.equal(await dialog.locator('button').innerText(),'Close');
  await dialog.locator('video').evaluate(v=>{v.pause();v.currentTime=1;});
  await page.getByRole('button',{name:'Close video',exact:true}).click();
  await page.getByRole('button',{name:'Success-rate heatmap',exact:true}).click();
  const successCells=await numericCells();
  const tableModels=[...data.reportData.top_four,'Pi-05','mix','gpt'];
  assert.deepEqual(successCells,data.queries.tasks.rows.flatMap(r=>tableModels.map(m=>r.rates[m])));
  await page.getByRole('button',{name:'Switch to Chinese',exact:true}).click();
  assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
  assert.deepEqual(await numericCells(),successCells,'Language switch reset the selected metric');
  await validateCharts();
  await page.getByRole('button',{name:'Score 热力表',exact:true}).click();
  assert.deepEqual(await numericCells(),originalScoreCells);
  assert.deepEqual(await page.locator('.rr-prose').allTextContents(),baseline.prose,'Original Chinese prose changed');
  await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(300);
  await compareChineseLayout('desktop');
  await page.screenshot({path:path.join(out,'zh-desktop.png')});
  assert.deepEqual(await page.locator('#cost tbody td').allTextContents(),costs);
  assert.deepEqual(await page.locator('#method .katex-mathml annotation').allTextContents(),math);
  assert.deepEqual(await page.locator('[data-component-id]').evaluateAll(xs=>xs.map(x=>x.dataset.componentId)),sourceIds);
  assert.deepEqual(await page.locator('.rr-video-card video').evaluateAll(xs=>xs.map(x=>x.getAttribute('src'))),videos);
  // Same media elements survive the switch (not just the same paths).
  await page.locator('.rr-video-card video').first().evaluate(v=>v.dataset.i18nSentinel='same-video');
  await page.reload({waitUntil:'domcontentloaded'});await page.locator('.rr-report[lang="zh-CN"]').waitFor();
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(300);
  await compareChineseLayout('mobile');await page.screenshot({path:path.join(out,'zh-mobile.png')});
  await page.locator('.rr-video-card video').first().evaluate(v=>v.dataset.i18nSentinel='same-video');
  await page.getByRole('button',{name:'切换为英文',exact:true}).click();
  assert.equal(await page.locator('.rr-video-card video').first().getAttribute('data-i18n-sentinel'),'same-video');
  await noChinese();await validateCharts();await englishLayout();
  await page.screenshot({path:path.join(out,'en-mobile.png')});
  await page.locator('.rr-methods').screenshot({path:path.join(out,'en-methods-mobile.png')});
  await page.locator('#cost').screenshot({path:path.join(out,'en-cost-mobile.png')});
  // No persisted browser preference overrides the default of a plain URL.
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('.rr-report[lang="en"]').waitFor();
  assert.deepEqual(errors,[]);
  const html=await(await page.request.get(base)).body();
  const result={status:'passed',html_sha256:crypto.createHash('sha256').update(html).digest('hex'),defaultEnglish:true,chineseLayoutUnchanged:true,originalChineseTextExact:true,allEnglishDisclosures:true,translatedReferenceHover:true,taskCount:10,bodyVideos:21,sourceComponentIdsUnchanged:true,mathUnchanged:true,quantitativeValuesUnchanged:true,languageSwitchPreservesMetric:true,mediaElementsPreserved:true,desktop:true,mobile:true,pageErrors:errors};
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
