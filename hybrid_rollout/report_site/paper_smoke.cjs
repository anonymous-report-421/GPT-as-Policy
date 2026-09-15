const {chromium}=require('/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const base=process.env.REPORT_URL||'http://127.0.0.1:8768/',out=path.resolve(process.env.REPORT_QA||'runtime/report_score_qa');
const data=JSON.parse(fs.readFileSync('hybrid_rollout/report_site/app/src/data.json','utf8'));
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('.rr-report').waitFor();await page.evaluate(()=>document.fonts.ready);
  // This legacy regression suite intentionally checks the preserved Chinese presentation.
  if(await page.locator('.rr-report').getAttribute('lang')==='en'){
    const caption=page.locator('.rr-report > .rr-opening-figure').first().locator('.rr-caption');
    assert((await caption.innerText()).includes('stratifying'));
    assert((await caption.innerText()).includes('official full-benchmark leaderboard'));
    assert.equal(await caption.locator('.katex').count(),1);
    await page.getByRole('button',{name:'Switch to Chinese',exact:true}).click();
  }
  const leadingCaption=page.locator('.rr-report > .rr-opening-figure').first().locator('.rr-caption');
  assert((await leadingCaption.innerText()).includes('分层选取'));
  assert((await leadingCaption.innerText()).includes('标准/随机场景配比'));
  assert.equal(await leadingCaption.locator('.katex').count(),1);
  assert.equal(await page.locator('.dashboard-topbar').count(),0);
  assert.equal(await page.locator('.rr-hero h1').innerText(),'GPT 6 Astra 作为具身策略');
  assert.deepEqual(await page.locator('.rr-author-name').allTextContents(),['Jiayi Su','Yixin Zheng','Mi Yan','Li Yi','Zhizheng Zhang','He Wang']);
  assert.equal(await page.locator('.rr-author > a[target="_blank"]').count(),6);
  assert.equal(await page.locator('.rr-author sup').count(),5);
  assert.equal(await page.locator('#equal-contribution-note').innerText(),'* 同等贡献');
  assert.equal(await page.locator('#corresponding-author-note').count(),1);
  assert(await page.locator('.rr-institution-brand img').evaluate(x=>x.complete&&x.naturalWidth>0));
  assert.equal(await page.locator('.rr-author-email, .rr-nav').count(),0);
  assert.equal(await page.locator('.rr-author-placeholder, .rr-affiliations').count(),0);
  const title=await page.locator('.rr-hero').boundingBox(),chart=await page.locator('.rr-report > .rr-opening-figure').first().boundingBox();
  assert(title.y+title.height<=chart.y,'Title must precede chart');
  const labFigure=page.locator('#opening-robolab'),labBox=await labFigure.boundingBox();
  assert(Math.abs(labBox.x-chart.x)<2&&Math.abs(labBox.width-chart.width)<2,'Opening charts must have equal width');
  assert(labBox.height<chart.height-80,'RoboLab card must be more compact than RoboDojo');
  assert(Math.abs(labBox.y-chart.y-chart.height)<2,'RoboLab must immediately follow RoboDojo');
  const lab=page.locator('[data-component-id="opening-robolab-ranking"]');
  await page.waitForFunction(()=>document.querySelectorAll('[data-component-id="opening-robolab-ranking"] .recharts-bar-rectangle').length===5);
  assert.deepEqual(await lab.locator('.recharts-label-list text').allTextContents().then(xs=>xs.map(x=>Number(x.replace('%','')))),[98,92,36,36,34]);
  assert.equal(await lab.locator('.rr-axis-icons img').count(),6);
  assert(await lab.locator('img').evaluateAll(xs=>xs.every(x=>x.complete&&x.naturalWidth>0)));
  assert.equal(await lab.locator('.katex').count(),2);
  assert.equal(await lab.locator('.rr-scope-marker').count(),0);
  assert((await labFigure.locator('.rr-caption').innerText()).includes('所选十任务'));
  assert(!/官方总榜|Cosmos3-Edge|GR00T|FAST/.test(await labFigure.innerText()));
  const heights=await lab.locator('.recharts-bar-rectangle').evaluateAll(xs=>xs.map(x=>x.getBoundingClientRect().height));
  assert(Math.max(...heights)-Math.min(...heights)<1,'RoboLab bar thickness must stay uniform');
  assert(heights.every(h=>h>=19&&h<=22),'RoboLab bars must retain their previous thickness');
  const labTicks=await lab.locator('.recharts-xAxis-tick-labels text').allTextContents();
  assert(labTicks.some(x=>Number(x.replace('%',''))===100));
  await labFigure.screenshot({path:path.join(out,'opening-robolab.png')});
  assert.equal(await page.locator('#teaser').count(),0);
  const body=await page.locator('body').innerText();
  assert(!/GPT[- ]only/.test(body),'Old direct-policy label remains visible');
  const staleNames=[...body.matchAll(/.{0,30}\bGPT\b(?!\s6\sAstra|-Policy-Eval|-6 Astra).{0,45}/g)].map(m=>m[0]);
  assert.deepEqual(staleNames,[],'Unqualified GPT label remains visible');
  assert((await page.locator('.rr-method h3').allTextContents()).includes('GPT 6 Astra（direct）'));
  assert(!/RoboDojo · 成功率排行榜|X 原帖的直接访问受限|原始链接保留于参考|\bv[123]\b|原始标注/.test(body));
  assert.equal(await page.locator('.katex-error').count(),0);
  assert(await page.locator('.rr-hero .katex').count()>0,'Subtitle needs rendered pi model');
  const math=await page.locator('#method .katex-mathml annotation').allTextContents();
  for(const x of ['o_t','p_t','h_t','\\ell','\\pi_{0.5}','\\mathbf{x}'])assert(math.some(s=>s.includes(x)),x+' missing');
  assert(!math.some(s=>/s_t|s_\{t/.test(s)),'Old proprioception symbol remains');
  assert.equal(await page.locator('.rr-choice').innerText().then(s=>s.includes('二选一 · OR')),true);
  assert.equal(await page.locator('[data-component-id="executed-step-share"] .recharts-wrapper').count(),0);
  assert(!/不启用 Fast|2 条.*缺分|48 条有分|完整性与中断|控制器的错误恢复实现存在版本差异|DAgger/.test(body));
  const chapterHeads=await page.locator('.rr-section > .rr-prose h2').allTextContents();
  assert(chapterHeads.some(s=>s.startsWith('5. 混合')));
  assert(chapterHeads.some(s=>s.startsWith('6. Direct')));
  assert(chapterHeads.some(s=>s.startsWith('8. 结论')));
  await page.evaluate(()=>{const p=document.querySelector('#abstract p'),r=document.createRange();r.selectNodeContents(p);const s=getSelection();s.removeAllRanges();s.addRange(r);document.dispatchEvent(new Event('selectionchange'));document.dispatchEvent(new MouseEvent('mouseup',{bubbles:true}));});
  await page.waitForTimeout(200);
  assert.equal(await page.locator('.dashboard-ask-panel, .dashboard-ask-composer').count(),0,'Selected-text Ask must be disabled');
  await page.evaluate(()=>getSelection().removeAllRanges());
  assert((await page.locator('.rr-method h3 .rr-math').boundingBox()).width<60,'Math label spacing is broken');
  await page.waitForFunction(()=>document.querySelectorAll('[data-component-id="opening-success-ranking"] .recharts-bar-rectangle').length===12);
  const bars=await page.locator('[data-component-id="opening-success-ranking"] .recharts-bar-rectangle').evaluateAll(xs=>xs.map(x=>{const r=x.getBoundingClientRect();return {x:r.x,w:r.width,h:r.height};}));
  assert(bars.every(b=>b.w>b.h*2));assert(Math.max(...bars.map(b=>b.x))-Math.min(...bars.map(b=>b.x))<2);
  const opening=page.locator('[data-component-id="opening-success-ranking"]');
  assert.equal(await opening.locator('.rr-axis-icons img').count(),13);
  assert(await opening.locator('img').evaluateAll(xs=>xs.every(x=>x.complete&&x.naturalWidth>0)));
  assert.equal(await opening.locator('.katex').count(),2);
  assert(await page.locator('.rr-benchmark-logo').evaluate(x=>x.complete&&x.naturalWidth>0));
  assert(chart.x<2&&Math.abs(chart.width-(await page.locator('main').boundingBox()).width)<2&&chart.width>=1420,'Opening figure must fill the scrollable page width');
  for(const id of ['opening-success-ranking','panel-score-ranking','panel-sr-ranking']){
    const root=page.locator(`[data-component-id="${id}"]`);
    assert.equal(await root.locator('.recharts-bar-rectangle').count(),12);
    const ticks=await root.locator('.recharts-xAxis-tick-labels text').allTextContents();
    assert(ticks.includes('0')||ticks.includes('0%'),id+': zero missing '+ticks);
    const maximum=id==='opening-success-ranking'?'80':'100';
    assert(ticks.includes(maximum)||ticks.includes(maximum+'%'),id+': upper bound missing '+ticks);
    assert.equal(Math.max(...ticks.map(t=>Number(t.replace('%','')))),Number(maximum));
    const labels=await root.locator('.recharts-label-list text').allTextContents();
    const metric=id==='panel-sr-ranking'?'sr':'score';
    const ordered=[...data.queries.models.rows].sort((a,b)=>b.score-a.score);
    const expected=ordered.map(r=>Number(r[metric].toFixed(2)));
    assert.deepEqual(labels.map(x=>Number(x.replace('%',''))),expected,id+' plotted values');
    if(id==='opening-success-ranking'){
      assert(labels.every(x=>!x.includes('%')));
      assert.equal(await root.locator('.chart-legend-button').count(),0);
    }
  }
  const alignment=await opening.locator('.rr-axis-label').evaluateAll(xs=>xs.map(x=>{const r=x.getBoundingClientRect(),label=x.querySelector('.rr-axis-model').getBoundingClientRect();return {y:r.y+r.height/2,x:label.x};}));
  await opening.screenshot({path:path.join(out,'axis-name-check.png')});
  for(const id of ['opening-success-ranking','panel-score-ranking','panel-sr-ranking']){
    const root=page.locator(`[data-component-id="${id}"]`);
    assert((await root.locator('[data-model="gpt"] .rr-axis-model').innerText()).includes('GPT 6 Astra（direct）'));
    assert((await root.locator('[data-model="mix"] .rr-axis-model').innerText()).includes('GPT 6 Astra'));
    // KaTeX can overhang its intrinsic flex item slightly without reaching
    // the actual clipping boundary (the containing foreignObject).
    const overflow=await root.locator('.rr-axis-model').evaluateAll(xs=>xs.filter(x=>x.getBoundingClientRect().left+Math.max(x.scrollWidth,x.getBoundingClientRect().width)>x.parentElement.getBoundingClientRect().right+1).map(x=>x.textContent));
    assert.deepEqual(overflow,[],id+': model names clipped');
  }
  assert(Math.max(...alignment.map(a=>a.x))-Math.min(...alignment.map(a=>a.x))<1,'Model names are not aligned');
  await page.screenshot({path:path.join(out,'homepage.png')});
  await page.locator('.rr-report > .rr-opening-figure').first().screenshot({path:path.join(out,'opening-score.png')});
  await page.locator('.rr-paired-metrics').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'both-metrics.png')});
  await page.locator('.rr-methods').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'formulas.png')});
  const modelIds=[...data.reportData.top_four,'Pi-05','mix','gpt'];
  const cells=page.locator('.rr-heat-table tbody td');
  assert.equal(await cells.count(),70);
  const contrasts=await cells.evaluateAll(xs=>xs.map(el=>{
    const l=color=>{const c=color.match(/[\d.]+/g).slice(0,3).map(Number).map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return c[0]*.2126+c[1]*.7152+c[2]*.0722;};
    const s=getComputedStyle(el),a=l(s.color),b=l(s.backgroundColor);return (Math.max(a,b)+.05)/(Math.min(a,b)+.05);
  }));assert(Math.min(...contrasts)>=4.5,'Heatmap text contrast');
  assert.deepEqual(await cells.evaluateAll(xs=>xs.map(x=>Number(x.dataset.value))),data.queries.tasks.rows.flatMap(t=>modelIds.map(m=>t.scores[m])));
  assert.equal(await page.locator('.rr-heat-table tbody sup').count(),0);
  await page.locator('.rr-heat-table').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(out,'score-heatmap.png')});
  await page.getByRole('button',{name:'成功率热力表',exact:true}).click();
  assert.deepEqual(await cells.evaluateAll(xs=>xs.map(x=>Number(x.dataset.value))),data.queries.tasks.rows.flatMap(t=>modelIds.map(m=>t.rates[m])));
  assert.equal(await page.locator('.rr-heat-table tbody sup').count(),0);
  assert((await page.locator('.rr-heat-table tfoot').innerText()).includes('48.00%'));
  assert((await page.locator('.rr-heat-table tfoot').innerText()).includes('26.00%'));
  await page.screenshot({path:path.join(out,'success-heatmap.png')});
  await page.getByRole('button',{name:'Score 热力表',exact:true}).click();
  assert((await page.locator('.rr-heat-table tfoot').innerText()).includes('37.81'));
  const feature=page.locator('#findings [data-component-id="featured-historical-case"]');
  assert.equal(await feature.count(),1);assert.equal(await feature.locator('.rr-link').count(),0);
  assert.equal(await page.locator('.rr-clip').count(),21);
  const captions=await page.locator('.rr-video-caption').allTextContents();
  assert.deepEqual(captions.map(t=>Number(t.match(/视频 (\d+)/)[1])),Array.from({length:21},(_,i)=>i+1));
  assert(await page.locator('.rr-clip-grid').evaluateAll(xs=>xs.every(x=>x.getBoundingClientRect().width<=761)),'Videos wider than prose');
  const proseBox=await page.locator('#abstract .rr-prose').boundingBox();
  const mediaWidths=await page.locator('.rr-clip-grid, .rr-table-scroll').evaluateAll(xs=>xs.map(x=>{const r=x.getBoundingClientRect();return {x:r.x,width:r.width};}));
  assert(mediaWidths.every(r=>Math.abs(r.x-proseBox.x)<2&&Math.abs(r.width-proseBox.width)<2),'Videos and tables must align with prose');
  assert(await page.locator('.rr-heat-table').evaluate(x=>x.scrollWidth<=x.clientWidth+2),'Desktop heat table must fit prose width');
  assert.equal(await page.locator('.rr-clip h3').count(),0,'Long video descriptions should be replaced by short captions');
  await page.locator('.rr-skill > summary').click();
  assert.equal(await page.locator('.rr-skill-source').count(),7);
  await page.locator('.rr-skill-source').first().locator('summary').click();
  assert((await page.locator('.rr-skill-source').first().innerText()).includes('robodojo_execute'));
  await page.locator('.rr-skill').screenshot({path:path.join(out,'skill-detail.png')});
  await page.locator('.rr-skill > summary').click();
  await page.locator('.rr-video-card video').evaluateAll(vs=>vs.forEach(v=>{v.preload='metadata';v.load();}));
  await page.waitForFunction(()=>[...document.querySelectorAll('.rr-video-card video')].every(v=>v.readyState>=1&&v.videoWidth>0));
  if(process.env.REPORT_PAPER_MEDIA==='1'){
    const videos=await page.locator('.rr-video-card video').evaluateAll(vs=>vs.map(v=>({width:v.videoWidth,height:v.videoHeight,src:v.currentSrc})));
    assert.equal(videos.length,21);
    assert(videos.every(v=>v.width===1280&&v.height===636&&new URL(v.src).searchParams.has('v')));
  }
  assert.equal(await page.locator('video[src*="imitate_sorting_sequence_a5_frames_493_1309.mp4"]').count(),1);
  await feature.scrollIntoViewIfNeeded();await page.waitForFunction(()=>document.querySelector('[data-component-id="featured-historical-case"] video').currentTime>.1);
  await feature.locator('.rr-video-card').click();await feature.locator('dialog[open]').waitFor();
  const video=feature.locator('dialog video');await page.waitForFunction(()=>document.querySelector('dialog[open] video').readyState>=2);
  assert.equal(await video.evaluate(v=>v.controls),true);assert(Math.abs(await video.evaluate(v=>v.duration)-32.68)<.01);
  await page.screenshot({path:path.join(out,'analysis-video.png')});await page.keyboard.press('Escape');
  const reference=page.locator('#motivation a[href="https://github.com/zjwzcx/Awesome-Astra-Embodied-AI"]').first();
  await reference.hover();await page.locator('.source-preview-card').waitFor();
  assert((await page.locator('.source-preview-card').innerText()).includes('Awesome Astra Embodied AI'));
  assert(!/读取受限|不是.*实验|不直接横比/.test(await page.locator('.source-preview-card').innerText()));
  await page.screenshot({path:path.join(out,'reference-hover.png')});await page.keyboard.press('Escape');
  await page.locator('[data-component-id="task-score-table"] button.menu-trigger').click();
  await page.getByRole('menuitem',{name:'View data source'}).click();
  await page.locator('.source-sidebar-body[aria-busy="false"]').waitFor();
  assert((await page.locator('.source-sidebar').innerText()).includes('RoboDojo'));
  await page.screenshot({path:path.join(out,'source-inspection.png')});await page.keyboard.press('Escape');
  for(const id of ['clip__faad002fd6c510f09db4c435__0','clip__6ba3a538a3405206181aa6c6__0']){
    const v=page.locator(`.rr-video-card video[src*="${id}.mp4"]`);
    await v.scrollIntoViewIfNeeded();
    await v.evaluate(async v=>{v.pause();v.currentTime=Math.min(8,v.duration/2);await new Promise(resolve=>v.addEventListener('seeked',resolve,{once:true}));});
    await v.screenshot({path:path.join(out,id+'.png')});
  }
  // The optional next-release sample is deliberately not part of production.
  if(process.env.CHECK_DEBUG_SAMPLE==='1'){
    const sample=await page.request.get(new URL('previews/debug_layout_sample.mp4',base).href,{headers:{Range:'bytes=0-1023'}});
    assert.equal(sample.status(),206);assert.equal((await sample.body()).length,1024);
  }
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>scrollTo(0,0));
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));await page.screenshot({path:path.join(out,'mobile.png')});
  await page.locator('.rr-methods').scrollIntoViewIfNeeded();assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
  await page.screenshot({path:path.join(out,'formulas-mobile.png')});
  assert.deepEqual(errors,[]);
  const html=await(await page.request.get(base)).body();
  const result={status:'passed',html_sha256:crypto.createHash('sha256').update(html).digest('hex'),titleFirst:true,authorsConfigured:true,authorEmailsOnlyInLinks:true,sectionNavigationRemoved:true,openingAxis:[0,80],lowerAxes:[0,100],horizontalBars:12,latex:true,heatmapCells:70,heatmapBothMetrics:true,scoreMissingnessPreserved:true,historicalClipInAnalysisOnly:true,bodyVideos:21,fullSkillDocuments:7,askSelectionDisabled:true,proprioSymbol:'p_t',exclusiveBranches:true,referenceHover:true,sourceInspection:true,mediaWidthMatchesProse:true,debugSampleRange:true,mobile:true,pageErrors:errors};
  result.debugSampleRange=process.env.CHECK_DEBUG_SAMPLE==='1';
  result.authorEmailsOnlyInLinks=false;
  result.authorProfileLinks=true;
  result.horizontalBars=12;
  result.lowerPanelBars=12;
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
