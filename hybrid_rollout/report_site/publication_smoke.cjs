const {chromium}=require(process.env.REPORT_PLAYWRIGHT||'/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const base=process.env.REPORT_URL||'http://127.0.0.1:8769/';
const out=path.resolve(process.env.REPORT_QA||'runtime/public_release_qa');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('.rr-report').waitFor();await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.rr-report').getAttribute('lang'),'en');
  const link=page.locator('.rr-repository-link');
  assert.equal(await link.getAttribute('href'),'https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy');
  assert.equal(await page.locator('.rr-project-links a[href^="gallery.html"]').count(),1);
  const bib=await page.request.get(new URL('citation.bib',base).href);
  assert.equal(bib.status(),200);assert((await bib.text()).includes('Su, Jiayi and Zheng, Yixin and Yan, Mi and Yi, Li and Zhang, Zhizheng and Wang, He'));
  assert((await page.locator('#citation').innerText()).includes('@misc{su2026astra'));
  const heading=page.locator('#robolab-results');await heading.scrollIntoViewIfNeeded();
  await page.waitForFunction(()=>document.querySelectorAll('[data-component-id="robolab-success-ranking"] .recharts-bar-rectangle').length===5);
  const values=await page.locator('[data-component-id="robolab-success-ranking"] .recharts-label-list text').allTextContents();
  assert.deepEqual(values.map(v=>Number(v.replace('%',''))),[98,92,36,36,34]);
  assert.equal(await page.locator('#robolab-task-details tbody td').count(),50);
  assert(!(await heading.innerText()).includes('initial states and execution budgets are not fully paired'));
  await heading.screenshot({path:path.join(out,'robolab-en.png')});
  for(const language of ['en','zh']){
   if(language==='zh')await page.getByRole('button',{name:'Switch to Chinese',exact:true}).click();
   assert(!(await heading.innerText()).includes(language==='zh'?'本节报告我们在 RoboLab 中部署':'Evaluation protocol.'));
   assert((await page.locator('#citation').innerText()).includes(language==='en'?'Citation':'引用本文'));
   await page.locator('.rr-skill > summary').click();
   await page.locator('.rr-skill-source').evaluateAll(xs=>xs.forEach(x=>x.open=true));
   if(language==='en'){
    const untranslated=await page.locator('.rr-report').evaluate(root=>{
     const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),bad=[];let n;
     while((n=walker.nextNode())){if(n.parentElement.closest('.rr-language-switch'))continue;const r=document.createRange();r.selectNodeContents(n);if(r.getClientRects().length&&/[\u4e00-\u9fff]/.test(n.textContent))bad.push(n.textContent);}
     return bad;
    });assert.deepEqual(untranslated,[]);
   }
   await page.locator('.rr-skill > summary').click();
   const expectedCells=await page.locator('#robolab-task-details tbody td').allTextContents();
   assert.equal(expectedCells.filter(x=>x==='100% (5/5)').length,20);
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:width===390?844:1050});
    await page.locator('#robolab-results').scrollIntoViewIfNeeded();
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
    const boxes=await page.locator('#abstract .rr-prose, #robolab-task-details .rr-table-scroll').evaluateAll(xs=>xs.map(x=>{const b=x.getBoundingClientRect();return {x:b.x,w:b.width};}));
    assert(Math.abs(boxes[0].x-boxes[1].x)<2&&Math.abs(boxes[0].w-boxes[1].w)<2);
    await page.screenshot({path:path.join(out,`robolab-${language}-${width}.png`)});
   }
  }
  await page.goto(new URL('gallery.html',base).href);await page.waitForFunction(()=>document.querySelector('#video').readyState>=2);
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  const gallery=await page.evaluate(()=>window.REPORT_DATA);
  assert.equal(gallery.cases.length,100);assert.equal(gallery.clips.length,43);assert.equal(gallery.tasks.length,10);
  if(process.env.REPORT_PAPER_MEDIA==='1'){
   for(const row of [...gallery.cases,...gallery.clips]){
    assert.match(row.video_sha256,/^[a-f0-9]{64}$/);
    assert.equal(row.video_url,row.video+'?v='+row.video_sha256.slice(0,16));
    assert(row.poster_url.startsWith(row.poster+'?v='));
   }
   assert.equal(await page.locator('#video').evaluate(v=>v.videoHeight),636);
   assert(new URL(await page.locator('#video').getAttribute('src'),base).searchParams.has('v'));
  }
  assert.equal(await page.locator('.rollout-button').count(),5);
  const first=new URL(page.url()).searchParams.get('id');
  await page.locator('#paired').click();
  const paired=new URL(page.url()).searchParams.get('id');
  assert.notEqual(first,paired);assert.equal(gallery.cases.find(x=>x.id===first).case_id,gallery.cases.find(x=>x.id===paired).case_id);
  await page.locator('#video').evaluate(async v=>{v.pause();if(v.readyState<1)await new Promise(r=>v.addEventListener('loadedmetadata',r,{once:true}));v.currentTime=v.duration/2;await new Promise(r=>v.addEventListener('seeked',r,{once:true}));});
  await page.locator('#video-card').click();await page.locator('#video-dialog[open]').waitFor();
  assert(await page.locator('#expanded-video').evaluate(v=>v.controls));await page.keyboard.press('Escape');
  await page.locator('#scope').selectOption('clips');
  await page.locator('#search').fill('no-matching-task');assert.equal(await page.locator('.task-button').count(),0);
  await page.locator('#search').fill('');assert(await page.locator('.task-button').count()>0);
  await page.locator('#scope').selectOption('cases');
  await page.locator('#language').click();await page.waitForFunction(()=>document.documentElement.lang==='zh-CN');
  assert((await page.locator('#task-count').innerText()).includes('个任务'));
  await page.screenshot({path:path.join(out,'gallery-mobile.png')});
  // Validate every original full-video URL, including both timeout results.
  for(let i=0;i<gallery.cases.length;i+=8){
   await Promise.all(gallery.cases.slice(i,i+8).map(async row=>{
    const response=await page.request.get(new URL(row.video_url||row.video,base).href,{headers:{Range:'bytes=0-1023'}});
    assert.equal(response.status(),206,row.id);assert.equal((await response.body()).length,1024,row.id);
   }));
  }
  assert.deepEqual(errors,[]);
  const result={status:'passed',bilingual:true,robolabRates:[98,92,36,36,34],robolabTableCells:50,robolabProseAligned:true,repositoryLink:true,bibtex:true,galleryCases:100,galleryClips:43,allFullVideoRanges:true,paperMediaVerified:process.env.REPORT_PAPER_MEDIA==='1',pairedSeedNavigation:true,videoSeekAndPopup:true,mobile:true,errors};
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
