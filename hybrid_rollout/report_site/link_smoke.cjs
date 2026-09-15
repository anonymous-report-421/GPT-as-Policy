// Check report assets and gallery navigation without pretending missing media exist.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(process.env.REPORT_PLAYWRIGHT||'/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const base=process.env.REPORT_URL||'http://127.0.0.1:8771/';
const output=process.env.REPORT_QA||'runtime/report-link-qa';
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage(),errors=[],checked=new Set();
  page.on('pageerror',error=>errors.push(error.message));
  let robolabAvailable=false;
  for(const language of ['en','zh']){
   const url=new URL(base);url.searchParams.set('lang',language);
   await page.goto(url.href,{waitUntil:'domcontentloaded'});await page.locator('.rr-report').waitFor();
   const links=await page.locator('.rr-report a[href]').evaluateAll(xs=>xs.map(a=>a.href));
   for(const href of new Set(links)){
    const target=new URL(href),current=new URL(page.url());
    if(target.origin!==current.origin)continue;
    if(target.hash&&target.pathname===current.pathname){
     assert(await page.evaluate(id=>!!document.getElementById(id),decodeURIComponent(target.hash.slice(1))),href);
    }
    target.hash='';
    if(!checked.has(target.href)){
     const response=await page.request.head(target.href);
     assert.equal(response.status(),200,target.href);checked.add(target.href);
    }
   }
   const galleries=await page.locator('.rr-project-links a[href*="gallery.html"]').evaluateAll(xs=>xs.map(a=>({href:a.href,text:a.textContent})));
   robolabAvailable=galleries.some(g=>new URL(g.href).pathname.endsWith('robolab-gallery.html'));
   for(const gallery of galleries){
    assert.equal(new URL(gallery.href).searchParams.get('lang'),language);
    assert(gallery.text.includes(language==='en'?'Video Gallery':'视频库'));
    await page.goto(gallery.href,{waitUntil:'domcontentloaded'});
    await page.locator('.rollout-button').first().waitFor();
    const back=new URL(await page.locator('#back-report').getAttribute('href'),page.url());
    assert.equal(back.searchParams.get('lang'),language);
    assert.equal((await page.request.head(back.href)).status(),200);
    for(const target of await page.locator('#benchmark-galleries a').evaluateAll(xs=>xs.map(a=>a.href))){
     assert.equal(new URL(target).searchParams.get('lang'),language);
     assert.equal((await page.request.head(target)).status(),200,target);
    }
   }
  }
  assert.deepEqual(errors,[]);
  if(process.env.REPORT_EXPECT_ROBOLAB==='1')assert(robolabAvailable,'Expected RoboLab gallery missing');
  const result={status:'passed',internalLinks:checked.size,languages:['en','zh'],galleryNavigation:true,robolabAvailable,
    robolabVideosChecked:false,note:'Real RoboLab video playback/Range checks belong to robolab_gallery_smoke.cjs.',errors};
  fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'links.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
