const {chromium}=require('/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const base=process.env.REPORT_URL||'http://127.0.0.1:8768/';
const out=path.resolve(process.env.REPORT_QA||'runtime/report_authors_qa');
const expected=['Jiayi Su','Yixin Zheng','Mi Yan','Li Yi','Zhizheng Zhang','He Wang'];
const expectedUrls=['https://shuyumo2003.github.io/','https://steveouo.github.io/','https://miyandoris.github.io/','https://ericyi.github.io/','https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en','https://hughw19.github.io/'];
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base,{waitUntil:'domcontentloaded'});await page.locator('.rr-authors').waitFor();await page.evaluate(()=>document.fonts.ready);
  assert.equal(await page.locator('.rr-report').getAttribute('lang'),'en');
  assert.equal(await page.locator('.rr-author').count(),6);
  assert.equal(await page.locator('.rr-author-placeholder, .rr-affiliations').count(),0);
  for(let i=0;i<expected.length;i++){
   const author=page.locator('.rr-author').nth(i),value=expected[i];
   assert.equal(await author.locator('.rr-author-name').innerText(),value);
   assert.equal(await author.locator('a').count(),1);
   assert.equal(await author.locator('a').getAttribute('href'),expectedUrls[i]);
   assert.equal(await author.locator('a').getAttribute('target'),'_blank');
   assert.equal(await author.locator('a').getAttribute('rel'),'noopener noreferrer');
   assert.deepEqual(await author.locator('sup').allTextContents(),i<2?['*']:i>=3?['†']:[]);
  }
  for(const language of ['en','zh']){
   if(language==='zh')await page.getByRole('button',{name:'Switch to Chinese',exact:true}).click();
   for(const [i,url] of expectedUrls.entries()){
    // Verify real click navigation without depending on external site availability.
    await page.context().route(url,route=>route.fulfill({status:200,contentType:'text/html',body:'<!doctype html><title>Author link check</title>'}));
    const [popup]=await Promise.all([page.waitForEvent('popup'),page.locator('.rr-author > a').nth(i).click()]);
    await popup.waitForLoadState('domcontentloaded');
    assert.equal(popup.url(),url);
    await popup.close();await page.context().unroute(url);
   }
   const quote=page.getByRole('blockquote',{name:'TL;DR'});
   assert((await quote.innerText()).includes(language==='en'?'reasoning alone does not guarantee timely responses or coherent, robust trajectories':'仅靠推理并不能保证及时响应，也不能保证连贯、稳健的运动轨迹'));
   assert((await quote.innerText()).includes(language==='en'?'learned action alongside reasoning':'习得的动作能力与推理相结合'));
   assert((await quote.innerText()).includes(language==='en'?'latency remains an unresolved challenge':'延迟仍是尚未解决的挑战'));
   assert((await quote.innerText()).includes('System 1'));
   assert.equal(await page.locator('#equal-contribution-note').innerText(),language==='en'?'* Equal contribution':'* 同等贡献');
   assert.equal(await page.locator('#corresponding-author-note').innerText(),language==='en'?'† Corresponding authors':'† 通讯作者');
   assert.equal(await page.locator('[aria-describedby="corresponding-author-note"]').count(),3);
   assert.equal(await quote.locator('h2').count(),1);
   assert.equal(await quote.locator('p, strong').count(),0);
   assert.equal(await quote.locator('.katex').count(),1);
   const abstract=page.locator('#abstract .rr-prose'),abstractText=await abstract.innerText();
   for(const metric of ['14.4%','85.6%','48%','62.60','26%','37.81','15.67%','24.43','44.8%'])assert(abstractText.includes(metric));
   assert(abstractText.includes(language==='en'?'latency remains a hugely unresolved issue to real-time deployment':'延迟仍是实时部署尚未解决的一大难题'));
   assert(abstractText.includes(language==='en'?'real-world physics has no pause button':'物理过程没有暂停键'));
   assert.equal(await abstract.locator('p').count(),3);
   assert.equal(await page.locator('.rr-institution-brand img').count(),1);
   assert.equal(await page.locator('.rr-institution-brand img').getAttribute('alt'),'Galbot');
   assert(await page.locator('.rr-institution-brand img').evaluate(x=>x.complete&&x.naturalWidth===1412&&x.naturalHeight===446));
   const logoSource=await page.locator('.rr-institution-brand img').getAttribute('src');
   const logoBytes=logoSource.startsWith('data:')?Buffer.from(logoSource.split(',')[1],'base64'):await(await page.request.get(new URL(logoSource,base).href)).body();
   assert.equal(crypto.createHash('sha256').update(logoBytes).digest('hex'),'760f9afbd645287eae72e67665b8e290f137ed7032c4171be071aa8e559fa9ba');
   const contexts=page.locator('.rr-video-card .rr-video-context');
   assert.equal(await contexts.count(),21);
   assert.deepEqual(await contexts.evaluateAll(xs=>xs.map(x=>x.dataset.method)),[...Array(16).fill('mix'),...Array(5).fill('gpt')]);
   assert(await contexts.locator('.rr-video-takeaway').evaluateAll(xs=>xs.every(x=>x.textContent.trim().length>12)));
   assert((await contexts.nth(0).innerText()).includes(language==='zh'?'行为分析：':'Clip takeaway:'));
   assert((await contexts.nth(0).innerText()).includes(language==='zh'?'混合控制':'Hybrid control'));
   assert((await contexts.nth(16).innerText()).includes('GPT 6 Astra（direct）'));
   assert(!(await page.locator('.rr-report').innerText()).includes('GPT 6 Astra Direct'));
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:width===390?844:1050});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
    assert(await page.locator('.rr-author').evaluateAll(xs=>xs.every(x=>x.scrollWidth<=x.clientWidth+1)));
    const logo=await page.locator('.rr-institution-brand img').boundingBox();
    assert(Math.abs(logo.width/logo.height-1412/446)<0.01,'Preserve the full supplied logo aspect ratio');
    assert(Math.abs(logo.width-(width===390?200:220))<1,'Use the smaller responsive logo size');
    const authors=await page.locator('.rr-authors').boundingBox(),brand=await page.locator('.rr-institution-brand').boundingBox(),card=await quote.boundingBox(),links=await page.locator('.rr-project-links').boundingBox(),prose=await page.locator('#abstract .rr-prose').boundingBox();
    assert(authors.y+authors.height<=brand.y&&brand.y+brand.height<=card.y&&card.y+card.height<=links.y);
    assert(Math.abs(card.x-prose.x)<2&&Math.abs(card.width-prose.width)<2,'Quote must match the prose column');
    await page.locator('.rr-hero').screenshot({path:path.join(out,`authors-${language}-${width}.png`)});
   }
   await page.setViewportSize({width:1440,height:1050});
   await abstract.screenshot({path:path.join(out,`abstract-${language}.png`)});
   for(const index of [0,16])await page.locator('.rr-clip').nth(index).screenshot({path:path.join(out,`video-${language}-${index}.png`)});
   await page.locator('.rr-video-card').nth(16).click();
   const dialog=page.locator('.rr-video-dialog[open]');await dialog.waitFor();
   assert.equal(await dialog.locator('.rr-video-context').getAttribute('data-method'),'gpt');
   assert((await dialog.locator('.rr-video-context').innerText()).includes('GPT 6 Astra（direct）'));
   assert(await dialog.locator('video').getAttribute('controls')!==null);
   await dialog.getByRole('button',{name:language==='zh'?'关闭视频':'Close video',exact:true}).click();
  }
  await page.getByRole('button',{name:'切换为英文',exact:true}).click();
  assert((await page.locator('.rr-tldr').innerText()).includes('System 2 reasoning'));
  assert.equal(await page.locator('.rr-author-email, .rr-nav').count(),0);
  assert.equal(await page.locator('.rr-clip').count(),21);assert.deepEqual(errors,[]);
  const html=await(await page.request.get(base)).body();
  for(const gallery of ['gallery.html','robolab-gallery.html']){
   await page.goto(new URL(gallery+'?lang=zh',base).href,{waitUntil:'domcontentloaded'});
   const id=gallery==='gallery.html'?'gpt':'pure_astra';
   assert.equal(await page.locator(`.method-tab[data-method="${id}"]`).innerText(),'GPT 6 Astra（direct）');
  }
  const result={status:'passed',authors:expected,authorLinks:expectedUrls,authorLinkClicks:12,equalContribution:expected.slice(0,2),galbotLogo:true,bilingualTakeaway:true,videoTakeaways:21,videoPolicies:{hybrid:16,direct:5},directName:'GPT 6 Astra（direct）',galleriesRenamed:2,desktop:true,mobile:true,affiliationNotInvented:true,bodyVideos:21,html_sha256:crypto.createHash('sha256').update(html).digest('hex'),pageErrors:errors};
  result.correspondingAuthors=expected.slice(3);
  result.realTimeLimitation=true;result.tokenReduction='44.8%';result.system2Takeaway=true;result.userAbstract=true;
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
