// Real browser checks for the static RoboLab gallery; no policy/model calls.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const execFileAsync=promisify(execFile);
const {chromium}=require(process.env.REPORT_BROWSER_RUNTIME||'playwright');
let rangeRetries=0;

async function videoRange(request,url){
  for(let attempt=0;attempt<3;attempt++){
    try{return await request.get(url,{headers:{Range:'bytes=0-1023'},timeout:30000});}
    catch(error){
      if(attempt===2||!/Timeout|ECONNRESET|ETIMEDOUT/.test(String(error)))throw error;
      rangeRetries++;
    }
  }
}

async function readyVideo(page,selector){
  await page.locator(selector).evaluate(video=>new Promise((resolve,reject)=>{
    if(video.readyState>=2)return resolve();
    const timer=setTimeout(()=>reject(Error('Video loading timed out')),15000);
    video.addEventListener('loadeddata',()=>{clearTimeout(timer);resolve();},{once:true});
    video.addEventListener('error',()=>{clearTimeout(timer);reject(Error('Video decode failed'));},{once:true});
  }));
}

(async()=>{
  const base=process.argv[2], output=process.argv[3];fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100}}), errors=[];
    page.on('pageerror',error=>errors.push(String(error)));
    await page.goto(base,{waitUntil:'domcontentloaded'});
    const link=page.getByRole('link',{name:'RoboLab Video Gallery',exact:true}).first();
    await link.click();await page.waitForURL(/robolab-gallery/);
    await readyVideo(page,'#video');
    const data=await page.evaluate(()=>window.REPORT_DATA);
    assert.equal(data.cases.length,150);assert.equal(data.tasks.length,10);
    assert.equal(await page.locator('.method-tab').count(),3);
    assert.equal(await page.locator('#benchmark-galleries a').count(),2);
    assert.equal(await page.locator('#paired').isVisible(),false);
    let cells=0;
    for(const method of data.methods){
      await page.locator(`.method-tab[data-method="${method.id}"]`).click();
      for(const task of data.tasks){
        await page.locator(`.task-button[data-task="${task.id}"]`).click();
        assert.equal(await page.locator('.rollout-button').count(),5);
        const rows=data.cases.filter(r=>r.method===method.id&&r.task===task.id);
        assert.equal(await page.locator('#task-average').innerText(),`${rows.filter(r=>r.status==='success').length*20}%`);
        await page.locator('.rollout-button').last().click();
        assert.equal(new URL(page.url()).searchParams.get('id'),rows[4].id);cells++;
      }
      await readyVideo(page,'#video');
      await page.locator('#video').evaluate(async v=>{await v.play();});
      await page.waitForTimeout(250);
      assert.ok(await page.locator('#video').evaluate(v=>v.currentTime>0&&!v.paused));
      await page.locator('#video-card').click();await readyVideo(page,'#expanded-video');
      await page.locator('#expanded-video').evaluate(v=>new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(Error('Seek timed out')),10000);
        v.addEventListener('seeked',()=>{clearTimeout(timer);resolve();},{once:true});
        v.currentTime=Math.min(v.duration/2,10);
      }));
      assert.ok(await page.locator('#expanded-video').evaluate(v=>v.currentTime>0));
      await page.locator('#close-dialog').click();
    }
    for(let start=0;start<data.cases.length;start+=5){
      await Promise.all(data.cases.slice(start,start+5).map(async row=>{
        const url=new URL(row.video_url||row.video,page.url()).href;
        if(process.env.REPORT_RANGE_TRANSPORT==='curl'){
          // curl respects the machine's existing proxy environment. Never print it
          // or put proxy credentials in process arguments; do not bypass failures.
          const {stdout}=await execFileAsync('curl',['--silent','--show-error','--fail','--location',
            '--connect-timeout','10','--max-time','40','--retry','2','--retry-all-errors','--range','0-1023',
            '--write-out','\n%{http_code}',url],{encoding:'buffer',maxBuffer:1024*1024,timeout:135000});
          assert.equal(stdout.subarray(-4).toString(),'\n206',row.id);
          assert.equal(stdout.length-4,1024,row.id);
        }else{
          const response=await videoRange(page.request,url);
          assert.equal(response.status(),206,row.id);assert.equal((await response.body()).length,1024,row.id);
          await response.dispose();
        }
      }));
    }
    await page.locator('#search').fill('no-such-task');assert.equal(await page.locator('.task-button').count(),0);
    await page.locator('#search').fill('');assert.equal(await page.locator('.task-button').count(),10);
    await page.locator('#language').click();await page.waitForURL(/lang=zh/);await readyVideo(page,'#video');
    assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
    assert.equal(await page.locator('#average-label').innerText(),'成功率');
    assert.equal(await page.locator('.rollout-button').count(),5);
    await page.locator('#language').click();await page.waitForURL(/lang=en/);await readyVideo(page,'#video');
    await page.screenshot({path:path.join(output,'gallery-desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.screenshot({path:path.join(output,'gallery-mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    const result={videos:150,taskMethodCells:cells,rangeRequests:150,rangeRetries,
      rangeTransport:process.env.REPORT_RANGE_TRANSPORT||'playwright',
      methodPlaybackAndSeek:3,bilingual:true,mobile:true,errors};
    fs.writeFileSync(path.join(output,'checks.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
