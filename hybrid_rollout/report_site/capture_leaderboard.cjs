// Read-only public reference capture, using the existing isolated browser.
const {chromium} = require('/tmp/rollout-review-browser.HpuzaB/node_modules/playwright');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1050}});
    const gallery = process.argv.includes('--gallery');
    await page.goto(gallery ? 'https://robodojo-benchmark.com/leaderboard/rollouts/OpenWAM-%CE%B1?bench=sim' : 'https://robodojo-benchmark.com/leaderboard',{waitUntil:'networkidle',timeout:45000});
    if (!gallery) await page.getByText('Top 10 of 40 models racing toward the unconquered peak.',{exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:gallery ? '/tmp/report-robodojo-gallery-reference.png' : '/tmp/report-robodojo-reference-viewport.png'});
    const info = await page.evaluate(()=>({title:document.title,text:document.body.innerText,
      images:[...document.images].map(i=>({src:i.currentSrc,alt:i.alt})),
      scripts:[...document.scripts].map(s=>s.src).filter(Boolean)}));
    fs.writeFileSync(gallery ? '/tmp/report-robodojo-gallery-reference.json' : '/tmp/report-robodojo-reference.json',JSON.stringify(info,null,2));
    console.log(JSON.stringify({...info,text:info.text.slice(0,1700)}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exit(1)});
