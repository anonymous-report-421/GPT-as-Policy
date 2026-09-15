'use strict';
const D=window.REPORT_DATA,$=id=>document.getElementById(id),v=$('video'),expanded=$('expanded-video'),dialog=$('video-dialog');
const language=new URLSearchParams(location.search).get('lang')==='zh'?'zh':'en';
const t=(zh,en)=>language==='zh'?zh:en;
const names=Object.fromEntries(D.tasks.map(row=>[row.id,language==='zh'?row.label:(D.translations?.[row.id]?.label||row.id)]));
const labels={
 '← 评测报告':'← Evaluation report','RoboDojo · 轨迹视频':'RoboDojo · Rollout videos',
 '轨迹视频库':'Rollout video gallery','选择任务与运行实例，查看完整操作过程。':'Select a task and rollout to view the complete execution.',
 '策略选择':'Policy selection','完整评测':'Full rollouts','行为片段':'Behavior clips','视频类别':'Video scope',
 '任务列表':'Task list','搜索任务…':'Search tasks…','搜索任务':'Search tasks','轨迹播放器':'Rollout player',
 'Score / 成功率':'Score / Success rate','运行实例':'Rollout instances','没有符合条件的视频。':'No matching videos.',
 '放大观看当前视频':'Expand current video','视频暂时无法加载，请通过下载链接查看。':'The video could not load. Try the download link.',
 '同 seed 对照':'Same-seed comparison','下载视频':'Download video','放大视频':'Expanded video','关闭视频':'Close video',
 '关闭':'Close','RoboDojo 具身策略评测 · 完整过程与行为片段':'RoboDojo embodied policy evaluation · Full rollouts and behavior clips',
};
if(language==='en'){
 const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
 while(walker.nextNode()){const n=walker.currentNode,key=n.textContent.trim();if(labels[key])n.textContent=n.textContent.replace(key,labels[key]);}
 document.querySelectorAll('[aria-label],[placeholder]').forEach(n=>['aria-label','placeholder'].forEach(a=>{if(labels[n.getAttribute(a)])n.setAttribute(a,labels[n.getAttribute(a)]);}));
}
document.documentElement.lang=language==='zh'?'zh-CN':'en';
document.title=t('轨迹视频库 · GPT 6 Astra 具身策略评测','Rollout Gallery · GPT 6 Astra as an Embodied Policy');
$('language').textContent=t('English','中文');
$('language').onclick=()=>{const url=new URL(location.href);url.searchParams.set('lang',language==='zh'?'en':'zh');location.href=url.href;};
$('back-report').href='index.html?lang='+language;
const benchmark=D.benchmark||'RoboDojo', isRoboLab=benchmark==='RoboLab';
const methods=(D.methods||[{id:'mix',label:'π0.5 + GPT 6 Astra'},{id:'gpt',label:'GPT 6 Astra（direct）'}]).map(row=>
 ({...row,label:['gpt','pure_astra'].includes(row.id)?'GPT 6 Astra（direct）':row.label}));
const methodName=m=>methods.find(row=>row.id===m)?.label||m;
const outcome={success:t('成功','Success'),failure:t('未成功','Failure'),idle:t('超时','Timeout')};
const requested=new URLSearchParams(location.search).get('id');
const initial=[...D.cases,...D.clips].find(r=>r.id===requested);
let method=initial?.method||methods[0].id,task=initial?.task||D.tasks[0].id,selected=null,visible=[];
if(initial&&D.clips.includes(initial))$('scope').value='clips';
const el=(tag,text)=>{const node=document.createElement(tag);node.textContent=text;return node;};
const dataset=()=> $('scope').value==='clips'?D.clips:D.cases;
const isClip=row=>D.clips.includes(row);
function updateUrl(row){const u=new URL(location.href);u.searchParams.set('id',row.id);history.replaceState(null,'',u);}
function select(row,update=true){
 selected=row;v.pause();v.poster=row.poster_url||row.poster||'';v.src=row.video_url||row.video;v.muted=true;v.load();v.play().catch(()=>{});$('video-error').hidden=true;
 const index=visible.findIndex(r=>r.id===row.id);
 $('run-label').textContent=methodName(row.method)+' · '+(isClip(row)?t('片段','Clip'):'Rollout')+' '+(index+1);
 $('result').textContent=outcome[row.status]||t('行为片段','Behavior clip');$('result').dataset.result=row.status||'clip';
 $('prompt').textContent=row.instruction;$('download').href=row.video_url||row.video;$('paired').hidden=D.paired===false||!row.seeds;
 const fields=isRoboLab?{[t('控制步','Control steps')]:row.limit==null?row.steps:row.steps+' / '+row.limit,[t('时长','Duration')]:row.duration.toFixed(2)+' s'}:row.seeds?{'Score':row.score==null?'—':(row.score*100).toFixed(1),[t('场景','Scene')]:row.variant==='random'?t('随机','Random'):t('标准','Standard'),'Layout':row.seeds.layout_id,'Seed':row.seeds.reset_seed,[t('控制步','Control steps')]:row.steps+' / '+row.limit}:{[t('区间','Range')]:row.start.toFixed(2)+'–'+row.end.toFixed(2)+' s',[t('时长','Duration')]:row.duration.toFixed(2)+' s'};
 $('facts').replaceChildren(...Object.entries(fields).map(([key,value])=>{const item=el('div','');item.append(el('dt',key),el('dd',String(value)));return item;}));
 document.querySelectorAll('.rollout-button').forEach(b=>b.setAttribute('aria-current',String(b.dataset.id===row.id)));
 if(update)updateUrl(row);
}
function renderTaskList(){
 const q=$('search').value.trim().toLowerCase(),rows=dataset().filter(r=>r.method===method);
 const tasks=D.tasks.filter(row=>rows.some(r=>r.task===row.id)&&(!q||(row.id+' '+row.label+' '+names[row.id]).toLowerCase().includes(q)));
 $('task-count').textContent=tasks.length+t(' 个任务',' tasks');
 $('task-list').replaceChildren(...tasks.map(row=>{const b=el('button','');b.className='task-button';b.setAttribute('aria-current',String(row.id===task));b.dataset.task=row.id;b.append(el('strong',names[row.id]),el('small',row.id),el('span',rows.filter(r=>r.task===row.id).length+t(' 条视频',' videos')));b.onclick=()=>{task=row.id;render();};return b;}));
 if(!tasks.length)$('task-list').append(el('p',t('没有匹配的任务。','No matching tasks.')));
}
function render(preferred){
 document.querySelectorAll('.method-tab').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.method===method)));
 const rows=dataset().filter(r=>r.method===method);if(!rows.some(r=>r.task===task))task=rows[0]?.task||task;
 visible=rows.filter(r=>r.task===task);renderTaskList();
 $('task-title').textContent=names[task]||task;$('task-id').textContent=task;
 const cases=D.cases.filter(r=>r.method===method&&r.task===task),scored=cases.filter(r=>r.score!=null);
 $('task-average').textContent=cases.length?(scored.length?(scored.reduce((s,r)=>s+r.score,0)*100/scored.length).toFixed(1):'—')+' / '+(cases.filter(r=>r.status==='success').length*100/cases.length).toFixed(0)+'%':'—';
 $('task-average').title=t('Score / 成功率；','Score / Success rate; ')+scored.length+t(' 条有分，',' scored, ')+cases.length+t(' 条评测',' evaluated');
 $('score-coverage').hidden=scored.length===cases.length;
 $('score-coverage').textContent=t('Score 基于 ','Score based on ')+scored.length+' / '+cases.length+t(' 条有分结果',' scored episodes');
 if(isRoboLab){$('average-label').textContent=t('成功率','Success rate');$('task-average').textContent=cases.length?(cases.filter(r=>r.status==='success').length*100/cases.length).toFixed(0)+'%':'—';$('task-average').title=cases.filter(r=>r.status==='success').length+' / '+cases.length;$('score-coverage').hidden=true;}
 $('count').textContent=visible.length+($('scope').value==='clips'?t(' 条片段',' clips'):t(' 条运行',' rollouts'));
 $('rollout-list').replaceChildren(...visible.map((row,i)=>{const b=el('button',(isClip(row)?t('片段','Clip'):'Rollout')+' '+(i+1));b.className='rollout-button';b.dataset.id=row.id;b.onclick=()=>select(row);return b;}));
 $('empty').hidden=!!visible.length;$('video-card').hidden=!visible.length;$('video-caption').hidden=!visible.length;
 if(visible.length)select(visible.find(r=>r.id===preferred?.id)||visible[0],!preferred);
 else{v.pause();v.removeAttribute('src');v.load();$('facts').replaceChildren();$('prompt').textContent='';}
}
document.querySelectorAll('.method-tab').forEach(b=>b.remove());
methods.forEach(row=>{const button=el('button',row.label);button.className='method-tab';button.dataset.method=row.id;button.type='button';$('scope').before(button);});
if(isRoboLab){
 document.querySelector('header>span').textContent='RoboLab · '+t('轨迹视频','Rollout videos');
 document.querySelector('h1').textContent=t('RoboLab 视频库','RoboLab Video Gallery');
 document.title=t('RoboLab 视频库','RoboLab Video Gallery')+' · GPT 6 Astra';
 document.querySelector('footer').textContent='RoboLab · 10 '+t('个任务','tasks')+' · 150 '+t('条运行','rollouts');
 $('scope').hidden=true;
}
const galleries=D.galleries||[];
$('benchmark-galleries').hidden=!galleries.length;
galleries.forEach(row=>{const a=el('a',row.benchmark+t(' 视频库',' Video Gallery'));const url=new URL(row.href,location.href);url.searchParams.set('lang',language);a.href=url.href;if(row.benchmark===benchmark)a.setAttribute('aria-current','page');$('benchmark-galleries').append(a);});
document.querySelectorAll('.method-tab').forEach(b=>b.onclick=()=>{method=b.dataset.method;render();});
$('scope').onchange=()=>render();$('search').oninput=renderTaskList;
$('paired').onclick=()=>{const other=D.cases.find(r=>r.case_id===selected.case_id&&r.method!==selected.method);if(other){method=other.method;$('scope').value='cases';task=other.task;render(other);updateUrl(other);}};
$('video-card').onclick=()=>{v.pause();$('dialog-title').textContent=names[task]+' · '+$('run-label').textContent;expanded.src=selected.video_url||selected.video;expanded.muted=true;expanded.onloadedmetadata=()=>{expanded.currentTime=v.currentTime;expanded.play().catch(()=>{});};dialog.showModal();};
$('close-dialog').onclick=()=>dialog.close();dialog.onclick=e=>{if(e.target===dialog)dialog.close();};
dialog.onclose=()=>{expanded.pause();expanded.removeAttribute('src');expanded.load();v.play().catch(()=>{});};
v.onerror=()=>{$('video-error').hidden=false;};
document.addEventListener('visibilitychange',()=>{if(document.hidden){v.pause();expanded.pause();}else if(!dialog.open)v.play().catch(()=>{});});
render(initial);
