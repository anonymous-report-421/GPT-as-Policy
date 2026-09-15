'use strict';
const $ = id => document.getElementById(id);
const video = $('video');
const state = {rows: [], labels: {}, current: null, draft: null, csrf: '', dirty: false,
  saving: false, refreshing: false, load: 0, clips: [], editing: -1, video: null, selectedStart: 0, selectedEnd: 0, previewEnd: null};
const draftKey = id => 'rollout-review-draft-v1:' + id;
const versionKey = 'rollout-review-context-version';
let selectedVersion = new URLSearchParams(location.search).get('version');
try {if (selectedVersion === null) selectedVersion = localStorage.getItem(versionKey);} catch {}
selectedVersion ??= 'v3';
if (![...$('versionFilter').options].some(o => o.value === selectedVersion)) $('versionFilter').add(new Option(selectedVersion, selectedVersion));
$('versionFilter').value = selectedVersion;
function rememberVersion() {
  try {localStorage.setItem(versionKey, selectedVersion);} catch {}
  const url = new URL(location.href); url.searchParams.set('version', selectedVersion);
  history.replaceState(null, '', url);
}
function clearRun() {
  ++state.load; video.pause(); state.current = state.draft = state.video = null;
  state.dirty = false; state.previewEnd = null; state.clips = [];
  video.removeAttribute('src'); video.load(); $('detail').hidden = true; $('empty').hidden = false;
  history.replaceState(null, '', location.pathname + location.search);
}
async function changeVersion() {
  if (state.saving || (state.dirty && !confirm('当前有未保存标注。保留浏览器草稿并切换版本？'))) {
    $('versionFilter').value = selectedVersion; return;
  }
  if (state.dirty) persistDraft();
  selectedVersion = $('versionFilter').value; rememberVersion();
  if (state.current && selectedVersion && state.current.context_version !== selectedVersion) clearRun();
  renderList(); if (!state.current) await openRun(visibleRows()[0]?.id);
}
const tags = text => [...new Set(text.split(/[,，\n]/).map(x => x.trim()).filter(Boolean))];
const time = s => {if (!Number.isFinite(s)) return '—'; const ms = Math.floor(Math.max(0, s) * 1000);
  return `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(ms % 1000).padStart(3, '0')}`;};
function message(text = '', error = false) {$('notice').textContent = text; $('notice').className = error ? 'error' : '';}
async function api(url, options = {}) {const response = await fetch(url, {cache: 'no-store', ...options});
  const data = await response.json(); if (!response.ok) {const e = new Error(data.error || `HTTP ${response.status}`); e.status = response.status; throw e;} return data;}
function node(tag, text, cls) {const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n;}
function badge(text, cls = '') {return node('span', text, 'badge ' + cls);}
function button(text, action, cls = '') {const b = node('button', text, cls); b.type = 'button'; b.onclick = action; return b;}
function pickFilter(id, values) {const el = $(id), current = el.value, first = el.options[0].text; el.replaceChildren(new Option(first, ''));
  [...new Set(values)].sort().forEach(v => el.add(new Option(v, v))); el.value = values.includes(current) ? current : '';}
function visibleRows() {const search = $('search').value.toLowerCase(); return state.rows.filter(r => {
  const a = r.annotation;
  return (!$('taskFilter').value || r.task === $('taskFilter').value) && (!$('versionFilter').value || r.context_version === $('versionFilter').value)
    && (!$('stateFilter').value || r.status === $('stateFilter').value) && (!$('methodFilter').value || r.method === $('methodFilter').value)
    && (!$('videoFilter').value || r.videos.length > 0)
    && (!$('reviewFilter').value || ({selected: a.selected, reviewed: a.reviewed, unreviewed: !a.reviewed, clips: a.clips.length > 0})[$('reviewFilter').value])
    && (!search || [r.task, r.path, r.case_id, r.context_version, ...a.tags, a.comment, ...a.clips.map(c => c.comment)].join(' ').toLowerCase().includes(search));
});}
function renderList() {const rows = visibleRows(); $('runs').replaceChildren(); $('listCount').textContent = `${rows.length} / ${state.rows.length} 条 attempt`;
  for (const r of rows) {const b = button('', () => openRun(r.id), 'run' + (state.current?.id === r.id ? ' active' : ''));
    b.append(node('strong', r.task)); const line = node('span'); line.append(badge(state.labels[r.status] || r.status, r.status), badge(r.context_version));
    if (r.annotation.selected) line.append(badge('★ 报告候选', 'star')); if (r.annotation.reviewed) line.append(badge('已标注')); b.append(line);
    b.append(node('span', `${r.variant} · layout ${r.seeds.layout_id ?? '—'} · attempt ${r.attempt ?? '—'} · ${r.method}`, 'meta'));
    b.append(node('span', `step ${r.step ?? '—'} / ${r.max_steps ?? '—'} · Score ${r.native_score === null ? '—' : (r.native_score * 100).toFixed(1)} · ${r.annotation.clips.length} 段`, 'meta'));
    b.append(node('span', r.experiment, 'meta')); $('runs').append(b);
  } if (!rows.length) $('runs').append(node('p', '没有符合当前筛选条件的记录。', 'muted'));}
async function refresh() {if (state.refreshing || document.hidden) return; state.refreshing = true;
  try {const data = await api('/api/runs'); state.rows = data.rows; state.labels = data.states; state.csrf = data.csrf;
    pickFilter('taskFilter', data.rows.map(r => r.task)); pickFilter('versionFilter', ['v1','v2','v3',selectedVersion,...data.rows.map(r => r.context_version)].filter(Boolean)); pickFilter('methodFilter', data.rows.map(r => r.method));
    const old = $('stateFilter').value; $('stateFilter').replaceChildren(new Option('全部结果', '')); Object.entries(data.states).forEach(([k,v]) => $('stateFilter').add(new Option(v,k))); $('stateFilter').value = old;
    $('catalogStatus').textContent = `${data.rows.length} 条 attempt · ${new Date().toLocaleTimeString()} 更新`;
    $('storage').textContent = `标注独立保存：${data.annotation_root}。不改原始轨迹，不调用模型或仿真。`; renderList();
    if (!state.current && data.rows.length) {const requested = new URLSearchParams(location.hash.slice(1)).get('run');
      const visible = visibleRows();
      await openRun(visible.some(r => r.id === requested) ? requested : visible[0]?.id);}
  } catch(e) {message('列表读取失败：' + e.message, true);} finally {state.refreshing = false;}}
function cleanClip(c) {return {video_key:c.video_key, start:c.start, end:c.end, comment:c.comment, tags:c.tags || []};}
function draftFrom(a) {return {revision:a.revision, reviewed:a.reviewed, selected:a.selected, tags:a.tags, comment:a.comment, clips:a.clips.map(cleanClip)};}
function persistDraft() {if (!state.current || !state.draft) return; try {localStorage.setItem(draftKey(state.current.id), JSON.stringify({annotation:state.draft, clipForm:{video_key:state.video?.key, start:state.selectedStart, end:state.selectedEnd, comment:$('clipComment').value, tags:$('clipTags').value}}));}
  catch {message('浏览器本地草稿缓存失败；请尽快保存到共享存储。', true);}}
function setDirty() {state.dirty = true; $('saveState').textContent = '有未保存修改'; $('saveState').className = 'dirty'; persistDraft();}
function readForm() {if (!state.draft) return; Object.assign(state.draft, {reviewed:$('reviewed').checked, selected:$('selected').checked, tags:tags($('tags').value), comment:$('comment').value, clips:state.clips.map(cleanClip)}); setDirty();}
function fillAnnotation(a) {state.draft = draftFrom(a); state.clips = state.draft.clips; $('reviewed').checked = a.reviewed; $('selected').checked = a.selected; $('tags').value = a.tags.join(', '); $('comment').value = a.comment;
  state.dirty = false; $('saveState').textContent = `已保存版本 r${a.revision}`; $('saveState').className = 'saved'; renderClips();}
async function openRun(id) {if (!id || state.saving) return; if (state.dirty && !confirm('当前有未保存标注。保留浏览器本地草稿并切换？')) return;
  const serial = ++state.load; video.pause(); message();
  try {const r = await api('/api/run?id=' + encodeURIComponent(id)); if (serial !== state.load) return;
    state.current = r; $('detail').hidden = false; $('empty').hidden = true; state.editing = -1; fillAnnotation(r.annotation);
    $('runTitle').textContent = r.task; $('runIdentity').textContent = `${r.case_id || r.path} · ${r.context_version} · attempt ${r.attempt ?? '—'}`;
    $('runFacts').replaceChildren(...[state.labels[r.status], `Score ${r.native_score === null ? '—' : (r.native_score * 100).toFixed(1)}`, `控制步 ${r.step ?? '—'} / ${r.max_steps ?? '—'}`,
      `${r.method} · ${r.model || '—'} / ${r.effort || '—'}`, `Student / Edit / EEF：${r.student_steps ?? '—'} / ${r.edited_steps ?? '—'} / ${r.recovery_steps ?? '—'}`].map(t => node('span',t,'fact')));
    $('instruction').textContent = r.instruction; $('metadata').textContent = JSON.stringify({path:r.absolute_path,seeds:r.seeds,context_version:r.context_version,method:r.method,native_status:r.outcome_status},null,2);
    $('videoSelect').replaceChildren(...r.videos.map(v => new Option(v.name,v.key))); loadVideo(r.videos[0]?.key);
    renderDecisions(); renderList(); history.replaceState(null,'','#run='+id); resetClipForm();
    const raw = localStorage.getItem(draftKey(id)); if (raw) {try {const d = JSON.parse(raw);
      if (d.annotation.revision === r.annotation.revision && confirm('这条 rollout 有浏览器未保存草稿，是否恢复？')) {
        fillAnnotation(d.annotation); state.dirty = true; $('saveState').textContent = '已恢复未保存草稿'; $('saveState').className = 'dirty';
        if (d.clipForm) {if (r.videos.some(v => v.key === d.clipForm.video_key)) loadVideo(d.clipForm.video_key);
          setSelection(d.clipForm.start,d.clipForm.end); $('clipComment').value = d.clipForm.comment; $('clipTags').value = d.clipForm.tags;}
      } else if (d.annotation.revision !== r.annotation.revision) message('发现旧版本本地草稿，服务器已有更新；未覆盖当前标注。可从浏览器存储保留旧稿后核对。', true);
    } catch {message('本地草稿无法读取；服务器标注未受影响。',true);}}
  } catch(e) {message('无法打开 rollout：'+e.message,true);}}
function loadVideo(key, seek = null) {const v = state.current?.videos.find(v => v.key === key); video.pause(); state.previewEnd = null; state.video = v || null; $('videoError').hidden = true;
  if (!v) {video.removeAttribute('src'); video.load(); $('videoError').textContent = '这条 attempt 尚无已收尾的 debug 视频。原始记录仍可标注。'; $('videoError').hidden = false;}
  else {$('videoSelect').value = key; video.src = '/video?id='+state.current.id+'&video='+key; video.load(); $('downloadVideo').href = video.src; $('downloadVideo').download = `${state.current.task}_${state.current.context_version}_${key}.mp4`;
    if (seek !== null) video.addEventListener('loadedmetadata', () => jump(seek), {once:true});}
  for (const id of ['prevFrame','nextFrame']) $(id).disabled = !v?.fps;
  $('seekStep').disabled = $('seekStepGo').disabled = !v?.step_mapping;
  $('pip').disabled = !document.pictureInPictureEnabled;
  setSelection(0,0); renderClips(); renderDecisions(); updateTime();}
function duration() {return Number.isFinite(video.duration) ? video.duration : state.video?.duration || 0;}
function clamp(t) {return Math.max(0,Math.min(Number(t) || 0,Math.max(0,duration()-0.0001)));}
function jump(t) {if (state.video && video.readyState >= 1) {video.currentTime = clamp(t); updateTime();}}
function togglePlay() {if (!state.video) return; if (video.paused) video.play().catch(e => message('播放失败：'+e.message,true)); else video.pause();}
function stepFrame(direction) {if (!state.video?.fps) return; video.pause(); state.previewEnd = null; const fps = state.video.fps;
  jump((Math.floor(video.currentTime*fps + 0.001)+direction+0.2)/fps);}
function setSelection(start,end) {const max = duration() || 1e6; state.selectedStart = Math.max(0,Math.min(Number(start)||0,max)); state.selectedEnd = Math.max(0,Math.min(Number(end)||0,max));
  $('clipStart').value = state.selectedStart.toFixed(3); $('clipEnd').value = state.selectedEnd.toFixed(3); renderSelection();}
function renderSelection() {const d = duration() || 1, a = state.selectedStart, b = state.selectedEnd;
  $('selectionBand').style.left = `${Math.min(a,b)/d*100}%`; $('selectionBand').style.width = `${Math.abs(b-a)/d*100}%`;
  $('selectionInfo').textContent = `${time(a)} → ${time(b)} · ${(b-a).toFixed(3)} 秒` + (state.video?.step_mapping ? ` · step ${Math.floor(a*state.video.fps)}–${Math.floor(b*state.video.fps)}` : ' · step 映射未核验');}
function updateTime() {const t = video.currentTime || 0, fps = state.video?.fps;
  $('clock').textContent = `${time(t)} / ${time(duration())}`;
  const frame = fps ? Math.min(Math.floor(t*fps+0.001),(state.video.frames || Infinity)-1) : null;
  $('frameInfo').textContent = fps ? `${fps} fps · frame ${frame}` + (state.video.step_mapping ? ` = step ${frame}` : '（时间标注，不推断 step）') : 'FPS 未知，逐帧 / step 跳转禁用';
  $('playhead').style.left = `${t/(duration()||1)*100}%`;
  if (!video.paused && state.previewEnd !== null && t >= state.previewEnd && !$('loop').checked) {video.pause(); state.previewEnd = null;}
  if ($('loop').checked && !video.paused && state.selectedEnd > state.selectedStart && t >= state.selectedEnd) jump(state.selectedStart);
  const step = state.video?.step_mapping ? frame : null;
  for (const el of $('decisions').children) el.classList.toggle('active',step!==null && step>=Number(el.dataset.start) && step<Number(el.dataset.end));}
function resetClipForm() {state.editing = -1; $('clipComment').value = ''; $('clipTags').value = ''; $('addClip').textContent = '添加片段到草稿'; $('cancelClip').hidden = true; renderClips();}
function addClip() {if (!state.video) return message('请先选择已有视频。',true); const start = state.selectedStart,end = state.selectedEnd;
  if (end < start) return message('终点不能早于起点。',true); if (!$('clipComment').value.trim() && !tags($('clipTags').value).length) return message('请填写片段评论或标签。',true);
  const clip = {video_key:state.video.key,start,end,comment:$('clipComment').value,tags:tags($('clipTags').value)};
  if (state.editing >= 0) state.clips[state.editing] = clip; else state.clips.push(clip);
  resetClipForm(); readForm(); message('片段已加入草稿；请保存全部标注。');}
function preview(start = state.selectedStart,end = state.selectedEnd) {if (!state.video || end < start) return; state.previewEnd = end > start ? end : null; jump(start);
  if (end > start) video.play().catch(e=>message(e.message,true)); else video.pause();}
function editClip(index, play = false) {const c = state.clips[index]; if (state.video?.key !== c.video_key) loadVideo(c.video_key,c.start);
  state.editing = index; setSelection(c.start,c.end); $('clipComment').value = c.comment; $('clipTags').value = c.tags.join(', ');
  $('addClip').textContent = '更新此片段到草稿'; $('cancelClip').hidden = false; renderClips(); if(play) {if(video.readyState>=1) preview(); else video.addEventListener('loadedmetadata',()=>preview(),{once:true});}}
function renderClips() {$('clips').replaceChildren(); $('clipMarkers').replaceChildren();
  state.clips.forEach((c,i) => {const card = node('div',undefined,'clip-item'+(state.editing===i?' editing':'')), row = node('div',undefined,'row');
    const v = state.current?.videos.find(v=>v.key===c.video_key); row.append(node('strong',`#${i+1} · ${time(c.start)}–${time(c.end)}`),button('回看',()=>editClip(i,true)),button('编辑',()=>editClip(i)),button('移除',()=>{if(confirm('从本次草稿移除此片段？已保存的历史版本仍保留。')){state.clips.splice(i,1);resetClipForm();readForm();}}));
    card.append(row,node('span',v?.name || c.video_key,'muted small'),node('p',c.comment)); c.tags.forEach(t=>card.append(badge(t))); $('clips').append(card);
    if (state.video?.key === c.video_key) {const m=node('div',undefined,'clip-marker');m.style.left=`${c.start/(duration()||1)*100}%`;m.style.width=`${(c.end-c.start)/(duration()||1)*100}%`;$('clipMarkers').append(m);}
  }); if(!state.clips.length)$('clips').append(node('p','尚无片段标注。可先播放视频，用 I / O 标出值得分析的一段。','muted'));}
function renderDecisions() {$('decisions').replaceChildren(); for(const d of state.current?.decisions || []) {const b=button('',()=>{if(state.video?.step_mapping)jump(d.start_step/state.video.fps);},'decision');
  b.dataset.start=d.start_step;b.dataset.end=d.end_step;b.append(node('strong',`#${d.index} · step ${d.start_step}–${d.end_step} · ${d.mode}`),node('p',d.reason));
  b.disabled=!state.video?.step_mapping;$('decisions').append(b);} if(!$('decisions').children.length)$('decisions').append(node('p','没有可读取的结构化公开决策。','muted'));}
async function save() {if(!state.current || state.saving)return;
  if($('clipComment').value.trim() || $('clipTags').value.trim())return message('当前片段编辑框还有内容。先“添加/更新片段到草稿”，或取消片段编辑，再保存。',true);
  readForm();state.saving=true;$('save').disabled=true;$('saveState').textContent='正在保存…';
  try {const a=await api('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':state.csrf},body:JSON.stringify({id:state.current.id,annotation:state.draft})});
    state.current.annotation=a; const r=state.rows.find(r=>r.id===state.current.id);if(r)r.annotation=a;fillAnnotation(a);localStorage.removeItem(draftKey(state.current.id));renderList();message(`已保存到共享存储 · r${a.revision} · ${new Date(a.updated_utc).toLocaleTimeString()}`);
  } catch(e){message(e.status===409?'保存冲突：另一页面已有新版本。当前草稿仍在，请先复制评论后重新读取服务器标注并合并。':'保存失败，草稿保留：'+e.message,true);$('saveState').textContent='未保存（草稿保留）';}
  finally{state.saving=false;$('save').disabled=false;}}
async function reloadAnnotation(){if(!state.current||state.saving)return;if(state.dirty&&!confirm('重新读取将放弃当前浏览器草稿，确定吗？'))return;
  try{const r=await api('/api/run?id='+state.current.id);state.current.annotation=r.annotation;fillAnnotation(r.annotation);resetClipForm();localStorage.removeItem(draftKey(state.current.id));message('已读取服务器最新标注。');}catch(e){message(e.message,true);}}
function adjacent(delta){const rows=visibleRows(),index=rows.findIndex(r=>r.id===state.current?.id);if(index>=0&&rows[index+delta])openRun(rows[index+delta].id);}
function fullscreen(){if(document.fullscreenElement)document.exitFullscreen().catch(()=>{});else $('viewer').requestFullscreen().catch(e=>message(e.message,true));}
$('refresh').onclick=refresh;['search','taskFilter','stateFilter','methodFilter','reviewFilter','videoFilter'].forEach(id=>$(id).addEventListener('input',renderList));
$('versionFilter').addEventListener('change',changeVersion);
['selected','reviewed','tags','comment'].forEach(id=>$(id).addEventListener('input',readForm));
['clipComment','clipTags'].forEach(id=>$(id).addEventListener('input',setDirty));
$('save').onclick=save;$('reloadAnnotation').onclick=reloadAnnotation;$('previous').onclick=()=>adjacent(-1);$('next').onclick=()=>adjacent(1);
$('videoSelect').onchange=()=>{if(($('clipComment').value||$('clipTags').value)&&!confirm('切换视频会清空当前未添加的片段编辑框，确定吗？')){$('videoSelect').value=state.video?.key||'';return;}resetClipForm();loadVideo($('videoSelect').value);};
$('play').onclick=togglePlay;$('back5').onclick=()=>jump(video.currentTime-5);$('forward5').onclick=()=>jump(video.currentTime+5);$('prevFrame').onclick=()=>stepFrame(-1);$('nextFrame').onclick=()=>stepFrame(1);
$('speed').onchange=()=>{video.playbackRate=Number($('speed').value);localStorage.setItem('rollout-review-speed',$('speed').value);};
try{const saved=localStorage.getItem('rollout-review-speed');if([...$('speed').options].some(o=>o.value===saved))$('speed').value=saved;}catch{}
$('zoom').onchange=()=>{const zoom=$('zoom').value;$('viewer').classList.toggle('zoomed',zoom!=='fit');video.style.width=zoom==='fit'?'100%':`${Number(zoom)*100}%`;};
$('seekTime').onclick=()=>jump($('seekSeconds').value);$('seekStepGo').onclick=()=>{if(state.video?.step_mapping)jump(Number($('seekStep').value)/state.video.fps);};
$('fullscreen').onclick=fullscreen;$('pip').onclick=()=>{if(document.pictureInPictureEnabled)video.requestPictureInPicture().catch(e=>message(e.message,true));};
$('markIn').onclick=()=>{setSelection(video.currentTime,Math.max(video.currentTime,state.selectedEnd));persistDraft();};
$('markOut').onclick=()=>{setSelection(Math.min(state.selectedStart,video.currentTime),video.currentTime);persistDraft();};
['clipStart','clipEnd'].forEach(id=>{ $(id).oninput=()=>{state.selectedStart=Number($('clipStart').value)||0;state.selectedEnd=Number($('clipEnd').value)||0;renderSelection();persistDraft();};
  $(id).onchange=()=>{setSelection($('clipStart').value,$('clipEnd').value);persistDraft();}; });
$('previewClip').onclick=()=>preview();$('addClip').onclick=addClip;$('cancelClip').onclick=resetClipForm;
let dragging=null, scrubFrame=null;
const timelineTime=e=>clamp((e.clientX-$('timeline').getBoundingClientRect().left)/$('timeline').getBoundingClientRect().width*duration());
function scrubTo(t) {if(!dragging)return;dragging.time=t;setSelection(Math.min(t,dragging.anchor),Math.max(t,dragging.anchor));
  // Coalesce pointer events, but seek during the drag (not just at its start).
  if(scrubFrame===null)scrubFrame=requestAnimationFrame(()=>{scrubFrame=null;if(dragging)jump(dragging.time);});}
function finishScrub(e,cancelled=false) {if(!dragging||e.pointerId!==dragging.pointerId)return;
  if(!cancelled)scrubTo(timelineTime(e));
  if(scrubFrame!==null){cancelAnimationFrame(scrubFrame);scrubFrame=null;}
  const {pointerId,time}=dragging;dragging=null;jump(time);video.pause();persistDraft();
  if($('timeline').hasPointerCapture(pointerId))$('timeline').releasePointerCapture(pointerId);}
$('timeline').onpointerdown=e=>{if(dragging||e.button!==0||!state.video||!duration()||video.readyState<1)return;
  e.preventDefault();$('timeline').focus({preventScroll:true});video.pause();state.previewEnd=null;
  const t=timelineTime(e);dragging={pointerId:e.pointerId,anchor:t,time:t};$('timeline').setPointerCapture(e.pointerId);setSelection(t,t);jump(t);};
$('timeline').onpointermove=e=>{if(dragging&&e.pointerId===dragging.pointerId)scrubTo(timelineTime(e));};
$('timeline').onpointerup=e=>finishScrub(e);
$('timeline').onpointercancel=e=>finishScrub(e,true);
$('timeline').onlostpointercapture=e=>finishScrub(e,true);
video.addEventListener('contextmenu',e=>e.preventDefault());
video.addEventListener('loadedmetadata',()=>{video.playbackRate=Number($('speed').value);updateTime();renderSelection();renderClips();});
video.addEventListener('timeupdate',updateTime);video.addEventListener('seeked',updateTime);video.addEventListener('ended',()=>{if($('loop').checked&&state.selectedEnd>state.selectedStart)preview();});
video.addEventListener('error',()=>{$('videoError').textContent='视频暂时无法播放。可能尚未收尾、网络读取失败或浏览器不支持编码；可尝试刷新或下载原视频。';$('videoError').hidden=false;});
if('requestVideoFrameCallback' in video){const frame=()=>{updateTime();video.requestVideoFrameCallback(frame);};video.requestVideoFrameCallback(frame);}
$('history').onclick=async()=>{try{const r=await api('/api/revisions?id='+state.current.id);$('historyContent').textContent=JSON.stringify(r.revisions,null,2);$('historyDialog').showModal();}catch(e){message(e.message,true);}};
$('closeHistory').onclick=()=>$('historyDialog').close();$('exportOpen').onclick=()=>$('exportDialog').showModal();$('closeExport').onclick=()=>$('exportDialog').close();
document.querySelectorAll('[data-export]').forEach(b=>b.onclick=()=>{const link=document.createElement('a');const params=new URLSearchParams({format:b.dataset.export,selected:$('exportSelected').checked?'1':'0'});if($('exportVersion').checked&&selectedVersion)params.set('version',selectedVersion);link.href='/api/export?'+params;link.download='rollout_review.'+b.dataset.export;link.click();});
['规划有效','纠错成功','视觉误判','重复尝试','动作精度','历史记忆','停止判断','恢复失败','值得展示'].forEach(t=>$('tagSuggestions').append(button(t,()=>{$('tags').value=[...new Set([...tags($('tags').value),t])].join(', ');readForm();})));
document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();save();return;}
  if(e.isComposing||e.ctrlKey||e.metaKey||e.altKey||dragging||/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||e.target.isContentEditable||document.querySelector('dialog[open]'))return;
  // Physical key codes also work when the keyboard emits Chinese punctuation.
  const key=({Comma:',',Period:'.'})[e.code]||e.key.toLowerCase(),actions={' ':togglePlay,k:togglePlay,j:()=>jump(video.currentTime-5),l:()=>jump(video.currentTime+5),arrowleft:()=>jump(video.currentTime-1),arrowright:()=>jump(video.currentTime+1),',':()=>stepFrame(-1),'.':()=>stepFrame(1),i:()=>$('markIn').click(),o:()=>$('markOut').click(),f:fullscreen,
    '[':()=>{const s=$('speed');s.selectedIndex=Math.max(0,s.selectedIndex-1);s.onchange();},']':()=>{const s=$('speed');s.selectedIndex=Math.min(s.options.length-1,s.selectedIndex+1);s.onchange();}};
  if(actions[key]){e.preventDefault();actions[key]();}});
window.addEventListener('beforeunload',e=>{if(state.dirty){persistDraft();e.preventDefault();e.returnValue='';}});
setInterval(refresh,45000);refresh();
