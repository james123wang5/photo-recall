const icons = {
 moon:'<path d="M20.6 13.3A8.7 8.7 0 0 1 10.7 3.4 8.8 8.8 0 1 0 20.6 13.3Z"/>',
 sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
 grid:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
 'grid-small':'<path d="M3 3h4v4H3Zm7 0h4v4h-4Zm7 0h4v4h-4ZM3 10h4v4H3Zm7 0h4v4h-4Zm7 0h4v4h-4ZM3 17h4v4H3Zm7 0h4v4h-4Zm7 0h4v4h-4Z"/>',
 'arrow-up':'<path d="M12 20V4m-6 6 6-6 6 6"/>',
 folder:'<path d="M3 7h6l2-3h9a1 1 0 0 1 1 1v13a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
 lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2"/>',
 refresh:'<path d="M20 8a8 8 0 0 0-14-3L3 8m0-5v5h5m-4 8a8 8 0 0 0 14 3l3-3m0 5v-5h-5"/>',
 search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
 sliders:'<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="8" cy="6" r="2" fill="white"/><circle cx="16" cy="12" r="2" fill="white"/><circle cx="10" cy="18" r="2" fill="white"/>',
 spark:'<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z"/>',
 image:'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 6-6 4 4 3-3 5 5"/>'
};
const $ = s => document.querySelector(s);
const escapeHTML = s => String(s ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const icon = name => `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.image}</svg>`;
function hydrateIcons(){document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));}
let state, currentLibrary, session, results = [], currentPhoto, pollTimer, busy = false, inspections = {}, lastResponse;
let searchSequence = 0;
let cancelInspection = false;
let manualInspectionRunning = false;
let verificationTimer, verification;
let jobHideTimer, visibleOrder = [], rankingPending = false;
let previewAppliedSequence = -1;
const cards = new Map();
let savedSessions;
try {savedSessions=JSON.parse(localStorage.getItem('photorecall-sessions') || '{}');} catch {savedSessions={};}
async function api(path, body, retried=false){
 const options = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-PhotoRecall-Token':state.token},body:JSON.stringify(body)};
 const res = await fetch(path,options); const data = await res.json();
 if(res.status===403 && body!==undefined && !retried && typeof data.detail==='string' && data.detail.includes('页面已过期')){
   const fresh=await fetch('/api/state').then(r=>r.json());state.token=fresh.token;return api(path,body,true);
 }
 if(!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求未完成，请检查输入后重试。');
 return data;
}
let toastTimer;
function toast(text){$('#toast').textContent=text;$('#toast').classList.remove('hidden');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.add('hidden'),6500);}
function renderLibraries(){
 $('#library-total').textContent=state.libraries.filter(l=>!l.is_demo).length;
 $('#mobile-library').innerHTML=state.libraries.map(lib=>`<option value="${escapeHTML(lib.id)}">${escapeHTML(lib.name)}</option>`).join('');
 $('#mobile-library').value=currentLibrary.id;
 $('#library-list').innerHTML=state.libraries.map(lib=>`<button class="library-button ${currentLibrary?.id===lib.id?'active':''}" data-library="${escapeHTML(lib.id)}" title="${escapeHTML(lib.name)}">${icon(lib.is_demo?'spark':'folder')}<span class="library-text"><strong>${escapeHTML(lib.name)}</strong><small>${lib.is_demo?'演示插画 · 体验流程':lib.semantic_count?'画面索引已建立':'文件夹索引'}</small></span><span class="count">${lib.count}</span></button>`).join('');
 document.querySelectorAll('[data-library]').forEach(button=>button.onclick=()=>selectLibrary(button.dataset.library));
}
async function selectLibrary(id){
 clearTimeout(pollTimer);clearTimeout(verificationTimer);clearTimeout(jobHideTimer); visibleOrder=[];cards.clear();$('#gallery').replaceChildren(); currentLibrary=state.libraries.find(x=>x.id===id) || state.libraries[0];
 session=null;inspections={};localStorage.setItem('photorecall-library',currentLibrary.id);
 $('#library-title').textContent=currentLibrary.name;$('#update-index').disabled=!!currentLibrary.is_demo;
 $('#job-panel').classList.add('hidden');$('#analysis-button').disabled=!!currentLibrary.is_demo;renderLibraries();
 await search('', 'append'); if(!currentLibrary.is_demo) pollJob();
}
function renderConditions(){
 const hasConditions=!!session?.conditions.length;
 $('.intro').classList.toggle('hidden',hasConditions);
 $('.condition-area').classList.toggle('hidden',!hasConditions);
 $('.welcome').classList.toggle('hidden',!!session?.messages.length);
 $('#conditions').innerHTML=session?.conditions.length?session.conditions.map(c=>`<span class="condition-chip ${escapeHTML(c.kind)}"><small>${{include:'包含',prefer:'可能',exclude:'排除'}[c.kind]}</small>${escapeHTML(c.label)}<button aria-label="移除 ${escapeHTML(c.label)} 条件" data-remove="${c.id}">×</button></span>`).join(''):'<span class="no-conditions">不用记得时间，先描述你记得的画面。</span>';
 document.querySelectorAll('[data-remove]').forEach(button=>button.onclick=()=>search('', 'append', button.dataset.remove));
}
function renderMessages(response){
 const history=session?.messages || [];
 const reply=response.mode==='semantic'?'主结果只显示核对较符合的照片。筛选会逐步进行，也可以继续补充线索。':response.message;
 $('#messages').innerHTML=history.map((msg,i)=>`<div class="user-message">${escapeHTML(msg.text)}</div>${i===history.length-1?`<div class="reply-message">${escapeHTML(reply)}</div>`:''}`).join('');
 $('.welcome').classList.toggle('hidden',!!history.length);$('#chat-scroll').scrollTop=$('#chat-scroll').scrollHeight;
}
function renderGallery({live=false}={}){
 const filtering=!!session?.conditions.length&&!currentLibrary.is_demo;
 const groups=PhotoMatches.partition(results),otherView=filtering&&$('#show-pending').checked;
 let ordered=(filtering?(otherView?groups.other:groups.matched):results).slice();
 $('#pending-control').classList.toggle('hidden',!filtering);
 $('#filter-summary').textContent=`${groups.matched.length} 张较符合 · ${groups.unreviewed.length} 张待筛选 · ${groups.uncertain.length} 张不确定 · ${groups.not_matched.length} 张未保留。已核对 ${results.length-groups.unreviewed.length} / ${results.length} 张候选${(lastResponse?.total||0)>results.length?`，全库 ${lastResponse.total} 张`:''}。`;
 $('#continue-filter').disabled=manualInspectionRunning||verification?.status==='running'||!groups.unreviewed.length;
 if($('#sort').value==='date')ordered.sort((a,b)=>(b.captured||'').localeCompare(a.captured||''));
 else if($('#sort').value==='base'){const ranks=new Map((verification?.base_order||[]).map((id,i)=>[id,i]));ordered.sort((a,b)=>(ranks.get(a.id)??a.rank_order??99999)-(ranks.get(b.id)??b.rank_order??99999));}
 else if(verification?.pipeline==='vnext')ordered.sort((a,b)=>(a.rank_order??99999)-(b.rank_order??99999));
 else ordered.sort((a,b)=>(b.inspection_tier??1)-(a.inspection_tier??1)||(b.inspection_score??0)-(a.inspection_score??0)||b.score-a.score);
 const nextOrder=ordered.map(p=>p.id);
 if(live && visibleOrder.length && $('#sort').value==='relevance'){
  rankingPending=nextOrder.join('|')!==visibleOrder.join('|');
  const positions=new Map(visibleOrder.map((id,i)=>[id,i]));
  ordered.sort((a,b)=>(positions.get(a.id)??Infinity)-(positions.get(b.id)??Infinity));
 }else rankingPending=false;
 $('#apply-ranking').classList.toggle('hidden',!rankingPending);
 $('#apply-ranking').textContent=verification?.status==='running'?'新排序 · 点击更新':'排序已完成 · 点击更新';
 const gallery=$('#gallery'), keep=new Set(nextOrder);
 for(const [id,card] of cards){if(!keep.has(id)){card.remove();cards.delete(id);}}
 ordered.forEach((photo,i)=>{
  let card=cards.get(photo.id);
  if(!card){
   card=document.createElement('article');card.className='photo-card';card.dataset.id=photo.id;
   card.innerHTML='<button class="photo-image-button"><img loading="lazy" decoding="async"><span class="photo-badge"></span><span class="rank-badge"></span></button><h3></h3><div class="card-meta"><span></span><span></span></div><div class="card-reasons"></div>';
   cards.set(photo.id,card);card.querySelector('button').onclick=()=>showPhoto(photo.id);
  }
  const button=card.querySelector('button'),img=card.querySelector('img');
  button.setAttribute('aria-label','查看 '+photo.name);
  if(img.getAttribute('src')!==photo.thumbnail)img.src=photo.thumbnail;
  img.alt=photo.name;
  const badge=card.querySelector('.photo-badge');
  badge.textContent=photo.engine==='demo'?'演示':photo.inspection_uncertain?'核对不确定':photo.inspection_tier===0?(photo.inspection_strict_conflict?'可能不符':'核对有分歧'):photo.inspection_tier===2?'较符合':'';
  badge.classList.toggle('hidden',!badge.textContent);
  card.querySelector('.rank-badge').textContent=String(i+1).padStart(2,'0');
  card.querySelector('h3').textContent=photo.name;
  card.querySelector('.card-meta span:first-child').textContent=photo.captured?photo.captured.slice(0,10).replaceAll('-','.'):'时间未知';
  card.querySelector('.card-meta span:last-child').textContent=photo.engine==='demo'?'演示插画':`${photo.width} × ${photo.height}`;
  const reason=card.querySelector('.card-reasons');
  reason.textContent=inspections[photo.id]?.inspection.summary || '';
  card.title=reason.textContent || (photo.reasons||[]).join(' · ');
  // Preserve the image elements; move only nodes whose order actually changed.
  if(gallery.children[i]!==card)gallery.insertBefore(card,gallery.children[i]||null);
 });
 visibleOrder=ordered.map(p=>p.id);
 $('#empty-state').classList.toggle('hidden',ordered.length>0);gallery.classList.toggle('hidden',ordered.length===0);
 if(!ordered.length)$('#empty-description').textContent=filtering?(otherView?'没有其他候选。':verification?.status==='running'?'正在筛选，核对较符合的照片会逐步出现。':groups.unreviewed.length?`暂未确认符合，仍有 ${groups.unreviewed.length} 张候选未核对。可以继续筛选或查看其他候选。`:'当前候选暂未确认匹配，可以查看其他候选或调整线索。'):currentLibrary.count?'暂时没有结果，试着移除一条线索。':'添加文件夹并建立索引后，照片会出现在这里。';
 $('#results-heading').textContent=filtering?(otherView?'其他候选':'较符合的照片'):session?.conditions.length?'搜索结果':'照片';
 $('#result-count').textContent=filtering?(otherView?`${ordered.length} 张 · 包含未核对、不确定和模型认为不符的照片`:`${ordered.length} 张 · 模型核对结果，仍可能误判`):`${ordered.length} 张照片`;
}
async function search(message='',mode='append',removeId=null){
 const requestLibrary=currentLibrary.id;const seq=++searchSequence;
 clearTimeout(verificationTimer);verification=null;$('#verification-panel').classList.add('hidden');$('#coverage-note').classList.add('hidden');$('#pending-control').classList.add('hidden');$('#show-pending').checked=false;
 $('#inspect-button').disabled=false;
 busy=true;$('#send-button').disabled=true;$('#composer-status').textContent='正在查找记忆里的照片…';
 try{
   const response=await api('/api/search',{library_id:requestLibrary,session_id:session?.id||savedSessions[requestLibrary]||null,message,mode,remove_id:removeId,limit:1000});
   if(seq!==searchSequence || currentLibrary.id!==requestLibrary)return;
   session=response.session;results=response.results;lastResponse=response;inspections={};results.forEach(p=>{if(p.inspection)inspections[p.id]={id:p.id,inspection:p.inspection,inspection_score:p.inspection_score,inspection_tier:p.inspection_tier};});savedSessions[requestLibrary]=session.id;localStorage.setItem('photorecall-sessions',JSON.stringify(savedSessions));
   verification=response.verification;$('#pending-control').classList.toggle('hidden',!verification?.enabled);
   $('#results-heading').textContent=session.conditions.length?'搜索结果':'照片';
   $('#result-count').textContent=`${response.total} 张${response.mode==='demo'?'演示插画':response.mode==='semantic'?'已检索':'候选'}${response.total>results.length?' · 展示前 '+results.length+' 张':''}`;
   $('#mode-note').innerHTML=icon('info')+`<span>${escapeHTML(response.mode==='metadata'?'画面搜索尚未启用，目前只搜索文件信息。':response.mode==='demo'?'这是演示照片库，可添加自己的文件夹开始搜索。':'')}</span>`;$('#mode-note').classList.toggle('hidden',response.mode==='semantic');
   $('#footer-note').textContent=response.mode==='demo'?'演示插画用于体验操作，AI 搜索效果需要用真实照片验证。':response.mode==='metadata'?'已建立文件索引；画面语义搜索尚未启用。':'主结果只保留核对较符合的照片；其他候选保留供复查，未核对不等于不符合。';
   renderConditions();renderMessages(response);renderGallery();
   if(verification?.enabled){renderVerification();if(verification.status==='running')verificationTimer=setTimeout(()=>pollVerification(seq,session.id),1200);}
 }catch(err){toast(err.message);}finally{if(seq===searchSequence){busy=false;$('#send-button').disabled=false;$('#composer-status').textContent='可以继续补充细节';}}
}
function renderCoverage(){
 const coverage=verification?.observation_coverage,total=lastResponse?.total||0,panel=$('#coverage-note');
 const incomplete=verification?.hybrid&&coverage&&total&&(coverage.caption<total||coverage.text_vectors<coverage.caption);
 panel.classList.toggle('hidden',!incomplete);
 if(incomplete)panel.textContent=`画面索引正在补齐：描述 ${coverage.caption}/${total} · 描述向量 ${coverage.text_vectors}/${total}。仍可检索全库图片；照片文字是可选辅助，不影响画面搜索。`;
}
function applyInspectionResponse(response){
 response.results.forEach(r=>{inspections[r.id]=r;const photo=results.find(p=>p.id===r.id);if(photo)Object.assign(photo,r);});
 const ranking=response.ranking;
 if(!ranking||ranking.key!==verification?.key||ranking.revision<(verification.revision||0))return;
 verification.revision=ranking.revision;
 const positions=new Map(ranking.ids.map((id,i)=>[id,i]));
 results.forEach(photo=>{if(positions.has(photo.id))photo.rank_order=positions.get(photo.id);});
}
function renderVerification(){
 renderCoverage();
 const panel=$('#verification-panel'),active=verification?.status==='running';
 const show=verification?.enabled&&(active||verification.status==='error'||verification.status==='paused');
 panel.classList.toggle('hidden',!show);
 $('#inspect-button').disabled=!!active;$('#expand-rerank').disabled=!!active;
 if(!show)return;
 const title=active?(verification.hybrid?(verification.stage==='recall'?'正在联合检索图片与文字…':`本地核对细节 ${verification.done} / ${verification.total}`):verification.stage==='recall'?'正在整理候选…':`${verification.first_preview_ready?'首批已排序 · ':'正在排序 · '}${verification.done} / ${verification.total}${verification.automatic_extra?' · 补查更多候选':''}`):verification.status==='error'?'排序未完成':'排序已暂停';
 panel.innerHTML=`<div class="job-top"><strong>${escapeHTML(title)}</strong>${active?'<button id="pause-verification">暂停</button>':'<button id="dismiss-verification" aria-label="关闭排序提示">×</button>'}</div>${verification.errors?.length?`<small>${escapeHTML(verification.errors[0])}</small>`:''}`;
 if(active)$('#pause-verification').onclick=async()=>{try{await api(`/api/sessions/${session.id}/verification/pause`,{});toast('当前图片完成后暂停');}catch(err){toast(err.message);}};
 else $('#dismiss-verification').onclick=()=>panel.classList.add('hidden');
}
async function pollVerification(seq,id){
 try{
  const updated=await api(`/api/sessions/${id}/verification`);if(seq!==searchSequence||session?.id!==id||updated.key!==verification?.key)return;
  if((updated.revision||0)<(verification.revision||0)){
   if(updated.status==='running'||verification.status==='running')verificationTimer=setTimeout(()=>pollVerification(seq,id),1200);
   return;
  }
  verification=updated;if(updated.pipeline==='vnext')results=updated.results;
  updated.results.forEach(r=>{const photo=results.find(p=>p.id===r.id);if(photo){Object.assign(photo,r);if(r.inspection)inspections[r.id]=r;}});
  const firstPreview=updated.first_preview_ready&&previewAppliedSequence!==seq;
  const applyPreview=firstPreview&&window.scrollY<=80&&!$('#photo-dialog').open;
  if(firstPreview)previewAppliedSequence=seq;
  renderGallery({live:!applyPreview&&(updated.status==='running'||window.scrollY>80||$('#photo-dialog').open)});renderVerification();if(updated.status==='running')verificationTimer=setTimeout(()=>pollVerification(seq,id),1200);
 }catch(err){if(seq===searchSequence){toast(err.message);$('#inspect-button').disabled=false;}}
}
function showPhoto(id){
 currentPhoto=results.find(x=>x.id===id);if(!currentPhoto)return;
 const p=currentPhoto,preview=$('#photo-preview'),source=`/api/assets/${p.id}/original`;
 preview.alt=p.name;
 if(preview.getAttribute('src')!==source){
  preview.classList.add('is-loading');$('#photo-loading').classList.remove('hidden');preview.setAttribute('aria-busy','true');
  preview.onload=()=>{
   if(currentPhoto?.id!==id)return;preview.classList.remove('is-loading');$('#photo-loading').classList.add('hidden');preview.setAttribute('aria-busy','false');
  };
  preview.onerror=()=>{
   if(currentPhoto?.id!==id)return;preview.classList.remove('is-loading');$('#photo-loading').classList.add('hidden');preview.setAttribute('aria-busy','false');toast('原图暂时无法载入，请检查硬盘连接。');
  };
  preview.src=source;
  if(preview.complete&&preview.naturalWidth)preview.onload();
 }
 $('#photo-name').textContent=p.name;$('#photo-kind').textContent=p.engine==='demo'?'演示插画 · 非 AI 检索结果':'本地照片';
 $('#photo-description').textContent=p.description || p.reasons.join(' · ');
 $('#photo-meta').innerHTML=`<div class="meta-row"><strong>拍摄时间</strong>${escapeHTML(p.captured?p.captured.replace('T',' '):'未知 · 不使用文件修改时间推测')}</div><div class="meta-row"><strong>原文件位置</strong>${escapeHTML(p.path)}</div>`;
 $('#inspection-detail').innerHTML=inspections[id]?inspections[id].inspection.conditions.map(c=>`<div class="inspection-item">${escapeHTML(c.term)} · ${{matched:'符合',contradicted:'可能不符',unknown:'无法判断'}[c.status]}${c.consistency==='inconsistent'?'（结论与依据矛盾，未据此调低排名）':''}<br>${escapeHTML(c.evidence)}</div>`).join(''):'';
 $('#original-link').href=`/api/assets/${p.id}/original`;$('#original-link').textContent=p.engine==='demo'?'打开插画 ↗':'打开原图 ↗';
 $('#inspect-photo').classList.toggle('hidden',p.engine==='demo');
 $('#reveal-button').classList.toggle('hidden',p.engine==='demo');
 const position=visibleOrder.indexOf(id);
 $('#photo-counter').textContent=position>=0?`${position+1} / ${visibleOrder.length}`:'';
 $('#photo-prev').disabled=position<=0;$('#photo-next').disabled=position<0||position>=visibleOrder.length-1;
 if(!$('#photo-dialog').open)$('#photo-dialog').showModal();
}
async function pollJob(){
 const id=currentLibrary.id;
 try{
  const job=await api(`/api/libraries/${id}/job`);if(currentLibrary.id!==id)return;
  if(job.status==='idle'){$('#job-panel').classList.add('hidden');return;}
  const active=['running','scanning'].includes(job.status);const panel=$('#job-panel');panel.classList.remove('hidden');
  const title={scanning:'正在清点照片',running:job.task==='retrieval'?'正在建立新版视觉索引':job.task==='hybrid'?'正在增强搜索信息':job.task==='analysis'?'正在分析画面':'正在建立索引',complete:job.task==='retrieval'?'新版视觉索引已更新':job.task==='hybrid'?'混合搜索信息已更新':job.task==='analysis'?'详细画面索引已更新':'索引已更新',paused:'已暂停，可继续',error:'索引遇到问题'}[job.status];
  panel.innerHTML=`<div class="job-top"><strong>${title} ${active&&job.total?`${job.done} / ${job.total} 项${job.photo_total?` · ${job.photo_total} 张照片`:""}`:''}</strong>${active?'<button id="pause-index">暂停</button>':'<button id="dismiss-index" aria-label="关闭索引提示">×</button>'}</div>${active?`<progress ${job.total?`value="${job.done}" max="${job.total}"`:''}></progress>`:''}<small>${escapeHTML(active?job.current:`处理 ${job.processed} ${job.task==='hybrid'?'项':'张'} · 复用 ${job.skipped} 项 · 移除失效记录 ${job.removed} 条`)}</small>${job.errors.length?`<details><summary>${job.errors.length} 个文件或步骤未完成</summary><ul>${job.errors.map(e=>`<li>${escapeHTML(e.file)} ${escapeHTML(e.message)}</li>`).join('')}</ul></details>`:''}`;
  if(!active)$('#dismiss-index').onclick=()=>panel.classList.add('hidden');
  if(active){$('#pause-index').onclick=async()=>{await api(`/api/libraries/${id}/pause`,{});toast('将在当前图片完成后暂停');};pollTimer=setTimeout(pollJob,1500);}
  else{state=await api('/api/state');if(currentLibrary.id!==id)return;currentLibrary=state.libraries.find(x=>x.id===id)||currentLibrary;renderLibraries();if(job.status==='complete'&&!job.errors.length){clearTimeout(jobHideTimer);jobHideTimer=setTimeout(()=>{if(currentLibrary.id===id)panel.classList.add('hidden');},3500);}if(!results.length&&!session?.conditions.length)await search();}
 }catch(err){toast(err.message);}
}
async function startIndex(){
 if(currentLibrary.is_demo)return;try{await api(`/api/libraries/${currentLibrary.id}/index`,{});pollJob();}catch(err){toast(err.message);}
}
async function refreshModels(){
 $('#model-status').innerHTML='<p class="field-hint">正在检查本地模型…</p>';
 try{const data=await api('/api/models');
 $('#model-status').innerHTML=`<div class="model-row"><span>${icon('image')}</span><div><strong>SigLIP 2 · 画面检索</strong><p>为图片建立可复用的视觉索引</p></div><span class="model-label ${data.siglip.enabled?'ready':''}">${data.siglip.enabled?'已启用':data.siglip.installed?'已准备 · 待启用':'尚未安装'}</span></div><div class="model-row"><span>${icon('spark')}</span><div><strong>Qwen3-VL · 候选核对</strong><p>指令版 · 本地逐项核对画面细节</p></div><span class="model-label ${data.ollama.ready?'ready':''}">${data.ollama.ready?'已准备':'尚未准备'}</span></div>`;
 $('#model-status').insertAdjacentHTML('beforeend',`<div class="model-row"><span>${icon('image')}</span><div><strong>本地目标检测 · 数量筛选</strong><p>检测可见目标位置，补充人数与物体数量依据</p></div><span class="model-label ${data.detector?.ready?'ready':''}">${data.detector?.ready?'已准备':'尚未准备'}</span></div>`);
 if(data.retrieval)for(const [key,label] of [['main','WeMM · 主检索'],['reranker','CORE · 多条件精排']]){
  const ready=data.retrieval[key].installed;$('#model-status').insertAdjacentHTML('beforeend',`<div class="model-row"><span>${icon('spark')}</span><div><strong>${label}</strong><p>分阶段加载 · 照片在本机处理</p></div><span class="model-label ${ready?'ready':''}">${ready?'已准备':'下载尚未完成'}</span></div>`);
 }
 if(data.hybrid){const h=data.hybrid;$('#model-status').insertAdjacentHTML('beforeend',`<div class="model-row"><div><strong>${escapeHTML(h.ocr.selected)} · 照片文字</strong><p>${escapeHTML(h.ocr.last_error||'离线识别，保存文字位置与来源')}</p></div><span class="model-label ready">本地</span></div><div class="model-row"><div><strong>${escapeHTML(h.description_model)}</strong><p>离线画面描述 · ${escapeHTML(h.text_embedding)} 文字检索</p></div><span class="model-label ${h.text_embedding_ready?'ready':''}">${h.text_embedding_ready?'已准备':'下载中'}</span></div>`);}
 $('#enable-semantic').disabled=!data.siglip.installed;$('#enable-semantic').textContent=data.siglip.enabled?'关闭画面搜索':'启用画面搜索';$('#enable-semantic').dataset.enabled=String(data.siglip.enabled);
 }catch(err){toast(err.message);}
}
hydrateIcons();
['#folder-button','#add-library'].forEach(s=>$(s).onclick=()=>$('#folder-dialog').showModal());
document.querySelectorAll('.dialog-close').forEach(b=>b.onclick=()=>b.closest('dialog').close());
document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
$('#settings-button').onclick=()=>{$('#settings-dialog').showModal();refreshModels();};$('#refresh-models').onclick=refreshModels;
$('#mobile-settings').onclick=$('#settings-button').onclick;$('#mobile-library').onchange=e=>selectLibrary(e.target.value);
$('#enable-semantic').onclick=async()=>{const enabled=$('#enable-semantic').dataset.enabled==='true';$('#enable-semantic').disabled=true;try{await api('/api/settings',{semantic:!enabled});toast(enabled?'已关闭画面搜索':'已启用；请更新真实照片库索引');await refreshModels();}catch(err){toast(err.message);$('#enable-semantic').disabled=false;}};
$('#pick-folder').onclick=async()=>{$('#pick-folder').disabled=true;try{const response=await api('/api/pick-folder',{});if(response.path)$('#folder-path').value=response.path;}catch(err){toast(err.message);}finally{$('#pick-folder').disabled=false;}};
$('#folder-form').onsubmit=async e=>{e.preventDefault();$('#confirm-folder').disabled=true;try{const lib=await api('/api/libraries',{path:$('#folder-path').value.trim()});state=await api('/api/state');$('#folder-dialog').close();await selectLibrary(lib.id);if($('#start-on-add').checked)await startIndex();toast('照片库已添加');}catch(err){toast(err.message);}finally{$('#confirm-folder').disabled=false;}};
$('#search-form').onsubmit=async e=>{e.preventDefault();const text=$('#query').value.trim();if(!text||busy)return;const seq=searchSequence+1;await search(text);if(searchSequence===seq&&session?.messages.at(-1)?.text===text&&$('#query').value.trim()===text)$('#query').value='';};
$('#query').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();$('#search-form').requestSubmit();}};
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>search(b.dataset.prompt,'replace'));
$('#reset-search').onclick=()=>search('','reset');$('#empty-reset').onclick=()=>search('','reset');$('#sort').onchange=()=>renderGallery();$('#apply-ranking').onclick=()=>renderGallery();$('#update-index').onclick=startIndex;
$('#show-pending').onchange=()=>renderGallery();
$('#continue-filter').onclick=()=>{if(!$('#inspect-button').disabled&&!manualInspectionRunning)$('#inspect-button').click();};
$('#analysis-button').onclick=async()=>{if(currentLibrary.is_demo)return;try{await api(`/api/libraries/${currentLibrary.id}/hybrid-index`,{});toast('先补齐画面描述与向量，再补充照片文字。可暂停，已完成的会保留。');pollJob();}catch(err){toast(err.message);}};
$('#expand-rerank').onclick=async()=>{if(!session?.conditions.length||verification?.pipeline!=='vnext'){toast('先描述想找的照片');return;}try{verification=await api(`/api/sessions/${session.id}/rerank`,{budget:Math.min(256,Math.max(128,(verification.total||64)*2))});renderVerification();verificationTimer=setTimeout(()=>pollVerification(searchSequence,session.id),1200);}catch(err){toast(err.message);}};
$('#visual-index-button').onclick=async()=>{if(currentLibrary.is_demo)return;try{await api(`/api/libraries/${currentLibrary.id}/retrieval-index`,{});$('#settings-dialog').close();pollJob();}catch(err){toast(err.message);}};
$('#cancel-inspection').onclick=()=>{cancelInspection=true;toast('当前照片核对完成后停止');};
$('#reveal-button').onclick=async()=>{try{await api(`/api/assets/${currentPhoto.id}/reveal`,{});}catch(err){toast(err.message);}};
$('#inspect-photo').onclick=async()=>{
 if(!session?.conditions.length){toast('先描述想找的照片');return;}
 const id=currentPhoto.id,sid=session.id,seq=searchSequence,button=$('#inspect-photo');button.disabled=true;button.textContent='正在核对…';
 try{
  if(verification?.status==='running')await api(`/api/sessions/${sid}/verification/pause`,{});
  const response=await api('/api/inspect',{session_id:sid,asset_ids:[id]});
  if(seq!==searchSequence||sid!==session?.id)return;
  applyInspectionResponse(response);
  renderGallery();if(currentPhoto?.id===id){const p=inspections[id].inspection;$('#inspection-detail').innerHTML=p.conditions.map(c=>`<div class="inspection-item">${escapeHTML(c.term)} · ${{matched:'符合',contradicted:'不符合',unknown:'无法判断'}[c.status]}<br>${escapeHTML(c.evidence)}</div>`).join('');}
 }catch(err){toast(err.message);}finally{button.disabled=false;button.textContent='核对这张的条件';}
};
$('#inspect-button').onclick=async()=>{
 if(manualInspectionRunning)return;
 if(!session?.conditions.length){toast('先描述想找的照片，再核对候选');return;}
 if(currentLibrary.is_demo){toast('演示库用于体验流程；请添加真实图片后进行 AI 核对');return;}
 if(!results.length){toast('没有候选照片可以核对');return;}
 let models;try{models=await api('/api/models');}catch(err){toast(err.message);return;}
 if(!models.ollama.ready){$('#settings-dialog').showModal();refreshModels();return;}
 if(manualInspectionRunning)return;
 manualInspectionRunning=true;$('#continue-filter').disabled=true;
 const inspectLibrary=currentLibrary.id, inspectSession=session.id, inspectSeq=searchSequence;
 cancelInspection=false;$('#cancel-inspection').classList.remove('hidden');
 $('#inspect-button').disabled=true;$('#inspect-button').innerHTML=icon('spark')+'<span>正在逐张核对，请稍候…</span>';
 try{
  // One image per request gives visible progress and bounded memory.
  const subset=PhotoMatches.candidateBatch(results,visibleOrder,$('#inspect-count').value);
  if(!subset.length){toast('当前候选已全部核对；不确定的照片可打开后单独复查');return;}
  for(let i=0;i<subset.length;i++){
   if(cancelInspection||currentLibrary.id!==inspectLibrary||session?.id!==inspectSession||searchSequence!==inspectSeq)break;
   $('#inspect-button').innerHTML=icon('spark')+`<span>正在核对 ${i+1} / ${subset.length}…</span>`;
   const response=await api('/api/inspect',{session_id:inspectSession,asset_ids:[subset[i].id]});
   if(currentLibrary.id!==inspectLibrary||session?.id!==inspectSession||searchSequence!==inspectSeq)break;
   applyInspectionResponse(response);renderGallery({live:true});
  }
  if(searchSequence===inspectSeq){renderGallery();toast('本批筛选完成，较符合的照片已加入主结果；其余可在其他候选中查看。');}
 }catch(err){toast(err.message);}finally{manualInspectionRunning=false;$('#inspect-button').disabled=false;$('#cancel-inspection').classList.add('hidden');$('#inspect-button').innerHTML=icon('spark')+'<span>核对候选细节</span><span>↗</span>';renderGallery();}
};
function applyTheme(theme){
 const value=theme==='dark'?'dark':'light';document.documentElement.dataset.theme=value;
 try{localStorage.setItem('photorecall-theme',value);}catch{}
 const label=value==='dark'?'切换浅色外观':'切换深色外观';
 $('#theme-toggle').innerHTML=icon(value==='dark'?'sun':'moon');
 $('#theme-toggle').setAttribute('aria-label',label);$('#theme-toggle').title=label;
}
function applyDensity(density){
 const value=density==='compact'?'compact':'comfortable';document.documentElement.dataset.density=value;
 try{localStorage.setItem('photorecall-density',value);}catch{}
 $('#density-compact').setAttribute('aria-pressed',String(value==='compact'));
 $('#density-comfortable').setAttribute('aria-pressed',String(value==='comfortable'));
}
function navigatePhoto(direction){
 const i=visibleOrder.indexOf(currentPhoto?.id),next=visibleOrder[i+direction];
 if(i>=0&&next)showPhoto(next);
}
$('#theme-toggle').onclick=()=>applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');
$('#density-compact').onclick=()=>applyDensity('compact');
$('#density-comfortable').onclick=()=>applyDensity('comfortable');
$('#photo-prev').onclick=()=>navigatePhoto(-1);$('#photo-next').onclick=()=>navigatePhoto(1);
const macPlatform=/Mac|iPhone|iPad/.test(navigator.platform||'');
$('.search-shortcut').textContent=macPlatform?'⌘ K':'Ctrl K';
document.addEventListener('keydown',event=>{
 if(event.isComposing||event.altKey)return;
 if((event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'&&!document.querySelector('dialog[open]')){event.preventDefault();$('#query').focus();return;}
 if($('#photo-dialog').open&&!event.metaKey&&!event.ctrlKey&&!/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)){
  if(event.key==='ArrowLeft'){event.preventDefault();navigatePhoto(-1);}
  if(event.key==='ArrowRight'){event.preventDefault();navigatePhoto(1);}
 }
});
applyTheme(document.documentElement.dataset.theme);applyDensity(document.documentElement.dataset.density);
async function init(){try{state=await api('/api/state');await selectLibrary(localStorage.getItem('photorecall-library')||'demo');}catch(err){$('#mode-note').textContent='本地服务连接失败，请重新启动项目。';toast(err.message);}}
init();
