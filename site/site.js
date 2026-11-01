const $ = id => document.getElementById(id);
const cache = new Map();
let manifest, selected, payload, viewerReady = false, requestVersion = 0;
const number = (value, digits=3) => value !== '' && value !== null && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : 'Unavailable';
const status = (message, error=false) => { $('load-status').textContent = message; $('load-status').classList.toggle('error', error); };
const link = (label, file) => { const a = document.createElement('a'); a.textContent=label; a.href=file; a.download=''; return a; };
function showView() {
  const name = location.hash.slice(1) || 'inspection';
  const view = ['overview','inspection','comparison','report'].includes(name) ? name : 'inspection';
  for (const el of document.querySelectorAll('.view')) el.hidden = el.id !== view;
  for (const a of document.querySelectorAll('[data-view]')) {
    if (a.dataset.view === view) a.setAttribute('aria-current','page'); else a.removeAttribute('aria-current');
  }
  if (view === 'report' && !$('report-frame').src) $('report-frame').src=$('report-frame').dataset.src;
  // Resizing a hidden WebGL canvas on re-entry is display-only.
  if (view === 'inspection' && viewerReady) $('viewer').contentWindow.dispatchEvent(new Event('resize'));
}
function deliver() {
  if (viewerReady && payload) $('viewer').contentWindow.postMessage({type:'streamlit:render',args:{data:payload}},location.origin);
}
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.source !== $('viewer').contentWindow) return;
  if (event.data?.type === 'streamlit:componentReady') { viewerReady=true; deliver(); }
  if (event.data?.type === 'streamlit:setFrameHeight' && Number.isFinite(event.data.height)) {
    $('viewer').height=String(Math.max(1000,Math.min(6000,event.data.height+20)));
  }
});
$('viewer').addEventListener('load',()=>{viewerReady=true;deliver();});
async function load(entry) {
  if (!cache.has(entry.participant_id)) {
    const promise = (async () => {
      const response = await fetch(`${entry.file}?v=${entry.sha256.slice(0,12)}`);
      if (!response.ok) throw new Error(`Saved participant file returned HTTP ${response.status}`);
      const buffer = await response.arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(b=>b.toString(16).padStart(2,'0')).join('');
      if (hash !== entry.sha256) throw new Error('Saved participant checksum mismatch; reload after the deployment completes.');
      const data = JSON.parse(new TextDecoder().decode(buffer));
      if (data.participant_id !== entry.participant_id) throw new Error('Saved participant identity mismatch');
      return data;
    })();
    cache.set(entry.participant_id,promise);
    promise.catch(()=>cache.delete(entry.participant_id));
  }
  return cache.get(entry.participant_id);
}
function metric(label,value) {
  const el=document.createElement('div');el.className='metric';
  const strong=document.createElement('strong');strong.textContent=value;
  const span=document.createElement('span');span.textContent=label;el.append(strong,span);return el;
}
function renderComparison(entry,data) {
  const container=$('comparison-content');container.replaceChildren();
  const heading=document.createElement('h3');heading.textContent=entry.participant_id;container.append(heading);
  const retained=document.createElement('div');retained.className='metrics';
  for (const r of data.primary_analysis) retained.append(metric(`${r.condition === 'baseline' ? 'Baseline' : 'FD-censored'} retained`,`${r.retained_volumes} vol · ${number(r.retained_minutes,2)} min`));
  retained.append(metric('Mean FD · 167 observed values',`${number(data.motion.mean_fd_mm)} mm`),metric('Additional FD > 0.5 mm frames',String(data.motion.high_fd_count)));
  container.append(retained);
  if (!entry.comparison) {
    const notice=document.createElement('div');notice.className='unavailable';
    const title=document.createElement('h3');title.textContent='Correlation matrices unavailable';
    const reason=document.createElement('p');reason.textContent=data.primary_analysis.find(r=>r.status!=='complete')?.reason || 'See saved methods and primary status for the reason.';
    const detail=document.createElement('p');detail.textContent='No substitute matrix is displayed. Inventory, motion, retention, original-frame exclusions and coverage remain inspectable.';
    notice.append(title,reason,detail);container.append(notice);
  } else {
    const measures=document.createElement('div');measures.className='metrics';
    const c=entry.comparison;
    for (const [key,label] of [['mean_absolute_change','Mean absolute Δr'],['max_absolute_change','Maximum absolute Δr'],['mean_signed_change','Mean signed Δr'],['edge_similarity','Edge-vector similarity']]) measures.append(metric(label,number(c[key],4)));
    container.append(measures);
    const note=document.createElement('p');note.className='note';note.textContent=`Changes use ${c.n_unique_edges} unique off-diagonal edges; difference = censored − baseline. Numerical diagonals are preserved in matrix CSVs.`;container.append(note);
    const figure=document.createElement('figure');figure.className='matrix-figure';
    const img=document.createElement('img');img.src=entry.exports['matrices.png'].file;img.alt=`${entry.participant_id}: baseline, FD-censored and difference correlation matrices in identical atlas order`;
    const caption=document.createElement('figcaption');caption.textContent='Original saved analysis figure; no correlations are recalculated in the browser.';figure.append(img,caption);container.append(figure);
  }
  const exports=document.createElement('div');exports.className='download-links';
  for (const name of ['correlation_baseline.csv','correlation_censored.csv','frames.csv','regions.csv','methods.json']) if (entry.exports[name]) exports.append(link(name,entry.exports[name].file));
  container.append(exports);
}
async function select(pid) {
  const entry=manifest.participants.find(p=>p.participant_id===pid);
  if (!entry) return;
  const version=++requestVersion;
  $('participant').value=pid;
  status(`Loading ${pid} · saved real-data display…`);
  // Do not leave an old participant visible while the selector says a new one.
  $('viewer').style.visibility='hidden';
  $('comparison-content').replaceChildren();
  payload=null;
  try {
    const data=await load(entry);
    if (version!==requestVersion) return;
    selected=pid;payload=data;deliver();
    $('viewer').style.visibility='visible';
    renderComparison(entry,data);
    for (const b of document.querySelectorAll('[data-participant]')) b.setAttribute('aria-pressed',String(b.dataset.participant===pid));
    status(`${pid} · verified saved artifacts · ${data.n_volumes} frames × ${data.tr_seconds} s · ${data.primary_status === 'complete' ? 'paired matrices available' : 'matrices unavailable; reason preserved'}`);
  } catch(error) {
    if(version===requestVersion) status(`${pid} could not be displayed: ${error.message} Existing artifacts are preserved. Retry by selecting the participant again.`,true);
  }
}
function sampleRows() {
  for (const entry of manifest.participants) {
    const s=entry.summary,tr=document.createElement('tr'),cell=document.createElement('td'),button=document.createElement('button');
    button.textContent=entry.participant_id;button.dataset.participant=entry.participant_id;button.dataset.caseView='inspection';cell.append(button);tr.append(cell);
    for (const value of [s.processing_status==='complete'?'Matrices available':'Matrices unavailable',Number(s.baseline_retained_volumes),Number(s.censored_retained_volumes),number(s.mean_fd_mm),`${s.n_fd_observed} / ${s.n_volumes}`]) { const td=document.createElement('td');td.textContent=value;tr.append(td); }
    $('sample-rows').append(tr);
  }
}
document.addEventListener('click',event=>{
  const button=event.target.closest('[data-participant]');
  if(button && manifest) {select(button.dataset.participant);if(button.dataset.caseView)location.hash=button.dataset.caseView;}
});
$('participant').addEventListener('change',()=>select($('participant').value));
window.addEventListener('hashchange',showView);showView();
try {
  const response=await fetch('data/manifest.json');
  if(!response.ok)throw new Error(`Manifest returned HTTP ${response.status}`);
  manifest=await response.json();
  if(manifest.data_kind!=='saved_real_results_demo'||manifest.participants.length!==8)throw new Error('Unsupported saved sample');
  $('participant').replaceChildren(...manifest.participants.map(p=>new Option(p.participant_id,p.participant_id)));
  $('participant').disabled=false;sampleRows();await select('sub-pixar123');
} catch(error) { status(`The saved demo could not load: ${error.message}. The static report and repository remain available.`,true); }
