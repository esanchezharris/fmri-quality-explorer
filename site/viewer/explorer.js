// One client-side state links spatial inspection, source observations and saved exclusions.
const $ = id => document.getElementById(id);
const standalone = Boolean($('coverage-data'));
const moduleFromNode = async id => {
  const url = URL.createObjectURL(new Blob([JSON.parse($(id).textContent)], {type:'text/javascript'}));
  try { return await import(url); } finally { URL.revokeObjectURL(url); }
};
const model = standalone ? await moduleFromNode('explorer-model') : await import('./explorer-model.mjs');
const {signalRange, linePath, nearestFrame, coverageLabel} = model;
const state = {data:null, region:14, source:'tutorial', side:'right', frame:0, mode:'slices', plane:'coronal', expanded:false, timer:null, load:0, loading:true};
let nv = null, library = null, viewerPromise = null, current = null, sourceCache = new Map(), lastLocation = null;
const number = (v, digits=2) => Number.isFinite(v) ? v.toFixed(digits) : 'Unavailable';
const region = () => state.data.regions.find(r => r.id === state.region);
const source = () => state.data.sources[state.source];
const observations = () => region()[state.source];
const text = (id, value) => { $(id).textContent = value; };

function sendHeight() {
  if (!standalone) window.parent.postMessage({isStreamlitMessage:true, type:'streamlit:setFrameHeight', height:Math.ceil(document.querySelector('.explorer').getBoundingClientRect().height)}, '*');
}

function setPressed(id, active) { $(id).setAttribute('aria-pressed', String(active)); }

function renderSupport() {
  const src = source(), support = observations().support;
  text('support-title', src.support_label);
  text('support-source', state.source === 'tutorial' ? 'Tutorial' : 'SPM8 archive');
  text('support-definition', src.denominator + ', stratified by supplied MNI x.');
  const container = $('side-bars');
  container.replaceChildren();
  for (const side of ['left','right']) {
    const count = support[side], row = document.createElement('div');
    row.className = 'side-stat';
    const heading = document.createElement('div'); heading.className = 'side-stat-heading';
    const name = document.createElement('span'); name.textContent = side === 'left' ? 'Left' : 'Right';
    const value = document.createElement('strong'); value.textContent = coverageLabel(count);
    value.dataset.side = side;
    heading.append(name, value);
    const track = document.createElement('div'); track.className = 'support-track';
    const fill = document.createElement('div'); fill.className = 'support-fill';
    fill.style.width = `${(count.lower || 0) * 100}%`;
    track.append(fill);
    if (count.unknown && count.total) {
      const missing = document.createElement('div'); missing.className = 'support-unknown';
      missing.style.left = `${count.lower * 100}%`; missing.style.width = `${(count.upper-count.lower) * 100}%`;
      track.append(missing);
    }
    const detail = document.createElement('span'); detail.className = 'support-count';
    detail.textContent = count.total ? `${count.included} of ${count.total} ${state.source === 'tutorial' ? 'voxel centers vary' : 'atlas centers included'}${count.unknown ? `; ${count.unknown} unknown` : ''}` : 'No parcel centers in this stratum';
    row.append(heading, track, detail); container.append(row);
  }
  const whole = support.whole;
  text('whole-support', `${coverageLabel(whole)} of the whole parcel: ${whole.included} of ${whole.total} centers${whole.unknown ? `; ${whole.unknown} unknown` : ''}.${support.midline.total ? ` Midline: ${support.midline.included}/${support.midline.total}.` : ''}`);
  text('support-note', src.note);
}

function renderResult() {
  const unavailable = state.data.primary_analysis.find(r => r.status !== 'complete');
  const status = $('result-status'); status.replaceChildren();
  const dot = document.createElement('i'); dot.className = 'status-dot'; status.append(dot);
  status.append(document.createTextNode(unavailable ? 'Matrix unavailable' : 'Matrices available'));
  status.classList.toggle('unavailable', Boolean(unavailable));
  if (unavailable) {
    const constants = state.data.regions.filter(r => r.tutorial.constant);
    text('result-reason', constants.length ? `The tutorial signal is constant in ${constants.map(r=>`region ${r.id}`).join(', ')}. Correlations were withheld. These are saved primary results.` : unavailable.reason || 'See the saved primary report for the specific failure.');
  } else {
    text('result-reason', 'Both saved tutorial-analysis conditions produced a matrix. Inspect spatial support separately.');
  }
  const pairs = $('retention-pairs'); pairs.replaceChildren();
  for (const row of state.data.primary_analysis) {
    const column = document.createElement('div'), amount = document.createElement('strong'), label = document.createElement('span');
    amount.textContent = `${row.retained_volumes} volumes`;
    label.textContent = `${row.condition === 'baseline' ? 'Baseline' : 'FD-censored'}: ${number(row.retained_minutes)} min retained`;
    column.append(amount, label); pairs.append(column);
  }
}

function renderDetails() {
  const src = source(), container = $('source-details'); container.replaceChildren();
  const facts = [
    ['Image', src.source_filename], ['Preprocessing', src.preprocessing],
    ['Support measure', `${src.support_label}. Denominator: ${src.denominator}.`],
    ['Atlas', `${state.data.atlas.name}, unsplit region ${state.region}: ${region().name}`],
    ['Source SHA-256', src.source_sha256],
    ['Matching mask source', src.mask_source_filename || 'Unavailable for this tutorial derivative'],
    ['Original frames', `0–${state.data.n_volumes-1}; TR ${state.data.tr_seconds} s; all frames used for the displayed mean and raw series.`],
    ['Display assets', `Generated ${state.data.generated_at_utc}. Intensities are source units and are not comparable across lineages.`],
  ];
  if(src.mask_source_sha256) facts.push(['Mask SHA-256',src.mask_source_sha256]);
  if (src.unknown_voxels) facts.push(['Undefined membership', `${src.unknown_voxels} source-mask locations remain unknown; selected parcel has ${observations().support.whole.unknown}.`]);
  for (const [label, value] of facts) {
    const line = document.createElement('div'); line.className = 'source-fact';
    const strong = document.createElement('strong'); strong.textContent = label + ': ';
    const content = document.createElement(label.includes('SHA') ? 'code' : 'span'); content.textContent = value;
    line.append(strong, content); container.append(line);
  }
}

function renderInfo() {
  if (!state.data) return;
  text('participant-name', state.data.participant_id);
  text('acquisition-label', `${state.data.n_volumes} frames, ${state.data.tr_seconds}-s TR`);
  $('region-select').value = String(state.region);
  setPressed('source-tutorial', state.source === 'tutorial'); setPressed('source-archive', state.source === 'archive');
  text('image-context', `Full-run mean, ${source().title}, ${source().grid_mm[0]}-mm grid`);
  text('resolution-note', state.source==='tutorial'
    ? 'Source: 4-mm voxels. Enlarging or smoothing the display cannot recover finer anatomical detail.'
    : 'Source: 2-mm normalized grid, already 5-mm smoothed. Display controls cannot recover finer anatomical detail.');
  text('overlay-label', state.source === 'tutorial' ? 'Varying centers' : 'Known mask');
  text('signal-description', 'Uncleaned source intensity; all acquired frames');
  const constant = observations().constant;
  text('signal-state', observations().series === null ? 'Unavailable' : constant ? 'Constant signal' : 'Source observations');
  $('signal-state').classList.toggle('constant', Boolean(constant));
  text('fd-denominator', `${state.data.motion.n_fd_observed}/${state.data.n_volumes} defined values`);
  renderSupport(); renderResult(); renderDetails(); drawCharts();
  requestAnimationFrame(sendHeight);
}

function chart(svg, values, limits, isFD) {
  const W = Math.max(280, svg.getBoundingClientRect().width), H = isFD ? 110 : 140;
  const left=44, right=14, top=12, bottom=24, times=state.data.motion.time_seconds;
  const end=times[times.length-1], plotW=W-left-right, plotH=H-top-bottom;
  const x = t => left + t / end * plotW;
  const y = value => H-bottom - (value-limits[0])/(limits[1]-limits[0])*plotH;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let content = '';
  const m=state.data.motion, half=state.data.tr_seconds/2;
  for (let i=0;i<times.length;i++) {
    const fill = !m.baseline_keep[i] ? '#e4ebf0' : !m.censored_keep[i] ? '#f8e2b7' : null;
    if (fill) {
      const start=Math.max(0,times[i]-half), stop=Math.min(end,times[i]+half);
      content += `<rect x="${x(start)}" y="${top}" width="${x(stop)-x(start)}" height="${plotH}" fill="${fill}"/>`;
    }
  }
  for(let i=0;i<3;i++) {
    const value=limits[0]+i*(limits[1]-limits[0])/2;
    content+=`<line class="grid" x1="${left}" x2="${W-right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left-7}" y="${y(value)+3}" text-anchor="end">${value.toFixed(2)}</text>`;
  }
  for(let i=0;i<5;i++) {
    const t=i*end/4;
    content+=`<text x="${x(t)}" y="${H-6}" text-anchor="middle">${t.toFixed(0)}</text>`;
  }
  if (isFD) content+=`<line class="threshold" x1="${left}" x2="${W-right}" y1="${y(.5)}" y2="${y(.5)}"/><text x="${W-right-4}" y="${y(.5)-4}" text-anchor="end">0.5 mm</text>`;
  content+=`<path class="series-line ${isFD?'fd-line':''}" d="${linePath(times, values, x, y)}"/>`;
  const selected=state.frame, cursor=x(times[selected]);
  content+=`<line class="cursor-line" x1="${cursor}" x2="${cursor}" y1="${top}" y2="${H-bottom}"/>`;
  if (values[selected] !== null && Number.isFinite(values[selected])) content+=`<circle class="cursor-dot ${isFD?'fd-dot':''}" cx="${cursor}" cy="${y(values[selected])}" r="3.5"/>`;
  svg.innerHTML=content;
  svg.dataset.start=left; svg.dataset.plotWidth=plotW; svg.dataset.width=W;
}

function drawCharts() {
  if (!state.data) return;
  const series=observations().series, range=signalRange(series);
  if (range) chart($('signal-chart'), series, range, false);
  else $('signal-chart').innerHTML='<text x="30" y="60">No source observations for this parcel on this grid.</text>';
  const finite=state.data.motion.fd.filter(v=>v!==null);
  const ymax=Math.max(.65, Math.max(...finite)*1.15);
  chart($('motion-chart'), state.data.motion.fd, [0,ymax], true);
  const frame=state.frame, m=state.data.motion;
  $('frame-range').value=String(frame);
  text('frame-readout', `Frame ${frame} / ${state.data.n_volumes-1} · ${m.time_seconds[frame]} s`);
  text('fd-readout', m.fd[frame] === null ? 'FD undefined at initial frame' : `FD ${m.fd[frame].toFixed(3)} mm`);
  for (const condition of ['baseline','censored']) {
    const keep=m[condition+'_keep'][frame], name=condition==='baseline'?'Baseline':'FD condition';
    text(condition+'-state', `${name}: ${keep?'retained':'excluded'}`);
    $(condition+'-state').className=keep?'retained':'excluded';
    $(condition+'-state').title=m[condition+'_reason'][frame] || 'No exclusion';
  }
}

function setFrame(frame) {
  state.frame=Math.max(0,Math.min(state.data.n_volumes-1,Number(frame))); drawCharts();
}

function selectRegion(id, shouldFocus=true) {
  const value=Number(id);
  if (!state.data.regions.some(r=>r.id===value)) return;
  state.region=value; renderInfo(); updateRegion();
  if (shouldFocus) focusRegion(state.side);
}

function updateRegion() {
  if (!nv || !current || state.loading) return;
  const labels=current.atlas.img, highlight=current.roi.img;
  for(let i=0;i<labels.length;i++) highlight[i]=labels[i]===state.region?1:0;
  current.roi.cal_min=.5; current.roi.cal_max=1; current.roi.opacity=$('show-region').checked ? 0.6 : 0;
  current.support.opacity=$('show-support').checked ? 0.22 : 0;
  current.unknown.opacity=$('show-support').checked ? 0.6 : 0;
  nv.updateGLVolume();
  applyDisplaySampling();
}

function applyDisplaySampling() {
  if(!nv || !current || state.loading) return;
  if(state.mode==='3d') return;
  // Keep the combined categorical overlay nearest-neighbor; interpolate only mean BOLD.
  nv.updateInterpolation(0,$('smooth-image').checked);
  nv.updateInterpolation(1,false);
  nv.drawScene();
}

function focusRegion(side) {
  state.side=side;
  setPressed('focus-left',side==='left'); setPressed('focus-right',side==='right');
  const points=region().focus[state.source];
  $('focus-left').disabled=!points.left; $('focus-right').disabled=!points.right;
  if(!nv || !current || state.loading) return;
  const mm=points[side] || points.left || points.right || points.midline;
  if(!mm) return;
  nv.scene.crosshairPos=nv.mm2frac(mm);
  nv.drawScene();
  text('crosshair-location', `Supplied MNI coordinates: ${mm.map(v=>v.toFixed(0)).join(', ')} mm`);
}

function viewerFailure(error) {
  state.loading=false; document.body.dataset.viewerStatus='unavailable';
  $('coverage-canvas').style.visibility='hidden';
  const message=$('viewer-message'); message.hidden=false; message.classList.add('error');
  message.textContent=`Brain viewer unavailable: ${error.message || error}. Saved counts and charts remain available. Use a browser with WebGL2 support.`;
  requestAnimationFrame(sendHeight);
}

async function ensureViewer() {
  if(viewerPromise) return viewerPromise;
  if(nv) return;
  viewerPromise=initializeViewer();
  try { await viewerPromise; } catch(error) { viewerPromise=null;nv=null;throw error; }
}

async function initializeViewer() {
  const probe=document.createElement('canvas');
  if(!probe.getContext('webgl2')) throw new Error('WebGL2 could not start');
  library=standalone?await moduleFromNode('niivue-library'):await import('./vendor/niivue.esm.js');
  nv=new library.Niivue({backColor:[16/255,43/255,59/255,1], crosshairColor:[.93,.77,.42,.85],
    fontColor:[.85,.93,.97,1], textHeight:.035, isColorbar:false, isOrientCube:true,
    isNearestInterpolation:true, isResizeCanvas:true, dragAndDropEnabled:false, drawingEnabled:false,
    multiplanarEqualSize:true, multiplanarShowRender:library.SHOW_RENDER.NEVER,
    crosshairWidth:.35, yoke3Dto2DZoom:true, logLevel:'error'});
  await nv.attachToCanvas($('coverage-canvas'));
  nv.addColormap('explorerAmber',{R:[220,220],G:[156,156],B:[52,52],A:[0,255],I:[0,255]});
  nv.addColormap('explorerCyan',{R:[53,53],G:[182,182],B:[196,196],A:[0,255],I:[0,255]});
  nv.addColormap('explorerUnknown',{R:[213,213],G:[118,118],B:[155,155],A:[0,255],I:[0,255]});
  nv.setMultiplanarLayout(library.MULTIPLANAR_TYPE.GRID);
  nv.onLocationChange=location=>{
    lastLocation=location;
    if(state.loading) return;
    const mm=location?.mm || nv.frac2mm(nv.scene.crosshairPos);
    if(mm && mm.length>=3) text('crosshair-location', `Supplied MNI coordinates: ${Array.from(mm).slice(0,3).map(v=>v.toFixed(1)).join(', ')} mm`);
  };
  // Contrast interaction refreshes textures after this callback; restore sampling afterward.
  nv.onIntensityChange=()=>queueMicrotask(applyDisplaySampling);
  nv.setRenderAzimuthElevation(120,15);
}

async function switchSource(name) {
  if(!state.data.sources[name]) return;
  const data=state.data, sourceImages=sourceCache;
  state.source=name; state.loading=true; current=null; renderInfo();
  const sequence=++state.load, message=$('viewer-message');
  message.hidden=false; message.classList.remove('error'); message.textContent=`Loading ${source().title.toLowerCase()}…`;
  $('coverage-canvas').style.visibility='hidden'; document.body.dataset.viewerStatus='loading';
  try {
    await ensureViewer();
    let cached=sourceImages.get(name);
    if(!cached) {
      const loaded={};
      for (const key of ['mean','atlas','support','unknown']) {
        loaded[key]=await library.NVImage.loadFromBase64({base64:data.sources[name].volumes[key],name:`${name}_${key}.nii.gz`,colorbarVisible:false});
      }
      loaded.roi=loaded.atlas.clone(); loaded.roi.name='Selected atlas region';
      loaded.mean.cal_min=data.sources[name].window[0]; loaded.mean.cal_max=data.sources[name].window[1]; loaded.mean.setColormap('gray');
      loaded.atlas.opacity=0;
      for(const [key,cmap] of [['support','explorerCyan'],['roi','explorerAmber'],['unknown','explorerUnknown']]) {
        loaded[key].setColormap(cmap); loaded[key].cal_min=.5; loaded[key].cal_max=1; loaded[key].opacity=0;
      }
      sourceImages.set(name,loaded); cached=loaded;
    }
    if(sequence!==state.load) return;
    for(const volume of [...nv.volumes]) nv.removeVolume(volume);
    for(const key of ['mean','atlas','support','roi','unknown']) nv.addVolume(cached[key]);
    // NiiVue may infer a window when a volume is added; enforce the recorded display window.
    cached.mean.cal_min=data.sources[name].window[0]; cached.mean.cal_max=data.sources[name].window[1];
    current=cached; state.loading=false;
    updateRegion(); focusRegion(state.side); setMode(state.mode);
    $('coverage-canvas').style.visibility='visible'; message.hidden=true;
    document.body.dataset.viewerStatus='ready'; sendHeight();
  } catch(error) { if(sequence===state.load) viewerFailure(error); }
}

function setMode(mode) {
  state.mode=mode; setPressed('view-slices',mode==='slices'); setPressed('view-3d',mode==='3d');
  $('slice-plane').disabled=mode==='3d';
  $('smooth-image').disabled=mode==='3d';
  const planes={coronal:library?.SLICE_TYPE.CORONAL,axial:library?.SLICE_TYPE.AXIAL,sagittal:library?.SLICE_TYPE.SAGITTAL,all:library?.SLICE_TYPE.MULTIPLANAR};
  if(nv && current) nv.setSliceType(mode==='slices'?planes[state.plane]:library.SLICE_TYPE.RENDER);
  applyDisplaySampling();
}

function zoomView(factor) {
  if(!nv || !current || state.loading) return;
  const pan=Array.from(nv.scene.pan2Dxyzmm);
  const scale=state.mode==='3d'?nv.scene.volScaleMultiplier:pan[3];
  pan[3]=Math.min(3,Math.max(.5,scale*factor));
  nv.setPan2Dxyzmm(pan);
}

function enlargeView() {
  state.expanded=!state.expanded;
  document.querySelector('.workspace').classList.toggle('expanded',state.expanded);
  text('enlarge-view',state.expanded?'Compact':'Enlarge');
  setPressed('enlarge-view',state.expanded);
  requestAnimationFrame(sendHeight);
}

function download(name, data, type) {
  const link=document.createElement('a'), url=URL.createObjectURL(new Blob([data],{type}));
  link.href=url; link.download=name; document.body.append(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

function exportSeries() {
  const m=state.data.motion, signal=observations().series;
  const columns=['participant_id','source','region_id','region_name','original_index','time_seconds','raw_signal','fd_from_tutorial_mm','fd_stored','baseline_keep','censored_keep','baseline_reason','censored_reason'];
  const cell=v=>v===null||v===undefined?'':typeof v==='string'?`"${v.replaceAll('"','""')}"`:String(v);
  const rows=m.original_index.map((index,i)=>[state.data.participant_id,state.source,state.region,region().name,index,m.time_seconds[i],signal?.[i]??null,m.fd[i],m.fd_stored[i],m.baseline_keep[i],m.censored_keep[i],m.baseline_reason[i],m.censored_reason[i]].map(cell).join(','));
  download(`${state.data.participant_id}-region-${state.region}-${state.source}.csv`,[columns.join(','),...rows].join('\n'),'text/csv;charset=utf-8');
}

function exportEvidence() {
  const src=source();
  const evidence={participant_id:state.data.participant_id,region_id:state.region,region_name:region().name,source:state.source,
    source_filename:src.source_filename,source_sha256:src.source_sha256,preprocessing:src.preprocessing,source_note:src.note,
    mask_source_filename:src.mask_source_filename,mask_source_sha256:src.mask_source_sha256,
    support_measure:src.support_label,denominator:src.denominator,support:observations().support,
    regional_signal_constant:observations().constant,primary_status:state.data.primary_status,primary_analysis:state.data.primary_analysis,
    original_frames:state.data.n_volumes,tr_seconds:state.data.tr_seconds,n_fd_observed:state.data.motion.n_fd_observed,
    atlas:state.data.atlas,settings:state.data.settings,prepared_with:state.data.prepared_with,
    display_only:{plane:state.plane,mode:state.mode,bold_2d_interpolation:$('smooth-image').checked?'linear':'nearest',categorical_2d_overlays:'nearest',volume_rendering:'NiiVue default trilinear ray-casting; visualization only',intensity_window:current?[current.mean.cal_min,current.mean.cal_max]:null,expanded:state.expanded},
    preparation:{asset_key:state.data.asset_key,inputs:state.data.preparation_inputs,generated_at_utc:state.data.generated_at_utc}};
  download(`${state.data.participant_id}-region-${state.region}-${state.source}-evidence.json`,JSON.stringify(evidence,null,2),'application/json');
}

function pause() { clearInterval(state.timer); state.timer=null; text('play-cursor','Play cursor'); setPressed('play-cursor',false); }

function registerControls() {
  let pointerStart=null;
  $('coverage-canvas').addEventListener('pointerdown',event=>{pointerStart=[event.clientX,event.clientY];});
  $('coverage-canvas').addEventListener('click',event=>{
    if(!nv || !current || state.loading || state.mode!=='slices' || !pointerStart ||
      Math.hypot(event.clientX-pointerStart[0],event.clientY-pointerStart[1])>5) return;
    const mm=Array.from(nv.frac2mm(nv.scene.crosshairPos)).slice(0,3);
    const xyz=Array.from(current.atlas.mm2vox(mm)).map(Math.round);
    const selected=current.atlas.getValue(...xyz);
    if(state.data.regions.some(r=>r.id===selected)) selectRegion(selected,false);
  });
  $('region-select').addEventListener('change',e=>selectRegion(e.target.value));
  $('source-tutorial').addEventListener('click',()=>switchSource('tutorial'));
  $('source-archive').addEventListener('click',()=>switchSource('archive'));
  $('view-slices').addEventListener('click',()=>setMode('slices')); $('view-3d').addEventListener('click',()=>setMode('3d'));
  $('slice-plane').addEventListener('change',event=>{state.plane=event.target.value;setMode('slices');});
  $('enlarge-view').addEventListener('click',enlargeView);
  $('zoom-in').addEventListener('click',()=>zoomView(1.2)); $('zoom-out').addEventListener('click',()=>zoomView(1/1.2));
  $('smooth-image').addEventListener('change',applyDisplaySampling);
  $('focus-left').addEventListener('click',()=>focusRegion('left')); $('focus-right').addEventListener('click',()=>focusRegion('right'));
  $('reset-view').addEventListener('click',()=>{if(nv){nv.setPan2Dxyzmm([0,0,0,1]);nv.setRenderAzimuthElevation(120,15);} focusRegion(state.side);});
  $('show-region').addEventListener('change',updateRegion); $('show-support').addEventListener('change',updateRegion);
  $('frame-range').addEventListener('input',e=>setFrame(e.target.value));
  $('play-cursor').addEventListener('click',()=>{if(state.timer){pause();return;}text('play-cursor','Pause cursor');setPressed('play-cursor',true);state.timer=setInterval(()=>{if(state.frame>=state.data.n_volumes-1){pause();return;}setFrame(state.frame+1);},140);});
  for(const id of ['signal-chart','motion-chart']) {
    $(id).addEventListener('pointermove',event=>{
      if(event.pointerType==='touch' && !event.buttons) return;
      const rect=event.currentTarget.getBoundingClientRect(), W=Number(event.currentTarget.dataset.width), start=Number(event.currentTarget.dataset.start), width=Number(event.currentTarget.dataset.plotWidth);
      const t=((event.clientX-rect.left)/rect.width*W-start)/width*state.data.motion.time_seconds.at(-1);
      setFrame(nearestFrame(t,state.data.motion.time_seconds));
    });
  }
  $('export-series').addEventListener('click',exportSeries); $('export-evidence').addEventListener('click',exportEvidence);
  new ResizeObserver(()=>{requestAnimationFrame(()=>{drawCharts();sendHeight();});}).observe(document.querySelector('.explorer'));
}

async function initialize(data) {
  if(!data || !Array.isArray(data.regions)) return;
  if(state.data?.participant_id===data.participant_id && state.data?.asset_key===data.asset_key) return;
  pause(); state.load++; sourceCache=new Map(); current=null;
  state.data=data; state.region=data.initial_region||14; state.source=data.initial_source||'tutorial'; state.frame=0; state.side='right';
  $('frame-range').max=String(data.n_volumes-1);
  const selector=$('region-select'); selector.replaceChildren();
  for(const r of [...data.regions.filter(r=>[14,37].includes(r.id)),...data.regions.filter(r=>![14,37].includes(r.id))]) {
    const option=document.createElement('option');option.value=String(r.id);option.textContent=`${r.id}: ${r.name}`;selector.append(option);
  }
  renderInfo(); await switchSource(state.source);
}

registerControls();
window.coverageExplorer={getState:()=>({participant:state.data?.participant_id,source:state.source,region:state.region,frame:state.frame,side:state.side,mode:state.mode,plane:state.plane,expanded:state.expanded,viewerStatus:document.body.dataset.viewerStatus,loading:state.loading}),selectRegion,switchSource,setFrame,get viewer(){return nv;},get lastLocation(){return lastLocation;}};
if(standalone) initialize(JSON.parse($('coverage-data').textContent));
else {
  window.addEventListener('message',event=>{
    if(event.source===window.parent && event.data?.type==='streamlit:render') initialize(event.data.args?.data);
  });
  window.parent.postMessage({isStreamlitMessage:true,type:'streamlit:componentReady',apiVersion:1},'*');
  sendHeight();
}
