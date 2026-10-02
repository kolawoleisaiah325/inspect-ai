'use strict';
(()=>{
  const byId=id=>document.getElementById(id);
  let selected=null, busy=false, imageUrl=null;
  const historyKey='inspectai-history-v1';
  let history=[];
  try{const stored=JSON.parse(localStorage.getItem(historyKey)||'[]');if(Array.isArray(stored))history=stored.slice(-50);}catch{}
  const setStatus=(text,error=false)=>{byId('request-status').textContent=text;byId('request-status').classList.toggle('error',error);};
  function setBusy(value){busy=value;byId('inspect-button').disabled=value||!selected;byId('image-upload').disabled=value;document.querySelectorAll('.sample-button').forEach(b=>b.disabled=value);byId('inspect-button').firstChild.textContent=value?'Inspecting… ':'Run live inspection ';}
  function clearResult(){byId('inspection-result').hidden=true;byId('heatmap-image').hidden=true;byId('heatmap-placeholder').hidden=false;byId('result-origin').textContent='Ready for inspection';}
  function showResult(result,origin,heatmap){
    byId('inspection-result').hidden=false;byId('inspection-result').dataset.decision=result.decision;
    byId('decision').textContent=result.decision==='review'?'Review needed':'No anomaly flagged';
    byId('decision-detail').textContent=result.decision==='review'?'Above the normal-calibrated threshold':'Below the normal-calibrated threshold';
    byId('relative-score').textContent=Number(result.relative_score).toFixed(2)+'×';
    byId('inference-time').textContent=Number(result.inference_ms).toFixed(0)+' ms';
    byId('timing-detail').textContent=origin==='Live API result'?'Model + scoring · CPU':'Recorded sample · local CPU';
    byId('result-origin').textContent=origin;
    byId('heatmap-image').src=heatmap||result.heatmap;byId('heatmap-image').hidden=false;byId('heatmap-placeholder').hidden=true;
  }
  function preview(blob,name){
    if(imageUrl)URL.revokeObjectURL(imageUrl);
    imageUrl=URL.createObjectURL(blob);
    for(const id of ['original-image','overlay-original']){byId(id).src=imageUrl;byId(id).hidden=false;}
    byId('original-image').onload=()=>{const image=byId('original-image');document.querySelectorAll('.image-stage').forEach(stage=>stage.style.aspectRatio=image.naturalWidth+'/'+image.naturalHeight);};
    byId('image-name').textContent=name;
  }
  async function selectSample(sample){
    if(busy)return;setBusy(true);clearResult();setStatus('Loading the sample…');
    try{
      const response=await fetch(sample.file);if(!response.ok)throw new Error('Sample unavailable. Please refresh.');
      const blob=await response.blob();selected={blob,name:sample.id+'.jpg'};preview(blob,selected.name);
      document.querySelectorAll('.sample-button').forEach(button=>button.classList.toggle('active',button.dataset.id===sample.id));
      showResult(sample.recorded_result,'Recorded sample · not live','samples/'+sample.id+'-heat.png');
      setStatus('Recorded example shown. Run live inspection to call the model API.');
    }catch(error){selected=null;setStatus(error.message,true);}finally{setBusy(false);}
  }
  byId('image-upload').addEventListener('change',()=>{
    if(busy)return;const file=byId('image-upload').files[0];if(!file)return;
    if(!['image/jpeg','image/png','image/webp'].includes(file.type)||file.size>2*1024*1024){setStatus('Choose a JPEG, PNG or WebP smaller than 2 MB.',true);return;}
    selected={blob:file,name:file.name};clearResult();preview(file,file.name);document.querySelectorAll('.sample-button').forEach(b=>b.classList.remove('active'));setStatus('Image selected. Nothing has been uploaded yet.');setBusy(false);
  });
  byId('inspect-button').addEventListener('click',async()=>{
    if(busy||!selected)return;setBusy(true);setStatus('Running real CPU inference. The first request may take longer…');
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),65000);
    try{
      const started=performance.now();const response=await fetch('/api/inspect',{method:'POST',headers:{'Content-Type':selected.blob.type||'application/octet-stream'},body:selected.blob,signal:controller.signal});
      const result=await response.json();if(!response.ok)throw new Error(result.detail||'Inspection unavailable. Please try again.');
      showResult(result,'Live API result');setStatus('Inspection complete · '+Math.round(performance.now()-started)+' ms round trip. Review the evidence before drawing conclusions.');
      history.push({id:crypto.randomUUID(),name:selected.name,decision:result.decision,relative_score:result.relative_score,inference_ms:result.inference_ms,model_version:result.model_version,time:new Date().toISOString(),review:'unreviewed'});history=history.slice(-50);saveHistory();renderHistory();
    }catch(error){setStatus(error.name==='AbortError'?'The API took too long. Your image was not saved. Please retry.':error.message,true);}finally{clearTimeout(timer);setBusy(false);}
  });
  byId('overlay-opacity').addEventListener('input',event=>{byId('heatmap-image').style.opacity=event.target.value/100;byId('opacity-value').textContent=event.target.value+'%';});
  function saveHistory(){try{localStorage.setItem(historyKey,JSON.stringify(history));}catch{}}
  function renderHistory(){
    const body=byId('history-body');body.replaceChildren();byId('clear-history').disabled=!history.length;byId('export-history').disabled=!history.length;
    if(!history.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=5;cell.textContent='No live inspections yet.';row.append(cell);body.append(row);return;}
    [...history].reverse().forEach(record=>{
      const row=document.createElement('tr');[record.name,record.decision==='review'?'Review needed':'No anomaly flagged',Number(record.relative_score).toFixed(2)+'×',Number(record.inference_ms).toFixed(0)+' ms'].forEach(text=>{const cell=document.createElement('td');cell.textContent=text;row.append(cell);});
      const cell=document.createElement('td'),select=document.createElement('select');select.setAttribute('aria-label','Review of '+record.name);
      [['unreviewed','Unreviewed'],['confirmed','Signal confirmed'],['false_alarm','False alarm'],['missed','Missed issue'],['uncertain','Uncertain']].forEach(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;select.append(option);});
      select.value=record.review;select.addEventListener('change',()=>{record.review=select.value;saveHistory();});cell.append(select);row.append(cell);body.append(row);
    });
  }
  byId('clear-history').addEventListener('click',()=>{history=[];saveHistory();renderHistory();});
  byId('export-history').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({model:'inspectai-bottle-v1',notice:'Local review metadata; no images. Scores are not probabilities.',inspections:history},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='inspectai-history.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  renderHistory();
  fetch('/api/health').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(()=>byId('service-status').textContent='● API ready · CPU').catch(()=>byId('service-status').textContent='API unavailable');
  fetch('samples.json').then(r=>{if(!r.ok)throw new Error('Samples unavailable.');return r.json();}).then(samples=>{
    samples.forEach(sample=>{const button=document.createElement('button');button.className='sample-button';button.dataset.id=sample.id;button.setAttribute('aria-label','Select '+sample.label+' sample');const image=document.createElement('img');image.src=sample.file;image.alt='';const text=document.createElement('span');text.textContent=sample.label;button.append(image,text);button.addEventListener('click',()=>selectSample(sample));byId('samples').append(button);});selectSample(samples[1]);
  }).catch(error=>setStatus(error.message,true));
  fetch('evaluation.json').then(r=>{if(!r.ok)throw new Error();return r.json();}).then(data=>{
    [[data.split.fit,'Normal images fitted'],[data.split.normal_calibration,'Normals for calibration'],[data.split.test,'Official test images']].forEach(([number,label])=>{const div=document.createElement('div'),strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=number;span.textContent=label;div.append(strong,span);byId('split-stats').append(div);});
    Object.entries(data.methods).forEach(([name,value])=>{const row=document.createElement('tr');[name==='pixel_baseline'?'Pixel baseline':'CNN + patch memory',(value.metrics.image_auroc*100).toFixed(1)+'%',value.metrics.false_negative+' / 63',value.metrics.false_positive+' / 20'].forEach(text=>{const cell=document.createElement('td');cell.textContent=text;row.append(cell);});byId('evaluation-body').append(row);});
  }).catch(()=>{byId('evaluation-body').textContent='Recorded evaluation could not be loaded.';});
})();
