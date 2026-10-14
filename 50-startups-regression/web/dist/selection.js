"use strict";
// Pure chart helper is also exercised by the non-browser client tests.
function selectionChart(rows, metric, active, maxK, best){
  const palette={Lasso:"#276e59","Forward SFS":"#c47b37",RFE:"#5579b0",ElasticNet:"#a35981",Exhaustive:"#7058a4"};
  const valid=rows.filter(row=>Number.isFinite(row[metric]));
  if(!valid.length)return '<p>No evaluated points for this configuration.</p>';
  const vals=valid.map(row=>row[metric]);let low=Math.min(...vals),high=Math.max(...vals);
  const pad=(high-low)*.12||Math.max(Math.abs(high)*.1,.1);low-=pad;high+=pad;
  const x=k=>60+(k-1)*350/Math.max(maxK-1,1),y=v=>225-(v-low)*190/(high-low);
  let content='';
  for(let i=0;i<5;i++){const v=low+(high-low)*i/4,py=y(v);content+='<line x1="60" x2="420" y1="'+py+'" y2="'+py+'" stroke="#e2e9e4"/><text x="54" y="'+(py+4)+'" text-anchor="end" font-size="10">'+escapeHTML(Number(v.toPrecision(4)))+'</text>';}
  const names=[...new Set(valid.map(row=>row.algorithm))];
  names.forEach(name=>{
    const series=valid.filter(row=>row.algorithm===name).sort((a,b)=>a.k-b.k);
    // Separate segments across compute-limited counts, so gaps remain visible.
    let previous=null;
    series.forEach(row=>{if(previous&&row.k===previous.k+1)content+='<line x1="'+x(previous.k)+'" y1="'+y(previous[metric])+'" x2="'+x(row.k)+'" y2="'+y(row[metric])+'" stroke="'+palette[name]+'" stroke-width="2"/>';previous=row;
      const chosen=row.algorithm===active.algorithm&&row.k===active.k;
      const recommended=row.algorithm===best.algorithm&&row.k===best.k;
      const title=name+' · k='+row.k+' · '+metric+'='+row[metric].toPrecision(6)+(recommended?' · training CV recommendation':'');
      content+='<circle tabindex="0" role="button" aria-label="'+escapeHTML(title)+'" data-selection-algorithm="'+escapeHTML(name)+'" data-selection-k="'+row.k+'" cx="'+x(row.k)+'" cy="'+y(row[metric])+'" r="'+(chosen?7:4)+'" fill="'+palette[name]+'" stroke="'+(recommended?'#e3b94c':'white')+'" stroke-width="'+(recommended?3:1)+'"><title>'+escapeHTML(title)+'</title></circle>';
    });
  });
  for(let k=1;k<=maxK;k++)if(maxK<=15||k===1||k===maxK||k%5===0)content+='<text x="'+x(k)+'" y="245" text-anchor="middle" font-size="11">'+k+'</text>';
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 275" role="img" aria-label="'+(metric==='rmse'?'Test RMSE':'Test R-squared')+' by encoded feature count" style="font-family:Arial,sans-serif;fill:#52615b">'+content+'<text x="230" y="266" text-anchor="middle" font-size="12">Number of encoded features (k)</text></svg><div class="selection-legend">'+names.map(name=>'<span style="border-color:'+palette[name]+'">'+escapeHTML(name)+'</span>').join('')+'</div>';
}
const SelectionExplorer=(()=>{
  const $=id=>document.getElementById(id),e=value=>escapeHTML(String(value));
  const fmt=(v,d=5)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumSignificantDigits:d});
  let getConfig,getPreview,result=null,config=null,revision=0,pending=null,predictRevision=0,descriptivePending=null;
  const abort=()=>{if(pending)pending.abort();pending=null;};
  function clearPrediction(){predictRevision++;abort();$('selection-prediction').textContent='';$('selection-contributions').innerHTML='';$('selection-predict-status').textContent='';}
  function reset(){revision++;clearDescriptive();clearPrediction();result=null;config=null;$('selection-results').hidden=true;$('selection-status').textContent='Configuration changed. Run feature selection for the current dataset and inputs.';refresh();}
  function refresh(){const p=getPreview?.(),c=getConfig?.();$('run-selection').disabled=!p||!c?.target||!c?.features?.length;}
  function active(){return {algorithm:$('selection-algorithm').value,k:Number($('selection-k').value)};}
  function current(){const a=active();return result?.results.find(r=>r.algorithm===a.algorithm&&r.k===a.k);}
  function save(content,name,type){const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function post(operation,payload,signal){const r=await fetch('/api/'+operation,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal});const body=await r.json();if(!r.ok)throw new Error(body.error||'Selection request failed.');return body;}
  async function run(){
    reset();config=JSON.parse(JSON.stringify(getConfig()));const runConfig=config,token=revision;
    if(!config.target||!config.features.length){$('selection-status').textContent='Choose a numerical target and one or more inputs.';return;}
    pending=new AbortController();const signal=pending.signal;
    $('run-selection').disabled=true;$('selection-status').textContent='Comparing five methods with nested training CV…';
    try{
      const reply=await post('selection',runConfig,signal);
      if(token!==revision||signal.aborted)return;
      result=reply;$('selection-results').hidden=false;
      $('selection-algorithm').innerHTML=['Lasso','Forward SFS','RFE','ElasticNet','Exhaustive'].map(n=>'<option>'+n+'</option>').join('');
      $('selection-algorithm').value=reply.best.algorithm;$('selection-k').max=reply.maxK;$('selection-k').value=reply.best.k;
      $('selection-methodology').textContent=reply.dataset.name+' · target '+reply.target+' · '+reply.methodology;
      const messages=[...reply.warnings,...reply.skipped.map(s=>s.algorithm+' k='+s.k+': '+s.reason)];
      $('selection-notices').hidden=!messages.length;$('selection-notices').textContent=messages.join(' ');
      $('selection-best').textContent='Training CV recommendation: '+reply.best.algorithm+' with '+reply.best.k+' encoded features. Full-input OLS baseline: test RMSE '+fmt(reply.baseline.rmse)+', R² '+fmt(reply.baseline.r2)+'.';
      $('selection-comparison').innerHTML=reply.results.map(r=>'<tr><td>'+e(r.algorithm)+'</td><td>'+r.k+'</td><td>'+e(r.features.join(', '))+'</td><td>'+fmt(r.cvRMSE)+'</td><td>'+fmt(r.rmse)+'</td><td>'+fmt(r.r2)+'</td><td>'+fmt(r.mae)+'</td><td>'+fmt(r.mse)+'</td></tr>').join('');
      const preview=getPreview();
      $('selection-predict-fields').innerHTML=runConfig.features.map((name,i)=>{const numeric=preview.columns.find(c=>c.name===name).kind==='numeric';return '<label class="field">'+e(name)+'<input id="selection-input-'+i+'" type="'+(numeric?'number':'text')+'" '+(numeric?'step="any"':'')+' value="'+e(reply.exampleInput[name]??'')+'"></label>';}).join('');
      render();$('selection-status').textContent='Completed '+reply.results.length+' algorithm / k configurations on '+reply.dataset.name+'.';
    }catch(error){if(token===revision&&!signal.aborted)$('selection-status').textContent=error.message;}
    finally{if(token===revision)refresh();}
  }
  function renderCharts(){if(!result)return;const a=active(),rows=$('selection-chart-mode').value==='all'?result.results:result.results.filter(r=>r.algorithm===a.algorithm);$('selection-rmse-chart').innerHTML=selectionChart(rows,'rmse',a,result.maxK,result.best);$('selection-r2-chart').innerHTML=selectionChart(rows,'r2',a,result.maxK,result.best);}
  function render(){
    clearDescriptive();clearPrediction();if(!result)return;const a=active(),r=current();$('selection-k-label').textContent=a.k+' / '+result.maxK;renderCharts();
    $('selection-detail').hidden=!r;
    if(!r){$('selection-status').textContent=result.skipped.find(s=>s.algorithm===a.algorithm&&s.k===a.k)?.reason||'This configuration could not be evaluated.';return;}
    $('selection-status').textContent=result.dataset.name+' · '+r.algorithm+' · k='+r.k+' · target '+result.target;
    $('selection-detail-title').textContent=r.algorithm+' · '+r.k+' encoded predictors';$('selection-chips').innerHTML=chips(r.features);
    const [intercept,...terms]=r.coefficients;
    $('selection-equation').textContent=result.target+' = '+fmt(intercept.coefficient,10)+terms.map(c=>(c.coefficient<0?' − ':' + ')+fmt(Math.abs(c.coefficient),10)+' × ['+c.feature+']').join('');
    const s=r.summary,metrics=[['Test RMSE',r.rmse],['Test R²',r.r2],['Test MAE',r.mae],['Test MSE',r.mse],['Nested CV RMSE',r.cvRMSE],['Training R²',s.trainR2],['Adjusted training R²',s.adjustedTrainR2],['Residual df',s.residualDf],['Training F',s.F],['F p-value',s.F_p],['Residual standard error',s.residualStdError],['AIC',s.AIC],['BIC',s.BIC],['Design rank',s.rank],['Scaled condition number',s.conditionNumber]];
    $('selection-summary').innerHTML=metrics.map(([label,v])=>'<span>'+e(label)+'<strong>'+fmt(v)+'</strong></span>').join('');
    $('selection-inference-note').textContent=r.inferenceNote;
    $('selection-coefficients').innerHTML=r.coefficients.map(c=>'<tr><td>'+e(c.feature)+'</td><td>'+fmt(c.coefficient,10)+'</td><td>'+fmt(c.stderr)+'</td><td>'+fmt(c.t)+'</td><td>'+fmt(c.p)+'</td><td>'+(c.ciLower==null?'—':'['+fmt(c.ciLower)+', '+fmt(c.ciUpper)+']')+'</td><td>'+e(c.vif==null?c.vifStatus:fmt(c.vif))+'</td></tr>').join('');
    $('selection-ranking').textContent=JSON.stringify(result.rankings[r.algorithm]||{method:r.algorithm,features:r.features,criterion:r.algorithm==='RFE'?'Remove smallest absolute standardized OLS coefficient.':'Minimize three-fold training RMSE; refit OLS.'},null,2);
    $('selection-scatter').innerHTML=scatterChart(r.predictions,false,result.target);$('selection-residual').innerHTML=scatterChart(r.predictions,true,result.target);
    $('selection-outliers').innerHTML=r.outliers.map((p,i)=>'<tr><td>'+(i+1)+'</td><td>'+p.Row_Index+'</td><td>'+fmt(p.Actual_Profit)+'</td><td>'+fmt(p.Predicted_Profit)+'</td><td>'+fmt(p.Residual)+'</td><td>'+fmt(p.absoluteError)+'</td><td>'+fmt(p.percentageError)+'</td></tr>').join('');
  }
  async function predict(event){
    event.preventDefault();if(!result||!current())return;clearPrediction();const token=revision,ptoken=predictRevision,a=active(),input={};
    config.features.forEach((f,i)=>{input[f]=$('selection-input-'+i).value.trim()||null;});
    pending=new AbortController();const signal=pending.signal;$('selection-predict-status').textContent='Calculating with '+a.algorithm+' / k='+a.k+'…';
    try{const reply=await post('selection-predict',{...config,...a,input},signal);
      if(token!==revision||ptoken!==predictRevision||signal.aborted||reply.fingerprint!==result?.fingerprint)return;
      $('selection-prediction').textContent=result.target+': '+fmt(reply.prediction,10);
      $('selection-contributions').innerHTML=reply.contributions.map(c=>'<tr><td>'+e(c.feature)+'</td><td>'+fmt(c.value,10)+'</td><td>'+fmt(c.coefficient,10)+'</td><td>'+fmt(c.contribution,10)+'</td></tr>').join('');
      $('selection-predict-status').textContent=reply.warnings.join(' ')||'Prediction uses the same training-fitted subset and preprocessing.';
    }catch(error){if(token===revision&&ptoken===predictRevision&&!signal.aborted)$('selection-predict-status').textContent=error.message;}
  }
  function clearDescriptive(){descriptivePending?.abort();descriptivePending=null;$('selection-descriptive-note').textContent='';for(const id of ['selection-descriptive-chart','selection-descriptive-outliers'])$(id).innerHTML='';}
  async function descriptive(){
    if(!result||!current())return;clearDescriptive();descriptivePending=new AbortController();const signal=descriptivePending.signal,token=revision,a=active();
    $('selection-descriptive-note').textContent='Fitting the descriptive model on all valid rows…';
    try{const reply=await post('selection-descriptive',{...config,...a},signal);
      if(signal.aborted||token!==revision||reply.fingerprint!==result?.fingerprint)return;
      $('selection-descriptive-note').textContent=reply.note;
      $('selection-descriptive-chart').innerHTML=scatterChart(reply.predictions,false,result.target,true);
      $('selection-descriptive-outliers').innerHTML=reply.outliers.map((p,i)=>'<tr><td>'+(i+1)+'</td><td>'+p.Row_Index+'</td><td>'+fmt(p.Actual_Profit)+'</td><td>'+fmt(p.Predicted_Profit)+'</td><td>'+fmt(p.Residual)+'</td></tr>').join('');
    }catch(error){if(!signal.aborted&&token===revision)$('selection-descriptive-note').textContent=error.message;}
  }
  function init(configCallback,previewCallback){
    getConfig=configCallback;getPreview=previewCallback;
    $('run-selection').addEventListener('click',run);
    $('selection-descriptive-run').addEventListener('click',descriptive);
    $('selection-algorithm').addEventListener('change',render);$('selection-k').addEventListener('input',render);$('selection-chart-mode').addEventListener('change',renderCharts);
    const choose=event=>{if(event.type==='keydown'&&!['Enter',' '].includes(event.key))return;const p=event.target.closest('[data-selection-k]');if(p){event.preventDefault();$('selection-algorithm').value=p.dataset.selectionAlgorithm;$('selection-k').value=p.dataset.selectionK;render();}};
    for(const id of ['selection-rmse-chart','selection-r2-chart']){$(id).addEventListener('click',choose);$(id).addEventListener('keydown',choose);}
    $('selection-predict-form').addEventListener('submit',predict);$('selection-predict-form').addEventListener('input',clearPrediction);
    $('selection-csv').addEventListener('click',()=>{if(!result)return;const keys=['algorithm','k','features','cvRMSE','cvR2','rmse','r2','mae','mse'];const q=v=>'"'+String(v??'').replace(/"/g,'""')+'"';save([keys.join(','),...result.results.map(r=>keys.map(k=>q(Array.isArray(r[k])?r[k].join(' | '):r[k])).join(','))].join('\r\n'),'selection_comparison.csv','text/csv');});
    $('selection-json').addEventListener('click',()=>{if(result)save(JSON.stringify(result,null,2),'selection_evidence.json','application/json');});
    $('selection-svg').addEventListener('click',()=>{if(!result)return;const rmse=$('selection-rmse-chart').querySelector('svg').innerHTML,r2=$('selection-r2-chart').querySelector('svg').innerHTML;save('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 325" style="font-family:Arial;fill:#52615b"><rect width="900" height="325" fill="white"/><text x="170" y="20">Test RMSE</text><text x="620" y="20">Test R²</text><g transform="translate(0 20)">'+rmse+'</g><g transform="translate(450 20)">'+r2+'</g><text x="20" y="315">Lasso: green · SFS: orange · RFE: blue · ElasticNet: pink · Exhaustive: purple</text></svg>','selection_charts.svg','image/svg+xml');});
    refresh();
  }
  return {init,reset,refresh};
})();
