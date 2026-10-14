"use strict";
// A revision changes for every dataset/configuration edit. Pending results can
// only commit to the revision that requested them, preventing stale displays.
class LatestDatasetSession {
  constructor(){this.revision=0;this.payload=null;this.preview=null;this.result=null;this.config=null;this.pending=null;}
  invalidate(){this.revision++;if(this.pending)this.pending.abort();this.pending=null;this.result=null;this.config=null;return this.revision;}
  select(payload){this.invalidate();this.payload=payload;this.preview=null;}
  begin(){if(this.pending)this.pending.abort();this.pending=new AbortController();return {revision:this.revision,signal:this.pending.signal};}
  current(task){return task.revision===this.revision&&!task.signal.aborted;}
  commit(task,result,config){if(!this.current(task))return false;this.result=result;this.config=config;return true;}
}
(function(){
  const session=new LatestDatasetSession();
  const names=["Multiple Linear Regression","Polynomial Regression","Decision Tree Regression","Random Forest Regression","Support Vector Regression","K-Nearest Neighbors Regression"];
  const el=id=>document.getElementById(id);
  const e=escapeHTML;
  const fmt=(value,digits=2)=>value==null?"—":number(value,digits);
  const pageActive=()=>!location.hash||location.hash==="#analysis";
  function setHeader(){el("workspace-dataset-label").textContent=pageActive()?(session.preview?.dataset.name||"Regression analysis"):"50 Startups · original study";el("workspace-status-label").textContent=pageActive()?(session.result?"Analysis complete":"Dataset workspace"):"Verified original study";}
  function status(message,error=false){setHeader();el("analysis-status").textContent=message;el("analysis-status").className="analysis-status"+(error?" error":"");}
  function hideResults(message){
    session.invalidate();
    SelectionExplorer.reset();
    el("analysis-results").hidden=true;
    el("download-analysis").disabled=true;
    el("analysis-predict-output").textContent="";
    el("analysis-predict-status").textContent="";
    el("run-analysis").disabled=!session.preview;
    status(message);
  }
  async function post(operation,payload,signal){
    const response=await fetch("/api/"+operation,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload),signal});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"The analysis request failed.");
    return result;
  }
  function selected(selector){return [...document.querySelectorAll(selector)].filter(input=>input.checked).map(input=>input.value);}
  function renderFeatures(){
    if(!session.preview)return;
    const target=el("target-select").value;
    el("input-feature-options").innerHTML=session.preview.columns.filter(c=>c.name!==target).map(c=>'<label><input type="checkbox" name="analysis-feature" value="'+e(c.name)+'" checked><span>'+e(c.name)+'</span><span class="column-kind">'+e(c.kind)+'</span></label>').join("");
    preparationText();
    SelectionExplorer.refresh();
  }
  function preparationText(){
    if(!session.preview)return;
    const features=selected('input[name="analysis-feature"]');
    const columns=session.preview.columns.filter(c=>features.includes(c.name));
    const numeric=columns.filter(c=>c.kind==="numeric").length;
    el("preprocessing-summary").textContent=numeric+" numeric / "+(columns.length-numeric)+" categorical inputs. Training median/mode imputation, one-hot encoding, and model-specific scaling inside pipelines.";
  }
  function renderPreview(){
    const d=session.preview;
    el("dataset-details").hidden=false;
    el("dataset-preview-title").textContent=d.dataset.name;
    el("dataset-source-label").textContent=d.dataset.source;
    const cards=[["Rows",d.rows],["Columns",d.columnCount],["Missing values",d.missingValues],["Target",el("target-select").value||"Choose target"]];
    el("dataset-statistics").innerHTML=cards.map(([title,value])=>'<article class="metric"><div class="metric-label">'+title+'</div><div class="metric-value">'+e(value)+'</div></article>').join("");
    el("dataset-preview-head").innerHTML="<tr>"+d.columns.map(c=>"<th>"+e(c.name)+"</th>").join("")+"</tr>";
    el("dataset-preview-body").innerHTML=d.preview.map(row=>"<tr>"+d.columns.map(c=>"<td>"+e(row[c.name]??"—")+"</td>").join("")+"</tr>").join("");
    el("column-statistics").innerHTML=d.columns.map(c=>"<tr><td>"+e(c.name)+"</td><td>"+e(c.dtype)+"</td><td>"+c.missing+"</td><td>"+c.unique+"</td><td>"+fmt(c.mean)+"</td><td>"+fmt(c.min)+"</td><td>"+fmt(c["50%"])+"</td><td>"+fmt(c.max)+"</td></tr>").join("");
    const correlations=d.correlations;
    el("dataset-correlations").innerHTML='<table class="correlation-table"><thead><tr><th></th>'+correlations.columns.map(c=>"<th>"+e(c)+"</th>").join("")+"</tr></thead><tbody>"+correlations.columns.map((name,i)=>"<tr><th>"+e(name)+"</th>"+correlations.values[i].map(value=>{
      const color=value==null?"#eef0eb":value>=0?"rgba(55,132,90,"+(0.08+Math.abs(value)*.5)+")":"rgba(184,116,65,"+(0.08+Math.abs(value)*.5)+")";
      return '<td style="background:'+color+'">'+fmt(value)+"</td>";
    }).join("")+"</tr>").join("")+"</tbody></table>";
    setHeader();
    SelectionExplorer.refresh();
  }
  async function loadPreview(autoRun){
    const task=session.begin();el("retry-dataset").hidden=true;
    try{
      const reply=await post("dataset",session.payload,task.signal);
      if(!session.current(task))return;
      session.preview=reply;
      el("target-select").innerHTML='<option value="">Select numerical target…</option>'+reply.columns.map(c=>'<option value="'+e(c.name)+'" '+(c.kind!=="numeric"?"disabled":"")+'>'+e(c.name)+(c.kind!=="numeric"?" (categorical)":"")+'</option>').join("");
      el("target-select").disabled=false;
      el("target-select").value=reply.dataset.defaultTarget||"";
      renderFeatures();renderPreview();
      el("run-analysis").disabled=false;
      if(!reply.columns.some(c=>c.kind==="numeric")){el("run-analysis").disabled=true;status("This CSV has no numerical target column. Upload a dataset with a numeric target.",true);return;}
      status(reply.dataset.name+" loaded. "+(autoRun?"Running the default comparison…":"Choose your target, inputs and models, then run analysis."));
      if(autoRun)await runAnalysis();
    }catch(error){
      if(error.name==="AbortError"||!session.current(task))return;
      status(error.message,true);el("target-select").disabled=true;el("run-analysis").disabled=true;el("retry-dataset").hidden=false;
    }
  }
  function switchDataset(){
    const dataset=el("dataset-select").value;
    el("retry-dataset").hidden=true;
    session.select({dataset});
    SelectionExplorer.reset();
    el("analysis-results").hidden=true;
    el("dataset-details").hidden=true;
    el("dataset-statistics").innerHTML="";
    el("input-feature-options").innerHTML="";
    el("dataset-preview-head").innerHTML="";
    el("dataset-preview-body").innerHTML="";
    el("target-select").innerHTML='<option value="">Select dataset…</option>';
    el("target-select").disabled=true;
    el("run-analysis").disabled=true;
    el("analysis-predict-output").textContent="";
    el("upload-controls").hidden=dataset!=="custom";
    el("dataset-upload").value="";
    el("preprocessing-summary").textContent="Select a dataset to inspect its preprocessing.";
    setHeader();
    if(dataset==="custom"){status("Upload a CSV to preview its columns and choose a numerical target.");return;}
    status("Loading "+(dataset==="boston"?"Boston Housing":"50 Startups")+"…");
    loadPreview(true);
  }
  async function upload(){
    hideResults("Reading CSV…");session.preview=null;el("dataset-details").hidden=true;
    el("target-select").disabled=true;el("input-feature-options").innerHTML="";el("run-analysis").disabled=true;SelectionExplorer.refresh();
    const file=el("dataset-upload").files[0],revision=session.revision;
    if(!file){status("Choose a CSV file.");return;}
    if(file.size>1000000){status("CSV upload must be no larger than 1 MB.",true);return;}
    try{
      const bytes=await file.arrayBuffer();
      if(revision!==session.revision||el("dataset-select").value!=="custom")return;
      const text=new TextDecoder("utf-8",{fatal:true}).decode(bytes);
      session.select({dataset:"custom",csv:text,filename:file.name});
      await loadPreview(false);
    }catch(error){if(revision===session.revision)status("Could not read this file. Use a valid UTF-8 CSV.",true);}
  }
  async function runAnalysis(){
    if(!session.preview)return;
    const target=el("target-select").value;
    const features=selected('input[name="analysis-feature"]'),models=selected('input[name="analysis-model"]');
    if(!target||!features.length||!models.length){hideResults("Choose a numerical target, at least one input feature, and at least one regression model.");return;}
    const config=JSON.parse(JSON.stringify({...session.payload,target,features,models}));
    hideResults("Running "+models.length+" model(s) on "+session.preview.dataset.name+"…");
    const task=session.begin();el("run-analysis").disabled=true;
    try{
      const reply=await post("analyze",config,task.signal);
      if(!session.commit(task,reply,config))return;
      renderResults();
      status("Analysis complete: "+reply.dataset.name+" · target "+reply.settings.target+" · "+reply.models.length+" models.");
    }catch(error){
      if(error.name==="AbortError"||!session.current(task))return;
      status(error.message,true);
    }finally{if(session.current(task))el("run-analysis").disabled=false;}
  }
  function renderDiagnostics(){
    if(!session.result)return;
    const r=session.result,rows=r.predictions.filter(p=>p.Model===el("analysis-chart-model").value);
    el("analysis-scatter").innerHTML=scatterChart(rows,false,r.settings.target);
    el("analysis-residual").innerHTML=scatterChart(rows,true,r.settings.target);
    el("analysis-predictions").innerHTML=rows.map(p=>"<tr><td>"+p.Row_Index+"</td><td>"+fmt(p.Actual_Profit)+"</td><td>"+fmt(p.Predicted_Profit)+"</td><td>"+fmt(p.Residual)+"</td></tr>").join("");
  }
  function renderResults(){
    const r=session.result,best=r.models.find(m=>m.Best_By_CV);
    el("analysis-results").hidden=false;
    el("download-analysis").disabled=false;
    el("analysis-results-heading").textContent=r.dataset.name+" · "+r.settings.target;
    el("analysis-result-settings").textContent=r.split.trainRows+" training / "+r.split.testRows+" test rows · five-fold CV · random_state=42";
    const cards=[["CV winner",best.Model],["Test R²",fmt(best.Test_R2,4)],["Test RMSE",fmt(best.Test_RMSE)],["Encoded inputs",best.Feature_Count]];
    el("analysis-metrics").innerHTML=cards.map(([label,value])=>'<article class="metric"><div class="metric-label">'+label+'</div><div class="metric-value '+(label==="CV winner"?"model-name-value":"")+'">'+e(value)+'</div></article>').join("");
    el("analysis-bars").innerHTML=comparisonBars(r.models);
    el("analysis-model-table").innerHTML=[...r.models].sort((a,b)=>a.CV_RMSE_Mean-b.CV_RMSE_Mean).map(m=>'<tr class="'+(m.Best_By_CV?"selected":"")+'"><td><span class="model-label">'+e(m.Model)+'</span>'+(m.Best_By_CV?'<span class="best-badge">CV WINNER</span>':"")+'</td><td>'+fmt(m.CV_RMSE_Mean)+' ± '+fmt(m.CV_RMSE_Std)+'</td><td>'+fmt(m.Test_R2,4)+'</td><td>'+fmt(m.Test_MAE)+'</td><td>'+fmt(m.Test_MSE)+'</td><td>'+fmt(m.Test_RMSE)+'</td></tr>').join("");
    const notices=[...r.warnings,...r.errors.map(error=>error.model+": "+error.error)];
    el("analysis-warnings").hidden=!notices.length;
    el("analysis-warnings").innerHTML=notices.map(message=>"<p>"+e(message)+"</p>").join("");
    const options=r.models.map(m=>"<option>"+e(m.Model)+"</option>").join("");
    el("analysis-chart-model").innerHTML=options;el("analysis-chart-model").value=r.bestModel;
    el("analysis-predict-model").innerHTML=options;el("analysis-predict-model").value=r.bestModel;
    renderDiagnostics();
    const p=r.preprocessing;
    el("analysis-preprocessing").textContent=p.numericFeatures.length+" numerical and "+p.categoricalFeatures.length+" categorical inputs. "+p.numericImputation+" / "+p.categoricalImputation+" imputation. "+p.encoding+". "+p.scaling+".";
    el("analysis-encoded-features").innerHTML=chips(p.encodedFeatures);
    el("analysis-workflow").innerHTML=r.workflow.map((phase,i)=>'<article class="method-card"><div class="method-number">0'+(i+1)+'</div><h3>'+e(phase.name)+'</h3><p>'+e(phase.description)+'</p></article>').join("");
    el("analysis-predict-fields").innerHTML=r.settings.features.map((name,i)=>{
      const value=r.exampleInput[name],numeric=p.numericFeatures.includes(name);
      return '<label class="field">'+e(name)+'<input id="analysis-input-'+i+'" data-feature="'+e(name)+'" type="'+(numeric?"number":"text")+'" '+(numeric?'step="any"':"")+' value="'+e(value)+'"></label>';
    }).join("");
    el("analysis-predict-output").textContent="";el("analysis-predict-status").textContent="";
  }
  function saveFile(content,name,type){
    const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function downloadResults(){
    if(!session.result)return;
    const columns=["Model","CV_RMSE_Mean","CV_RMSE_Std","Test_R2","Test_MAE","Test_MSE","Test_RMSE","Train_R2","Feature_Count"];
    const quote=value=>'"'+String(value??"").replace(/"/g,'""')+'"';
    const csv=[columns.join(","),...session.result.models.map(row=>columns.map(key=>quote(row[key])).join(","))].join("\r\n");
    saveFile(csv,"regression_results.csv","text/csv;charset=utf-8");
  }
  async function predictInput(event){
    event.preventDefault();if(!session.result)return;
    const task=session.begin(),r=session.result,model=el("analysis-predict-model").value,input={};
    el("analysis-predict-output").textContent="";
    r.settings.features.forEach((name,i)=>{const value=el("analysis-input-"+i).value.trim();input[name]=value===""?null:value;});
    el("analysis-predict-status").textContent="Calculating "+r.settings.target+"…";
    try{
      const reply=await post("predict",{...session.config,model,input},task.signal);
      if(!session.current(task)||!session.result||reply.fingerprint!==session.result.fingerprint)return;
      el("analysis-predict-output").textContent=r.settings.target+": "+fmt(reply.prediction,4);
      el("analysis-predict-status").textContent=model+(reply.warnings.length?" · "+reply.warnings.join(" "):"");
    }catch(error){if(error.name!=="AbortError"&&session.current(task))el("analysis-predict-status").textContent=error.message;}
  }
  async function downloadModel(){
    if(!session.result)return;
    const task=session.begin(),button=el("download-analysis-model");
    button.disabled=true;
    try{
      const response=await fetch("/api/model",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({...session.config,model:el("analysis-predict-model").value}),signal:task.signal});
      if(!response.ok){const reply=await response.json();throw new Error(reply.error);}
      const blob=await response.blob();if(!session.current(task))return;
      saveFile(blob,"regression_model.joblib","application/octet-stream");
    }catch(error){if(error.name!=="AbortError"&&session.current(task))el("analysis-predict-status").textContent=error.message;}
    finally{button.disabled=false;}
  }
  function initAnalysis(){
    SelectionExplorer.init(()=>({...session.payload,target:el("target-select").value,features:selected('input[name="analysis-feature"]')}),()=>session.preview);
    el("regression-model-options").innerHTML=names.map(name=>'<label><input type="checkbox" name="analysis-model" value="'+e(name)+'" checked><span>'+e(name)+'</span></label>').join("");
    el("dataset-select").addEventListener("change",switchDataset);
    el("dataset-upload").addEventListener("change",upload);
    el("target-select").addEventListener("change",()=>{hideResults("Target changed. Run analysis for the new configuration.");renderFeatures();renderPreview();});
    el("input-feature-options").addEventListener("change",()=>{hideResults("Input features changed. Run analysis to update the results.");preparationText();});
    el("regression-model-options").addEventListener("change",()=>hideResults("Model selection changed. Run analysis to update the results."));
    el("select-all-features").addEventListener("click",()=>{document.querySelectorAll('input[name="analysis-feature"]').forEach(input=>input.checked=true);hideResults("Input features changed. Run analysis to update the results.");preparationText();});
    el("clear-features").addEventListener("click",()=>{document.querySelectorAll('input[name="analysis-feature"]').forEach(input=>input.checked=false);hideResults("Select at least one input feature.");preparationText();});
    el("run-analysis").addEventListener("click",runAnalysis);
    el("retry-dataset").addEventListener("click",()=>loadPreview(el("dataset-select").value!=="custom"));
    el("analysis-chart-model").addEventListener("change",renderDiagnostics);
    el("analysis-predict-form").addEventListener("submit",predictInput);
    el("analysis-predict-form").addEventListener("input",()=>{session.pending?.abort();el("analysis-predict-output").textContent="";el("analysis-predict-status").textContent="Inputs changed. Predict again to update.";});
    el("analysis-predict-model").addEventListener("change",()=>{session.pending?.abort();el("analysis-predict-output").textContent="";el("analysis-predict-status").textContent="Model changed. Predict again to update.";});
    el("download-analysis").addEventListener("click",downloadResults);
    el("download-analysis-model").addEventListener("click",downloadModel);
    window.addEventListener("hashchange",setHeader);
    switchDataset();
  }
  initAnalysis();
})();
