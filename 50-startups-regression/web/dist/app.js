"use strict";
const $ = (id) => document.getElementById(id);
const number = (value, digits = 2) => Number(value).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
let data, selectedCount = 3, lastScenario;
const pageNames = {analysis:"Dataset analysis", overview:"Overview", models:"Model comparison", features:"Feature selection", predict:"Predict profit", method:"Methodology"};
const svg = (width,height,content,label) => '<svg class="chart" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+escapeHTML(label)+'">'+content+'</svg>';
const chips = (values) => values.map(v=>'<span>'+escapeHTML(v)+'</span>').join("");

function showPage() {
  const requested = location.hash.slice(1);
  const page = Object.hasOwn(pageNames,requested) ? requested : "analysis";
  document.querySelectorAll(".page").forEach(el=>el.hidden=el.id!==page);
  document.querySelectorAll("nav a").forEach(a=>{
    const active=a.dataset.page===page;
    a.classList.toggle("active",active);
    if(active)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");
  });
  $("breadcrumb").textContent=pageNames[page];
  document.title=pageNames[page]+" · StartupLab";
}
function overview() {
  const best=data.models.find(m=>m.Best_By_CV);
  const cards=[
    ["CV RMSE",number(best.CV_RMSE_Mean),"5-fold outer validation","↘"],
    ["Test R²",number(best.Test_R2,4),"On ten unseen observations","◎"],
    ["Test RMSE",number(best.Test_RMSE),"Error in profit units","⌁"],
    ["Active features",String(best.Feature_Count)+" / 5","Three spending variables","▤"]
  ];
  $("overview-metrics").innerHTML=cards.map(([label,value,note,icon])=>'<article class="metric"><div class="metric-label">'+label+'<span aria-hidden="true">'+icon+'</span></div><div class="metric-value">'+value+'</div><div class="metric-note">'+note+'</div></article>').join("");
  $("winner-name").textContent=best.Model;
  $("winner-params").innerHTML="<span>α "+number(best.alpha,2)+"</span><span>L1 ratio "+number(best.l1_ratio,1)+"</span>";
  $("winner-features").innerHTML=chips(best.Selected_Features.split(" | "));
  $("overview-bars").innerHTML=comparisonBars(data.models);
}
function comparisonBars(models) {
  const sorted=[...models].sort((a,b)=>a.CV_RMSE_Mean-b.CV_RMSE_Mean);
  const w=650,left=177,right=90,top=15,rowHeight=38,h=top+sorted.length*rowHeight+45;
  const max=Math.max(1,...sorted.map(m=>m.CV_RMSE_Mean+m.CV_RMSE_Std))*1.18;
  const scale=v=>left+Math.max(0,v)/max*(w-left-right);
  let parts="";
  sorted.forEach((m,i)=>{
    const y=top+i*rowHeight;
    parts+='<text style="font-size:10px" x="'+(left-12)+'" y="'+(y+16)+'" text-anchor="end">'+escapeHTML(m.Model)+'</text>';
    parts+='<rect x="'+left+'" y="'+y+'" width="'+(scale(m.CV_RMSE_Mean)-left)+'" height="25" rx="3" fill="'+(m.CV_RMSE_Mean===sorted[0].CV_RMSE_Mean?"#27775d":"#d7e5d1")+'"><title>'+escapeHTML(m.Model)+": "+number(m.CV_RMSE_Mean)+" ± "+number(m.CV_RMSE_Std)+'</title></rect>';
    const a=scale(m.CV_RMSE_Mean-m.CV_RMSE_Std),b=scale(m.CV_RMSE_Mean+m.CV_RMSE_Std);
    parts+='<path d="M'+a+' '+(y+12)+'H'+b+' M'+a+' '+(y+8)+'V'+(y+17)+' M'+b+' '+(y+8)+'V'+(y+17)+'" fill="none" stroke="#6b8666" stroke-width="1"/>';
    parts+='<text x="'+(b+8)+'" y="'+(y+16)+'">'+number(m.CV_RMSE_Mean,max>10000?0:2)+'</text>';
  });
  for(let i=0;i<=3;i++){const t=max*i/3;parts+='<text x="'+scale(t)+'" y="'+(h-16)+'" text-anchor="middle">'+(max>10000?number(t/1000,0)+"k":number(t,1))+'</text>';}
  return svg(w,h,parts,"Cross-validation RMSE comparison, with standard-deviation error bars");
}

function modelTable() {
  const key=$("model-sort").value;
  const ordered=[...data.models].sort((a,b)=>key==="Test_R2"?b[key]-a[key]:a[key]-b[key]);
  const min=Math.min(...data.models.map(m=>m.CV_RMSE_Mean));
  $("model-table").innerHTML=ordered.map(m=>'<tr class="'+(m.Best_By_CV?"selected":"")+'"><td><span class="model-label">'+escapeHTML(m.Model)+'</span>'+(m.Best_By_CV?'<span class="best-badge">SAVED WINNER</span>':m.CV_RMSE_Mean===min?'<span class="best-badge">TIED BEST CV</span>':"")+'</td><td>'+m.Feature_Count+'</td><td>'+number(m.CV_RMSE_Mean)+' ± '+number(m.CV_RMSE_Std)+'</td><td>'+number(m.Test_RMSE)+'</td><td>'+number(m.Test_R2,4)+'</td><td>'+number(m.Test_MAE)+'</td></tr>').join("");
}
function scatterChart(rows,residual=false,targetLabel="profit",descriptive=false) {
  targetLabel=escapeHTML(targetLabel);
  const w=540,h=320,left=65,right=20,top=20,bottom=55;
  const xValues=rows.map(r=>residual?r.Predicted_Profit:r.Actual_Profit);
  const yValues=rows.map(r=>residual?r.Residual:r.Predicted_Profit);
  let lowX,highX,lowY,highY;
  if(residual){const min=Math.min(...xValues),max=Math.max(...xValues),span=max-min||1;lowX=min-span*.1;highX=max+span*.1;let bound=Math.max(1, ...yValues.map(Math.abs))*1.2;lowY=-bound;highY=bound;}
  else{const min=Math.min(...xValues,...yValues),max=Math.max(...xValues,...yValues),span=max-min||1;lowX=lowY=min-span*.08;highX=highY=max+span*.08;}
  if(lowX===highX){lowX-=1;highX+=1;}
  if(lowY===highY){lowY-=1;highY+=1;}
  const tick=v=>Math.abs(v)>=10000?number(v/1000,0)+"k":number(v,Math.abs(v)<10?1:0);
  const sx=v=>left+(v-lowX)/(highX-lowX)*(w-left-right);
  const sy=v=>h-bottom-(v-lowY)/(highY-lowY)*(h-top-bottom);
  let parts="";
  for(let i=0;i<=4;i++){
    const xv=lowX+(highX-lowX)*i/4,yv=lowY+(highY-lowY)*i/4;
    parts+='<line class="grid-line" x1="'+left+'" y1="'+sy(yv)+'" x2="'+(w-right)+'" y2="'+sy(yv)+'"/>';
    parts+='<text x="'+(left-9)+'" y="'+(sy(yv)+4)+'" text-anchor="end">'+tick(yv)+'</text>';
    parts+='<text x="'+sx(xv)+'" y="'+(h-bottom+21)+'" text-anchor="middle">'+tick(xv)+'</text>';
  }
  const line=residual?'M'+left+' '+sy(0)+'H'+(w-right):'M'+sx(lowX)+' '+sy(lowX)+'L'+sx(highX)+' '+sy(highX);
  parts+='<path d="'+line+'" stroke="#b48b65" stroke-dasharray="5 5" fill="none" stroke-width="1.4"/>';
  rows.forEach((r,i)=>parts+='<circle tabindex="0" cx="'+sx(xValues[i])+'" cy="'+sy(yValues[i])+'" r="5" fill="#3b896e" stroke="white" stroke-width="1.5"><title>Row '+r.Row_Index+" · Actual: "+number(r.Actual_Profit)+" · Predicted: "+number(r.Predicted_Profit)+" · Residual: "+number(r.Residual)+'</title></circle>');
  if(descriptive&&!residual){
    const meanX=xValues.reduce((a,b)=>a+b,0)/rows.length,meanY=yValues.reduce((a,b)=>a+b,0)/rows.length;
    const variance=xValues.reduce((sum,x)=>sum+(x-meanX)**2,0);
    if(variance>0){const slope=xValues.reduce((sum,x,i)=>sum+(x-meanX)*(yValues[i]-meanY),0)/variance;
      const lo=Math.min(...xValues),hi=Math.max(...xValues);
      parts+='<path d="M'+sx(lo)+' '+sy(meanY+slope*(lo-meanX))+'L'+sx(hi)+' '+sy(meanY+slope*(hi-meanX))+'" stroke="#c47b37" fill="none" stroke-width="2"><title>Descriptive trend, not a validation model</title></path>';
    }
    [...rows].sort((a,b)=>Math.abs(b.Residual)-Math.abs(a.Residual)).slice(0,10).forEach((r,i)=>{
      parts+='<circle cx="'+sx(r.Actual_Profit)+'" cy="'+sy(r.Predicted_Profit)+'" r="9" fill="none" stroke="#a35981" stroke-width="1.5"/><text x="'+(sx(r.Actual_Profit)+9)+'" y="'+(sy(r.Predicted_Profit)-8)+'" font-size="10">'+(i+1)+'</text>';
    });
  }
  parts+='<text class="axis-label" x="'+(w/2)+'" y="'+(h-7)+'" text-anchor="middle">'+(residual?"Predicted "+targetLabel:"Actual "+targetLabel)+'</text>';
  parts+='<text class="axis-label" transform="translate(13 '+h/2+') rotate(-90)" text-anchor="middle">'+(residual?"Actual − predicted":"Predicted "+targetLabel)+'</text>';
  return svg(w,h,parts,residual?"Residuals against predicted target":"Actual versus predicted target with perfect-prediction line");
}
function diagnostics() {
  const rows=data.predictions.filter(r=>r.Model===$("diagnostic-model").value);
  $("scatter-chart").innerHTML=scatterChart(rows);
  $("residual-chart").innerHTML=scatterChart(rows,true);
  $("prediction-table").innerHTML=rows.map(r=>'<tr><td>'+r.Row_Index+'</td><td>'+number(r.Actual_Profit)+'</td><td>'+number(r.Predicted_Profit)+'</td><td>'+number(r.Residual)+'</td></tr>').join("");
}
function featureChart(key,label) {
  const w=540,h=300,left=68,right=25,top=25,bottom=50;
  const values=data.featureCounts.map(r=>r[key]);
  const min=Math.min(...values),max=Math.max(...values),pad=(max-min)*.2;
  const sx=n=>left+(n-1)/4*(w-left-right);
  const sy=v=>h-bottom-(v-(min-pad))/(max-min+pad*2)*(h-top-bottom);
  let parts="";
  for(let i=0;i<=4;i++){
    const v=min-pad+(max-min+pad*2)*i/4;
    parts+='<line class="grid-line" x1="'+left+'" y1="'+sy(v)+'" x2="'+(w-right)+'" y2="'+sy(v)+'"/><text x="'+(left-9)+'" y="'+(sy(v)+4)+'" text-anchor="end">'+number(v,key==="r2"?3:0)+'</text>';
  }
  parts+='<path d="'+data.featureCounts.map((r,i)=>(i?"L":"M")+sx(r.count)+" "+sy(r[key])).join(" ")+'" fill="none" stroke="#3d8466" stroke-width="2.2"/>';
  data.featureCounts.forEach(r=>{
    const active=r.count===selectedCount;
    if(active)parts+='<circle cx="'+sx(r.count)+'" cy="'+sy(r[key])+'" r="11" fill="#deedd7"/>';
    parts+='<circle tabindex="0" cx="'+sx(r.count)+'" cy="'+sy(r[key])+'" r="'+(active?6:4)+'" fill="#34795b"><title>'+r.count+" features · "+label+": "+number(r[key],key==="r2"?6:2)+'</title></circle><text x="'+sx(r.count)+'" y="'+(h-bottom+22)+'" text-anchor="middle">'+r.count+'</text>';
  });
  parts+='<text class="axis-label" x="'+w/2+'" y="'+(h-5)+'" text-anchor="middle">Number of features</text>';
  return svg(w,h,parts,label+" by number of features");
}
function featureView() {
  const selected=data.featureCounts.find(r=>r.count===selectedCount);
  $("feature-buttons").innerHTML=data.featureCounts.map(r=>'<button type="button" data-count="'+r.count+'" class="'+(r.count===selectedCount?"active":"")+'" aria-pressed="'+(r.count===selectedCount)+'">'+r.count+'</button>').join("");
  $("feature-rmse-chart").innerHTML=featureChart("rmse","Mean CV RMSE");
  $("feature-r2-chart").innerHTML=featureChart("r2","Mean CV R-squared");
  $("selected-feature-heading").textContent=selectedCount+" selected feature"+(selectedCount===1?"":"");
  $("selected-feature-chips").innerHTML=chips(selected.features);
  $("feature-table").innerHTML=data.featureCounts.map(r=>'<tr class="'+(r.count===selectedCount?"selected":"")+'"><td>'+r.count+'</td><td>'+escapeHTML(r.features.join(", "))+'</td><td>'+number(r.rmse,6)+'</td><td>'+number(r.r2,6)+'</td></tr>').join("");
}
function predictFormula(model, input) {
  const values={"R&D Spend":input["R&D Spend"],"Administration":input.Administration,"Marketing Spend":input["Marketing Spend"],"State_Florida":input.State==="Florida"?1:0,"State_New York":input.State==="New York"?1:0};
  return model.intercept+model.weights.reduce((sum,w,i)=>sum+w*values[model.features[i]],0);
}
function calculatePrediction() {
  const fields=[["rd","R&D Spend"],["admin","Administration"],["marketing","Marketing Spend"]];
  const input={State:$("state").value};
  $("form-error").textContent="";
  fields.forEach(([id])=>$(id).removeAttribute("aria-invalid"));
  for(const [id,name] of fields){
    const raw=$(id).value.trim(),value=Number(raw);
    if(raw===""||!Number.isFinite(value)||value<0){
      $("form-error").textContent="Enter a valid, non-negative value for "+name+".";
      $(id).setAttribute("aria-invalid","true");$(id).focus();$("download-scenario").disabled=true;return false;
    }
    input[name]=value;
  }
  const key=$("predict-model").value, model=data.predictionModels[key], value=predictFormula(model,input);
  if(!Number.isFinite(value)){ $("form-error").textContent="These inputs are too large. Please enter smaller values.";return false; }
  const metric=data.models.find(m=>m.Model===model.name);
  $("profit-value").textContent=number(value);
  $("prediction-model-label").textContent=model.name+" · fitted on 40 training observations";
  $("prediction-rmse").textContent=number(metric.Test_RMSE);
  $("prediction-r2").textContent=number(metric.Test_R2,4);
  const values=[input["R&D Spend"],input.Administration,input["Marketing Spend"],input.State==="Florida"?1:0,input.State==="New York"?1:0];
  const contributions=[["Intercept",model.intercept],...model.features.map((name,i)=>[name,model.weights[i]*values[i]])];
  const maximum=Math.max(...contributions.map(([,v])=>Math.abs(v)),1);
  $("contributions").innerHTML=contributions.map(([name,v])=>'<div class="contribution-row"><span>'+escapeHTML(name)+'</span><div class="contribution-track"><i class="'+(v<0?"negative":"")+'" style="width:'+(Math.abs(v)/maximum*100)+'%"></i></div><strong>'+(v>=0?"+":"−")+number(Math.abs(v))+'</strong></div>').join("");
  lastScenario={model:model.name,...input,predicted_profit:value};
  $("download-scenario").disabled=false;return true;
}
function validateExport() {
  for(const [key,cases] of Object.entries(data.validationCases)){
    for(const item of cases){
      const actual=predictFormula(data.predictionModels[key],item.input);
      if(Math.abs(actual-item.expected)>1e-7)throw new Error("Model verification failed. Please export the model data again.");
    }
  }
}
function methodCards() {
  const phases=[
    ["01","Business understanding","Predict profit from R&D, administration, marketing spend, and state. The goal is to study associations and predictive performance."],
    ["02","Data understanding","50 rows, four original inputs, and Profit as the target. Inspect distributions, missing values, and numerical correlations."],
    ["03","Data preparation","Split 80/20 with random_state=42. One-hot encode State, drop California as the reference, and fit preprocessing within training data."],
    ["04","Modeling","Compare OLS, exhaustive selection, ElasticNetCV, forward SFS, RFECV, and LassoCV with training-only selection."],
    ["05","Evaluation","Rank methods by five-fold outer CV RMSE. Refit on training rows and report MAE, MSE, RMSE, and R² on the shared test set."],
    ["06","Deployment","Save and reload the trained pipelines. This app reproduces their linear predictions and lets you explore the verified results."]
  ];
  $("method-cards").innerHTML=phases.map(([n,title,text])=>'<article class="method-card"><div class="method-number">'+n+'</div><h3>'+title+'</h3><p>'+text+'</p></article>').join("");
}
async function init() {
  try {
    const response=await fetch("data/project.json");
    if(!response.ok)throw new Error("The experiment data could not be loaded.");
    data=await response.json();
    validateExport();
    overview();modelTable();methodCards();
    $("diagnostic-model").innerHTML=data.models.map(m=>'<option>'+escapeHTML(m.Model)+'</option>').join("");
    $("diagnostic-model").value=data.metadata.best_model;
    diagnostics();featureView();calculatePrediction();
    $("model-sort").addEventListener("change",modelTable);
    $("diagnostic-model").addEventListener("change",diagnostics);
    $("feature-buttons").addEventListener("click",e=>{const button=e.target.closest("[data-count]");if(button){selectedCount=Number(button.dataset.count);featureView();}});
    $("predict-form").addEventListener("submit",e=>{e.preventDefault();calculatePrediction();});
    $("predict-model").addEventListener("change",calculatePrediction);
    $("predict-form").addEventListener("input",()=>{$("download-scenario").disabled=true;$("prediction-model-label").textContent="Inputs changed · select Predict profit to update.";});
    document.querySelectorAll("[data-preset]").forEach(button=>button.addEventListener("click",()=>{
      const values=button.dataset.preset==="lean"?[50000,80000,100000]:[150000,120000,300000];
      ["rd","admin","marketing"].forEach((id,i)=>$(id).value=values[i]);$("state").value="New York";calculatePrediction();
    }));
    $("download-scenario").addEventListener("click",()=>{
      if(!lastScenario)return;
      const blob=new Blob([JSON.stringify(lastScenario,null,2)],{type:"application/json"});
      const url=URL.createObjectURL(blob), a=document.createElement("a");
      a.href=url;a.download="startup-profit-prediction.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });
    $("view-report").addEventListener("click",()=>$("report-dialog").showModal());
    $("close-report").addEventListener("click",()=>$("report-dialog").close());
    $("report-dialog").addEventListener("click",e=>{if(e.target===$("report-dialog"))$("report-dialog").close();});
    window.addEventListener("hashchange",()=>{showPage();window.scrollTo(0,0);});
    $("load-state").hidden=true;showPage();
  } catch(error) {
    $("load-state").textContent=error.message+" Refresh the page or restart the local preview.";
    $("load-state").setAttribute("role","alert");
    console.error(error);
  }
}
init();
