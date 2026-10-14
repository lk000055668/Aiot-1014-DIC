// Browser-independent checks against Python-generated reference predictions.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root=path.dirname(fileURLToPath(import.meta.url));
const data=JSON.parse(fs.readFileSync(path.join(root,"dist/data/project.json"),"utf8"));
const html=fs.readFileSync(path.join(root,"dist/index.html"),"utf8");
const script=fs.readFileSync(path.join(root,"dist/app.js"),"utf8");
const ids=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
for(const match of script.matchAll(/\$\("([^"]+)"\)/g)) assert(ids.has(match[1]),"Missing HTML element: "+match[1]);

// A minimal form adapter tests the actual calculation/validation functions,
// without claiming to exercise a real browser or CSS layout.
const elements=new Map([...ids].map(id=>[id,{
  value:"",textContent:"",innerHTML:"",disabled:false,attributes:{},
  setAttribute(name,value){this.attributes[name]=value;},
  removeAttribute(name){delete this.attributes[name];},
  focus(){this.focused=true;}
}]));
const context=vm.createContext({document:{getElementById:id=>elements.get(id)},console});
vm.runInContext(script.replace(/\binit\(\);\s*$/,""),context);
context.fixture=data;
vm.runInContext("data=fixture;",context);
const predict=vm.runInContext("predictFormula",context);
let checked=0;
for(const [key,cases] of Object.entries(data.validationCases)){
  for(const item of cases){
    assert(Math.abs(predict(data.predictionModels[key],item.input)-item.expected)<1e-7);
    checked++;
  }
}
function fill(values,key="best"){
  for(const [id,value] of Object.entries(values))elements.get(id).value=String(value);
  elements.get("predict-model").value=key;
}
fill({rd:150000,admin:120000,marketing:300000,state:"New York"});
assert.equal(vm.runInContext("calculatePrediction()",context),true);
assert.equal(elements.get("profit-value").textContent,"174,433.12");
fill({rd:150000,admin:120000,marketing:300000,state:"New York"},"baseline");
assert.equal(vm.runInContext("calculatePrediction()",context),true);
assert.equal(elements.get("profit-value").textContent,"175,581.62");
fill({rd:"",admin:120000,marketing:300000,state:"New York"});
assert.equal(vm.runInContext("calculatePrediction()",context),false);
assert.match(elements.get("form-error").textContent,/R&D Spend/);
assert(elements.get("download-scenario").disabled);
fill({rd:-1,admin:120000,marketing:300000,state:"New York"});
assert.equal(vm.runInContext("calculatePrediction()",context),false);
fill({rd:0,admin:0,marketing:0,state:"California"});
assert.equal(vm.runInContext("calculatePrediction()",context),true);
assert.equal(elements.get("form-error").textContent,"");
for(const model of data.models){
  context.rows=data.predictions.filter(p=>p.Model===model.Model);
  for(const residual of [false,true]){
    const chart=vm.runInContext("scatterChart(rows,"+residual+")",context);
    assert(!/NaN|undefined/.test(chart));
    assert.equal((chart.match(/<circle /g)||[]).length,10);
  }
}
for(const key of ["rmse","r2"]){
  const chart=vm.runInContext('featureChart("'+key+'","Metric")',context);
  assert(!/NaN|undefined/.test(chart));
}
console.log("PASS: "+checked+" JavaScript predictions match the original Python pipelines.");
console.log("PASS: example predictions, baseline switch, blank/negative validation, and zero inputs.");
console.log("PASS: all referenced element IDs exist; all model diagnostic and feature charts contain finite coordinates.");
console.log("Browser rendering/interaction QA still requires the user's local visual review.");
