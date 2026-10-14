import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import {fileURLToPath} from "node:url";

const root=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(root,"dist/index.html"),"utf8");
const app=fs.readFileSync(path.join(root,"dist/app.js"),"utf8");
const client=fs.readFileSync(path.join(root,"dist/analysis.js"),"utf8");
const ids=new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
for(const match of client.matchAll(/\bel\("([^"]+)"\)/g)){
  assert(ids.has(match[1]),"Missing control: "+match[1]);
}
new vm.Script(client); // Syntax check the complete client, including UI callbacks.
const context=vm.createContext({AbortController});
vm.runInContext(client.slice(0,client.indexOf("(function(){")),context);
const Session=vm.runInContext("LatestDatasetSession",context);
const state=new Session();
state.select({dataset:"startups"});
const oldStartups=state.begin();
state.preview={dataset:{kind:"startups"}};
state.result={dataset:{kind:"startups"}};
state.select({dataset:"boston"});
assert.equal(state.preview,null);
assert.equal(state.result,null);
assert(oldStartups.signal.aborted);
assert(!state.commit(oldStartups,{dataset:{kind:"startups"}},{dataset:"startups"}));
const boston=state.begin();
assert(state.commit(boston,{dataset:{kind:"boston"}},{dataset:"boston"}));
state.select({dataset:"custom",csv:"x,y\n1,2"});
assert.equal(state.result,null);
assert.equal(state.config,null);
assert(!state.commit(boston,{dataset:{kind:"boston"}},{dataset:"boston"}));
const custom=state.begin();
assert(state.commit(custom,{dataset:{kind:"custom"}},{dataset:"custom",target:"y"}));
const staleConfig=state.begin();
state.invalidate();
assert.equal(state.result,null);
assert(!state.commit(staleConfig,{models:["old"]},{target:"old"}));
const firstRun=state.begin(),secondRun=state.begin();
assert(firstRun.signal.aborted);
assert(!state.current(firstRun));
assert(state.current(secondRun));

const charts=vm.createContext({document:{getElementById:()=>({})}});
vm.runInContext(app.replace(/\binit\(\);\s*$/,""),charts);
charts.models=[
  {Model:"Linear",CV_RMSE_Mean:3.2,CV_RMSE_Std:1.5},
  {Model:"Forest",CV_RMSE_Mean:2.7,CV_RMSE_Std:0.4}
];
let svg=vm.runInContext("comparisonBars(models)",charts);
assert(!/NaN|undefined/.test(svg));
assert(svg.includes("2.70"));
charts.rows=[
  {Row_Index:0,Actual_Profit:-10,Predicted_Profit:-8,Residual:-2},
  {Row_Index:1,Actual_Profit:-5,Predicted_Profit:-7,Residual:2},
  {Row_Index:2,Actual_Profit:0,Predicted_Profit:1,Residual:-1}
];
for(const residual of [true,false]){
  svg=vm.runInContext('scatterChart(rows,'+residual+',"medv")',charts);
  assert(!/NaN|undefined/.test(svg));
  assert(svg.includes("medv"));
}
charts.rows=[{Row_Index:0,Actual_Profit:0,Predicted_Profit:0,Residual:0}];
assert(!/NaN|undefined/.test(vm.runInContext("scatterChart(rows)",charts)));
console.log("PASS: switching Startups → Boston → Custom cancels and rejects every stale response.");
console.log("PASS: feature/target changes clear results; newer requests supersede earlier ones.");
console.log("PASS: all control IDs exist and the complete analysis client parses.");
console.log("PASS: shared charts support housing-scale, negative and constant values.");
console.log("Real browser layout/interaction QA is unavailable in this session.");
