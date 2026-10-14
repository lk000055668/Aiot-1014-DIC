import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {Blob} from 'node:buffer';
import {fileURLToPath} from 'node:url';
const root=path.dirname(fileURLToPath(import.meta.url));
const html=fs.readFileSync(path.join(root,'dist/index.html'),'utf8');
const source=fs.readFileSync(path.join(root,'dist/selection.js'),'utf8');
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(new Set(ids).size,ids.length,'Duplicate HTML IDs');
for(const match of source.matchAll(/\$\('([^']+)'\)/g))assert(ids.includes(match[1]),'Missing selection control '+match[1]);
new vm.Script(source);
const elements=new Map();
class Element {
  constructor(id){this.id=id;this.value='';this.hidden=false;this.disabled=false;this.listeners={};this.textContent='';this._html='';}
  addEventListener(name,callback){this.listeners[name]=callback;}
  get innerHTML(){return this._html;}
  set innerHTML(value){this._html=value;for(const m of value.matchAll(/<input id="([^"]+)"[^>]*value="([^"]*)"/g))get(m[1]).value=m[2];}
  querySelector(selector){if(selector==='svg')return {innerHTML:this._html.match(/<svg[^>]*>([\s\S]*?)<\/svg>/)[1]};throw Error(selector);}
  click(){downloads.push(this);}
  fire(name,event={preventDefault(){}}){return this.listeners[name]?.(event);}
}
function get(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);}
ids.forEach(get);
let requests=[],downloads=[],blobs=[];
const context=vm.createContext({console,AbortController,document:{getElementById:get,createElement:()=>new Element('download')},fetch:(url,options)=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject})),setTimeout:(f)=>f(),URL:{createObjectURL:blob=>{blobs.push(blob);return 'blob:test';},revokeObjectURL(){}},Blob});
vm.runInContext(fs.readFileSync(path.join(root,'dist/app.js'),'utf8').replace(/\binit\(\);\s*$/,''),context);
vm.runInContext(source,context);
const explorer=vm.runInContext('SelectionExplorer',context);
const data=JSON.parse(fs.readFileSync(path.join(root,'selection_startups_evidence.json'),'utf8'));
let config={dataset:'startups',target:'Profit',features:['R&D Spend','Administration','Marketing Spend','State']};
let preview={columns:config.features.map(name=>({name,kind:name==='State'?'categorical':'numeric'}))};
explorer.init(()=>config,()=>preview);
assert.equal(get('run-selection').disabled,false);
get('selection-chart-mode').value='all';
const run=get('run-selection').fire('click');
assert.equal(requests.length,1);
requests[0].resolve({ok:true,json:async()=>data});await run;
assert.equal(get('selection-results').hidden,false);
assert(get('selection-equation').textContent.startsWith('Profit = '));
assert(get('selection-coefficients').innerHTML.includes('Intercept'));
assert(!/NaN|undefined/.test(get('selection-rmse-chart').innerHTML));
assert.equal(get('selection-k').max,5);
get('selection-csv').fire('click');get('selection-json').fire('click');get('selection-svg').fire('click');
assert.equal(downloads.length,3);
assert((await blobs[0].text()).includes('cvRMSE'));
assert.equal(JSON.parse(await blobs[1].text()).results.length,25);
assert((await blobs[2].text()).includes('<svg'));
const description=get('selection-descriptive-run').fire('click');const descriptReq=requests.at(-1);
get('selection-k').value=2;get('selection-k').fire('input');assert(descriptReq.options.signal.aborted);
descriptReq.resolve({ok:true,json:async()=>({fingerprint:data.fingerprint,note:'OLD REFIT',predictions:[],outliers:[]})});await description;
assert.equal(get('selection-descriptive-note').textContent,'');
get('selection-k').value=1;get('selection-k').fire('input');
const one=get('selection-equation').textContent;
get('selection-k').value=4;get('selection-k').fire('input');
assert.notEqual(get('selection-equation').textContent,one);
assert.equal((get('selection-equation').textContent.match(/×/g)||[]).length,4);
get('selection-algorithm').value='Lasso';get('selection-algorithm').fire('change');
assert(get('selection-detail-title').textContent.includes('Lasso'));
// A response arriving after a k change must not display a stale prediction.
const prediction=get('selection-predict-form').fire('submit');const pending=requests.at(-1);
get('selection-k').value=2;get('selection-k').fire('input');
assert(pending.options.signal.aborted);
pending.resolve({ok:true,json:async()=>({prediction:999,fingerprint:data.fingerprint,contributions:[],warnings:[]})});await prediction;
assert.equal(get('selection-prediction').textContent,'');
// Dataset changes clear both the details and contribution output, even if old HTTP completes.
const oldRun=get('run-selection').fire('click');const oldRequest=requests.at(-1);
config={dataset:'boston',target:'medv',features:['rm','lstat']};preview={columns:config.features.map(name=>({name,kind:'numeric'}))};explorer.reset();
assert(oldRequest.options.signal.aborted);assert.equal(get('selection-results').hidden,true);
oldRequest.resolve({ok:true,json:async()=>data});await oldRun;assert.equal(get('selection-results').hidden,true);
const boston=JSON.parse(fs.readFileSync(path.join(root,'selection_boston_evidence.json'),'utf8'));
const newRun=get('run-selection').fire('click');requests.at(-1).resolve({ok:true,json:async()=>boston});await newRun;
assert.equal(get('selection-k').max,13);
assert(get('selection-equation').textContent.startsWith('medv = '));
get('selection-algorithm').value='Exhaustive';get('selection-k').value=6;get('selection-k').fire('input');
assert.equal(get('selection-detail').hidden,true);assert(get('selection-status').textContent.includes('limit'));
config={dataset:'custom',target:'',features:[]};preview=null;explorer.reset();assert(get('run-selection').disabled);assert(get('selection-results').hidden);
// Constant, negative and missing chart points remain finite and correctly labeled.
context.rows=[{algorithm:'Lasso',k:1,rmse:0,r2:-2},{algorithm:'Lasso',k:3,rmse:0,r2:null}];
for(const metric of ['rmse','r2']){
  const svg=vm.runInContext('selectionChart(rows,"'+metric+'",{algorithm:"Lasso",k:1},3,{algorithm:"Lasso",k:1})',context);
  assert(!/NaN|undefined/.test(svg));assert(svg.includes('data-selection-k="1"'));
}
context.attack=[{Row_Index:0,Actual_Profit:0,Predicted_Profit:1,Residual:-1}];
const escaped=vm.runInContext('scatterChart(attack,false,"<img src=x onerror=alert(1)>",true)',context);
assert(!escaped.includes('<img'));assert(escaped.includes('&lt;img'));
console.log('PASS: CSV/JSON/SVG exports, descriptive stale-response cancellation, and uploaded-label escaping.');
console.log('PASS: complete selection controller executes without JS exceptions in a minimal DOM adapter.');
console.log('PASS: k and algorithm changes update equations/tables/charts; Boston max k is 13.');
console.log('PASS: stale analysis and prediction replies are rejected after dataset/k changes.');
console.log('PASS: exhaustive gaps, disabled custom target, HTML IDs, finite chart coordinates.');
console.log('This is a controller test, not a real-browser layout or console certification.');
