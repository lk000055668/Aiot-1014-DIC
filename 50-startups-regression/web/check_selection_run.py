import json, time
from threadpoolctl import threadpool_limits
from selection_workflow import analyze_selection
from regression_workflow import load_builtin, SPECS
for name in ('startups', 'boston'):
    data = load_builtin(name)
    target = SPECS[name]['target']
    payload = {'dataset':name, 'target':target, 'features':[c for c in data if c != target]}
    begin = time.monotonic()
    with threadpool_limits(1):
        result = analyze_selection(payload)
    with open('selection_'+name+'_evidence.json','w') as f:
        json.dump(result,f,allow_nan=False)
    print(name, 'seconds', time.monotonic()-begin, 'configurations',len(result['results']), 'best', result['best'], flush=True)
