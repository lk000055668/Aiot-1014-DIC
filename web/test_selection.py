"""HTTP feature parity tests, plus independent statsmodels and leakage checks.
Run from web/: python test_selection.py
Optional independent verification deps: pip install statsmodels flask
"""
from pathlib import Path
from tempfile import gettempdir
import importlib.util
import itertools
import json
import sys
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

# Temporary audit dependencies are optional and do not enter app requirements.
sys.path.append(str(Path(gettempdir())/'regression-audit-testdeps'))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from threadpoolctl import threadpool_limits
from test_platform import call
import regression_workflow as workflow
import selection_workflow as selection


class SelectionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(63)
        n = 70
        data = pd.DataFrame({'spending':rng.normal(40,7,n),'age':rng.normal(6,2,n),
            'region':np.resize(['east','west','north'],n)})
        data['outcome'] = 7+2*data.spending-.7*data.age+3*(data.region=='west')+rng.normal(0,2,n)
        data.loc[2,'spending'] = np.nan
        data.loc[3,'region'] = None
        data.loc[[4,5],'outcome'] = np.nan
        cls.data = data
        cls.configs = {
            'startups': {'dataset':'startups','target':'Profit','features':workflow.SPECS['startups']['columns'][:-1]},
            'boston': {'dataset':'boston','target':'medv','features':workflow.BOSTON_COLUMNS[:-1]},
            'custom': {'dataset':'custom','csv':data.to_csv(index=False),'filename':'parity.csv','target':'outcome','features':['spending','age','region']},
        }
        cls.results = {}
        cls.timings = {}
        for name, config in cls.configs.items():
            begin = time.monotonic()
            cls.results[name] = call('selection',config)
            cls.timings[name] = time.monotonic()-begin
            print(name, len(cls.results[name]['results']), 'evaluated configurations', round(cls.timings[name],2),'seconds',flush=True)
        Path('selection_verification.json').write_text(json.dumps({name:{'maxK':r['maxK'],'configurations':len(r['results']),
            'best':r['best'],'skipped':r['skipped'],'seconds':cls.timings[name]} for name,r in cls.results.items()},indent=2))

    def test_all_algorithms_counts_and_metrics(self):
        for name,result in self.results.items():
            self.assertEqual(set(r['algorithm'] for r in result['results']),set(selection.ALGORITHMS))
            for algorithm in selection.ALGORITHMS[:-1]:
                self.assertEqual({r['k'] for r in result['results'] if r['algorithm']==algorithm},set(range(1,result['maxK']+1)))
            for row in result['results']:
                self.assertEqual(len(row['features']),row['k'])
                self.assertEqual(len(set(row['features'])),row['k'])
                actual = np.array([r['Actual_Profit'] for r in row['predictions']])
                predicted = np.array([r['Predicted_Profit'] for r in row['predictions']])
                self.assertAlmostEqual(row['mse'],np.mean((actual-predicted)**2),places=6)
                self.assertAlmostEqual(row['rmse'],np.sqrt(row['mse']),places=8)
                self.assertAlmostEqual(row['mae'],np.mean(abs(actual-predicted)),places=8)
                self.assertAlmostEqual(row['r2'],r2_score(actual,predicted),places=10)
                self.assertAlmostEqual(row['cvRMSE'],np.mean([f['rmse'] for f in row['cvFolds']]),places=8)
                self.assertEqual(row['outliers'],sorted(row['predictions'],key=lambda r:-r['absoluteError'])[:10])
            winner=min(result['results'],key=lambda r:(r['cvRMSE'],r['algorithm'],r['k']))
            self.assertEqual(result['best'],{'algorithm':winner['algorithm'],'k':winner['k']})
            self.assertEqual(set(result['split']['trainingIndices'])&set(result['split']['testingIndices']),set())
        self.assertEqual(self.results['startups']['maxK'],5)
        self.assertEqual(self.results['boston']['maxK'],13)
        self.assertEqual(self.results['custom']['maxK'],4)

    def test_independent_statistics_equations_and_vif(self):
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        for name,config in self.configs.items():
            c = workflow.prepare_analysis(config)
            pre = c['preprocessing']
            train = pd.DataFrame(pre.transform(c['train_X']),columns=c['encoded'])
            test = pd.DataFrame(pre.transform(c['test_X']),columns=c['encoded'])
            for row in self.results[name]['results']:
                X = train[row['features']].to_numpy()
                design = sm.add_constant(X,has_constant='add')
                fit = sm.OLS(c['train_y'].to_numpy(),design).fit()
                coefficients = row['coefficients']
                np.testing.assert_allclose([v['coefficient'] for v in coefficients],fit.params,rtol=1e-7,atol=1e-7)
                self.assertTrue(row['inferenceValid'])
                for key,expected in [('stderr',fit.bse),('t',fit.tvalues),('p',fit.pvalues),('ciLower',fit.conf_int()[:,0]),('ciUpper',fit.conf_int()[:,1])]:
                    np.testing.assert_allclose([v[key] for v in coefficients],expected,rtol=1e-6,atol=1e-7)
                for j in range(1,len(coefficients)):
                    self.assertAlmostEqual(coefficients[j]['vif'],variance_inflation_factor(design,j),places=5)
                summary=row['summary']
                np.testing.assert_allclose([summary['F'],summary['F_p'],summary['AIC'],summary['BIC'],summary['adjustedTrainR2']],
                    [fit.fvalue,fit.f_pvalue,fit.aic,fit.bic,fit.rsquared_adj],rtol=1e-6,atol=1e-7)
                eq = coefficients[0]['coefficient']+test[row['features']].to_numpy()@np.array([c['coefficient'] for c in coefficients[1:]])
                np.testing.assert_allclose(eq,[p['Predicted_Profit'] for p in row['predictions']],rtol=1e-9,atol=1e-7)

    def test_predictions_contributions_all_algorithms_all_datasets(self):
        for name,config in self.configs.items():
            for algorithm in selection.ALGORITHMS:
                for k in [1,self.results[name]['maxK']]:
                    result=call('selection-predict',{**config,'algorithm':algorithm,'k':k,'input':self.results[name]['exampleInput']})
                    expected=next(r for r in self.results[name]['results'] if r['algorithm']==algorithm and r['k']==k)
                    self.assertEqual(result['features'],expected['features'])
                    self.assertEqual(result['fingerprint'],self.results[name]['fingerprint'])
                    self.assertAlmostEqual(result['prediction'],sum(c['contribution'] for c in result['contributions']),places=7)
                    np.testing.assert_allclose([c['coefficient'] for c in result['contributions']],[c['coefficient'] for c in expected['coefficients']],rtol=1e-10)
                    context=workflow.prepare_analysis(config)
                    sample=workflow.input_frame(context,self.results[name]['exampleInput'])
                    encoded=context['preprocessing'].transform(sample)[0]
                    values=dict(zip(context['encoded'],encoded))
                    actual=[1]+[values[f] for f in expected['features']]
                    np.testing.assert_allclose([c['value'] for c in result['contributions']],actual)
        config=self.configs['custom']
        r=call('selection-predict',{**config,'algorithm':'Lasso','k':4,'input':{'spending':None,'age':None,'region':'unseen'}})
        self.assertTrue(np.isfinite(r['prediction']))
        self.assertTrue(r['warnings'])

    def test_exhaustive_limits_invalid_requests_and_target_switch(self):
        self.assertEqual({r['k'] for r in self.results['boston']['results'] if r['algorithm']=='Exhaustive'},{1,2,11,12,13})
        self.assertEqual({s['k'] for s in self.results['boston']['skipped']},set(range(3,11)))
        for algorithm,k in [('Exhaustive',6),('RFE',0),('Lasso',14),('Fake',1),('RFE',True),([],1)]:
            with self.assertRaises(HTTPError) as caught:
                call('selection-predict',{**self.configs['boston'],'algorithm':algorithm,'k':k,'input':{}})
            self.assertEqual(caught.exception.code,400)
            self.assertIn('error',json.loads(caught.exception.read()))
            caught.exception.close()
        for patch_config in [{'target':'region'},{'features':[]},{'csv':'a,a\n1,2'}]:
            with self.assertRaises(HTTPError) as caught:
                call('selection',{**self.configs['custom'],**patch_config})
            self.assertEqual(caught.exception.code,400)
            caught.exception.close()
        alternate={**self.configs['custom'],'target':'age','features':['spending','region','outcome']}
        result=call('selection',alternate)
        self.assertEqual(result['target'],'age')
        self.assertNotEqual(result['fingerprint'],self.results['custom']['fingerprint'])
        for r in result['results']:
            self.assertNotIn('age',r['features'])

    def test_no_holdout_influence_and_fold_local_preprocessing(self):
        config=self.configs['custom']
        data=self.data.copy()
        train,test=train_test_split(data.index[data.outcome.notna()],test_size=.2,random_state=42)
        data.loc[test,'outcome']+=10000
        data.loc[test,'spending']=1e7
        data.loc[test,'region']='test_only'
        changed=call('selection',{**config,'csv':data.to_csv(index=False)})
        original=self.results['custom']
        self.assertEqual(changed['best'],original['best'])
        for a,b in zip(original['results'],changed['results']):
            self.assertEqual(a['features'],b['features'])
            self.assertEqual(a['coefficients'],b['coefficients'])
            self.assertEqual(a['cvFolds'],b['cvFolds'])
        # Observe actual fit_transform row indices, including inner partitions.
        from sklearn.compose import ColumnTransformer
        original_fit=ColumnTransformer.fit_transform
        seen=[]
        def record(transformer,X,*args,**kwargs):
            seen.append(set(X.index))
            return original_fit(transformer,X,*args,**kwargs)
        c=workflow.prepare_analysis(config)
        with patch.object(ColumnTransformer,'fit_transform',record),threadpool_limits(1):
            selection.analyze_selection(config)
        self.assertTrue(all(rows<=set(train) for rows in seen))
        self.assertTrue(any(len(rows)<len(train)*.7 for rows in seen))

    def test_singular_design_exact_fit_and_zero_percentage(self):
        X=np.column_stack([np.arange(30),np.arange(30)*2,np.ones(30)])
        y=np.arange(30)*3+np.sin(np.arange(30))
        model=LinearRegression().fit(X,y)
        report=selection.ols_statistics(X,y,model,['x','duplicate','constant'])
        self.assertFalse(report['inferenceValid'])
        self.assertTrue(all(c['p'] is None for c in report['coefficients']))
        self.assertIn('infinite',report['coefficients'][1]['vifStatus'])
        self.assertEqual(report['coefficients'][3]['vifStatus'],'constant predictor')
        exact=selection.ols_statistics(X[:,:1],np.arange(30)*2,LinearRegression().fit(X[:,:1],np.arange(30)*2),['x'])
        self.assertFalse(exact['inferenceValid'])
        with self.assertRaises(workflow.AnalysisError):
            selection.check_time(time.monotonic()-1)

    def test_descriptive_refit_is_separate(self):
        for name,config in self.configs.items():
            response=call('selection-descriptive',{**config,'algorithm':'RFE','k':2})
            context=workflow.prepare_analysis(config)
            X=pd.concat([context['train_X'],context['test_X']]).sort_index()
            y=pd.concat([context['train_y'],context['test_y']]).sort_index()
            encoded=pd.DataFrame(context['preprocessing'].transform(X),columns=context['encoded'])
            values=encoded[response['features']].to_numpy()
            expected=LinearRegression().fit(values,y).predict(values)
            np.testing.assert_allclose(expected,[r['Predicted_Profit'] for r in response['predictions']],rtol=1e-9,atol=1e-7)
            self.assertEqual(len(response['predictions']),len(X))
            self.assertIn('not out-of-sample',response['note'])
            self.assertEqual(response['outliers'],sorted(response['predictions'],key=lambda r:-abs(r['Residual']))[:10])
        # A zero target must produce an undefined percentage, never infinity.
        data=self.data.copy();data['outcome']=data['outcome'].fillna(0);data.loc[0:20,'outcome']=0
        result=call('selection',{**self.configs['custom'],'csv':data.to_csv(index=False)})
        zero_rows=[p for p in result['results'][0]['predictions'] if p['Actual_Profit']==0]
        self.assertTrue(zero_rows)
        self.assertTrue(all(r['percentageError'] is None for r in zero_rows))

    def test_reference_api_and_registry(self):
        ref=Path(gettempdir())/'regression-reference-audit'
        spec=importlib.util.spec_from_file_location('reference_app',ref/'app.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        client=module.app.test_client()
        registry=client.get('/api/models').get_json()
        self.assertEqual(set(registry),set(selection.ALGORITHMS))
        self.assertEqual(len(client.get('/api/summary').get_json()),25)
        for algo in registry:
            for k in range(1,6):
                reply=client.post('/api/predict',json={'algo':algo,'k':k,'rd_spend':100000,'marketing_spend':150000,'admin_spend':120000,'state':'Florida'})
                self.assertEqual(reply.status_code,200)
                body=reply.get_json()
                self.assertAlmostEqual(body['predicted_profit'],sum(c['contribution'] for c in body['breakdown']),places=8)
        self.assertEqual(client.post('/api/predict',json={'algo':'invalid'}).status_code,400)
        self.assertEqual(client.get('/').status_code,200)

if __name__=='__main__':
    unittest.main(verbosity=2)
