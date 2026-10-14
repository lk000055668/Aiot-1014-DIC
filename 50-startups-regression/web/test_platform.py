"""Integration tests for the actual local API and reusable workflow."""
from io import BytesIO
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import regression_workflow as workflow

URL = "http://127.0.0.1:8000/api/"

def call(operation, payload, binary=False):
    request = Request(URL+operation, data=json.dumps(payload).encode(),
                      headers={"Content-Type":"application/json"})
    with urlopen(request, timeout=90) as response:
        raw = response.read()
        return raw if binary else json.loads(raw)


class PlatformIntegrationTests(unittest.TestCase):
    summaries = {}
    results = {}

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n = 65
        frame = pd.DataFrame({
            "amount": np.arange(n, dtype=float),
            "offset": rng.normal(4, 1, n),
            "group": np.tile(["red", "green", "blue", "red", "blue"], 13),
        })
        frame["target"] = frame.amount * 2.5 + frame.offset * 4 + (frame.group=="blue") * 3
        frame.loc[[1, 8], "target"] = np.nan
        frame.loc[[3, 17, 29], "amount"] = np.nan
        frame.loc[[4, 19], "group"] = None
        cls.custom = {"dataset":"custom", "filename":"mixed.csv", "csv":frame.to_csv(index=False)}
        cls.datasets = [{"dataset":"startups"}, {"dataset":"boston"}, cls.custom]
        for payload in cls.datasets:
            key = payload["dataset"]
            summary = call("dataset", payload)
            cls.summaries[key] = summary
            target = summary["dataset"]["defaultTarget"] or "target"
            features = [c["name"] for c in summary["columns"] if c["name"] != target]
            config = {**payload, "target":target, "features":features, "models":workflow.MODEL_NAMES}
            result = call("analyze", config)
            cls.results[key] = (config, result)
            print(key, "loaded", summary["rows"], "rows; winner", result["bestModel"], flush=True)
        # Store only test summaries (not user uploaded rows or pipelines).
        report = {key: {"rows":cls.summaries[key]["rows"], "columns":cls.summaries[key]["columnCount"],
                        "target":r["settings"]["target"], "bestModel":r["bestModel"],
                        "models":r["models"], "warnings":r["warnings"], "errors":r["errors"]}
                  for key, (_,r) in cls.results.items()}
        Path("verification_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def test_boston_schema(self):
        boston = self.summaries["boston"]
        self.assertEqual((boston["rows"], boston["columnCount"]), (506,14))
        self.assertEqual([c["name"] for c in boston["columns"]],workflow.BOSTON_COLUMNS)
        self.assertEqual(boston["dataset"]["defaultTarget"], "medv")
        self.assertTrue(all(c["kind"]=="numeric" for c in boston["columns"]))

    def test_all_six_models_and_metrics(self):
        for dataset, (config,result) in self.results.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(len(result["models"]),6)
                self.assertEqual(result["errors"],[])
                by_name = {row["Model"]:row for row in result["models"]}
                for name, model in by_name.items():
                    rows = [p for p in result["predictions"] if p["Model"]==name]
                    actual = np.array([p["Actual_Profit"] for p in rows])
                    predicted = np.array([p["Predicted_Profit"] for p in rows])
                    mse = mean_squared_error(actual,predicted)
                    np.testing.assert_allclose(
                        [model["Test_R2"],model["Test_MAE"],model["Test_MSE"],model["Test_RMSE"]],
                        [r2_score(actual,predicted),mean_absolute_error(actual,predicted),mse,np.sqrt(mse)]
                    )
                    scores = np.asarray(model["CV_RMSE_Folds"])
                    self.assertEqual(len(scores),5)
                    np.testing.assert_allclose([model["CV_RMSE_Mean"],model["CV_RMSE_Std"]],
                                               [scores.mean(),scores.std()])
                    self.assertTrue(all(np.isfinite([model[key] for key in
                        ("Test_R2","Test_MAE","Test_MSE","Test_RMSE","CV_RMSE_Mean")])))
                self.assertEqual(result["bestModel"],min(result["models"],key=lambda r:r["CV_RMSE_Mean"])["Model"])

    def test_splits_missing_values_and_original_baseline(self):
        for _,result in self.results.values():
            training=set(result["split"]["trainingIndices"])
            testing=set(result["split"]["testingIndices"])
            self.assertFalse(training & testing)
            validation_rows=set()
            for fold in result["split"]["cvFolds"]:
                train,valid=set(fold["train"]),set(fold["validation"])
                self.assertFalse(train & valid)
                self.assertEqual(train|valid,training)
                self.assertFalse((train|valid)&testing)
                validation_rows|=valid
            self.assertEqual(validation_rows,training)
        custom=self.results["custom"][1]
        self.assertEqual(custom["preprocessing"]["missingTargetsDropped"],2)
        self.assertEqual(custom["split"]["trainRows"]+custom["split"]["testRows"],63)
        baseline=next(m for m in self.results["startups"][1]["models"] if m["Model"]=="Multiple Linear Regression")
        self.assertAlmostEqual(baseline["Test_RMSE"],9055.95732345847,places=6)
        self.assertEqual(baseline["Feature_Count"],5)
        self.assertEqual(self.results["boston"][1]["preprocessing"]["categoricalFeatures"],[])

    def test_prediction_and_downloaded_pipeline(self):
        for key,(config,result) in self.results.items():
            payload={**config,"model":result["bestModel"],"input":result["exampleInput"]}
            predicted=call("predict",payload)
            self.assertEqual(predicted["fingerprint"],result["fingerprint"])
            pipeline=joblib.load(BytesIO(call("model",payload,binary=True)))
            expected=float(pipeline.predict(pd.DataFrame([payload["input"]]))[0])
            self.assertAlmostEqual(predicted["prediction"],expected,places=8)
        config,result=self.results["custom"]
        sample={**result["exampleInput"],"group":"previously-unseen","amount":None}
        response=call("predict",{**config,"model":"Multiple Linear Regression","input":sample})
        self.assertTrue(np.isfinite(response["prediction"]))

    def test_incompatible_requests(self):
        invalid=[
            ("dataset",{"dataset":"custom","csv":"a,a\n1,2\n"}),
            ("dataset",{"dataset":"custom","csv":'a,b\n"unterminated,2\n'}),
            ("dataset",{"dataset":"custom","csv":"a,b\n"}),
            ("dataset",{"dataset":"custom","csv":"a,b\n1,2,3\n"}),
            ("analyze",{**self.results["custom"][0],"target":"group"}),
            ("analyze",{**self.results["boston"][0],"features":["medv"]}),
            ("analyze",{**self.results["boston"][0],"features":[]}),
            ("analyze",{**self.results["boston"][0],"models":["unknown"]}),
            ("analyze",{**self.results["boston"][0],"target":["medv"]}),
            ("dataset",{"dataset":[]}),
            ("analyze",{**self.results["boston"][0],"models":[{}]}),
            ("analyze",{"dataset":"custom","csv":"category,y\n"+"\n".join(f"only,{i}" for i in range(30)),"target":"y","features":["category"]}),
            ("analyze",{"dataset":"custom","csv":"x,y\n1,2\n2,3\n","target":"y","features":["x"]}),
        ]
        for operation,payload in invalid:
            with self.subTest(operation=operation,payload=payload.get("target")):
                with self.assertRaises(HTTPError) as error:
                    call(operation,payload)
                self.assertEqual(error.exception.code,400)
                self.assertIn("error",json.loads(error.exception.read()))
                error.exception.close()

    def test_network_failure_and_schema_error(self):
        workflow.load_builtin.cache_clear()
        with patch.object(workflow.pd,"read_csv",side_effect=OSError("offline")):
            with self.assertRaisesRegex(workflow.AnalysisError,"download failed"):
                workflow.load_builtin("boston")
        with patch.object(workflow.pd,"read_csv",return_value=pd.DataFrame({"wrong":[1]})):
            with self.assertRaisesRegex(workflow.AnalysisError,"schema"):
                workflow.load_builtin("boston")

    def test_training_only_imputation(self):
        frame=pd.DataFrame({"x":[1.,2.,3.,100000.],"category":["a","b","a","new"]})
        preprocessing=workflow.build_preprocessing(["x"],["category"]).fit(frame.iloc[:3])
        imputer=preprocessing.named_transformers_["numeric"].named_steps["imputer"]
        self.assertEqual(imputer.statistics_[0],2.)
        encoder=preprocessing.named_transformers_["state"].named_steps["encoder"]
        self.assertEqual(encoder.categories_[0].tolist(),["a","b"])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(np.isfinite(preprocessing.transform(frame.iloc[3:])).all())

if __name__=="__main__":
    unittest.main(verbosity=2)
