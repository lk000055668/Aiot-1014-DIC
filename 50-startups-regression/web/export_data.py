"""Export verified results and equivalent linear prediction formulas for the web."""
from pathlib import Path
import json
import shutil
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))  # Resolve FeatureSubset in persisted pipelines.
DIST = Path(__file__).resolve().parent / "dist"
OUTPUTS = PROJECT / "outputs"

def records(name):
    return json.loads(pd.read_csv(OUTPUTS / name).to_json(orient="records", double_precision=15))

def export_model(filename, name):
    model = joblib.load(PROJECT / filename)
    names = model.named_steps["preprocessing"].get_feature_names_out().tolist()
    indices = np.arange(len(names))
    if "selection" in model.named_steps:
        indices = model.named_steps["selection"].indices_
    reg = model.named_steps["regression"]
    coef = np.asarray(reg.coef_, dtype=float)
    intercept = float(reg.intercept_)
    if "scaling" in model.named_steps:
        scaler = model.named_steps["scaling"]
        intercept -= float(np.dot(coef, scaler.mean_ / scaler.scale_))
        coef = coef / scaler.scale_
    weights = np.zeros(len(names))
    weights[indices] = coef
    result = {"name": name, "features": names, "weights": weights.tolist(), "intercept": intercept}
    rng = np.random.default_rng(42)
    samples = pd.DataFrame({
        "R&D Spend": rng.uniform(0, 180000, 60),
        "Administration": rng.uniform(50000, 180000, 60),
        "Marketing Spend": rng.uniform(0, 480000, 60),
        "State": np.tile(["California", "Florida", "New York"], 20),
    })
    samples.loc[0] = [150000, 120000, 300000, "New York"]
    encoded = model.named_steps["preprocessing"].transform(samples)
    expected = model.predict(samples)
    np.testing.assert_allclose(encoded @ weights + intercept, expected, rtol=1e-12, atol=1e-8)
    fixtures = [{"input": row, "expected": float(value)}
                for row, value in zip(samples.to_dict(orient="records"), expected)]
    return result, fixtures

def main():
    (DIST / "data").mkdir(parents=True, exist_ok=True)
    (DIST / "downloads").mkdir(exist_ok=True)
    metadata = json.loads((OUTPUTS / "run_metadata.json").read_text())
    best, best_tests = export_model("best_startup_profit_model.pkl", metadata["best_model"])
    baseline, baseline_tests = export_model("startup_profit_model.pkl", "Baseline OLS")
    counts = [{
        "count": int(r["Number of Features"]),
        "features": r["Selected Features"].strip("[]").split(", "),
        "rmse": r["RMSE"], "r2": r["R-squared"]
    } for r in records("feature_count_results.csv")]
    data = {
        "models": records("model_comparison.csv"),
        "predictions": records("prediction_results_all_models.csv"),
        "featureCounts": counts,
        "metadata": metadata,
        "predictionModels": {"best": best, "baseline": baseline},
        "validationCases": {"best": best_tests, "baseline": baseline_tests}
    }
    (DIST / "data/project.json").write_text(
        json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
    )
    for name in [
        "model_comparison.csv", "prediction_results_all_models.csv",
        "feature_count_results.csv", "teacher_feature_count_report.png",
        "correlation_matrix.png", "teacher_results.md",
        "nested_cv_results.csv"
    ]:
        shutil.copy2(OUTPUTS / name, DIST / "downloads" / name)
    print("Exported results and 120 verified prediction cases to", DIST)

if __name__ == "__main__":
    main()
