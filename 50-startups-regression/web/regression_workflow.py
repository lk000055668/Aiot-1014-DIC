"""Reusable CRISP-DM regression workflow for built-in and uploaded CSV datasets."""
from functools import lru_cache
from hashlib import sha256
from io import StringIO, BytesIO
import csv
import json
import socket
import warnings
from urllib.error import URLError

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

STARTUP_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)
BOSTON_URL = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
BOSTON_COLUMNS = ["crim", "zn", "indus", "chas", "nox", "rm", "age",
                  "dis", "rad", "tax", "ptratio", "b", "lstat", "medv"]
SPECS = {
    "startups": {"name": "50 Startups Dataset", "url": STARTUP_URL,
                 "target": "Profit", "rows": 50,
                 "columns": ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]},
    "boston": {"name": "Boston Housing Dataset", "url": BOSTON_URL,
               "target": "medv", "rows": 506, "columns": BOSTON_COLUMNS},
}
MODEL_NAMES = ["Multiple Linear Regression", "Polynomial Regression",
               "Decision Tree Regression", "Random Forest Regression",
               "Support Vector Regression", "K-Nearest Neighbors Regression"]
MAX_CSV_BYTES = 1_000_000
MAX_ROWS = 2000
MAX_COLUMNS = 50
MAX_ENCODED_FEATURES = 60


class AnalysisError(ValueError):
    """An actionable validation error that may safely be returned to the UI."""


def build_preprocessing(numeric_features, categorical_features):
    """Shared by the original OLS project and all interactive regression models."""
    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]), list(numeric_features)))
    if categorical_features:
        # Keep the original transformer name so existing OLS inspection code works.
        transformers.append(("state", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_features)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def normalize(data):
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    if data.columns.duplicated().any() or any(not name for name in data.columns):
        raise AnalysisError("CSV column names must be nonempty and unique.")
    if len(data) == 0 or len(data.columns) < 2:
        raise AnalysisError("Provide a CSV with data rows, at least one input, and one target.")
    if len(data) > MAX_ROWS or len(data.columns) > MAX_COLUMNS:
        raise AnalysisError(f"Interactive analysis supports up to {MAX_ROWS} rows and {MAX_COLUMNS} columns.")
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]) and not pd.api.types.is_bool_dtype(data[column]):
            data[column] = data[column].replace([np.inf, -np.inf], np.nan)
        else:
            values = data[column].astype(object).map(
                lambda value: np.nan if pd.isna(value) or str(value).strip() == "" else str(value).strip()
            )
            numeric = pd.to_numeric(values, errors="coerce")
            if values.notna().any() and numeric[values.notna()].notna().all():
                data[column] = numeric.replace([np.inf, -np.inf], np.nan)
            else:
                data[column] = values
    return data


@lru_cache(maxsize=2)
def load_builtin(dataset):
    if dataset not in SPECS:
        raise AnalysisError("Choose a supported dataset.")
    spec = SPECS[dataset]
    previous_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(25)
        data = pd.read_csv(spec["url"])
    except (OSError, URLError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AnalysisError("Dataset download failed. Check internet access and retry.") from exc
    finally:
        socket.setdefaulttimeout(previous_timeout)
    if data.shape != (spec["rows"], len(spec["columns"])) or list(data.columns) != spec["columns"]:
        raise AnalysisError(f"{spec['name']} does not match the expected row count or column schema.")
    data = normalize(data)
    if dataset == "boston" and any(not pd.api.types.is_numeric_dtype(data[c]) for c in BOSTON_COLUMNS):
        raise AnalysisError("Boston Housing must contain 14 numerical columns, including medv.")
    if not pd.api.types.is_numeric_dtype(data[spec["target"]]):
        raise AnalysisError("The default target must be numerical.")
    return data


def load_custom(text):
    if not isinstance(text, str) or not text.strip():
        raise AnalysisError("Choose a nonempty UTF-8 CSV file.")
    if len(text.encode("utf-8")) > MAX_CSV_BYTES:
        raise AnalysisError("CSV upload must be no larger than 1 MB.")
    text = text.lstrip("\ufeff")
    try:
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader)
        cleaned = [column.strip() for column in header]
        if not cleaned or len(set(cleaned)) != len(cleaned) or any(not c for c in cleaned):
            raise AnalysisError("CSV column names must be nonempty and unique.")
        for row in reader:
            if len(row) > len(header):
                raise AnalysisError("A CSV row contains more fields than the header.")
        data = pd.read_csv(StringIO(text))
    except (csv.Error, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AnalysisError("Invalid CSV. Check commas, quoting, and the header row.") from exc
    return normalize(data)


def load_request(payload):
    dataset = payload.get("dataset")
    if not isinstance(dataset, str):
        raise AnalysisError("Choose a supported dataset.")
    if dataset == "custom":
        data = load_custom(payload.get("csv"))
        name = str(payload.get("filename", "Custom Dataset"))[:100]
        source = "User-uploaded CSV"
        target = None
        identity = sha256(payload["csv"].encode()).hexdigest()
    elif dataset in SPECS:
        data = load_builtin(dataset).copy()
        name, source, target = SPECS[dataset]["name"], SPECS[dataset]["url"], SPECS[dataset]["target"]
        identity = dataset
    else:
        raise AnalysisError("Choose 50 Startups, Boston Housing, or Upload Custom Dataset.")
    return data, {"id": identity, "kind": dataset, "name": name, "source": source, "defaultTarget": target}


def frame_records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=12))


def preview_dataset(payload):
    data, info = load_request(payload)
    columns = []
    for name in data.columns:
        numeric = pd.api.types.is_numeric_dtype(data[name])
        row = {"name": name, "dtype": str(data[name].dtype),
               "kind": "numeric" if numeric else "categorical",
               "missing": int(data[name].isna().sum()), "unique": int(data[name].nunique())}
        if numeric:
            for stat, value in data[name].describe().items():
                row[stat] = None if not np.isfinite(value) else float(value)
        else:
            counts = data[name].value_counts()
            row["top"] = None if counts.empty else str(counts.index[0])
        columns.append(row)
    numeric = data.select_dtypes(include="number")
    return {
        "dataset": info, "rows": len(data), "columnCount": len(data.columns),
        "columns": columns, "preview": frame_records(data.head(10)),
        "correlations": {"columns": numeric.columns.tolist(),
                         "values": json.loads(numeric.corr().to_json(orient="values"))},
        "missingValues": int(data.isna().sum().sum()),
        "preprocessing": {"numeric": "Training median imputation",
                          "categorical": "Training mode imputation + one-hot encoding (drop first, ignore unknown)"},
        "modelOptions": MODEL_NAMES,
    }


def prepare_analysis(payload):
    data, info = load_request(payload)
    target, features = payload.get("target"), payload.get("features")
    if not isinstance(target, str) or target not in data.columns:
        raise AnalysisError("Select a target variable from this dataset.")
    if not isinstance(features, list) or not features or any(not isinstance(f, str) for f in features):
        raise AnalysisError("Select at least one input feature.")
    if len(features) != len(set(features)) or target in features or any(f not in data.columns for f in features):
        raise AnalysisError("Inputs must be unique dataset columns and must exclude the target.")
    values = pd.to_numeric(data[target], errors="coerce")
    invalid = data[target].notna() & values.isna()
    if invalid.any():
        raise AnalysisError("Regression requires a numerical target. The selected column contains text.")
    values = values.replace([np.inf, -np.inf], np.nan)
    keep = values.notna()
    removed = int((~keep).sum())
    X, y = data.loc[keep, features].copy(), values.loc[keep].astype(float)
    if len(y) < 20:
        raise AnalysisError("At least 20 rows with a valid numerical target are required for 80/20 splitting and five-fold CV.")
    if y.nunique() < 2:
        raise AnalysisError("The target is constant; choose a target with variation.")
    numeric = [f for f in features if pd.api.types.is_numeric_dtype(X[f])]
    categorical = [f for f in features if f not in numeric]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessing = build_preprocessing(numeric, categorical)
    fitted = preprocessing.fit(train_X)
    encoded_names = fitted.get_feature_names_out().tolist()
    if not encoded_names:
        raise AnalysisError("Encoding produced no input features. Select a varying numerical or categorical input.")
    if len(encoded_names) > MAX_ENCODED_FEATURES:
        raise AnalysisError(f"Encoding creates {len(encoded_names)} inputs; select fewer columns or lower-cardinality categories (maximum {MAX_ENCODED_FEATURES}).")
    models = payload.get("models", MODEL_NAMES)
    if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models) or len(models) != len(set(models)) or any(m not in MODEL_NAMES for m in models):
        raise AnalysisError("Select one or more supported regression models.")
    settings = {"target": target, "features": features, "models": models, "randomState": 42,
                "testSize": 0.2, "folds": 5}
    fingerprint = sha256(json.dumps({"dataset": info["id"], **settings}, sort_keys=True).encode()).hexdigest()
    return {"dataset": info, "settings": settings, "fingerprint": fingerprint,
            "train_X": train_X, "test_X": test_X, "train_y": train_y, "test_y": test_y,
            "numeric": numeric, "categorical": categorical, "encoded": encoded_names,
            "preprocessing": preprocessing, "removed": removed}


def make_model(name, preprocessing):
    steps = [("preprocessing", preprocessing)]
    if name in ("Polynomial Regression", "Support Vector Regression", "K-Nearest Neighbors Regression"):
        steps.append(("scaling", StandardScaler()))
    if name == "Multiple Linear Regression":
        estimator = LinearRegression()
    elif name == "Polynomial Regression":
        steps.extend([("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                      ("polynomial_scaling", StandardScaler())])
        estimator = LinearRegression()
    elif name == "Decision Tree Regression":
        estimator = DecisionTreeRegressor(max_depth=6, min_samples_leaf=2, random_state=42)
    elif name == "Random Forest Regression":
        estimator = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1)
    elif name == "Support Vector Regression":
        # Target scaling makes C and epsilon comparable across target units.
        estimator = TransformedTargetRegressor(regressor=SVR(C=10, epsilon=0.1, kernel="rbf"),
                                               transformer=StandardScaler())
    elif name == "K-Nearest Neighbors Regression":
        estimator = KNeighborsRegressor(n_neighbors=5, weights="distance")
    else:
        raise AnalysisError("Unsupported model.")
    steps.append(("regression", estimator))
    return Pipeline(steps)


def fresh_model(context, name):
    return make_model(name, build_preprocessing(context["numeric"], context["categorical"]))


def analyze(payload):
    context = prepare_analysis(payload)
    X, y, test_X, test_y = (context[key] for key in ("train_X", "train_y", "test_X", "test_y"))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(X, y))
    reports, predictions, fitted_models, errors = [], [], {}, []
    messages = []
    if context["removed"]:
        messages.append(f"Dropped {context['removed']} rows with missing target values; targets are not imputed.")
    for name in context["settings"]["models"]:
        try:
            model = fresh_model(context, name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = cross_validate(model, X, y, cv=splits,
                    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                    error_score="raise", n_jobs=1)
                model.fit(X, y)
            for warning in caught:
                message = str(warning.message)
                if message not in messages:
                    messages.append(message)
            rmse = -scores["test_rmse"]
            if not np.isfinite(rmse).all():
                raise AnalysisError("Cross-validation returned non-finite values.")
            fitted_models[name] = model
            # Freeze model selection scores before evaluating the test set below.
            reports.append({"Model": name, "Selected_Features": " | ".join(context["encoded"]),
                            "Feature_Count": len(context["encoded"]),
                            "CV_RMSE_Mean": float(rmse.mean()), "CV_RMSE_Std": float(rmse.std()),
                            "CV_R2_Mean": float(scores["test_r2"].mean()),
                            "CV_RMSE_Folds": rmse.tolist()})
        except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError) as exc:
            errors.append({"model": name, "error": str(exc)})
    if not reports:
        raise AnalysisError("No selected model could fit this configuration. " + "; ".join(e["error"] for e in errors))
    best = min(reports, key=lambda row: row["CV_RMSE_Mean"])["Model"]
    for report in reports:
        name, model = report["Model"], fitted_models[report["Model"]]
        predicted = model.predict(test_X)
        if not np.isfinite(predicted).all():
            raise AnalysisError(f"{name} produced non-finite predictions.")
        mse = mean_squared_error(test_y, predicted)
        report.update({"Test_R2": float(r2_score(test_y, predicted)),
                       "Test_MAE": float(mean_absolute_error(test_y, predicted)),
                       "Test_MSE": float(mse), "Test_RMSE": float(np.sqrt(mse)),
                       "Train_R2": float(model.score(X, y)), "Best_By_CV": name == best})
        if "polynomial" in model.named_steps:
            report["Expanded_Feature_Count"] = int(model.named_steps["polynomial"].n_output_features_)
        predictions.extend({"Model": name, "Row_Index": int(index), "Actual_Profit": float(actual),
                            "Predicted_Profit": float(value), "Residual": float(actual-value)}
                           for index, actual, value in zip(test_X.index, test_y, predicted))
    example = {}
    for column in context["settings"]["features"]:
        if column in context["numeric"]:
            value = X[column].median()
            example[column] = 0.0 if pd.isna(value) else float(value)
        else:
            mode = X[column].dropna().mode()
            example[column] = "" if mode.empty else str(mode.iloc[0])
    buffer = BytesIO()
    joblib.dump(fitted_models[best], buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(restored.predict(test_X), fitted_models[best].predict(test_X))
    preprocessing = {
        "numericFeatures": context["numeric"], "categoricalFeatures": context["categorical"],
        "encodedFeatures": context["encoded"], "missingTargetsDropped": context["removed"],
        "numericImputation": "Training median", "categoricalImputation": "Training mode",
        "encoding": 'OneHotEncoder(drop="first", handle_unknown="ignore")',
        "scaling": "StandardScaler inside polynomial, SVR and KNN pipelines; SVR target scaled inside the estimator",
    }
    phases = [
        {"name": "Business Understanding", "description": f"Predict {context['settings']['target']} from {len(context['settings']['features'])} selected inputs in {context['dataset']['name']}."},
        {"name": "Data Understanding", "description": f"{len(X)+len(test_X)} valid target rows. Inspect the preview, column statistics and correlations before interpreting results."},
        {"name": "Data Preparation", "description": f"{len(X)} training / {len(test_X)} testing rows; {len(context['encoded'])} encoded inputs. All imputation, encoding and scaling fit within training folds."},
        {"name": "Modeling", "description": f"Fit {len(reports)} selected regression models using fixed, declared settings. No test-based parameter tuning."},
        {"name": "Evaluation", "description": "Rank by five-fold training CV RMSE, then evaluate all models on the same holdout using R², MAE, MSE and RMSE."},
        {"name": "Deployment", "description": f"{best} is the CV winner. Its complete preprocessing/model pipeline passed a save/reload prediction check and can be downloaded."},
    ]
    return {"dataset": context["dataset"], "settings": context["settings"], "fingerprint": context["fingerprint"],
            "models": reports, "predictions": predictions, "bestModel": best,
            "split": {"trainRows": len(X), "testRows": len(test_X),
                      "trainingIndices": X.index.tolist(), "testingIndices": test_X.index.tolist(),
                      "cvFolds": [{"train": X.index[a].tolist(), "validation": X.index[b].tolist()} for a,b in splits]},
            "preprocessing": preprocessing, "workflow": phases, "warnings": messages, "errors": errors,
            "exampleInput": example, "deployment": {"reloadVerified": True, "model": best}}


def fit_for_deployment(payload):
    context = prepare_analysis(payload)
    name = payload.get("model")
    if name not in context["settings"]["models"]:
        raise AnalysisError("Choose a model from the analysis configuration.")
    model = fresh_model(context, name).fit(context["train_X"], context["train_y"])
    return context, model


def input_frame(context, sample):
    if not isinstance(sample, dict) or any(f not in sample for f in context["settings"]["features"]):
        raise AnalysisError("Provide every selected input feature.")
    row = {}
    for feature in context["settings"]["features"]:
        value = sample[feature]
        if feature in context["numeric"]:
            if value is None or value == "":
                row[feature] = np.nan
            else:
                try:
                    row[feature] = float(value)
                except (ValueError, TypeError):
                    raise AnalysisError(f"{feature} must be numerical.")
                if not np.isfinite(row[feature]):
                    raise AnalysisError(f"{feature} must be finite.")
        else:
            row[feature] = np.nan if value is None or value == "" else str(value)
    return pd.DataFrame([row])


def predict(payload):
    context, model = fit_for_deployment(payload)
    sample = input_frame(context, payload.get("input"))
    with warnings.catch_warnings(record=True) as caught:
        value = float(model.predict(sample)[0])
    if not np.isfinite(value):
        raise AnalysisError("Prediction was not finite. Check the input values.")
    return {"prediction": value, "model": payload["model"], "target": context["settings"]["target"],
            "fingerprint": context["fingerprint"], "warnings": list(dict.fromkeys(str(w.message) for w in caught))}


def export_pipeline(payload):
    context, model = fit_for_deployment(payload)
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue(), "regression_model.joblib"
