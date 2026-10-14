"""Live, training-only feature-selection paths and exploratory OLS inference.

All selectors refit unpenalized OLS. Selection CV is internal to each outer fold.
Candidate encodings/imputers/scalers are learned on each training partition.
"""
from itertools import combinations
from math import comb
from time import monotonic
from hashlib import sha256
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from regression_workflow import AnalysisError, prepare_analysis, build_preprocessing, input_frame

ALGORITHMS = ["Lasso", "Forward SFS", "RFE", "ElasticNet", "Exhaustive"]
MAX_SUBSETS_PER_K = 256
MAX_PATH_FEATURES = 30
TIME_BUDGET = 45


def finite(value):
    return float(value) if np.isfinite(value) else None


def check_time(deadline):
    if monotonic() > deadline:
        raise AnalysisError("Feature selection exceeded the 45-second compute budget. Select fewer inputs and retry.")


def ols_fit(X, y):
    return LinearRegression().fit(X, y)


def subset_score(folds, subset):
    errors = []
    for train, valid, y, truth in folds:
        # Frames align independently fitted fold vocabularies. A category absent
        # from a training fold is a zero column, not a learned validation category.
        a = train.reindex(columns=subset, fill_value=0).to_numpy()
        b = valid.reindex(columns=subset, fill_value=0).to_numpy()
        predicted = ols_fit(a, y).predict(b)
        errors.append(np.sqrt(mean_squared_error(truth, predicted)))
    return float(np.mean(errors))


def training_paths(context, X, y, deadline, algorithms=None):
    algorithms = ALGORITHMS if algorithms is None else algorithms
    pre = build_preprocessing(context["numeric"], context["categorical"])
    values = pre.fit_transform(X)
    names = pre.get_feature_names_out().tolist()
    p = len(names)
    if not p or p > MAX_PATH_FEATURES:
        raise AnalysisError(f"Selection paths support 1–{MAX_PATH_FEATURES} encoded inputs. Reduce the selected input columns.")
    frame = pd.DataFrame(values, columns=names)
    folds, penalized_folds = [], []
    for a, b in KFold(3, shuffle=True, random_state=42).split(X):
        fold_pre = clone(pre)
        train = fold_pre.fit_transform(X.iloc[a])
        valid = fold_pre.transform(X.iloc[b])
        fold_names = fold_pre.get_feature_names_out().tolist()
        if not fold_names:
            # Entire categorical inputs can be constant within a small fold.
            train = np.zeros((len(a), 1)); valid = np.zeros((len(b), 1))
            fold_names = ["__constant_fold__"]
        folds.append((pd.DataFrame(train, columns=fold_names), pd.DataFrame(valid, columns=fold_names), y.iloc[a], y.iloc[b]))
        scale = StandardScaler().fit(train)
        target_scale = StandardScaler().fit(np.asarray(y.iloc[a]).reshape(-1, 1))
        penalized_folds.append((scale.transform(train), scale.transform(valid),
            target_scale.transform(np.asarray(y.iloc[a]).reshape(-1, 1)).ravel(), y.iloc[b], target_scale))
    paths, metadata, skipped = {}, {}, []
    scaled = StandardScaler().fit_transform(values)
    ys = StandardScaler().fit_transform(np.asarray(y).reshape(-1, 1)).ravel()
    for name in [a for a in ("Lasso", "ElasticNet") if a in algorithms]:
        best = None
        ratios = [1.] if name == "Lasso" else [.2, .5, .8]
        for alpha in [.001, .01, .1, 1.]:
            for ratio in ratios:
                check_time(deadline)
                errors = []
                for tx, vx, ty, vy, target_scale in penalized_folds:
                    estimator = Lasso(alpha=alpha, max_iter=20000) if name == "Lasso" else ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=20000)
                    estimator.fit(tx, ty)
                    pred = target_scale.inverse_transform(estimator.predict(vx).reshape(-1, 1)).ravel()
                    errors.append(np.sqrt(mean_squared_error(vy, pred)))
                candidate = (float(np.mean(errors)), alpha, ratio)
                if best is None or candidate < best:
                    best = candidate
        _, alpha, ratio = best
        estimator = Lasso(alpha=alpha, max_iter=20000) if name == "Lasso" else ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=20000)
        estimator.fit(scaled, ys)
        order = sorted(range(p), key=lambda j: (-abs(estimator.coef_[j]), names[j]))
        paths[name] = {k: [names[j] for j in order[:k]] for k in range(1, p+1)}
        metadata[name] = {"alpha": alpha, "l1_ratio": ratio,
            "ranking": [{"feature": names[j], "standardizedCoefficient": float(estimator.coef_[j])} for j in order],
            "note": "Rank by absolute standardized penalized coefficient; zero ties use feature name. Each top-k subset is refitted with OLS."}
    if "Forward SFS" in algorithms:
        selected = []
        paths["Forward SFS"] = {}
        for k in range(1, p+1):
            check_time(deadline)
            remaining = [f for f in names if f not in selected]
            feature = min(remaining, key=lambda f: (subset_score(folds, selected+[f]), f))
            selected = selected+[feature]
            paths["Forward SFS"][k] = selected
    if "RFE" in algorithms:
        remaining = list(range(p))
        paths["RFE"] = {}
        while remaining:
            check_time(deadline)
            paths["RFE"][len(remaining)] = [names[j] for j in remaining]
            coefficients = ols_fit(scaled[:, remaining], y).coef_
            remove = min(range(len(remaining)), key=lambda j: (abs(coefficients[j]), names[remaining[j]]))
            remaining.pop(remove)
    if "Exhaustive" in algorithms:
        paths["Exhaustive"] = {}
        for k in range(1, p+1):
            count = comb(p, k)
            if count > MAX_SUBSETS_PER_K:
                skipped.append({"algorithm": "Exhaustive", "k": k, "candidates": count,
                    "reason": f"{count:,} subsets exceed the {MAX_SUBSETS_PER_K} subsets-per-k interactive limit. Narrow the inputs for exact search."})
                continue
            best = None
            for subset in combinations(names, k):
                check_time(deadline)
                score = subset_score(folds, list(subset))
                if best is None or score < best[0]:
                    best = (score, list(subset))
            paths["Exhaustive"][k] = best[1]
    return pre, frame, paths, metadata, skipped


def ols_statistics(values, y, model, names):
    """Classical OLS inference, conditional on the chosen design; not selective inference."""
    n, p = values.shape
    design = np.column_stack([np.ones(n), values])
    # Scale columns before rank/covariance operations to avoid unit-driven loss.
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0] = 1
    normalized = design/scales
    rank = int(np.linalg.matrix_rank(normalized))
    df = n-rank
    residual = np.asarray(y)-model.predict(values)
    sse = float(residual @ residual)
    tss = float(np.sum((np.asarray(y)-np.mean(y))**2))
    valid = rank == p+1 and df > 0 and sse > np.finfo(float).eps * max(tss, 1.)
    beta = np.r_[model.intercept_, model.coef_]
    if valid:
        pinv = np.linalg.pinv(normalized)
        covariance = (pinv @ pinv.T)/(scales[:, None]*scales[None, :]) * (sse/df)
        stderr = np.sqrt(np.maximum(np.diag(covariance), 0))
        tvalues = np.divide(beta, stderr, out=np.full_like(beta, np.nan), where=stderr>0)
        pvalues = 2*stats.t.sf(np.abs(tvalues), df)
        interval = stats.t.ppf(.975, df)*stderr
    else:
        stderr = tvalues = pvalues = interval = np.full(p+1, np.nan)
    rows = []
    for j, name in enumerate(["Intercept"]+names):
        vif, vif_status = None, "not applicable"
        if j:
            column = values[:, j-1]
            if np.ptp(column) == 0:
                vif_status = "constant predictor"
            elif p == 1:
                vif, vif_status = 1., "finite"
            else:
                others = np.delete(values, j-1, axis=1)
                aux = ols_fit(others, column)
                unexplained = np.sum((column-aux.predict(others))**2)/np.sum((column-column.mean())**2)
                if unexplained <= 1e-12:
                    vif_status = "infinite (perfect collinearity)"
                else:
                    vif, vif_status = float(1/unexplained), "finite"
        rows.append({"feature": name, "coefficient": float(beta[j]), "stderr": finite(stderr[j]),
            "t": finite(tvalues[j]), "p": finite(pvalues[j]),
            "ciLower": finite(beta[j]-interval[j]), "ciUpper": finite(beta[j]+interval[j]),
            "vif": vif, "vifStatus": vif_status})
    r2 = 1-sse/tss if tss > 0 else np.nan
    f = ((tss-sse)/(rank-1))/(sse/df) if valid and rank > 1 else np.nan
    log_likelihood = -.5*n*(np.log(2*np.pi)+1+np.log(sse/n)) if sse > 0 else np.nan
    return {"coefficients": rows, "summary": {"n": n, "rank": rank, "residualDf": df,
        "trainR2": finite(r2), "adjustedTrainR2": finite(1-(1-r2)*(n-1)/df) if df > 0 else None,
        "residualStdError": finite(np.sqrt(sse/df)) if df > 0 else None,
        "F": finite(f), "F_p": finite(stats.f.sf(f, rank-1, df)) if np.isfinite(f) else None,
        "AIC": finite(2*rank-2*log_likelihood), "BIC": finite(np.log(n)*rank-2*log_likelihood),
        "conditionNumber": finite(np.linalg.cond(normalized))},
        "inferenceValid": bool(valid),
        "inferenceNote": ("Classical homoskedastic OLS inference assumes independent errors and a fixed design. After feature selection these intervals and p-values are exploratory, not selection-adjusted or causal."
            if valid else "Inference is unavailable: rank deficiency, insufficient residual degrees of freedom, or effectively exact fit. Predictions use the OLS least-squares solution; individual coefficients may be non-identifiable.")}


def context_for_selection(payload):
    context = prepare_analysis(payload)
    if len(context["encoded"]) > MAX_PATH_FEATURES:
        raise AnalysisError(f"Feature selection supports at most {MAX_PATH_FEATURES} encoded inputs; reduce the input selection. The six-model comparison remains available.")
    return context


def analyze_selection(payload):
    context = context_for_selection(payload)
    X, y = context["train_X"], context["train_y"]
    deadline = monotonic()+TIME_BUDGET
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pre, frame, paths, metadata, skipped = training_paths(context, X, y, deadline)
        p = len(frame.columns)
        cv_results = {(name, k): [] for name in ALGORITHMS for k in paths[name]}
        folds = []
        for a, b in KFold(5, shuffle=True, random_state=42).split(X):
            check_time(deadline)
            fold_pre, fold_frame, fold_paths, _, _ = training_paths(context, X.iloc[a], y.iloc[a], deadline)
            valid = pd.DataFrame(fold_pre.transform(X.iloc[b]), columns=fold_frame.columns)
            folds.append({"train": X.index[a].tolist(), "validation": X.index[b].tolist()})
            for (name, k), scores in cv_results.items():
                effective_k = min(k, len(fold_frame.columns))
                subset = fold_paths[name].get(effective_k)
                if subset is None:
                    continue
                model = ols_fit(fold_frame[subset].to_numpy(), y.iloc[a])
                pred = model.predict(valid[subset].to_numpy())
                scores.append({"rmse": float(np.sqrt(mean_squared_error(y.iloc[b], pred))),
                    "r2": finite(r2_score(y.iloc[b], pred)), "features": subset, "effectiveK": effective_k})
        # Choose the configuration from nested training CV, before any holdout scoring.
        eligible = [(float(np.mean([f["rmse"] for f in scores])), name, k)
                    for (name, k), scores in cv_results.items() if len(scores)==5]
        if not eligible:
            raise AnalysisError("No selection configuration completed all five validation folds.")
        _, best_name, best_k = min(eligible)
        test = pd.DataFrame(pre.transform(context["test_X"]), columns=frame.columns)
        results = []
        for name in ALGORITHMS:
            for k, subset in sorted(paths[name].items()):
                scores = cv_results[(name, k)]
                if len(scores) != 5:
                    skipped.append({"algorithm": name, "k": k, "reason": "Not all outer folds support this exact search within the compute limit."})
                    continue
                model = ols_fit(frame[subset].to_numpy(), y)
                pred = model.predict(test[subset].to_numpy())
                truth = context["test_y"]
                mse = float(mean_squared_error(truth, pred))
                details = ols_statistics(frame[subset].to_numpy(), y, model, subset)
                predictions = [{"Row_Index": int(index), "Actual_Profit": float(actual),
                    "Predicted_Profit": float(value), "Residual": float(actual-value),
                    "absoluteError": float(abs(actual-value)),
                    "percentageError": float(abs(actual-value)/abs(actual)*100) if actual != 0 else None}
                    for index, actual, value in zip(context["test_X"].index, truth, pred)]
                results.append({"algorithm": name, "k": k, "features": subset,
                    "cvRMSE": float(np.mean([s["rmse"] for s in scores])),
                    "cvR2": finite(np.mean([s["r2"] for s in scores if s["r2"] is not None])),
                    "cvFolds": scores, "rmse": float(np.sqrt(mse)), "mse": mse,
                    "mae": float(mean_absolute_error(truth, pred)), "r2": finite(r2_score(truth, pred)),
                    "predictions": predictions,
                    "outliers": sorted(predictions, key=lambda row: -row["absoluteError"])[:10], **details})
        # The all-input OLS baseline is identical for all algorithms at full k.
        baseline = next(row for row in results if row["k"] == p)
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    if context["removed"]:
        messages.append(f"Dropped {context['removed']} rows with missing targets.")
    if any(f["effectiveK"] != row["k"] for row in results for f in row["cvFolds"]):
        messages.append("Some training folds contain fewer encoded categories; their effective k is capped at the fold's available inputs. See fold evidence in the JSON export.")
    example = {}
    for feature in context["settings"]["features"]:
        if feature in context["numeric"]:
            value = X[feature].median()
            example[feature] = float(value) if pd.notna(value) else None
        else:
            mode = X[feature].dropna().mode()
            example[feature] = str(mode.iloc[0]) if len(mode) else None
    return {"dataset": context["dataset"], "target": context["settings"]["target"],
        "fingerprint": context["fingerprint"], "maxK": p, "results": results,
        "baseline": {key: baseline[key] for key in ("k", "features", "rmse", "r2", "mae", "mse")},
        "best": {"algorithm": best_name, "k": best_k}, "rankings": metadata,
        "skipped": skipped, "warnings": messages, "exampleInput": example,
        "split": {"trainingIndices": X.index.tolist(), "testingIndices": context["test_X"].index.tolist(), "cvFolds": folds},
        "methodology": "80/20 seed 42; five outer training folds; three inner selection folds. Preprocessing, scaling and selection refit inside training partitions. Curves are holdout diagnostics; the recommendation uses nested training CV only. k counts encoded predictors, excluding the intercept. Classical post-selection inference is exploratory."}


def predict_selection(payload):
    context = context_for_selection(payload)
    algorithm, k = payload.get("algorithm"), payload.get("k")
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS or type(k) is not int or not 1 <= k <= len(context["encoded"]):
        raise AnalysisError("Choose a supported selection algorithm and an integer k within the encoded feature count.")
    if algorithm == "Exhaustive" and comb(len(context["encoded"]), k) > MAX_SUBSETS_PER_K:
        raise AnalysisError("This exhaustive subset count exceeds the interactive compute limit. Reduce the input features or choose another k.")
    with warnings.catch_warnings(record=True) as caught:
        pre, frame, paths, _, _ = training_paths(context, context["train_X"], context["train_y"], monotonic()+TIME_BUDGET, [algorithm])
        subset = paths[algorithm][k]
        model = ols_fit(frame[subset].to_numpy(), context["train_y"])
        sample = input_frame(context, payload.get("input"))
        encoded = pd.DataFrame(pre.transform(sample), columns=frame.columns)[subset].to_numpy()[0]
        value = float(model.predict(encoded.reshape(1, -1))[0])
    if not np.isfinite(value):
        raise AnalysisError("Prediction is not finite. Check input values.")
    contributions = [{"feature": "Intercept", "value": 1., "coefficient": float(model.intercept_), "contribution": float(model.intercept_)}]
    contributions += [{"feature": name, "value": float(v), "coefficient": float(c), "contribution": float(v*c)} for name, v, c in zip(subset, encoded, model.coef_)]
    return {"prediction": value, "algorithm": algorithm, "k": k, "features": subset,
        "fingerprint": context["fingerprint"], "contributions": contributions,
        "warnings": list(dict.fromkeys(str(w.message) for w in caught))}


def descriptive_selection(payload):
    """Explicit all-row refit for descriptive diagnostics; never used for selection."""
    context = context_for_selection(payload)
    algorithm, k = payload.get("algorithm"), payload.get("k")
    if not isinstance(algorithm, str) or algorithm not in ALGORITHMS or type(k) is not int or not 1 <= k <= len(context["encoded"]):
        raise AnalysisError("Choose a supported selection algorithm and an integer k within the encoded feature count.")
    if algorithm == "Exhaustive" and comb(len(context["encoded"]), k) > MAX_SUBSETS_PER_K:
        raise AnalysisError("This exhaustive count exceeds the interactive compute limit.")
    with warnings.catch_warnings(record=True):
        pre, frame, paths, _, _ = training_paths(context, context["train_X"], context["train_y"], monotonic()+TIME_BUDGET, [algorithm])
        subset = paths[algorithm][k]
        # Keep training-derived imputers, vocabulary and selection fixed. Only
        # descriptive OLS coefficients are refitted using all valid target rows.
        all_X = pd.concat([context["train_X"], context["test_X"]]).sort_index()
        all_y = pd.concat([context["train_y"], context["test_y"]]).sort_index()
        encoded = pd.DataFrame(pre.transform(all_X), columns=frame.columns)[subset].to_numpy()
        model = ols_fit(encoded, all_y)
        predicted = model.predict(encoded)
    rows = [{"Row_Index": int(index), "Actual_Profit": float(actual), "Predicted_Profit": float(value),
             "Residual": float(actual-value)} for index, actual, value in zip(all_X.index, all_y, predicted)]
    return {"algorithm": algorithm, "k": k, "fingerprint": context["fingerprint"], "features": subset,
            "predictions": rows, "outliers": sorted(rows, key=lambda row: -abs(row["Residual"]))[:10],
            "note": "Descriptive all-row OLS refit, including held-out rows. This is not out-of-sample evaluation and does not replace the selected training model, inference tables or prediction calculator. Preprocessing and feature selection remain training-derived."}
