"""Training-only model selection and nested validation for the existing project."""

from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METHODS = ("Baseline OLS", "Exhaustive", "ElasticNetCV", "Forward SFS", "RFECV", "LassoCV")
ALPHAS = np.logspace(-3, 5, 35)
L1_RATIOS = (0.1, 0.5, 0.9, 1.0)
COEFFICIENT_TOLERANCE = 1e-6


def folds():
    """Use the same deterministic five folds for every method."""
    return KFold(n_splits=5, shuffle=True, random_state=42)


class FeatureSubset(BaseEstimator, TransformerMixin):
    """Keep specified encoded columns; remains cloneable and Joblib-compatible."""

    def __init__(self, indices=None):
        self.indices = indices

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.indices_ = np.arange(X.shape[1]) if self.indices is None else np.asarray(self.indices)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.indices_]


def subset_pipeline(template, indices):
    return Pipeline([
        ("preprocessing", clone(template.named_steps["preprocessing"])),
        ("selection", FeatureSubset(indices)),
        ("regression", LinearRegression()),
    ])


def selected_features(model):
    names = model.named_steps["preprocessing"].get_feature_names_out()
    if "selection" in model.named_steps:
        return names[model.named_steps["selection"].indices_].tolist()
    if isinstance(model.named_steps["regression"], (LassoCV, ElasticNetCV)):
        active = np.abs(model.named_steps["regression"].coef_) > COEFFICIENT_TOLERANCE
        return names[active].tolist()
    return names.tolist()


def hyperparameters(model):
    regression = model.named_steps["regression"]
    if isinstance(regression, ElasticNetCV):
        return {"alpha": float(regression.alpha_), "l1_ratio": float(regression.l1_ratio_)}
    if isinstance(regression, LassoCV):
        return {"alpha": float(regression.alpha_)}
    return {}


def audit_selector_encoding(preprocessing, X, y):
    """Check that pre-encoding cannot leak fold statistics for this clean dataset.

    SFS/RFECV operate on encoded matrices. On these complete observations the
    imputation statistics are unused, and every inner training fold learns the
    same state encoding. Assert equivalence to fold-local transformations rather
    than assuming it. If future data break equivalence, fail with an explanation.
    Scaling is NOT performed here; selectors fit it inside their own CV folds.
    """
    for train, validation in folds().split(X, y):
        local = clone(preprocessing).fit(X.iloc[train])
        if list(local.get_feature_names_out()) != list(preprocessing.get_feature_names_out()):
            raise ValueError("Selector CV has a different state schema; revise fold-local encoding.")
        for rows in (train, validation):
            try:
                np.testing.assert_allclose(
                    local.transform(X.iloc[rows]), preprocessing.transform(X.iloc[rows]),
                    rtol=0, atol=0,
                )
            except AssertionError as exc:
                raise ValueError(
                    "SFS/RFECV pre-encoding is not fold-invariant (for example, missing inputs). "
                    "Use fold-local selector preprocessing before comparing this changed dataset."
                ) from exc


def fit_method(method, template, X, y):
    """Fit one complete selection procedure using ONLY the supplied training rows."""
    names = clone(template.named_steps["preprocessing"]).fit(X).get_feature_names_out()
    count = len(names)
    search_rows = []

    if method == "Baseline OLS":
        return clone(template).fit(X, y), search_rows

    if method == "Exhaustive":
        subsets = [subset for size in range(1, count + 1)
                   for subset in combinations(range(count), size)]
        search = GridSearchCV(
            subset_pipeline(template, None),
            {"selection__indices": subsets},
            scoring="neg_root_mean_squared_error", cv=folds(),
            refit=True, error_score="raise", n_jobs=1,
        ).fit(X, y)
        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            indices = params["selection__indices"]
            search_rows.append({
                "Model": method, "Selected_Features": " | ".join(names[list(indices)]),
                "Feature_Count": len(indices), "Inner_CV_RMSE": -float(score),
            })
        return search.best_estimator_, search_rows

    if method in ("ElasticNetCV", "LassoCV"):
        # Native CV estimators cannot refit an upstream scaler inside their CV.
        # Search singleton-alpha candidates as complete raw-input pipelines.
        # Thus the DECISION-MAKING CV fits imputation, encoding and scaling
        # independently in every inner fold. Native singleton CV scores cannot
        # change a candidate's hyperparameters, so they cause no tuning leakage.
        regression = (
            ElasticNetCV(alphas=[1.0], l1_ratio=1.0, cv=5,
                         max_iter=100000, tol=1e-7, n_jobs=1)
            if method == "ElasticNetCV" else
            LassoCV(alphas=[1.0], cv=5, max_iter=100000, tol=1e-7, n_jobs=1)
        )
        candidate = Pipeline([
            ("preprocessing", clone(template.named_steps["preprocessing"])),
            ("scaling", StandardScaler()),
            ("regression", regression),
        ])
        grid = {"regression__alphas": [(float(alpha),) for alpha in ALPHAS]}
        if method == "ElasticNetCV":
            grid["regression__l1_ratio"] = list(L1_RATIOS)
        search = GridSearchCV(
            candidate, grid, scoring="neg_root_mean_squared_error",
            cv=folds(), refit=True, error_score="raise", n_jobs=1,
        ).fit(X, y)
        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
            search_rows.append({
                "Model": method, "Alpha": params["regression__alphas"][0],
                "L1_Ratio": params.get("regression__l1_ratio", 1.0),
                "Inner_CV_RMSE": -float(score),
            })
        return search.best_estimator_, search_rows

    preprocessing = clone(template.named_steps["preprocessing"]).fit(X)
    audit_selector_encoding(preprocessing, X, y)
    encoded = preprocessing.transform(X)
    # Scaling inside the selector's estimator is fitted anew for every inner
    # fold. It also makes RFECV coefficient magnitudes comparable across units.
    selection_estimator = Pipeline([
        ("scaling", StandardScaler()),
        ("regression", LinearRegression()),
    ])

    if method == "Forward SFS":
        candidates = []
        # SFS requires fewer than all columns; the all-feature option is explicit.
        for size in range(1, count + 1):
            if size == count:
                indices = tuple(range(count))
            else:
                selector = SequentialFeatureSelector(
                    selection_estimator, n_features_to_select=size,
                    direction="forward", scoring="neg_root_mean_squared_error",
                    cv=folds(), n_jobs=1,
                ).fit(encoded, y)
                indices = tuple(np.flatnonzero(selector.get_support()).tolist())
            candidate = subset_pipeline(template, indices)
            score = -cross_val_score(
                candidate, X, y, cv=folds(),
                scoring="neg_root_mean_squared_error", error_score="raise",
            ).mean()
            search_rows.append({
                "Model": method, "Selected_Features": " | ".join(names[list(indices)]),
                "Feature_Count": size, "Inner_CV_RMSE": float(score),
            })
            candidates.append((float(score), size, candidate))
        # If scores tie exactly, choose the smaller subset.
        chosen = min(candidates, key=lambda entry: (entry[0], entry[1]))[2]
        return chosen.fit(X, y), search_rows

    if method == "RFECV":
        selector = RFECV(
            selection_estimator, step=1, min_features_to_select=1,
            scoring="neg_root_mean_squared_error", cv=folds(), n_jobs=1,
            importance_getter="named_steps.regression.coef_",
        ).fit(encoded, y)
        indices = tuple(np.flatnonzero(selector.get_support()).tolist())
        for size, score in enumerate(selector.cv_results_["mean_test_score"], start=1):
            search_rows.append({
                "Model": method, "Feature_Count": size, "Inner_CV_RMSE": -float(score),
            })
        # The scaled selector uses the same OLS solution; refit an unscaled OLS
        # pipeline on its selected columns for understandable coefficients.
        return subset_pipeline(template, indices).fit(X, y), search_rows
    raise ValueError(f"Unknown method: {method}")


def fit_with_convergence_check(method, template, X, y):
    # Scaled features, a broad alpha grid and a high iteration limit normally
    # converge. A warning becomes a clear failure instead of silently accepting
    # unreliable candidate scores.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            return fit_method(method, template, X, y)
        except ConvergenceWarning as exc:
            raise RuntimeError(
                f"{method} did not converge. Increase max_iter or refine ALPHAS "
                "before trusting the comparison."
            ) from exc


def train_comparison(template, X_train, y_train, output_dir):
    """Select the winner from training-only outer CV; no test argument is accepted."""
    outer_splits = list(folds().split(X_train, y_train))
    models, validation_rows, search_rows, summary_rows = {}, [], [], []
    for method in METHODS:
        print(f"\nComparing {method}: five outer folds with training-only selection...", flush=True)
        rmse_scores = []
        for fold_number, (train, validation) in enumerate(outer_splits, start=1):
            assert not set(X_train.index[train]) & set(X_train.index[validation])
            model, _ = fit_with_convergence_check(
                method, template, X_train.iloc[train], y_train.iloc[train]
            )
            predictions = model.predict(X_train.iloc[validation])
            rmse = float(np.sqrt(mean_squared_error(y_train.iloc[validation], predictions)))
            rmse_scores.append(rmse)
            validation_rows.append({
                "Model": method, "Fold": fold_number, "RMSE": rmse,
                "Training_Row_Indices": " | ".join(map(str, X_train.index[train])),
                "Validation_Row_Indices": " | ".join(map(str, X_train.index[validation])),
                "Selected_Features": " | ".join(selected_features(model)),
                **hyperparameters(model),
            })
            print(f"  Outer fold {fold_number}: RMSE {rmse:,.2f}", flush=True)
        # Refit the whole training set; never include any of the held-out test rows.
        model, details = fit_with_convergence_check(method, template, X_train, y_train)
        models[method] = model
        search_rows.extend(details)
        summary_rows.append({
            "Model": method,
            "Selected_Features": " | ".join(selected_features(model)) or "(intercept only)",
            "Feature_Count": len(selected_features(model)),
            "CV_RMSE_Mean": float(np.mean(rmse_scores)),
            "CV_RMSE_Std": float(np.std(rmse_scores)),
            **hyperparameters(model),
        })
        print("  Final selected features:", selected_features(model), flush=True)
        if hyperparameters(model):
            print("  Optimal hyperparameters:", hyperparameters(model), flush=True)
    summary = pd.DataFrame(summary_rows)
    best_name = summary.loc[summary["CV_RMSE_Mean"].idxmin(), "Model"]
    pd.DataFrame(validation_rows).to_csv(output_dir / "nested_cv_results.csv", index=False)
    pd.DataFrame(search_rows).to_csv(output_dir / "selection_search_results.csv", index=False)
    print(f"\nBest model by training-only outer CV RMSE: {best_name}", flush=True)
    return models, summary, best_name


def evaluate_comparison(models, summary, best_name, X_train, y_train, X_test, y_test, output_dir):
    """Evaluate the frozen six methods on the shared test set without reranking."""
    prediction_rows = []
    summary = summary.copy()
    for row, method in zip(summary.index, summary["Model"]):
        model = models[method]
        predicted = model.predict(X_test)
        mse = mean_squared_error(y_test, predicted)
        metrics = {
            "Test_MAE": mean_absolute_error(y_test, predicted),
            "Test_MSE": mse, "Test_RMSE": np.sqrt(mse),
            "Test_R2": r2_score(y_test, predicted),
            "Train_R2": model.score(X_train, y_train),
        }
        for name, value in metrics.items():
            summary.loc[row, name] = float(value)
        prediction_rows.extend({
            "Model": method, "Row_Index": int(index),
            "Actual_Profit": float(actual), "Predicted_Profit": float(prediction),
            "Residual": float(actual - prediction),
        } for index, actual, prediction in zip(X_test.index, y_test, predicted))
    summary["Best_By_CV"] = summary["Model"] == best_name
    summary.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "prediction_results_all_models.csv", index=False)
    predictions.loc[predictions.Model == best_name].to_csv(
        output_dir / "best_prediction_results.csv", index=False
    )
    display = summary[["Model", "Selected_Features", "CV_RMSE_Mean", "CV_RMSE_Std",
                       "Test_MAE", "Test_MSE", "Test_RMSE", "Test_R2", "Train_R2"]].copy()
    display["CV RMSE (mean +/- std)"] = [
        f"{mean:,.2f} +/- {std:,.2f}"
        for mean, std in zip(display.CV_RMSE_Mean, display.CV_RMSE_Std)
    ]
    display = display.drop(columns=["CV_RMSE_Mean", "CV_RMSE_Std"])
    print("\nSix-method comparison:\n" + display.to_string(
        index=False, float_format=lambda value: f"{value:,.4f}"
    ))
    print(f"\nSelected before test evaluation: {best_name}. Test scores do not change this choice.")
    print("Dropping weak features can reduce variance; regularization shrinks coefficients.")
    print("RFECV ranks standardized OLS effects; forward and exhaustive searches follow different paths.")
    print("The numerical results, rather than assumptions about any method, determine this ranking.")
    print("Small CV differences relative to fold variability are not strong evidence of superiority.")
    save_comparison_plots(summary, models, output_dir)
    return summary


def save_comparison_plots(summary, models, output_dir):
    ordered = summary.sort_values("CV_RMSE_Mean", kind="stable")
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered.CV_RMSE_Mean, xerr=ordered.CV_RMSE_Std,
            capsize=4, color="teal", alpha=0.85)
    ax.set_yticks(positions, ordered.Model)
    ax.invert_yaxis()
    ax.set(xlabel="Cross-validation RMSE", title="50 Startups \u2014 Model Comparison")
    end = (ordered.CV_RMSE_Mean + ordered.CV_RMSE_Std).max()
    ax.set_xlim(0, end * 1.22)
    for position, (_, row) in zip(positions, ordered.iterrows()):
        ax.text(row.CV_RMSE_Mean + row.CV_RMSE_Std + end * 0.015, position,
                f"{row.CV_RMSE_Mean:,.2f}", va="center")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_rmse_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(summary.Model, summary.Test_RMSE, color="steelblue")
    ax.invert_yaxis()
    limit = summary.Test_RMSE.max()
    ax.set_xlim(0, limit * 1.18)
    for position, value in enumerate(summary.Test_RMSE):
        ax.text(value + limit * 0.015, position, f"{value:,.2f}", va="center")
    ax.set(xlabel="Test RMSE", title="50 Startups: Shared Test Set Performance")
    fig.tight_layout()
    fig.savefig(output_dir / "test_rmse_comparison.png", dpi=160)
    plt.close(fig)

    methods = ["Exhaustive", "Forward SFS", "RFECV", "LassoCV"]
    names = models["Baseline OLS"].named_steps["preprocessing"].get_feature_names_out().tolist()
    selected = np.array([[int(name in selected_features(models[method]))
                          for name in names] for method in methods])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(selected, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_yticks(range(len(methods)), methods)
    ax.set_title("Feature Selection Comparison (Full Training Set)")
    ax.grid(False)
    for row in range(len(methods)):
        for column in range(len(names)):
            value = selected[row, column]
            ax.text(column, row, "Selected" if value else "-",
                    ha="center", va="center", color="white" if value else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_selection_comparison.png", dpi=160)
    plt.close(fig)
