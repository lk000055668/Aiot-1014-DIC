"""50 Startups multiple linear regression using the six CRISP-DM phases."""

from pathlib import Path
import socket
import json
import sys
from urllib.error import URLError

try:
    import joblib
    import matplotlib
    matplotlib.use("Agg")  # Save plots without opening GUI windows.
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from web.regression_workflow import build_preprocessing
    from feature_count_report import create_feature_count_report
    from model_comparison import (
        train_comparison, evaluate_comparison, selected_features, hyperparameters
    )
except ImportError as exc:
    print(f"Missing dependency: {exc}. Run: python -m pip install -r requirements.txt",
          file=sys.stderr)
    raise SystemExit(1)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "startup_profit_model.pkl"
# Pin the source revision for reproducible assignment results.
DATA_URL = (
    "https://gist.githubusercontent.com/GaneshSparkz/"
    "b5662effbdae8746f7f7d8ed70c42b2d/raw/"
    "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv"
)
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
INPUT_FEATURES = NUMERIC_FEATURES + ["State"]
TARGET = "Profit"


def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def load_data():
    """Read directly from GitHub without saving a local dataset."""
    old_timeout = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(30)
        data = pd.read_csv(DATA_URL)
    except (OSError, URLError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            "Could not load the GitHub CSV. Check your internet connection, "
            f"proxy/firewall and dataset URL. Details: {exc}"
        ) from exc
    finally:
        socket.setdefaulttimeout(old_timeout)
    expected = INPUT_FEATURES + [TARGET]
    if list(data.columns) != expected:
        raise ValueError(f"Expected columns {expected}; received {list(data.columns)}.")
    if data.empty:
        raise ValueError("The remote dataset has no observations.")
    if len(data) != 50:
        print(f"Warning: expected 50 rows, received {len(data)}.")
    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Column {column} contains infinite values.")
    # Normalize blank/missing state labels for SimpleImputer.
    data["State"] = data["State"].str.strip().replace("", np.nan)
    data["State"] = data["State"].astype(object).where(data["State"].notna(), np.nan)
    return data


def build_pipeline():
    return Pipeline([
        ("preprocessing", build_preprocessing(NUMERIC_FEATURES, ["State"])),
        ("regression", LinearRegression()),
    ])


def save_evaluation_plots(actual, predicted, residuals, prefix="", model_name=""):
    title_suffix = f" - {model_name}" if model_name else ""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, color="teal", s=65, alpha=0.8)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "r--", label="Perfect prediction")
    ax.set(xlabel="Actual profit", ylabel="Predicted profit",
           title="Actual vs. Predicted Profit (Test Set)" + title_suffix)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{prefix}actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residuals, color="darkorange", s=65, alpha=0.8)
    ax.axhline(0, color="red", linestyle="--", label="Zero residual")
    ax.set(xlabel="Predicted profit", ylabel="Residual (actual - predicted)",
           title="Residual Plot (Test Set)" + title_suffix)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{prefix}residual_plot.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 140)
    sns.set_theme(style="whitegrid")

    # Phase 1: Business Understanding
    heading("Phase 1: Business Understanding")
    print("Objective: predict startup profit from spending and geographic state.")
    print("Inputs (X): " + ", ".join(INPUT_FEATURES))
    print("Target (y): Profit. The target is never included among input features.")
    print("Predictions support planning; coefficients describe associations, not causation.")

    # Phase 2: Data Understanding
    heading("Phase 2: Data Understanding")
    data = load_data()
    print(f"Loaded directly from: {DATA_URL}")
    print("\nFirst 5 rows:\n" + data.head().to_string(index=False))
    print(f"\nDataset dimensions: {data.shape[0]} observations x {data.shape[1]} columns")
    print("Original columns:", list(data.columns))
    print("\nData types:\n" + data.dtypes.to_string())
    print("\nDescriptive statistics:\n" + data.describe(include="all").to_string())
    print("\nMissing values:\n" + data.isna().sum().to_string())
    correlations = data[NUMERIC_FEATURES + [TARGET]].corr()
    print("\nNumerical correlation matrix (exploratory):\n" + correlations.to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("50 Startups: Numerical Correlations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160)
    plt.close(fig)

    # Phase 3: Data Preparation
    heading("Phase 3: Data Preparation")
    missing_targets = int(data[TARGET].isna().sum())
    if missing_targets:
        print(f"Dropping {missing_targets} rows with missing Profit; targets are not imputed.")
    clean_data = data.dropna(subset=[TARGET])
    if len(clean_data) < 15:
        raise ValueError("Too few labeled rows for a holdout and 5-fold cross-validation.")
    X = clean_data[INPUT_FEATURES].copy()
    y = clean_data[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"80/20 split, random_state=42: {len(X_train)} training / {len(X_test)} testing rows.")
    print("Missing numeric inputs: training median; missing State: training mode.")
    print('State encoding: drop="first", handle_unknown="ignore", sparse_output=False.')
    pipeline = build_pipeline()
    preprocessing = pipeline.named_steps["preprocessing"]
    # Fit on training rows only to inspect encoding. Phase 4 refits the same rows.
    preprocessing.fit(X_train)
    feature_names = preprocessing.get_feature_names_out()
    encoder = preprocessing.named_transformers_["state"].named_steps["encoder"]
    reference_state = encoder.categories_[0][encoder.drop_idx_[0]]
    print("Learned state categories:", encoder.categories_[0].tolist())
    print("Reference state:", reference_state)
    print("\nEncoded input feature names:")
    for number, name in enumerate(feature_names, start=1):
        print(f"  {number}. {name}")
    print(f"Total input features after encoding: {len(feature_names)}")
    if len(feature_names) != 5:
        raise ValueError("Expected five inputs: three spending variables and two state indicators.")
    encoded = pd.DataFrame(preprocessing.transform(X), columns=feature_names, index=X.index)
    encoded["Profit (target, excluded from X)"] = y
    encoded["Split"] = np.where(encoded.index.isin(X_train.index), "train", "test")
    print("\nEncoded dataset (training-fitted transformation; target shown separately):")
    print(encoded.to_string(index=True))

    # Phase 4: Modeling
    heading("Phase 4: Modeling")
    # Fit all preprocessing steps and the regression on training data only.
    pipeline.fit(X_train, y_train)
    regression = pipeline.named_steps["regression"]
    print(f"Intercept: {regression.intercept_:.6f}")
    print("\nRegression coefficients:")
    coefficients = pd.Series(regression.coef_, index=feature_names, name="Coefficient")
    print(coefficients.to_string(float_format=lambda value: f"{value:.6f}"))
    equation = f"Profit = {regression.intercept_:.6f}"
    for name, coefficient in coefficients.items():
        sign = "+" if coefficient >= 0 else "-"
        equation += f" {sign} {abs(coefficient):.6f} * [{name}]"
    print("\nLearned regression equation:\n" + equation)
    print(f"State indicators are 0 or 1; {reference_state} has both indicators equal to 0.")

    # Compare entire selection procedures without giving this function test data.
    models, comparison, best_name = train_comparison(
        pipeline, X_train, y_train, OUTPUT_DIR
    )

    # Phase 5: Evaluation
    heading("Phase 5: Evaluation")
    predicted = pipeline.predict(X_test)
    print("Baseline OLS diagnostics (preserved from the original project):")
    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = float(np.sqrt(mse))  # Compatible across scikit-learn versions.
    train_r2 = pipeline.score(X_train, y_train)
    test_r2 = r2_score(y_test, predicted)
    print(f"Test MAE:  {mae:,.4f}")
    print(f"Test MSE:  {mse:,.4f}")
    print(f"Test RMSE: {rmse:,.4f}")
    print(f"Training R-squared (R2): {train_r2:.6f}")
    print(f"Testing R-squared (R2):  {test_r2:.6f}")
    # Each fold clones the full pipeline and learns its own imputer and encoder.
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=folds, scoring="r2", error_score="raise"
    )
    print("\n5-fold cross-validation on training data only (R2):")
    for number, score in enumerate(cv_scores, start=1):
        print(f"  Fold {number}: {score:.6f}")
    cv_mean = float(cv_scores.mean())
    print(f"CV mean R2: {cv_mean:.6f}; standard deviation: {cv_scores.std():.6f}")
    residuals = y_test.to_numpy() - predicted
    results = pd.DataFrame({
        "Row_Index": X_test.index,
        "Actual_Profit": y_test.to_numpy(),
        "Predicted_Profit": predicted,
        "Residual": residuals,
    })
    print("\nActual vs. predicted profit (test rows):\n" +
          results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    results.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)
    save_evaluation_plots(y_test.to_numpy(), predicted, residuals)

    print("\nMetric interpretation:")
    print("MAE: average absolute error, in profit units; smaller is better.")
    print("MSE: average squared error, in squared profit units; penalizes large errors.")
    print("RMSE: square root of MSE, in profit units; smaller is better.")
    print("R2: 1 is perfect; 0 matches a constant mean benchmark; negative is worse.")
    print("Training R2 measures fit on seen rows; test/CV R2 measure fit on held-out rows.")
    if train_r2 - test_r2 > 0.10 or train_r2 - cv_mean > 0.10:
        print("Fit discussion: the training-to-validation gap suggests possible overfitting.")
    elif train_r2 < 0.60 and test_r2 < 0.60:
        print("Fit discussion: low training and test R2 suggest possible underfitting.")
    else:
        print("Fit discussion: R2 scores show no obvious severe overfitting or underfitting.")
    print("This is a heuristic assessment: 50 rows and 10 test rows make estimates uncertain.")
    print("Inspect residual patterns; linearity and constant error variance are not guaranteed.")

    comparison = evaluate_comparison(
        models, comparison, best_name, X_train, y_train, X_test, y_test, OUTPUT_DIR
    )
    create_feature_count_report(pipeline, X_train, y_train, OUTPUT_DIR)

    best_model = models[best_name]
    best_predictions = best_model.predict(X_test)
    save_evaluation_plots(
        y_test.to_numpy(), best_predictions, y_test.to_numpy() - best_predictions,
        prefix="best_", model_name=best_name,
    )

    # Phase 6: Deployment
    heading("Phase 6: Deployment")
    joblib.dump(pipeline, MODEL_PATH)
    loaded_pipeline = joblib.load(MODEL_PATH)
    new_startup = pd.DataFrame([{
        "R&D Spend": 150000,
        "Administration": 120000,
        "Marketing Spend": 300000,
        "State": "New York",
    }])
    new_prediction = float(loaded_pipeline.predict(new_startup)[0])
    # Verify persistence preserves predictions on every test row.
    np.testing.assert_allclose(loaded_pipeline.predict(X_test), predicted)
    print(f"Saved and reloaded complete pipeline: {MODEL_PATH}")
    print("\nNew startup inputs:\n" + new_startup.to_string(index=False))
    print(f"Predicted profit for the new startup: {new_prediction:,.2f}")
    best_model_path = BASE_DIR / "best_startup_profit_model.pkl"
    joblib.dump(best_model, best_model_path)
    loaded_best = joblib.load(best_model_path)
    np.testing.assert_allclose(loaded_best.predict(X_test), best_predictions)
    best_example_profit = float(loaded_best.predict(new_startup)[0])
    print(f"Best model saved and reloaded: {best_model_path}")
    print(f"Best model ({best_name}) example predicted profit: {best_example_profit:,.2f}")
    metadata = {
        "best_model": best_name, "selection_metric": "training-only outer 5-fold CV RMSE",
        "random_state": 42, "outer_folds": 5, "inner_folds": 5,
        "training_rows": X_train.index.tolist(), "testing_rows": X_test.index.tolist(),
        "best_selected_features": selected_features(best_model),
        "best_hyperparameters": hyperparameters(best_model),
        "best_example_profit": best_example_profit,
        "baseline_example_profit": new_prediction, "dataset_url": DATA_URL,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nPlots and prediction CSV saved in: {OUTPUT_DIR}")
    print("Completed all six CRISP-DM phases successfully.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, TypeError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("If an API is unavailable, install requirements with this Python interpreter.",
              file=sys.stderr)
        raise SystemExit(1)
