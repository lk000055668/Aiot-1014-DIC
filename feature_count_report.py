"""Create the teacher-style feature-count chart and table for the existing project."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_comparison import folds, subset_pipeline, audit_selector_encoding


def forward_subset(preprocessing, X, y, size):
    """Fit SFS inside the supplied training rows, never on validation rows."""
    feature_names = preprocessing.get_feature_names_out()
    if size == len(feature_names):
        return tuple(range(size))
    estimator = Pipeline([
        ("scaling", StandardScaler()),
        ("regression", LinearRegression()),
    ])
    selector = SequentialFeatureSelector(
        estimator, n_features_to_select=size, direction="forward",
        scoring="neg_root_mean_squared_error", cv=folds(), n_jobs=1,
    ).fit(preprocessing.transform(X), y)
    return tuple(np.flatnonzero(selector.get_support()).tolist())


def draw_curves(axes, results):
    counts = results["Number of Features"]
    settings = [
        ("RMSE", "RMSE by Number of Features", "RMSE"),
        ("R-squared", "R-squared by Number of Features", "R-squared"),
    ]
    for ax, (column, title, ylabel) in zip(axes, settings):
        ax.plot(counts, results[column], marker="o", linewidth=1.5,
                markersize=5, color="#1f77b4")
        ax.set(title=title, xlabel="Number of Features", ylabel=ylabel)
        ax.set_xticks(counts)
        ax.set_xlim(0.8, len(counts) + 0.2)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(False)


def draw_table(ax, results):
    ax.axis("off")
    rows = [
        [int(row["Number of Features"]), row["Selected Features"],
         f'{row["RMSE"]:.6f}', f'{row["R-squared"]:.6f}']
        for _, row in results.iterrows()
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Number of Features", "Selected Features", "RMSE", "R-squared"],
        colWidths=[0.16, 0.60, 0.12, 0.12], cellLoc="left",
        bbox=[0, 0.10, 1, 0.83],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.PAD = 0.04
        cell.set_edgecolor("white")
        cell.set_linewidth(0.5)
        cell.set_facecolor("white" if row == 0 or row % 2 == 0 else "#f4f5f6")
        if row == 0:
            cell.set_text_props(weight="bold")
        if column >= 2:
            cell.get_text().set_ha("right")
        elif row == 0 and column == 1:
            cell.get_text().set_ha("center")


def save_teacher_figures(results, output_dir):
    note = ("5-fold outer CV on 40 training rows; feature selection uses 5 inner folds. "
            "Feature lists show the fit on all training rows.")
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": False,
             "axes.facecolor": "white", "figure.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
        draw_curves(axes, results)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        fig.text(0.5, 0.025, note, ha="center", fontsize=8)
        fig.savefig(output_dir / "feature_count_comparison.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12.8, 2.6))
        draw_table(ax, results)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.text(0.5, 0.035, note, ha="center", fontsize=8)
        fig.savefig(output_dir / "feature_count_results.png", dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(12.8, 8.1))
        grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.65])
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
        draw_curves(axes, results)
        draw_table(fig.add_subplot(grid[1, :]), results)
        fig.suptitle("50 Startups - Forward Feature Selection", fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=2.0, w_pad=2.8)
        fig.text(0.5, 0.025, note, ha="center", fontsize=8)
        fig.savefig(output_dir / "teacher_feature_count_report.png", dpi=180)
        plt.close(fig)


def create_feature_count_report(template, X_train, y_train, output_dir):
    """Compare fixed feature counts by nested CV on training data only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_template = template.named_steps["preprocessing"]
    final_preprocessing = clone(preprocessing_template).fit(X_train)
    names = final_preprocessing.get_feature_names_out()
    audit_selector_encoding(final_preprocessing, X_train, y_train)
    outer_splits = list(folds().split(X_train, y_train))

    # Fit and audit each outer training preprocessor once; reuse its deterministic
    # encoding for the different feature counts. Validation targets stay excluded.
    outer_preprocessors = []
    for train, validation in outer_splits:
        local = clone(preprocessing_template).fit(X_train.iloc[train])
        audit_selector_encoding(local, X_train.iloc[train], y_train.iloc[train])
        outer_preprocessors.append(local)

    summaries, fold_rows, prediction_rows = [], [], []
    for size in range(1, len(names) + 1):
        print(f"\nFeature-count comparison: {size} input feature(s)...", flush=True)
        rmses, r2_scores = [], []
        for number, ((train, validation), preprocessing) in enumerate(
            zip(outer_splits, outer_preprocessors), start=1
        ):
            train_X, train_y = X_train.iloc[train], y_train.iloc[train]
            indices = forward_subset(preprocessing, train_X, train_y, size)
            model = subset_pipeline(template, indices).fit(train_X, train_y)
            predictions = model.predict(X_train.iloc[validation])
            actual = y_train.iloc[validation]
            rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
            r2 = float(r2_score(actual, predictions))
            rmses.append(rmse)
            r2_scores.append(r2)
            selected = preprocessing.get_feature_names_out()[list(indices)].tolist()
            fold_rows.append({
                "Number of Features": size, "Fold": number,
                "Selected Features": "[" + ", ".join(selected) + "]",
                "RMSE": rmse, "R-squared": r2,
                "Training Row Indices": " | ".join(map(str, train_X.index)),
                "Validation Row Indices": " | ".join(map(str, actual.index)),
            })
            prediction_rows.extend({
                "Number of Features": size, "Fold": number, "Row Index": int(index),
                "Actual Profit": float(value), "Predicted Profit": float(prediction),
            } for index, value, prediction in zip(actual.index, actual, predictions))
        # This final feature list is descriptive; outer metrics above have already
        # been calculated from independently selected subsets in each outer fold.
        indices = forward_subset(final_preprocessing, X_train, y_train, size)
        selected = names[list(indices)].tolist()
        summaries.append({
            "Number of Features": size,
            "Selected Features": "[" + ", ".join(selected) + "]",
            "RMSE": float(np.mean(rmses)),
            "R-squared": float(np.mean(r2_scores)),
        })
        print(f"  Mean outer RMSE: {np.mean(rmses):,.4f}; "
              f"mean outer R-squared: {np.mean(r2_scores):.6f}", flush=True)

    results = pd.DataFrame(summaries)
    results.to_csv(output_dir / "feature_count_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "feature_count_cv_folds.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "feature_count_cv_predictions.csv", index=False
    )
    save_teacher_figures(results, output_dir)
    print("\nFeature-count results (training-only nested cross-validation):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    best_row = results.loc[results.RMSE.idxmin()]
    print(f"Lowest mean CV RMSE in this comparison: {int(best_row['Number of Features'])} features.")
    print("Teacher report saved:", output_dir / "teacher_feature_count_report.png")
    return results


if __name__ == "__main__":
    # Generate just these teaching outputs without repeating all six model searches.
    from main import load_data, build_pipeline, INPUT_FEATURES, TARGET, OUTPUT_DIR
    from sklearn.model_selection import train_test_split

    try:
        data = load_data().dropna(subset=[TARGET])
        X_train, _, y_train, _ = train_test_split(
            data[INPUT_FEATURES], data[TARGET], test_size=0.2, random_state=42
        )
        create_feature_count_report(build_pipeline(), X_train, y_train, OUTPUT_DIR)
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
