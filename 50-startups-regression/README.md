# 50 Startups Profit Prediction

A university Data Mining project using multiple linear regression and all six
CRISP-DM phases to predict startup profit from spending and geographic state.

## Dataset source

[GaneshSparkz's 50 Startups Gist](https://gist.github.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d)
contains 50 observations and five columns. The script calls `pandas.read_csv()`
with this revision-pinned raw CSV URL:

https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv

No manual download or local dataset copy is needed. Internet access is required
each time the script runs.

## Install and run in VS Code

Use Python 3.10 or newer. Open this project folder in VS Code, select your Python
interpreter, then run these commands in its terminal:

```powershell
python -m pip install -r requirements.txt
python main.py
```

An optional virtual environment can be created with `python -m venv .venv`.
If you use it, select `.venv\Scripts\python.exe` in VS Code and install the
requirements using that interpreter.

From the parent L5 directory, you can also run:

```powershell
python 50-startups-regression/main.py
```

All generated paths are relative to the script's directory, regardless of the
terminal's current directory. Plots are saved with a noninteractive Matplotlib
backend, so execution does not wait for plot windows to close.

## CRISP-DM methodology

1. **Business understanding:** predict profit to support financial planning.
   The input variables describe R&D, administration and marketing spending,
   plus the startup's state; Profit is the target.
2. **Data understanding:** print the first five rows, dimensions, original
   columns, types, descriptive statistics, missing counts, and a numerical
   correlation matrix; save a correlation heatmap.
3. **Data preparation:** separate X and y, split 80/20 with
   `random_state=42`, and prepare numerical imputation and state encoding.
   Display all transformed rows and input feature names, with the target
   clearly labeled separately.
4. **Modeling:** preserve the baseline equation and add six regression methods,
   with training-only feature selection and regularization tuning.
5. **Evaluation:** compare five-fold outer RMSE for all six training procedures,
   select the winner, evaluate all six on the shared test set, and preserve
   the original baseline R2 diagnostics, predictions and plots.
6. **Deployment:** save/reload the baseline and CV-selected pipelines, verify
   identical test predictions, and predict a new startup's profit.

## Original and encoded features

| Role | Columns |
| --- | --- |
| Original inputs (4) | R&D Spend, Administration, Marketing Spend, State |
| Encoded inputs (5) | R&D Spend, Administration, Marketing Spend, State_Florida, State_New York |
| Target (1) | Profit |

One-hot encoding represents states using binary indicator columns. California
is the alphabetically first training category and is the reference: both
indicators are zero. Florida has `State_Florida=1`; New York has
`State_New York=1`. Using `drop="first"` avoids perfect multicollinearity
between the full set of state indicators and the intercept.
`handle_unknown="ignore"` allows a new state to be predicted without an error;
its indicators become zero, so it receives the reference state's effect
(and Scikit-learn may issue a warning). This does not establish accuracy for
unseen states. `sparse_output=False` returns dense encoded arrays and requires
Scikit-learn 1.2 or newer, as specified in requirements.txt.

The encoder and imputers learn only from training rows. Phase 3 fits the
preprocessor for inspection; Phase 4 refits it on the same training rows as
part of fitting the whole pipeline. Cross-validation clones the whole pipeline
and refits preprocessing separately inside every fold. Test rows are never
used to learn preprocessing or regression coefficients.

Missing spending values use training medians; missing State uses its training
mode. Rows with missing Profit are removed because a target cannot be learned
from an unlabeled observation. Zero spending is a valid value and is retained.
The original dataset has no missing values. The whole-data correlation matrix
is descriptive only and does not drive feature selection or model tuning.

## Multiple linear regression

The model learns this equation by minimizing the sum of squared training errors:

```text
Profit = intercept
       + b1 * R&D Spend
       + b2 * Administration
       + b3 * Marketing Spend
       + b4 * State_Florida
       + b5 * State_New York
```

Spending coefficients describe predicted profit change per additional spending
unit, holding the other inputs fixed. State coefficients describe differences
from California at equal spending. The script prints actual learned values.
No scaling is needed for this ordinary least squares implementation.
Coefficients are associations and should not be interpreted as causal effects.

## Evaluation and interpretation

- **MAE:** average absolute prediction error in profit units.
- **MSE:** average squared prediction error, in squared profit units; emphasizes
  large errors.
- **RMSE:** square root of MSE, in profit units.
- **R2:** explained variation relative to a constant mean prediction. A score
  of 1 is perfect, 0 matches that benchmark, and a negative score is worse.
- **Training versus test R2:** a large performance gap suggests overfitting.
  Poor scores on both may suggest underfitting.
- **Five-fold CV R2:** estimates performance across five training-only validation
  splits. The printed mean and standard deviation show typical performance
  and variation. Standard deviation is not a confidence interval.

The program uses a simple heuristic to discuss overfitting/underfitting.
Inspect the residual plot for curvature, systematic bias, or changing spread.
A positive residual means actual profit exceeds the prediction. In the
actual/predicted plot, points near the diagonal are more accurate.

## Six-method comparison

The original OLS baseline, six CRISP-DM stages, preprocessing, EDA, three plots,
prediction CSV, and model file are preserved. The new model_comparison.py module
extends the Modeling and Evaluation stages.

| Method | Training-only procedure |
| --- | --- |
| Baseline OLS | LinearRegression with all five encoded features |
| Exhaustive | All 31 nonempty encoded subsets; select by five-fold RMSE |
| ElasticNetCV | Pipeline CV search over alpha and l1_ratio |
| Forward SFS | Forward SequentialFeatureSelector for 1–4 features; compare these subsets and the all-five option |
| RFECV | Recursive elimination using standardized LinearRegression coefficients and five-fold RMSE |
| LassoCV | Pipeline CV search over alpha; absolute coefficients above 1e-6 count as selected |

Florida and New York indicators are individual candidate features. Regularized
models' active features are determined from coefficients in standardized units.

### Fair validation

All six methods use the same 40/10 train/test split and the same five shuffled
outer folds, with random_state=42. Each complete selection procedure fits on
32 outer training observations and predicts the remaining eight. Inner feature
and parameter selection uses five folds. Baseline OLS has no inner tuning.
Reported CV RMSE mean and population standard deviation come from the five
OUTER fold scores; inner tuning scores are exported separately.

The winner is fixed by outer CV mean RMSE before test evaluation. The six
methods are then evaluated on the shared 10 test observations. Test performance
does not change the winner. Final pipelines are fitted on the 40 training rows.

Imputation and encoding stay inside raw-input pipelines. Exhaustive selection
refits preprocessing in every candidate fold. StandardScaler is used for
Lasso/ElasticNet and inside the SFS/RFECV estimators. A runtime audit checks that
SFS/RFECV pre-encoding gives EXACTLY the same transformed train/validation values
as fitting preprocessing independently within each selector fold. This is true
for these complete data: imputation has no effect, and folds contain the same
three states. If changed data with missing values or categories break that
equivalence, comparison stops with a clear error instead of reporting leakage.

A plain scaler followed by LassoCV or ElasticNetCV cannot refit the upstream
scaler inside native CV. Therefore GridSearchCV searches full raw-input
pipelines, each containing a CV estimator with one alpha (and one l1_ratio for
ElasticNet). The surrounding inner search chooses the optimal candidate and
refits preprocessing within each decision-making fold. Native singleton-CV
scores cannot influence parameter choice, which avoids scaling leakage.

The alpha grid has 35 logarithmic values from 0.001 to 100000.
ElasticNet l1_ratios are 0.1, 0.5, 0.9, and 1.0; 1.0 is equivalent to Lasso.
"Optimal" means best within this finite reproducible grid. Models use
max_iter=100000 and tol=1e-7. Convergence warnings become errors rather than
being suppressed. The nested search takes longer than the original script
and prints progress after every outer fold.

### Added outputs

| File in outputs/ | Contents |
| --- | --- |
| model_comparison.csv | Six methods, selected features, CV mean/std, all test metrics, train R2, alpha/ratio and winner flag |
| nested_cv_results.csv | 30 outer-fold results with fold row indices, features and parameters |
| selection_search_results.csv | Final-training inner candidate scores, including all 31 exhaustive subsets and five SFS sizes |
| prediction_results_all_models.csv | 60 test predictions with actual profit and residuals |
| best_prediction_results.csv | Ten predictions from the CV winner |
| cv_rmse_comparison.png | Sorted horizontal CV RMSE chart with standard-deviation error bars |
| test_rmse_comparison.png | Shared test RMSE for all six models |
| feature_selection_comparison.png | Exhaustive, SFS, RFECV and Lasso feature matrix |
| best_actual_vs_predicted.png | Winner's scatter plot with perfect-prediction line |
| best_residual_plot.png | Winner's residual plot with zero-reference line |
| run_metadata.json | Dataset URL, split indices, selection rule, winner and example predictions |

startup_profit_model.pkl remains the original OLS pipeline.
best_startup_profit_model.pkl contains the CV-selected pipeline, including
preprocessing, selection and scaling where applicable. Use this latter filename
in the deployment example to predict with the selected model. Keep
model_comparison.py available when loading its FeatureSubset transformer.
No new dependencies are required.

Choosing the best of six outer scores can introduce selection optimism;
the held-out test set remains the final check. With 50 rows, differences small
relative to fold variability do not prove superiority. Removing weak features
and shrinking coefficients can reduce variance. The measured results determine
the ranking; no method is universally better.

## Deployment example

The saved pipeline accepts a DataFrame with the four original input columns:

```python
from pathlib import Path
import joblib
import pandas as pd

model = joblib.load(Path("startup_profit_model.pkl"))
startup = pd.DataFrame([{
    "R&D Spend": 150000,
    "Administration": 120000,
    "Marketing Spend": 300000,
    "State": "New York",
}])
print(f"Predicted profit: {model.predict(startup)[0]:,.2f}")
```

The deployed model is the same pipeline trained on the 40 training rows used in
the holdout experiment. It is not refitted on the test rows.

## Generated files

```text
50-startups-regression/
|-- main.py
|-- model_comparison.py
|-- requirements.txt
|-- README.md
|-- startup_profit_model.pkl
|-- best_startup_profit_model.pkl
|-- outputs/
    |-- correlation_matrix.png
    |-- actual_vs_predicted.png
    |-- residual_plot.png
    |-- prediction_results.csv
```

The prediction CSV has Row_Index (original zero-based dataset row),
Actual_Profit, Predicted_Profit, and Residual. The model contains both fitted
preprocessing and regression. Re-running overwrites these generated artifacts.

## Limitations and troubleshooting

There are only 50 observations, including just 10 test observations. A single
split can give optimistic or pessimistic results, and five-fold scores can vary.
The model assumes an additive linear relationship; outliers and correlated
spending variables can affect its coefficients. Predictions outside the observed
spending ranges or for other states have limited support. Currency and the
profit measurement period are not documented in the source CSV.

If a dependency is missing, the script prints an installation command. Install
requirements using the same interpreter that runs the script. Network failures,
invalid CSV structure, and output permission errors produce clear error messages
and a nonzero exit status. For download failures, check internet access, GitHub
availability, proxy/firewall settings, and the raw URL. Keep Python and package
versions consistent when loading a saved model.

## Preserved baseline execution results

The complete script ran successfully on the original 50 observations, using
40 training rows and 10 test rows with `random_state=42`. No dependency
installation was necessary in the supplied environment.

| Metric | Calculated value |
| --- | ---: |
| Test MAE | 6,961.4778 |
| Test MSE | 82,010,363.0443 |
| Test RMSE | 9,055.9573 |
| Training R2 | 0.953702 |
| Testing R2 | 0.898727 |
| 5-fold training CV mean R2 | 0.934004 |
| 5-fold training CV standard deviation | 0.032301 |
| Example startup predicted profit | 175,581.62 |

Fold R2 scores were 0.953948, 0.954346, 0.893257, 0.971281, and 0.897187.
The training/test gap is approximately 0.055, with no obvious severe overfitting
or underfitting by the program's heuristic. The small dataset still limits
confidence in performance on future startups.

Verified artifacts include all three valid PNG plots, 10 prediction CSV rows,
five model input features, and unchanged predictions after model reload.
Additional prediction checks confirmed missing inputs and an unknown state can
be processed. Initial sandbox network access was restricted; the complete run
succeeded with network access and read the raw CSV directly without a local
copy. Tested with Python 3.14.7, pandas 3.0.6, NumPy 2.5.3, Scikit-learn 1.9.1,
Matplotlib 3.11.2, Seaborn 0.13.2, and Joblib 1.6.0. Small floating-point
differences are possible across library versions.

## Verified six-method results

The enhanced script completed all six CRISP-DM stages without errors or
convergence warnings. No dependencies needed installation.

| Model | Selected features | Outer CV RMSE mean ± std | Test RMSE | Test R2 |
| --- | --- | ---: | ---: | ---: |
| Baseline OLS | All five encoded features | 9,550.68 ± 2,936.12 | 9,055.96 | 0.898727 |
| Exhaustive | Three spending variables | 9,532.69 ± 2,928.29 | 8,995.91 | 0.900065 |
| ElasticNetCV | Three spending variables | 9,229.58 ± 3,230.98 | 8,397.00 | 0.912929 |
| Forward SFS | Three spending variables | 9,532.69 ± 2,928.29 | 8,995.91 | 0.900065 |
| RFECV | Three spending variables | 9,519.09 ± 3,103.80 | 8,995.91 | 0.900065 |
| LassoCV | Three spending variables | 9,229.58 ± 3,230.98 | 8,397.00 | 0.912929 |

The three spending variables are R&D Spend, Administration, and Marketing Spend.
The baseline additionally uses State_Florida and State_New York.

ElasticNetCV and LassoCV tie for the best outer CV mean. The first method in
the fixed method order wins exact ties, so ElasticNetCV is saved as the selected
model. Its alpha is 762.6985859023436 and l1_ratio is 1.0; LassoCV selects the
same alpha. Because the chosen ElasticNet ratio is 1.0, these fitted methods
are equivalent, explaining their identical scores.

The selected model has test MAE 6,660.0808, test MSE 70,509,535.5800, and
training R2 0.9528. Its deployment example predicts profit **174,433.12**.
The preserved baseline example remains **175,581.62**.

The state indicators were not retained in the final selected models. Removing
them gives a small OLS validation improvement here; regularization gives the
lowest mean RMSE. However, fold-to-fold standard deviations of approximately
2,900–3,200 exceed the differences between method means. These data do not
establish a decisive advantage for one method.

Verification independently recomputed all six test metrics and outer CV
mean/std values from exported predictions and folds. All 30 outer-fold records
used matching splits across methods, and no holdout row appeared in outer
training/validation data. The selector encoding-equivalence audits passed.
All 31 exhaustive subsets, all five SFS feature counts, 140 ElasticNet parameter
pairs, and 35 Lasso alphas were evaluated on final training data. All eight PNGs
were valid, and both saved pipelines reproduced their example predictions.

## Teacher-style feature-count report

The new feature_count_report.py adds the requested two line plots and a table
showing one through five selected features. These reports are also generated
during the Evaluation stage when main.py runs.

To regenerate only these outputs:

```powershell
python feature_count_report.py
```

- outputs/teacher_feature_count_report.png: combined two-panel chart and table.
- outputs/feature_count_comparison.png: the two plots separately.
- outputs/feature_count_results.png: the striped results table.
- outputs/feature_count_results.csv: the four-column numerical table.
- outputs/teacher_results.md: a short explanation and table for submission.
- outputs/feature_count_cv_folds.csv: 25 outer-fold scores and selected features.
- outputs/feature_count_cv_predictions.csv: 200 out-of-fold predictions.

The metrics are five-fold outer CV means on the same 40 training observations.
Within each outer fold, forward SFS selects a subset of the specified size
using five inner folds. The feature lists in the summary table are selected
on all 40 training observations; individual outer folds can select different
subsets. The 10 test observations are not used to make these curves.
R-squared is the mean of fold R2 scores, rather than R2 from pooled predictions.

The smallest mean RMSE in this feature-count experiment uses three inputs:
R&D Spend, Administration, and Marketing Spend. Its mean outer RMSE is
9,266.777665 and mean outer R2 is 0.938115. One input has the highest mean
fold R2 (0.938374); averaging fold R2 normalizes by each fold's target variance,
so its ranking need not match mean fold RMSE. Model selection uses RMSE.

The results use this project's existing split and nested CV configuration,
so their numerical values differ from the example screenshot. Verification
recomputed every fold score from predictions, checked all training/validation
indices, confirmed the five-feature result matches the existing OLS CV score,
and validated the three generated PNGs.

## Local web app

The interactive web version is in web/. It includes model comparisons,
feature-count charts, teacher-report downloads, CRISP-DM notes, and predictions
using the saved trained models.

Start it from this project folder:

```powershell
python web/serve.py
```

Then open http://127.0.0.1:8000. After retraining, run
python web/export_data.py to refresh the exported results and model formulas.

See web/README.md for validation details and the prepared Vercel configuration.
The current handoff is for local review; no Vercel deployment has been made.

## Boston Housing and custom-dataset platform

The existing web app now includes a Dataset analysis workspace with 50 Startups,
Boston Housing and user-uploaded CSVs. Boston is validated as 506 rows with
13 numerical inputs and target medv. Custom uploads support target/input/model
selection, mixed column types and missing values.

The live workspace compares Multiple Linear Regression, Polynomial Regression,
Decision Tree, Random Forest, SVR and KNN using shared training/CV splits and
Pipeline/ColumnTransformer preprocessing. The original OLS/feature-selection
study and its plots remain available as the original 50 Startups study.

Start python web/serve.py and open http://127.0.0.1:8000/#analysis.
See web/README.md for limits, model settings, tests and the updated Vercel API
configuration. All three datasets and all six model families were tested locally.


## Reference feature parity audit

The web workspace now includes live five-method feature selection, dynamic encoded feature counts, OLS inference, equations, VIF, prediction contributions and outlier diagnostics for every supported dataset. See [FEATURE_PARITY_REPORT.md](FEATURE_PARITY_REPORT.md) for the original comparison matrix, implementation details, passed tests and explicit compute/browser limits. Start the local app with `python web/serve.py` and open http://127.0.0.1:8000/#analysis.
