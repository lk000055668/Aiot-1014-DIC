# StartupLab interactive regression platform

The existing website now has a Dataset analysis workspace backed by Python.
Its original 50 Startups study, feature-selection methods, teacher-report charts,
downloads, and browser profit calculator are preserved under the original-study
navigation.

## Run locally

From the Aiot-1014-DIC repository root:

```powershell
python -m pip install -r web/requirements.txt
python web/serve.py
```

Open http://127.0.0.1:8000/#analysis. Press Ctrl+C in the server terminal to stop.
Use --port 8001 if needed. The server serves only dist/ plus the JSON API.

## Dataset controls

- **50 Startups Dataset:** retains the original revision-pinned GitHub Gist URL,
  50 observations, four original inputs and default target Profit.
- **Boston Housing Dataset:** loads directly with pandas from
  https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv.
  The loader verifies 506 rows, 14 columns, the exact expected column names,
  numerical types, and default target medv. No deprecated Boston loader is used.
- **Upload Custom Dataset:** accepts a UTF-8 CSV. Select a numerical target,
  input columns and one or more model families, then Run analysis.

Built-in dataset changes automatically load the preview, statistics, columns,
default target and a comparison using the selected models. Custom uploads wait
for the user to choose a target. Target, input or model changes clear previous
results until the new analysis runs. In-flight requests are cancelled; revision
checks reject late responses from old dataset/configuration selections.

Uploads are limited to 1 MB, 2,000 rows and 50 columns; encoded inputs are limited
to 60 to bound interactive model fitting. CSV data is sent to the Python app
server for each operation and is not written to disk. The browser holds custom
CSV text for the current session. Built-in datasets are cached in server memory.
Rows with missing target values are removed and reported. Missing inputs are
imputed within training folds. Invalid quoting, duplicate headers, nonnumeric
targets, constant targets and incompatible configurations produce clear errors.

## Shared workflow

regression_workflow.py supplies dataset validation, previews, preprocessing,
analysis, prediction and model export. The original main.py now uses the same
build_preprocessing function, retaining its original transformer names and
encoding behavior.

Numerical inputs use training-median imputation. Categorical inputs use
training-mode imputation and OneHotEncoder(drop="first", handle_unknown="ignore").
Every model is a Pipeline with ColumnTransformer, cloned and fitted inside each
CV training fold. StandardScaler is used for polynomial, SVR and KNN models.
SVR also scales the target within TransformedTargetRegressor.

All methods use an 80/20 split with random_state=42 and the same five shuffled
training folds. Fixed settings are compared without hyperparameter tuning.
The winner is selected by mean training CV RMSE before holdout evaluation.
Every successful method reports R2, MAE, MSE, RMSE, training R2, and CV RMSE
mean/std. The shared charts render model comparisons, actual/predicted values
and residuals for arbitrary target scales, including negative values.

| Model family | Fixed settings |
| --- | --- |
| Multiple Linear Regression | Ordinary least squares |
| Polynomial Regression | Degree 2, no bias expansion; scaled before and after expansion |
| Decision Tree Regression | max_depth=6, min_samples_leaf=2, random_state=42 |
| Random Forest Regression | 100 trees, min_samples_leaf=2, random_state=42 |
| Support Vector Regression | RBF, C=10, epsilon=0.1, scaled features and target |
| K-Nearest Neighbors Regression | 5 neighbors, distance weights, scaled inputs |

The original six OLS/feature-selection approaches are distinct from these six
model families. They remain available in the original 50 Startups study.
Its reported winner is not mixed with the new family-comparison results.

## CRISP-DM and deployment

The analysis shows all six CRISP-DM phases, dataset and preparation details,
model metrics, diagnostics, and a target prediction form. Each analysis checks
that saving/reloading the winning full pipeline preserves holdout predictions.
Users can download the selected model as a .joblib file and model results as CSV.

Prediction and model download reconstruct the same deterministic training split
and fit only the chosen model. Requests carry the full configuration, making
them independent of server-side sessions or temporary files. Test rows remain
excluded. Downloaded pipelines contain their fitted preprocessing and estimator;
load them with compatible Python/scikit-learn versions.

api/index.py is shared by the local server and the prepared Vercel file-based
Python function. It implements /api/dataset, /api/analyze, /api/predict,
/api/model and /api/health. API responses use no-store caching.

## Preserve/refresh the original study

After rerunning the original main.py, refresh only its archived web snapshot:

```powershell
python web/export_data.py
```

That snapshot's 120 reference prediction checks remain independent of the new
live API. The legacy fitted pickle files are not served publicly.

## Verification

With the local server running, from web/:

```powershell
python test_platform.py
node verify_platform.mjs
node verify_web.mjs
python check_http.py
```

The JavaScript scripts require Node.js on PATH, but Node.js is not needed to
serve or use the app. The scripts were executed using the available Node runtime.

Tests cover all three dataset options, all six model families, metric
recalculation, identical train/CV splits, missing targets and inputs, unseen
categories, invalid CSV/configurations, network/schema errors, prediction and
downloaded-pipeline equivalence, training-only preprocessing, and stale-response
rejection. The original prediction/check suite also passes.

The session has no connected browser automation. API and JavaScript state/chart
checks passed; actual browser layout and interaction QA need local review.

## Vercel — after local review

Project root: web. The static output remains dist/.
The updated vercel.json routes /api/:operation to api/index.py, configures
a 60-second function duration, and excludes unnecessary files from its bundle.
requirements.txt contains only API runtime packages. Python source is now
required by the deployment; the old static-only ignore rule has been removed.

Reference:
https://vercel.com/docs/functions/runtimes/python/api-directory

No Vercel deployment or account change has been performed. Hosted runtime,
dependency packaging, and request-duration behavior must be verified at deployment.


## Live feature selection and OLS inference

In Dataset analysis, choose a dataset, numeric target and inputs, then select **Compare five selectors**. This sits alongside the existing six-regression-model comparison. Use the algorithm dropdown and encoded-feature-count slider to inspect equations, coefficients, classical OLS tests/intervals, VIF, residuals and predictions. The chart display can overlay all five methods while retaining an explicit detail algorithm.

Selection methods: Lasso ranking, Forward SFS, RFE, ElasticNet ranking and exact exhaustive search. Each subset is refitted with OLS. The full-k model is the all-input baseline. Five outer training folds evaluate the selection procedure; three inner folds perform selection/tuning. The 80/20 seed-42 split and all original dataset sources remain.

The recommendation uses nested training CV. Charts display holdout diagnostics; confidence intervals and p-values after feature selection are labeled exploratory, not selection-adjusted or causal. Undefined inference/VIF is explicitly labeled. The optional descriptive all-row refit is separate from holdout evaluation and prediction.

Selection supports up to 30 encoded predictors and 45 seconds per request. Exhaustive search evaluates at most 256 combinations per k. For Boston with all 13 inputs, counts 3–10 are skipped with an explanation; other methods cover all counts. Narrow the input selection for larger exact searches. The six-model workspace retains its existing 60-encoded-input limit.

New POST routes: /api/selection, /api/selection-predict, /api/selection-descriptive. Selection payloads reuse dataset/csv/target/features; prediction adds algorithm, integer k and input. JSON evidence, CSV results and SVG chart downloads are available in the panel.

Verification: install requirements-test.txt, run check_selection_run.py to generate controller fixtures, then test_selection.py and verify_selection.mjs. Existing test_platform.py, verify_platform.mjs, verify_web.mjs and check_http.py remain. The reference API audit test uses the temporary regression-reference-audit clone. See ../FEATURE_PARITY_REPORT.md for the full audit, evidence, reproducible commands and unverified browser/deployment limits.
