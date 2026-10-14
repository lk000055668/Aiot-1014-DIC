# Feature Parity Report

## Audit baseline — before implementation

Date: 2026-10-14. Reference commit: `8b65e3bd9745e650ae7bea28f9562c059d5029ed`. Inspected separate temporary clone at `C:/Users/JunKai/AppData/Local/Temp/regression-reference-audit`.

Sources: [repository](https://github.com/huanchen1107/AIoT-DIC2-MultipleLR), [demo](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/). Read README, main.py, feature_progression_analysis.py, app.py, api/index.py, both HTML frontends and inline JavaScript, JSON/JS model registry, CSV artifacts, requirements and deployment workflow. The registry generation code for full inferential statistics is not present in this checkout.

Existing project inspected: main.py, model_comparison.py, feature_count_report.py, web/regression_workflow.py, API, local server, export script, both browser scripts, HTML/CSS, tests, artifacts and documentation. Existing selectors operate only in the original Startups study; the new dataset workspace currently compares model families, not feature selection methods.

### Complete functional inventory / initial comparison matrix

| Feature | Reference Implementation | My Implementation (before) | Status | Required Changes | Priority |
|---|---|---|---|---|---|
| OLS | main.py baseline; web full-k OLS | Live shared pipeline | Complete | Add detailed OLS inference | P1 |
| Lasso (L1) ranking | Standardized LassoCV top-k, then OLS | Archived Startups LassoCV only | Partial | Live dataset-aware ranking path | P0 |
| Forward SFS | Training CV greedy subsets | Archived Startups SFS only | Partial | Live training-only selection | P0 |
| RFE | Standardized OLS backward removal | Archived RFECV only | Partial | Exact-k RFE on active data | P0 |
| ElasticNet ranking | Standardized ElasticNetCV top-k | Archived Startups regularized estimator | Partial | Live top-k ranking + OLS | P0 |
| Exhaustive selection | Progression selects on test RMSE (leakage) | Archived training-CV Startups search | Partial | Bounded training-only search | P0 |
| Compare five selectors | Five overlaid series; ALL detail falls back to Lasso | Archive summary only | Partial | Live comparison across active data | P0 |
| Interactive k | Fixed 1–5 buttons and chart clicks | Archived SFS 1–5 | Partial | Dynamic encoded feature count | P0 |
| RMSE vs k | Precomputed test RMSE | Archived nested-CV SFS chart | Partial | Live algorithm paths | P0 |
| R² vs k | Precomputed test R² | Archived nested-CV SFS chart | Partial | Live algorithm paths | P0 |
| Dynamic equation | Registry coefficients; Profit hardcoded | No current-dataset equation | Missing | Equation from fitted OLS coefficients | P0 |
| Coefficient table | Precomputed registry beta | Archived coefficients only | Partial | Active OLS coefficients | P0 |
| SE / t / p | Registry numbers displayed | Absent | Missing | Training OLS covariance, df and t distribution | P0 |
| 95% confidence intervals | Registry intervals | Absent | Missing | Classical exploratory OLS intervals | P0 |
| VIF | Registry; main.py statsmodels | Absent | Missing | Intercept-aware auxiliary regressions | P1 |
| Interactive prediction | Hardcoded Startup inputs; API and JS | Live six-model prediction on all datasets | Complete | Support selected subset / k | P0 |
| Feature contributions | Coefficient × input in API and frontend | Archived calculator contributions only | Partial | Encoded contributions + intercept | P1 |
| Residuals / top outliers | Web test top10; CLI descriptive full-data refit | Live residual chart and unsorted table | Partial | Top10 absolute residuals and percentage error | P1 |
| Evaluation / summary | MAE MSE RMSE R², adjusted R², CV, F | Live four metrics and CV | Partial | Training adjusted R², F and model summary | P1 |
| CRISP-DM | Six CLI phases; web badge/footer | Live six phases | Complete | Extend explanations for feature selection | P1 |
| Chart interaction / optimum | Hover/click; hardcoded k2 optimal and k5 warning | Archive k buttons; no live selector charts | Partial | Data-derived CV winner; selectable chart points | P1 |
| CSV / figure exports | Offline CSV/PNG generation | Existing CSV/teacher PNG/model downloads | Complete | Live selector CSV and chart export | P2 |
| Actual-predicted scatter | CLI scatter / reference diagonal / outlier labels | Shared live scatter with diagonal | Complete | Reuse for selected subset | P1 |
| Summary / prediction APIs | Flask serves precomputed model registry | Dynamic Python JSON APIs | Complete | Add selection and prediction endpoints | P1 |

### Reference discrepancies and additional behavior

- Main CLI uses seed 42; progression/registry uses seed 0. Different holdout scores are expected. Preserve this project’s seed 42 and source URL.
- Exhaustive progression evaluates candidates against the test set. This must not be reproduced; choose on training CV only.
- Main CLI CV evaluates already-selected fixed subsets (explicitly documented in source); it is not nested selection evaluation.
- Frontend hardcodes k=2 optimum, k=5 overfitting claims, a significance badge and F fallback 347.8. These are not valid for arbitrary data.
- Compare All overlays all five curves but coefficient/prediction details silently use Lasso. Our detail model must be explicit.
- Web registry supplies coefficients/statistics/outliers; API reads this static registry. Browser predictions are calculated locally. CLI full-data refit outliers are descriptive, while web outliers are holdout.
- Extras: adjusted R², F/p model test, chart hover/click, feature chips, input sliders, CSV/PNG exports, actual-versus-predicted reference line, full-data descriptive scatter trend, Flask summary endpoint, static GitHub Pages fallback and Vercel adapter. Hosting technology and visual styling are not analytical parity requirements.
- Existing Boston, upload, missing-value handling, six regression families, model downloads and all legacy pages must remain.

### Baseline behavior evidence

Initial test command from project root failed because existing tests expect web/ on the Python path; corrected to documented `python test_platform.py` from web/. All seven baseline integration tests passed before implementation. Browser demo retrieval via web reader failed; source behavior is distinguished from browser verification.

## Implementation and final verification

The initial matrix above was written before application implementation changes. Initial Complete: 6/24; Partial: 14/24; Missing: 4/24. Final coverage and verification follow.


### Final comparison matrix

The descriptive refit was included in the initial additional-feature inventory and is counted separately as capability 25 here. Complete means implemented within the supported bounds; it does not imply real-browser QA passed.

| Feature | Final status | Changes implemented / retained | Passed evidence |
|---|---|---|---|
| OLS | Complete | All-input baseline and selected subsets refit OLS | Independent coefficients and holdout metrics on 102 configurations |
| Lasso (L1) ranking | Complete | Fold-local scaling, inner-CV alpha, magnitude ranking and OLS refit | All k on all three datasets |
| Forward SFS | Complete | Greedy three-fold inner training RMSE selection | All k; outer-fold refits and holdout perturbation tests |
| RFE | Complete | Backward standardized OLS coefficient elimination | All k on all datasets |
| ElasticNet ranking | Complete | Inner-CV alpha/L1 ratio, standardized ranking, OLS refit | All k on all datasets |
| Exhaustive selection | Complete (bounded) | Exact training-CV combinations, bounded to 256 candidates per k | All Startups/custom counts; Boston k=1,2,11,12,13; unsafe counts rejected |
| Compare five selectors | Complete | Overlay curves, full comparison table, explicit detail algorithm | API and client controller checks |
| Interactive k | Complete | Slider from 1 to training-encoded feature count | Startups5, Boston13, custom4; k changes equation and tables |
| RMSE vs k | Complete | Holdout metrics with visible gaps for skipped counts | Independent metric recomputation and finite SVG coordinates |
| R² vs k | Complete | Holdout R² including negative values | Independent sklearn scores and chart fixtures |
| Dynamic equation | Complete | Original-unit OLS coefficients and encoded terms | Equation matches every holdout prediction |
| Coefficient table | Complete | Intercept and selected predictors for current configuration | Statsmodels agreement on all 102 configurations |
| SE / t / p | Complete | Classical training OLS inference with rank/df guards | Statsmodels agreement and singular-design fixture |
| 95% confidence intervals | Complete | Student-t interval, unavailable if inference is unidentified | Statsmodels confidence interval agreement |
| VIF | Complete | Intercept-aware auxiliary regression, constant/infinite labels | Statsmodels agreement and collinearity tests |
| Interactive prediction | Complete | Current dataset, algorithm, k and fitted preprocessing | 30 predictions across 5 algorithms × 2 counts × 3 datasets |
| Feature contributions | Complete | Intercept plus encoded value × coefficient | Sums and actual encoded values verified independently |
| Residuals / top outliers | Complete | Top10 absolute holdout errors and percent errors | Independent ordering and zero-target percentage handling |
| Evaluation / summary | Complete | Four test metrics, nested CV, training adjusted R², F/p, df/error, AIC/BIC, condition number | Independent metrics and Statsmodels summary checks |
| CRISP-DM | Complete | Existing six phases retained with selection methodology | Original API suite and selection controller execution |
| Chart interaction / optimum | Complete | Keyboard-focusable points and training-CV recommendation | Controller and SVG tests; real-browser interaction unverified |
| CSV / figure exports | Complete | Selection CSV/JSON/SVG plus preserved teacher PNG | Actual export callbacks/content and HTTP downloads |
| Actual-predicted scatter | Complete | Shared current-target scatter, residual chart and reference line | Finite chart fixtures and all-model compatibility |
| Summary / prediction APIs | Complete | New selection, selection-predict, selection-descriptive routes; old APIs retained | HTTP tests and reference Flask test-client execution |
| Descriptive all-row refit (additional capability) | Complete | Optional full-data coefficient refit; separate trend and ranked outliers | Independent all-row predictions on all datasets and stale-response test |


### Changes and architecture

- Added web/selection_workflow.py, reusing regression_workflow.py dataset loading, validation, splitting, ColumnTransformer, imputers, encoding and shared prediction input validation. The professor's source was a functional reference, not copied into the app.
- Extended the existing Python API with POST /api/selection, /api/selection-predict and /api/selection-descriptive. Existing dataset, six-model analysis, prediction and downloadable pipeline routes remain.
- Added selection.js and a panel within the existing dataset workspace. Original navigation, archived Startups results, teacher reports, model comparison and calculator remain.
- Dataset/target/input edits abort requests and hide old results. Algorithm/k changes update equations, statistics, diagnostics and charts together and invalidate older predictions/descriptive refits. Comparing all five curves does not silently change the detail algorithm.
- Escaped uploaded target names in shared SVG labels. User-provided names and table values are escaped or assigned through textContent.
- Added SciPy to runtime requirements for statistical distributions. Statsmodels and Flask are test-only requirements in web/requirements-test.txt; audit dependencies were installed in a separate temporary directory.
- Updated Vercel exclusions to omit tests/evidence. No deployment was performed; the user's local-review-first requirement remains in effect.

### Statistical and evaluation design

1. The existing 80/20 split and seed 42 remain. The Startups data URL is unchanged. Boston still loads through pandas from the supplied URL and validates 506 rows, 14 numerical columns and medv.
2. k counts encoded predictors, excluding the intercept. RFE is exact-k backward elimination, distinct from archived RFECV. Lasso/ElasticNet are ranking selectors followed by OLS; the displayed inference is for OLS, not a penalized estimator.
3. Each of five outer training folds reconstructs selection paths from that fold's training partition. SFS/exhaustive use three inner folds. Lasso/ElasticNet tune with separately fitted inner-fold preprocessing, input scaling and target scaling. Alpha candidates are 0.001, 0.01, 0.1, 1; ElasticNet L1 ratios are 0.2, 0.5, 0.8. Zero-coefficient ties sort deterministically by feature name.
4. The recommendation is frozen using mean outer training RMSE before holdout scores are calculated. Holdout curves are exploratory: choosing a model after examining them would consume the untouched test set. The minimum CV score over many configurations is not itself an unbiased final estimate.
5. Inner-fold vocabulary is learned only from inner training rows; absent categories align as zero columns. When an outer fold has fewer encoded categories, effective k is capped to available predictors and reported in fold evidence with a warning. Every fold subset and split is available in the JSON download.
6. Equation and calculator use the same training OLS coefficients. Standard errors use residual variance and the design pseudoinverse; Student-t inference uses n minus design rank degrees of freedom. VIF auxiliary regressions include an intercept. Adjusted R², F, AIC and BIC refer to the training fit and are labeled accordingly.
7. Classical post-selection inference is explicitly exploratory, conditional on the selected design, not selection-adjusted or causal. Rank-deficient, saturated and effectively exact-fit designs display unavailable inference instead of fabricated significance. Constant predictors and infinite VIF are distinguished. Zero targets yield undefined percentage errors.
8. The optional descriptive refit uses all valid target rows only on request, retains training-derived preprocessing/selection, and never replaces holdout scores, statistical tables or the training-model predictor.

### Preserved capabilities

All three dataset options, custom target/input selection, automatic type detection, numerical/categorical imputation, one-hot encoding, all six regression families, consistent splits, four evaluation metrics, fitted-model downloads, CRISP-DM cards and original Startups pages remain. The original 120 browser prediction fixtures still match their exported Python pipelines.

### Test results and evidence

| Verification | Final result | Evidence |
|---|---|---|
| Existing API integration suite | 7/7 passed, 7.543 seconds | web/test_platform.py; web/verification_results.json |
| Feature parity / independent statistics suite | 8/8 passed, 27.159 seconds | web/test_selection.py; web/selection_verification.json |
| Startups selection | 25 configurations, max k=5 | 3.31 seconds in final concurrent test run; CV recommendation RFE k=3 |
| Boston selection | 57 configurations, max k=13 | 9.88 seconds; CV recommendation full-input OLS (Exhaustive k=13) |
| Mixed custom CSV | 20 configurations, max k=4 | 1.40 seconds; selected target, missing inputs/categories and missing targets tested |
| Independent Statsmodels inference | All 102 configurations passed | Coefficients, SE, t, p, CI, VIF, F/p, adjusted R², AIC/BIC |
| Leakage checks | Passed | Changing only holdout values leaves training subsets/statistics/CV unchanged; recorded preprocessing fit indices stay in training partitions |
| Predictions/contributions | Passed | Five selectors at k=1/full for all three datasets; imputation and unseen categories |
| Invalid/rank-deficient inputs | Passed | Invalid k/algorithm/target/CSV, unsafe exhaustive count, singular/constant/exact-fit design, zero target, deadline guard |
| Descriptive refits | Passed on all three datasets | All-row sklearn predictions agree; top10 ordering correct and explicitly not held-out |
| Real selection JS controller in minimal DOM adapter | Passed | k/algorithm equations/tables/charts, export content, stale analysis/prediction/descriptive responses, safe custom labels |
| Existing JS suites | Passed | 120 original predictions, input validation, chart coordinates and dataset/configuration invalidation |
| Local HTTP assets/downloads | 14/14 return HTTP 200 | web/check_http.py, including selection.js and all legacy report downloads |
| Reference Flask API | Passed | 25 configurations; model/summary routes; prediction contribution sums; invalid algorithm 400; template rendered |
| Actual browser rendering / console | NOT VERIFIED | No connected browser surfaces available |
| Vercel deployment | NOT PERFORMED | Local review requested first |

Run these commands from web/, with the local server running:

~~~text
python serve.py
python -m pip install -r requirements-test.txt
python check_selection_run.py
python test_selection.py
python test_platform.py
python check_http.py
node verify_selection.mjs
node verify_platform.mjs
node verify_web.mjs
~~~

The environment lacks a node executable on PATH, so the three MJS checks were executed through the available embedded Node runtime. check_selection_run.py creates the built-in fixtures consumed by verify_selection.mjs. The reference test expects the separately cloned repository at the temporary regression-reference-audit directory. Test dependencies can be installed normally with requirements-test.txt; this audit used temporary isolated dependencies.

Before changes, all seven existing integration tests passed. Initial root-level test invocation failed due to its import path; using the documented web directory resolved it. A reference Flask test-loader issue initially caused an incorrect template root; registering its imported module in sys.modules resolved that harness issue without changing reference code. The new API JSON test caught a NumPy boolean serialization issue, which was corrected before final testing.

The supplied live demo could not be read by the web reader. Browser inventory returned apps=[], browsers=[]. Thus deployed demo behavior, actual rendered responsiveness, native keyboard interaction and browser console are not certified. Syntax checks and execution of real JavaScript callbacks in a minimal DOM adapter are useful automated evidence, not browser rendering tests.

### Remaining limitations

- Feature-selection paths support up to 30 encoded inputs and a 45-second compute budget. The preserved model-family workspace supports 60 encoded inputs. Upload limits remain 1 MB, 2,000 rows and 50 columns.
- Exact exhaustive search allows at most 256 subsets per k. Boston k=3–10 is explicitly skipped, not approximated. Narrow input columns to examine those sizes exactly.
- Real-browser visual/console review remains necessary. The local server is available at http://127.0.0.1:8000/#analysis.
- New dataset analysis requires the Python API; a static precomputed registry cannot provide it. Vercel runtime packaging and deployment have not been verified.
- Fixed selector grids and model settings are documented. Results need not reproduce the professor's seed-0, partly test-selected benchmark values.
- Aborting a browser request rejects stale results but does not forcibly stop an already-running Python thread. The compute budget bounds that work. Application endpoints do not save uploaded datasets.

### Final feature coverage

- **Analytical capability implementation: 25/25 = 100%**, within the declared input and exhaustive-search bounds.
- **Configuration execution in the main three-dataset fixtures: 102/110 = 92.73%**: Startups 25/25, Boston 57/65, custom 20/20. Eight Boston exhaustive counts are intentionally blocked and are not claimed as fitted or tested models.
- Each implemented analytical capability has automated server, mathematical, controller or export evidence as identified in the matrix. This does **not** establish browser end-to-end coverage. **Real-browser visual/console verification: not performed.**
