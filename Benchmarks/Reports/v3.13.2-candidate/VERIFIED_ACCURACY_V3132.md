# Verified Scientific Correctness & Accuracy Report — SwiftSci `v3.13.2-candidate` (Stage A)

- **Status:** `CANDIDATE` (Stage A — AC Power Verified, Uncommitted Working Tree)
- **Candidate Base Commit:** `0cf99f0470d0757d0a02bd3244c62d821ab144b0` (`v3.13.1`, `dirty: true`)
- **Stage A Run Snapshot `tree_sha256`:** `5abd55baabb35be660ab990e97eb37bd009c9d8a955b09c447d290a083bdee30`
- **Final pre-release review source fingerprint:** `953e773629eabf731530b82edd6d74813e8d5f4b22bfec25b412ba9951eb4a9c` (four DocC pages and one whitespace-only edit in `ARIMA.swift`; benchmark executable inputs, scripts, and thresholds are unchanged apart from that source whitespace).
- **Mandatory G-037 Verdict:** **`FIXED AND VERIFIED`**

---

## 1. Mandatory Release Gate G-037: ARIMA / SARIMA `fittedValues` Correctness

**Explicit Verdict:** **`FIXED AND VERIFIED`**

### 1.1 Root Cause Analysis (`v3.13.1`)

In `v3.13.1`, both `ARIMAModel.integrate(fittedDiff:original:d:isForecast:)` (`Sources/SwiftForecast/Core/ARIMA.swift`) and `SARIMAModel.integrate(fittedDiff:original:d:D:s:isForecast:)` (`Sources/SwiftForecast/Core/SARIMAModel.swift`) reconstructed historical in-sample `fittedValues` (`isForecast == false`) via **open-loop cumulative summation** instead of closed-loop one-step-ahead integration anchored on the observed series:

- Non-seasonal integration initialized `var lastVal = levelSeries[0]` and iterated `val = lastVal + diffVal; lastVal = val` — feeding back prior *predicted* levels across the entire history.
- Seasonal integration iterated `val = reconstructed[i] + current[i]` — feeding back prior *predicted* seasonal levels instead of observed `subLevelSeries[i]`.
- Because `diffResiduals[t] = diffSeries[t] - diffFitted[t]` was computed closed-loop, `residuals` reflected true one-step innovations, whereas `fittedValues` drifted as a cumulative random walk. For `ARIMAModel` with exogenous variables (`ARIMAX`), `fittedValues` also omitted `withExogenousComponent(...)`.
- Consequently, Saura had to reconstruct aligned historical lines via the display-side workaround `fitted[t] = actual[t] - residual[t - warmUp]`.

### 1.2 Fix & Mathematical Contract (`v3.13.2-candidate`)

- `ARIMAModel.integrate` (`isForecast == false`) now reconstructs `reconstructed.append(levelSeries[i] + diffVal)` after seeding `levelSeries[0]`, and `performFit` applies `withExogenousComponent(integratedARIMA, exog: exog)`.
- `SARIMAModel.integrate` (`isForecast == false`) reconstructs both non-seasonal (`d`) and seasonal (`D * s`) layers closed-loop from observed `subLevelSeries[i] + diffVal` after seeding initial boundary points.
- Exact mathematical definitions & index alignment:
  - `warmUp = order.d` (`ARIMAModel`) or `warmUp = order.d + seasonalOrder.D * seasonalOrder.s` (`SARIMAModel`)
  - `fittedValues.count == series.count`
  - `residuals.count == series.count - warmUp`
  - For `t in 0..<warmUp`: `fittedValues[t] == series[t]` (seed boundary observations).
  - For `t in warmUp..<series.count`: `fittedValues[t]` is the **one-step-ahead prediction on the original data scale**, satisfying the exact identity:

$$\text{residuals}[t - \text{warmUp}] = \text{series}[t] - \text{fittedValues}[t]$$

### 1.3 G-037 Verification Evidence

- **Regression Test Suite (`Tests/SwiftForecastTests/ARIMASARIMAFittedValuesTests.swift`):**
  - Verifies exact causal one-step prediction equality and `series[t] - residuals[t - warmUp] == fittedValues[t]` across `ARIMAModel` (`d = 0, 1, 2`), trivial `ARIMA(0, 1, 0)`, `ARIMAX(1, 1, 0)` & `(0, 1, 0)`, `SARIMAModel` (`D = 1, 2`), `AutoARIMA` on `AirPassengers` (`n = 120`), and non-leakage under future `+5000.0` terminal perturbation at `t = 143` (tolerances `< 1e-9` / `< 1e-10`).
- **Stage A Scientific Validation (`Benchmarks/Runs/v3.13.2-candidate/scientific_validation.json`):**
  All 9 applicable ARIMA/SARIMA/AutoARIMA workloads and 6 rolling-origin folds passed G-037.
  - `ts_sarima_airpassengers_144_g037`: `warm_up=13`, `fitted_count=144`, `residuals_count=131`, `max_warmup_diff=0.0`, `max_identity_diff=5.684e-14`, `saura_formula_exact_match=true`, in-sample `RMSE=28.3051`, `R²=0.8554`, `late_1960_max_err=23.7976`.
  - `sarima_ap_ro` (origins `108/120/132`): `max_warmup_diff=0.0`, `max_identity_diff=5.684e-14`, `saura_formula_exact_match=true` across all 3 origins.
  - `ts_arima_syn500_matched_hr`, `ts_arima_dgp_syn500_matched_hr`, `arima_dgp_ro` (3 origins): `warm_up=1`, `max_warmup_diff=0.0`, `max_identity_diff=1.421e-14`, `saura_formula_exact_match=true`.
  - `ts_arima_syn50k_vs_statespace`, `ts_arima_dgp_syn50k_matched_hr`: `warm_up=1`, `fitted_count=49976`, `residuals_count=49975`, `max_warmup_diff=0.0`, `max_identity_diff=5.684e-14`.
  - `ts_auto_arima_valid`: `warm_up=0`, `max_warmup_diff=0.0`, `max_identity_diff=0.0`.

---

## 2. Other Targeted Patch Items

- **G-033 (`ARIMA`, `SARIMAModel`, `AutoARIMA`, `ForecastValidation`):**
  Added `ForecastValidation.ensureFinite`, `ForecastValidation.ensureMAInvertible`, `AutoARIMA.isMAPolynomialInvertible(coefficients:)` (negating MA/SMA coefficients to match Schur-Cohn `1 - sum a_k B^k` convention), and `resetFittedState()`. Non-invertible or non-finite models fail closed (`isFitted == false`) with typed `ForecastError`. Verified by `ts_sarima_expect_fail_syn500` & `ts_sarima_expect_fail_syn50k` (`SwiftMLError`, `is_fitted=false`), `ts_auto_arima_poly_sign_check` (`[true, true]`), `ts_auto_arima_poly_unstable_check` (`[false, false]`), and `ts_auto_arima_zero_variance` (`[0,0,0,0,0,0,1]`).
- **SwiftStats DocC (`Stats.skewness` and `Stats.kurtosis`):**
  Public `ddof = 1` sample-denominator behavior preserved under API Contract Freeze; DocC in `Stats+Descriptive.swift` and `DescriptiveStatistics.md` updated to document exact formulas and differences vs standard Fisher `g1/g2` (`scipy.stats(..., bias=True)`). Verified on `stats_ddof1_small5` (`kurt=-1.912`, `diff=4.44e-16`) and `stats_ddof1_100k` (`skew_diff=2.71e-17`, `kurt_diff=2.13e-14`).
- **G-035 / G-036 (`TreeSHAP`, `ModelInterpretability.md`, `ShapleyAttributions.md`):**
  Corrected complexity to `O(T · 2^K · L)` exact (`K <= 12`) and deterministic `2048`-permutation Monte Carlo (`K > 12`); removed false GBDT claim; explicitly deferred G-035 (GBDT TreeSHAP) and G-036 (classifier probability TreeSHAP). Exact coalition oracle verified on `explain_treeshap_dt_k4` (`max_coalition_phi_diff=0.0`, `additivity=1.78e-15`), `explain_treeshap_rf_k8` (`4.44e-16`), and MC additivity on `explain_treeshap_dt_k16_mc` (`7.37e-14`).

---

## 3. CPU Acceptance & Auxiliary Conformance Controls (`acceptance-cpu`)

- **CPU Conformance Coverage:** `16/16` profiles complete (`coverage_complete: true`), **`0` SwiftSci failures**.
- **Passing Profiles (`14/16`):** `vision-conformance` (`16/16`), `boundary-cpu-conformance` (`8/8`), `certification` (`54/54`), `controlled-conformance` (`20/20`), `dataframe-conformance` (`48/48`), `migration-smoke` (`72/72`), `model-conformance` (`6/6`), `neural-cpu-conformance` (`10/10`), `public-smoke` (`22/22`), `smoke` (`28/28`), `supervised-conformance` (`12/12`), `scientific-workflow` (`8/8`), `persisted-workflow` (`12/12`), `neural-shaped-cpu-conformance` (`12/12`).
- **Expected Pandas Known Failures (`2/16` profiles, `4` cases):** `numerical-binary64` (`36/38`: `nist-wampler4-binary64`, `nist-wampler5-binary64` `pandas`) and `numerical-conformance` (`36/38`: `nist-wampler4-decimal`, `nist-wampler5-decimal` `pandas`). `regression-policy.json`: `blocking: []`.
- **Pretrained Inspection Preflight (`check_pretrained_inspection.py`):** `passed` (`60` cases, `0` failures, `generation_executed: false`).
- **Workflow Worker Mutation Controls (`check_public_workflow_workers.py`):** `passed` (`100` controls, `0` ineligible).

---

## 4. Scientific Validation Accuracy Matrix (`scientific_validation.json`: `39/39` Passed)

### 4.1 Rolling-Origin Time-Series Backtests (`3/3` Passed)

| Suite | Origins | Swift Mean RMSE | Python Mean RMSE | Seasonal Naive Mean RMSE | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `hw_syn500_ro` | `428, 452, 476` (`h=24`) | **`1.5110`** | `1.5540` | `5.3479` | Passed (`1.5249 / 1.4609 / 1.5472`) |
| `arima_dgp_ro` | `428, 452, 476` (`h=24`) | `5.2235` | `5.0715` | `4.2768` | Passed + G-037 `<= 1.42e-14` |
| `sarima_ap_ro` | `108, 120, 132` (`h=12`) | `32.5422` | `30.9928` | `38.9917` | Passed (`23.0768 / 47.8993 / 26.6507`) + G-037 `5.68e-14` |

### 4.2 Supervised ML, Clustering, Anomaly Detection, Explainability, Stats, & LLM

| Workload | Key SwiftSci Accuracy Metrics | Python Baseline Metrics | Gate Outcome |
| --- | --- | --- | --- |
| `ts_hw_syn500_opt` / `_fixed` / `syn50k_fixed` | `RMSE = 0.7500 / 1.5472 / 1.4210`, `R² = 0.9869 / 0.9444 / 0.9459` | `RMSE = 0.7121 / 1.6272 / 1.4358` | Passed (`MASE 0.19–0.84`) |
| `ts_sarima_airpassengers_110_010` / `_011_011` | `RMSE = 75.1505 / 73.4972` (beats Naive `76.9946`) | `RMSE = 74.2522 / 71.9600` | Passed (`RMSE gap <= 1.54`) + G-037 |
| `anomaly_syn144_s04` | Precision `1.0`, Recall `1.0`, `F1 = 1.0` (`[30, 75, 120]`) | — | Passed |
| `anomaly_syn500_holdout` (`thr=5`) | Precision `0.8333`, Recall `1.0`, `F1 = 0.9091` (`tp=5, fp=1`) | — | Passed |
| `anomaly_syn500_trended_diagnostic` (`thr=3`) | Precision `0.20`, Recall `1.0`, `F1 = 0.3333` (`tp=5, fp=20`) | — | Passed (documented `medTrend` edge effect) |
| `ml_ols_regression` | `R² = 0.998018`, `RMSE = 0.245219` | `R² = 0.998018`, `max_abs_pred_diff = 5.51e-14` | Passed |
| `ml_logistic_regression` | Acc `0.9600`, Macro-F1 `0.9600` | Acc `0.9575`, Macro-F1 `0.9575` | Passed (GD vs L-BFGS) |
| `ml_linear_svc` | Acc `0.9575`, Macro-F1 `0.9575` | Acc `0.9575`, Macro-F1 `0.9575` (`gap = 0.0`) | Passed |
| `ml_decision_tree_regressor` / `_classifier` | Reg `R² = 0.6004`, Cls Acc `0.7450` | Reg `R² = 0.6015`, Cls Acc `0.7450` (`gap = 0.0`) | Passed |
| `ml_random_forest_regressor` / `_classifier` | Reg `R² = 0.6044`, Cls Acc `0.8600` | Reg `R² = 0.6620` (`gap=0.0576`), Cls Acc `0.8500` | Passed |
| `ml_gbdt_regressor` | `R² = 0.7289`, `RMSE = 1.4225` | `R² = 0.7311`, `RMSE = 1.4167` (`gap = 0.0022`) | Passed |
| `ml_hist_gbdt_regressor` / `_classifier` | Reg `R² = 0.7861`, Cls Acc `0.8625` | Reg `R² = 0.7821`, Cls Acc `0.8650` | Passed |
| `cluster_pca` / `cluster_kmeans` | `EVR = 0.97188233`, `WCSS = 6528.64305` | `EVR abs diff = 4.44e-16`, `WCSS rel gap = 5.57e-16` | Passed |
| `explain_treeshap_dt_k4` / `_rf_k8` / `_dt_k16_mc` | Coalition diff `0.0 / 4.44e-16` (`K=4/8`), additivity `<= 7.37e-14` | Exact oracle verified (`K<=12`); seeded 2048 MC (`K=16`) | Passed (G-035/G-036 deferred) |
| `stats_ddof1_small5` / `_100k` | `kurt=-1.912` (`small5`), `skew_diff=2.71e-17` (`100k`) | Analytic SciPy conversion diff `<= 2.13e-14` | Passed |
| `llm_kvcache_ablation_p24_d16` / `_p64_d24` | Greedy decode `max_abs_logit_diff = 3.58e-07` / `4.77e-07` | Internal ablation vs full-forward | Passed |
