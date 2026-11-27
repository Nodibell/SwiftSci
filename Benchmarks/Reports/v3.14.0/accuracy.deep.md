# SwiftSci 3.14.0 Deep Accuracy & Conformance Audit

> **Navigation**: [← v3.14.0 Evidence Index](README.md) · [Accuracy Overview](../../../ACCURACY/3.14.0.md) · [Deep Performance Audit](performance.deep.md) · [Performance Overview](../../../PERFORMANCE/3.14.0.md)

Branch: `release/3.14.0` · Base commit: `10c11b0c586663484e0f982f7afd52834d3924d8` (`dirty: true` with `3.14.0` changes)

- Source tree SHA-256 (`summary.json` / `suite.json`): `c6cd7882049ccb65c485c5d113880a44615a5fbb553137ef13abe7bfb79fa2b7`
- Benchmark worker SHA-256 (`SwiftSciBenchmarkWorker`, on disk & `scientific_validation.json`): `0f75343624dbaf306c452262adb5e675b2436cf60aa538e505b336edd74e9968`
- Scientific validation JSON SHA-256: `e0d38b0d9aa527d36ca6b84bde511278014e0f7ca8bdd8b5c7a8f9e8cb22dbdc`
- Release-gate `acceptance.json` SHA-256: `d09e640e37638601761cf36829e36d193232011bb28551c19f2828f5dc2df15b`
- Release-gate `summary.json` SHA-256: `7e771fa8d48b96a3b942bf7ac195f5c2ac94d3beafbd0ed5913e759d8273aefa`
- Comparison `suite.json` SHA-256: `ab111e4068d067e21931894e0c7465b8d0575f21a1b3d7cbfe67a543af215222`
- Comparison `measurements.json` SHA-256: `751a7d3c63905827db358c560ddcbaade8462a80a076cd78177cf75e6e5a0eea`
- Comparison `cpu-acceptance/acceptance.json` SHA-256: `f28fb2282ed026b6804a4011e907d2fb42852cf6f88467f89330858ebb21d5da`
- Pretrained controls `controls.json` SHA-256: `b45fb4a526421ca31a704e05c96ff62d977957158d45900b86ef70ace65be58b`
- Workflow controls `controls.json` SHA-256: `5182daeefe3dccd82daef818a3901c68e01083eafdcba637c525873147231e5f`
- `treeshap_oracle/oracle_payload.json` SHA-256: `544a1448731f1d6655ec2b04a996749f1d551ea2c454e8299a83da1c30c96f04`
- [Release evidence index](README.md)

## Release validation

- **Scientific suite**: `39/39` workloads passed, `3/3` rolling-origin backtests passed, `2/2` expected-error checks verified (`all_gates_passed: true`, AC power).
- **Standardized comparison (`v3.14.0-release-gate/comparison`, AC `production-default`)**: `6/6` profiles passed (`standard`, `extended`, `public-data`, `nist`, `migration`, `tabular-migration`), `10,300` validated measurements across `412` engine-case rows, `0` failures.
- **CPU acceptance (`16` profiles)**: `14/16` profiles passed strictly (`400/404` total engine-case executions) and recorded `4` known **pandas** Wampler 4/5 comparator mismatches in `numerical-binary64` and `numerical-conformance`. SwiftSci passed `202/202` engine-case executions; `regression-policy.json` recorded `blocking: []`.
- **Auxiliary controls**: `60/60` pretrained inspection controls (`pretrained-inspection/controls.json`) and `100/100` workflow mutation controls (`workflow-controls/controls.json`) passed.
- **TreeSHAP Oracle (`shap 0.54.1`)**: Independent 5-model cross-check (`DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostedTreesRegressor`, `DecisionTreeClassifier`, `RandomForestClassifier`) against Python `shap.TreeExplainer(feature_perturbation="tree_path_dependent")` and Swift `TreeSHAP(weighting: .empiricalCover)` passed all checks with `maxShapDiff <= 1.776e-15` (all `< 1e-10`).
- **Swift, DocC & Cookbook verification**: `xcodebuild test` + `swift test -c release` recorded `1,919` total tests (`1,912` passed / `2,814` dynamic runs, `7` env-gated skips, `0` failures). `15/15` DocC archives built cleanly with `--warnings-as-errors`, `verify_doc_coverage.py` reported `100.00% (2641/2641 matched declarations documented)` (`0` placeholders), and `CookbookVerification` passed `71/71` recipes (`0` failed).

## 16-Profile CPU Acceptance Summary

Both `v3.14.0-release-gate/acceptance/acceptance.json` and `v3.14.0-release-gate/comparison/cpu-acceptance/acceptance.json` produced identical profile counts (`400/404` passed):

| Profile | Engine Cases | Passed | Failures | Status | Run SHA-256 (release-gate) |
|---|---:|---:|---:|---|---|
| `vision-conformance` | 16 | 16 | 0 | `passed` | `a3d80a5175a61af248e94ce9657e733f5c05ef05445e17372ef8727d7e86bc1c` |
| `boundary-cpu-conformance` | 8 | 8 | 0 | `passed` | `476194dcdaf7848517dbe83d5be3a795faa7309985202246efed72f02b6d4084` |
| `certification` | 54 | 54 | 0 | `passed` | `cc19a6e94b2c5952abb236ea237f709fa4176eadb58911534c6f1c71cb21ed1d` |
| `controlled-conformance` | 20 | 20 | 0 | `passed` | `32065b2ba747e15381523d54908ca09bafada4e51e4bde60f5acb9eebe3dfa20` |
| `dataframe-conformance` | 48 | 48 | 0 | `passed` | `1b1b4ea2c5652b44c73bb865633b9c376905fb686b907292517ae7f5eac7868c` |
| `migration-smoke` | 72 | 72 | 0 | `passed` | `a80cec7fd0d324a5a6eac0d50851d0baa252ccee7636fb27e78fb2fad548f90e` |
| `model-conformance` | 6 | 6 | 0 | `passed` | `427724a5017a6a236167f5fb5ccd37c4c76bd0bb219d75d1e301b22447eeb33f` |
| `neural-cpu-conformance` | 10 | 10 | 0 | `passed` | `81cad4ccdfe9acc7ee8677f5d9a846f23118cc9d0a5822e88e0c3170edae07ce` |
| `numerical-binary64` | 38 | 36 | 2 (pandas) | `failed` | `f47559bc9a71d2ad1b0f1ab805c95e7dcae226c78e27a835e4eeddc3df7ae3b9` |
| `numerical-conformance` | 38 | 36 | 2 (pandas) | `failed` | `c24f50bf849822062cf147fa43e488bf24fae9a6c3a495c88dc9ee6bb13e115e` |
| `public-smoke` | 22 | 22 | 0 | `passed` | `a9a59c3c20d68bd2f7128109ffb3d08056947f3d6b9c90487c6c12b7deb9fffd` |
| `smoke` | 28 | 28 | 0 | `passed` | `ee1f7b663014b3a311fd2c24cc11631b38b46e4625bc8e1b71f2d552ea3a61cd` |
| `supervised-conformance` | 12 | 12 | 0 | `passed` | `dcb8a2422b3832dce1d98f2cdbeb02224be24db40c9f1a8946a426fd2fefc6ae` |
| `scientific-workflow` | 8 | 8 | 0 | `passed` | `15c302c086ca3d2f774e9ed5eeb5d00bed97353f23ee72d8cb3f2d11437c05b5` |
| `persisted-workflow` | 12 | 12 | 0 | `passed` | `9749825a6f13530c231e9dbf5f8a4833d27c6b74c25f7375f88a9c958b3b5dce` |
| `neural-shaped-cpu-conformance` | 12 | 12 | 0 | `passed` | `9e1c507d133741be14a125fba0a07f5a5063772afc443e69969898dddf60540b` |

## Reference methods and tolerances

Each standardized contract declares shape, output ordering, units and tolerances. For finite numeric values the harness applies `abs(actual - expected) <= atol + rtol * abs(expected)`; NaN and infinity are checked separately where declared by the reference. Tabular workflows also check null semantics and round trips, including independent Parquet readback. Scientific checks use NumPy/SciPy, matched Python reference methods, scikit-learn and statsmodels where applicable, independently enumerated `2^K` coalitions, and Python `shap.TreeExplainer` (`shap 0.54.1`) path-dependent TreeSHAP fixtures.

### NIST univariate cases (`comparison/measurements.json`)

All `27` mean, sample variance and sample standard deviation cases passed against NIST Statistical Reference Dataset values.

| NIST dataset | Rows | Gates passed | Mean abs error | Variance abs error | StdDev abs error | Max absolute SwiftSci error |
|---|---:|---:|---:|---:|---:|---:|
| `nist-lew` | 200 | 3/3 | `0` | `7.27596e-11` | `1.13687e-13` | `7.27596e-11` |
| `nist-lottery` | 218 | 3/3 | `3.41061e-13` | `4.36557e-11` | `5.68434e-14` | `4.36557e-11` |
| `nist-mavro` | 50 | 3/3 | `0` | `2.78197e-20` | `3.24176e-17` | `3.24176e-17` |
| `nist-michelso` | 100 | 3/3 | `0` | `1.79544e-16` | `1.13798e-15` | `1.13798e-15` |
| `nist-numacc1` | 3 | 3/3 | `0` | `0` | `0` | `0` |
| `nist-numacc2` | 1,001 | 3/3 | `0` | `1.73472e-18` | `1.38778e-17` | `1.38778e-17` |
| `nist-numacc3` | 1,001 | 3/3 | `3.49246e-10` | `6.98492e-12` | `3.49246e-11` | `3.49246e-10` |
| `nist-numacc4` | 1,001 | 3/3 | `5.58794e-09` | `1.11759e-10` | `5.58794e-10` | `5.58794e-09` |
| `nist-pidigits` | 5,000 | 3/3 | `0` | `1.06581e-14` | `1.77636e-15` | `1.06581e-14` |

## 3.14.0 TreeSHAP Oracle (`shap.TreeExplainer`, `shap 0.54.1`)

Cross-checked `TreeSHAP(permutationSeed: 42, weighting: .empiricalCover)` against Python `shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")` on `12` held-out 5-feature test instances (`90` training rows, `seed = 20260314`):

| Model | Trees / Config | `maxShapDiff` | `maxBaseDiff` | `maxPredDiff` | `maxEffDiff` | Gate (`< 1e-10`) |
|---|---|---:|---:|---:|---:|---|
| `DecisionTreeRegressor` | `max_depth=4, min_samples_split=4` | `1.332e-15` | `2.220e-16` | `0.000e+00` | `4.441e-16` | `PASS` |
| `RandomForestRegressor` | `n_estimators=5, max_depth=4` | `1.776e-15` | `2.429e-17` | `0.000e+00` | `1.776e-15` | `PASS` |
| `GradientBoostedTreesRegressor` (G-035) | `n_estimators=6, lr=0.2, max_depth=3` | `1.332e-15` | `5.551e-17` | `8.882e-16` | `8.882e-16` | `PASS` |
| `DecisionTreeClassifier` (G-036, per-class) | `3` classes, `max_depth=4` | `1.665e-16` | `5.551e-17` | `0.000e+00` | `2.220e-16` | `PASS` |
| `RandomForestClassifier` (G-036, per-class) | `5` trees, `3` classes, `max_depth=4` | `1.110e-16` | `5.551e-17` | `0.000e+00` | `2.220e-16` | `PASS` |

## 3.14.0 Numerical & Correctness Gates

| Check | Result | Scope |
|---|---|---|
| **G-035**: `TreeSHAP` for `GradientBoostedTreesRegressor` | Local accuracy, batch parity, `2^K` coalition & `shap.TreeExplainer` `1.332e-15` | `explain`, `explainDetailed`, `explainBatch` across `.squaredError`, `.absoluteError`, `.huber`; `.uniformLeaf` & `.empiricalCover` weighting; `FlatTreeNode.cover` preserved in `GradientBoostedTreesModelState` |
| **G-036**: Per-class probability `TreeSHAP` & `ForestProbabilityMode` | Local accuracy, class balance, `2^K` coalition & `shap.TreeExplainer` `<= 1.665e-16` | `explainClassDetailed`, `explainAllClassesDetailed`, `explainBatchClass` on `DecisionTreeClassifier` and `RandomForestClassifier` (`.softLeafDistribution` & `.hardVote`) across `C = 2, 3, 5` |
| **G-038**: `ReActAgent` chat-templated turns, `ChatTemplate` stop sequences & `GenerationConfiguration` | Passed in `ReActAgent3140Tests`, `ChatTemplate3140Tests`, `GenerationConfigStopSeq3140Tests` | `stream(query:chatTemplate:...)`, `ChatTemplate.defaultStopSequences` across all 6 families, `LLMOptions.stopSequences` / `stopTokenIDs`, `GenerationConfiguration` JSON parsing |
| **G-039**: `SwiftVision` dataset metadata, cropping & split `ImageFolder` loaders | Passed in `VisionDatasets3140Tests` | `YOLODatasetMetadata` YAML parsing, `ImageDataset.crop` / `cropAll` / `sampleBackgroundCrop`, `ImageDataset.loadFolder` / `loadSplitFolder` |
| **G-040**: `TrainedVisionClassifier`, `OneVsRestClassifier` & Core ML Vision pipeline export | Compiled Core ML `VisionFeaturePrint.Scene` + `MLModel` parity (`OneVsRest`, `MLP`, `RandomForest`) | `OneVsRestClassifier` `ClassifierEstimator` conformance (`getBias`, `getWeights`, `getWeightsAndBias`, `decisionFunction`, `predictProbability`), `CoreMLExporter.exportVisionClassifierPipeline`, end-to-end `TrainedVisionClassifier` folder training |
| OLS (`ml_ols_regression`) | Max prediction difference `5.50671e-14` | Matched seeded fixture (`r2 = 0.998018`) |
| Logistic regression (`ml_logistic_regression`) | Holdout accuracy `0.9600` (SwiftSci) vs `0.9575` (scikit-learn) | One fixed split |
| LinearSVC (`ml_linear_svc`) | Holdout accuracy `0.9575` (SwiftSci) vs `0.9575` (scikit-learn) | Exact `0.0` accuracy gap |
| PCA (`cluster_pca`) | Total explained-variance ratio difference `4.44089e-16` | Seeded projection fixture (`1500 × 4`) |
| K-Means (`cluster_kmeans`) | WCSS relative gap `5.57234e-16` | `6528.643055` vs `6528.643055` |
| Decision tree TreeSHAP (`explain_treeshap_dt_k4`) | Coalition-oracle diff `0`, additivity error `1.77636e-15` | `25` instances, `1` tree |
| Random forest TreeSHAP (`explain_treeshap_rf_k8`) | Coalition-oracle diff `4.44089e-16`, additivity error `4.44089e-16` | `25` instances, `10` trees |
| Sample moments (`stats_ddof1_100k`) | Skewness diff `2.38524e-17`, excess kurtosis diff `2.13163e-14` | `100,000` values vs SciPy translated to `ddof=1` |
| G-037 one-step fitted values | Max warmup diff `0.0`; max residual identity diff `5.68434e-14` | Verified across `ts_arima_*` and `ts_sarima_*` workloads |
| LLM KV-cache ablation | Max abs logit diff `4.76837e-07` (`p24_d16`) / `3.57628e-07` (`p64_d24`) | Synthetic Transformer forward vs KV-cache equivalence |

## Rolling-origin mean RMSE

| Backtest | SwiftSci | Python reference | Seasonal naive |
|---|---:|---:|---:|
| `hw_syn500_ro` | `1.511027` | `1.554041` | `5.347851` |
| `arima_dgp_ro` | `5.223510` | `5.071532` | `4.276752` |
| `sarima_ap_ro` | `32.542239` | `30.992804` | `38.991669` |

## Scientific workload outcomes (39/39 passed)

| Workload | Group | Gate | Key Metric / Notes |
|---|---|---|---|
| `ts_hw_syn500_opt` | `SwiftForecast` | Passed | RMSE `0.750040` (Swift) vs `0.712150` (Python) |
| `ts_hw_syn500_fixed` | `SwiftForecast` | Passed | RMSE `1.547220` (Swift) vs `1.627154` (Python) |
| `ts_hw_syn50k_fixed` | `SwiftForecast` | Passed | RMSE `1.420978` (Swift) vs `1.435845` (Python) |
| `ts_arima_syn500_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_arima_syn500_vs_statespace` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_arima_syn50k_vs_statespace` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_arima_dgp_syn50k_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_arima_dgp_syn500_matched_hr` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `1.42109e-14` |
| `ts_sarima_expect_fail_syn500` | `SwiftForecast` | Expected error verified | Rejects invalid/non-invertible polynomial |
| `ts_sarima_expect_fail_syn50k` | `SwiftForecast` | Expected error verified | Rejects invalid/non-invertible polynomial |
| `ts_sarima_airpassengers_110_010` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_sarima_airpassengers_011_011` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_sarima_airpassengers_144_g037` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `5.68434e-14` |
| `ts_auto_arima_valid` | `SwiftForecast` | Passed | G-037 warmup diff `0.0`, identity diff `0.0` |
| `ts_auto_arima_zero_variance` | `SwiftForecast` | Passed | Constant-series guard verified (`max_abs_err_from_42 = 0.0`) |
| `ts_auto_arima_poly_sign_check` | `SwiftForecast` | Passed | Schur–Cohn sign alignment verified |
| `ts_auto_arima_poly_unstable_check` | `SwiftForecast` | Passed | Unstable polynomial rejection verified |
| `anomaly_syn144_s04` | `SwiftForecast` | Passed | SHESD seasonal anomaly detection (`f1 = 1.0`) |
| `anomaly_syn500_holdout` | `SwiftForecast` | Passed | SHESD holdout anomaly detection (`f1 = 0.9091`) |
| `anomaly_syn500_trended_diagnostic` | `SwiftForecast` | Passed | SHESD trended diagnostic (`recall = 1.0`) |
| `ml_ols_regression` | `SwiftML` | Passed | Max pred diff `5.50671e-14` |
| `ml_logistic_regression` | `SwiftML` | Passed | Accuracy `0.9600` vs `0.9575` |
| `ml_linear_svc` | `SwiftML` | Passed | Accuracy `0.9575` vs `0.9575` (`gap = 0.0`) |
| `ml_decision_tree_regressor` | `SwiftML` | Passed | R² `0.600367` vs `0.601527` |
| `ml_decision_tree_classifier` | `SwiftML` | Passed | Accuracy `0.7450` vs `0.7450` (`gap = 0.0`) |
| `ml_random_forest_regressor` | `SwiftML` | Passed | R² `0.604400` vs `0.662001` |
| `ml_random_forest_classifier` | `SwiftML` | Passed | Accuracy `0.8600` vs `0.8500` |
| `ml_gbdt_regressor` | `SwiftML` | Passed | R² `0.728881` vs `0.731315` |
| `ml_hist_gbdt_regressor` | `SwiftML` | Passed | R² `0.786131` vs `0.777853` |
| `ml_hist_gbdt_classifier` | `SwiftML` | Passed | Accuracy `0.8625` vs `0.8700` |
| `cluster_pca` | `SwiftCluster` | Passed | EVR abs diff `4.44089e-16` |
| `cluster_kmeans` | `SwiftCluster` | Passed | WCSS rel gap `5.57234e-16` |
| `explain_treeshap_dt_k4` | `SwiftExplain` | Passed | Coalition-oracle diff `0.0`, additivity `1.77636e-15` |
| `explain_treeshap_rf_k8` | `SwiftExplain` | Passed | Coalition-oracle diff `4.44089e-16`, additivity `4.44089e-16` |
| `explain_treeshap_dt_k16_mc` | `SwiftExplain` | Passed | MC (`2,048` permutations) additivity `7.37188e-14` |
| `stats_ddof1_small5` | `SwiftStats` | Passed | Skew diff `0.0`, kurt diff `4.44089e-16` |
| `stats_ddof1_100k` | `SwiftStats` | Passed | Skew diff `2.38524e-17`, kurt diff `2.13163e-14` |
| `llm_kvcache_ablation_p24_d16` | `SwiftLLM` | Passed | Max abs logit diff `4.76837e-07` |
| `llm_kvcache_ablation_p64_d24` | `SwiftLLM` | Passed | Max abs logit diff `3.57628e-07` |

Note: `scientific_validation.json` carries forward the `39`-workload standardized gate baseline (whose legacy `deferred_limitations` field reflects the prior `v3.13.2` runner metadata). Both **G-035** (`GradientBoostedTreesRegressor` `TreeSHAP`) and **G-036** (per-class probability `TreeSHAP`) are implemented and verified in `3.14.0` via `TreeSHAP3140Tests` and the `shap 0.54.1` `TreeSHAPOracle` cross-check above.

## Comparator failures and environment skips

| CPU profile | pandas case | Reported mismatch |
|---|---|---|
| `numerical-binary64` | `nist-wampler4-binary64` | `Output mismatch at 0: 1.0000000028854616 != 1.0` |
| `numerical-binary64` | `nist-wampler5-binary64` | `Output mismatch at 0: 1.0000003429656807 != 1.0` |
| `numerical-conformance` | `nist-wampler4-decimal` | `Output mismatch at 0: 1.0000000028854616 != 1.0` |
| `numerical-conformance` | `nist-wampler5-decimal` | `Output mismatch at 0: 1.0000003429656807 != 1.0` |

- **Environment-gated Swift tests**: `7` environment-gated Swift tests (external HuggingFace checkpoint / live network / optional fixture checks) were skipped across `xcodebuild test` and `swift test -c release` (`1,919` total; `1,912` passed / `2,814` dynamic runs; `0` failures).

See the [evidence index](README.md) for gate thresholds and cryptographic hashes, and the [deep performance audit](performance.deep.md) for full 6-profile comparison tables.
