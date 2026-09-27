# Frozen supervised datasets

This pack separates data preparation, numerical conformance and predictive quality. Its three current cases check train-only scaling on WDBC and red wine, then train-only OLS regression on red wine. WDBC classifier training is not yet part of this profile.

## Sources and attribution

- Red wine quality has 1,599 records and eleven physicochemical predictors. The original quality score is the regression target. Source bytes already reside in `../uci/winequality-red.csv`; this pack does not duplicate them. Cite Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J., 2009, [Wine Quality, UCI](https://doi.org/10.24432/C56S3T). UCI distributes the data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This pack converts decimal text to binary64, assigns zero-based source-row IDs and adds splits and independent references. It does not filter on the quality target.
- Wisconsin diagnostic breast cancer has 569 records and thirty predictors. Original IDs and diagnoses are retained in `originals/wdbc.data`. Derived labels use malignant = 1 and benign = 0. IDs and diagnoses never enter the predictor matrix. See [creator credit and license](originals/ATTRIBUTION.md) and the [column schema](originals/wdbc-schema-v1.json). Physical feature units are unspecified by the source; values remain in the published units.

`sources.lock.json` pins raw byte counts and SHA-256 digests. UCI does not supply release identifiers for these files, so content hashes define the versions. Input manifests pin the lock, derived payload and reference. Original records remain unchanged; derived JSON, splits and reference calculations are additions and do not imply creator endorsement.

## Split and preprocessing rules

Each feature row becomes a JSON array of Python binary64 `float.hex()` strings, with comma and colon separators and no spaces. Prefix its ASCII bytes with `SwiftSci-supervised-split-v1` and a NUL byte, hash with SHA-256, interpret the full digest as a nonnegative integer, then take modulo ten. Buckets 0 through 5 are training, 6 through 7 validation and 8 through 9 test. Rows retain source order inside each split. IDs and targets do not affect assignment. This is a grouped content split, not an exact-count or stratified split.

Identical predictor rows stay in the same partition even when targets differ. The workers reject overlaps, omissions, invalid indices and identical features crossing a partition boundary. The frozen artifacts contain both row indices and original row IDs.

| Dataset | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Red wine | 959 | 298 | 342 |
| WDBC | 328 | 115 | 126 |

Population standardization fits only training features. A standard deviation below `1e-12` becomes one, matching the declared SwiftSci API rule. Validation and test rows use those fitted parameters. Targets never affect scaling. The suite checks every mean, effective scale and transformed value. Every repeat starts with a fresh scaler and model.

## Model and metric contract

The wine model fits an intercept and all eleven standardized predictors using explicit CPU `LinearRegression`. The Python adapter uses SciPy QR with `gelsy`; SwiftSci attempts analytical QR and can fall back to gradient descent. The public API exposes the device but not the selected solver. This contract compares outputs and does not claim identical algorithms or matched solver performance.

The independent generator uses mpmath 1.4.1 at 80 decimal digits on the exact binary64 raw values. It calculates scaling, QR coefficients, every train/validation/test prediction, and held-out RMSE, MAE and R-squared. It also checks the same metrics for a constant predictor fitted to the training-target mean. Neither runtime worker imports this reference calculation. All outputs use absolute tolerance `1e-8` and relative tolerance `1e-9`, fixed before running SwiftSci.

The current model has no hyperparameter search. Validation and test are reported separately; neither affects fitting. The numerical acceptance gate checks implementation agreement with the independent calculation. It does not impose a predictive-quality threshold or establish usefulness for deployment. A future quality gate must freeze its threshold using training/validation evidence before evaluating a new held-out test split. Repeated development against this published split does not make it a fresh final evaluation set.

## Reproduce

```sh
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/build_supervised_fixtures.py
Benchmarks/.venv-standardized/bin/python Benchmarks/Tools/bench.py prepare --profile supervised-conformance
```

Run `supervised-conformance` with the worker arguments in the [benchmark guide](../../README.md). The profile uses one checked warmup and two checked samples per case. Input decoding stays outside timing. Split gathering, construction, scaling, fitting, prediction, metrics and complete output materialization are timed together. These small conformance timings are diagnostic and are not the formal performance baseline.

Regeneration tests compare every derived byte. Leakage tests perturb held-out features and targets and require unchanged fitted parameters and training outputs. Full-output checks prevent a matching aggregate metric from concealing wrong predictions. Large trained models, classification quality and shared optimization trajectories require additional contracts.
