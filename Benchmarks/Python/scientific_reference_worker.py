"""Isolated Python reference worker for SwiftSci v3.13.2 scientific validation."""
import argparse
import json
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import shap
import sklearn
import statsmodels
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def _environment_versions():
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "shap": shap.__version__,
    }


def _timed_call(warmups, samples, fn):
    timings = []
    last = None
    for index in range(warmups + samples):
        start = time.perf_counter_ns()
        last = fn()
        elapsed = time.perf_counter_ns() - start
        if index >= warmups:
            timings.append(int(elapsed))
    return timings, last


def _run_holt_winters(req):
    series = np.asarray(req["series"], dtype=np.float64)
    horizon = int(req["horizon"])
    period = int(req.get("period") or 12)
    warmups = int(req["warmups"])
    samples = int(req["samples"])

    alpha = req.get("alpha")
    beta = req.get("beta")
    gamma = req.get("gamma")

    def fit_once():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=period,
            )
            if alpha is not None:
                return model.fit(
                    smoothing_level=float(alpha),
                    smoothing_trend=float(beta if beta is not None else 0.1),
                    smoothing_seasonal=float(gamma if gamma is not None else 0.1),
                    optimized=False,
                )
            return model.fit(optimized=True)

    fit_ns, fitted_res = _timed_call(warmups, samples, fit_once)
    infer_ns, fc = _timed_call(
        warmups, samples, lambda: np.asarray(fitted_res.forecast(steps=horizon), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": fc.tolist(),
        "fitted_values": np.asarray(fitted_res.fittedvalues, dtype=np.float64).tolist(),
    }


def _run_arima(req):
    series = np.asarray(req["series"], dtype=np.float64)
    horizon = int(req["horizon"])
    order = (int(req.get("p", 1)), int(req.get("d", 1)), int(req.get("q", 1)))
    method = str(req.get("arima_method") or "statespace")
    warmups = int(req["warmups"])
    samples = int(req["samples"])

    def fit_once():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "hannan_rissanen":
                return ARIMA(series, order=order).fit(method="hannan_rissanen")
            return ARIMA(series, order=order).fit()

    fit_ns, fitted_res = _timed_call(warmups, samples, fit_once)
    infer_ns, fc = _timed_call(
        warmups, samples, lambda: np.asarray(fitted_res.forecast(steps=horizon), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": fc.tolist(),
        "fitted_values": np.asarray(fitted_res.fittedvalues, dtype=np.float64).tolist(),
        "metrics": {"aic": float(fitted_res.aic)} if np.isfinite(fitted_res.aic) else None,
    }


def _run_sarima(req):
    series = np.asarray(req["series"], dtype=np.float64)
    horizon = int(req["horizon"])
    order = (int(req.get("p", 1)), int(req.get("d", 1)), int(req.get("q", 1)))
    seasonal = (
        int(req.get("P", 0)),
        int(req.get("D", 0)),
        int(req.get("Q", 0)),
        int(req.get("s", 12)),
    )
    warmups = int(req["warmups"])
    samples = int(req["samples"])

    def fit_once():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(series, order=order, seasonal_order=seasonal).fit(disp=False)

    fit_ns, fitted_res = _timed_call(warmups, samples, fit_once)
    infer_ns, fc = _timed_call(
        warmups, samples, lambda: np.asarray(fitted_res.forecast(steps=horizon), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": fc.tolist(),
        "fitted_values": np.asarray(fitted_res.fittedvalues, dtype=np.float64).tolist(),
        "metrics": {"aic": float(fitted_res.aic)} if np.isfinite(fitted_res.aic) else None,
    }


def _run_seasonal_naive(req):
    series = req["series"]
    horizon = int(req["horizon"])
    period = int(req.get("period") or 12)
    warmups = int(req["warmups"])
    samples = int(req["samples"])
    n = len(series)

    def predict():
        return [float(series[n - period + (i % period)]) for i in range(horizon)]

    timings, fc = _timed_call(warmups, samples, predict)
    return {
        "fit_samples_ns": timings,
        "infer_samples_ns": timings,
        "predictions": fc,
    }


def _supervised_arrays(req):
    x_train = np.asarray(req["x_train"], dtype=np.float64)
    y_train = np.asarray(req["y_train"], dtype=np.float64)
    x_test = np.asarray(req["x_test"], dtype=np.float64)
    return x_train, y_train, x_test


def _run_supervised_regressor(req, make_model):
    x_train, y_train, x_test = _supervised_arrays(req)
    warmups = int(req["warmups"])
    samples = int(req["samples"])
    fit_ns, model = _timed_call(warmups, samples, lambda: make_model().fit(x_train, y_train))
    infer_ns, preds = _timed_call(
        warmups, samples, lambda: np.asarray(model.predict(x_test), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": preds.tolist(),
    }


def _run_supervised_classifier(req, make_model):
    x_train, y_train, x_test = _supervised_arrays(req)
    y_int = y_train.astype(np.int64)
    warmups = int(req["warmups"])
    samples = int(req["samples"])
    fit_ns, model = _timed_call(warmups, samples, lambda: make_model().fit(x_train, y_int))

    def infer_once():
        preds = np.asarray(model.predict(x_test), dtype=np.float64)
        probs = (
            np.asarray(model.predict_proba(x_test), dtype=np.float64).tolist()
            if hasattr(model, "predict_proba")
            else None
        )
        scores = (
            np.asarray(model.decision_function(x_test), dtype=np.float64).tolist()
            if hasattr(model, "decision_function")
            else None
        )
        return preds, probs, scores

    infer_ns, out = _timed_call(warmups, samples, infer_once)
    preds, probs, scores = out
    result = {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": preds.tolist(),
    }
    if probs is not None:
        result["probabilities"] = probs
    if scores is not None:
        result["anomaly_scores"] = scores
    return result


def _run_pca(req):
    x_train = np.asarray(req["x_train"], dtype=np.float64)
    x_test = np.asarray(req.get("x_test") or req["x_train"], dtype=np.float64)
    k = int(req.get("n_components") or 10)
    warmups = int(req["warmups"])
    samples = int(req["samples"])
    fit_ns, model = _timed_call(
        warmups, samples, lambda: PCA(n_components=k, svd_solver="full").fit(x_train)
    )
    infer_ns, proj = _timed_call(
        warmups, samples, lambda: np.asarray(model.transform(x_test), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "matrix_output": proj.tolist(),
        "metrics": {
            "explained_variance_ratio_sum": float(np.sum(model.explained_variance_ratio_)),
        },
    }


def _run_kmeans(req):
    x_train = np.asarray(req["x_train"], dtype=np.float64)
    x_test = np.asarray(req.get("x_test") or req["x_train"], dtype=np.float64)
    k = int(req.get("n_clusters") or 3)
    seed = int(req.get("seed") or 42)
    warmups = int(req["warmups"])
    samples = int(req["samples"])
    fit_ns, model = _timed_call(
        warmups,
        samples,
        lambda: KMeans(
            n_clusters=k, max_iter=50, tol=1e-4, n_init=1, init="k-means++", random_state=seed
        ).fit(x_train),
    )
    infer_ns, labels = _timed_call(
        warmups, samples, lambda: np.asarray(model.predict(x_test), dtype=np.float64)
    )
    return {
        "fit_samples_ns": fit_ns,
        "infer_samples_ns": infer_ns,
        "predictions": labels.tolist(),
        "matrix_output": model.cluster_centers_.tolist(),
        "metrics": {"wcss": float(model.inertia_)},
    }


def _run_treeshap(req, is_forest):
    x_train, y_train, x_test = _supervised_arrays(req)
    instances = np.asarray(req.get("instances") or x_test[:25], dtype=np.float64)
    seed = int(req.get("seed") or 42)
    max_depth = int(req.get("max_depth") or 4)
    if is_forest:
        n_est = int(req.get("n_estimators") or 20)
        model = RandomForestRegressor(
            n_estimators=n_est, max_depth=max_depth, random_state=seed, n_jobs=1
        ).fit(x_train, y_train)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=seed).fit(x_train, y_train)
    warmups = int(req["warmups"])
    samples = int(req["samples"])

    def explain_once():
        explainer = shap.TreeExplainer(model)
        sv = np.asarray(explainer.shap_values(instances), dtype=np.float64)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        return sv, base

    explain_ns, (sv, base) = _timed_call(warmups, samples, explain_once)
    preds = np.asarray(model.predict(instances), dtype=np.float64)
    return {
        "fit_samples_ns": explain_ns,
        "infer_samples_ns": explain_ns,
        "matrix_output": sv.tolist(),
        "shap_base_values": [base] * len(instances),
        "shap_predictions": preds.tolist(),
    }


def _run_stats_skew_kurt(req):
    vals = np.asarray(req["series"], dtype=np.float64)
    warmups = int(req["warmups"])
    samples = int(req["samples"])

    def eval_once():
        return (
            float(scipy.stats.skew(vals, bias=True)),
            float(scipy.stats.skew(vals, bias=False)),
            float(scipy.stats.kurtosis(vals, bias=True)),
            float(scipy.stats.kurtosis(vals, bias=False)),
        )

    timings, (sk_t, sk_f, ku_t, ku_f) = _timed_call(warmups, samples, eval_once)
    return {
        "fit_samples_ns": timings,
        "infer_samples_ns": timings,
        "metrics": {
            "skew_bias_true": sk_t,
            "skew_bias_false": sk_f,
            "kurt_bias_true": ku_t,
            "kurt_bias_false": ku_f,
        },
    }


def dispatch(req):
    op = req["operation"]
    seed = int(req.get("seed") or 42)
    if op == "holt_winters":
        return _run_holt_winters(req)
    if op == "arima_fit_forecast":
        return _run_arima(req)
    if op == "sarima_fit_forecast":
        return _run_sarima(req)
    if op == "seasonal_naive":
        return _run_seasonal_naive(req)
    if op == "ols_regression":
        return _run_supervised_regressor(req, lambda: LinearRegression())
    if op == "logistic_regression":
        return _run_supervised_classifier(
            req, lambda: LogisticRegression(max_iter=1000, random_state=seed)
        )
    if op == "linear_svc":
        c_param = float(req.get("c_param") or 1.0)
        return _run_supervised_classifier(
            req, lambda: LinearSVC(C=c_param, max_iter=2000, random_state=seed)
        )
    if op == "decision_tree_regressor":
        md = int(req.get("max_depth") or 5)
        return _run_supervised_regressor(
            req, lambda: DecisionTreeRegressor(max_depth=md, random_state=seed)
        )
    if op == "decision_tree_classifier":
        md = int(req.get("max_depth") or 5)
        return _run_supervised_classifier(
            req, lambda: DecisionTreeClassifier(max_depth=md, criterion="gini", random_state=seed)
        )
    if op == "random_forest_regressor":
        n_est = int(req.get("n_estimators") or 50)
        md = int(req.get("max_depth") or 6)
        return _run_supervised_regressor(
            req,
            lambda: RandomForestRegressor(
                n_estimators=n_est, max_depth=md, random_state=seed, n_jobs=1
            ),
        )
    if op == "random_forest_classifier":
        n_est = int(req.get("n_estimators") or 50)
        md = int(req.get("max_depth") or 6)
        return _run_supervised_classifier(
            req,
            lambda: RandomForestClassifier(
                n_estimators=n_est, max_depth=md, criterion="gini", random_state=seed, n_jobs=1
            ),
        )
    if op == "gbdt_regressor":
        n_est = int(req.get("n_estimators") or 50)
        md = int(req.get("max_depth") or 3)
        lr = float(req.get("learning_rate") or 0.1)
        return _run_supervised_regressor(
            req,
            lambda: GradientBoostingRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=seed
            ),
        )
    if op == "hist_gbdt_regressor":
        n_est = int(req.get("n_estimators") or 50)
        md = int(req.get("max_depth") or 5)
        mb = min(255, int(req.get("max_bins") or 255))
        lr = float(req.get("learning_rate") or 0.1)
        return _run_supervised_regressor(
            req,
            lambda: HistGradientBoostingRegressor(
                max_iter=n_est, learning_rate=lr, max_depth=md, max_bins=mb, random_state=seed
            ),
        )
    if op == "hist_gbdt_classifier":
        n_est = int(req.get("n_estimators") or 50)
        md = int(req.get("max_depth") or 5)
        mb = min(255, int(req.get("max_bins") or 255))
        lr = float(req.get("learning_rate") or 0.1)
        return _run_supervised_classifier(
            req,
            lambda: HistGradientBoostingClassifier(
                max_iter=n_est, learning_rate=lr, max_depth=md, max_bins=mb, random_state=seed
            ),
        )
    if op == "pca_fit_transform":
        return _run_pca(req)
    if op == "kmeans_fit_predict":
        return _run_kmeans(req)
    if op == "treeshap_decision_tree_regressor":
        return _run_treeshap(req, is_forest=False)
    if op == "treeshap_random_forest_regressor":
        return _run_treeshap(req, is_forest=True)
    if op == "stats_skewness_kurtosis":
        return _run_stats_skew_kurt(req)
    raise ValueError(f"Unsupported Python reference operation: {op}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SwiftSci Python scientific reference worker")
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args(argv)
    req = json.loads(args.request.read_text(encoding="utf-8"))
    payload = {
        "workload_id": req["workload_id"],
        "operation": req["operation"],
        "versions": _environment_versions(),
        "error_occurred": False,
    }
    try:
        payload.update(dispatch(req))
    except Exception as exc:  # noqa: BLE001
        payload["error_occurred"] = True
        payload["error_type"] = type(exc).__name__
        payload["error_message"] = str(exc)
    payload["peak_rss_bytes"] = _rss_bytes()
    args.response.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
