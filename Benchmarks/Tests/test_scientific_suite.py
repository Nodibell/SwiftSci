"""Unit tests for the scientific validation suite and gate logic."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
import scientific_validation_suite as svs  # noqa: E402

PY311_ENGINE = Path(
    os.environ.get(
        "SWIFTSCI_SCIENTIFIC_PYTHON",
        str(Path.home() / ".pyenv/versions/3.11.9/bin/python3"),
    )
)


class ScientificSuiteStatsAndGatingTests(unittest.TestCase):
    def test_median_of_medians_iqr_and_resolution_flag(self):
        # Asymmetric process distributions verify pure median-of-medians (not trimmed mean)
        proc_runs = [
            [200.0, 210.0, 220.0, 230.0, 9000.0],  # median = 220
            [240.0, 250.0, 260.0, 270.0, 9000.0],  # median = 260
            [300.0, 310.0, 320.0, 330.0, 9000.0],  # median = 320
        ]
        summary = svs.aggregate_process_runs(proc_runs, resolution_threshold_ns=1000.0)
        self.assertEqual(summary["median_of_medians_ns"], 260.0)
        self.assertEqual(summary["min_ns"], 220.0)
        self.assertEqual(summary["max_ns"], 320.0)
        self.assertEqual(summary["iqr_ns"], 50.0)
        self.assertTrue(summary["resolution_limited"])

        fast = svs.summarize_samples([1500.0, 2000.0, 2500.0])
        self.assertEqual(fast["median_ns"], 2000.0)
        self.assertFalse(fast["resolution_limited"])

    def test_rss_summary_tracks_process_scope(self):
        rss = svs.compute_rss_summary([10 * 1024 * 1024, 12 * 1024 * 1024, 14 * 1024 * 1024])
        self.assertEqual(rss["metric_scope"], "whole_process_peak_rss_ru_maxrss")
        self.assertAlmostEqual(rss["median_mib"], 12.0, places=9)
        self.assertAlmostEqual(rss["iqr_mib"], 2.0, places=9)

    def test_expect_error_gate_excludes_speedup_and_requires_fail_closed(self):
        spec = {"gate_kind": "expect_error", "expect_error": True}
        passed_gate = svs.evaluate_case_gate(
            spec,
            swift_resp={"error_occurred": True, "is_fitted": False, "error_type": "ForecastError"},
            py_resp={"error_occurred": False, "predictions": [1.0]},
        )
        self.assertTrue(passed_gate["passed"])
        self.assertTrue(passed_gate["expect_error_verified"])
        self.assertEqual(passed_gate["speedup_fit"]["reason"], "EXCLUDED_EXPECTED_ERROR")

        # Must fail if is_fitted remained True
        bad_gate = svs.evaluate_case_gate(
            spec,
            swift_resp={"error_occurred": True, "is_fitted": True},
        )
        self.assertFalse(bad_gate["passed"])

    def test_non_comparable_nonfinite_and_resolution_gates_block_speedups(self):
        timing_ok = svs.aggregate_process_runs([[5000.0, 5100.0, 5200.0]])
        timing_lt_1us = svs.aggregate_process_runs([[400.0, 420.0, 450.0]])

        non_comp_spec = {
            "gate_kind": "supervised_reg",
            "actual": [1.0, 2.0, 3.0],
            "min_r2": 0.9,
            "comparable_estimator": False,
        }
        non_comp_gate = svs.evaluate_case_gate(
            non_comp_spec,
            swift_resp={"predictions": [1.0, 2.0, 3.0]},
            py_resp={"predictions": [1.0, 2.0, 3.0]},
            fit_swift=timing_ok, infer_swift=timing_ok,
            fit_py=timing_ok, infer_py=timing_ok,
        )
        self.assertTrue(non_comp_gate["passed"])
        self.assertEqual(non_comp_gate["speedup_fit"]["reason"], "NON_COMPARABLE_ESTIMATOR_OR_WEIGHTS")

        nonfinite_spec = {
            "gate_kind": "supervised_reg",
            "actual": [1.0, 2.0, 3.0],
            "min_r2": 0.9,
            "comparable_estimator": True,
        }
        nonfinite_gate = svs.evaluate_case_gate(
            nonfinite_spec,
            swift_resp={"predictions": [1.0, float("nan"), 3.0]},
            py_resp={"predictions": [1.0, 2.0, 3.0]},
            fit_swift=timing_ok, infer_swift=timing_ok,
            fit_py=timing_ok, infer_py=timing_ok,
        )
        self.assertFalse(nonfinite_gate["passed"])
        self.assertEqual(nonfinite_gate["speedup_fit"]["reason"], "ACCURACY_OR_FINITENESS_GATE_FAILED")

        res_gate = svs.evaluate_case_gate(
            nonfinite_spec,
            swift_resp={"predictions": [1.0, 2.0, 3.0]},
            py_resp={"predictions": [1.0, 2.0, 3.0]},
            fit_swift=timing_lt_1us, infer_swift=timing_ok,
            fit_py=timing_ok, infer_py=timing_ok,
        )
        self.assertTrue(res_gate["passed"])
        self.assertEqual(res_gate["speedup_fit"]["reason"], "EXCLUDED_RESOLUTION_LIMITED_LT_1000NS")
        self.assertTrue(res_gate["speedup_infer"]["eligible"])

    def test_g037_alignment_checker(self):
        series = [10.0, 12.0, 15.0, 19.0, 22.0]
        fitted = [10.0, 12.0, 14.5, 18.0, 23.0]
        # warm_up = 2 -> residuals corresponds to indices t = 2, 3, 4
        residuals = [15.0 - 14.5, 19.0 - 18.0, 22.0 - 23.0]
        ok = svs.verify_g037_alignment(series, fitted, residuals)
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["warm_up"], 2)
        self.assertLess(ok["max_identity_diff"], 1e-12)

        drifted = [10.0, 12.0, 14.5, 17.5, 23.0]
        bad = svs.verify_g037_alignment(series, drifted, residuals)
        self.assertFalse(bad["passed"])

    def test_exact_2k_treeshap_coalition_verifier(self):
        # Asymmetric tree of depth 2 over features f0 (threshold 0.5) and f1 (threshold 1.0):
        # Root splits on f0 <= 0.5 -> left=leaf(10), right=node(f1 <= 1.0 -> left=leaf(20), right=leaf(40)).
        # Uniform leaf weights: w(leaf)=1 -> w(right)=2, w(root)=3.
        # E[f(X)] = (1/3)*10 + (2/3)*((20+40)/2) = 10/3 + 20 = 70/3.
        # For x = [1.0, 2.0]: pred = 40.0.
        # v({}) = 70/3; v({f0}) = (20+40)/2 = 30; v({f1}) = (1/3)*10 + (2/3)*40 = 30; v({f0,f1}) = 40.
        # Shapley values: phi_0 = 0.5*(30 - 70/3) + 0.5*(40 - 30) = 25/3;
        # phi_1 = 0.5*(30 - 70/3) + 0.5*(40 - 30) = 25/3. Sum + base = 70/3 + 50/3 = 40.
        flat_tree = [
            [0.0, 0.5, 1.0, 2.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, -1.0, 10.0, 1.0],
            [1.0, 1.0, 3.0, 4.0, 0.0, 0.0],
            [-1.0, 0.0, -1.0, -1.0, 20.0, 1.0],
            [-1.0, 0.0, -1.0, -1.0, 40.0, 1.0],
        ]
        res = svs.verify_treeshap_exact_coalitions(
            flat_trees=[flat_tree],
            instances=[[1.0, 2.0]],
            shap_matrix=[[25.0 / 3.0, 25.0 / 3.0]],
            base_values=[70.0 / 3.0],
            predictions=[40.0],
            num_features=2,
        )
        self.assertTrue(res["passed"])
        self.assertLess(res["max_coalition_phi_diff"], 1e-12)
        self.assertLess(res["max_additivity_diff"], 1e-12)

    def test_python_reference_worker_smoke(self):
        if not PY311_ENGINE.exists():
            self.skipTest("Python 3.11 scientific stack not installed at expected path")
        py_worker = ROOT / "Benchmarks/Python/scientific_reference_worker.py"
        with tempfile.TemporaryDirectory() as tmp:
            req_p = Path(tmp) / "req.json"
            res_p = Path(tmp) / "res.json"
            req_p.write_text(
                json.dumps({
                    "schema_version": 1,
                    "workload_id": "smoke_stats",
                    "operation": "stats_skewness_kurtosis",
                    "warmups": 1,
                    "samples": 2,
                    "series": [1.0, 2.0, 3.0, 4.0, 5.0],
                }),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [str(PY311_ENGINE), str(py_worker), str(req_p), str(res_p)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            out = json.loads(res_p.read_text(encoding="utf-8"))
            self.assertAlmostEqual(out["metrics"]["kurt_bias_true"], -1.3, places=9)
            self.assertGreater(out["peak_rss_bytes"], 0)

    def test_forecast_and_regression_gap_gates_suppress_speedups(self):
        timing_ok = svs.aggregate_process_runs([[5000.0, 5100.0, 5200.0]])
        fc_spec = {
            "gate_kind": "forecast",
            "train_series": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "actual": [10.0, 20.0],
            "period": 2,
            "beat_seasonal_naive_rmse": True,
            "max_rmse_gap_vs_python": 1.0,
            "comparable_estimator": True,
        }
        fc_fail = svs.evaluate_case_gate(
            fc_spec,
            swift_resp={"predictions": [12.0, 22.0]},
            py_resp={"predictions": [10.0, 20.0]},
            fit_swift=timing_ok,
            infer_swift=timing_ok,
            fit_py=timing_ok,
            infer_py=timing_ok,
        )
        self.assertFalse(fc_fail["passed"])
        self.assertEqual(fc_fail["speedup_fit"]["reason"], "ACCURACY_OR_FINITENESS_GATE_FAILED")
        self.assertEqual(fc_fail["speedup_infer"]["reason"], "ACCURACY_OR_FINITENESS_GATE_FAILED")

        reg_spec = {
            "gate_kind": "supervised_reg",
            "actual": [1.0, 2.0, 3.0, 4.0],
            "min_r2": 0.5,
            "max_r2_gap_vs_python": 0.05,
            "comparable_estimator": True,
        }
        reg_fail = svs.evaluate_case_gate(
            reg_spec,
            swift_resp={"predictions": [1.5, 2.0, 2.5, 3.5]},
            py_resp={"predictions": [1.0, 2.0, 3.0, 4.0]},
            fit_swift=timing_ok,
            infer_swift=timing_ok,
            fit_py=timing_ok,
            infer_py=timing_ok,
        )
        self.assertFalse(reg_fail["passed"])
        self.assertGreater(reg_fail["metrics"]["r2_gap_vs_python"], 0.05)
        self.assertEqual(reg_fail["speedup_fit"]["reason"], "ACCURACY_OR_FINITENESS_GATE_FAILED")

    def test_cli_metadata_defaults_and_overrides(self):
        defaults = svs.build_arg_parser().parse_args([
            "--swift-worker", "/tmp/worker",
            "--python-engine", "/tmp/python3",
            "--output", "/tmp/out.json",
        ])
        self.assertEqual(defaults.status, "LOCAL")
        self.assertEqual(defaults.label, "scientific-validation")

        custom = svs.build_arg_parser().parse_args([
            "--swift-worker", "/tmp/worker",
            "--python-engine", "/tmp/python3",
            "--output", "/tmp/out.json",
            "--status", "RELEASE",
            "--label", "v3.13.2",
        ])
        self.assertEqual(custom.status, "RELEASE")
        self.assertEqual(custom.label, "v3.13.2")


if __name__ == "__main__":
    unittest.main()
