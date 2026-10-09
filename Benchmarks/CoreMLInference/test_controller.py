"""Contract checks for the bounded inference comparison controller."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import run


class ControllerTests(unittest.TestCase):
    def record(self):
        sample = {"validated": True, "mismatchedValues": 0, "elapsed_ns": 2_000_000,
                  "maximum_absolute_error": 0.0}
        return dict(workload="linear128", rows=32, maximumBatchSize=2, mode="coreml-cpu",
                    matrixInput=False, executionTraceVerified=False,
                    firstPrediction=sample, warmPredictions=[sample], residentBytesPerSample=[123],
                    reservedBytesAfterPrediction=0, parameterSHA256="weights", inputSHA256="inputs",
                    modelSHA256="vector-model", warmupFailures=0,
                    residentBytesAfterWarmup=2**20, residentBytesAfterSamples=2**20,
                    plannedDevices=["dense: cpu"])

    def test_sweep_is_bounded_and_has_no_duplicates(self):
        cases = run.plan_cases(True)
        self.assertEqual(len(cases), len(set(cases)))
        self.assertLess(len(cases), 200)
        self.assertEqual({c[0] for c in cases}, set(run.WORKLOADS))
        for workload, rows, batch, mode in cases:
            self.assertLessEqual(batch, rows)
            self.assertLessEqual(rows, 256)
            if mode.startswith("native-"):
                self.assertEqual(workload, "linear128")
            if mode.endswith(("-matrix", "-matrix-adapter")):
                self.assertEqual(batch, rows)
        for workload in run.WORKLOADS:
            self.assertIn((workload, 256, 256, "coreml-neural-matrix-adapter"), cases)
            self.assertIn((workload, 256, 32, "coreml-neural"), cases)
            self.assertIn((workload, 256, 256, "coreml-neural-matrix"), cases)

    def test_larger_batch_sweep_preserves_matching_paths(self):
        cases = run.plan_cases(True, (1024,))
        self.assertEqual(len(cases), 56)
        self.assertTrue(all(case[1] == 1024 for case in cases))
        self.assertIn(("mlp512", 1024, 1024, "coreml-neural-matrix"), cases)
        self.assertIn(("mlp512", 1024, 32, "coreml-neural"), cases)

    def test_original_six_modes_remain_available(self):
        self.assertEqual(len(run.plan_cases(False)), 6)
        self.assertTrue(all(c[:3] == ("linear128", 256, 1) for c in run.plan_cases(False)))

    def test_rejects_inconsistent_or_incomplete_records(self):
        case = ("linear128", 32, 2, "coreml-cpu")
        good = self.record()
        run.validate_record(good, case, 1)
        mutations = [("reservedBytesAfterPrediction", 1), ("mode", "coreml-neural"),
                     ("matrixInput", True), ("executionTraceVerified", True),
                     ("residentBytesPerSample", []), ("warmPredictions", []),
                     ("firstPrediction", {"validated": False})]
        for key, value in mutations:
            record = copy.deepcopy(good)
            record[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                run.validate_record(record, case, 1)

    def outside_reference(self):
        record = self.record()
        record["firstPrediction"] = dict(record["firstPrediction"], validated=False,
                                         mismatchedValues=3, maximum_absolute_error=0.0003)
        record["warmupFailures"] = 1
        return record

    def test_accuracy_exceedance_is_a_completed_measurement(self):
        record = self.outside_reference()
        run.validate_record(record, ("linear128", 32, 2, "coreml-cpu"), 1)
        self.assertFalse(run.reference_accuracy_met(record))
        self.assertEqual(run.completion_exit_code([record], False), 0)
        self.assertEqual(run.completion_exit_code([record], True), 2)
        self.assertEqual(run.completion_exit_code([self.record()], True), 0)

    def test_warmup_accuracy_exceedance_remains_visible(self):
        record = self.record()
        record["warmupFailures"] = 1
        self.assertFalse(run.reference_accuracy_met(record))
        self.assertIn("Outside tolerance", run.report([record]))
        self.assertEqual(run.completion_exit_code([record], True), 2)

    def test_report_pairs_all_timings_with_accuracy_without_claiming_quality(self):
        text = run.report([self.record(), self.outside_reference()])
        self.assertIn("Within tolerance", text)
        self.assertIn("Outside tolerance | 2.000 | 2.000 | 16000", text)
        self.assertNotIn("FAIL", text)
        self.assertNotIn("n/a", text)
        self.assertIn("Application suitability is not assessed", text)
        self.assertIn("1e-5", text)

    def test_assessment_separates_execution_accuracy_and_application_suitability(self):
        assessment = run.assessment([self.outside_reference()])
        self.assertEqual(assessment["reference_contract"]["absolute_tolerance"], 1e-5)
        self.assertEqual(assessment["reference_contract"]["relative_tolerance"], 1e-5)
        cell = assessment["cases"][0]
        self.assertEqual(cell["execution"], "completed")
        self.assertEqual(cell["reference_accuracy"], "outside_tolerance")
        self.assertEqual(cell["application_suitability"], "not_assessed")

    def test_rejects_inconsistent_or_nonfinite_measurements(self):
        for changes in ({"validated": True, "mismatchedValues": 2},
                        {"validated": False, "mismatchedValues": 0},
                        {"maximum_absolute_error": float("nan")},
                        {"elapsed_ns": float("inf")}, {"elapsed_ns": -1}):
            record = self.record()
            record["firstPrediction"] = dict(record["firstPrediction"], **changes)
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                run.validate_record(record, ("linear128", 32, 2, "coreml-cpu"), 1)

    def test_public_matrix_path_requires_adapter_evidence(self):
        record = dict(self.record(), matrixInput=True, maximumBatchSize=32,
                      mode="coreml-cpu-matrix-adapter", publicAdapter=True)
        case = ("linear128", 32, 32, "coreml-cpu-matrix-adapter")
        run.validate_record(record, case, 1)
        with self.assertRaises(RuntimeError):
            run.validate_record(dict(record, publicAdapter=False), case, 1)

    def test_multiclass_counts_every_output_value(self):
        record = dict(self.outside_reference(), outputColumns=7)
        record["firstPrediction"]["mismatchedValues"] = 200
        run.validate_record(record, ("linear128", 32, 2, "coreml-cpu"), 1)
        record["firstPrediction"]["mismatchedValues"] = 225
        with self.assertRaises(RuntimeError):
            run.validate_record(record, ("linear128", 32, 2, "coreml-cpu"), 1)
        for count in [0, -1, 1025, 1.5, True]:
            with self.assertRaises(RuntimeError):
                run.validate_record(dict(self.record(), outputColumns=count), ("linear128", 32, 2, "coreml-cpu"), 1)

    def test_worker_runtime_error_still_stops_collection(self):
        with TemporaryDirectory() as directory, patch("run.subprocess.Popen") as process, patch("run.os.killpg"):
            process.return_value.pid = 123
            process.return_value.wait.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "failed with exit 1"):
                run.run_case(Path("unused-worker"), Path(directory),
                             ("linear128", 32, 2, "coreml-cpu"), 1, {})

    def test_comparability_checks_parameters_and_distinguishes_model_shape(self):
        first = self.record()
        matrix = dict(first, matrixInput=True, modelSHA256="matrix-model")
        run.validate_comparability([first, matrix])
        for key in ("parameterSHA256", "inputSHA256"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                run.validate_comparability([first, dict(matrix, **{key: "changed"})])
        with self.assertRaises(RuntimeError):
            run.validate_comparability([first, dict(first, modelSHA256="changed")])


if __name__ == "__main__":
    unittest.main()
