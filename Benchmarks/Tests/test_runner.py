import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from test_contracts import ROOT
from contracts import *
from datasets import *
from runner import run
from reporting import summarize, comparison


class RunnerTests(unittest.TestCase):
    def make_plan(self):
        profile = dict(
            schema_version=1,
            id="smoke",
            warmups=0,
            samples=1,
            batches=1,
            timeout_seconds=1,
            cases=[dict(id="tiny-target", dataset="tiny", workload="target-v1")],
        )
        dataset = load_manifest(ROOT, "tiny")
        workload = load_workload(ROOT, "target-v1")
        return dict(
            schema_version=1,
            profile=profile,
            cases=[
                dict(
                    case=profile["cases"][0],
                    dataset=dataset,
                    workload=workload,
                    case_key="a" * 64,
                )
            ],
            engines={},
            contract_hash="c" * 64,
            environment={},
        )

    def test_crash_timeout_missing_and_wrong_response_cannot_certify(self):
        scripts = [
            "raise SystemExit(7)",
            "import time;time.sleep(5)",
            "pass",
            'import json,sys;json.dump({},open(sys.argv[2],"w"))',
        ]
        for script in scripts:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                p = self.make_plan()
                prepare(root, p["cases"][0]["dataset"])
                worker = root / "fake.py"
                worker.write_text(script)
                p["engines"] = {"fake": dict(command=[sys.executable, str(worker)])}
                p["source"] = {}
                with patch("builtins.print"):
                    result = run(root, p, root / "run")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(
                    read_json(root / "run/certificate.json")["status"], "failed"
                )
                self.assertIsNone(summarize(result)[0]["median_ns"])

    def test_incomplete_batches_are_not_passed(self):
        p = self.make_plan()
        p["profile"]["batches"] = 2
        result = dict(
            schema_version=1,
            status="passed",
            plan=p,
            events=[
                dict(
                    case_id="tiny-target",
                    engine="x",
                    status="passed",
                    result=dict(
                        samples=[dict(elapsed_ns=10, maximum_absolute_error=0)],
                        peak_rss_bytes=1,
                    ),
                )
            ],
        )
        self.assertEqual(summarize(result)[0]["status"], "failed")


class AuditTests(unittest.TestCase):
    def passing_run(self):
        p = RunnerTests().make_plan()
        p["engines"] = {"swiftsci": {}}
        result = dict(
            schema_version=1,
            case_key="a" * 64,
            status="passed",
            peak_rss_bytes=100,
            engine_version="test",
            samples=[
                dict(
                    elapsed_ns=2000,
                    output_sha256="b" * 64,
                    maximum_absolute_error=0,
                    validated=True,
                )
            ],
        )
        event = dict(
            case_id="tiny-target",
            case_key="a" * 64,
            engine="swiftsci",
            batch=0,
            status="passed",
            result=result,
        )
        return dict(schema_version=1, status="passed", plan=p, events=[event])

    def test_complete_run_compares_but_missing_duplicate_or_mismatched_cases_fail(self):
        from reporting import audit

        good = self.passing_run()
        self.assertTrue(audit(good))
        self.assertEqual(comparison(good, good)[0]["speedup"], 1)
        for events in [[], good["events"] * 2]:
            bad = copy.deepcopy(good)
            bad["events"] = events
            with self.assertRaises(ContractError):
                audit(bad)
        for field, value in [
            ("engine", "unknown"),
            ("batch", 1),
            ("case_key", "z" * 64),
            ("status", "failed"),
        ]:
            bad = copy.deepcopy(good)
            bad["events"][0][field] = value
            with self.assertRaises(ContractError):
                audit(bad)

    def test_summary_retains_worst_measured_absolute_error(self):
        good = self.passing_run()
        good["plan"]["profile"]["batches"] = 2
        second = copy.deepcopy(good["events"][0])
        second["batch"] = 1
        good["events"][0]["result"]["samples"][0]["maximum_absolute_error"] = 1e-9
        second["result"]["samples"][0]["maximum_absolute_error"] = 3e-9
        good["events"].append(second)
        self.assertEqual(summarize(good)[0]["maximum_absolute_error"], 3e-9)

    def test_unresolved_timing_has_no_speedup(self):
        good = self.passing_run()
        good["events"][0]["result"]["samples"][0]["elapsed_ns"] = 10
        self.assertIsNone(comparison(good, good)[0]["speedup"])

    def test_environment_changes_refuse_comparison(self):
        good = self.passing_run()
        other = copy.deepcopy(good)
        other["plan"]["environment"] = {"machine": "different"}
        with self.assertRaises(ContractError):
            comparison(good, other)


if __name__ == "__main__":
    unittest.main()
