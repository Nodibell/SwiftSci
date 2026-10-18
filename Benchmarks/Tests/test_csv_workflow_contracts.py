"""Known-schema ingestion is required for each timed CSV workflow."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
from contracts import ContractError, load_profile, load_workload
from csv_contract import MATCHED_OPERATIONS, schema_for
from datasets import load_manifest, resolve_workload
from runner import plan


class CSVWorkflowContractTests(unittest.TestCase):
    def test_schema_resolution_covers_each_registered_workflow(self):
        for op in MATCHED_OPERATIONS:
            datasets = ("wine-quality-red",) if op == "wine-pipeline" else ("tiny", "mixed-tiny")
            for dataset in datasets:
                with self.subTest(operation=op, dataset=dataset):
                    manifest = load_manifest(ROOT, dataset)
                    workload = resolve_workload(manifest, load_workload(ROOT, op + "-v2"))
                    self.assertEqual(workload["csv_schema"], schema_for(manifest["kind"]))
                    historical = resolve_workload(manifest, load_workload(ROOT, op + "-v1"))
                    self.assertNotIn("csv_schema", historical)

    def test_planner_rejects_unmatched_historical_contracts(self):
        for op in MATCHED_OPERATIONS:
            dataset = "wine-quality-red" if op == "wine-pipeline" else "tiny"
            profile = dict(cases=[dict(id="old", dataset=dataset, workload=op + "-v1")])
            with (self.subTest(operation=op),
                  patch("runner.command", return_value="versions"),
                  patch("runner.subprocess.check_output", return_value="{}")):
                with self.assertRaisesRegex(ContractError, "unmatched inference"):
                    plan(ROOT, profile, ["pandas"], None, sys.executable)

    def test_active_profiles_use_matched_workflow_contracts(self):
        for path in (ROOT / "Benchmarks/Specs/profiles").glob("*.json"):
            for case in load_profile(ROOT, path.stem)["cases"]:
                workload = load_workload(ROOT, case["workload"])
                if workload["operation"] in MATCHED_OPERATIONS:
                    self.assertEqual(workload["id"], workload["operation"] + "-v2", path)


if __name__ == "__main__":
    unittest.main()
