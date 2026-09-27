"""Migration coverage and reference-role regressions without legacy imports."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
from contracts import ContractError, read_json
from inventory import INVENTORY_PATH, discover, reconcile


class LegacyInventoryTests(unittest.TestCase):
    def setUp(self):
        self.inventory = read_json(ROOT / INVENTORY_PATH)

    def core_entry(self):
        return next(
            x for x in self.inventory["entries"] if x["suite"] == "standardized-core"
        )

    def test_complete_registration_coverage(self):
        summary = reconcile(ROOT, self.inventory)
        self.assertEqual(summary["entries"], 135)
        self.assertEqual(
            summary["by_kind"],
            {"timed": 111, "accuracy-result": 22, "untimed-diagnostic": 2},
        )
        self.assertEqual(summary["by_suite"], {"standardized-core": 60, "research": 75})
        self.assertEqual(summary["replacement_workloads"], 30)

    def test_missing_and_duplicate_entries_fail(self):
        self.inventory["entries"].pop()
        with self.assertRaisesRegex(ContractError, "Unclassified"):
            reconcile(ROOT, self.inventory)
        self.inventory["entries"].append(self.inventory["entries"][0])
        with self.assertRaisesRegex(ContractError, "Duplicate inventory"):
            reconcile(ROOT, self.inventory)

    def test_dangling_legacy_name_fails(self):
        self.inventory["entries"][0]["name"] += " renamed"
        with self.assertRaisesRegex(ContractError, "Dangling legacy"):
            reconcile(ROOT, self.inventory)

    def test_missing_replacement_fails(self):
        self.core_entry()["replacements"] = []
        with self.assertRaisesRegex(ContractError, "Suite/replacement"):
            reconcile(ROOT, self.inventory)

    def test_wrong_operation_fails(self):
        self.core_entry()["replacements"][0]["operation"] = "not-this-operation"
        with self.assertRaisesRegex(ContractError, "operation mismatch"):
            reconcile(ROOT, self.inventory)

    def test_dangling_workload_fails(self):
        self.core_entry()["replacements"][0]["workload"] = "nonexistent-v1"
        with self.assertRaises(FileNotFoundError):
            reconcile(ROOT, self.inventory)

    def test_profile_without_workload_fails(self):
        entry = next(
            x
            for x in self.inventory["entries"]
            if any(r["operation"] == "parquet-read" for r in x["replacements"])
        )
        entry["replacements"][0]["profiles"] = ["nist"]
        with self.assertRaisesRegex(ContractError, "absent from profile"):
            reconcile(ROOT, self.inventory)

    def test_research_cannot_claim_core_replacement(self):
        entry = next(x for x in self.inventory["entries"] if x["suite"] == "research")
        entry["replacements"] = copy.deepcopy(self.core_entry()["replacements"])
        with self.assertRaisesRegex(ContractError, "Suite/replacement"):
            reconcile(ROOT, self.inventory)

    def test_kiraa_role_cannot_drift(self):
        for field, value in [
            ("role", "oracle"),
            ("optional", False),
            ("official", True),
            ("known_buggy", False),
            ("correctness_oracle", True),
            ("certification_engine", True),
        ]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.inventory)
                changed["experimental_references"][0][field] = value
                with self.assertRaisesRegex(ContractError, "Kiraa"):
                    reconcile(ROOT, changed)

    def test_deterministic_candidates_remain_research_until_contract_exists(self):
        for entry in self.inventory["entries"]:
            if entry["name"].startswith(("VectorStore Cosine Search", "Kalman Filter")):
                self.assertEqual(entry["suite"], "research")
                self.assertEqual(entry["replacements"], [])

    def test_discovery_detects_new_static_registrations_and_rejects_dynamic_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            swift = root / "Benchmarks/Swift"
            python = root / "Benchmarks/Python"
            swift.mkdir(parents=True)
            python.mkdir(parents=True)
            (swift / "AccuracyBenchmarks.swift").write_text("")
            (swift / "ExtraBenchmarks.swift").write_text(
                'BenchmarkRunner.run(name: "New Swift", module: module) {}'
            )
            (python / "benchmarks.py").write_text(
                'run_benchmark("New Python", "module", callback)'
            )
            (python / "accuracy_benchmarks.py").write_text(
                'results["new_accuracy"] = {}'
            )
            self.assertEqual(
                {x["name"] for x in discover(root)},
                {"New Swift", "New Python", "new_accuracy"},
            )
            (python / "benchmarks.py").write_text(
                'run_benchmark(dynamic_name, "module", callback)'
            )
            with self.assertRaisesRegex(ContractError, "Dynamic Python"):
                discover(root)
            (python / "benchmarks.py").write_text("")
            (swift / "ExtraBenchmarks.swift").write_text(
                "BenchmarkRunner.run(name: dynamicName, module: module) {}"
            )
            with self.assertRaisesRegex(ContractError, "Unrecognized Swift"):
                discover(root)


if __name__ == "__main__":
    unittest.main()
