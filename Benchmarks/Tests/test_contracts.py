import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
from contracts import *
from datasets import *
from reporting import *


class Contracts(unittest.TestCase):
    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            for value in ['{"a":1,"a":2}', '{"a":NaN}']:
                p.write_text(value)
                with self.assertRaises(ValueError):
                    read_json(p)

    def test_manifest_matches_generated_bytes(self):
        for name in ["tiny", "medium", "large"]:
            m = load_manifest(ROOT, name)
            data = generate_table(m["rows"], m["groups"])
            self.assertEqual(digest(data), m["sha256"])
            self.assertEqual(len(data), m["size_bytes"])

    def test_corruption_never_regenerates(self):
        m = load_manifest(ROOT, "tiny")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = prepare(root, m)
            p.write_bytes(b"wrong")
            with self.assertRaises(ContractError):
                prepare(root, m)

    def test_validating_values(self):
        for values in [[], [float("nan")], [float("inf")], [2]]:
            with self.assertRaises(ContractError):
                validate_values(values, [1], 0, 0)
        self.assertGreater(validate_values([1.001], [1], 0.002, 0), 0)

    def test_reference_known_by_hand(self):
        m = load_manifest(ROOT, "tiny")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepare(root, m)
            result = reference(root, m, load_workload(ROOT, "target-v1"))
            self.assertEqual(result[:3], [-126, -116.75, -107.5])
            self.assertEqual(
                len(reference(root, m, load_workload(ROOT, "group-sum-v1"))), 14
            )

    def test_nist_values_and_input(self):
        m = load_manifest(ROOT, "nist-numacc4")
        data = verified_file(ROOT / m["fixture"], m["sha256"], m["size_bytes"])
        values = [float(x) for x in data.decode().splitlines()[60:] if x.strip()]
        self.assertEqual(len(values), 1001)
        self.assertAlmostEqual(math.fsum(values) / len(values), m["certified"]["mean"])
        self.assertAlmostEqual(
            math.sqrt(
                math.fsum((x - m["certified"]["mean"]) ** 2 for x in values) / 1000
            ),
            0.1,
            places=8,
        )

    def test_failed_zero_and_unvalidated_worker_rejected(self):
        req = dict(case_key="key", samples=1)
        sample = dict(
            elapsed_ns=100,
            output_sha256="a" * 64,
            maximum_absolute_error=0,
            validated=True,
        )
        base = dict(
            schema_version=1,
            case_key="key",
            samples=[sample],
            status="passed",
            peak_rss_bytes=100,
            engine_version="test",
        )
        validate_worker(base, req)
        for k, v in [("status", "failed"), ("case_key", "other"), ("samples", [])]:
            bad = copy.deepcopy(base)
            bad[k] = v
            with self.assertRaises(ContractError):
                validate_worker(bad, req)
        for k, v in [
            ("elapsed_ns", 0),
            ("validated", False),
            ("maximum_absolute_error", float("nan")),
        ]:
            bad = copy.deepcopy(base)
            bad["samples"][0][k] = v
            with self.assertRaises(ContractError):
                validate_worker(bad, req)

    def test_path_cannot_escape_repository(self):
        with self.assertRaises(ContractError):
            repository_file(ROOT, "../outside")

    def test_all_profiles_resolve(self):
        for name in ["smoke", "standard", "extended", "certification"]:
            profile = load_profile(ROOT, name)
            for case in profile["cases"]:
                load_manifest(ROOT, case["dataset"])
                load_workload(ROOT, case["workload"])

    def test_comparison_refuses_failed_or_different_contract(self):
        baseline = dict(schema_version=1, status="failed")
        with self.assertRaises(ContractError):
            comparison(baseline, baseline)
        baseline = dict(
            schema_version=1,
            status="passed",
            plan=dict(contract_hash="a", environment={}),
        )
        other = copy.deepcopy(baseline)
        other["plan"]["contract_hash"] = "b"
        with self.assertRaises(ContractError):
            comparison(baseline, other)


if __name__ == "__main__":
    unittest.main()
