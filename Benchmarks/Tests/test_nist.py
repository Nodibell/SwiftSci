import copy
from decimal import Decimal, localcontext
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Benchmarks/Tools"))
from contracts import ContractError, load_profile, load_workload, verified_file
from datasets import load_manifest, parse_univariate, resolve_workload

NAMES = [
    "pidigits",
    "lottery",
    "lew",
    "mavro",
    "michelso",
    "numacc1",
    "numacc2",
    "numacc3",
    "numacc4",
]


class NISTTests(unittest.TestCase):
    def test_all_nine_original_files_and_decimal_reference_answers(self):
        for name in NAMES:
            with self.subTest(dataset=name):
                m = load_manifest(ROOT, "nist-" + name)
                data = verified_file(ROOT / m["fixture"], m["sha256"], m["size_bytes"])
                lines = data.decode("ascii").splitlines()
                marker = next(
                    i for i, line in enumerate(lines) if line.strip() == "Data: Y"
                )
                self.assertEqual(m["data_start_line"], marker + 3)
                mean = Decimal(
                    next(line for line in lines if line.startswith("Sample Mean"))
                    .split(":")[1]
                    .split()[0]
                )
                sd = Decimal(
                    next(
                        line
                        for line in lines
                        if line.startswith("Sample Standard Deviation")
                    )
                    .split(":")[1]
                    .split()[0]
                )
                count = int(
                    next(
                        line
                        for line in lines
                        if line.startswith("Number of Observations:")
                    ).split(":")[1]
                )
                self.assertEqual(m["rows"], count)
                self.assertEqual(m["certified"]["mean"], float(mean))
                self.assertEqual(m["certified"]["stddev"], float(sd))
                values = parse_univariate(data, m["data_start_line"], m["rows"])
                self.assertEqual(len(values), count)
                with localcontext() as context:
                    context.prec = 60
                    decimals = [
                        Decimal(t.strip())
                        for t in lines[m["data_start_line"] - 1 :]
                        if t.strip()
                    ]
                    exact_mean = sum(decimals) / len(decimals)
                    exact_variance = sum((v - exact_mean) ** 2 for v in decimals) / (
                        len(decimals) - 1
                    )
                    self.assertEqual(m["certified"]["variance"], float(sd * sd))
                    for actual, expected in [
                        (exact_mean, mean),
                        (exact_variance.sqrt(), sd),
                        (exact_variance, sd * sd),
                    ]:
                        self.assertLessEqual(
                            abs(actual - expected), abs(expected) * Decimal("2e-14")
                        )
                for operation, relative in [
                    ("mean", 1e-14),
                    ("stddev", 1e-12),
                    ("variance", 2e-12),
                ]:
                    self.assertEqual(m["tolerances"][operation]["rtol"], relative)
                budget = 2 * max(math.ulp(v) for v in values)
                self.assertEqual(m["tolerances"]["mean"]["atol"], budget)
                self.assertEqual(m["tolerances"]["stddev"]["atol"], budget)
                self.assertEqual(
                    m["tolerances"]["variance"]["atol"],
                    2 * float(sd) * budget + budget * budget,
                )

    def test_profiles_cover_every_dataset_and_statistic_once(self):
        expected = {
            ("nist-" + name, op + "-v1")
            for name in NAMES
            for op in ["mean", "variance", "stddev"]
        }
        for profile in ["certification", "nist"]:
            p = load_profile(ROOT, profile)
            self.assertEqual(len(p["cases"]), 27)
            self.assertEqual(
                {(c["dataset"], c["workload"]) for c in p["cases"]}, expected
            )

    def test_variable_header_length_crlf_and_scientific_notation(self):
        self.assertEqual(
            parse_univariate(b"Header\r\n\r\n1e-3\r\n -2.5\r\n", 3, 2), [0.001, -2.5]
        )
        self.assertEqual(parse_univariate(b"1\n2\n", 1, 2), [1, 2])

    def test_truncated_extra_nonfinite_invalid_and_bad_metadata_are_rejected(self):
        for data, start, rows in [
            (b"1\n", 1, 2),
            (b"1\n2\n3\n", 1, 2),
            (b"NaN\n1\n", 1, 2),
            (b"inf\n1\n", 1, 2),
            (b"x\n1\n", 1, 2),
            (b"\xff\n1\n", 1, 2),
            (b"1\n2\n", 99, 2),
            (b"1\n2\n", 0, 2),
            (b"1\n", 1, 1),
        ]:
            with (
                self.subTest(data=data, start=start, rows=rows),
                self.assertRaises(ValueError),
            ):
                parse_univariate(data, start, rows)

    def test_only_declared_statistics_can_use_nist_inputs(self):
        m = load_manifest(ROOT, "nist-mavro")
        for operation in ["mean", "stddev", "variance"]:
            w = load_workload(ROOT, operation + "-v1")
            resolved = resolve_workload(m, w)
            self.assertEqual(resolved["atol"], m["tolerances"][operation]["atol"])
            self.assertEqual(w, load_workload(ROOT, operation + "-v1"))
        with self.assertRaises(ContractError):
            resolve_workload(m, load_workload(ROOT, "filter-v1"))

    def test_manifest_rejects_missing_or_invalid_reference_metadata(self):
        base = load_manifest(ROOT, "nist-mavro")
        for key, value in [
            ("data_start_line", 0),
            ("certified", {}),
            ("tolerances", {}),
            ("tolerances", dict(base["tolerances"], mean={"atol": -1, "rtol": 0})),
        ]:
            invalid = copy.deepcopy(base)
            invalid[key] = value
            with (
                self.subTest(key=key),
                patch("datasets.read_json", return_value=invalid),
                self.assertRaises(ContractError),
            ):
                load_manifest(ROOT, "nist-mavro")


if __name__ == "__main__":
    unittest.main()
