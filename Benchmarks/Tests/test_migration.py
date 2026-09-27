import unittest
import tempfile
from pathlib import Path
from test_contracts import ROOT
from contracts import load_workload
from datasets import (
    average_ranks,
    correlation,
    grouped_reference,
    generate_table,
    load_manifest,
    prepare,
    reference,
)


class MigrationReferences(unittest.TestCase):
    def test_tied_ranks_and_correlations(self):
        self.assertEqual(average_ranks([4, 1, 4, 2]), [3.5, 1, 3.5, 2])
        self.assertAlmostEqual(correlation([1, 2, 3], [2, 4, 6]), 1)
        self.assertAlmostEqual(correlation([1, 2, 3], [6, 4, 2]), -1)

    def test_chunk_group_reference_and_retained_join_values(self):
        rows = [(0, 1, 2, 10), (1, 0, 4, 20), (2, 1, 6, 30)]
        self.assertEqual(grouped_reference(rows), [[0, 4, 20], [1, 8, 20]])
        dataset = load_manifest(ROOT, "tiny")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prepare(root, dataset)

            def result(name):
                return reference(root, dataset, load_workload(ROOT, name + "-v1"))

            self.assertEqual(result("csv-stream-read"), result("csv-read"))
            self.assertEqual(result("csv-stream-filter"), result("filter"))
            joined = result("inner-join")
            original = result("csv-read")
            self.assertEqual(len(joined), dataset["rows"] * 5)
            for i in range(dataset["rows"]):
                self.assertEqual(joined[i * 5 : i * 5 + 4], original[i * 4 : i * 4 + 4])
                self.assertEqual(joined[i * 5 + 4], i / 4)
            grouped = result("group-sum-mean")
            stream = result("csv-stream-group")
            self.assertEqual(
                stream,
                [
                    v
                    for i in range(0, len(grouped), 3)
                    for v in [0, *grouped[i : i + 3]]
                ],
            )
            self.assertEqual(result("row-sum"), [sum(original[2::4])])

    def test_stream_groups_preserve_chunk_boundaries(self):
        import datasets
        from unittest.mock import patch

        rows = [(i, 0, float(i), float(i)) for i in range(10001)]
        with patch.object(datasets, "values", return_value=rows):
            result = reference(
                ROOT, {"kind": "table-v1"}, {"operation": "csv-stream-group"}
            )
        self.assertEqual(result, [0, 0, 49995000, 4999.5, 1, 0, 10000, 10000])


class MixedFixtures(unittest.TestCase):
    def test_pinned_mixed_and_parquet_values_agree(self):
        from datasets import values

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            csv = load_manifest(ROOT, "mixed-tiny")
            parquet = load_manifest(ROOT, "parquet-tiny")
            prepare(root, csv)
            prepare(root, parquet)
            self.assertEqual(values(root, csv), values(root, parquet))
            self.assertEqual(values(root, csv)[0], (0, 0, -126, -87.5, 1))
            matrix = reference(root, csv, load_workload(ROOT, "flat-matrix-v1"))
            self.assertEqual(len(matrix), 2 * csv["rows"])

    def test_every_parquet_write_requires_independent_artifact(self):
        import pyarrow.parquet as pq
        import pyarrow as pa
        from runner import validate_parquet_artifacts

        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "response.json"
            path = Path(str(base) + ".sample1.parquet")
            request = dict(warmups=1, samples=2, rows=1, atol=0, rtol=0)
            table = pa.table(
                dict(id=[0], group=["category0"], x=[2.0], y=[3.0], flag=[True])
            )
            pq.write_table(table, path)
            with self.assertRaises(FileNotFoundError):
                validate_parquet_artifacts(base, request, [0, 0, 2, 3, 1])
            pq.write_table(table, Path(str(base) + ".sample2.parquet"))
            self.assertEqual(
                len(validate_parquet_artifacts(base, request, [0, 0, 2, 3, 1])), 2
            )
            with self.assertRaises(ValueError):
                validate_parquet_artifacts(base, request, [0, 0, 9, 3, 1])


class NumericalReferences(unittest.TestCase):
    def test_welch_closed_form_df_two(self):
        import math
        from numerical_reference import welch

        result = welch([1, 3], [1, 3])
        critical = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
        expected = [0, 1, 2, -critical * math.sqrt(2), critical * math.sqrt(2), 0]
        for a, e in zip(result, expected):
            self.assertAlmostEqual(a, e, places=12)

    def test_metrics_ranks_and_text_reference(self):
        import math
        from numerical_reference import regression_metrics, auc, tfidf

        expected = [math.sqrt(5 / 3), 1, 100 / 3, -1 / 14]
        for a, e in zip(regression_metrics([1, 2, 4], [1, 3, 2]), expected):
            self.assertAlmostEqual(a, e, places=12)
        self.assertEqual(auc([0, 1, 0, 1], [0, 0, 1, 1]), 0.5)
        self.assertEqual(auc([0, 0, 1, 1], [1, 2, 3, 4]), 1)
        self.assertEqual(auc([0, 0, 1, 1], [4, 3, 2, 1]), 0)
        idf = math.log(1.5) + 1
        self.assertEqual(tfidf(2), [idf / 3, 2 / 3, 0, 0, 0.5, idf / 2])


class MigrationContracts(unittest.TestCase):
    def test_workload_schema_matches_executable_registry(self):
        import json
        from contracts import WORKLOADS, load_profile

        schema = json.loads(
            (ROOT / "Benchmarks/Specs/schemas/workload.json").read_text()
        )
        self.assertEqual(set(schema["properties"]["operation"]["enum"]), WORKLOADS)
        for name in ["migration-smoke", "migration"]:
            for case in load_profile(ROOT, name)["cases"]:
                self.assertIn(
                    load_workload(ROOT, case["workload"])["operation"], WORKLOADS
                )

    def test_paired_student_and_anova_references(self):
        from numerical_reference import welch, anova

        student = welch([1, 3], [1, 3], method="student")
        self.assertEqual(student[:3], [0, 1, 2])
        paired = welch([1, 2, 3], [2, 2, 2], method="paired")
        self.assertEqual(paired[:3], [0, 1, 2])
        # Equal group means give F=0, p=1 and zero explained variance.
        self.assertEqual(anova([[1, 3], [0, 4], [2, 2]]), [0, 1, 2, 3, 0])


class ArtifactAudit(unittest.TestCase):
    def test_missing_changed_and_unvalidated_artifacts_fail(self):
        from contracts import digest
        from reporting import audit_artifacts

        event = dict(case_id="write", engine="swiftsci", batch=0)
        profile = dict(warmups=1, samples=1)
        with self.assertRaises(ValueError):
            audit_artifacts(event, profile)
        data = b"validated artifact bytes"
        record = dict(
            file="write-swiftsci-0.response.json.sample1.parquet",
            sha256=digest(data),
            bytes=len(data),
            independently_validated=True,
        )
        event["artifacts"] = [record]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / record["file"]
            path.write_bytes(data)
            audit_artifacts(event, profile, root)
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                audit_artifacts(event, profile, root)
            record["independently_validated"] = False
            with self.assertRaises(ValueError):
                audit_artifacts(event, profile)
