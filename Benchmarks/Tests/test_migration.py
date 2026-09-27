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
