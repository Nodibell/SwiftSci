import json
import csv
import io
import tempfile
import unittest
from pathlib import Path
from test_contracts import ROOT
from contracts import ContractError, digest, load_profile, load_workload, validate_values
from datasets import load_manifest, prepare, reference, resolve_workload, values
from public_datasets import generate_h2o, group_reference, wine_reference


class PublicData(unittest.TestCase):
    def test_all_public_cases_resolve_and_validate(self):
        for profile in ["public-smoke", "public-data"]:
            for case in load_profile(ROOT, profile)["cases"]:
                dataset = load_manifest(ROOT, case["dataset"])
                resolve_workload(dataset, load_workload(ROOT, case["workload"]))

    def test_generated_bytes_are_pinned_at_both_scales(self):
        for size in ["tiny", "medium"]:
            for distribution in ["uniform", "hot-key"]:
                manifest = load_manifest(ROOT, f"h2o-{size}-{distribution}")
                data = generate_h2o(manifest["rows"], manifest["groups"], distribution)
                self.assertEqual(digest(data), manifest["sha256"])
                self.assertEqual(len(data), manifest["size_bytes"])
                rows = list(csv.DictReader(io.StringIO(data.decode())))
                unique_pairs = {(r["id1"], r["id2"]) for r in rows}
                self.assertGreater(len(unique_pairs), manifest["groups"])
                hot = sum(r["id1"] == "key0" for r in rows) / len(rows)
                self.assertGreater(hot, 0.79) if distribution == "hot-key" else self.assertLess(hot, 0.2)

    def test_hand_computed_all_five_queries_and_numeric_key_order(self):
        rows = [
            dict(id1="key10", id2="key2", id3="key0", id4="0", id5="0", id6="1", v1="1", v2="2", v3="0.25"),
            dict(id1="key2", id2="key2", id3="key0", id4="0", id5="1", id6="1", v1="3", v2="4", v3="0.75"),
            dict(id1="key10", id2="key1", id3="key1", id4="1", id5="2", id6="2", v1="5", v2="6", v3="1.25"),
        ]
        expected = {
            "h2o-q1": [2, 3, 10, 6],
            "h2o-q2": [2, 2, 3, 10, 1, 5, 10, 2, 1],
            "h2o-q3": [0, 4, 0.5, 1, 5, 1.25],
            "h2o-q4": [0, 2, 3, 0.5, 1, 5, 6, 1.25],
            "h2o-q5": [1, 4, 6, 1, 2, 5, 6, 1.25],
        }
        for query, answer in expected.items():
            self.assertEqual(group_reference(rows, query), answer)

    def test_unrelated_workloads_and_datasets_rejected(self):
        for dataset, workload in [("tiny", "h2o-q1"), ("h2o-tiny-uniform", "group-sum"), ("wine-quality-red", "csv-read"), ("tiny", "wine-pipeline"), ("h2o-tiny-uniform", "wine-pipeline"), ("wine-quality-red", "h2o-q1")]:
            with self.assertRaises(ContractError):
                resolve_workload(load_manifest(ROOT, dataset), load_workload(ROOT, workload + "-v1"))

    def test_wine_pin_transformation_and_full_pipeline(self):
        manifest = load_manifest(ROOT, "wine-quality-red")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / manifest["fixture"]
            source.parent.mkdir(parents=True)
            source.write_bytes((ROOT / manifest["fixture"]).read_bytes())
            prepared = prepare(root, manifest)
            self.assertEqual(digest(prepared.read_bytes()), manifest["sha256"])
            rows = values(root, manifest)
            self.assertEqual(len(rows), 1599)
            output = wine_reference(rows)
            self.assertEqual(len(output), 855 * 13)
            self.assertEqual(output[:10], [544, 371, 528, 99, 102, 234, 235, 236, 238, 239])
            self.assertEqual(output[845:855], [1132, 1228, 1477, 142, 144, 467, 588, 821, 1269, 1270])
            self.assertEqual(sum(output[:855]), 720283)
            self.assertEqual(sum(output[855:1710]), 5365)
            matrix = output[1710:]
            for j in range(11):
                column = matrix[j::11]
                self.assertAlmostEqual(sum(column), 0, places=10)
                self.assertAlmostEqual(sum(v*v for v in column) / 855, 1, places=12)
            wrong = output.copy()
            wrong[0], wrong[1] = wrong[1], wrong[0]
            with self.assertRaises(ContractError):
                validate_values(wrong, output, 1e-10, 1e-10)

    def test_wine_source_corruption_rejected_before_transformation(self):
        manifest = load_manifest(ROOT, "wine-quality-red")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / manifest["fixture"]
            source.parent.mkdir(parents=True)
            source.write_bytes(b"wrong")
            with self.assertRaises(ContractError):
                prepare(root, manifest)
            self.assertFalse((root / "Benchmarks/Data").exists())

    def test_public_manifests_reject_missing_or_invalid_provenance(self):
        mutations = [
            ("h2o-tiny-uniform", "distribution", None),
            ("h2o-tiny-uniform", "distribution", "unknown"),
            ("h2o-tiny-uniform", "groups", 258),
            ("wine-quality-red", "source_sha256", None),
            ("wine-quality-red", "source_sha256", "z" * 64),
            ("wine-quality-red", "source_size_bytes", None),
            ("wine-quality-red", "source_size_bytes", 0),
            ("wine-quality-red", "source_size_bytes", True),
            ("wine-quality-red", "fixture", None),
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            directory = root / "Benchmarks/Specs/datasets"
            directory.mkdir(parents=True)
            for name, field, value in mutations:
                with self.subTest(dataset=name, field=field, value=value):
                    manifest = load_manifest(ROOT, name)
                    if value is None:
                        del manifest[field]
                    else:
                        manifest[field] = value
                    (directory / (name + ".json")).write_text(json.dumps(manifest))
                    with self.assertRaises(ContractError):
                        load_manifest(root, name)

    def test_generated_cache_corruption_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = load_manifest(ROOT, "h2o-tiny-uniform")
            path = prepare(root, manifest)
            path.write_bytes(b"wrong")
            with self.assertRaises(ContractError):
                prepare(root, manifest)
            self.assertEqual(path.read_bytes(), b"wrong")
