"""Reject schema mismatches before accepting CSV comparison evidence."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from contracts import ContractError, load_workload
from csv_contract import requested_schema, schema_for, validate_schema
from datasets import load_manifest, resolve_workload
from reporting import validate_worker


class CSVComparisonTests(unittest.TestCase):
    def test_complete_schema_is_required_and_output_types_are_checked(self):
        for kind in ('table-v1', 'mixed-table-v1', 'wine-quality-v1'):
            expected = schema_for(kind)
            request = dict(dataset_kind=kind, csv_schema=expected)
            self.assertEqual(validate_schema(expected, request), expected)
            for bad in (None, {}, dict(expected, x='Float'), dict(expected, id='Double')):
                with self.subTest(kind=kind, schema=bad):
                    with self.assertRaises(ContractError):
                        requested_schema(dict(request, csv_schema=bad))
                    with self.assertRaises(ContractError):
                        validate_schema(bad, request)

    def test_schema_is_part_of_the_resolved_versioned_workload(self):
        for dataset in ('tiny', 'mixed-tiny'):
            manifest = load_manifest(ROOT, dataset)
            workload = resolve_workload(manifest, load_workload(ROOT, 'csv-read-v2'))
            self.assertEqual(workload['csv_schema'], schema_for(manifest['kind']))
        with self.assertRaises(ContractError):
            schema_for('parquet-table-v1')

    def test_missing_or_mismatched_worker_schema_cannot_pass(self):
        schema = schema_for('table-v1')
        request = dict(case_key='key', samples=1, csv_schema=schema)
        result = dict(schema_version=1, case_key='key', status='passed', peak_rss_bytes=1,
                      engine_version='test', csv_schema=schema,
                      samples=[dict(elapsed_ns=1001, timing_resolved=True, output_sha256='a'*64,
                                    maximum_absolute_error=0, validated=True)])
        validate_worker(result, request)
        for bad in (None, {}, dict(schema, x='Float')):
            altered = copy.deepcopy(result)
            altered['csv_schema'] = bad
            with self.assertRaises(ContractError):
                validate_worker(altered, request)


if __name__ == '__main__':
    unittest.main()
