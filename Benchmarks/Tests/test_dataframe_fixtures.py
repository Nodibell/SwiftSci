import copy
import importlib.util
import math
import sys
import unittest
from test_contracts import ROOT
from contracts import ContractError, read_json, validate_values, load_profile
from dataframe_fixtures import cell, decoded_columns, encode_frames, expected_values, float_words, validate_input
from datasets import load_manifest
from numerical_fixtures import source_and_input, expected_values as pinned_values, validate_manifest
sys.path.insert(0, str(ROOT / 'Benchmarks/Python'))
from dataframe_semantic_workloads import prepare, execute


class DataframeFixtures(unittest.TestCase):
    def payload(self, slug):
        return read_json(ROOT / f'Benchmarks/Fixtures/dataframe/inputs/dataframe-{slug}.json')

    def test_word_encoding_distinguishes_precision_null_nan_and_signed_zero(self):
        def encoded(value):
            return encode_frames([[{'name': 'i', 'type': 'int64', 'values': [value]}]])
        self.assertEqual(encoded(2**53 + 1)[-2:], [2097152, 1])
        self.assertEqual(encoded(-(2**63))[-2:], [2147483648, 0])
        self.assertEqual(encoded(2**63 - 1)[-2:], [2147483647, 4294967295])
        self.assertNotEqual(encoded(2**53), encoded(2**53 + 1))
        self.assertNotEqual(float_words(-0.0), float_words(0.0))
        self.assertEqual(float_words(math.nan), [2146959360, 0])
        nil = encode_frames([[{'name': 'f', 'type': 'float64', 'values': [None]}]])
        nan = encode_frames([[{'name': 'f', 'type': 'float64', 'values': [math.nan]}]])
        self.assertNotEqual(nil, nan)

    def test_hand_checked_filter_rows(self):
        expected = {'eq': [3, 4], 'ne': [1, 2, 5, 6], 'lt': [2], 'le': [2, 3, 4],
                    'gt': [5, 6], 'ge': [3, 4, 5, 6], 'isnull': [0], 'isnotnull': [1, 2, 3, 4, 5, 6]}
        for name, indices in expected.items():
            p = self.payload('filter-float-' + name)
            source = decoded_columns(p)
            selected = [dict(c, values=[c['values'][i] for i in indices]) for c in source]
            self.assertEqual(expected_values(p), encode_frames([source, selected]))
        p = self.payload('filter-int64-exact')
        source = decoded_columns(p)
        selected = [dict(c, values=[c['values'][i] for i in [1, 3, 5]]) for c in source]
        self.assertEqual(expected_values(p), encode_frames([source, selected]))

    def test_hand_checked_stable_descending_order(self):
        for slug, indices in [('sort-int64-desc', [3, 1, 5, 4, 0, 2]),
                              ('sort-float-desc', [6, 0, 4, 2, 3, 5, 1])]:
            p = self.payload(slug)
            source = decoded_columns(p)
            result = [dict(c, values=[c['values'][i] for i in indices]) for c in source]
            self.assertEqual(expected_values(p), encode_frames([source, result]))

    def test_hand_checked_matrix_positions_and_conversion(self):
        p = self.payload('matrix-nonsquare')
        source = decoded_columns(p)
        flat = [1.25, float(2**53), math.nan, -7.0, -2.0, math.nan]
        values = encode_frames([source]) + [3, 2]
        for v in flat + flat:
            values += float_words(v)
        values += [3]
        for v in [1.0, 0.0, math.nan]:
            values += float_words(v)
        self.assertEqual(expected_values(p), values)

    def test_hand_checked_group_counts_and_missing_distinction(self):
        p = self.payload('group-int64')
        source = decoded_columns(p)
        keys = {'name': 'key', 'type': 'utf8', 'values': ['9007199254740992', '9007199254740993', None]}
        counts = [keys] + [{'name': name, 'type': 'int64', 'values': [2, 1, 2]} for name in ['v', 'other']]
        present = [keys, {'name': 'v_count', 'type': 'float64', 'values': [2.0, 0.0, 1.0]}]
        self.assertEqual(expected_values(p), encode_frames([source, counts, present]))

    def test_all_pinned_answers_and_pandas_outputs(self):
        profile = load_profile(ROOT, 'dataframe-conformance')
        for case in profile['cases']:
            with self.subTest(case=case['id']):
                manifest = load_manifest(ROOT, case['dataset'])
                source_and_input(ROOT, manifest)
                p = read_json(ROOT / manifest['fixture'])
                answer = pinned_values(ROOT, manifest, p)
                self.assertEqual(execute(prepare(p)), answer)
                # Source fixture is unchanged by repeated execution.
                self.assertEqual(execute(prepare(p)), answer)
                for index in (0, len(answer)//2, len(answer)-1):
                    wrong = list(answer); wrong[index] += 1
                    with self.assertRaises(ContractError):
                        validate_values(wrong, answer, 0, 0)

    def test_strict_input_boundaries(self):
        base = self.payload('gather-int64')
        invalid = []
        for value in ['+1', '01', '-0', '9223372036854775808', '-9223372036854775809', 1, True]:
            p = copy.deepcopy(base); p['columns'][0]['values'][0] = value; invalid.append(p)
        for indices in [[-1], [6], [True], [1.0], [1.5]]:
            p = copy.deepcopy(base); p['parameters']['indices'] = indices; invalid.append(p)
        p = copy.deepcopy(base); p['columns'][1]['name'] = 'id'; invalid.append(p)
        p = copy.deepcopy(base); p['columns'][0]['values'].pop(); invalid.append(p)
        p = copy.deepcopy(base); p['parameters']['unknown'] = 1; invalid.append(p)
        for slug, key, value in [('sort-int64-desc', 'ascending', 1), ('select-order', 'columns', ['id','id']),
                                 ('select-order', 'columns', []), ('matrix-nullable', 'target', 'missing'),
                                 ('replace-isolation', 'values', ['1']), ('group-int64', 'value', 'key')]:
            p = self.payload(slug); p['parameters'][key] = value; invalid.append(p)
        p = self.payload('sort-float-asc'); p['columns'][0]['values'][0] = 'NaN'; invalid.append(p)
        for p in invalid:
            with self.subTest(payload=p):
                with self.assertRaises(ContractError):
                    validate_input(p, 'dataframe-semantics', len(base['columns'][0]['values']) if p['action']=='gather' else len(p['columns'][0]['values']))
        manifest = load_manifest(ROOT, 'dataframe-gather-int64')
        manifest['tolerances']['rtol'] = 1e-9
        with self.assertRaises(ContractError): validate_manifest(manifest)

    def test_generator_reconstructs_every_input(self):
        path = ROOT / 'Benchmarks/Fixtures/dataframe/generate.py'
        spec = importlib.util.spec_from_file_location('dataframe_generator', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for name, p in module.cases():
            self.assertEqual(p, self.payload(name))
