import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from fractions import Fraction
from test_contracts import ROOT
from contracts import ContractError, load_profile, load_workload, validate_values
from datasets import load_manifest, prepare, reference, resolve_workload
from numerical_fixtures import validate_input


class ExactModels(unittest.TestCase):
    def test_pinned_cases_have_complete_outputs(self):
        cases = load_profile(ROOT, 'model-conformance')['cases']
        self.assertEqual(len(cases), 3)
        lengths = {'pca-oblique-k2': 51, 'pca-oblique-k1': 45, 'multinomial-nb-noncontiguous': 17}
        for case in cases:
            m = load_manifest(ROOT, case['dataset'])
            w = resolve_workload(m, load_workload(ROOT, case['workload']))
            prepare(ROOT, m)
            self.assertEqual(len(reference(ROOT, m, w)), lengths[m['id']])
            self.assertEqual((w['atol'], w['rtol']), (1e-12, 1e-12))

    def test_stdlib_rational_generator_reproduces_all_bytes(self):
        original = ROOT / 'Benchmarks/Fixtures/exact-models'
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shutil.copyfile(original / 'source-spec.json', base / 'source-spec.json')
            subprocess.run([sys.executable, str(ROOT / 'Benchmarks/Tools/build_exact_model_fixtures.py'), '--base-dir', str(base)], check=True, capture_output=True)
            self.assertEqual((original / 'inventory.json').read_bytes(), (base / 'inventory.json').read_bytes())
            for part in ['inputs', 'references']:
                for path in (original / part).glob('*.json'):
                    self.assertEqual(path.read_bytes(), (base / part / path.name).read_bytes())

    def test_retained_variance_is_not_total_variance(self):
        m = load_manifest(ROOT, 'pca-oblique-k1')
        prepare(ROOT, m)
        expected = reference(ROOT, m, load_workload(ROOT, 'pca-cpu-v1'))
        self.assertEqual(expected[:4], [10., -2., 200/3, .8])
        wrong = expected.copy(); wrong[3] = 1
        with self.assertRaises(ContractError):
            validate_values(wrong, expected, 1e-12, 1e-12)

    def test_class_indices_and_noncontiguous_labels_are_distinct(self):
        m = load_manifest(ROOT, 'multinomial-nb-noncontiguous')
        prepare(ROOT, m)
        result = reference(ROOT, m, load_workload(ROOT, 'multinomial-nb-cpu-v1'))
        self.assertEqual(result[:2], [10, 20])
        self.assertEqual(result[-5:], [1, 1, 0, 1, 0])
        self.assertEqual(result[2:4], [1/3, 2/3])
        for i in range(2, 12, 2):
            self.assertAlmostEqual(sum(result[i:i+2]), 1)

    def test_cross_gram_detects_inconsistent_transform_signs(self):
        def gram(a, b):
            return [sum(x*y for x, y in zip(row, other)) for row in a for other in b]
        training = [[10, 0], [-10, 0], [0, 5], [0, -5]]
        query = [[5, 0], [0, 5], [0, 0]]
        flip = lambda rows: [[-row[0], row[1]] for row in rows]
        self.assertEqual(gram(query, query), gram(flip(query), flip(query)))
        self.assertEqual(gram(training, query), gram(flip(training), flip(query)))
        self.assertNotEqual(gram(training, query), gram(training, flip(query)))
        ref = json.loads((ROOT / 'Benchmarks/Fixtures/exact-models/references/pca-oblique-k2.json').read_text())
        part = next(v for v in ref['outputOrder'] if v['name'] == 'trainingQueryScoreGram')
        actual = ref['exactRationalReference']['values'][part['offset']:part['offset']+part['count']]
        self.assertEqual(actual, gram(training, query))

    def test_invalid_model_shapes_and_smoothing_rejected(self):
        base = ROOT / 'Benchmarks/Fixtures/exact-models/inputs'
        nb = json.loads((base/'multinomial-nb-noncontiguous.json').read_text())
        for key, value in [('alpha', 0), ('alpha', True), ('targets', [10]), ('query', [[-1, 0]]), ('query', [[1]]), ('query', [])]:
            payload = dict(nb, **{key: value})
            with self.assertRaises(ContractError):
                validate_input(payload, 'multinomial-nb-cpu', 3)
        pca = json.loads((base/'pca-oblique-k2.json').read_text())
        for k in [0, 3, True]:
            with self.assertRaises(ContractError):
                validate_input(dict(pca, n_components=k), 'pca-cpu', 4)
