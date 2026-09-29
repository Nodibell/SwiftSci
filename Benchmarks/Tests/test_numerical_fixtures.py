import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from test_contracts import ROOT
from contracts import ContractError, load_profile, load_workload
from datasets import load_manifest, prepare, reference, resolve_workload
from numerical_fixtures import validate_input


class NumericalFixtures(unittest.TestCase):
    def fixture(self, name):
        manifest = load_manifest(ROOT, name)
        workload = resolve_workload(manifest, load_workload(ROOT, manifest['operation'] + '-v1'))
        return manifest, workload

    def test_all_38_cases_have_pinned_inputs_and_complete_answers(self):
        for profile in ['numerical-conformance', 'numerical-binary64']:
            cases = load_profile(ROOT, profile)['cases']
            self.assertEqual(len(cases), 19)
            for case in cases:
                manifest, workload = self.fixture(case['dataset'])
                path = prepare(ROOT, manifest)
                payload = json.loads(path.read_text())
                values = reference(ROOT, manifest, workload)
                expected_length = len(payload['features'][0]) + 2 + manifest['rows'] if manifest['operation'] == 'ols-cpu' else 3
                self.assertEqual(len(values), expected_length)
                self.assertEqual(workload['rtol'], 1e-9)

    def test_decimal_and_binary64_answers_are_not_conflated(self):
        outputs = []
        for basis in ['decimal', 'binary64']:
            m, w = self.fixture('nist-smls09-' + basis)
            prepare(ROOT, m)
            outputs.append(reference(ROOT, m, w))
        self.assertEqual(outputs[0], [2001.0, 8.0, 18000.0])
        self.assertAlmostEqual(outputs[1][0], 2001.1349262209505, places=9)
        self.assertGreater(abs(outputs[0][0] - outputs[1][0]), .13)

    def test_reconstruction_matches_every_committed_byte(self):
        original = ROOT / 'Benchmarks/Fixtures/nist-models'
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            shutil.copytree(original / 'originals', base / 'originals')
            shutil.copyfile(original / 'sources.lock.json', base / 'sources.lock.json')
            subprocess.run([sys.executable, str(ROOT / 'Benchmarks/Tools/build_nist_model_fixtures.py'), '--base-dir', str(base)], check=True, capture_output=True)
            for sub in ['inputs', 'references']:
                for p in (original / sub).glob('*.json'):
                    self.assertEqual(p.read_bytes(), (base / sub / p.name).read_bytes(), p.name)

    def test_corruption_of_source_input_reference_and_cache_is_rejected(self):
        manifest, workload = self.fixture('nist-norris-decimal')
        for target in ['source_fixture', 'fixture', 'reference_fixture', 'cache']:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                for key in ['source_fixture', 'fixture', 'reference_fixture']:
                    p = root / manifest[key]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / manifest[key], p)
                cached = prepare(root, manifest)
                bad = cached if target == 'cache' else root / manifest[target]
                bad.write_bytes(b'{}')
                with self.assertRaises(ContractError):
                    reference(root, manifest, workload)

    def test_payload_boundary_rejects_ragged_nonfinite_and_answer_leakage(self):
        valid = dict(operation='ols-cpu', features=[[1.], [2.], [3.]], targets=[2., 4., 6.])
        invalid = []
        p = copy.deepcopy(valid); p['features'][1] = [1., 2.]; invalid.append(p)
        p = copy.deepcopy(valid); p['targets'][0] = float('nan'); invalid.append(p)
        p = copy.deepcopy(valid); p['expected'] = [0., 2.]; invalid.append(p)
        p = copy.deepcopy(valid); p['features'][0][0] = True; invalid.append(p)
        for payload in invalid:
            with self.assertRaises(ContractError):
                validate_input(payload, 'ols-cpu', 3)
        with self.assertRaises(ContractError):
            validate_input(dict(operation='nist-anova', groups=[[1.], [2.]]), 'nist-anova', 2)

    def test_checksum_valid_sibling_input_cannot_borrow_another_reference(self):
        m, w = self.fixture('nist-smls01-decimal')
        other, _ = self.fixture('nist-smls04-decimal')
        for key in ['fixture', 'sha256', 'size_bytes']:
            m[key] = other[key]
        prepare(ROOT, m)
        with self.assertRaisesRegex(ContractError, 'input identity'):
            reference(ROOT, m, w)

    def test_dataset_operation_pairing_is_explicit(self):
        m, _ = self.fixture('nist-norris-decimal')
        with self.assertRaises(ContractError):
            resolve_workload(m, load_workload(ROOT, 'nist-anova-v1'))
        with self.assertRaises(ContractError):
            resolve_workload(load_manifest(ROOT, 'tiny'), load_workload(ROOT, 'ols-cpu-v1'))
