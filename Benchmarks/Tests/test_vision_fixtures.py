import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from test_contracts import ROOT
from contracts import ContractError, digest, read_json, validate_values
from vision_fixtures import OPERATION, TOLERANCES, geometry, output_count, validate_input
from vision_reference import reference
sys.path.insert(0, str(ROOT/'Benchmarks/Python'))
from vision_workloads import execute, prepare


class VisionFixtures(unittest.TestCase):
    def payload(self, name='rgb-odd-identity'):
        return read_json(ROOT/f'Benchmarks/Fixtures/vision/inputs/vision-cpu-{name}.json')

    def test_pinned_bytes_complete_reference_and_runtime(self):
        profile = read_json(ROOT/'Benchmarks/Specs/profiles/vision-conformance.json')
        self.assertEqual(len(profile['cases']), 8)
        for case in profile['cases']:
            with self.subTest(case=case['id']):
                manifest = read_json(ROOT/f"Benchmarks/Specs/datasets/{case['dataset']}.json")
                for prefix in ('', 'source_', 'reference_'):
                    data = (ROOT/manifest[prefix+'fixture']).read_bytes()
                    self.assertEqual(digest(data), manifest[prefix+'sha256'])
                    self.assertEqual(len(data), manifest[prefix+'size_bytes'])
                lock_path = ROOT/manifest['source_fixture']
                for source in read_json(lock_path)['sources']:
                    data = (lock_path.parent/source['path']).read_bytes()
                    self.assertEqual(digest(data), source['sha256'])
                    self.assertEqual(len(data), source['bytes'])
                payload = read_json(ROOT/manifest['fixture'])
                validate_input(payload, OPERATION, manifest['rows'])
                expected = read_json(ROOT/manifest['reference_fixture'])['independentReference']['values']
                self.assertEqual(expected, reference(payload))
                self.assertEqual(len(expected), output_count(payload))
                self.assertEqual(manifest['tolerances'], TOLERANCES)
                prepared = prepare(payload)
                actual = execute(prepared)
                self.assertEqual(actual, execute(prepared))
                validate_values(actual, expected, **TOLERANCES)

    def test_hand_computed_planar_layout_and_gray_expansion(self):
        payload = self.payload()
        payload.update(width=2, height=1, target_width=2, target_height=1,
                       pixels=[0, .125, .25, .5, .75, 1])
        expected = [1, 1, 2, 3, 0, .25, .75, .125, .5, 1]
        self.assertEqual(reference(payload), expected)
        self.assertEqual(execute(prepare(payload)), expected)
        payload.update(channels=1, pixels=[.25, .75])
        expected = [1, 1, 2, 3, .25, .25, .25, .75, .75, .75]
        self.assertEqual(reference(payload), expected)
        self.assertEqual(execute(prepare(payload)), expected)

    def test_rounding_and_asymmetric_padding_positions(self):
        payload = self.payload('rgb-up-round-half')
        self.assertEqual(geometry(payload), (5, 3, 1, 0))
        values = reference(payload)[4:]
        padding = values[:3]
        self.assertEqual(values[3:6], [.125, .5, .875])
        self.assertEqual(values[5*3:6*3], [.125, .5, .875])
        self.assertEqual(values[6*3:8*3], padding*2)
        payload = self.payload('gray-tall-odd-padding')
        self.assertEqual(geometry(payload), (2, 5, 1, 0))
        values = reference(payload)[4:]
        self.assertEqual(values[:3], [.125]*3)
        self.assertEqual(values[3*3:5*3], [.125]*6)

    def test_every_output_value_is_checked(self):
        payload = self.payload('rgb-up-round-half')
        expected = reference(payload)
        for index in range(len(expected)):
            changed = expected.copy()
            changed[index] += .001
            with self.subTest(index=index), self.assertRaises(ContractError):
                validate_values(changed, expected, **TOLERANCES)
        for changed in (expected[:-1], expected+[0]):
            with self.assertRaises(ContractError):
                validate_values(changed, expected, **TOLERANCES)

    def test_boundary_rejects_malformed_and_unsupported_inputs(self):
        base = self.payload()
        bad = []
        for key, values in {
            'width': [True, 5.0, 0, -1, 33], 'height': [False, 0],
            'target_width': [0, 33, 2.5], 'target_height': [0, '3'],
            'channels': [True, 2, 4], 'device': ['auto', 'gpu'],
            'layout': ['HWC'], 'encoding': ['uint8'], 'operation': ['other'],
            'padding_color': [True, -.1, 1.01, float('inf')],
        }.items():
            for value in values:
                payload = copy.deepcopy(base); payload[key] = value; bad.append(payload)
        for value in (True, -.1, 1.01, .1, float('nan'), float('inf')):
            payload = copy.deepcopy(base); payload['pixels'][0] = value; bad.append(payload)
        for pixels in ([], base['pixels'][:-1], base['pixels']+[0]):
            payload = copy.deepcopy(base); payload['pixels'] = pixels; bad.append(payload)
        payload = copy.deepcopy(base); payload['expected'] = []; bad.append(payload)
        payload = copy.deepcopy(base); del payload['layout']; bad.append(payload)
        payload = copy.deepcopy(base); payload['target_width'] = 3; bad.append(payload)
        for payload in bad:
            with self.assertRaises(ContractError):
                validate_input(payload, OPERATION, 15)
        with self.assertRaises(ContractError): validate_input(base, OPERATION, 14)
        with self.assertRaises(ContractError): validate_input(base, OPERATION, True)

    def test_generator_reconstructs_original_inputs(self):
        path = ROOT/'Benchmarks/Fixtures/vision/generate.py'
        spec = importlib.util.spec_from_file_location('vision_generator', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for name, payload in module.cases():
            self.assertEqual(payload, read_json(ROOT/f'Benchmarks/Fixtures/vision/inputs/{name}.json'))


if __name__ == '__main__':
    unittest.main()
