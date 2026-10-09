import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'Benchmarks/Python'), str(ROOT / 'Benchmarks/Tools')]
from contracts import digest, read_json, validate_values
from neural_reference import reference as neural_reference
from boundary_reference import reference as boundary_reference
from boundary_sweep import reference as sweep_reference
from numerical_fixtures import validate_input
from mlx_workloads import MLXWorkload, check_supported


@unittest.skipUnless(importlib.util.find_spec('mlx'), 'Install requirements-mlx.txt for native MLX comparisons')
class MLXWorkersTests(unittest.TestCase):
    def payload(self, family, name):
        return read_json(ROOT / f'Benchmarks/Fixtures/{family}/inputs/{name}.json')

    def test_cpu_decoders_match_independent_logits(self):
        profile = read_json(ROOT / 'Benchmarks/Specs/profiles/neural-cpu-conformance.json')
        for case in profile['cases']:
            with self.subTest(case=case['id']):
                payload = self.payload('neural', case['dataset'])
                validate_values(MLXWorkload(payload).execute(), neural_reference(payload), 2e-5, 2e-5)

    def test_cache_resets_and_causal_prefix_is_preserved(self):
        for position in ('learned', 'rope'):
            payload = self.payload('neural', f'neural-cpu-{position}-cached')
            worker = MLXWorkload(payload)
            first = worker.execute()
            validate_values(worker.execute(), first, 0, 0)
            self.assertEqual(first[-4:], [3, 2, 3, 4])
            full = copy.deepcopy(payload)
            full['execution'] = 'full'
            validate_values(first[:-4], MLXWorkload(full).execute(), 2e-5, 2e-5)
            changed = copy.deepcopy(full)
            for row in changed['tokens']:
                row[-1] = (row[-1] + 1) % 7
            modified = MLXWorkload(changed).execute()
            for batch in range(len(full['tokens'])):
                start = 3 + batch * 4 * 7
                validate_values(modified[start:start+21], first[start:start+21], 2e-5, 2e-5)
            self.assertNotEqual(modified, first[:-4])

    def test_cpu_boundary_precision_alignment_and_isolation(self):
        profile = read_json(ROOT / 'Benchmarks/Specs/profiles/boundary-cpu-conformance.json')
        for case in profile['cases']:
            with self.subTest(case=case['id']):
                payload = self.payload('boundary', case['dataset'])
                tolerance = 1e-12 if payload['dtype'] == 'float64' else 2e-5
                output = MLXWorkload(payload).execute()
                validate_values(output, boundary_reference(payload), tolerance, 0)
                self.assertEqual(output[-1], 1)

    def test_isolation_detects_altered_target_tensor(self):
        import mlx.core as mx
        import mlx_boundary_workloads as boundary
        payload = self.payload('boundary', 'boundary-cpu-float64-narrow')
        original = boundary.convert

        def altered_target(data, source):
            prepared = list(original(data, source))
            target = prepared[2].copy()
            target[0] += 19
            prepared[4] = mx.array(target, dtype=mx.float64)
            return tuple(prepared)

        with patch.object(boundary, 'convert', side_effect=altered_target):
            self.assertEqual(MLXWorkload(payload).execute()[-1], 0)

    def test_stage_boundaries_have_complete_outputs(self):
        for dtype in ('float32', 'float64'):
            for stage in ('conversion', 'prepared', 'pipeline'):
                with self.subTest(dtype=dtype, stage=stage):
                    payload = dict(operation='dataframe-model-sweep', recipe='dyadic-v1',
                                   device='cpu', dtype=dtype, stage=stage, rows=128, columns=8)
                    worker = MLXWorkload(payload)
                    gold = sweep_reference(payload)
                    for _ in range(2):
                        validate_values(worker.execute(), gold, 2e-5 if dtype == 'float32' else 1e-12, 0)

    def test_no_device_fallback_or_loader_substitution(self):
        import mlx.core as mx
        payload = self.payload('neural', 'neural-cpu-learned-full')
        payload['loading'] = 'public-loader'
        with self.assertRaisesRegex(ValueError, 'no equivalent'):
            check_supported(payload)
        payload['loading'] = 'direct'
        payload['device'] = 'gpu'
        with patch.object(mx.metal, 'is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'no CPU fallback'):
                MLXWorkload(payload)
        payload['device'] = 'auto'
        with self.assertRaises(ValueError):
            validate_input(payload, payload['operation'], 8)

    def test_worker_rejects_corrupted_answers_and_inputs(self):
        payload = self.payload('neural', 'neural-cpu-rope-cached')
        gold = neural_reference(payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for corruption in ('none', 'answer', 'cache-count', 'input'):
                with self.subTest(corruption=corruption):
                    data = copy.deepcopy(payload)
                    expected = list(gold)
                    if corruption == 'answer': expected[3] += 1
                    if corruption == 'cache-count': expected[-1] += 1
                    if corruption == 'input': data['tokens'][0][0] = 999
                    input_path, expected_path = root/'input.json', root/'expected.f64'
                    input_path.write_text(json.dumps(data))
                    expected_path.write_bytes(np.asarray(expected, dtype='<f8').tobytes())
                    request = dict(schema_version=1, case_key='a'*64, operation='decoder-fixed-f32',
                                   dataset_kind='numerical-fixture-v1', input_path=str(input_path),
                                   input_sha256=digest(input_path.read_bytes()), input_bytes=input_path.stat().st_size,
                                   expected_path=str(expected_path), expected_sha256=digest(expected_path.read_bytes()),
                                   rows=8, warmups=0, samples=1, atol=2e-5, rtol=2e-5)
                    req, dest = root/'request.json', root/'response.json'
                    req.write_text(json.dumps(request))
                    run = subprocess.run([sys.executable, str(ROOT/'Benchmarks/Python/mlx_worker.py'), str(req), str(dest)], capture_output=True, text=True)
                    response = read_json(dest)
                    self.assertEqual(run.returncode, 0 if corruption == 'none' else 1, run.stderr)
                    self.assertEqual(response['status'], 'passed' if corruption == 'none' else 'failed')
                    if corruption != 'none': self.assertEqual(response['samples'], [])

    @unittest.skipUnless(os.environ.get('SWIFTSCI_MLX_TEST_GPU') == '1', 'Explicit Metal test invocation required')
    def test_explicit_metal_decoder_and_boundary(self):
        for family, profile in [('neural', 'neural-conformance'), ('boundary', 'boundary-conformance')]:
            for case in read_json(ROOT/f'Benchmarks/Specs/profiles/{profile}.json')['cases']:
                payload = self.payload(family, case['dataset'])
                if payload['device'] != 'gpu': continue
                with self.subTest(case=case['id']):
                    gold = neural_reference(payload) if family == 'neural' else boundary_reference(payload)
                    validate_values(MLXWorkload(payload).execute(), gold, 2e-5, 2e-5 if family == 'neural' else 0)
