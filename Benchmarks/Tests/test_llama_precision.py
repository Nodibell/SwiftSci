import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
sys.path.insert(0, str(ROOT / 'Benchmarks/Python'))
from llama_precision_benchmark import summarize


class PrecisionSummaryTests(unittest.TestCase):
    def runs(self):
        runs = []
        for mode, elapsed in [('bf16', 10), ('head-float32', 15), ('shared-float32', 15), ('float32', 20)]:
            for block in range(2):
                samples = [dict(warmup=True, elapsed_ns=1000, tokens=[1, 2], peak_active_device_bytes=100)]
                samples += [dict(warmup=False, elapsed_ns=elapsed + block, tokens=[1, 2], peak_active_device_bytes=100)] * 5
                runs.append(dict(precision=mode, retained_parameter_bytes=20, additional_output_weight_bytes=0,
                                 cases=[dict(id='fixture', samples=samples)]))
        return runs

    def test_balanced_blocks_exclude_warmup(self):
        result = summarize(self.runs())
        self.assertEqual(result['bf16']['cases'][0]['median_elapsed_ns'], 10.5)
        self.assertEqual(result['bf16']['cases'][0]['block_medians_ns'], [10, 11])
        self.assertAlmostEqual(result['head-float32']['cases'][0]['elapsed_ratio_to_bf16'], 15.5/10.5)

    def test_incomplete_or_changed_trajectories_fail(self):
        runs = self.runs()
        with self.assertRaises(ValueError):
            summarize(runs[:-1])
        changed = copy.deepcopy(runs)
        changed[0]['cases'][0]['samples'][1] = dict(warmup=False, elapsed_ns=10, tokens=[1, 3], peak_active_device_bytes=100)
        with self.assertRaises(ValueError):
            summarize(changed)


@unittest.skipUnless(importlib.util.find_spec('mlx') and os.environ.get('SWIFTSCI_MLX_TEST_GPU') == '1',
                     'Explicit optional Metal test invocation required')
class MixedPrecisionAdapterTests(unittest.TestCase):
    def test_wider_projection_preserves_close_score_without_changing_body_weights(self):
        from types import SimpleNamespace
        import mlx.core as mx
        import mlx.nn as nn
        from llama_precision import prepare_precision

        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(3, 4)
                self.embed_tokens.weight = mx.array([[1, 0, 0, 0], [1, 1/128, 0, 0], [0, 0, 1, 0]], dtype=mx.bfloat16)

            def __call__(self, inputs, cache=None, input_embeddings=None):
                return self.embed_tokens(inputs) if input_embeddings is None else input_embeddings

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_type = 'llama'
                self.args = SimpleNamespace(tie_word_embeddings=True)
                self.model = Body()
                self.layers = []

            def __call__(self, inputs, cache=None, input_embeddings=None):
                return self.model.embed_tokens.as_linear(self.model(inputs, cache, input_embeddings))

        if not mx.metal.is_available():
            self.skipTest('Metal unavailable')
        with mx.stream(mx.new_stream(mx.gpu)):
            base = Model()
            before = base.model.embed_tokens.weight.tolist()
            hidden = mx.array([[[20, 5, 0, 0]]], dtype=mx.bfloat16)
            tokens = mx.array([[0]])
            native = base(tokens, input_embeddings=hidden)
            mixed, metadata = prepare_precision(base, 'head-float32')
            precise = mixed(tokens, input_embeddings=hidden)
            mx.eval(native, precise)
            self.assertEqual(native.tolist()[0][0][:2], [20., 20.])
            self.assertEqual(precise.tolist()[0][0][:2], [20., 20.0390625])
            self.assertEqual(int(mx.argmax(native).item()), 0)
            self.assertEqual(int(mx.argmax(precise).item()), 1)
            self.assertEqual(base.model.embed_tokens.weight.tolist(), before)
            self.assertEqual(base.model.embed_tokens.weight.dtype, mx.bfloat16)
            self.assertEqual(metadata['expected_cache_dtype'], 'mlx.core.bfloat16')
            self.assertEqual(metadata['additional_output_weight_bytes'], 3 * 4 * 4)
            base.args.tie_word_embeddings = False
            with self.assertRaises(ValueError):
                prepare_precision(base, 'head-float32')

    def test_shared_storage_preserves_inputs_and_projection_without_duplicate_matrix(self):
        from types import SimpleNamespace
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from llama_precision import prepare_precision

        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(3, 4)
                self.embed_tokens.weight = mx.array([[1, 0, 0, 0], [1, 1/128, 0, 0], [0, 0, 1, 0]], dtype=mx.bfloat16)
                self.scale = mx.ones((4,), dtype=mx.bfloat16)

            def __call__(self, inputs, cache=None, input_embeddings=None):
                hidden = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
                return hidden * self.scale

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model_type = 'llama'
                self.args = SimpleNamespace(tie_word_embeddings=True)
                self.model = Body()
                self.layers = []

        if not mx.metal.is_available():
            self.skipTest('Metal unavailable')
        with mx.stream(mx.new_stream(mx.gpu)):
            base = Model()
            tokens = mx.array([[2, 0, 1, 1]])
            before = base.model.embed_tokens(tokens).tolist()
            separate, separate_meta = prepare_precision(Model(), 'head-float32')
            shared, metadata = prepare_precision(base, 'shared-float32')
            gathered = base.model.embed_tokens(tokens)
            self.assertEqual(gathered.dtype, mx.bfloat16)
            self.assertEqual(gathered.tolist(), before)
            self.assertEqual(base.model.scale.dtype, mx.bfloat16)
            self.assertEqual(base.model.scale.tolist(), [1, 1, 1, 1])
            self.assertEqual(metadata['body_parameter_dtypes'], ['mlx.core.bfloat16'])
            self.assertEqual(metadata['expected_cache_dtype'], 'mlx.core.bfloat16')
            self.assertEqual(metadata['retained_parameter_bytes'], 3 * 4 * 4 + 4 * 2)
            self.assertEqual(separate_meta['retained_parameter_bytes'] - metadata['retained_parameter_bytes'], 3 * 4 * 2)
            self.assertEqual(metadata['additional_output_weight_bytes'], 0)
            matrices = [v for _, v in tree_flatten(shared.parameters()) if v.ndim == 2]
            self.assertEqual(len(matrices), 1)
            self.assertIs(matrices[0], base.model.embed_tokens.weight)
            hidden = mx.array([[[20, 5, 0, 0]]], dtype=mx.bfloat16)
            precise = shared(tokens, input_embeddings=hidden)
            self.assertEqual(precise.dtype, mx.float32)
            self.assertEqual(precise.tolist()[0][0][:2], [20, 20.0390625])
            self.assertEqual(shared(tokens).tolist(), separate(tokens).tolist())
            base.args.tie_word_embeddings = False
            with self.assertRaises(ValueError):
                prepare_precision(base, 'shared-float32')
