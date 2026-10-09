import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from llama_cpp_compare import require_conversion, token_agreement, visible_tokens
from pretrained_llama import MANIFEST, sha
from verify_llama_gguf import mapped_name, restore_hf_rows, rope_factors


class LlamaComparisonTests(unittest.TestCase):
    def test_eos_normalization_preserves_all_content_tokens(self):
        self.assertEqual(visible_tokens([7, 8, 128009], [128009]), [7, 8])
        self.assertEqual(visible_tokens([7, 128009, 8], [128009]), [7, 128009, 8])
        self.assertEqual(visible_tokens([], [128009]), [])
        self.assertEqual(token_agreement([1, 2, 3], [1, 4, 3])['common_prefix_tokens'], 1)
        self.assertFalse(token_agreement([1], [1, 2])['equal'])

    def test_conversion_evidence_cannot_be_reused_for_another_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'model.gguf'
            artifact.write_bytes(b'fixture')
            report = dict(status='passed', purpose='pinned-llama-conversion-parity',
                checkpoint_manifest_sha256=sha(MANIFEST),
                verifier_sha256=sha(ROOT / 'Benchmarks/Tools/verify_llama_gguf.py'),
                gguf_sha256=sha(artifact), gguf_bytes=7,
                tensor_count=146, tensors=[dict(exact=True)] * 146)
            require_conversion(report, artifact)
            for change in [dict(status='failed'), dict(checkpoint_manifest_sha256='changed'),
                           dict(verifier_sha256='changed'), dict(tensor_count=145),
                           dict(tensors=[dict(exact=False)] * 146)]:
                with self.assertRaises(ValueError):
                    require_conversion({**report, **change}, artifact)
            artifact.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                require_conversion(report, artifact)

    def test_conversion_layout_and_names(self):
        import numpy as np
        original = np.arange(32).reshape(8, 4)
        # Two heads, each with split-half rotary rows. GGUF stores pairs together.
        interleaved = original[[0, 2, 1, 3, 4, 6, 5, 7], :]
        np.testing.assert_array_equal(restore_hf_rows(interleaved, 2), original)
        self.assertEqual(mapped_name('model.layers.15.self_attn.k_proj.weight'), 'blk.15.attn_k.weight')
        self.assertEqual(mapped_name('model.embed_tokens.weight'), 'token_embd.weight')
        for name in ['model.layers.0.unknown.weight', 'modelXlayersX0Xself_attn.k_proj.weight', 'lm_head.weight']:
            with self.assertRaises(ValueError):
                mapped_name(name)

    def test_rope_frequency_ranges(self):
        result = rope_factors(dict(hidden_size=2048, num_attention_heads=32, rope_theta=500000,
            rope_scaling=dict(factor=32, low_freq_factor=1, high_freq_factor=4,
                              original_max_position_embeddings=8192)))
        self.assertEqual(len(result), 32)
        self.assertEqual(result[0], 1)
        self.assertEqual(result[-1], 32)
        self.assertTrue(all(1 <= x <= 32 for x in result))
        self.assertTrue(any(1 < x < 32 for x in result))

    def test_template_date_is_an_explicit_fixture_input(self):
        prompts = json.loads((ROOT / 'Benchmarks/Fixtures/pretrained/prompts.json').read_text())
        self.assertEqual(prompts['template_date'], '29 Sep 2026')
