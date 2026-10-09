#!/usr/bin/env python3
"""Verify this pinned Llama BF16 conversion, independently of converter code."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import struct

from pretrained_llama import MANIFEST, sha, verify_checkpoint


def mapped_name(name):
    direct = {'model.embed_tokens.weight': 'token_embd.weight', 'model.norm.weight': 'output_norm.weight'}
    if name in direct:
        return direct[name]
    match = re.fullmatch(r'model\.layers\.(\d+)\.(.+)\.weight', name)
    suffixes = {'input_layernorm': 'attn_norm', 'post_attention_layernorm': 'ffn_norm',
                'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
                'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
                'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up', 'mlp.down_proj': 'ffn_down'}
    if match is None or match[2] not in suffixes:
        raise ValueError('Unexpected source tensor: ' + name)
    return f'blk.{match[1]}.{suffixes[match[2]]}.weight'


def restore_hf_rows(array, heads):
    """Reverse GGUF interleaved RoPE rows into HF split-half rows."""
    return array.reshape(heads, -1, 2, array.shape[1]).swapaxes(1, 2).reshape(array.shape)


def rope_factors(config):
    scaling = config['rope_scaling']
    dim = config['hidden_size'] // config['num_attention_heads']
    result = []
    for i in range(0, dim, 2):
        wavelength = 2 * math.pi * config['rope_theta'] ** (i / dim)
        high = scaling['original_max_position_embeddings'] / scaling['high_freq_factor']
        low = scaling['original_max_position_embeddings'] / scaling['low_freq_factor']
        if wavelength < high:
            value = 1.0
        elif wavelength > low:
            value = scaling['factor']
        else:
            blend = (scaling['original_max_position_embeddings'] / wavelength - scaling['low_freq_factor']) / (scaling['high_freq_factor'] - scaling['low_freq_factor'])
            value = 1 / (blend + (1 - blend) / scaling['factor'])
        result.append(value)
    return result


def verify(model, gguf_path):
    import numpy as np
    import gguf
    manifest = json.loads(MANIFEST.read_text())
    verify_checkpoint(model, manifest)
    before = sha(gguf_path)
    config = json.loads((model / 'config.json').read_text())
    reader = gguf.GGUFReader(gguf_path)
    tensors = {t.name: t for t in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError('Duplicate GGUF tensor names')
    with (model / 'model.safetensors').open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    header.pop('__metadata__', None)
    expected_names = {mapped_name(n) for n in header} | {'rope_freqs.weight'}
    if set(tensors) != expected_names:
        raise ValueError('GGUF tensor inventory differs from the tied-embedding checkpoint')
    results = []
    for name, spec in header.items():
        if spec['dtype'] != 'BF16':
            raise ValueError('Expected BF16 source: ' + name)
        original = np.memmap(model / 'model.safetensors', mode='r', dtype='<u2',
                             offset=8 + header_size + spec['data_offsets'][0], shape=tuple(spec['shape']))
        tensor = tensors[mapped_name(name)]
        if tuple(reversed(tensor.shape)) != original.shape:
            raise ValueError('Shape mismatch: ' + name)
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            actual = tensor.data.view('<u2').reshape(original.shape)
            expected = original
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F32 and original.ndim == 1:
            actual = tensor.data
            expected = (original.astype('<u4') << 16).view('<f4')
        else:
            raise ValueError('Unexpected GGUF tensor dtype: ' + name)
        if name.endswith('q_proj.weight'):
            actual = restore_hf_rows(actual, config['num_attention_heads'])
        elif name.endswith('k_proj.weight'):
            actual = restore_hf_rows(actual, config['num_key_value_heads'])
        if not np.array_equal(actual, expected):
            raise ValueError('Weight values differ: ' + name)
        results.append(dict(source=name, gguf=tensor.name, shape=spec['shape'], exact=True,
                            stored_type=tensor.tensor_type.name))
    rope = tensors['rope_freqs.weight']
    if rope.tensor_type != gguf.GGMLQuantizationType.F32:
        raise ValueError('Unexpected rotary table dtype')
    expected_rope = np.asarray(rope_factors(config))
    if rope.data.shape != expected_rope.shape or not np.allclose(rope.data, expected_rope, rtol=1e-6, atol=0):
        raise ValueError('Rotary frequency factors differ')
    checks = {'general.architecture': 'llama', 'llama.block_count': config['num_hidden_layers'],
              'llama.context_length': config['max_position_embeddings'],
              'llama.embedding_length': config['hidden_size'],
              'llama.feed_forward_length': config['intermediate_size'],
              'llama.attention.head_count': config['num_attention_heads'],
              'llama.attention.head_count_kv': config['num_key_value_heads'],
              'llama.rope.freq_base': config['rope_theta'], 'llama.vocab_size': config['vocab_size'],
              'llama.rope.dimension_count': config['hidden_size'] // config['num_attention_heads'],
              'tokenizer.ggml.bos_token_id': config['bos_token_id'],
              'tokenizer.chat_template': json.loads((model / 'tokenizer_config.json').read_text())['chat_template']}
    for name, expected in checks.items():
        if name not in reader.fields or reader.fields[name].contents() != expected:
            raise ValueError('GGUF metadata mismatch: ' + name)
    epsilon = reader.fields['llama.attention.layer_norm_rms_epsilon'].contents()
    if not math.isclose(epsilon, config['rms_norm_eps'], rel_tol=1e-6):
        raise ValueError('RMS normalization epsilon differs')
    if reader.fields['tokenizer.ggml.eos_token_id'].contents() not in config['eos_token_id']:
        raise ValueError('Unexpected EOS token')
    verify_checkpoint(model, manifest)
    if sha(gguf_path) != before:
        raise ValueError('GGUF changed during verification')
    return dict(schema_version=1, purpose='pinned-llama-conversion-parity', status='passed',
                checkpoint_manifest_sha256=sha(MANIFEST), gguf_sha256=before, gguf_bytes=gguf_path.stat().st_size,
                tensor_count=len(results), tensors=results, metadata_checked=list(checks) + ['llama.attention.layer_norm_rms_epsilon', 'tokenizer.ggml.eos_token_id'],
                rope_max_relative_error=float(np.max(np.abs(rope.data - expected_rope) / expected_rope)),
                rope_rtol=1e-6, inference_numerical_conformance='not-tested',
                verifier_sha256=sha(Path(__file__)),
                versions={n: importlib.metadata.version(n) for n in ['numpy', 'gguf']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a fresh output path')
    try:
        report = verify(args.model, args.gguf)
    except Exception as error:
        args.output.write_text(json.dumps(dict(status='failed', error=f'{type(error).__name__}: {error}'), indent=2) + '\n')
        raise
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Verified {report['tensor_count']} checkpoint tensors exactly; rotary factors and metadata passed.")


if __name__ == '__main__':
    main()
