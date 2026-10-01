#!/usr/bin/env python3
"""Exercise the checkpoint preflight with tiny synthetic tensors, including invalid inputs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess


def tokenizer():
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    vocab, next_scalar = {}, 256
    for byte in range(256):
        scalar = byte if byte in visible else next_scalar
        if byte not in visible:
            next_scalar += 1
        vocab[chr(scalar)] = byte
    bos = '<|begin_of_text|>'
    special = [dict(id=i, content=text, special=True, single_word=False,
                    lstrip=False, rstrip=False, normalized=False)
               for i, text in [(1000, bos), (1001, '<|eot_id|>')]]
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    token = lambda kind, identifier, type_id: {kind: dict(id=identifier, type_id=type_id)}
    return dict(
        version='1.0', normalizer=None, truncation=None, padding=None, added_tokens=special,
        model=dict(type='BPE', vocab=vocab, merges=[], ignore_merges=True, dropout=None,
                   unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None,
                   fuse_unk=False, byte_fallback=False),
        pre_tokenizer=dict(type='Sequence', pretokenizers=[
            dict(type='Split', pattern=dict(Regex=pattern), behavior='Isolated', invert=False),
            dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=False)]),
        decoder=dict(type='ByteLevel', add_prefix_space=True, trim_offsets=True, use_regex=True),
        post_processor=dict(type='Sequence', processors=[
            dict(type='ByteLevel', add_prefix_space=True, trim_offsets=False, use_regex=True),
            dict(type='TemplateProcessing',
                 single=[token('SpecialToken', bos, 0), token('Sequence', 'A', 0)],
                 pair=[token('SpecialToken', bos, 0), token('Sequence', 'A', 0),
                       token('SpecialToken', bos, 1), token('Sequence', 'B', 1)],
                 special_tokens={bos: dict(id=bos, ids=[1000], tokens=[bos])})]))


def fixture():
    config = dict(model_type='llama', hidden_act='silu', attention_bias=False, mlp_bias=False, head_dim=4, vocab_size=1002, hidden_size=8,
                  num_attention_heads=2, num_key_value_heads=1, num_hidden_layers=1,
                  intermediate_size=12, max_position_embeddings=131072,
                  rms_norm_eps=1e-5, rope_theta=500000, tie_word_embeddings=True,
                  rope_scaling=dict(rope_type='llama3', factor=32, low_freq_factor=1,
                                    high_freq_factor=4, original_max_position_embeddings=8192))
    shapes = {'model.embed_tokens.weight': [1002, 8], 'model.norm.weight': [8]}
    for key, shape in {
        'self_attn.q_proj.weight': [8, 8], 'self_attn.k_proj.weight': [4, 8],
        'self_attn.v_proj.weight': [4, 8], 'self_attn.o_proj.weight': [8, 8],
        'mlp.gate_proj.weight': [12, 8], 'mlp.up_proj.weight': [12, 8],
        'mlp.down_proj.weight': [8, 12], 'input_layernorm.weight': [8],
        'post_attention_layernorm.weight': [8]
    }.items():
        shapes['model.layers.0.' + key] = shape
    texts = ['abc', ' café 世界\t\n', '<|begin_of_text|>abc<|eot_id|>']
    cases = [dict(id=str(i), text=text, raw_tokens=list(text.encode('utf-8')))
             for i, text in enumerate(texts)]
    cases[-1]['raw_tokens'] = [1000, 97, 98, 99, 1001]
    return config, shapes, tokenizer(), cases


def write_tensors(path, shapes):
    header, data = {}, bytearray()
    for key, shape in sorted(shapes.items()):
        start = len(data)
        data.extend(bytes(4 * math.prod(shape)))
        header[key] = dict(dtype='F32', shape=shape, data_offsets=[start, len(data)])
    encoded = json.dumps(header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swift-worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Run only named controls; repeat to select more than one')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mutations = [('valid-tied', None, None, 'preflight-passed'),
                 ('valid-untied', 'untied', None, 'preflight-passed')]
    for key in fixture()[1]:
        mutations += [('missing-' + key, 'missing', key, 'missing-tensor'),
                      ('shape-' + key, 'shape', key, 'tensor-shape')]
    mutations += [('extra-tensor', 'extra', None, 'unexpected-tensor'),
                  ('wrong-token-oracle', 'tokens', None, 'tokenizer-parity'),
                  ('unsupported-tokenizer', 'tokenizer', None, 'failed')]
    for key in ['model_type', 'vocab_size', 'num_hidden_layers', 'intermediate_size',
                'num_attention_heads', 'num_key_value_heads', 'hidden_size',
                'hidden_act', 'attention_bias', 'mlp_bias', 'head_dim',
                'rms_norm_eps', 'rope_theta', 'tie_word_embeddings', 'rope_scaling']:
        mutations.append(('missing-config-' + key, 'config-missing', key, 'failed'))
    for key, value in [('model_type', 'other'), ('vocab_size', 0),
                       ('num_hidden_layers', 0), ('intermediate_size', 0),
                       ('hidden_act', 'gelu'), ('attention_bias', True),
                       ('mlp_bias', True), ('head_dim', 8), ('num_attention_heads', 3), ('num_key_value_heads', 3),
                       ('hidden_size', 6), ('rms_norm_eps', 0), ('rope_theta', -1)]:
        mutations.append(('invalid-' + key, 'config-value', (key, value), 'failed'))
    for key, value in [('factor', 0), ('low_freq_factor', 0), ('high_freq_factor', 1),
                       ('original_max_position_embeddings', 0), ('rope_type', 'unknown')]:
        mutations.append(('invalid-scaling-' + key, 'scaling', (key, value), 'failed'))
    if args.case:
        unknown = set(args.case) - {row[0] for row in mutations}
        if unknown:
            parser.error('Unknown controls: ' + ', '.join(sorted(unknown)))
        mutations = [row for row in mutations if row[0] in args.case]
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    report = dict(purpose='pretrained-preflight-regression', generation_executed=False,
                  worker=str(args.swift_worker.resolve()), worker_sha256=sha(args.swift_worker),
                  checker_sha256=sha(Path(__file__)), status='running', cases=[])
    for name, mutation, value, expected in mutations:
        directory = args.output / name
        directory.mkdir()
        config, shapes, tok, cases = copy.deepcopy(fixture())
        if mutation == 'untied':
            config['tie_word_embeddings'] = False
            shapes['lm_head.weight'] = [1002, 8]
        elif mutation == 'missing': del shapes[value]
        elif mutation == 'shape': shapes[value][0] += 1
        elif mutation == 'extra': shapes['unexpected.weight'] = [1]
        elif mutation == 'tokens': cases[0]['raw_tokens'] = [0]
        elif mutation == 'tokenizer': tok['normalizer'] = dict(type='Lowercase')
        elif mutation == 'config-missing': del config[value]
        elif mutation == 'config-value': config[value[0]] = value[1]
        elif mutation == 'scaling': config['rope_scaling'][value[0]] = value[1]
        (directory / 'config.json').write_text(json.dumps(config))
        (directory / 'tokenizer.json').write_text(json.dumps(tok))
        write_tensors(directory / 'model.safetensors', shapes)
        request, response = directory / 'request.json', directory / 'response.json'
        request.write_text(json.dumps(dict(model_path=str(directory.resolve()), cases=cases)))
        with (directory / 'worker.log').open('w') as log:
            completed = subprocess.run([str(args.swift_worker.resolve()), '--llama-inspect',
                                        str(request.resolve()), str(response.resolve())],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=120)
        result = json.loads(response.read_text()) if response.exists() else {}
        if expected == 'preflight-passed':
            passed = (completed.returncode == 0 and result.get('status') == expected
                      and result.get('weight_loading_verified') is True
                      and result.get('generation_executed') is False
                      and result.get('inspection_device') == 'cpu'
                      and result.get('expected_tensor_count') == len(shapes)
                      and result.get('findings') == []
                      and len(result.get('tokenizer_cases', [])) == len(cases)
                      and all(x.get('equal') is True and x.get('decoded_equal') is True
                              for x in result.get('tokenizer_cases', [])))
        elif expected == 'failed':
            passed = completed.returncode == 2 and result.get('status') == 'failed' and bool(result.get('error'))
        else:
            passed = (completed.returncode == 1 and result.get('status') == 'blocked'
                      and result.get('weight_loading_verified') is False
                      and result.get('generation_executed') is False
                      and result.get('inspection_device') == 'cpu'
                      and any(x.get('id') == expected for x in result.get('findings', [])))
        report['cases'].append(dict(name=name, expected=expected, passed=bool(passed),
                                    exit_code=completed.returncode, status=result.get('status')))
        (args.output / 'controls.json').write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'passed' if all(x['passed'] for x in report['cases']) else 'failed'
    (args.output / 'controls.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], cases=len(report['cases']),
                          failures=[x['name'] for x in report['cases'] if not x['passed']])))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
