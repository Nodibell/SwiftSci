"""Capture MLX scores on shared token prefixes without changing model code."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
from llama_score_analysis import validate_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--precision', choices=['bf16', 'float32'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    validate_request(request)
    if not mx.metal.is_available():
        raise RuntimeError('Metal is required; no CPU fallback')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', purpose='shared-prefix-score-probe',
                  precision=args.precision, request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                  device='gpu', numerical_certificate=False, cases=[],
                  versions={n: importlib.metadata.version(n) for n in ['mlx', 'mlx-metal', 'mlx-lm', 'numpy']})
    arrays = {}
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            model, tokenizer = load(str(args.model), tokenizer_config={'trust_remote_code': False})
            model.eval()
            if args.precision == 'float32':
                model.set_dtype(mx.float32)
            mx.eval(model.parameters())
            mx.synchronize()
            report['parameter_dtypes'] = sorted({str(v.dtype) for _, v in tree_flatten(model.parameters())})
            expected_dtype = 'mlx.core.' + ('bfloat16' if args.precision == 'bf16' else 'float32')
            if report['parameter_dtypes'] != [expected_dtype]:
                raise ValueError('Model parameter dtype differs from the requested precision')
            for case in request['cases']:
                rendered = tokenizer.apply_chat_template([dict(role='user', content=case['text'])],
                    tokenize=False, add_generation_prompt=True, date_string=request['template_date'])
                if rendered != case['rendered_prompt'] or tokenizer.encode(rendered, add_special_tokens=False) != case['prompt_tokens']:
                    raise ValueError('Probe prompt differs from the recorded input')
                raw_scores, normalized_scores = [], []

                def capture(tokens, logits):
                    raw_scores.append(logits[0])
                    return logits

                def teacher_force(logprobs):
                    index = len(normalized_scores)
                    normalized_scores.append(logprobs[0])
                    if index < len(case['forced_tokens']):
                        return mx.array([case['forced_tokens'][index]], dtype=mx.uint32)
                    return mx.argmax(logprobs, axis=-1)

                cache = make_prompt_cache(model)
                generated = []
                for response in stream_generate(model, tokenizer, case['prompt_tokens'],
                        max_tokens=len(case['forced_tokens']) + 1, prompt_cache=cache,
                        sampler=teacher_force, logits_processors=[capture]):
                    generated.append(response.token)
                mx.synchronize()
                if generated[:-1] != case['forced_tokens'] or len(generated) != len(case['forced_tokens']) + 1:
                    raise ValueError('Generation did not consume the exact forced prefix')
                cache_dtypes = sorted({str(t.dtype) for layer in cache for t in layer.state})
                if cache_dtypes != [expected_dtype]:
                    raise ValueError('Cache precision differs from model precision')
                row = dict(id=case['id'], cache_dtypes=cache_dtypes, generated_tokens=generated, probes=[])
                for position in case['positions']:
                    raw, native = raw_scores[position], normalized_scores[position]
                    if raw.shape != (128256,) or native.shape != raw.shape:
                        raise ValueError('Unexpected score shape')
                    raw_array = np.asarray(raw.astype(mx.float32))
                    native_array = np.asarray(native.astype(mx.float32))
                    rounded_array = np.asarray(raw.astype(mx.bfloat16).astype(mx.float32))
                    if not np.all(np.isfinite(raw_array)) or not np.all(np.isfinite(native_array)):
                        raise ValueError('Nonfinite MLX scores')
                    key = f"{case['id']}-{position}"
                    arrays[key + '-logits'] = raw_array
                    arrays[key + '-native-logprobs'] = native_array
                    row['probes'].append(dict(position=position, array_key=key,
                        logits_dtype=str(raw.dtype), native_logprobs_dtype=str(native.dtype),
                        raw_argmax=int(np.argmax(raw_array)), native_argmax=int(np.argmax(native_array)),
                        raw_top_tie_count=int(np.count_nonzero(raw_array == raw_array.max())),
                        rounded_bf16_argmax=int(np.argmax(rounded_array)),
                        rounded_bf16_top_tie_count=int(np.count_nonzero(rounded_array == rounded_array.max())),
                        top_tokens=[dict(id=int(i), text=tokenizer.decode([int(i)]), logit=float(raw_array[i]))
                                    for i in np.argsort(-raw_array, kind='stable')[:20]]))
                report['cases'].append(row)
        np.savez_compressed(args.output / 'scores.npz', **arrays)
        with (args.output / 'scores.npz').open('rb') as stream:
            report['scores_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'scores.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
