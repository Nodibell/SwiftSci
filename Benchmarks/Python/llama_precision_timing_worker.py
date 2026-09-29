"""Measure a fixed teacher-forced Llama trajectory under one precision policy."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from llama_precision import MODES, prepare_precision

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
from llama_score_analysis import validate_request


def execute(model, tokenizer, case, expected_cache_dtype):
    cache = make_prompt_cache(model)
    index = 0

    def choose(logprobs):
        nonlocal index
        position = index
        index += 1
        if position < len(case['forced_tokens']):
            return mx.array([case['forced_tokens'][position]], dtype=mx.uint32)
        return mx.argmax(logprobs, axis=-1)

    tokens = []
    for response in stream_generate(model, tokenizer, case['prompt_tokens'],
            max_tokens=len(case['forced_tokens']) + 1, prompt_cache=cache, sampler=choose):
        tokens.append(response.token)
    mx.synchronize()
    if tokens[:-1] != case['forced_tokens'] or len(tokens) != len(case['forced_tokens']) + 1:
        raise ValueError('Timing run changed the forced trajectory')
    if {str(t.dtype) for layer in cache for t in layer.state} != {expected_cache_dtype}:
        raise ValueError('Timing cache precision differs from the experiment contract')
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--precision', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse-cases', action='store_true')
    args = parser.parse_args()
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    validate_request(request)
    if args.output.exists():
        raise ValueError('Choose a fresh timing output')
    if not mx.metal.is_available():
        raise RuntimeError('Metal required; no CPU fallback')
    report = dict(schema_version=1, status='running', purpose='fixed-trajectory-precision-timing',
                  precision=args.precision, warmups=2, measurements=5, device='gpu', cases=[],
                  request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                  versions={n: importlib.metadata.version(n) for n in ['mlx', 'mlx-metal', 'mlx-lm']})
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            start = time.perf_counter_ns()
            model, tokenizer = load(str(args.model), tokenizer_config={'trust_remote_code': False})
            model.eval()
            model, precision = prepare_precision(model, args.precision)
            report.update(precision)
            report['load_and_prepare_ns'] = time.perf_counter_ns() - start
            mx.clear_cache()
            report['active_device_bytes_after_prepare'] = mx.get_active_memory()
            cases = list(reversed(request['cases'])) if args.reverse_cases else request['cases']
            for case in cases:
                rendered = tokenizer.apply_chat_template([dict(role='user', content=case['text'])],
                    tokenize=False, add_generation_prompt=True, date_string=request['template_date'])
                if tokenizer.encode(rendered, add_special_tokens=False) != case['prompt_tokens']:
                    raise ValueError('Timing prompt differs')
                row = dict(id=case['id'], prompt_tokens=len(case['prompt_tokens']),
                           output_steps=len(case['forced_tokens']) + 1, samples=[])
                for repetition in range(7):
                    mx.synchronize()
                    mx.reset_peak_memory()
                    start = time.perf_counter_ns()
                    tokens = execute(model, tokenizer, case, precision['expected_cache_dtype'])
                    elapsed = time.perf_counter_ns() - start
                    row['samples'].append(dict(warmup=repetition < 2, elapsed_ns=elapsed,
                        tokens=tokens, peak_active_device_bytes=mx.get_peak_memory(),
                        allocator_cache_bytes_after=mx.get_cache_memory()))
                if not all(s['tokens'] == row['samples'][0]['tokens'] for s in row['samples']):
                    raise ValueError('Timing replay changed token decisions')
                report['cases'].append(row)
            report['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
