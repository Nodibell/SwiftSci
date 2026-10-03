"""Opt-in free-running generation and true batching on a pinned local Llama model."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from llama_precision import MODES, prepare_precision
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
from pretrained_llama import sha
from llama_generation_cases import build_cases, task_result


def cache_dtypes(cache, expected):
    observed = sorted({str(t.dtype) for layer in cache for t in layer.state})
    if observed != [expected]:
        raise ValueError('Unexpected generation cache dtype')
    return observed


def single(model, tokenizer, case, ids, suite, dtype):
    cache = make_prompt_cache(model)
    tokens = []
    start = time.perf_counter_ns()
    first = None
    for response in stream_generate(model, tokenizer, ids, max_tokens=suite['max_new_tokens'],
            sampler=make_sampler(temp=0), prompt_cache=cache, prefill_step_size=suite['prefill_step_size']):
        if first is None:
            first = time.perf_counter_ns() - start
        tokens.append(response.token)
    mx.synchronize()
    elapsed = time.perf_counter_ns() - start
    if not tokens or response.finish_reason not in ('stop', 'length'):
        raise ValueError('Incomplete single generation')
    visible = tokens[:-1] if tokens[-1] in tokenizer.eos_token_ids else tokens
    text = tokenizer.decode(visible)
    return dict(tokens=visible, finish_reason=response.finish_reason, text=text,
        elapsed_ns=elapsed, time_to_first_token_ns=first,
        library_prompt_tps=response.prompt_tps, library_generation_tps=response.generation_tps,
        library_prompt_tokens=response.prompt_tokens, library_generation_tokens=response.generation_tokens,
        peak_active_device_bytes=mx.get_peak_memory(), cache_dtypes=cache_dtypes(cache, dtype),
        task=task_result(case, text))


def batch(model, tokenizer, cases, prompts, suite, dtype):
    start = time.perf_counter_ns()
    gen = BatchGenerator(model, max_tokens=suite['max_new_tokens'],
        stop_tokens=[[t] for t in tokenizer.eos_token_ids], sampler=make_sampler(temp=0),
        prefill_batch_size=len(cases), completion_batch_size=len(cases), prefill_step_size=suite['prefill_step_size'])
    results = {}
    try:
        uids = gen.insert([prompts[c['id']]['tokens'] for c in cases], [suite['max_new_tokens']] * len(cases))
        outputs = {uid: dict(tokens=[]) for uid in uids}
        with gen.stats() as stats:
            while responses := gen.next_generated():
                for response in responses:
                    row = outputs[response.uid]
                    row.setdefault('time_to_first_token_ns', time.perf_counter_ns() - start)
                    if response.finish_reason != 'stop':
                        row['tokens'].append(response.token)
                    if response.finish_reason is not None:
                        row['finish_reason'] = response.finish_reason
                        row['cache_dtypes'] = cache_dtypes(response.prompt_cache, dtype)
        mx.synchronize()
        elapsed = time.perf_counter_ns() - start
        for uid, case in zip(uids, cases):
            row = outputs[uid]
            if row.get('finish_reason') not in ('stop', 'length'):
                raise ValueError('Incomplete batch generation')
            row['text'] = tokenizer.decode(row['tokens'])
            row['task'] = task_result(case, row['text'])
            results[case['id']] = row
        return dict(outputs=results, elapsed_ns=elapsed, library_stats=asdict(stats),
                    peak_active_device_bytes=mx.get_peak_memory())
    finally:
        gen.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'smoke', 'suite', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--precision', choices=MODES, required=True)
    args = parser.parse_args()
    smoke, suite = json.loads(args.smoke.read_text()), json.loads(args.suite.read_text())
    cases = build_cases(smoke, suite)
    if args.output.exists() or not mx.metal.is_available():
        raise ValueError('Fresh output and Metal are required')
    report = dict(schema_version=1, status='running', precision=args.precision,
        suite_sha256=sha(args.suite), smoke_sha256=sha(args.smoke), prompts={}, singles={}, batches={},
        versions={n: importlib.metadata.version(n) for n in ['mlx', 'mlx-metal', 'mlx-lm', 'transformers', 'tokenizers']})
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
            for case in cases:
                rendered = tokenizer.apply_chat_template([dict(role='user', content=case['text'])],
                    tokenize=False, add_generation_prompt=True, date_string=smoke['template_date'])
                ids = tokenizer.encode(rendered, add_special_tokens=False)
                if not 0 < len(ids) <= suite['max_prompt_tokens']:
                    raise ValueError('Prompt length outside generation contract')
                report['prompts'][case['id']] = dict(text=case['text'], rendered=rendered, tokens=ids)
                samples = []
                for repetition in range(suite['repetitions']):
                    mx.synchronize(); mx.reset_peak_memory()
                    sample = single(model, tokenizer, case, ids, suite, precision['expected_cache_dtype'])
                    sample['warmup'] = repetition == 0
                    samples.append(sample)
                report['singles'][case['id']] = samples
                print(args.precision, case['id'], len(ids), 'tokens completed', flush=True)
            by_id = {c['id']: c for c in cases}
            for group in suite['batches']:
                samples = []
                for repetition in range(suite['repetitions']):
                    mx.synchronize(); mx.reset_peak_memory()
                    sample = batch(model, tokenizer, [by_id[k] for k in group['cases']], report['prompts'],
                                   suite, precision['expected_cache_dtype'])
                    sample['warmup'] = repetition == 0
                    samples.append(sample)
                report['batches'][group['id']] = samples
                print(args.precision, group['id'], 'batch completed', flush=True)
            report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
