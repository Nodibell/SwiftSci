#!/usr/bin/env python3
"""Compare pinned MLX-LM and llama.cpp integration behavior, not numerical certification."""
import argparse
import json
import os
import platform
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request

from pretrained_llama import MANIFEST, PROMPTS, ROOT, sha, verify_checkpoint


def visible_tokens(tokens, eos_ids):
    tokens = list(tokens)
    if tokens and tokens[-1] in eos_ids:
        tokens.pop()
    return tokens


def token_agreement(left, right):
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return dict(equal=left == right, common_prefix_tokens=prefix,
                left_count=len(left), right_count=len(right))


def require_conversion(report, gguf_path):
    if report.get('status') != 'passed' or report.get('purpose') != 'pinned-llama-conversion-parity':
        raise ValueError('A passing conversion report is required')
    if report.get('checkpoint_manifest_sha256') != sha(MANIFEST):
        raise ValueError('Conversion checkpoint identity differs')
    if report.get('verifier_sha256') != sha(Path(__file__).with_name('verify_llama_gguf.py')):
        raise ValueError('Conversion verifier changed; verify again')
    if report.get('gguf_sha256') != sha(gguf_path) or report.get('gguf_bytes') != gguf_path.stat().st_size:
        raise ValueError('GGUF differs from the verified artifact')
    if report.get('tensor_count') != 146 or len(report.get('tensors', [])) != 146 or not all(t.get('exact') is True for t in report['tensors']):
        raise ValueError('Incomplete weight verification')


def main():
    from runner import source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--conversion-report', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True, help='Pinned MLX-LM environment')
    parser.add_argument('--server', type=Path, required=True, help='llama-server from llama.cpp 7fe450e19')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model, gguf_path, server, output = [p.resolve() for p in (args.model, args.gguf, args.server, args.output)]
    manifest = json.loads(MANIFEST.read_text())
    verify_checkpoint(model, manifest)
    conversion = json.loads(args.conversion_report.read_text())
    require_conversion(conversion, gguf_path)
    version = subprocess.check_output([str(server), '--version'], stderr=subprocess.STDOUT, text=True)
    if '7fe450e19' not in version:
        raise ValueError('This comparison pins llama.cpp commit 7fe450e19')
    source = source_identity(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(platform=platform.platform(), machine=platform.machine(),
                       chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                       memory_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)))
    record = dict(schema_version=1, environment=environment, purpose='pretrained-cross-runtime-diagnostic', status='running',
                  source=source, checkpoint_manifest_sha256=sha(MANIFEST), prompts_sha256=sha(PROMPTS),
                  gguf_sha256=conversion['gguf_sha256'], conversion=conversion, server_version=version,
                  server_sha256=sha(server), numerical_conformance='not-tested',
                  native_swiftsci_generation='not-executed', cases=[])
    process = None
    try:
        with (output / 'python-environment.txt').open('w') as stream:
            subprocess.run([str(args.python.absolute()), '-m', 'pip', 'freeze'], check=True, stdout=stream)
        with (output / 'reference.log').open('w') as log:
            subprocess.run([str(args.python.absolute()), str(ROOT / 'Benchmarks/Python/llama_reference_worker.py'),
                            '--model', str(model), '--prompts', str(PROMPTS), '--output', str(output / 'reference.json')],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        reference = json.loads((output / 'reference.json').read_text())
        if reference['status'] != 'passed':
            raise ValueError('MLX reference failed')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        api_key = secrets.token_hex(32)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_'))}
        env['LLAMA_API_KEY'] = api_key
        command = [str(server), '--model', str(gguf_path), '--host', '127.0.0.1', '--port', str(port),
                   '--ctx-size', '512', '--parallel', '1', '--gpu-layers', '99', '--flash-attn', 'on',
                   '--cache-type-k', 'bf16', '--cache-type-v', 'bf16', '--no-cache-prompt', '--cache-ram', '0',
                   '--offline', '--no-webui', '-lv', '4']
        record['server_command'] = command
        # Ignore system proxies for the temporary loopback server.
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(path, data=None, timeout=120):
            req = urllib.request.Request(f'http://127.0.0.1:{port}/{path}',
                data=None if data is None else json.dumps(data).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key})
            with http.open(req, timeout=timeout) as response:
                return json.load(response)

        start = time.perf_counter_ns()
        with (output / 'server.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError('llama.cpp exited before readiness; inspect server.log')
            try:
                if request('health', timeout=2).get('status') == 'ok':
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('llama.cpp readiness timeout')
            time.sleep(0.25)
        record['server_startup_to_ready_ns'] = time.perf_counter_ns() - start
        runtime_log = (output / 'server.log').read_text()
        if 'using device MTL' not in runtime_log or 'offloaded 17/17 layers to GPU' not in runtime_log or 'K (bf16)' not in runtime_log or 'V (bf16)' not in runtime_log:
            raise ValueError('Expected Metal offload and BF16 KV cache were not confirmed')
        record['metal_offload_confirmed'] = True
        record['kv_cache_precision'] = 'bf16'
        record['server_properties'] = request('props')
        props = record['server_properties']
        if props['model_path'] != str(gguf_path) or props['model_ftype'] != 'BF16' or props['total_slots'] != 1:
            raise ValueError('Unexpected server model or slot configuration')
        eos = json.loads((model / 'config.json').read_text())['eos_token_id']
        for case in reference['cases']:
            if any(s['cache_dtypes'] != ['mlx.core.bfloat16'] for s in case['samples']):
                raise ValueError('MLX cache is not BF16')
            row = dict(id=case['id'], samples=[])
            raw = request('tokenize', dict(content=case['text'], add_special=False, parse_special=True))['tokens']
            chat = request('tokenize', dict(content=case['rendered_prompt'], add_special=False, parse_special=True))['tokens']
            row.update(raw_token_agreement=token_agreement(case['raw_tokens'], raw),
                       chat_token_agreement=token_agreement(case['prompt_tokens'], chat))
            record['cases'].append(row)
            if not row['raw_token_agreement']['equal'] or not row['chat_token_agreement']['equal']:
                raise ValueError('Tokenizer mismatch: ' + case['id'])
            for sample in range(3):
                start = time.perf_counter_ns()
                result = request('completion', dict(prompt=case['prompt_tokens'], n_predict=32, temperature=0,
                    seed=0, cache_prompt=False, return_tokens=True, repeat_penalty=1.0,
                    presence_penalty=0.0, frequency_penalty=0.0, samplers=['temperature']))
                elapsed = time.perf_counter_ns() - start
                if result['truncated'] or result['timings']['cache_n'] != 0 or result['tokens_evaluated'] != len(chat):
                    raise ValueError('Prompt truncated, cached, or changed')
                if not result['tokens'] or result['stop_type'] not in ('eos', 'limit'):
                    raise ValueError('Missing generation or unexpected stop')
                row['samples'].append(dict(warmup=sample == 0, request_elapsed_ns=elapsed, response=result))
            first = row['samples'][0]['response']
            row['greedy_repeat_equal'] = all(s['response']['tokens'] == first['tokens'] and s['response']['stop_type'] == first['stop_type'] for s in row['samples'])
            if not row['greedy_repeat_equal']:
                raise ValueError('llama.cpp greedy replay differs')
            mlx_first = case['samples'][0]
            row['cross_runtime_tokens'] = token_agreement(visible_tokens(mlx_first['tokens'], eos), visible_tokens(first['tokens'], eos))
            row['cross_runtime_stop_equal'] = mlx_first['finish_reason'] == {'eos': 'stop', 'limit': 'length'}[first['stop_type']]
        verify_checkpoint(model, manifest)
        require_conversion(conversion, gguf_path)
        if source_identity(ROOT) != source or sha(PROMPTS) != record['prompts_sha256'] or sha(server) != record['server_sha256']:
            raise ValueError('Source, prompts, or runtime changed during comparison')
        record['integration_checks'] = 'passed'
        record['cross_runtime_exact_generations'] = sum(c['cross_runtime_tokens']['equal'] and c['cross_runtime_stop_equal'] for c in record['cases'])
        record['status'] = 'matched' if record['cross_runtime_exact_generations'] == len(record['cases']) else 'completed-with-differences'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        (output / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ['status', 'cross_runtime_exact_generations', 'numerical_conformance', 'native_swiftsci_generation']}, indent=2))


if __name__ == '__main__':
    main()
