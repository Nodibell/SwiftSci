#!/usr/bin/env python3
"""Run bounded real-model generation diagnostics; never certify native SwiftSci."""
import argparse
import json
from pathlib import Path
import platform
import subprocess

from pretrained_llama import MANIFEST, PROMPTS, ROOT, sha, verify_checkpoint
from llama_generation_cases import build_cases, summarize, validate_run

SUITE = ROOT / 'Benchmarks/Fixtures/pretrained/generation-suite.json'


def main():
    from runner import source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'python', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    verify_checkpoint(args.model, manifest)
    suite, smoke = json.loads(SUITE.read_text()), json.loads(PROMPTS.read_text())
    cases = build_cases(smoke, suite)
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [MANIFEST, PROMPTS, SUITE]}
    report = dict(schema_version=1, status='running', purpose='free-running-generation-and-batching',
        source=source, input_sha256=inputs, platform=platform.platform(), runs=[],
        chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        numerical_certificate=False, production_performance_baseline=False, native_swiftsci_generation='not-executed')
    try:
        with (args.output / 'python-environment.txt').open('w') as stream:
            subprocess.run([str(args.python.absolute()), '-m', 'pip', 'freeze'], check=True, stdout=stream)
        for mode in ['bf16', 'head-float32', 'shared-float32', 'float32']:
            output = args.output / (mode + '.json')
            with (args.output / (mode + '.log')).open('w') as log:
                subprocess.run([str(args.python.absolute()), str(ROOT / 'Benchmarks/Python/llama_generation_worker.py'),
                    '--model', str(args.model.resolve()), '--smoke', str(PROMPTS), '--suite', str(SUITE),
                    '--precision', mode, '--output', str(output.resolve())],
                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            result = json.loads(output.read_text())
            if result['precision'] != mode or result['suite_sha256'] != sha(SUITE) or result['smoke_sha256'] != sha(PROMPTS):
                raise ValueError('Worker identity or fixture mismatch')
            validate_run(result, cases, suite)
            if report['runs'] and result['versions'] != report['runs'][0]['versions']:
                raise ValueError('Dependency versions changed across policies')
            report['runs'].append(result)
            print(mode + ' completed', flush=True)
        report['summary'] = summarize(report['runs'])
        verify_checkpoint(args.model, manifest)
        if source_identity(ROOT) != source or any(sha(ROOT / p) != h for p, h in inputs.items()):
            raise ValueError('Source or inputs changed during generation')
        report['status'] = 'completed-with-observations'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'generation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
