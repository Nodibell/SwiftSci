#!/usr/bin/env python3
"""Opt-in short-context precision experiment; not a production performance baseline."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess

from pretrained_llama import MANIFEST, ROOT, sha, verify_checkpoint
from llama_score_analysis import validate_request


def summarize(runs):
    modes = ['bf16', 'head-float32', 'shared-float32', 'float32']
    case_ids = {c['id'] for c in runs[0]['cases']}
    summary = {}
    for mode in modes:
        selected = [r for r in runs if r['precision'] == mode]
        if len(selected) != 2 or any({c['id'] for c in r['cases']} != case_ids for r in selected):
            raise ValueError('Expected two complete blocks per precision')
        rows = []
        for case_id in sorted(case_ids):
            cases = [next(c for c in r['cases'] if c['id'] == case_id) for r in selected]
            samples = [s for c in cases for s in c['samples'] if not s['warmup']]
            if len(samples) != 10 or any(s['tokens'] != samples[0]['tokens'] for s in samples):
                raise ValueError('Incomplete timing samples or inconsistent predictions')
            rows.append(dict(id=case_id, median_elapsed_ns=statistics.median(s['elapsed_ns'] for s in samples),
                block_medians_ns=[statistics.median(s['elapsed_ns'] for s in c['samples'] if not s['warmup']) for c in cases],
                min_elapsed_ns=min(s['elapsed_ns'] for s in samples), max_elapsed_ns=max(s['elapsed_ns'] for s in samples),
                max_active_device_bytes=max(s['peak_active_device_bytes'] for s in samples)))
        if selected[0]['retained_parameter_bytes'] != selected[1]['retained_parameter_bytes']:
            raise ValueError('Model parameter memory differs between blocks')
        summary[mode] = dict(retained_parameter_bytes=selected[0]['retained_parameter_bytes'],
                            additional_output_weight_bytes=selected[0]['additional_output_weight_bytes'], cases=rows)
    for mode in modes:
        for row in summary[mode]['cases']:
            baseline = next(c for c in summary['bf16']['cases'] if c['id'] == row['id'])
            row['elapsed_ratio_to_bf16'] = row['median_elapsed_ns'] / baseline['median_elapsed_ns']
    return summary


def main():
    from runner import source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--score-run', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    verify_checkpoint(args.model, manifest)
    request_path = args.score_run / 'score-request.json'
    request_hash = sha(request_path)
    validate_request(json.loads(request_path.read_text()))
    expected = {}
    score_hashes = {}
    for mode in ['bf16', 'head-float32', 'shared-float32', 'float32']:
        path = args.score_run / ('scores-' + mode) / 'scores.json'
        score_hashes[mode] = sha(path)
        score = json.loads(path.read_text())
        if score['status'] != 'completed' or score['request_sha256'] != request_hash:
            raise ValueError('Expected completed score probes for the same request')
        expected[mode] = {c['id']: c['generated_tokens'] for c in score['cases']}
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT)
    report = dict(schema_version=1, status='running', purpose='short-context-precision-experiment',
                  source=source, checkpoint_manifest_sha256=sha(MANIFEST), request_sha256=request_hash,
                  score_metadata_sha256=score_hashes, platform=platform.platform(), runs=[],
                  numerical_certificate=False, production_performance_baseline=False,
                  chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip())
    try:
        with (args.output / 'python-environment.txt').open('w') as stream:
            subprocess.run([str(args.python.absolute()), '-m', 'pip', 'freeze'], check=True, stdout=stream)
        for block, modes in enumerate([['bf16', 'head-float32', 'shared-float32', 'float32'], ['float32', 'shared-float32', 'head-float32', 'bf16']]):
            for mode in modes:
                name = f'block-{block + 1}-{mode}'
                command = [str(args.python.absolute()), str(ROOT / 'Benchmarks/Python/llama_precision_timing_worker.py'),
                    '--model', str(args.model.resolve()), '--request', str(request_path.resolve()), '--precision', mode,
                    '--output', str((args.output / (name + '.json')).resolve())]
                if block == 1:
                    command.append('--reverse-cases')
                with (args.output / (name + '.log')).open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                run = json.loads((args.output / (name + '.json')).read_text())
                if run['status'] != 'completed' or run['request_sha256'] != request_hash:
                    raise ValueError('Timing worker failed or input identity changed')
                for case in run['cases']:
                    if any(s['tokens'] != expected[mode][case['id']] for s in case['samples']):
                        raise ValueError('Timing output differs from score-probe output')
                report['runs'].append(run)
                print(name + ' completed', flush=True)
        report['summary'] = summarize(report['runs'])
        verify_checkpoint(args.model, manifest)
        if source_identity(ROOT) != source or sha(request_path) != request_hash or sha(MANIFEST) != report['checkpoint_manifest_sha256']:
            raise ValueError('Source or inputs changed during measurement')
        for mode, digest in score_hashes.items():
            if sha(args.score_run / ('scores-' + mode) / 'scores.json') != digest:
                raise ValueError('Score evidence changed during measurement')
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (args.output / 'precision-timing.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
