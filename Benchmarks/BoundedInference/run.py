#!/usr/bin/env python3
"""Compare file-backed Core ML batch windows in serial bounded processes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/CoreMLInference'))
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from bounded_process import execute
from runner import environment, power_state, verify_uninstrumented, source_identity, metal_build_record
from contracts import read_json, require, verified_file


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--smoke', action='store_true')
    modes.add_argument('--sustained', action='store_true')
    args = parser.parse_args()
    build = read_json(Path(str(args.worker) + '.build.json'))
    verified_file(args.worker, build['binary_sha256'])
    require(build.get('coverage_instrumentation') == 'absent', 'Worker needs a recorded uninstrumented build')
    verify_uninstrumented(args.worker)
    require(build.get('metal') == metal_build_record(args.worker), 'Metal resources changed; rebuild')
    require(build['environment'] == environment(), 'Build and run toolchains differ; rebuild')
    require(build['source']['tree_sha256'] == source_identity(ROOT)['tree_sha256'], 'Worker source is stale; rebuild')
    args.output.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    files = {name: sha(ROOT / name) for name in sorted(set(names))
             if (name.startswith(('Sources/', 'Benchmarks/Worker/', 'Benchmarks/BoundedInference/'))
                 or name in ('Package.swift', 'Package.resolved')) and (ROOT / name).is_file()}
    fingerprint = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       files=files, worker_sha256=sha(args.worker), raw_sha256=sha(args.raw),
                       environment=environment(), power_state=power_state(), build=build)
    (args.output / 'fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    if args.sustained:
        from sustained import run
        run(args)
        return
    cases = [(rows, policy, delay) for rows in (1024, 8192) for policy in ('cpu', 'neural') for delay in (0, 5)]
    if args.smoke:
        cases = cases[:1]
    records = []
    for rows, policy, delay in cases:
        name = f'{rows}-{policy}-delay{delay}'
        request = dict(model=str((args.models / f'models-{rows}/program16.mlpackage').resolve()),
                       rawFixture=str(args.raw.resolve()), rows=rows, policy=policy, batches=8,
                       consumerDelayMilliseconds=delay)
        path = args.output / f'{name}-request.json'
        result = args.output / f'{name}-result.json'
        path.write_text(json.dumps(request, indent=2) + '\n')
        print(f'Running {name}', flush=True)
        execute([args.worker.resolve(), '--bounded-coreml-workflow', path.resolve(), result.resolve()],
                args.output / f'{name}.log', 240)
        records.append(json.loads(result.read_text()))
    lines = ['# Bounded inference comparison', '',
             'Covertype training-only preprocessing and a frozen Float16 classifier. Each run processes eight file-backed batches. A row count of 8,192 repeats the recorded held-out population eight times.', '',
             'Medians of three interleaved samples after one warmup per variant. Timing includes compiled-model loading, file reads, decoding, preparation, prediction, output hashing, optional consumer delay, and shutdown. Fitting, fixture conversion, and model compilation are excluded.', '',
             '| Rows | Policy | Consumer delay ms | Mode | Window / slots | Wall ms | First result ms | Rows/s | Max outstanding | Sampled RSS MiB |',
             '| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        variants = sorted({(s['mode'], s['window'], s['slots']) for s in record['samples']})
        for mode, window, slots in variants:
            samples = [s for s in record['samples'] if (s['mode'], s['window'], s['slots']) == (mode, window, slots)]
            wall = statistics.median(s['wallSeconds'] for s in samples)
            first = statistics.median(s['firstResultSeconds'] for s in samples)
            outstanding = max(s['maximumOutstanding'] for s in samples)
            rss = max(s['sampledResidentBytes'] for s in samples) / 2**20
            lines.append(f"| {record['rows']} | {record['policy']} | {record['consumerDelayMilliseconds']} | {mode} | {window} / {slots} | {wall*1000:.2f} | {first*1000:.2f} | {record['rows']*record['batches']/wall:.0f} | {outstanding} | {rss:.1f} |")
    thermals = sorted({s['thermalState'] for r in records for s in r['samples']})
    lines += ['', 'Every output hash matched the serial direct API reference within its compute policy. Local row identity and ordered delivery were checked. The worker rejects an undrained reservation or exceeded batch window.', '',
              'Memory figures are sampled process residency, not allocator peaks or an enforced memory cap. The serial reference budgets its pool and preparation separately, so its reservation peak is not directly comparable to the pipeline whole-run reservation.', '',
              f'Thermal states: {thermals}. The neural policy permits CPU and Neural Engine execution; it does not establish device placement. Source and executable fingerprints are in fingerprint.json.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
