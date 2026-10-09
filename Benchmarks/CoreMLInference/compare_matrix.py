#!/usr/bin/env python3
"""Compare two recorded production workers on identical fixed-matrix inference cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
from run import run_case, reference_accuracy_met


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(records):
    lines = ['# Production Core ML matrix comparison', '',
             'Warm end-to-end adapter latency includes admission, packing, prediction, output extraction, and the same benchmark result flattening. Loading, model compilation, and hashing are outside timing.', '',
             'Speedup is before divided by after. Each value is the median of the per-process medians. Ratios near one require repeat measurements before drawing a conclusion.', '',
             '| Workload | Rows | Allowed devices | Before ms | After ms | Speedup | Double reference |',
             '|---|---:|---|---:|---:|---:|---|']
    keys = sorted({tuple(r['case']) for r in records})
    for key in keys:
        group = [r for r in records if tuple(r['case']) == key]
        times = {v: statistics.median(statistics.median(s['elapsed_ns'] for s in r['result']['warmPredictions'])
                                     for r in group if r['variant'] == v) / 1e6 for v in ['before', 'after']}
        accuracy = 'within tolerance' if all(reference_accuracy_met(r['result']) for r in group) else 'outside tolerance'
        lines.append(f'| {key[0]} | {key[1]} | {key[3]} | {times["before"]:.4f} | {times["after"]:.4f} | {times["before"]/times["after"]:.3f}x | {accuracy} |')
    states = [r[k] for r in records for k in ['thermal_before', 'thermal_after']]
    lines += ['', f'Thermal states observed: {sorted(set(states))}. State 0 is nominal. Background work can affect these paired timings.', '', 'Compute-unit preferences are not proof of hardware placement. Precision exceedances remain visible. Source, model, input, output, worker, and thermal evidence are recorded beside this report. Sampled RSS does not prove leak freedom.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['before-worker', 'after-worker', 'before-build', 'after-build', 'fixtures', 'output']:
        parser.add_argument('--' + field, required=True, type=Path)
    parser.add_argument('--samples', type=int, default=64, choices=range(1, 513))
    parser.add_argument('--repeats', type=int, default=3, choices=range(1, 6))
    args = parser.parse_args()
    workers = {'before': args.before_worker.resolve(), 'after': args.after_worker.resolve()}
    builds = {'before': json.loads(args.before_build.read_text()), 'after': json.loads(args.after_build.read_text())}
    for variant, worker in workers.items():
        if sha(worker) != builds[variant]['binary_sha256'] or builds[variant]['coverage_instrumentation'] != 'absent':
            raise ValueError('Unverified or instrumented worker: ' + variant)
    old, new = (builds[v]['source']['files'] for v in ['before', 'after'])
    for path in old:
        if path.startswith('Benchmarks/Worker/') or path.startswith('Benchmarks/Support/'):
            if old[path] != new.get(path):
                raise ValueError('Comparison workers use different benchmark logic: ' + path)
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_files = [args.fixtures / f'fixture-512-{r}.json' for r in [1024, 8192]]
    provenance = dict(builds=builds, workers={k: str(v) for k, v in workers.items()},
                      fixtures={str(f.resolve()): sha(f) for f in fixture_files},
                      runner_sha256=sha(Path(__file__)), samples=args.samples, repeats=args.repeats)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    thermal_source = args.output / 'thermal.swift'
    thermal_source.write_text('import Foundation\nprint(ProcessInfo.processInfo.thermalState.rawValue)\n')
    thermal = args.output / 'thermal'
    subprocess.run(['xcrun', 'swiftc', str(thermal_source), '-o', str(thermal)], check=True, timeout=60)
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    workloads = [('linear128', r, None) for r in [1, 32, 1024]]
    workloads += [('mlp512', r, None) for r in [32, 1024]]
    workloads += [('covertype-mlp512', r, str(f.resolve())) for r, f in zip([1024, 8192], fixture_files)]
    pairs = [(repeat, w, r, fixture, policy) for repeat in range(args.repeats)
             for w, r, fixture in workloads for policy in ['cpu', 'gpu', 'neural']]
    random.Random(20261008).shuffle(pairs)
    records = []
    for index, (repeat, workload, rows, fixture, policy) in enumerate(pairs):
        case = (workload, rows, rows, f'coreml-{policy}-matrix-adapter')
        order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
        pair = []
        for variant in order:
            directory = args.output / f'{index:02d}-{variant}'
            directory.mkdir()
            case_env = dict(env)
            case_env.pop('SWIFTSCI_COREML_FIXTURE', None)
            if fixture: case_env['SWIFTSCI_COREML_FIXTURE'] = fixture
            state_before = int(subprocess.check_output([str(thermal)], text=True))
            host_load = os.getloadavg()
            result = run_case(workers[variant], directory, case, args.samples, case_env)
            state_after = int(subprocess.check_output([str(thermal)], text=True))
            if result['reservedBytesAfterPrediction'] != 0:
                raise ValueError('Leaked admission reservation')
            row = dict(case=case, repeat=repeat, variant=variant, thermal_before=state_before,
                       thermal_after=state_after, host_load_before=host_load, result=result)
            records.append(row); pair.append(result)
            (args.output / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
        for field in ['parameterSHA256', 'modelSHA256', 'inputSHA256', 'logicalInputPackingBytes',
                      'logicalOutputCopyBytes', 'reservationPeakBytes']:
            if pair[0][field] != pair[1][field]: raise ValueError('Comparison contract changed: ' + field)
        hashes = {s['output_sha256'] for r in pair for s in [r['firstPrediction'], *r['warmPredictions']]}
        if len(hashes) != 1: raise ValueError('Before/after outputs disagree: ' + str(case))
        (args.output / 'REPORT.md').write_text(report(records))
        print(f'{index+1}/{len(pairs)} matched: {workload} {rows} {policy}', flush=True)
    for variant, worker in workers.items():
        if sha(worker) != builds[variant]['binary_sha256']: raise ValueError('Worker changed during comparison')
    for path, expected in provenance['fixtures'].items():
        if sha(Path(path)) != expected: raise ValueError('Fixture changed during comparison')


if __name__ == '__main__':
    main()
