#!/usr/bin/env python3
"""Compare synchronous replicas with native async prediction on the same model artifact."""
import argparse
import json
from pathlib import Path
import random

from concurrency import ROOT, digest, save, run_case, percentile_upper
from runner import source_identity, verify_uninstrumented, metal_build_record


def report(records):
    lines = ['# Native async Core ML trial', '',
             'Identical model, input preparation, output copies, and admission estimates. '
             'Latency includes queueing; throughput includes output validation. No execution tracing is implied.', '',
             '| Policy | Execution | Callers/models/slots | Requests/s | Mean ms | p95 upper ms | RSS last/max MiB |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        q = r['request']; c = r['callers']; n = sum(x['latency']['count'] for x in c)
        mean = 1000 * sum(x['latency']['totalSeconds'] for x in c) / n
        p95 = percentile_upper(c, .95)
        tail = 'overflow' if p95 is None else f'{1000 * p95:.2f}'
        rss = [x['residentBytes'] / 1048576 for x in r['memory']]
        shape = '/'.join(str(q[k]) for k in ('callers', 'instances', 'admissionSlots'))
        lines.append(f"| {q['policy']} | {q['execution']} | {shape} | {n/r['elapsedSeconds']:.1f} | "
                     f'{mean:.2f} | {tail} | {rss[-1]:.1f}/{max(rss):.1f} |')
    lines += ['', 'The package-only trial does not change the public default. Each request owns its buffers; '
              'the shared model uses the documented native async operation. Cancellation probes, retained-output '
              'checks, source fingerprints, and RSS limits are the same as the concurrency qualification. '
              'The fixture remains a legacy neural-network model; results do not generalize to ML Programs.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('Duration must be between 1 and 300 seconds')
    worker = args.worker.resolve(); prepared = args.prepared.resolve()
    source = source_identity(ROOT)
    build = json.loads(Path(str(worker) + '.build.json').read_text())
    if source['tree_sha256'] != build['source']['tree_sha256'] or digest(worker) != build['binary_sha256']:
        raise RuntimeError('Worker/source mismatch')
    verify_uninstrumented(worker)
    if metal_build_record(worker) != build['metal']:
        raise RuntimeError('Metal fingerprint differs')
    prepared_record = json.loads((prepared / 'provenance.json').read_text())
    for name, sha in prepared_record['files'].items():
        if digest(prepared / name) != sha:
            raise RuntimeError('Prepared artifact changed: ' + name)
    cases = [(p, mode, callers, models, slots) for p in ('cpu', 'neural')
             for mode, callers, models, slots in (
                 ('synchronous', 1, 1, 1), ('synchronous', 4, 1, 2), ('synchronous', 4, 4, 2),
                 ('asynchronous', 1, 1, 1), ('asynchronous', 4, 1, 1),
                 ('asynchronous', 4, 1, 2), ('asynchronous', 4, 1, 4))]
    random.Random(20261008).shuffle(cases)
    args.output.mkdir(parents=True, exist_ok=False)
    sha = digest(Path(__file__))
    save(args.output / 'provenance.json', dict(source=source, build=build, controller_sha256=sha,
         shared_controller_sha256=digest(ROOT / 'Benchmarks/CoreMLInference/concurrency.py'),
         prepared=prepared_record, cases=cases, seconds=args.seconds))
    records = []
    for i, (policy, mode, callers, models, slots) in enumerate(cases):
        name = f'{i:02d}-{policy}-{mode}-c{callers}-m{models}-s{slots}'
        request = dict(fixture=str(prepared / 'fixture-512-8192.json'), rows=8192,
            policy=policy, execution=mode, callers=callers, instances=models, admissionSlots=slots,
            seconds=args.seconds, rssLimitBytes=1_073_741_824)
        print(f'{i+1}/{len(cases)} {name}', flush=True)
        records.append(run_case(worker, args.output, name, request))
        save(args.output / 'results.json', records)
        (args.output / 'REPORT.md').write_text(report(records))
    if source_identity(ROOT) != source or digest(worker) != build['binary_sha256'] or digest(Path(__file__)) != sha:
        raise RuntimeError('Source or worker changed during run')
    for name, expected in prepared_record['files'].items():
        if digest(prepared / name) != expected:
            raise RuntimeError('Prepared artifact changed during run')
    if len({r['modelSHA256'] for r in records}) != 1:
        raise RuntimeError('Model artifact changed')
    for policy in ('cpu', 'neural'):
        if len({tuple(r['outputHashes']) for r in records if r['request']['policy'] == policy}) != 1:
            raise RuntimeError('Execution modes disagree on output')
    print('Async comparison completed', flush=True)


if __name__ == '__main__':
    main()
