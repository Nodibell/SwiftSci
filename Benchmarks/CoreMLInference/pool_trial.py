#!/usr/bin/env python3
"""Compare fresh and persistent Core ML matrix buffers with bounded process lifetimes."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess

from concurrency import ROOT, digest, save, run_case, percentile_upper
from program_pool import verify_models, validate_backing_counts, backing_report
from runner import source_identity, verify_uninstrumented, metal_build_record, environment


def report(records):
    lines = ['# Persistent Core ML pool trial', '',
             'Identical model, precision, input preparation, and owned output copies. '
             'Latency includes queueing; throughput includes output validation. No execution tracing is implied.', '',
             '| Policy | Execution | Rows | I/O bytes | Callers/models/slots | Requests/s | Mean ms | p95 upper ms | RSS last/max MiB |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        q = r['request']; c = r['callers']; n = sum(x['latency']['count'] for x in c)
        mean = 1000 * sum(x['latency']['totalSeconds'] for x in c) / n
        p95 = percentile_upper(c, .95)
        tail = 'overflow' if p95 is None else f'{1000 * p95:.2f}'
        rss = [x['residentBytes'] / 1048576 for x in r['memory']]
        element_bytes = r.get('transportElementBytes', 4 if q.get('modelPackage') else 8)
        shape = '/'.join(str(q[k]) for k in ('callers', 'instances', 'admissionSlots'))
        lines.append(f"| {q['policy']} | {q['execution']} | {q['rows']} | {element_bytes} | {shape} | {n/r['elapsedSeconds']:.1f} | "
                     f'{mean:.2f} | {tail} | {rss[-1]:.1f}/{max(rss):.1f} |')
    lines += ['', 'Fresh synchronous mode is the public predictor default. Fresh asynchronous mode uses '
              'the package comparison entry point. Persistent mode uses the public scoped matrix pool. '
              'Model loading and preflight are outside timing. Throughput includes full-output hashing.', '',
              'Persistent mode reserves a 32 MiB model/buffer allowance plus one request allowance per slot '
              'throughout the scope. Fresh modes reserve only active request allowances; their model memory '
              'is outside the budget. These are different accounting policies, not equivalent RSS caps. '
              'Every persistent sample must retain its whole quota; after scope exit the quota must be zero. '
              'The same conservative per-request estimate is used in all modes.', '',
              'Fresh modes check two queued cancellations. Persistent cancellation is covered by the pool '
              'test suite and is not claimed by this timing run. Every mode checks invalid-input recovery, '
              'owner release, retained outputs, and identical output hashes within each policy and shape.', '',
              'RSS is sampled once per second and does not prove leak freedom. Compute plans do not prove '
              'Neural Engine execution. Strict Double-reference differences remain recorded separately. '
              'Thermal state and host activity can affect timings; these are exploratory results, not a formal baseline. '
              'The model is a trained Covertype legacy neural network, not an ML Program.']
    if records and records[0]['request'].get('modelPackage'):
        precision = ' and '.join(sorted({'Float16' if r.get('transportElementBytes', 4) == 2 else 'Float32' for r in records}))
        lines[-1] = lines[-1].replace('The model is a trained Covertype legacy neural network, not an ML Program.',
            f'The model is a trained Covertype ML Program with {precision} I/O and Float16 compute precision specified by the model.')
    return '\n'.join(lines) + '\n' + backing_report(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--models', type=Path, help='Use verified program32 packages in models-1024 and models-8192')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('Duration must be between 1 and 300 seconds')
    worker = args.worker.resolve(); prepared = args.prepared.resolve()
    source = source_identity(ROOT)
    build = json.loads(Path(str(worker) + '.build.json').read_text())
    if source['tree_sha256'] != build['source']['tree_sha256'] or digest(worker) != build['binary_sha256']:
        raise RuntimeError('Worker/source mismatch')
    verify_uninstrumented(worker)
    libraries = subprocess.check_output(["xcrun", "otool", "-L", str(worker)], text=True)
    if any(name in libraries for name in ("libclang_rt.tsan", "libclang_rt.asan", "libclang_rt.ubsan")):
        raise RuntimeError("Timing worker contains sanitizer instrumentation")
    if metal_build_record(worker) != build['metal']:
        raise RuntimeError('Metal fingerprint differs')
    prepared_record = json.loads((prepared / 'provenance.json').read_text())
    for name, sha in prepared_record['files'].items():
        if digest(prepared / name) != sha:
            raise RuntimeError('Prepared artifact changed: ' + name)
    models = args.models.resolve() if args.models else None
    manifests = verify_models(models, prepared) if models else None
    helper = Path(__file__).with_name('program_pool.py')
    helper_sha = digest(helper)
    cases = [(p, mode, rows, slots) for p in ('cpu', 'neural')
             for rows in (1024, 8192) for slots in (1, 2)
             for mode in ('synchronous', 'asynchronous', 'persistent')]
    random.Random(20261008).shuffle(cases)
    args.output.mkdir(parents=True, exist_ok=False)
    sha = digest(Path(__file__))
    shared = ROOT / "Benchmarks/CoreMLInference/concurrency.py"
    shared_sha = digest(shared)
    save(args.output / 'provenance.json', dict(source=source, build=build, controller_sha256=sha,
         shared_controller_sha256=shared_sha, load_average_start=os.getloadavg(),
         prepared=prepared_record, cases=cases, seconds=args.seconds, environment=environment(), models=manifests, program_helper_sha256=helper_sha))
    records = []
    for i, (policy, mode, rows, slots) in enumerate(cases):
        callers = 1 if slots == 1 else 4
        instances = 1
        name = f'{i:02d}-{policy}-{mode}-r{rows}-c{callers}-m{instances}-s{slots}'
        request = dict(fixture=str(prepared / f'fixture-512-{rows}.json'), rows=rows,
            policy=policy, execution=mode, callers=callers, instances=instances, admissionSlots=slots,
            seconds=args.seconds, rssLimitBytes=1_073_741_824)
        if models:
            request['modelPackage'] = str(models / f'models-{rows}' / 'program32.mlpackage')
        if mode == 'persistent':
            request['retainedBytes'] = 33_554_432
        print(f'{i+1}/{len(cases)} {name}', flush=True)
        record = run_case(worker, args.output, name, request)
        validate_backing_counts(record)
        records.append(record)
        save(args.output / 'results.json', records)
        (args.output / 'REPORT.md').write_text(report(records))
    if source_identity(ROOT) != source or digest(worker) != build['binary_sha256'] or digest(Path(__file__)) != sha:
        raise RuntimeError('Source or worker changed during run')
    for name, expected in prepared_record['files'].items():
        if digest(prepared / name) != expected:
            raise RuntimeError('Prepared artifact changed during run')
    if digest(helper) != helper_sha or (models and verify_models(models, prepared) != manifests):
        raise RuntimeError('Program helper or model provenance changed during run')
    if digest(shared) != shared_sha:
        raise RuntimeError('Shared controller changed during run')
    save(args.output / 'completed.json', dict(load_average_end=os.getloadavg(), cases=len(records)))
    for rows in (1024, 8192):
        subset = [r for r in records if r['request']['rows'] == rows]
        if len({r['modelSHA256'] for r in subset}) != 1:
            raise RuntimeError('Model artifact changed')
        for policy in ('cpu', 'neural'):
            selected = [r for r in subset if r['request']['policy'] == policy]
            if len({tuple(r['outputHashes']) for r in selected}) != 1:
                raise RuntimeError('Execution modes disagree on output')
            if len({r['preflightReferenceMismatches'] for r in selected}) != 1:
                raise RuntimeError('Execution modes disagree on reference accuracy')
    print('Persistent pool comparison completed', flush=True)


if __name__ == '__main__':
    main()
