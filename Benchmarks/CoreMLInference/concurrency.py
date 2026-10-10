#!/usr/bin/env python3
"""Bounded public-API concurrency and sustained-load qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from runner import source_identity, verify_uninstrumented, metal_build_record


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def validate(record, request):
    if record['request'] != request:
        raise ValueError('Request identity changed')
    if record['executionTraceVerified']:
        raise ValueError('Execution tracing was not performed')
    if not record['predictorsReleased'] or not record['retainedOutputsVerified']:
        raise ValueError('Ownership checks failed')
    if record['finalReservedBytes'] != 0 or record['finalQueuedCount'] != 0:
        raise ValueError('Admission did not drain')
    persistent = request.get('execution') == 'persistent'
    if record['cancelledRequestsVerified'] != (0 if persistent else 2):
        raise ValueError('Cancellation checks incomplete')
    estimate = record['requestReservationBytes']
    if not 0 < estimate <= record['budgetPeakBytes'] <= record['budgetLimitBytes']:
        raise ValueError('Invalid peak reservation')
    retained = request.get('retainedBytes', 0)
    if persistent and (request['instances'] != 1 or not 16_777_216 <= retained <= 268_435_456):
        raise ValueError('Invalid persistent pool allowance')
    if not persistent and retained:
        raise ValueError('Unexpected lifetime allowance')
    if record['budgetLimitBytes'] != estimate * request['admissionSlots'] + retained:
        raise ValueError('Admission slots changed')
    if len(record['callers']) != request['callers']:
        raise ValueError('Missing callers')
    if [c['caller'] for c in record['callers']] != list(range(request['callers'])):
        raise ValueError('Duplicate callers')
    if len(set(record['outputHashes'])) != 2:
        raise ValueError('Input variants are not distinct')
    for caller in record['callers']:
        stats = caller['latency']
        if stats['count'] < 1 or len(stats['buckets']) != 256 or sum(stats['buckets']) != stats['count']:
            raise ValueError('Latency histogram incomplete')
        if any(type(n) is not int or n < 0 for n in stats['buckets']):
            raise ValueError('Invalid histogram count')
        for key in ('totalSeconds', 'minimumSeconds', 'maximumSeconds'):
            if not math.isfinite(stats[key]) or stats[key] <= 0:
                raise ValueError('Invalid latency')
        mean = stats['totalSeconds'] / stats['count']
        if not stats['minimumSeconds'] <= mean <= stats['maximumSeconds']:
            raise ValueError('Inconsistent latency summary')
    if not 0 < len(record['memory']) <= request['seconds'] + 2:
        raise ValueError('Missing or unbounded memory telemetry')
    for point in record['memory']:
        if not 0 < point['residentBytes'] <= request['rssLimitBytes']:
            raise ValueError('RSS limit exceeded')
        if persistent and point['reservedBytes'] != record['budgetLimitBytes']:
            raise ValueError('Persistent quota released before scope exit')
        if not 0 <= point['reservedBytes'] <= record['budgetLimitBytes']:
            raise ValueError('Reservation exceeds budget')
    if not request['seconds'] <= record['elapsedSeconds'] <= request['seconds'] + 30:
        raise ValueError('Run duration outside bounds')


def percentile_upper(callers, fraction):
    buckets = [sum(c['latency']['buckets'][i] for c in callers) for i in range(256)]
    target = math.ceil(sum(buckets) * fraction)
    count = 0
    for i, n in enumerate(buckets):
        count += n
        if count >= target:
            return None if i == 255 else 1e-6 * 1.08 ** i
    raise ValueError('Empty histogram')


def run_case(worker, output, name, request):
    request_path = output / (name + '.request.json')
    result_path = output / (name + '.json')
    save(request_path, request)
    with (output / (name + '.log')).open('w') as log:
        process = subprocess.Popen([str(worker), '--coreml-concurrency', str(request_path), str(result_path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=request['seconds'] + 120)
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    if code != 0:
        raise RuntimeError(f'{name} failed with exit {code}; inspect {name}.log')
    record = json.loads(result_path.read_text())
    validate(record, request)
    return record


def report(records):
    lines = ['# Core ML concurrency qualification', '',
        'Every request uses the public matrix predictor. Latency includes actor scheduling, admission wait, '
        'packing, prediction, and output copying. Those components are not separately instrumented. '
        'Throughput also includes complete-output hashing. Percentiles are histogram upper bounds with '
        '8% bucket spacing.', '',
        '| Policy | Callers/models/slots | Seconds | Requests | Requests/s | Mean ms | p95 ≤ ms | RSS first/last/max MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for record in records:
        request = record['request']; callers = record['callers']
        count = sum(c['latency']['count'] for c in callers)
        mean = sum(c['latency']['totalSeconds'] for c in callers) / count
        rss = [p['residentBytes'] / 1048576 for p in record['memory']]
        p95 = percentile_upper(callers, .95)
        percentile = f'{p95 * 1000:.2f}' if p95 is not None else 'overflow'
        shape = '/'.join(str(request[k]) for k in ('callers', 'instances', 'admissionSlots'))
        lines.append(f"| {request['policy']} | {shape} | {record['elapsedSeconds']:.1f} | {count:,} | "
                     f"{count / record['elapsedSeconds']:.1f} | {mean * 1000:.2f} | {percentile} | "
                     f'{rss[0]:.1f}/{rss[-1]:.1f}/{max(rss):.1f} |')
    lines += ['', 'All completed cases verified two cancelled queued requests, rejection cleanup, distinct '
        'normal/reversed outputs, predictor release, and retained-output ownership. Every returned output '
        'must match its policy-specific preflight hash. Strict Double-reference differences remain recorded '
        'separately; repeatability is not numerical certification.', '',
        'The admission budget estimates request allocations. Persistent models, retained results, and Core ML '
        'caches are outside it. RSS samples are process measurements, not allocation traces or proof of '
        'leak freedom. The RSS cutoff is sampled once per second; the external timeout bounds stalled runs. '
        'Compute plans do not prove actual Neural Engine execution.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True, help='Verified Covertype prepared directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sweep-seconds', type=int, default=15)
    parser.add_argument('--sustain-seconds', type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.sweep_seconds <= 60 or not 0 <= args.sustain_seconds <= 1800:
        parser.error('Sweep must be 1...60 seconds; sustained duration 0...1800')
    worker = args.worker.resolve(); prepared = args.prepared.resolve()
    source = source_identity(ROOT)
    build = json.loads(Path(str(worker) + '.build.json').read_text())
    if source['tree_sha256'] != build['source']['tree_sha256'] or digest(worker) != build['binary_sha256']:
        raise RuntimeError('Worker/source fingerprint differs')
    verify_uninstrumented(worker)
    if metal_build_record(worker) != build['metal']:
        raise RuntimeError('Metal fingerprint differs')
    provenance = json.loads((prepared / 'provenance.json').read_text())
    artifacts = {name: digest(prepared / name) for name in provenance['files']}
    if artifacts != provenance['files']:
        raise RuntimeError('Prepared evidence changed')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [(policy, callers, models, slots, args.sweep_seconds)
             for policy in ('cpu', 'neural') for callers, models, slots in ((1, 1, 1), (4, 1, 1), (4, 4, 1), (4, 4, 2))]
    random.Random(20261008).shuffle(cases)
    if args.sustain_seconds:
        cases.extend((policy, 4, 4, 2, args.sustain_seconds) for policy in ('cpu', 'neural'))
    records = []
    controller_sha = digest(Path(__file__))
    save(args.output / 'provenance.json', {'source': source, 'build': build, 'controller_sha256': controller_sha,
         'prepared_files': artifacts, 'cases': cases, 'fixture': 'fixture-512-8192.json'})
    for i, (policy, callers, instances, slots, seconds) in enumerate(cases):
        name = f'{i:02d}-{policy}-c{callers}-m{instances}-s{slots}-{seconds}s'
        request = dict(fixture=str(prepared / 'fixture-512-8192.json'), rows=8192, policy=policy,
                       callers=callers, instances=instances, admissionSlots=slots, seconds=seconds,
                       rssLimitBytes=1_073_741_824)
        print(f'{i + 1}/{len(cases)} {name}', flush=True)
        record = run_case(worker, args.output, name, request)
        records.append(record)
        save(args.output / 'results.json', records)
        (args.output / 'REPORT.md').write_text(report(records))
    if source_identity(ROOT) != source or digest(Path(__file__)) != controller_sha or digest(worker) != build['binary_sha256']:
        raise RuntimeError('Sources or binary changed during stress')
    if any(digest(prepared / name) != expected for name, expected in artifacts.items()):
        raise RuntimeError('Prepared artifacts changed during stress')
    if len({r['modelSHA256'] for r in records}) != 1:
        raise RuntimeError('Cases used different models')
    for policy in ('cpu', 'neural'):
        if len({tuple(r['outputHashes']) for r in records if r['request']['policy'] == policy}) != 1:
            raise RuntimeError('Fresh processes disagree on policy outputs')
    print('Completed; see REPORT.md', flush=True)


if __name__ == '__main__':
    main()
