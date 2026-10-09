#!/usr/bin/env python3
"""Compare direct and retained production inputs with complete preparation cost."""
import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from bounded_process import execute


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule(samples, seed):
    rng = random.Random(seed)
    cases = []
    for _ in range(samples):
        reuses = [1, 2, 4, 16, 64]
        rng.shuffle(reuses)
        for reuse in reuses:
            modes = ['direct', 'retained']
            rng.shuffle(modes)
            cases += [dict(mode=mode, reuse=reuse) for mode in modes]
    return cases


def validate(record):
    q = record['request']
    if not record['poolReleased'] or record['finalPoolReservedBytes'] or record['finalInputReservedBytes']:
        raise ValueError('Outstanding owner or reservation')
    if record['transportElementBytes'] not in [2, 4] or not record['outputHash']:
        raise ValueError('Missing transport identity')
    actual = [dict(mode=s['mode'], reuse=s['reuse']) for s in record['samples']]
    if actual != q['schedule']:
        raise ValueError('Incomplete or reordered samples')
    for sample in record['samples']:
        if not all(math.isfinite(sample[k]) for k in ['preparationSeconds', 'predictionSeconds', 'totalSeconds', 'releaseSeconds']):
            raise ValueError('Nonfinite duration')
        if sample['inputReservedBytes'] < 0:
            raise ValueError('Negative reservation')
        if not 0 < sample['residentBytes'] < 1_073_741_824:
            raise ValueError('RSS cutoff')
        if sample['predictionSeconds'] <= 0 or sample['preparationSeconds'] < 0 or sample['releaseSeconds'] < 0:
            raise ValueError('Invalid duration')
        if sample['totalSeconds'] + 1e-9 < sample['predictionSeconds'] + sample['preparationSeconds']:
            raise ValueError('Preparation excluded from total')
        if (sample['inputReservedBytes'] > 0) != (sample['mode'] == 'retained'):
            raise ValueError('Incorrect retained input accounting')


def report(records):
    grouped = collections.defaultdict(list)
    for record in records:
        q = record['request']
        for reuse in [1, 2, 4, 16, 64]:
            pair = {m: [s for s in record['samples'] if s['reuse'] == reuse and s['mode'] == m]
                    for m in ['direct', 'retained']}
            if not all(pair.values()):
                continue
            medians = {m: statistics.median(s['totalSeconds'] for s in items) for m, items in pair.items()}
            key = q['rows'], q['policy'], q['callers'], q['slots'], reuse
            grouped[key].append((medians['direct'], medians['retained'],
                                 pair['retained'][0]['inputReservedBytes']))
    lines = ['# Production prepared-input comparison', '',
             'Totals include one preparation and the complete prediction burst. Model loading, source construction, '
             'output validation, and final asynchronous reservation release are outside timing. Release time is recorded separately. '
             'Inputs have the same column order and repeated-row mapping in both paths.', '',
             '| Rows | Policy | Callers/slots | Reuse | Direct ms | Retained ms | Speedup | Process speedup range | Extra reservation MiB |',
             '|---:|---|---|---:|---:|---:|---:|---|---:|']
    for (rows, policy, callers, slots, reuse), cases in sorted(grouped.items()):
        direct = statistics.median(c[0] for c in cases)
        retained = statistics.median(c[1] for c in cases)
        ratios = [c[0] / c[1] for c in cases]
        lines.append(f'| {rows} | {policy} | {callers}/{slots} | {reuse} | {direct*1000:.3f} | '
                     f'{retained*1000:.3f} | {direct/retained:.2f}x | {min(ratios):.2f}–{max(ratios):.2f}x | '
                     f'{statistics.median(c[2] for c in cases)/1048576:.3f} |')
    lines += ['', 'A speedup above 1 favors retained preparation. Ranges summarize independent processes, '
              'not confidence intervals. Reservations are estimates, not physical-memory savings. '
              'RSS samples can miss peaks. Neural policy allows CPU and Neural Engine execution and does not prove device placement. '
              'Host activity and thermal state can change these exploratory timings.', '',
              'Direct and retained outputs must match exactly within each artifact and policy. '
              'Differences from the strict Double reference remain recorded; passing transport validation does not certify application accuracy.']
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['worker', 'models', 'fixtures', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3, choices=range(1, 6))
    p.add_argument('--samples', type=int, default=5, choices=range(3, 16))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    paths = [args.worker.resolve(), Path(__file__).resolve()]
    for folder in ['Sources/SwiftML', 'Sources/SwiftPreprocessing', 'Sources/SwiftDataFrame', 'Benchmarks/Worker']:
        paths += list((root / folder).rglob('*.swift'))
    requests = []
    for rows in [1024, 8192]:
        model = args.models / f'models-{rows}' / 'program16.mlpackage'
        fixture = args.fixtures / f'fixture-512-{rows}.json'
        paths += [f for f in model.rglob('*') if f.is_file()]
        paths.append(fixture)
        # Freeze referenced binary fixture files as well as the JSON descriptor.
        for key in ['inputBinary', 'expectedBinary']:
            entry = json.loads(fixture.read_text()).get(key)
            if entry:
                paths.append(fixture.parent / entry['path'])
        for policy in ['cpu', 'neural']:
            for callers, slots in [(1, 1), (4, 2)]:
                for repeat in range(args.repeats):
                    requests.append(dict(model=str(model.resolve()), fixture=str(fixture.resolve()), rows=rows,
                        policy=policy, callers=callers, slots=slots,
                        schedule=schedule(args.samples, 20261009 + repeat)))
    identities = {str(path.resolve()): digest(path) for path in paths}
    (args.output / 'provenance.json').write_text(json.dumps(dict(files=identities), indent=2)+'\n')
    random.Random(20261009).shuffle(requests)
    records = []
    for index, request in enumerate(requests):
        name = f'{index:02d}-{request["rows"]}-{request["policy"]}-{request["callers"]}'
        q, result = args.output / (name+'.request.json'), args.output / (name+'.json')
        q.write_text(json.dumps(request, indent=2)+'\n')
        print(f'{index+1}/{len(requests)} {name}', flush=True)
        execute([args.worker.resolve(), '--coreml-prepared-workflow', q, result], args.output / (name+'.log'), 120)
        record = json.loads(result.read_text())
        if record['request'] != request:
            raise ValueError('Request differs')
        validate(record)
        peers = [r for r in records if r['request']['rows'] == request['rows'] and r['request']['policy'] == request['policy']]
        if any(r['outputHash'] != record['outputHash'] or r['modelSHA256'] != record['modelSHA256'] for r in peers):
            raise ValueError('Artifact or outputs changed across processes')
        records.append(record)
        (args.output / 'results.json').write_text(json.dumps(records, indent=2)+'\n')
        (args.output / 'REPORT.md').write_text(report(records))
    for path, sha in identities.items():
        if digest(Path(path)) != sha:
            raise ValueError('Source or artifact changed: '+path)


if __name__ == '__main__':
    main()
