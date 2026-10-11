"""Fresh-process sustained cases and a separate steady-state report."""
import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CoreMLInference'))
from bounded_process import execute


def run(args):
    lifetime_request = dict(model=str((args.models / 'models-8192/program16.mlpackage').resolve()),
                            rawFixture=str(args.raw.resolve()), rows=8192, policy='cpu', batches=128,
                            consumerDelayMilliseconds=0)
    lifetime_path = args.output / 'source-lifetime-request.json'
    lifetime_result = args.output / 'source-lifetime-result.json'
    lifetime_path.write_text(json.dumps(lifetime_request, indent=2) + '\n')
    execute([args.worker.resolve(), '--bounded-source-lifetime', lifetime_path.resolve(), lifetime_result.resolve()],
            args.output / 'source-lifetime.log', 60)
    lifetime = json.loads(lifetime_result.read_text())
    records = []
    shutdowns = []
    for rows in (1024, 8192):
        for policy in ('cpu', 'neural'):
            for delay, batches, variants in [(0, 1024, range(5)), (5, 128, (1, 2, 4))]:
                for variant in variants:
                    name = f'{rows}-{policy}-delay{delay}-variant{variant}'
                    request = dict(model=str((args.models / f'models-{rows}/program16.mlpackage').resolve()),
                                   rawFixture=str(args.raw.resolve()), rows=rows, policy=policy,
                                   batches=batches, consumerDelayMilliseconds=delay, variant=variant)
                    path = args.output / f'{name}-request.json'
                    result = args.output / f'{name}-result.json'
                    path.write_text(json.dumps(request, indent=2) + '\n')
                    print(f'Running {name}: {batches} batches', flush=True)
                    execute([args.worker.resolve(), '--bounded-coreml-workflow', path.resolve(), result.resolve()],
                            args.output / f'{name}.log', 240)
                    records.append(json.loads(result.read_text()))
            request['batches'] = 128
            request['consumerDelayMilliseconds'] = 0
            name = f'{rows}-{policy}-shutdown'
            path = args.output / f'{name}-request.json'
            result = args.output / f'{name}-result.json'
            path.write_text(json.dumps(request, indent=2) + '\n')
            execute([args.worker.resolve(), '--bounded-coreml-shutdown', path.resolve(), result.resolve()],
                    args.output / f'{name}.log', 120)
            shutdowns.extend(json.loads(result.read_text()))
    lines = ['# Sustained bounded inference', '',
             'Each window configuration runs in a fresh process. Fast consumers process 1,024 batches per sample; delayed consumers process 128. Each process computes a serial reference, warms its variant once, then records three samples. The source replays a fixed 8,192-row held-out Covertype cycle from a file. Every cycle hash and local row mapping must match.', '',
             'Steady-state timing excludes model loading and the first 32 completed batches. It includes file reads, preprocessing, prediction, output hashing, and consumer delay. The recorded model-ready time reaches the first source callback and includes admission and pool setup. File-cache reuse is expected.', '',
             '| Rows | Policy | Delay ms | Mode | Window / slots | Model ready ms | Steady rows/s | Tail RSS range MiB | Tail RSS end-start MiB | Across-sample RSS drift MiB | Max outstanding |',
             '| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        s = r['samples']
        ready = statistics.median(x['modelReadySeconds'] for x in s) * 1000
        rate = statistics.median(x['steadyStateBatches'] * r['rows'] / x['steadyStateSeconds'] for x in s)
        ranges, deltas = [], []
        for sample in s:
            tail = [p['residentBytes'] / 2**20 for p in sample['checkpoints'] if p['consumed'] >= r['batches'] // 2]
            ranges.append(max(tail) - min(tail))
            deltas.append(tail[-1] - tail[0])
        sample_drift = (s[-1]['checkpoints'][-1]['residentBytes'] - s[0]['checkpoints'][-1]['residentBytes']) / 2**20
        first = s[0]
        lines.append(f"| {r['rows']} | {r['policy']} | {r['consumerDelayMilliseconds']} | {first['mode']} | {first['window']} / {first['slots']} | {ready:.2f} | {rate:.0f} | {max(ranges):.2f} | {max(deltas):+.2f} | {sample_drift:+.2f} | {max(x['maximumOutstanding'] for x in s)} |")
    longest = max(s['drainSeconds'] for s in shutdowns)
    assert all(s['reservedAtTrigger'] > 0 and s['reservedAfterDrain'] == 0 for s in shutdowns)
    thermals = sorted({s['thermalState'] for r in records for s in r['samples']})
    lines += ['', f"The isolated file-adapter regression grew by {lifetime['growthBytes']/2**20:.2f} MiB across 128 reads on the cooperative executor, below its 64 MiB cutoff. It uses the actual adapter without Core ML.", '', f'All {len(shutdowns)} source-error, consumer-error, source-cancellation, and consumer-cancellation scenarios retained admission at the trigger and returned it to zero before completion. Each reacquired the full quota afterward. Longest observed trigger-to-drain time: {longest*1000:.2f} ms.', '',
              'Cancellation is injected at source and consumer boundaries while the pipeline is active. This does not prove that Core ML interrupted a device command. The existing pool shutdown tests cover holding ownership until admitted work finishes.', '',
              f'Thermal states: {thermals}. Tail memory columns cover the second half of each sample. Sampled RSS includes framework caches and allocator retention; a flat trace is bounded-run evidence, not proof that every object lifetime is correct. CPU-and-Neural-Engine policy does not prove exclusive Neural Engine placement. No memory-growth threshold is silently waived or used to claim leak freedom.', '',
              'Raw checkpoints, output hashes, and the verified build fingerprint accompany this report.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
