#!/usr/bin/env python3
"""Run bounded, isolated Core ML fusion comparisons and summarize wall time."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'CoreMLInference'))
from bounded_process import execute


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    fingerprint = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   'status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                   'worker_sha256': sha(args.worker), 'raw_sha256': sha(args.raw), 'platform': platform.platform()}
    (args.output / 'fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    cases = [(r, p, c) for r in [1024, 8192] for p in ['cpu', 'neural'] for c in [1, 4]]
    if args.smoke:
        cases = cases[:1]
    records = []
    for rows, policy, callers in cases:
        name = f'{rows}-{policy}-{callers}'
        request = {'model': str((args.models / f'models-{rows}/program16.mlpackage').resolve()),
                   'fixture': str((args.prepared / f'fixture-512-{rows}.json').resolve()),
                   'rawFixture': str(args.raw.resolve()), 'rows': rows, 'policy': policy,
                   'callers': callers, 'slots': min(callers, 2)}
        path = args.output / f'{name}-request.json'
        path.write_text(json.dumps(request, indent=2) + '\n')
        result = args.output / f'{name}-result.json'
        print(f'Running {name}', flush=True)
        execute([args.worker.resolve(), '--fused-coreml-workflow', path.resolve(), result.resolve()],
                args.output / f'{name}.log', 240)
        record = json.loads(result.read_text())
        if not record['poolReleased'] or record['finalPoolReservedBytes'] or record['finalInputReservedBytes']:
            raise RuntimeError('Pool or input ownership did not drain')
        records.append(record)
    lines = ['# Covertype fused preprocessing and Core ML', '',
             'Frozen 54 → 512 → 512 → 7 model with Float16 transport. Swift imputation and scaling fit the original 11,340 training rows. Inputs are from the recorded held-out selection.', '',
             'Each cell is a median of five interleaved samples after warmup. Wall time includes preprocessing, packing, prediction, owned output, and input reservation cleanup. Model loading and fitting are excluded.', '',
             '| Rows | Policy | Callers / slots | Native ms | Staged packed ms | Fused copied ms | Fused direct ms | Copied / direct |',
             '| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |']
    for record in records:
        medians = {mode: statistics.median(s['wallSeconds'] for s in record['samples'] if s['mode'] == mode) * 1000
                   for mode in ['native', 'staged-packed', 'fused-packed', 'fused-direct']}
        lines.append(f"| {record['rows']} | {record['policy']} | {record['callers']} / {record['slots']} | {medians['native']:.3f} | {medians['staged-packed']:.3f} | {medians['fused-packed']:.3f} | {medians['fused-direct']:.3f} | {medians['fused-packed']/medians['fused-direct']:.2f}x |")
    lines += ['', '## Uniquely owned input', '',
              'The consuming reference receives fresh, unique column buffers before timing. Fixture copying is excluded. This is a one-caller comparison; shared-input results above retain their source snapshots.', '',
              '| Rows | Policy | Consuming native ms | Direct fused ms | Consuming / fused |',
              '| ---: | :--- | ---: | ---: | ---: |']
    for record in records:
        if record['callers'] != 1:
            continue
        owned = statistics.median(s['wallSeconds'] for s in record['samples'] if s['mode'] == 'native-owned') * 1000
        direct = statistics.median(s['wallSeconds'] for s in record['samples'] if s['mode'] == 'fused-direct') * 1000
        lines.append(f"| {record['rows']} | {record['policy']} | {owned:.3f} | {direct:.3f} | {owned/direct:.2f}x |")
    thermals = sorted({s['thermalState'] for r in records for s in r['samples']})
    rss = max(s['residentBytes'] for r in records for s in r['samples']) / (1024 * 1024)
    lines += ['', 'All recorded predictions matched the native path exactly within each compute policy, including row identity. All pool owners and reservations drained.', '',
              f'Thermal states: {thermals}. Maximum sampled process residency: {rss:.1f} MiB. These samples are not peak-allocation measurements.', '',
              'The neural policy permits Core ML to use CPU and Neural Engine; it does not prove exclusive Neural Engine execution. Float16 model accuracy remains the separately qualified model property.', '',
              'The copied packed paths include an additional owned copy. Direct fusion fills reserved model-typed storage outside the pool actor and avoids that intermediate array and copy. Every path still copies prepared storage into an exclusive prediction slot. Native uses the existing public preparation API. These are package-only experiments, not a public API proposal or a dispatch threshold.', '',
              f"Source head: `{fingerprint['head']}`. See fingerprint.json and individual results for provenance, timings, and cleanup evidence."]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
