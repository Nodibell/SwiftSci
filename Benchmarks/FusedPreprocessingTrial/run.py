#!/usr/bin/env python3
"""Run the bounded fusion experiment and write a readable timing report."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    result = args.output / 'results.json'
    if result.exists():
        parser.error('Use a new output directory; existing results are not overwritten')
    worker = args.worker.resolve(strict=True)
    digest = hashlib.sha256()
    with worker.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    fingerprint = {
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'worker_sha256': digest.hexdigest(),
        'platform': platform.platform(),
        'machine': platform.machine(),
    }
    (args.output / 'fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    execute([worker, '--fused-preprocessing-trial', result.resolve()], args.output / 'worker.log', 180)
    data = json.loads(result.read_text())
    lines = ['# Fused preprocessing trial', '', data['scope'] + '.', '', data['ownership'] + '.', '',
             f"Validation cases passed: {data['validationCases']}. Numeric cases check both row and tiled fusion; one case checks schema rejection.", '',
             'Times are medians of nine interleaved samples after two warmups per mode.', '',
             '| Rows | Columns | Missing | Staged ms | Fused row ms | Fused tile ms | Staged / tile |',
             '| ---: | ---: | :--- | ---: | ---: | ---: | ---: |']
    groups = {}
    for sample in data['samples']:
        key = (sample['rows'], sample['columns'], sample['missing'])
        groups.setdefault(key, {}).setdefault(sample['mode'], []).append(sample['seconds'])
    for (rows, columns, missing), modes in groups.items():
        times = {mode: statistics.median(values) * 1000 for mode, values in modes.items()}
        ratio = times['staged'] / times['fused-tile']
        lines.append(f"| {rows} | {columns} | {missing} | {times['staged']:.4f} | {times['fused-row']:.4f} | {times['fused-tile']:.4f} | {ratio:.2f}x |")
    thermal = sorted({sample['thermalState'] for sample in data['samples']})
    lines += ['', f'Thermal states observed: {thermal}. Zero is nominal.', '',
              'The largest input contains 4,194,304 Doubles. Fusion removes the full 32 MiB transformed-column intermediate and retains bounded row/tile scratch plus the final 16 MiB Float32 output. These are logical buffer sizes, not measured process-memory savings.', '',
              'This comparison retains the input snapshot for every mode. It does not measure uniquely owned input, inference, GPU/Neural Engine work, concurrent callers, or a production dataset.', '',
              f"Recorded source head: `{fingerprint['head']}`. See fingerprint.json for working-tree status and the executable hash."]
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
