#!/usr/bin/env python3
"""Compare persistent model ownership against unchanged mixed-workload baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/CoreMLInference'))
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from bounded_process import execute
from contracts import read_json, require, verified_file
from runner import environment, power_state, source_identity, verify_uninstrumented, metal_build_record
from report import validate, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    worker = args.worker.resolve()
    build = read_json(Path(str(worker) + '.build.json'))
    verified_file(worker, build['binary_sha256'])
    verify_uninstrumented(worker)
    require(build['coverage_instrumentation'] == 'absent', 'Instrumented build record')
    require(build['source']['tree_sha256'] == source_identity(ROOT)['tree_sha256'], 'Worker source is stale')
    require(build['environment'] == environment(), 'Toolchain differs from build')
    require(build['metal'] == metal_build_record(worker), 'Metal resources differ from build')
    args.output.mkdir(parents=True, exist_ok=False)
    fingerprint = dict(build=build, environment=environment(), power_state=power_state(),
                       head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                       controller={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')})
    (args.output / 'fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
    batches = 32 if args.smoke else 256
    cases = [(rows, policy, quantum) for rows in (1024, 8192)
             for policy in ('cpu', 'neural') for quantum in (8,)]
    if args.smoke:
        cases = cases[:1]
    records = []
    for rows, policy, quantum in cases:
        name = f'{rows}-{policy}-quantum{quantum}'
        request = dict(model=str((args.models / f'models-{rows}/program16.mlpackage').resolve()),
                       rawFixture=str(args.raw.resolve()), rows=rows, policy=policy, batches=batches, quantum=quantum)
        request_path = args.output / f'{name}-request.json'
        output = args.output / f'{name}-result.json'
        model = Path(request['model'])
        fingerprint.setdefault('models', {})[str(model)] = {
            str(p.relative_to(model)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model.rglob('*')) if p.is_file()}
        (args.output / 'fingerprint.json').write_text(json.dumps(fingerprint, indent=2) + '\n')
        request_path.write_text(json.dumps(request, indent=2) + '\n')
        print(f'Running {name}', flush=True)
        execute([worker, '--persistent-model-trial', request_path.resolve(), output.resolve()],
                args.output / f'{name}.log', 360)
        record = json.loads(output.read_text())
        validate(record, batches)
        records.append((request, record))
    report = render(records)
    (args.output / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
