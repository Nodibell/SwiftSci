#!/usr/bin/env python3
"""Record bounded allocation and device traces of verified public-pool workloads."""
import argparse
import json
import os
import signal
import shutil
import plistlib
from pathlib import Path
import subprocess

from concurrency import ROOT, digest, save, validate
from program_pool import validate_backing_counts, verify_models
from runner import source_identity, verify_uninstrumented, metal_build_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path, help='Completed ML Program comparison directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 10:
        parser.error('Profile duration must be 1...10 seconds per worker')
    worker = args.worker.resolve()
    build = json.loads(Path(str(worker) + '.build.json').read_text())
    source = source_identity(ROOT)
    if digest(worker) != build['binary_sha256'] or source['tree_sha256'] != build['source']['tree_sha256']:
        raise RuntimeError('Source/worker fingerprint mismatch')
    verify_uninstrumented(worker)
    if metal_build_record(worker) != build['metal']:
        raise RuntimeError('Metal build fingerprint mismatch')
    refs = json.loads((args.reference / 'results.json').read_text())
    provenance = json.loads((args.reference / 'provenance.json').read_text())
    first = refs[0]['request']
    prepared = Path(first['fixture']).parent
    models = Path(first['modelPackage']).parent.parent
    if verify_models(models, prepared) != provenance['models']:
        raise RuntimeError('Reference model files changed')
    args.output.mkdir(parents=True, exist_ok=False)
    profiling_worker = args.output.resolve() / 'ProfilePoolWorker'
    shutil.copy2(worker, profiling_worker)
    entitlements = args.output / 'profiling-entitlements.plist'
    entitlements.write_bytes(plistlib.dumps({'com.apple.security.get-task-allow': True}))
    subprocess.run(['codesign', '--force', '--sign', '-', '--entitlements', str(entitlements),
                    str(profiling_worker)], check=True)
    for resource in list(worker.parent.glob('*.bundle')) + list(worker.parent.glob('*.metallib')):
        (args.output / resource.name).symlink_to(resource)
    controller = digest(Path(__file__))
    save(args.output / 'provenance.json', dict(source=source, build=build, controller_sha256=controller,
        reference_results_sha256=digest(args.reference / 'results.json'), performance_evidence=False,
        profiling_worker_sha256=digest(profiling_worker), profiling_entitlements_sha256=digest(entitlements)))
    records = []
    cases = [(kind, policy, mode) for kind in ('allocations', 'devices')
             for policy in (('cpu', 'neural') if kind == 'allocations' else ('neural',))
             for mode in ('asynchronous', 'persistent')]
    for kind, policy, mode in cases:
        base = next(r for r in refs if r['request']['rows'] == 1024 and r['request']['admissionSlots'] == 2
                    and r['request']['policy'] == policy and r['request']['execution'] == mode)
        request = dict(base['request'], seconds=args.seconds)
        name = f'{kind}-{policy}-{mode}'
        request_path = args.output / (name + '.request.json')
        result_path = args.output / (name + '.json')
        trace = args.output / (name + '.trace')
        save(request_path, request)
        command = ['xcrun', 'xctrace', 'record', '--template',
                   'Allocations' if kind == 'allocations' else 'Time Profiler']
        if kind == 'devices':
            command += ['--instrument', 'Core ML', '--instrument', 'Neural Engine',
                        '--instrument', 'Points of Interest']
        command += ['--time-limit', f'{args.seconds + 40}s', '--output', str(trace), '--no-prompt',
                    '--launch', '--', str(profiling_worker), '--coreml-concurrency', str(request_path), str(result_path)]
        save(args.output / (name + '.command.json'), command)
        print(name, flush=True)
        with (args.output / (name + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.seconds + 90)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                # xctrace launches through a service, so its suspended child may be outside our process group.
                owned = str(profiling_worker) + ' --coreml-concurrency ' + str(request_path) + ' '
                for line in subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines():
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2 and parts[1].startswith(owned):
                        try:
                            os.kill(int(parts[0]), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            if code != 0:
                raise RuntimeError(f'{name}: recorder exited {code}; inspect its log')
        record = json.loads(result_path.read_text())
        validate(record, request)
        validate_backing_counts(record)
        if record['modelSHA256'] != base['modelSHA256'] or record['outputHashes'] != base['outputHashes']:
            raise RuntimeError('Profiled model or outputs differ from the reference')
        subprocess.run(['xcrun', 'xctrace', 'export', '--input', str(trace), '--toc', '--output',
                        str(args.output / (name + '.toc.xml'))], check=True, timeout=60)
        records.append(dict(name=name, record=record))
        save(args.output / 'results.json', records)
    if source_identity(ROOT) != source or digest(worker) != build['binary_sha256'] or digest(Path(__file__)) != controller:
        raise RuntimeError('Source, controller or worker changed during profiling')
    if verify_models(models, prepared) != provenance['models']:
        raise RuntimeError('Model files changed during profiling')
    print('Traces recorded. Inspect allocation lifetimes and device attribution before drawing conclusions.', flush=True)


if __name__ == '__main__':
    main()
