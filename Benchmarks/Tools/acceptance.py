#!/usr/bin/env python3
"""Run every selected conformance profile and retain failures without stopping coverage."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from contracts import fields, identity, load_profile, read_json, require, digest, write_json
from reporting import audit, validate_worker, audit_artifacts
from runner import source_identity
from datasets import cache_path, reference, binary, load_manifest, resolve_workload
from contracts import load_workload, verified_file

POLICY = 'Benchmarks/Specs/acceptance.json'


def load_policy(root):
    policy = read_json(root / POLICY)
    fields(policy, ['schema_version', 'profiles', 'aliases', 'deferred'])
    require(policy['schema_version'] == 1, 'Unknown acceptance policy')
    names = []
    for entry in policy['profiles']:
        fields(entry, ['profile', 'tier', 'purpose'])
        require(entry['tier'] in ('cpu', 'apple', 'sweep'), 'Unknown execution tier')
        require(isinstance(entry['purpose'], str) and entry['purpose'], 'Missing profile purpose')
        load_profile(root, entry['profile'])
        names.append(entry['profile'])
    require(len(names) == len(set(names)), 'Duplicate acceptance profile')
    for alias, canonical in policy['aliases'].items():
        require(canonical in names and alias not in names, 'Invalid profile alias')
        left, right = load_profile(root, alias), load_profile(root, canonical)
        require({k: v for k, v in left.items() if k != 'id'} ==
                {k: v for k, v in right.items() if k != 'id'}, 'Alias profiles differ')
    available = {p.stem for p in (root / 'Benchmarks/Specs/profiles').glob('*.json') if ' ' not in p.name}
    require(set(names) | set(policy['aliases']) == available, 'Acceptance policy omits or invents profiles')
    require(isinstance(policy['deferred'], list) and all(isinstance(x, str) and x for x in policy['deferred']), 'Invalid deferred coverage')
    return policy


def contract_identity(root):
    names = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    return identity({name: digest((root / name).read_bytes()) for name in names
                     if name.startswith(('Benchmarks/Specs/', 'Benchmarks/Fixtures/')) and (root / name).is_file()})


def inspect_run(directory, profile, source, engines, root):
    run = read_json(directory / 'run.json')
    certificate = read_json(directory / 'certificate.json')
    require(certificate['run_sha256'] == digest((directory / 'run.json').read_bytes()), 'Certificate checksum mismatch')
    require(run['plan']['source'] == source == certificate['source'], 'Acceptance source changed')
    require(run['plan']['profile'] == profile, 'Unexpected run profile')
    require(set(run['plan']['engines']) == set(engines), 'Engine coverage differs')
    require(certificate['contract_hash'] == run['plan']['contract_hash'], 'Contract identity mismatch')
    cases = run['plan']['cases']
    require([c['case'] for c in cases] == profile['cases'], 'Planned cases differ')
    for c in cases:
        dataset = load_manifest(root, c['case']['dataset'])
        workload = resolve_workload(dataset, load_workload(root, c['case']['workload']))
        require(c['dataset'] == dataset and c['workload'] == workload, 'Resolved contract differs')
        require(c['case_key'] == identity({k: c[k] for k in ('case', 'dataset', 'workload')}), 'Invalid case identity')
    expected = {(c['case']['id'], e, b): c['case_key'] for c in cases for e in engines for b in range(profile['batches'])}
    events = run['events']
    require(events == [json.loads(s) for s in (directory / 'events.jsonl').read_text().splitlines()], 'Event log differs')
    observed = [(e['case_id'], e['engine'], e['batch']) for e in events]
    require(len(observed) == len(set(observed)) and set(observed) == set(expected), 'Incomplete or duplicate event coverage')
    failures, infrastructure = [], []
    for event, key in zip(events, observed):
        require(event['case_key'] == expected[key], 'Event identity differs')
        token = f'{key[0]}-{key[1]}-{key[2]}'
        request = read_json(directory / (token + '.request.json'))
        case = next(c for c in cases if c['case']['id'] == key[0])
        dataset, workload = case['dataset'], case['workload']
        input_path = cache_path(root, dataset)
        verified_file(input_path, dataset['sha256'], dataset['size_bytes'])
        expected_path = directory / (key[0] + '.expected.f64')
        expected_bytes = binary(reference(root, dataset, workload))
        require(expected_path.read_bytes() == expected_bytes, 'Independent expected output differs')
        expected_request = dict(schema_version=1, case_key=expected[key], operation=workload['operation'],
            dataset_kind=dataset['kind'], input_path=str(input_path), input_sha256=dataset['sha256'],
            input_bytes=dataset['size_bytes'], expected_path=str(expected_path), expected_sha256=digest(expected_bytes),
            rows=dataset['rows'], warmups=profile['warmups'], samples=profile['samples'],
            atol=workload['atol'], rtol=workload['rtol'])
        if dataset['kind'] == 'nist-univariate-v1': expected_request['input_skip_rows'] = dataset['data_start_line'] - 1
        require(request == expected_request, 'Request contract differs')
        response_path = directory / (token + '.response.json')
        if event['status'] == 'passed':
            response = read_json(response_path)
            require(response == event['result'], 'Worker response differs')
            validate_worker(response, request)
            if next(c for c in cases if c['case']['id'] == key[0])['workload']['operation'] == 'parquet-write':
                audit_artifacts(event, profile, directory)
        else:
            require(event['status'] == 'failed', 'Invalid event status')
            detail = dict(case=key[0], engine=key[1], batch=key[2], error=event.get('error', ''))
            try:
                response = read_json(response_path)
                fields(response, ['schema_version','case_key','status','samples','peak_rss_bytes','engine_version','error'])
                require(response['schema_version'] == 1 and response['samples'] == [] and
                        type(response['peak_rss_bytes']) is int and response['peak_rss_bytes'] >= 0 and
                        isinstance(response['engine_version'],str) and response['engine_version'], 'Invalid failed response fields')
                require(event.get('failed_result') == response, 'Failed response differs from event')
                require(response['status'] == 'failed' and response['case_key'] == expected[key], 'Invalid failed response')
                require(isinstance(response.get('error'), str) and response['error'], 'Missing worker error')
                detail['worker_error'] = response['error']
                failures.append(detail)
            except (OSError, ValueError, KeyError) as error:
                detail['artifact_error'] = str(error)
                infrastructure.append(detail)
    status = 'failed' if failures or infrastructure else 'passed'
    require(run['status'] == status == certificate['status'], 'Run status differs')
    require(certificate['validated_samples'] == sum(len(e['result']['samples']) for e in events if e['status'] == 'passed'), 'Sample total differs')
    if status == 'passed':
        audit(run, directory)
    return dict(status=status, coverage_complete=not infrastructure, engine_cases=len(events),
                passed=sum(e['status'] == 'passed' for e in events), failures=failures,
                infrastructure_errors=infrastructure, run_sha256=certificate['run_sha256'],
                certificate_sha256=digest((directory / 'certificate.json').read_bytes()))


def execute(root, policy, tier, output, worker, python, invoke=subprocess.run):
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity(root)
    frozen_contract = contract_identity(root)
    selected = [p for p in policy['profiles'] if tier == 'all' or p['tier'] == tier or (tier == 'apple' and p['tier'] == 'cpu')]
    require(selected, 'No selected profiles')
    report = dict(schema_version=1, source=source, policy_sha256=digest((root / POLICY).read_bytes()),
                  contract_sha256=frozen_contract, tier=tier, profiles=[], status='running', coverage_complete=False)
    write_json(output / 'acceptance.json', report)
    for entry in selected:
        name = entry['profile']
        print(f'Acceptance: {name}', flush=True)
        record = dict(profile=name, tier=entry['tier'])
        try:
            require(source_identity(root) == source and contract_identity(root) == frozen_contract, 'Source or contract changed during acceptance')
            profile = load_profile(root, name)
            base = [python, str(root / 'Benchmarks/Tools/bench.py')]
            with (output / (name + '.prepare.log')).open('w') as log:
                prepared = invoke(base + ['prepare', '--profile', name], cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            require(prepared.returncode == 0, 'Fixture preparation failed')
            with (output / (name + '.run.log')).open('w') as log:
                finished = invoke(base + ['run', '--profile', name, '--engines', 'swiftsci,pandas', '--swift-worker', worker, '--python', python, '--output', str(output / name)], cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=600 + len(profile['cases'])*profile['batches']*2*profile['timeout_seconds'])
            record.update(inspect_run(output / name, load_profile(root, name), source, ['swiftsci', 'pandas'], root))
            require(finished.returncode == (0 if record['status'] == 'passed' else 1), 'Unexpected runner exit')
            require(source_identity(root) == source and contract_identity(root) == frozen_contract, 'Source or contract changed during acceptance')
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            record.update(status='infrastructure-error', coverage_complete=False, error=str(error))
        report['profiles'].append(record)
        write_json(output / 'acceptance.json', report)
    report['coverage_complete'] = all(p['coverage_complete'] for p in report['profiles'])
    report['status'] = 'passed' if all(p['status'] == 'passed' for p in report['profiles']) else 'failed'
    write_json(output / 'acceptance.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', choices=['cpu', 'apple', 'sweep', 'all'], default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--swift-worker', required=True)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = execute(root, load_policy(root), args.tier, args.output.resolve(), args.swift_worker, args.python)
    print(json.dumps({k: report[k] for k in ('status', 'coverage_complete')}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
