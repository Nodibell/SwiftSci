import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
import acceptance
from acceptance import POLICY, execute, inspect_run, load_policy
from contracts import ContractError, digest, identity, read_json, write_json
from datasets import binary, cache_path


class AcceptanceTests(unittest.TestCase):
    def fixture(self, directory, failed=False, missing_response=False):
        root = directory / 'repository'
        source = dict(commit='a'*40, tree_sha256='b'*64, files={})
        case = dict(id='tiny', dataset='tiny', workload='row-sum-v1')
        profile = dict(id='example', cases=[case], batches=1, samples=1, warmups=0,
                       timeout_seconds=5)
        data = b'id,group,x,y\n0,0,1,2\n'
        dataset = dict(id='tiny', kind='table-v1', rows=1,
                       sha256=digest(data), size_bytes=len(data))
        workload = dict(operation='row-sum', atol=0, rtol=0)
        input_path = cache_path(root, dataset)
        input_path.parent.mkdir(parents=True)
        input_path.write_bytes(data)
        expected_path = directory / 'tiny.expected.f64'
        expected = binary([1])
        expected_path.write_bytes(expected)
        spec = dict(case=case, dataset=copy.deepcopy(dataset), workload=copy.deepcopy(workload))
        spec['case_key'] = identity(spec)
        key = spec['case_key']
        request = dict(schema_version=1, case_key=key, operation='row-sum',
                       dataset_kind='table-v1', input_path=str(input_path),
                       input_sha256=dataset['sha256'], input_bytes=len(data),
                       expected_path=str(expected_path), expected_sha256=digest(expected),
                       rows=1, warmups=0, samples=1, atol=0, rtol=0)
        response = dict(schema_version=1, case_key=key, status='failed' if failed else 'passed',
                        samples=[] if failed else [dict(elapsed_ns=1000, output_sha256=digest(expected),
                                                       maximum_absolute_error=0, validated=True)],
                        peak_rss_bytes=1024, engine_version='test')
        if failed:
            response['error'] = 'Output mismatch'
        event = dict(case_id='tiny', engine='swiftsci', batch=0,
                     case_key=key, status=response['status'])
        if failed:
            event.update(error='Worker exited 1', failed_result=copy.deepcopy(response))
        else:
            event['result'] = copy.deepcopy(response)
        run = dict(schema_version=1, status=response['status'], events=[event],
                   plan=dict(source=source, profile=profile, engines={'swiftsci': {}},
                             cases=[spec], contract_hash='d'*64))
        request_path = directory / 'tiny-swiftsci-0.request.json'
        response_path = directory / 'tiny-swiftsci-0.response.json'
        write_json(request_path, request)
        if not missing_response:
            write_json(response_path, response)
        fixture = SimpleNamespace(directory=directory, root=root, source=source, profile=profile,
                                  dataset=dataset, workload=workload, run=run,
                                  request_path=request_path, response_path=response_path,
                                  expected_path=expected_path, input_path=input_path)
        self.persist(fixture)
        return fixture

    def persist(self, fixture):
        directory, run = fixture.directory, fixture.run
        write_json(directory / 'run.json', run)
        (directory / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in run['events']))
        write_json(directory / 'certificate.json', dict(
            run_sha256=digest((directory / 'run.json').read_bytes()), source=fixture.source,
            contract_hash=run['plan']['contract_hash'], status=run['status'],
            validated_samples=sum(len(e['result']['samples']) for e in run['events']
                                  if e['status'] == 'passed')))

    def inspect(self, fixture):
        # Keep real request, input-byte, expected-byte, response and certificate checks.
        with patch('acceptance.load_manifest', return_value=fixture.dataset), \
             patch('acceptance.load_workload', return_value=fixture.workload), \
             patch('acceptance.reference', return_value=[1]):
            return inspect_run(fixture.directory, fixture.profile, fixture.source,
                               ['swiftsci'], fixture.root)

    def test_complete_passing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.inspect(self.fixture(Path(temporary)))
            self.assertEqual(result['status'], 'passed')
            self.assertTrue(result['coverage_complete'])
            self.assertEqual(result['passed'], 1)

    def test_failed_response_stays_failed_with_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.inspect(self.fixture(Path(temporary), failed=True))
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['coverage_complete'])
            self.assertEqual(result['failures'][0]['worker_error'], 'Output mismatch')
            self.assertEqual(result['infrastructure_errors'], [])

    def test_worker_crash_is_not_numerical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.inspect(self.fixture(Path(temporary), failed=True, missing_response=True))
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['coverage_complete'])
            self.assertEqual(result['failures'], [])
            self.assertEqual(len(result['infrastructure_errors']), 1)

    def test_incomplete_duplicate_and_source_drift_rejected(self):
        for mutation in ('missing', 'duplicate', 'source'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                if mutation == 'missing':
                    fixture.run['events'] = []
                elif mutation == 'duplicate':
                    fixture.run['events'] *= 2
                else:
                    fixture.run['plan']['source'] = dict(fixture.source, commit='z'*40)
                self.persist(fixture)
                with self.assertRaises(ContractError):
                    self.inspect(fixture)

    def test_modified_response_and_certificate_rejected(self):
        for mutation in ('response', 'certificate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                path = fixture.response_path if mutation == 'response' else fixture.directory/'certificate.json'
                value = read_json(path)
                if mutation == 'response':
                    value['samples'][0]['maximum_absolute_error'] = 1
                else:
                    value['run_sha256'] = '0'*64
                write_json(path, value)
                with self.assertRaises(ContractError):
                    self.inspect(fixture)

    def test_every_request_field_is_bound_to_the_resolved_contract(self):
        changes = {'schema_version': 2, 'case_key': '0'*64, 'operation': 'mean',
                   'dataset_kind': 'mixed-table-v1', 'input_path': '/wrong/input',
                   'input_sha256': '0'*64, 'input_bytes': 1,
                   'expected_path': '/wrong/expected', 'expected_sha256': '0'*64,
                   'rows': 2, 'warmups': 1, 'samples': 2, 'atol': 1, 'rtol': 1,
                   'unexpected': True}
        for key, changed in changes.items():
            with self.subTest(field=key), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                request = read_json(fixture.request_path)
                request[key] = changed
                write_json(fixture.request_path, request)
                with self.assertRaisesRegex(ContractError, 'Request contract differs'):
                    self.inspect(fixture)
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            request = read_json(fixture.request_path)
            del request['warmups']
            write_json(fixture.request_path, request)
            with self.assertRaisesRegex(ContractError, 'Request contract differs'):
                self.inspect(fixture)

    def test_changed_expected_bytes_rejected_even_with_matching_request_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            corrupted = binary([2])
            fixture.expected_path.write_bytes(corrupted)
            request = read_json(fixture.request_path)
            request['expected_sha256'] = digest(corrupted)
            write_json(fixture.request_path, request)
            with self.assertRaisesRegex(ContractError, 'Independent expected output differs'):
                self.inspect(fixture)

    def test_changed_input_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture.input_path.write_bytes(b'changed input')
            with self.assertRaises(ContractError):
                self.inspect(fixture)

    def test_resolved_manifest_and_workload_must_match_authoritative_records(self):
        for section, field, value in [('dataset', 'rows', 2), ('workload', 'atol', 1)]:
            with self.subTest(section=section), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                fixture.run['plan']['cases'][0][section][field] = value
                self.persist(fixture)
                with self.assertRaisesRegex(ContractError, 'Resolved contract differs'):
                    self.inspect(fixture)

    def test_malformed_or_stale_failed_response_is_infrastructure_failure(self):
        mutations = {'schema_version': 2, 'case_key': '0'*64, 'status': 'passed',
                     'samples': [{}], 'peak_rss_bytes': 0, 'engine_version': '',
                     'error': '', 'unexpected': True}
        for key, value in mutations.items():
            with self.subTest(field=key), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary), failed=True)
                response = read_json(fixture.response_path)
                response[key] = value
                # Matching a malformed record in the event must not legitimize it.
                fixture.run['events'][0]['failed_result'] = copy.deepcopy(response)
                write_json(fixture.response_path, response)
                self.persist(fixture)
                result = self.inspect(fixture)
                self.assertFalse(result['coverage_complete'])
                self.assertEqual(result['failures'], [])
                self.assertEqual(len(result['infrastructure_errors']), 1)
        for malformed in ('{"status":', 'null', '{"schema_version":1}'):
            with self.subTest(response=malformed), tempfile.TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary), failed=True)
                fixture.response_path.write_text(malformed)
                result = self.inspect(fixture)
                self.assertFalse(result['coverage_complete'])
                self.assertEqual(result['failures'], [])
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary), failed=True)
            stale = read_json(fixture.response_path)
            stale['error'] = 'An older failure with the same case key'
            write_json(fixture.response_path, stale)
            result = self.inspect(fixture)
            self.assertFalse(result['coverage_complete'])
            self.assertIn('Failed response differs', result['infrastructure_errors'][0]['artifact_error'])

    def execution_fixture(self, root):
        policy = dict(profiles=[dict(profile=name, tier='cpu') for name in ('one', 'two')])
        write_json(root/POLICY, policy)
        spec_path = root/'Benchmarks/Specs/profiles/one.json'
        write_json(spec_path, {'fixture': 'original'})
        profile = dict(cases=[{'id': 'tiny'}], batches=1, samples=1, warmups=0, timeout_seconds=5)
        success = dict(status='passed', coverage_complete=True, engine_cases=2, passed=2,
                       failures=[], infrastructure_errors=[])
        return policy, profile, spec_path, success

    def test_preparation_failure_does_not_skip_later_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy, profile, _, _ = self.execution_fixture(root)
            calls = []
            def invoke(args, **kwargs):
                calls.append(args)
                return SimpleNamespace(returncode=1)
            with patch('acceptance.source_identity', return_value={'commit': 'fixed'}), \
                 patch('acceptance.contract_identity', return_value='contract'), \
                 patch('acceptance.load_profile', return_value=profile):
                report = execute(root, policy, 'cpu', root/'run', 'worker', 'python', invoke)
            self.assertEqual(len(calls), 2)
            self.assertEqual([p['profile'] for p in report['profiles']], ['one', 'two'])
            self.assertTrue(all(p['status'] == 'infrastructure-error' for p in report['profiles']))
            self.assertFalse(report['coverage_complete'])
            self.assertEqual(report['status'], 'failed')

    def test_timeout_does_not_skip_later_profiles(self):
        for timeout_stage in ('prepare', 'run'):
            with self.subTest(stage=timeout_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                policy, profile, _, success = self.execution_fixture(root)
                calls = []
                def invoke(args, **kwargs):
                    stage, name = args[2], args[args.index('--profile')+1]
                    calls.append((name, stage))
                    if name == 'one' and stage == timeout_stage:
                        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
                    return SimpleNamespace(returncode=0)
                with patch('acceptance.source_identity', return_value={'commit': 'fixed'}), \
                     patch('acceptance.contract_identity', return_value='contract'), \
                     patch('acceptance.load_profile', return_value=profile), \
                     patch('acceptance.inspect_run', return_value=success) as inspector:
                    report = execute(root, policy, 'cpu', root/'run', 'worker', 'python', invoke)
                self.assertEqual(calls[-2:], [('two', 'prepare'), ('two', 'run')])
                self.assertEqual([p['status'] for p in report['profiles']], ['infrastructure-error', 'passed'])
                self.assertEqual(inspector.call_count, 1)
                self.assertEqual(inspector.call_args.args[-1], root)
                self.assertFalse(report['coverage_complete'])
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(read_json(root/'run/acceptance.json'), report)

    def test_mid_execution_spec_or_policy_drift_invalidates_the_frozen_contract(self):
        for changed_file in ('spec', 'policy'):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                policy, profile, spec_path, success = self.execution_fixture(root)
                names = POLICY+'\n'+str(spec_path.relative_to(root))+'\n'
                original_policy_hash = digest((root/POLICY).read_bytes())
                calls = []
                def invoke(args, **kwargs):
                    stage, name = args[2], args[args.index('--profile')+1]
                    calls.append((name, stage))
                    if stage == 'run':
                        path = spec_path if changed_file == 'spec' else root/POLICY
                        path.write_text(path.read_text()+'\n')
                    return SimpleNamespace(returncode=0)
                with patch('acceptance.source_identity', return_value={'commit': 'fixed'}), \
                     patch('acceptance.subprocess.check_output', return_value=names), \
                     patch('acceptance.load_profile', return_value=profile), \
                     patch('acceptance.inspect_run', return_value=success):
                    initial_contract = acceptance.contract_identity(root)
                    report = execute(root, policy, 'cpu', root/'run', 'worker', 'python', invoke)
                    self.assertNotEqual(acceptance.contract_identity(root), initial_contract)
                self.assertEqual(report['contract_sha256'], initial_contract)
                self.assertEqual(report['policy_sha256'], original_policy_hash)
                self.assertEqual(calls, [('one', 'prepare'), ('one', 'run')])
                self.assertTrue(all(p['status'] == 'infrastructure-error' for p in report['profiles']))
                self.assertTrue(all('Source or contract changed' in p['error'] for p in report['profiles']))
                self.assertFalse(report['coverage_complete'])
                self.assertEqual(report['status'], 'failed')

    def test_mid_execution_source_drift_invalidates_the_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy, profile, _, success = self.execution_fixture(root)
            source = {'commit': 'original'}
            calls = []
            def invoke(args, **kwargs):
                calls.append(args[2])
                if args[2] == 'run':
                    source['commit'] = 'changed'
                return SimpleNamespace(returncode=0)
            with patch('acceptance.source_identity', side_effect=lambda _: dict(source)), \
                 patch('acceptance.contract_identity', return_value='contract'), \
                 patch('acceptance.load_profile', return_value=profile), \
                 patch('acceptance.inspect_run', return_value=success):
                report = execute(root, policy, 'cpu', root/'run', 'worker', 'python', invoke)
            self.assertEqual(report['source'], {'commit': 'original'})
            self.assertEqual(calls, ['prepare', 'run'])
            self.assertTrue(all(p['status'] == 'infrastructure-error' for p in report['profiles']))
            self.assertFalse(report['coverage_complete'])

    def test_policy_covers_every_profile(self):
        load_policy(Path(acceptance.__file__).resolve().parents[2])


if __name__ == '__main__':
    unittest.main()
