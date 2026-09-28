import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
from regression_policy import evaluate


class RegressionPolicyTests(unittest.TestCase):
    def fixture(self):
        key = ('example', 'case', 'swiftsci', 0)
        binding = dict(case_key='a'*64, profile_key='b'*64, reference_basis='exact-binary64')
        expected = {key: binding, ('example', 'case', 'pandas', 0): binding}
        entry = dict(profile='example', case='case', engine='swiftsci', case_key='a'*64,
                     reference_basis='exact-binary64', classification='implementation-defect',
                     errors=['Output mismatch: 2.0 versus 1.0'], reason='Recorded defect',
                     evidence=['https://example.org/run'])
        baseline = dict(entries=[entry])
        failure = dict(case='case', engine='swiftsci', batch=0, case_key='a'*64,
                       worker_error=entry['errors'][0], error='Worker exited 1; see case-swiftsci-0.log')
        report = dict(status='failed', coverage_complete=True, profiles=[
            dict(profile='example', status='failed', coverage_complete=True,
                 engine_cases=2, passed=1, failures=[failure], infrastructure_errors=[])])
        return report, baseline, expected

    def test_known_failure_passes_policy_without_mutating_conformance(self):
        report, baseline, expected = self.fixture()
        before = copy.deepcopy(report)
        result = evaluate(report, baseline, expected)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['conformance_status'], 'failed')
        self.assertEqual(len(result['known_failures']), 1)
        self.assertEqual(result['known_failures'][0]['classification'], 'implementation-defect')
        self.assertEqual(report, before)

    def test_unlisted_or_different_error_or_contract_blocks(self):
        for mutation in ('unlisted', 'error', 'contract', 'engine'):
            with self.subTest(mutation=mutation):
                report, baseline, expected = self.fixture()
                if mutation == 'unlisted': baseline['entries'] = []
                elif mutation == 'error': report['profiles'][0]['failures'][0]['worker_error'] = 'Output mismatch: 3.0 versus 1.0'
                elif mutation == 'contract': baseline['entries'][0]['case_key'] = 'b'*64
                else: baseline['entries'][0]['engine'] = 'pandas'
                self.assertEqual(evaluate(report, baseline, expected)['status'], 'failed')

    def test_crash_with_known_response_cannot_pass(self):
        for error in ('Worker exited -11; see case-swiftsci-0.log', 'Command timed out'):
            with self.subTest(error=error):
                report, baseline, expected = self.fixture()
                report['profiles'][0]['failures'][0]['error'] = error
                self.assertEqual(evaluate(report, baseline, expected)['status'], 'failed')

    def test_profile_sampling_and_reference_basis_are_reviewed(self):
        from unittest.mock import patch
        from regression_policy import expected_cases, coverage_identity
        profile = dict(id='example', cases=[dict(id='case', dataset='data', workload='work')],
                       batches=1, samples=2, warmups=1, timeout_seconds=60)
        dataset = dict(reference_basis='original-decimal')
        def resolved():
            with patch('regression_policy.load_profile', return_value=profile), \
                 patch('regression_policy.load_manifest', return_value=dataset), \
                 patch('regression_policy.load_workload', return_value={}), \
                 patch('regression_policy.resolve_workload', return_value={}):
                return expected_cases(Path('/unused'), dict(profiles=[dict(tier='cpu', profile='example')]))
        before = coverage_identity(resolved())
        profile['samples'] = 1
        self.assertNotEqual(before, coverage_identity(resolved()))

    def test_unexpected_pass_requires_baseline_removal(self):
        report, baseline, expected = self.fixture()
        report.update(status='passed')
        report['profiles'][0].update(status='passed', passed=2, failures=[])
        result = evaluate(report, baseline, expected)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('unexpected-pass', [x['kind'] for x in result['blocking']])

    def test_coverage_and_infrastructure_cannot_be_allowlisted(self):
        for mutation in ('empty', 'missing-profile', 'coverage', 'count', 'duplicate', 'infrastructure', 'missing-response'):
            with self.subTest(mutation=mutation):
                report, baseline, expected = self.fixture()
                record = report['profiles'][0]
                if mutation == 'empty': report['profiles'] = []
                elif mutation == 'missing-profile': expected[('missing','case','swiftsci',0)] = expected[('example','case','swiftsci',0)]
                elif mutation == 'coverage': report['coverage_complete'] = False
                elif mutation == 'count': record['engine_cases'] = 1
                elif mutation == 'duplicate': record['failures'] *= 2
                elif mutation == 'infrastructure': record['infrastructure_errors'] = [{'error':'crash'}]
                else: record['failures'][0].pop('worker_error')
                self.assertEqual(evaluate(report, baseline, expected)['status'], 'failed')

    def test_all_batches_must_match_and_reference_role_is_preserved(self):
        report, baseline, expected = self.fixture()
        expected[('example','case','swiftsci',1)] = expected[('example','case','swiftsci',0)]
        report['profiles'][0].update(engine_cases=3, passed=2)
        self.assertEqual(evaluate(report, baseline, expected)['status'], 'failed')
        report, baseline, expected = self.fixture()
        entry = baseline['entries'][0]
        entry.update(engine='pandas', classification='comparator-accuracy')
        report['profiles'][0]['failures'][0].update(engine='pandas', error='Worker exited 1; see case-pandas-0.log')
        result = evaluate(report, baseline, expected)
        self.assertEqual(result['known_failures'][0]['engine'], 'pandas')
        self.assertEqual(result['known_failures'][0]['reference_basis'], 'exact-binary64')

    def test_no_failures_with_empty_inventory_passes(self):
        report, baseline, expected = self.fixture()
        baseline['entries'] = []
        report['status'] = 'passed'
        report['profiles'][0].update(status='passed', passed=2, failures=[])
        self.assertEqual(evaluate(report, baseline, expected)['status'], 'passed')

    def test_inventory_loader_rejects_stale_or_malformed_exceptions(self):
        import json
        import tempfile
        from regression_policy import BASELINE, coverage_identity, load_baseline
        from contracts import ContractError
        _, baseline, expected = self.fixture()
        baseline.update(schema_version=1, tier='cpu', coverage_sha256=coverage_identity(expected))
        for mutation in ('none', 'coverage', 'duplicate', 'classification', 'empty-errors', 'unknown-case', 'comparator', 'implementation-role', 'reference-basis'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                candidate = copy.deepcopy(baseline)
                if mutation == 'coverage': candidate['coverage_sha256'] = 'b'*64
                elif mutation == 'duplicate': candidate['entries'] *= 2
                elif mutation == 'classification': candidate['entries'][0]['classification'] = 'ignored'
                elif mutation == 'empty-errors': candidate['entries'][0]['errors'] = []
                elif mutation == 'unknown-case': candidate['entries'][0]['case'] = 'other'
                elif mutation == 'comparator': candidate['entries'][0]['engine'] = 'pandas'
                elif mutation == 'implementation-role': candidate['entries'][0]['classification'] = 'comparator-accuracy'
                elif mutation == 'reference-basis': candidate['entries'][0]['reference_basis'] = 'original-decimal'
                path = Path(temp) / BASELINE
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(candidate))
                if mutation == 'none': load_baseline(Path(temp), expected)
                else:
                    with self.assertRaises(ContractError): load_baseline(Path(temp), expected)
