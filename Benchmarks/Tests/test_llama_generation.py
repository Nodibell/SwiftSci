import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from llama_generation_cases import build_cases, task_result, same_output, summarize, validate_run


class GenerationCasesTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / 'Benchmarks/Fixtures/pretrained'
        self.smoke = json.loads((directory / 'prompts.json').read_text())
        self.suite = json.loads((directory / 'generation-suite.json').read_text())

    def test_records_are_reproducible_with_checked_answers(self):
        cases = build_cases(self.smoke, self.suite)
        self.assertEqual(cases, build_cases(self.smoke, self.suite))
        self.assertEqual(len(cases), 7)
        case = cases[-1]
        self.assertEqual(case['expected_json'], dict(sensor='S0256', value=402))
        self.assertIn('sensor=S0256 value=402 unit=C', case['text'])
        self.assertEqual(case['text'].count('unit=C'), 512)

    def test_invalid_bounds_or_batch_members_fail(self):
        for key, value in [('max_new_tokens', 257), ('record_counts', [16, 16]), ('repetitions', 1)]:
            bad = dict(self.suite, **{key: value})
            with self.assertRaises(ValueError): build_cases(self.smoke, bad)
        bad = copy.deepcopy(self.suite)
        bad['batches'][0]['cases'][0] = 'missing'
        with self.assertRaises(ValueError): build_cases(self.smoke, bad)

    def test_task_check_does_not_confuse_agreement_with_correctness(self):
        case = dict(expected_json=dict(sensor='S0001', value=48))
        self.assertEqual(task_result(case, '{"value":48,"sensor":"S0001"}')['status'], 'passed')
        for text in ['{"value":49,"sensor":"S0001"}', '{"value":48.0,"sensor":"S0001"}',
                     '```json\n{"value":48,"sensor":"S0001"}\n```', '{}', 'null']:
            self.assertEqual(task_result(case, text)['status'], 'failed')
        self.assertEqual(task_result({}, 'anything')['status'], 'not-graded')

    def test_matching_tokens_require_matching_stop_reason(self):
        self.assertFalse(same_output(dict(tokens=[1], finish_reason='stop'), dict(tokens=[1], finish_reason='length')))

    def reports(self):
        sample = dict(tokens=[1, 2], finish_reason='stop', task=dict(status='passed'))
        return [dict(precision=m, status='completed', prompts={'a': dict(tokens=[3])},
                     singles={'a': [copy.deepcopy(sample) for _ in range(3)]},
                     batches={'pair': [dict(outputs={'a': copy.deepcopy(sample)}) for _ in range(3)]})
                for m in ['bf16', 'head-float32', 'shared-float32', 'float32']]

    def test_differences_are_visible_not_counted_as_conformance(self):
        runs = self.reports()
        runs[2]['batches']['pair'][1]['outputs']['a']['tokens'] = [1, 9]
        result = summarize(runs)
        self.assertEqual(result['policies']['shared-float32']['repeatable_trajectories'], 1)
        self.assertFalse(result['shared_vs_separate'][-1]['equal'])
        with self.assertRaises(ValueError): summarize(runs[:-1])
        runs[0]['singles']['a'].pop()
        with self.assertRaises(ValueError): summarize(runs)

    def test_expected_inventory_rejects_consistently_missing_cases(self):
        run = self.reports()[0]
        suite = dict(max_prompt_tokens=100, repetitions=3, batches=[dict(id='pair', cases=['a'])])
        validate_run(run, [dict(id='a')], suite)
        with self.assertRaises(ValueError):
            validate_run(run, [dict(id='a'), dict(id='missing')], suite)
        run['batches']['pair'][0]['outputs'] = {}
        with self.assertRaises(ValueError):
            validate_run(run, [dict(id='a')], suite)
