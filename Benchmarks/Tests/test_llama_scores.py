import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from llama_score_analysis import candidate_metrics, make_request, normalized_logits, validate_request


def case():
    return dict(id='fixture', text='fixture', rendered_prompt='fixture', prompt_tokens=[1, 2],
                forced_tokens=[3, 4], positions=[0, 2])


class LlamaScoreTests(unittest.TestCase):
    def test_log_probabilities_are_shift_invariant_and_normalized(self):
        logits = np.array([10001., 10002., 9999.])
        result = normalized_logits(logits)
        np.testing.assert_allclose(result, normalized_logits(logits - 10000), rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(np.exp(result).sum()), 1., places=14)
        for invalid in [[1., float('nan')], [float('inf'), 1.], [[1., 2.]]]:
            with self.assertRaises(ValueError):
                normalized_logits(invalid)

    def test_rank_flip_has_opposite_margins(self):
        mlx = np.array([2., 1., 0.])
        cpp = normalized_logits([1., 2., 0.])
        candidates = [dict(id=i, logprob=float(cpp[i])) for i in [1, 0, 2]]
        result = candidate_metrics(mlx, candidates, 0, 1)
        self.assertEqual(result['mlx_minus_cpp_choice_logit_margin'], 1.)
        self.assertEqual(result['cpp_same_choice_logprob_margin'], -1.)
        self.assertAlmostEqual(result['candidate_logprob_rmse'], math.sqrt(2/3), places=14)

    def test_invalid_or_incomplete_candidate_scores_fail(self):
        logits = np.array([2., 1., 0.])
        for candidates in [[dict(id=0, logprob=-1.)],
                           [dict(id=0, logprob=-1.), dict(id=0, logprob=-2.)],
                           [dict(id=0, logprob=float('nan')), dict(id=1, logprob=-2.)],
                           [dict(id=0, logprob=-1.), dict(id=1, logprob=None)],
                           [dict(id=0, logprob=-1.), dict(id=1, logprob=-2.), dict(id=-1, logprob=-3.)]]:
            with self.assertRaises(ValueError):
                candidate_metrics(logits, candidates, 0, 1)

    def test_requests_require_valid_tokens_and_positions(self):
        request = dict(schema_version=1, template_date='29 Sep 2026', cases=[case()])
        validate_request(request)
        for change in [dict(prompt_tokens=[]), dict(prompt_tokens=[True]), dict(forced_tokens=[128256]),
                       dict(positions=[3]), dict(positions=[0, 0]), dict(positions=[2, 0]),
                       dict(forced_tokens=[1]*32), dict(prompt_tokens=[1]*510)]:
            changed = copy.deepcopy(request)
            changed['cases'][0].update(change)
            with self.assertRaises(ValueError):
                validate_request(changed)
        request['cases'].append(case())
        with self.assertRaises(ValueError):
            validate_request(request)

    def test_probe_stops_before_first_different_token(self):
        ref_case = case()
        ref_case['samples'] = [dict(tokens=[9, 8, 7])]
        comparison = dict(cases=[dict(id='fixture', samples=[dict(response=dict(tokens=[9, 8, 6]))])])
        reference = dict(template_date='29 Sep 2026', cases=[ref_case])
        request = make_request(comparison, reference, [128009])
        row = request['cases'][0]
        self.assertEqual(row['forced_tokens'], [9, 8])
        self.assertEqual(row['positions'], [0, 2])
        self.assertEqual(row['first_difference'], 2)
        comparison['cases'][0]['samples'][0]['response']['tokens'] = [9, 8, 7]
        row = make_request(comparison, reference, [128009])['cases'][0]
        self.assertEqual(row['positions'], [0, 2])
        self.assertIsNone(row['first_difference'])
