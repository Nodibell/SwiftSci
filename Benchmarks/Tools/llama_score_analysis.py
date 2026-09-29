"""Shared-prefix candidate score diagnostics; neither runtime is a numeric oracle."""
import json
import math
import subprocess

import numpy as np
from pretrained_llama import ROOT, sha


def validate_request(request):
    if request.get('schema_version') != 1 or not request.get('template_date'):
        raise ValueError('Invalid score request version or template date')
    cases = request.get('cases', [])
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Score cases must be nonempty and uniquely named')
    for case in cases:
        prompt, forced, positions = case['prompt_tokens'], case['forced_tokens'], case['positions']
        if not prompt or len(prompt) + len(forced) >= 512 or len(forced) >= 32:
            raise ValueError('Probe is outside the short-context contract')
        if any(type(t) is not int or not 0 <= t < 128256 for t in prompt + forced):
            raise ValueError('Invalid probe token ID')
        if not positions or positions != sorted(set(positions)) or any(type(p) is not int or not 0 <= p <= len(forced) for p in positions):
            raise ValueError('Invalid score positions')


def make_request(comparison, reference, eos):
    from llama_cpp_compare import visible_tokens, token_agreement
    reference_cases = {c['id']: c for c in reference['cases']}
    if len(reference_cases) != len(reference['cases']) or set(reference_cases) != {c['id'] for c in comparison['cases']}:
        raise ValueError('Comparison/reference cases differ')
    request = dict(schema_version=1, template_date=reference['template_date'], cases=[])
    for result in comparison['cases']:
        case = reference_cases[result['id']]
        left = visible_tokens(case['samples'][0]['tokens'], eos)
        right = visible_tokens(result['samples'][0]['response']['tokens'], eos)
        shared = token_agreement(left, right)['common_prefix_tokens']
        if not left or not right:
            raise ValueError('No content tokens to probe')
        decision = min(shared, len(left) - 1, len(right) - 1)
        request['cases'].append(dict(id=case['id'], text=case['text'], rendered_prompt=case['rendered_prompt'],
            prompt_tokens=case['prompt_tokens'], forced_tokens=left[:decision],
            positions=sorted({0, decision}), decision_position=decision,
            first_difference=shared if shared < min(len(left), len(right)) else None))
    validate_request(request)
    return request


def normalized_logits(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError('Expected a finite score vector')
    shifted = values - np.max(values)
    return shifted - np.log(np.sum(np.exp(shifted)))


def candidate_metrics(mlx_logits, cpp_candidates, mlx_token, cpp_token):
    ids = [c['id'] for c in cpp_candidates]
    if len(set(ids)) != len(ids) or mlx_token not in ids or cpp_token not in ids:
        raise ValueError('Candidate list is duplicated or omits a competing token')
    if any(type(i) is not int or not 0 <= i < len(mlx_logits) for i in ids):
        raise ValueError('Candidate token out of bounds')
    cpp = {c['id']: c['logprob'] for c in cpp_candidates}
    if any(v is None or not math.isfinite(v) for v in cpp.values()):
        raise ValueError('Nonfinite candidate log probability')
    precise = normalized_logits(mlx_logits)
    residual = np.array([precise[i] - cpp[i] for i in ids])
    return dict(mlx_minus_cpp_choice_logit_margin=float(mlx_logits[mlx_token] - mlx_logits[cpp_token]),
                cpp_same_choice_logprob_margin=float(cpp[mlx_token] - cpp[cpp_token]),
                candidate_count=len(ids), candidate_logprob_rmse=float(np.sqrt(np.mean(residual**2))),
                candidate_logprob_max_abs_difference=float(np.max(np.abs(residual))))


def run_diagnostics(comparison, reference, model, python, output, eos):
    request = make_request(comparison, reference, eos)
    request_path = output / 'score-request.json'
    request_path.write_text(json.dumps(request, indent=2) + '\n')
    score_sets = {}
    for precision in ['bf16', 'head-float32', 'float32']:
        destination = output / ('scores-' + precision)
        with (output / ('scores-' + precision + '.log')).open('w') as log:
            subprocess.run([str(python.absolute()), str(ROOT / 'Benchmarks/Python/llama_score_worker.py'),
                '--model', str(model), '--request', str(request_path), '--precision', precision,
                '--output', str(destination)], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        metadata = json.loads((destination / 'scores.json').read_text())
        if metadata['status'] != 'completed' or metadata['request_sha256'] != sha(request_path) or metadata['scores_sha256'] != sha(destination / 'scores.npz'):
            raise ValueError('Score worker failed or artifacts changed')
        with np.load(destination / 'scores.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        score_sets[precision] = (metadata, arrays)
    results_by_id = {c['id']: c for c in comparison['cases']}
    reference_by_id = {c['id']: c for c in reference['cases']}
    report = dict(schema_version=1, status='completed-with-observations', purpose='shared-prefix-score-diagnostic',
                  request_sha256=sha(request_path), numerical_certificate=False,
                  scope='llama.cpp top-20 pre-sampling log probabilities; MLX full-vocabulary logits',
                  model_precision_experiment='BF16 checkpoint promoted exactly to Float32; not higher-precision original weights',
                  score_artifacts={p: sha(output / ('scores-' + p) / 'scores.npz') for p in score_sets},
                  precision_memory={p: {k: m[k] for k in ['retained_parameter_bytes', 'additional_output_weight_bytes', 'body_parameter_dtypes']} for p, (m, _) in score_sets.items()}, probes=[])
    for case in request['cases']:
        result = results_by_id[case['id']]
        baseline = reference_by_id[case['id']]['samples'][0]['tokens']
        bf_meta, bf_arrays = score_sets['bf16']
        fp_meta, fp_arrays = score_sets['float32']
        head_meta, head_arrays = score_sets['head-float32']
        bf_case = next(c for c in bf_meta['cases'] if c['id'] == case['id'])
        fp_case = next(c for c in fp_meta['cases'] if c['id'] == case['id'])
        head_case = next(c for c in head_meta['cases'] if c['id'] == case['id'])
        for position in case['positions']:
            cpp_scores = [s['response']['completion_probabilities'][position] for s in result['samples']]
            if not all(s == cpp_scores[0] for s in cpp_scores):
                raise ValueError('llama.cpp score replay differs')
            cpp = cpp_scores[0]
            if len(cpp['top_logprobs']) != 20:
                raise ValueError('Expected exactly 20 pre-sampling candidates')
            cpp_token = result['samples'][0]['response']['tokens'][position]
            if cpp['id'] != cpp_token:
                raise ValueError('llama.cpp score index differs from generated token')
            bf_probe = next(p for p in bf_case['probes'] if p['position'] == position)
            fp_probe = next(p for p in fp_case['probes'] if p['position'] == position)
            head_probe = next(p for p in head_case['probes'] if p['position'] == position)
            if bf_probe['native_argmax'] != baseline[position]:
                raise ValueError('Instrumented BF16 prediction differs from the uninstrumented baseline')
            key = bf_probe['array_key']
            bf_logits, fp_logits = bf_arrays[key + '-logits'], fp_arrays[key + '-logits']
            centered_delta = (bf_logits.astype(np.float64) - fp_logits) - np.mean(bf_logits.astype(np.float64) - fp_logits)
            report['probes'].append(dict(id=case['id'], position=position,
                is_first_difference=position == case['first_difference'],
                mlx_bf16_token=baseline[position], cpp_token=cpp_token,
                mlx_bf16_raw_argmax=bf_probe['raw_argmax'], mlx_float32_token=fp_probe['native_argmax'],
                mlx_float32_matches_cpp=fp_probe['native_argmax'] == cpp_token,
                mlx_head_float32_token=head_probe['native_argmax'],
                mlx_head_float32_matches_cpp=head_probe['native_argmax'] == cpp_token,
                head_float32_top_tie_count=head_probe['raw_top_tie_count'],
                head_float32=candidate_metrics(head_arrays[key + '-logits'], cpp['top_logprobs'], baseline[position], cpp_token),
                bf16_top_tie_count=bf_probe['raw_top_tie_count'],
                float32_top_tie_count=fp_probe['raw_top_tie_count'],
                float32_rounded_bf16_token=fp_probe['rounded_bf16_argmax'],
                float32_rounded_bf16_top_tie_count=fp_probe['rounded_bf16_top_tie_count'],
                float32_rounding_reproduces_bf16_choice=fp_probe['rounded_bf16_argmax'] == baseline[position],
                bf16=candidate_metrics(bf_logits, cpp['top_logprobs'], baseline[position], cpp_token),
                float32=candidate_metrics(fp_logits, cpp['top_logprobs'], baseline[position], cpp_token),
                bf16_to_float32_centered_logit_rmse=float(np.sqrt(np.mean(centered_delta**2))),
                bf16_to_float32_centered_logit_max_abs=float(np.max(np.abs(centered_delta))),
                normalization_changed_argmax=bf_probe['raw_argmax'] != bf_probe['native_argmax'],
                cpp_candidates=cpp['top_logprobs'], mlx_bf16_top_tokens=bf_probe['top_tokens'],
                mlx_float32_top_tokens=fp_probe['top_tokens']))
    (output / 'score-analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    return dict(status=report['status'], probe_count=len(report['probes']), numerical_certificate=False,
                float32_agrees_with_cpp=sum(p['mlx_float32_matches_cpp'] for p in report['probes']),
                head_float32_agrees_with_cpp=sum(p['mlx_head_float32_matches_cpp'] for p in report['probes']))
