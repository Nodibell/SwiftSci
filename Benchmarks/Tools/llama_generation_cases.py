"""Deterministic inputs and observations for opt-in free-running Llama tests."""
import json


def build_cases(smoke, suite):
    if suite.get('schema_version') != 1 or smoke.get('schema_version') != 1:
        raise ValueError('Unsupported generation fixture version')
    for name, low, high in [('max_prompt_tokens', 1, 16384), ('max_new_tokens', 1, 256),
                            ('repetitions', 3, 3), ('prefill_step_size', 1, 2048)]:
        value = suite.get(name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('Invalid generation bound: ' + name)
    counts = suite['record_counts']
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or not 2 <= n <= 512 for n in counts):
        raise ValueError('Invalid sensor record counts')
    cases = [dict(c) for c in smoke['cases']]
    for count in counts:
        records = [dict(sensor=f'S{i:04d}', value=(i * 37 + 11) % 1009) for i in range(count)]
        target = records[count // 2]
        text = 'Read these sensor observations. Each record is independent.\n'
        text += '\n'.join(f"sensor={r['sensor']} value={r['value']} unit=C" for r in records)
        text += f"\nReturn only a JSON object with exactly two fields named \"sensor\" and \"value\" for {target['sensor']}. Do not add markdown or explanations."
        cases.append(dict(id=f'sensor-records-{count}', text=text, expected_json=target))
    ids = [c['id'] for c in cases]
    if not cases or len(set(ids)) != len(ids) or any(not c['text'] for c in cases):
        raise ValueError('Generation cases must have unique IDs and nonempty text')
    batches = suite['batches']
    if not batches or len({b['id'] for b in batches}) != len(batches):
        raise ValueError('Batch IDs must be unique')
    for batch in batches:
        members = batch['cases']
        if not 2 <= len(members) <= 4 or len(set(members)) != len(members) or any(k not in ids for k in members):
            raise ValueError('Invalid batch membership')
    return cases


def task_result(case, text):
    if 'expected_json' not in case:
        return dict(status='not-graded')
    try:
        value = json.loads(text)
    except ValueError:
        return dict(status='failed', reason='Output is not a single JSON value')
    expected = case['expected_json']
    passed = (isinstance(value, dict) and set(value) == set(expected)
              and value.get('sensor') == expected['sensor']
              and type(value.get('value')) is int and value['value'] == expected['value'])
    return dict(status='passed' if passed else 'failed', expected=expected, actual=value)


def same_output(left, right):
    return left['tokens'] == right['tokens'] and left['finish_reason'] == right['finish_reason']


def validate_run(run, cases, suite):
    ids = {c['id'] for c in cases}
    groups = {g['id']: set(g['cases']) for g in suite['batches']}
    if run['status'] != 'completed' or set(run['singles']) != ids or set(run['prompts']) != ids or set(run['batches']) != set(groups):
        raise ValueError('Worker omitted a required generation case or batch')
    for key in ids:
        tokens = run['prompts'][key]['tokens']
        if not 0 < len(tokens) <= suite['max_prompt_tokens'] or any(type(t) is not int or not 0 <= t < 128256 for t in tokens):
            raise ValueError('Worker prompt tokens violate the checkpoint contract')
        if len(run['singles'][key]) != suite['repetitions']:
            raise ValueError('Worker omitted single-generation repetitions')
    for key, members in groups.items():
        samples = run['batches'][key]
        if len(samples) != suite['repetitions'] or any(set(s['outputs']) != members for s in samples):
            raise ValueError('Worker omitted batch outputs or repetitions')


def summarize(runs):
    expected_modes = {'bf16', 'head-float32', 'shared-float32', 'float32'}
    if len(runs) != 4 or {r['precision'] for r in runs} != expected_modes:
        raise ValueError('Expected all four precision reports')
    keyed = {r['precision']: r for r in runs}
    baseline = keyed['bf16']
    expected_cases = set(baseline['singles'])
    expected_batches = set(baseline['batches'])
    summary = dict(policies={}, shared_vs_separate=[], batch_vs_single=[])
    for mode, run in keyed.items():
        if run['status'] != 'completed' or set(run['singles']) != expected_cases or set(run['batches']) != expected_batches:
            raise ValueError('Incomplete generation report')
        repeats, graded = [], []
        for case_id in expected_cases:
            if run['prompts'][case_id]['tokens'] != baseline['prompts'][case_id]['tokens']:
                raise ValueError('Prompt tokens changed across precision policies')
            samples = run['singles'][case_id]
            if len(samples) != 3:
                raise ValueError('Missing single-generation repetitions')
            repeats.append(all(same_output(s, samples[0]) for s in samples))
            graded.extend(s['task']['status'] for s in samples if s['task']['status'] != 'not-graded')
        for batch_id, samples in run['batches'].items():
            if len(samples) != 3 or any(set(s['outputs']) != set(samples[0]['outputs']) for s in samples):
                raise ValueError('Incomplete batch-generation repetitions')
            if set(samples[0]['outputs']) != set(baseline['batches'][batch_id][0]['outputs']):
                raise ValueError('Batch membership changed across policies')
            for case_id in samples[0]['outputs']:
                repeats.append(all(same_output(s['outputs'][case_id], samples[0]['outputs'][case_id]) for s in samples))
                summary['batch_vs_single'].append(dict(precision=mode, batch=batch_id, case=case_id,
                    equal=all(same_output(s['outputs'][case_id], run['singles'][case_id][0]) for s in samples)))
                graded.extend(s['outputs'][case_id]['task']['status'] for s in samples
                              if s['outputs'][case_id]['task']['status'] != 'not-graded')
        summary['policies'][mode] = dict(repeatable_trajectories=sum(repeats), trajectories=len(repeats),
            task_checks_passed=graded.count('passed'), task_checks=len(graded),
            length_limited_single_samples=sum(s['finish_reason'] == 'length' for a in run['singles'].values() for s in a))
    left, right = keyed['head-float32'], keyed['shared-float32']
    for case_id in sorted(expected_cases):
        summary['shared_vs_separate'].append(dict(kind='single', case=case_id,
            equal=all(same_output(a, b) for a, b in zip(left['singles'][case_id], right['singles'][case_id]))))
    for batch_id in sorted(expected_batches):
        for case_id in left['batches'][batch_id][0]['outputs']:
            summary['shared_vs_separate'].append(dict(kind='batch', batch=batch_id, case=case_id,
                equal=all(same_output(a['outputs'][case_id], b['outputs'][case_id])
                          for a, b in zip(left['batches'][batch_id], right['batches'][batch_id]))))
    return summary
