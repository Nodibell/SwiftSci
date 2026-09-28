"""Reviewed CPU failure inventory, separate from numerical conformance."""
from contracts import fields, identity, require, read_json, load_profile, load_workload
from datasets import load_manifest, resolve_workload

BASELINE = 'Benchmarks/Specs/known-failures.json'
CLASSIFICATIONS = {'implementation-defect', 'input-representation', 'comparator-accuracy'}


def expected_cases(root, policy):
    expected = {}
    for entry in policy['profiles']:
        if entry['tier'] != 'cpu':
            continue
        profile = load_profile(root, entry['profile'])
        for case in profile['cases']:
            dataset = load_manifest(root, case['dataset'])
            workload = resolve_workload(dataset, load_workload(root, case['workload']))
            case_key = identity(dict(case=case, dataset=dataset, workload=workload))
            for engine in ('swiftsci', 'pandas'):
                for batch in range(profile['batches']):
                    expected[(profile['id'], case['id'], engine, batch)] = dict(
                        case_key=case_key, profile_key=identity(profile),
                        reference_basis=dataset.get('reference_basis', 'fixture-defined'))
    return expected


def coverage_identity(expected):
    return identity([dict(profile=k[0], case=k[1], engine=k[2], batch=k[3], **v)
                     for k, v in sorted(expected.items())])


def load_baseline(root, expected):
    baseline = read_json(root / BASELINE)
    fields(baseline, ['schema_version', 'tier', 'coverage_sha256', 'entries'])
    require(baseline['schema_version'] == 1 and baseline['tier'] == 'cpu', 'Unsupported failure baseline')
    require(baseline['coverage_sha256'] == coverage_identity(expected),
            'CPU coverage/contracts changed; review the failure baseline')
    require(isinstance(baseline['entries'], list), 'Invalid baseline entries')
    keys = set()
    for entry in baseline['entries']:
        fields(entry, ['profile', 'case', 'engine', 'case_key', 'classification',
                       'reference_basis', 'errors', 'reason', 'evidence'])
        require(all(isinstance(entry[k], str) and entry[k] for k in
                    ('profile', 'case', 'engine', 'case_key', 'classification', 'reference_basis', 'reason')),
                'Invalid baseline identity or explanation')
        key = (entry['profile'], entry['case'], entry['engine'])
        require(key not in keys, 'Duplicate failure baseline entry')
        keys.add(key)
        binding = expected.get((*key, 0), {})
        require(binding.get('case_key') == entry['case_key'], 'Stale failure baseline contract')
        require(binding.get('reference_basis') == entry['reference_basis'], 'Reference basis differs from dataset')
        require(entry['classification'] in CLASSIFICATIONS, 'Unknown failure classification')
        require(entry['classification'] == 'input-representation' or
                (entry['engine'], entry['classification']) in
                {('swiftsci', 'implementation-defect'), ('pandas', 'comparator-accuracy')},
                'Failure classification differs from engine role')
        for field in ('errors', 'evidence'):
            values = entry[field]
            require(isinstance(values, list) and values and
                    all(isinstance(v, str) and v for v in values) and len(values) == len(set(values)),
                    f'Invalid baseline {field}')
    return baseline


def evaluate(report, baseline, expected):
    """Consume the verified acceptance summary; never edit runs or certificates."""
    blocking, known = [], []
    def block(kind, **detail):
        blocking.append(dict(kind=kind, **detail))
    inventory = {(e['profile'], e['case'], e['engine']): e for e in baseline['entries']}
    profiles = report['profiles']
    names = [p['profile'] for p in profiles]
    if len(names) != len(set(names)) or set(names) != {k[0] for k in expected}:
        block('profile-coverage')
    if report['coverage_complete'] is not True:
        block('incomplete-coverage')
    observed = {}
    for profile in profiles:
        name = profile['profile']
        count = sum(k[0] == name for k in expected)
        failures = profile.get('failures', [])
        if (profile.get('coverage_complete') is not True or profile.get('infrastructure_errors') or
                profile.get('status') not in ('passed', 'failed')):
            block('infrastructure', profile=name)
        if (profile.get('engine_cases') != count or
                profile.get('passed', -1) + len(failures) != count):
            block('case-coverage', profile=name)
        if profile.get('status') != ('failed' if failures else 'passed'):
            block('inconsistent-status', profile=name)
        for failure in failures:
            key = (name, failure['case'], failure['engine'], failure['batch'])
            if key in observed or expected.get(key, {}).get('case_key') != failure.get('case_key'):
                block('invalid-failure-identity', profile=name, case=failure['case'])
            observed[key] = failure
            entry = inventory.get(key[:3])
            token = f'{key[1]}-{key[2]}-{key[3]}'
            if failure.get('error') != f'Worker exited 1; see {token}.log':
                block('unexpected-worker-exit', profile=name, **failure)
            if entry is None:
                block('unclassified-failure', profile=name, **failure)
            elif (entry['case_key'] != failure.get('case_key') or
                  failure.get('worker_error') not in entry['errors']):
                block('changed-failure', profile=name, **failure)
            else:
                known.append(dict(profile=name, **failure, classification=entry['classification'],
                                  reference_basis=entry['reference_basis'], reason=entry['reason']))
    for key, entry in inventory.items():
        applicable = [k for k in expected if k[:3] == key]
        if not applicable or any(expected[k]['case_key'] != entry['case_key'] for k in applicable):
            block('stale-baseline', profile=key[0], case=key[1], engine=key[2])
        for execution in applicable:
            if execution not in observed:
                block('unexpected-pass', profile=key[0], case=key[1], engine=key[2], batch=execution[3])
    if report['status'] != ('failed' if observed or not report['coverage_complete'] else 'passed'):
        block('inconsistent-conformance-status')
    return dict(schema_version=1, status='failed' if blocking else 'passed',
                conformance_status=report['status'], known_failures=known, blocking=blocking)
