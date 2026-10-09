"""ML Program provenance and output-backing observations for pool_trial.py."""
import json
from concurrency import digest


def verify_models(models, prepared):
    manifests = {}
    for rows in (1024, 8192):
        folder = models / f'models-{rows}'
        manifest = json.loads((folder / 'manifest.json').read_text())
        if manifest['fixture_sha256'] != digest(prepared / f'fixture-512-{rows}.json'):
            raise RuntimeError('Program weights/input provenance differs from the reference fixture')
        if (manifest['rows'], manifest['width'], manifest['outputs'], manifest['program_compute_precision']) != (rows, 54, 7, 'float16'):
            raise RuntimeError('Unexpected trained program precision or dimensions')
        for name, expected in manifest['files'].items():
            if digest(folder / name) != expected:
                raise RuntimeError('Model artifact changed: ' + name)
        manifests[str(rows)] = manifest
    return manifests


def validate_backing_counts(record):
    if record['request']['execution'] != 'persistent':
        if record.get('poolPredictions') is not None or record.get('poolOutputBackingIdentityMatches') is not None:
            raise ValueError('Fresh predictor claimed pool reuse statistics')
        return
    expected = 2 + sum(c['latency']['count'] for c in record['callers'])
    count = record.get('poolPredictions')
    matches = record.get('poolOutputBackingIdentityMatches')
    if type(count) is not int or count != expected:
        raise ValueError('Missing successful pool predictions or invalid requests counted as successful')
    if type(matches) is not int or not 0 <= matches <= count:
        raise ValueError('Invalid output backing observation count')


def backing_report(records):
    lines = ['', '## Output-backing observations', '',
             'Identity matches prove that Core ML returned the supplied output object. They do not prove '
             'the absence of internal copies. A different object can still share storage; this counter '
             'does not detect that case. Adoption is observed, not required for numerical correctness.', '',
             '| Policy | Rows | Slots | Successful predictions including preflight | Output identity matches |',
             '|---|---:|---:|---:|---:|']
    for r in records:
        q = r['request']
        if q['execution'] == 'persistent':
            lines.append(f"| {q['policy']} | {q['rows']} | {q['admissionSlots']} | "
                         f"{r['poolPredictions']} | {r['poolOutputBackingIdentityMatches']} |")
    return '\n'.join(lines) + '\n'
