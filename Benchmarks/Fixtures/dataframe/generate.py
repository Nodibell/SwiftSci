#!/usr/bin/env python3
"""Regenerate original semantic inputs and exact scalar references, without libraries under test."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from contracts import digest, write_json
from dataframe_fixtures import validate_input, expected_values


def cases():
    def col(name, dtype, values):
        return {'name': name, 'type': dtype, 'values': values}
    exact = [col('id', 'int64', ['-9223372036854775808', '9007199254740993', None, '9223372036854775807', '-9007199254740993', '9007199254740993']),
             col('label', 'utf8', ['α', 'null', '', 'a|b', '__null__', '🚀'])]
    floats = [col('value', 'float64', [None, 'NaN', '-Inf', -0.0, 0.0, 1.5, '+Inf']),
              col('row', 'int64', list(map(str, range(7))))]
    numeric = [col('x', 'float64', [1.25, None, -2.0]),
               col('id', 'int64', ['9007199254740993', '-7', None]),
               col('flag', 'bool', [True, False, None])]
    groups = [col('k1', 'utf8', ['a|b', 'a', None, 'null', '__null__', 'a|b', None, 'a']),
              col('k2', 'utf8', ['c', 'b|c', '', '', '', 'c', '', 'b|c']),
              col('v', 'float64', [1.0, None, 'NaN', 4.0, None, 6.0, None, 8.0])]
    integer_groups = [col('key', 'int64', ['9007199254740992', '9007199254740993', None, '9007199254740992', None]),
                      col('v', 'int64', ['1', None, '3', '4', None]),
                      col('other', 'float64', [None, 2.0, 'NaN', None, None])]
    specs = [
        ('gather-int64', 'gather', exact, {'indices': [3, 2, 1, 0, 4, 2, 1]}),
        ('gather-empty', 'gather', exact, {'indices': []}),
        ('select-order', 'select', exact, {'columns': ['label', 'id']}),
        ('sort-int64-asc', 'sort', exact, {'column': 'id', 'ascending': True}),
        ('sort-int64-desc', 'sort', exact, {'column': 'id', 'ascending': False}),
        ('filter-int64-exact', 'filter', exact, {'column': 'id', 'predicate': 'gt', 'value': '9007199254740992'}),
        ('filter-empty', 'filter', exact, {'column': 'id', 'predicate': 'gt', 'value': '9223372036854775807'}),
        ('matrix-nullable', 'matrix', numeric, {'columns': ['flag', 'id', 'x'], 'target': 'x'}),
        ('matrix-nonsquare', 'matrix', numeric, {'columns': ['x', 'id'], 'target': 'flag'}),
        ('matrix-duplicate', 'matrix', numeric, {'columns': ['id', 'id'], 'target': 'id'}),
        ('matrix-empty-selection', 'matrix', numeric, {'columns': [], 'target': 'x'}),
        ('replace-isolation', 'replace', exact, {'column': 'id', 'values': ['42', None, '1', '-2', '3', '4']}),
        ('group-string-components', 'group-count', groups, {'columns': ['k1', 'k2'], 'value': 'v'}),
        ('group-int64', 'group-count', integer_groups, {'columns': ['key'], 'value': 'v'}),
    ]
    for predicate in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isNull', 'isNotNull'):
        specs.append(('filter-float-' + predicate.lower(), 'filter', floats,
                      {'column': 'value', 'predicate': predicate, 'value': None if predicate.startswith('is') else 0.0}))
    sortable = [col('value', 'float64', [2.0, None, -0.0, 0.0, 2.0, '-Inf', '+Inf']), floats[1]]
    for ascending in (True, False):
        specs.append(('sort-float-' + ('asc' if ascending else 'desc'), 'sort', sortable,
                      {'column': 'value', 'ascending': ascending}))
    return [(name, {'operation': 'dataframe-semantics', 'action': action, 'columns': columns, 'parameters': params}) for name, action, columns, params in specs]


def generate():
    source = Path(__file__).read_bytes()
    source_path = str(Path(__file__).relative_to(ROOT))
    entries = []
    for slug, payload in cases():
        name = 'dataframe-' + slug
        rows = len(payload['columns'][0]['values'])
        validate_input(payload, 'dataframe-semantics', rows)
        path = ROOT / f'Benchmarks/Fixtures/dataframe/inputs/{name}.json'
        write_json(path, payload)
        raw = path.read_bytes()
        reference = {'operation': 'dataframe-semantics',
                     'inputIdentity': {'sha256': digest(raw), 'bytes': len(raw)},
                     'source': {'sha256': digest(source)}, 'model': {'observations': rows},
                     'independentReference': {'values': expected_values(payload)}}
        refpath = ROOT / f'Benchmarks/Fixtures/dataframe/references/{name}.json'
        write_json(refpath, reference)
        ref = refpath.read_bytes()
        manifest = {'schema_version': 1, 'id': name, 'kind': 'numerical-fixture-v1',
                    'rows': rows, 'sha256': digest(raw), 'size_bytes': len(raw),
                    'source': 'Original bounded fixtures; see Benchmarks/Fixtures/dataframe/README.md',
                    'license': 'MIT, repository LICENSE', 'generator_version': 1,
                    'fixture': str(path.relative_to(ROOT)), 'source_fixture': source_path,
                    'source_sha256': digest(source), 'source_size_bytes': len(source),
                    'reference_fixture': str(refpath.relative_to(ROOT)),
                    'reference_sha256': digest(ref), 'reference_size_bytes': len(ref),
                    'reference_basis': 'dataframe-reference-v1', 'operation': 'dataframe-semantics',
                    'tolerances': {'atol': 0, 'rtol': 0}}
        write_json(ROOT / f'Benchmarks/Specs/datasets/{name}.json', manifest)
        entries.append({'id': name, 'dataset': name, 'workload': 'dataframe-semantics-v1'})
    write_json(ROOT / 'Benchmarks/Specs/profiles/dataframe-conformance.json',
               {'schema_version': 1, 'id': 'dataframe-conformance', 'warmups': 1, 'samples': 2,
                'batches': 1, 'timeout_seconds': 120, 'cases': entries})


if __name__ == '__main__':
    generate()
