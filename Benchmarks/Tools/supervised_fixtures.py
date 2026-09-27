"""Boundaries for frozen supervised splits and complete numerical outputs."""
import math
from contracts import fields, require

OPERATIONS = {'supervised-scale', 'supervised-ols-cpu'}
SPLITS = ('train', 'validation', 'test')


def validate_input(payload, operation, rows):
    fields(payload, ['operation', 'row_ids', 'feature_names', 'features', 'targets', 'splits'])
    require(operation in OPERATIONS and payload['operation'] == operation, 'Supervised operation mismatch')
    ids, names = payload['row_ids'], payload['feature_names']
    for values in [ids, names]:
        require(isinstance(values, list) and values and all(isinstance(v, str) and v.strip() for v in values), 'Invalid supervised identities')
        require(len(set(values)) == len(values), 'Duplicate supervised identities')
    require(len(ids) == rows, 'Supervised row count mismatch')
    matrix = payload['features']
    require(isinstance(matrix, list) and len(matrix) == rows, 'Invalid supervised matrix')
    require(all(isinstance(row, list) and len(row) == len(names) and all(type(v) in (int, float) and math.isfinite(v) for v in row) for row in matrix), 'Invalid supervised feature values')
    targets = payload['targets']
    require(isinstance(targets, list) and len(targets) == rows and all(type(v) in (int, float) and math.isfinite(v) for v in targets), 'Invalid supervised targets')
    fields(payload['splits'], list(SPLITS))
    flat = []
    owners = {}
    for name in SPLITS:
        indices = payload['splits'][name]
        require(isinstance(indices, list) and indices and all(type(i) is int and 0 <= i < rows for i in indices), 'Invalid split indices')
        flat += indices
        for i in indices:
            key = tuple(matrix[i])
            require(key not in owners or owners[key] == name, 'Identical features cross split boundary')
            owners[key] = name
    require(len(flat) == rows and len(set(flat)) == rows, 'Splits must be disjoint and exhaustive')
    require(len(payload['splits']['train']) > len(names), 'Insufficient supervised training rows')
    if operation == "supervised-ols-cpu":
        require(all(len({targets[i] for i in payload["splits"][name]}) > 1 for name in ("validation", "test")), "Constant held-out targets are outside R2 contract")
    return payload


def output_count(payload):
    n, p = len(payload['features']), len(payload['feature_names'])
    return 2*p + (n*p if payload['operation'] == 'supervised-scale' else 1+p+n+1+12)
