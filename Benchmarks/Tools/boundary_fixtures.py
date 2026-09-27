"""Bounded dataframe-to-MLX contract with no reference answers in worker inputs."""
import math
from contracts import fields, require

OPERATION = 'dataframe-model'


def validate_input(p, operation, rows):
    fields(p, ['operation', 'device', 'dtype', 'row_ids', 'feature_names', 'feature_order',
               'features', 'targets', 'filter_values', 'filter_threshold', 'sort_keys',
               'ascending', 'weights', 'bias'])
    require(operation == p['operation'] == OPERATION, 'Boundary operation mismatch')
    require(p['device'] in ('cpu', 'gpu') and p['dtype'] in ('float32', 'float64'), 'Invalid device or dtype')
    require((p['device'], p['dtype']) != ('gpu', 'float64'), 'MLX Metal does not support Float64')
    require(type(rows) is int and 2 <= rows <= 100000, 'Invalid boundary rows')
    names = p['feature_names']
    require(isinstance(names, list) and 1 <= len(names) <= 64 and
            all(isinstance(n, str) and n and n not in ('row_id', 'target', 'filter', 'sort') for n in names), 'Invalid feature names')
    require(len(set(names)) == len(names), 'Duplicate feature names')
    order = p['feature_order']
    require(isinstance(order, list) and len(order) == len(names) and
            all(isinstance(n, str) for n in order) and set(order) == set(names), 'Invalid feature order')
    ids = p['row_ids']
    require(isinstance(ids, list) and len(ids) == rows and all(type(x) is int and 0 <= x <= 2**31-1 for x in ids)
            and len(set(ids)) == rows, 'Invalid row IDs')
    def vector(x, count):
        require(isinstance(x, list) and len(x) == count and
                all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1024 for v in x), 'Invalid bounded numeric vector')
    require(isinstance(p['features'], list) and len(p['features']) == rows, 'Invalid feature rows')
    for row in p['features']: vector(row, len(names))
    for key in ('targets', 'filter_values', 'sort_keys'): vector(p[key], rows)
    vector(p['weights'], len(names))
    vector([p['filter_threshold'], p['bias']], 2)
    require(type(p['ascending']) is bool, 'Invalid sort direction')
    require(any(v >= p['filter_threshold'] for v in p['filter_values']), 'Empty model input')
    return p


def selected_indices(p):
    return sorted((i for i, v in enumerate(p['filter_values']) if v >= p['filter_threshold']),
                  key=lambda i: p['sort_keys'][i], reverse=not p['ascending'])


def output_count(p):
    n, d = len(selected_indices(p)), len(p['feature_names'])
    return 4 + n * (2*d + 4)
