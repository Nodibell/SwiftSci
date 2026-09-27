"""Input and output contracts for controlled CPU model, forecast and search cases."""

import math
from contracts import fields, require, positive

OPERATIONS = {
    'linear-fixed-cpu', 'logistic-fixed-cpu', 'kmeans-one-cpu',
    'kalman-fixed-cpu', 'vector-cosine', 'kernel-shap',
}


def scalar(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected a finite number')


def vector(value, width=None):
    require(isinstance(value, list) and value, 'Expected a nonempty vector')
    for v in value:
        scalar(v)
    require(width is None or len(value) == width, 'Vector width mismatch')


def matrix(value, rows=None, width=None):
    require(isinstance(value, list) and value, 'Expected a nonempty matrix')
    require(rows is None or len(value) == rows, 'Matrix row count mismatch')
    vector(value[0], width)
    for row in value:
        vector(row, len(value[0]))


def covariance(value, size, strictly_positive=False):
    matrix(value, size, size)
    require(all(value[i][j] == value[j][i] for i in range(size) for j in range(size)), 'Asymmetric covariance')
    require(all(value[i][i] >= 0 for i in range(size)), 'Negative covariance diagonal')
    determinant = value[0][0] if size == 1 else value[0][0]*value[1][1]-value[0][1]*value[1][0]
    require(determinant > 0 if strictly_positive else determinant >= 0, 'Invalid covariance')


def validate_input(payload, operation, rows):
    require(operation in OPERATIONS, 'Unknown controlled operation')
    require(isinstance(payload, dict) and payload.get('operation') == operation, 'Controlled operation mismatch')
    if operation in ('linear-fixed-cpu', 'logistic-fixed-cpu'):
        fields(payload, ['operation', 'weights', 'bias', 'features'])
        vector(payload['weights']); scalar(payload['bias'])
        matrix(payload['features'], rows, len(payload['weights']))
    elif operation == 'kmeans-one-cpu':
        fields(payload, ['operation', 'features', 'query', 'n_clusters', 'max_iterations', 'tolerance', 'seed'])
        matrix(payload['features'], rows)
        matrix(payload['query'], width=len(payload['features'][0]))
        require(type(payload['n_clusters']) is int and payload['n_clusters'] == 1, 'Only one-cluster conformance is supported')
        positive(payload['max_iterations'], 'max_iterations', 2)
        scalar(payload['tolerance']); require(payload['tolerance'] > 0, 'Invalid clustering tolerance')
        require(type(payload['seed']) is int and 0 <= payload['seed'] < 2**63, 'Invalid seed')
    elif operation == 'kalman-fixed-cpu':
        fields(payload, ['operation', 'state_size', 'observation_size', 'transition', 'observation_matrix', 'process_noise', 'measurement_noise', 'initial_mean', 'initial_covariance', 'observations'])
        n, m = payload['state_size'], payload['observation_size']
        require(type(n) is int and n in (1, 2) and type(m) is int and m in (1, 2), 'Controlled Kalman dimensions must be one or two')
        matrix(payload['transition'], n, n); matrix(payload['observation_matrix'], m, n)
        covariance(payload['process_noise'], n); covariance(payload['measurement_noise'], m, True)
        vector(payload['initial_mean'], n); covariance(payload['initial_covariance'], n)
        matrix(payload['observations'], rows, m)
    elif operation == 'vector-cosine':
        fields(payload, ['operation', 'vectors', 'query', 'top_k'])
        matrix(payload['vectors'], rows)
        vector(payload['query'], len(payload['vectors'][0]))
        require(all(any(v != 0 for v in row) for row in payload['vectors'] + [payload['query']]), 'Zero-norm cosine policy is not certified')
        require(type(payload['top_k']) is int and 1 <= payload['top_k'] <= rows, 'Invalid top K')
    else:
        fields(payload, ['operation', 'instance', 'background', 'model'])
        vector(payload['instance'], 2); matrix(payload['background'], rows, 2)
        fields(payload['model'], ['bias', 'weights', 'interaction'])
        scalar(payload['model']['bias']); vector(payload['model']['weights'], 2); scalar(payload['model']['interaction'])
    return payload


def output_count(payload):
    op = payload['operation']
    if op in ('linear-fixed-cpu', 'logistic-fixed-cpu'):
        return 1 + len(payload['weights']) + len(payload['features']) * (3 if op == 'logistic-fixed-cpu' else 1)
    if op == 'kmeans-one-cpu':
        return len(payload['features'][0]) + len(payload['features']) + len(payload['query']) + 1
    if op == 'kalman-fixed-cpu':
        n = payload['state_size']
        return (len(payload['observations'])+1)*(n+n*n)
    if op == 'vector-cosine':
        return payload['top_k']*2
    require(op == 'kernel-shap', 'Unknown controlled output')
    return 4
