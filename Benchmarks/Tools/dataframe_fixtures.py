"""Lossless dataframe fixture contracts and a scalar independent reference."""

import math
import operator
import struct
from contracts import fields, require

OPERATION = 'dataframe-semantics'
TYPES = {'int64': 1, 'float64': 2, 'bool': 3, 'utf8': 4}
PREDICATES = {'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt,
              'le': operator.le, 'gt': operator.gt, 'ge': operator.ge}


def cell(value, dtype):
    if value is None:
        return None
    if dtype == 'int64':
        require(isinstance(value, str), 'Int64 requires decimal string')
        try:
            result = int(value)
        except ValueError:
            require(False, 'Invalid Int64 decimal')
        require(str(result) == value and -(2**63) <= result < 2**63, 'Noncanonical or overflowing Int64')
        return result
    if dtype == 'float64':
        if isinstance(value, str):
            require(value in ('NaN', '+Inf', '-Inf'), 'Unknown float tag')
            return {'NaN': math.nan, '+Inf': math.inf, '-Inf': -math.inf}[value]
        require(type(value) in (int, float) and math.isfinite(value), 'Invalid Float64 cell')
        return float(value)
    require(type(value) is (bool if dtype == 'bool' else str), 'Invalid typed cell')
    return value


def validate_input(payload, operation, rows):
    fields(payload, ['operation', 'action', 'columns', 'parameters'])
    require(operation == OPERATION == payload['operation'], 'Dataframe operation mismatch')
    columns = payload['columns']
    require(isinstance(columns, list) and columns and type(rows) is int and rows > 0, 'Missing dataframe columns')
    names = {}
    for column in columns:
        fields(column, ['name', 'type', 'values'])
        name = column['name']
        require(isinstance(name, str) and name and name not in names, 'Duplicate or invalid column name')
        require(column['type'] in TYPES, 'Unknown column type')
        require(isinstance(column['values'], list) and len(column['values']) == rows, 'Ragged dataframe')
        names[name] = column['type']
        for value in column['values']:
            cell(value, column['type'])
    action, params = payload['action'], payload['parameters']
    shape = {'gather': ['indices'], 'select': ['columns'], 'sort': ['column', 'ascending'],
             'filter': ['column', 'predicate', 'value'], 'matrix': ['columns', 'target'],
             'replace': ['column', 'values'], 'group-count': ['columns', 'value']}
    require(action in shape, 'Unknown dataframe action')
    fields(params, shape[action])

    def named(name, allowed=TYPES):
        require(isinstance(name, str) and name in names and names[name] in allowed, 'Unknown or unsupported column')

    if action == 'gather':
        require(isinstance(params['indices'], list), 'Invalid gather indices')
        require(all(type(i) is int and 0 <= i < rows for i in params['indices']), 'Gather index out of bounds')
    if action in ('select', 'matrix', 'group-count'):
        keys = params['columns']
        require(isinstance(keys, list), 'Invalid column selection')
        for key in keys:
            named(key, ('int64', 'float64', 'bool') if action == 'matrix' else TYPES)
        if action != 'matrix':
            require(keys and len(set(keys)) == len(keys), 'Empty or duplicate column selection')
    if action == 'matrix':
        named(params['target'], ('int64', 'float64', 'bool'))
    if action == 'group-count':
        require(1 <= len(params['columns']) <= 2, 'Expected one or two grouping keys')
        for key in params['columns']:
            named(key, ('int64', 'utf8'))
        named(params['value'], ('int64', 'float64'))
        require(params['value'] not in params['columns'], 'Grouped value is a key')
    if action in ('sort', 'filter', 'replace'):
        named(params['column'], ('int64',) if action == 'replace' else ('int64', 'float64'))
        dtype = names[params['column']]
        if action == 'sort':
            require(type(params['ascending']) is bool, 'Invalid sorting direction')
            source = next(c for c in columns if c['name'] == params['column'])
            require('NaN' not in source['values'], 'NaN sorting has no declared ordering')
        if action == 'filter':
            predicate = params['predicate']
            require(predicate in set(PREDICATES) | {'isNull', 'isNotNull'}, 'Unknown predicate')
            if predicate in ('isNull', 'isNotNull'):
                require(params['value'] is None, 'Null predicate must have null operand')
            else:
                require(params['value'] is not None, 'Missing comparison operand')
                cell(params['value'], dtype)
        if action == 'replace':
            require(isinstance(params['values'], list) and len(params['values']) == rows, 'Replacement length mismatch')
            for value in params['values']:
                cell(value, dtype)
    return payload


def decoded_columns(payload):
    return [dict(c, values=[cell(v, c['type']) for v in c['values']]) for c in payload['columns']]


def words(bits):
    return [bits >> 32, bits & 0xffffffff]


def float_words(value):
    bits = 0x7ff8000000000000 if math.isnan(value) else struct.unpack('>Q', struct.pack('>d', value))[0]
    return words(bits)


def text_words(value):
    data = value.encode('utf-8')
    return [len(data), *data]


def encode_frames(frames):
    result = [len(frames)]
    for columns in frames:
        rows = len(columns[0]['values']) if columns else 0
        result += [rows, len(columns)]
        for c in columns:
            require(len(c['values']) == rows, 'Ragged output frame')
            result += [TYPES[c['type']], *text_words(c['name']), sum(v is None for v in c['values'])]
            for value in c['values']:
                result.append(0 if value is None else 1)
                if value is None:
                    continue
                if c['type'] == 'int64':
                    require(type(value) is int and -(2**63) <= value < 2**63, 'Invalid Int64 output')
                    result += words(value & 0xffffffffffffffff)
                elif c['type'] == 'float64':
                    result += float_words(value)
                elif c['type'] == 'bool':
                    result.append(int(value))
                else:
                    result += text_words(value)
    return result


def expected_values(payload):
    """Apply the declared scalar contract without importing any tested library."""
    source = decoded_columns(payload)
    by_name = {c['name']: c for c in source}
    n = len(source[0]['values'])
    action, p = payload['action'], payload['parameters']

    def gather(indices):
        return [dict(c, values=[c['values'][i] for i in indices]) for c in source] if indices else []

    if action == 'gather':
        result = gather(p['indices'])
    elif action == 'select':
        result = [by_name[name] for name in p['columns']]
    elif action == 'sort':
        values = by_name[p['column']]['values']
        present = sorted((i for i in range(n) if values[i] is not None), key=lambda i: values[i], reverse=not p['ascending'])
        result = gather(present + [i for i in range(n) if values[i] is None])
    elif action == 'filter':
        c = by_name[p['column']]
        operand = cell(p['value'], c['type'])
        predicate = p['predicate']
        def accepted(v):
            if predicate == 'isNull':
                return v is None
            if predicate == 'isNotNull':
                return v is not None
            return v is not None and PREDICATES[predicate](v, operand)
        result = gather([i for i, value in enumerate(c['values']) if accepted(value)])
    elif action == 'replace':
        name = p['column']
        result = [dict(c, values=[cell(v, c['type']) for v in p['values']]) if c['name'] == name else c for c in source]
        return encode_frames([source, result, [by_name[name]], [by_name[name]]])
    elif action == 'matrix':
        result = encode_frames([source]) + [n, len(p['columns'])]
        flat = [math.nan if by_name[name]['values'][i] is None else float(by_name[name]['values'][i]) for i in range(n) for name in p['columns']]
        for value in flat + flat:
            result += float_words(value)
        result.append(n)
        for value in by_name[p['target']]['values']:
            result += float_words(math.nan if value is None else float(value))
        return result
    else:
        groups = {}
        for i in range(n):
            key = tuple(by_name[name]['values'][i] for name in p['columns'])
            groups.setdefault(key, []).append(i)
        keys = [{'name': name, 'type': 'utf8', 'values': [None if key[j] is None else str(key[j]) for key in groups]} for j, name in enumerate(p['columns'])]
        counts = keys + [{'name': c['name'], 'type': 'int64', 'values': [len(indices) for indices in groups.values()]} for c in source if c['name'] not in p['columns'] and c['type'] in ('int64', 'float64')]
        value = by_name[p['value']]
        present = keys + [{'name': value['name'] + '_count', 'type': 'float64', 'values': [float(sum(value['values'][i] is not None for i in indices)) for indices in groups.values()]}]
        return encode_frames([source, counts, present])
    return encode_frames([source, result])
