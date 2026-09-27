"""Strict fixed Float32 decoder inputs; dimensions are bounded by this version."""
import math
import struct
from contracts import fields, require

OPERATION = 'decoder-fixed-f32'
SHAPES = {
    'embedding.weight': [7, 8], 'posEmbedding.weight': [8, 8],
    'finalNorm.weight': [8], 'lmHead.weight': [7, 8],
    'layers.0.norm1.weight': [8], 'layers.0.norm2.weight': [8],
    **{f'layers.0.attention.{p}_proj.weight': [8, 8] for p in ('query', 'key', 'value', 'out')},
    'layers.0.ffn.gate.weight': [6, 8], 'layers.0.ffn.up.weight': [6, 8],
    'layers.0.ffn.down.weight': [8, 6],
}


def validate_input(payload, operation, rows):
    fields(payload, ['operation', 'device', 'position', 'execution', 'loading', 'tokens', 'weights'])
    require(operation == OPERATION == payload['operation'], 'Neural operation mismatch')
    for key, choices in [('device', ('cpu', 'gpu')), ('position', ('learned', 'rope')),
                         ('execution', ('full', 'cached')), ('loading', ('direct', 'public-loader'))]:
        require(payload[key] in choices, f'Unknown neural {key}')
    tokens = payload['tokens']
    require(isinstance(tokens, list) and 1 <= len(tokens) <= 2, 'Invalid token batch')
    require(all(isinstance(row, list) and 1 <= len(row) <= 4 for row in tokens), 'Invalid token sequence')
    width = len(tokens[0])
    require(all(len(row) == width for row in tokens), 'Ragged tokens')
    require(all(type(v) is int and 0 <= v < 7 for row in tokens for v in row), 'Invalid token ID')
    require(type(rows) is int and rows == len(tokens)*width, 'Neural observation count mismatch')
    require(payload['execution'] != 'cached' or width == 4, 'Cached fixture requires four tokens')
    weights = payload['weights']
    fields(weights, SHAPES)
    for name, shape in SHAPES.items():
        weight = weights[name]
        fields(weight, ['shape', 'values'])
        require(isinstance(weight['shape'], list) and all(type(v) is int for v in weight['shape']) and weight['shape'] == shape, 'Wrong weight shape')
        values = weight['values']
        require(isinstance(values, list) and len(values) == math.prod(shape), 'Wrong weight count')
        for v in values:
            require(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2, 'Invalid bounded weight')
            require(struct.unpack('<f', struct.pack('<f', v))[0] == v, 'Weight is not exactly Float32')
    return payload


def output_count(payload):
    return 3 + len(payload['tokens'])*len(payload['tokens'][0])*7 + (4 if payload['execution']=='cached' else 0)
