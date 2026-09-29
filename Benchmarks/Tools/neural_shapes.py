"""Bounded one-layer decoder shape contract, separate from the frozen tiny pack."""
import math
import struct
from contracts import fields, require

OPERATION = 'decoder-shaped-f32'
DEFAULT_CONFIG = dict(vocab_size=7, hidden_dim=8, num_heads=2, intermediate_size=6, max_seq_len=8)


def layout(payload):
    return payload.get('config', DEFAULT_CONFIG)


def chunk_sizes(payload):
    if payload['execution'] == 'full':
        return []
    return payload.get('chunk_sizes', [2, 1, 1])


def shapes(config):
    v, h, f, length = [config[k] for k in ('vocab_size', 'hidden_dim', 'intermediate_size', 'max_seq_len')]
    return {
        'embedding.weight': [v, h], 'posEmbedding.weight': [length, h],
        'finalNorm.weight': [h], 'lmHead.weight': [v, h],
        'layers.0.norm1.weight': [h], 'layers.0.norm2.weight': [h],
        **{f'layers.0.attention.{p}_proj.weight': [h, h] for p in ('query', 'key', 'value', 'out')},
        'layers.0.ffn.gate.weight': [f, h], 'layers.0.ffn.up.weight': [f, h],
        'layers.0.ffn.down.weight': [h, f],
    }


def validate_input(p, operation, rows):
    fields(p, ['operation', 'device', 'position', 'execution', 'loading', 'tokens', 'weights', 'config', 'chunk_sizes'])
    require(operation == p['operation'] == OPERATION, 'Shaped decoder operation mismatch')
    require(p['device'] in ('cpu', 'gpu') and p['position'] in ('learned', 'rope'), 'Invalid decoder placement or position')
    require(p['execution'] in ('full', 'cached') and p['loading'] == 'direct', 'Invalid shaped execution or loading')
    c=p['config'];fields(c, DEFAULT_CONFIG)
    for key, maximum in [('vocab_size', 32), ('hidden_dim', 32), ('num_heads', 4), ('intermediate_size', 64), ('max_seq_len', 128)]:
        require(type(c[key]) is int and 1 <= c[key] <= maximum, 'Invalid bounded decoder config: '+key)
    require(c['hidden_dim'] % c['num_heads'] == 0 and (c['hidden_dim']//c['num_heads']) % 2 == 0, 'Head width must be even')
    tokens=p['tokens']
    require(isinstance(tokens,list) and 1 <= len(tokens) <= 2, 'Invalid decoder batch')
    require(all(isinstance(row,list) and 1 <= len(row) <= c['max_seq_len'] for row in tokens), 'Invalid decoder sequence')
    length=len(tokens[0])
    require(all(len(row)==length for row in tokens), 'Ragged decoder tokens')
    require(all(type(v) is int and 0 <= v < c['vocab_size'] for row in tokens for v in row), 'Invalid decoder token')
    require(type(rows) is int and rows == len(tokens)*length, 'Decoder observation count mismatch')
    chunks=p['chunk_sizes']
    require(isinstance(chunks,list) and all(type(n) is int and n > 0 for n in chunks), 'Invalid cache chunks')
    require((p['execution']=='full' and chunks==[]) or
            (p['execution']=='cached' and len(chunks)>=2 and sum(chunks)==length and all(n==1 for n in chunks[1:])),
            'Cache contract requires a prefix followed by single tokens')
    expected=shapes(c); fields(p['weights'],expected)
    for name,shape in expected.items():
        weight=p['weights'][name];fields(weight,['shape','values'])
        require(isinstance(weight['shape'],list) and all(type(n) is int for n in weight['shape']) and weight['shape']==shape, 'Invalid decoder weight shape')
        values=weight['values']
        require(isinstance(values,list) and len(values)==math.prod(shape), 'Invalid decoder weight count')
        for value in values:
            require(type(value) in (int,float) and math.isfinite(value) and abs(value)<=2, 'Invalid decoder weight')
            require(struct.unpack('<f',struct.pack('<f',value))[0]==value, 'Decoder weight must be exact Float32')
    return p


def output_count(p):
    return 3 + len(p['tokens'])*len(p['tokens'][0])*p['config']['vocab_size'] + (1+len(p['chunk_sizes']) if p['execution']=='cached' else 0)
