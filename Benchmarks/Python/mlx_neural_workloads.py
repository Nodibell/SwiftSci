"""Fixed decoder arithmetic on the fixture-selected MLX device.

The worker owns the stream scope. All tensors and cache state are fresh per call.
"""
import math
import mlx.core as mx


def execute(payload):
    weights = {}
    for name, spec in sorted(payload['weights'].items()):
        tensor = mx.array(spec['values'], dtype=mx.float32).reshape(spec['shape'])
        mx.eval(tensor)
        mx.synchronize()
        if tensor.dtype != mx.float32 or tensor.reshape(-1).tolist() != spec['values']:
            raise ValueError('Fixed weight readback differs: ' + name)
        weights[name] = tensor
    tokens = mx.array(payload['tokens'], dtype=mx.int32)
    prefix = 'layers.0.'
    cache = None

    def linear(values, name):
        return values @ weights[name].T

    def norm(values, name):
        denominator = mx.sqrt(mx.mean(values * values, axis=-1, keepdims=True)
                              + 2 ** -10)
        return (values / denominator) * weights[name]

    def rotate(values, offset):
        positions = mx.arange(offset, offset + values.shape[2], dtype=mx.float32)
        frequencies = mx.power(10000.0, mx.arange(2, dtype=mx.float32) / 2.0)
        angles = positions[:, None] / frequencies[None, :]
        cosine, sine = mx.cos(angles), mx.sin(angles)
        return mx.concatenate((values[..., :2] * cosine - values[..., 2:] * sine,
                               values[..., 2:] * cosine + values[..., :2] * sine), axis=-1)

    def forward(chunk):
        nonlocal cache
        batch, length = chunk.shape
        offset = 0 if cache is None else cache[0].shape[2]
        values = weights['embedding.weight'][chunk]
        if payload['position'] == 'learned':
            values = values + weights['posEmbedding.weight'][offset:offset + length]
        normalized = norm(values, prefix + 'norm1.weight')
        projected = [linear(normalized, prefix + 'attention.' + kind + '_proj.weight')
                     .reshape(batch, length, 2, 4).transpose(0, 2, 1, 3)
                     for kind in ('query', 'key', 'value')]
        query, key, value = projected
        if payload['position'] == 'rope':
            query, key = rotate(query, offset), rotate(key, offset)
        if cache is not None:
            key = mx.concatenate((cache[0], key), axis=2)
            value = mx.concatenate((cache[1], value), axis=2)
        cache = (key, value)
        scores = (query @ key.swapaxes(-1, -2)) * 0.5
        causal = mx.arange(key.shape[2])[None, :] <= (offset + mx.arange(length))[:, None]
        scores = mx.where(causal, scores, -float("inf"))
        exponentials = mx.exp(scores - mx.max(scores, axis=-1, keepdims=True))
        probabilities = exponentials / mx.sum(exponentials, axis=-1, keepdims=True)
        attention = (probabilities @ value).transpose(0, 2, 1, 3).reshape(batch, length, 8)
        residual = values + linear(attention, prefix + 'attention.out_proj.weight')
        normalized = norm(residual, prefix + 'norm2.weight')
        gate = linear(normalized, prefix + 'ffn.gate.weight')
        up = linear(normalized, prefix + 'ffn.up.weight')
        activated = (gate / (1.0 + mx.exp(-gate))) * up
        hidden = residual + linear(activated, prefix + 'ffn.down.weight')
        return linear(norm(hidden, 'finalNorm.weight'), 'lmHead.weight')

    counts = []
    if payload['execution'] == 'cached':
        chunks = []
        for start, stop in ((0, 2), (2, 3), (3, 4)):
            chunk = forward(tokens[:, start:stop])
            mx.eval(chunk)
            chunks.append(chunk)
            counts.append(cache[0].shape[2])
        logits = mx.concatenate(chunks, axis=1)
    else:
        logits = forward(tokens)
    expected_shape = (*tokens.shape, 7)
    if tuple(logits.shape) != expected_shape or logits.dtype != mx.float32:
        raise ValueError('Decoder produced an invalid shape, dtype, or nonfinite output')
    mx.eval(logits)
    mx.synchronize()
    values = logits.reshape(-1).tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Decoder produced nonfinite logits")
    result = list(logits.shape) + values
    if counts:
        result += [len(counts)] + counts
    return result
