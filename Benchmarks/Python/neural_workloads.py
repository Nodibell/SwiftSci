"""NumPy CPU float32 decoder, with independent incremental key/value caches."""
import numpy as np
from neural_shapes import layout, chunk_sizes


def prepare(payload):
    """Decode inputs outside timing; execute creates fresh weights and cache state."""
    return {
        'config': layout(payload), 'chunk_sizes': chunk_sizes(payload),
        'tokens': np.asarray(payload['tokens'], dtype=np.int32),
        'position': payload['position'],
        'execution': payload['execution'],
        'weights': {name: np.asarray(spec['values'], dtype=np.float32).reshape(spec['shape'])
                    for name, spec in payload['weights'].items()},
    }


def execute(prepared):
    weights = {name: value.copy() for name, value in prepared['weights'].items()}
    tokens = prepared['tokens'].copy()
    config = prepared['config']
    width, heads, vocab = [config[k] for k in ('hidden_dim', 'num_heads', 'vocab_size')]
    head_width = width // heads
    half = head_width // 2
    prefix = 'layers.0.'
    cache = None

    def linear(values, name):
        return values @ weights[name].T

    def norm(values, name):
        denominator = np.sqrt(np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
                              + np.float32(2 ** -10))
        return (values / denominator) * weights[name]

    def rotate(values, offset):
        positions = np.arange(offset, offset + values.shape[2], dtype=np.float32)
        frequencies = np.power(np.float32(10000), np.arange(half, dtype=np.float32) / np.float32(half))
        angles = positions[:, None] / frequencies[None, :]
        cosine, sine = np.cos(angles), np.sin(angles)
        return np.concatenate((values[..., :half] * cosine - values[..., half:] * sine,
                               values[..., half:] * cosine + values[..., :half] * sine), axis=-1)

    def forward(chunk):
        nonlocal cache
        batch, length = chunk.shape
        offset = 0 if cache is None else cache[0].shape[2]
        values = weights['embedding.weight'][chunk]
        if prepared['position'] == 'learned':
            values = values + weights['posEmbedding.weight'][offset:offset + length]
        normalized = norm(values, prefix + 'norm1.weight')
        projected = [linear(normalized, prefix + 'attention.' + kind + '_proj.weight')
                     .reshape(batch, length, heads, head_width).transpose(0, 2, 1, 3)
                     for kind in ('query', 'key', 'value')]
        query, key, value = projected
        if prepared['position'] == 'rope':
            query, key = rotate(query, offset), rotate(key, offset)
        if cache is not None:
            key = np.concatenate((cache[0], key), axis=2)
            value = np.concatenate((cache[1], value), axis=2)
        cache = (key, value)
        scores = (query @ key.swapaxes(-1, -2)) * np.float32(head_width ** -.5)
        causal = np.arange(key.shape[2])[None, :] <= (offset + np.arange(length))[:, None]
        scores = np.where(causal, scores, np.float32(-np.inf))
        exponentials = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True, dtype=np.float32)
        attention = (probabilities @ value).transpose(0, 2, 1, 3).reshape(batch, length, width)
        residual = values + linear(attention, prefix + 'attention.out_proj.weight')
        normalized = norm(residual, prefix + 'norm2.weight')
        gate = linear(normalized, prefix + 'ffn.gate.weight')
        up = linear(normalized, prefix + 'ffn.up.weight')
        activated = (gate / (np.float32(1) + np.exp(-gate))) * up
        hidden = residual + linear(activated, prefix + 'ffn.down.weight')
        return linear(norm(hidden, 'finalNorm.weight'), 'lmHead.weight')

    counts = []
    if prepared['execution'] == 'cached':
        chunks = []
        stops = np.cumsum(prepared['chunk_sizes']).tolist()
        ranges = zip([0] + stops[:-1], stops)
        for start, stop in ranges:
            chunks.append(forward(tokens[:, start:stop]))
            counts.append(cache[0].shape[2])
        logits = np.concatenate(chunks, axis=1)
    else:
        logits = forward(tokens)
    expected_shape = (*tokens.shape, vocab)
    if logits.shape != expected_shape or logits.dtype != np.dtype(np.float32) or not np.isfinite(logits).all():
        raise ValueError('Decoder produced an invalid shape, dtype, or nonfinite output')
    result = list(logits.shape) + logits.astype(np.float64).ravel().tolist()
    if counts:
        result += [len(counts)] + counts
    return result
