"""Scalar decoder oracle: exact binary32 inputs, evaluated at 80 decimal digits."""
import mpmath as mp
from neural_shapes import layout, chunk_sizes


def reference(payload):
    """Return all causal logits; cached expectations come from full-sequence math."""
    config = layout(payload)
    hidden, heads, vocab = [config[k] for k in ('hidden_dim', 'num_heads', 'vocab_size')]
    head_width = hidden // heads
    half = head_width // 2
    with mp.workdps(80):
        weights = {}
        for name, spec in payload['weights'].items():
            values = [mp.mpf(value) for value in spec['values']]
            if len(spec['shape']) == 2:
                width = spec['shape'][1]
                values = [values[i:i + width] for i in range(0, len(values), width)]
            weights[name] = values

        def linear(vector, name):
            return [mp.fsum(a * b for a, b in zip(vector, row)) for row in weights[name]]

        def norm(vector, name):
            denominator = mp.sqrt(mp.fsum(value * value for value in vector) / hidden + mp.mpf(2) ** -10)
            return [value * scale / denominator for value, scale in zip(vector, weights[name])]

        def rotate(head, position):
            angles = [mp.mpf(position) / mp.power(10000, mp.mpf(j) / half) for j in range(half)]
            return ([head[j] * mp.cos(angles[j]) - head[j + half] * mp.sin(angles[j]) for j in range(half)]
                    + [head[j + half] * mp.cos(angles[j]) + head[j] * mp.sin(angles[j]) for j in range(half)])

        logits = []
        prefix = 'layers.0.'
        for sequence in payload['tokens']:
            embedded = []
            for position, token in enumerate(sequence):
                vector = list(weights['embedding.weight'][token])
                if payload['position'] == 'learned':
                    vector = [a + b for a, b in zip(vector, weights['posEmbedding.weight'][position])]
                embedded.append(vector)
            projections = {}
            for kind in ('query', 'key', 'value'):
                rows = [linear(norm(row, prefix + 'norm1.weight'), prefix + 'attention.' + kind + '_proj.weight')
                        for row in embedded]
                split = [[row[i:i+head_width] for i in range(0, hidden, head_width)] for row in rows]
                if payload['position'] == 'rope' and kind != 'value':
                    split = [[rotate(head, position) for head in pair] for position, pair in enumerate(split)]
                projections[kind] = split
            for position, vector in enumerate(embedded):
                attention = []
                for head in range(heads):
                    scores = [mp.fsum(a * b for a, b in zip(projections['query'][position][head],
                                                          projections['key'][j][head])) / mp.sqrt(head_width)
                              for j in range(position + 1)]
                    maximum = max(scores)
                    exponentials = [mp.exp(score - maximum) for score in scores]
                    denominator = mp.fsum(exponentials)
                    attention.extend(mp.fsum(probability * projections['value'][j][head][component]
                                              for j, probability in enumerate(exponentials)) / denominator
                                     for component in range(head_width))
                projected = linear(attention, prefix + 'attention.out_proj.weight')
                residual = [a + b for a, b in zip(vector, projected)]
                normalized = norm(residual, prefix + 'norm2.weight')
                gate = linear(normalized, prefix + 'ffn.gate.weight')
                up = linear(normalized, prefix + 'ffn.up.weight')
                activated = [g / (1 + mp.exp(-g)) * u for g, u in zip(gate, up)]
                down = linear(activated, prefix + 'ffn.down.weight')
                final = norm([a + b for a, b in zip(residual, down)], 'finalNorm.weight')
                logits.extend(linear(final, 'lmHead.weight'))
        result = [len(payload['tokens']), len(payload['tokens'][0]), vocab] + [float(value) for value in logits]
        if payload['execution'] == 'cached':
            sizes = chunk_sizes(payload)
            result += [len(sizes)] + [sum(sizes[:i+1]) for i in range(len(sizes))]
        return result
