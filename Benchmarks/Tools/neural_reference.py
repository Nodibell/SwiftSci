"""Scalar decoder oracle: exact binary32 inputs, evaluated at 80 decimal digits."""
import mpmath as mp


def reference(payload):
    """Return all causal logits; cached expectations come from full-sequence math."""
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
            denominator = mp.sqrt(mp.fsum(value * value for value in vector) / 8 + mp.mpf(2) ** -10)
            return [value * scale / denominator for value, scale in zip(vector, weights[name])]

        def rotate(head, position):
            angles = [mp.mpf(position) / mp.power(10000, mp.mpf(j) / 2) for j in range(2)]
            return ([head[j] * mp.cos(angles[j]) - head[j + 2] * mp.sin(angles[j]) for j in range(2)]
                    + [head[j + 2] * mp.cos(angles[j]) + head[j] * mp.sin(angles[j]) for j in range(2)])

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
                heads = [[row[:4], row[4:]] for row in rows]
                if payload['position'] == 'rope' and kind != 'value':
                    heads = [[rotate(head, position) for head in pair] for position, pair in enumerate(heads)]
                projections[kind] = heads
            for position, vector in enumerate(embedded):
                attention = []
                for head in range(2):
                    scores = [mp.fsum(a * b for a, b in zip(projections['query'][position][head],
                                                          projections['key'][j][head])) / 2
                              for j in range(position + 1)]
                    maximum = max(scores)
                    exponentials = [mp.exp(score - maximum) for score in scores]
                    denominator = mp.fsum(exponentials)
                    attention.extend(mp.fsum(probability * projections['value'][j][head][component]
                                              for j, probability in enumerate(exponentials)) / denominator
                                     for component in range(4))
                projected = linear(attention, prefix + 'attention.out_proj.weight')
                residual = [a + b for a, b in zip(vector, projected)]
                normalized = norm(residual, prefix + 'norm2.weight')
                gate = linear(normalized, prefix + 'ffn.gate.weight')
                up = linear(normalized, prefix + 'ffn.up.weight')
                activated = [g / (1 + mp.exp(-g)) * u for g, u in zip(gate, up)]
                down = linear(activated, prefix + 'ffn.down.weight')
                final = norm([a + b for a, b in zip(residual, down)], 'finalNorm.weight')
                logits.extend(linear(final, 'lmHead.weight'))
        result = [len(payload['tokens']), len(payload['tokens'][0]), 7] + [float(value) for value in logits]
        if payload['execution'] == 'cached':
            result += [3, 2, 3, 4]
        return result
