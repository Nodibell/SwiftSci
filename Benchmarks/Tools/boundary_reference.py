"""Scalar oracle. Decimal arithmetic is independent of pandas, NumPy and MLX."""
from decimal import Decimal, localcontext
import struct


def reference(p):
    # Keep row selection here independent of runtime dataframe operations.
    selected = [i for i in range(len(p['row_ids'])) if p['filter_values'][i] >= p['filter_threshold']]
    selected.sort(key=lambda i: p['sort_keys'][i], reverse=not p['ascending'])
    columns = [p['feature_names'].index(name) for name in p['feature_order']]
    matrix = [p['features'][i][j] for i in selected for j in columns]
    targets = [p['targets'][i] for i in selected]
    cast = (lambda x: struct.unpack('<f', struct.pack('<f', x))[0]) if p['dtype'] == 'float32' else float
    tensor = list(map(cast, matrix))
    weights = list(map(cast, p['weights']))
    bias = cast(p['bias'])
    width = len(columns)
    with localcontext() as ctx:
        ctx.prec = 100
        predictions = [float(sum((Decimal.from_float(tensor[r*width+c]) * Decimal.from_float(weights[c])
                                  for c in range(width)), Decimal.from_float(bias))) for r in range(len(selected))]
        residuals = [float(Decimal.from_float(pred) - Decimal.from_float(cast(target)))
                     for pred, target in zip(predictions, targets)]
    return [len(selected), width, 32 if p['dtype']=='float32' else 64] + [p['row_ids'][i] for i in selected] + matrix + targets + tensor + predictions + residuals + [1]
