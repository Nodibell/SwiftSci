"""Pandas conversion and NumPy affine computation on CPU for both device requests."""
import numpy as np
import pandas as pd


def prepare(payload):
    return payload


def source_frame(p):
    frame = pd.DataFrame(p['features'], columns=p['feature_names'], dtype=np.float64)
    frame['row_id'] = p['row_ids']
    frame['target'] = p['targets']
    frame['filter'] = p['filter_values']
    frame['sort'] = p['sort_keys']
    return frame


def convert(p, source):
    selected = source.loc[source['filter'] >= p['filter_threshold']].sort_values('sort', ascending=p['ascending'], kind='stable')
    matrix = selected[p['feature_order']].to_numpy(dtype=np.float64, copy=True)
    target = selected['target'].to_numpy(dtype=np.float64, copy=True)
    dtype = np.float32 if p['dtype']=='float32' else np.float64
    return selected, matrix, target, matrix.astype(dtype, copy=True), np.asarray(p['weights'], dtype=dtype), dtype(p['bias'])


def compute(prepared):
    selected, matrix, target, tensor, weights, bias = prepared
    prediction = tensor @ weights + bias
    residual = prediction - target.astype(tensor.dtype)
    return prediction, residual


def execute(p):
    source = source_frame(p)
    prepared = convert(p, source)
    selected, matrix, target, tensor, weights, bias = prepared
    prediction, residual = compute(prepared)
    output = [len(selected), matrix.shape[1], tensor.dtype.itemsize*8] + selected['row_id'].tolist() + matrix.ravel().tolist() + target.tolist() + tensor.ravel().tolist() + prediction.tolist() + residual.tolist()
    snapshot = source.copy(deep=True)
    selected_snapshot = selected.copy(deep=True)
    matrix[0, 0] += 17
    target[0] += 19
    copied = source.copy(deep=True)
    copied[p['feature_names'][0]] = -23.0
    isolated = source.equals(snapshot) and selected.equals(selected_snapshot) and tensor.ravel()[0] != matrix[0, 0] and copied[p['feature_names'][0]].eq(-23).all()
    return output + [int(isolated)]
