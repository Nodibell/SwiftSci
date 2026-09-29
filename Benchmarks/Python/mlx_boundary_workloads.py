"""Pandas row preparation and native MLX affine arithmetic with complete readback."""
import mlx.core as mx
from boundary_workloads import source_frame
from boundary_sweep import expand


def convert(payload, source):
    selected = source.loc[source['filter'] >= payload['filter_threshold']].sort_values(
        'sort', ascending=payload['ascending'], kind='stable')
    matrix = selected[payload['feature_order']].to_numpy(dtype='float64', copy=True)
    target = selected['target'].to_numpy(dtype='float64', copy=True)
    dtype = mx.float32 if payload['dtype'] == 'float32' else mx.float64
    tensor = mx.array(matrix.copy(), dtype=dtype)
    targets = mx.array(target.copy(), dtype=dtype)
    weights = mx.array(payload['weights'], dtype=dtype)
    bias = mx.array(payload['bias'], dtype=dtype)
    mx.eval(tensor, targets, weights, bias)
    mx.synchronize()
    return selected, matrix, target, tensor, targets, weights, bias


def compute(prepared):
    _, _, _, tensor, targets, weights, bias = prepared
    prediction = tensor @ weights + bias
    residual = prediction - targets
    mx.eval(prediction, residual)
    mx.synchronize()
    return prediction.reshape(-1).tolist(), residual.reshape(-1).tolist()


def conversion_output(prepared):
    selected, matrix, target, tensor, _, _, _ = prepared
    return ([len(selected), matrix.shape[1], tensor.itemsize * 8]
            + selected['row_id'].tolist() + matrix.ravel().tolist()
            + target.tolist() + tensor.reshape(-1).tolist())


def prepare(payload):
    stage = payload.get('stage', 'pipeline')
    concrete = expand(payload) if payload['operation'] == 'dataframe-model-sweep' else payload
    source = source_frame(concrete) if stage in ('conversion', 'prepared') else None
    prepared = convert(concrete, source) if stage == 'prepared' else None
    return stage, concrete, source, prepared


def execute(state):
    stage, payload, source, prepared = state
    if stage == 'prepared':
        prediction, residual = compute(prepared)
        tensor = prepared[3]
        return [tensor.shape[0], tensor.shape[1], tensor.itemsize * 8] + prediction + residual
    if source is None:
        source = source_frame(payload)
    prepared = convert(payload, source)
    output = conversion_output(prepared)
    if stage == 'conversion':
        return output
    prediction, residual = compute(prepared)
    selected, matrix, target, tensor, targets, _, _ = prepared
    snapshot, selected_snapshot = source.copy(deep=True), selected.copy(deep=True)
    matrix[0, 0] += 17
    target[0] += 19
    copied = source.copy(deep=True)
    copied[payload['feature_names'][0]] = -23.0
    isolated = (source.equals(snapshot) and selected.equals(selected_snapshot)
                and tensor.reshape(-1)[0].item() != matrix[0, 0]
                and targets[0].item() != target[0]
                and copied[payload['feature_names'][0]].eq(-23).all())
    return output + prediction + residual + [int(isolated)]
