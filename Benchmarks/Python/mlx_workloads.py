"""Supported MLX comparison contracts and explicit execution placement."""
OPERATIONS = frozenset({'decoder-fixed-f32', 'dataframe-model', 'dataframe-model-sweep'})


def check_supported(payload):
    if payload['operation'] not in OPERATIONS:
        raise ValueError('Unsupported MLX comparison operation: ' + payload['operation'])
    if payload['operation'] == 'decoder-fixed-f32' and payload['loading'] != 'direct':
        raise ValueError('The Swift public weight loader has no equivalent Python MLX contract')


class MLXWorkload:
    def __init__(self, payload):
        check_supported(payload)
        import mlx.core as mx
        self.mx, self.payload = mx, payload
        if payload['device'] == 'gpu' and not mx.metal.is_available():
            raise ValueError('Requested Metal GPU is unavailable; no CPU fallback')
        self.device = mx.cpu if payload['device'] == 'cpu' else mx.gpu
        self.stream = mx.new_stream(self.device)
        with mx.stream(self.stream):
            if payload['operation'] == 'decoder-fixed-f32':
                from mlx_neural_workloads import execute
                self.state = payload
            else:
                from mlx_boundary_workloads import prepare, execute
                self.state = prepare(payload)
        self.operation = execute

    def execute(self):
        with self.mx.stream(self.stream):
            output = self.operation(self.state)
            self.mx.synchronize()
            return output
