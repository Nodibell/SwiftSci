"""Experimental precision adapters for the pinned tied-embedding Llama reference."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

MODES = ('bf16', 'head-float32', 'float32')


class Float32OutputProjection(nn.Module):
    """Keep the original body; retain an exact Float32 copy of the output weights."""
    def __init__(self, base):
        super().__init__()
        if base.model_type != 'llama' or not base.args.tie_word_embeddings:
            raise ValueError('This experiment requires Llama with tied output embeddings')
        self.base = base
        self.output_weight = base.model.embed_tokens.weight.astype(mx.float32)

    @property
    def layers(self):
        return self.base.layers

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.base.model(inputs, cache=cache, input_embeddings=input_embeddings)
        return hidden.astype(mx.float32) @ self.output_weight.T


def prepare_precision(model, mode):
    if mode not in MODES:
        raise ValueError('Unknown Llama precision experiment')
    if mode == 'float32':
        model.set_dtype(mx.float32)
    elif mode == 'head-float32':
        model = Float32OutputProjection(model)
    mx.eval(model.parameters())
    mx.synchronize()
    body = model.base if mode == 'head-float32' else model
    expected_body = 'mlx.core.float32' if mode == 'float32' else 'mlx.core.bfloat16'
    body_dtypes = sorted({str(v.dtype) for _, v in tree_flatten(body.parameters())})
    if body_dtypes != [expected_body]:
        raise ValueError('Model body dtype differs from the experiment contract')
    parameters = {id(v): v for _, v in tree_flatten(model.parameters())}
    metadata = dict(body_parameter_dtypes=body_dtypes,
        parameter_dtypes=sorted({str(v.dtype) for v in parameters.values()}),
        expected_cache_dtype=expected_body,
        expected_logits_dtype='mlx.core.bfloat16' if mode == 'bf16' else 'mlx.core.float32',
        retained_parameter_bytes=sum(v.nbytes for v in parameters.values()),
        additional_output_weight_bytes=model.output_weight.nbytes if mode == 'head-float32' else 0)
    return model, metadata
