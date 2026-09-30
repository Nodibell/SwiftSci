"""Experimental precision adapters for the pinned tied-embedding Llama reference."""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

MODES = ('bf16', 'head-float32', 'shared-float32', 'float32')


class SharedFloat32Embedding(nn.Module):
    """Store one wider matrix and narrow only gathered input rows."""
    def __init__(self, weight):
        super().__init__()
        self.weight = weight.astype(mx.float32)

    def __call__(self, inputs):
        return self.weight[inputs].astype(mx.bfloat16)


class Float32OutputProjection(nn.Module):
    """Keep the original body; retain an exact Float32 copy of the output weights."""
    def __init__(self, base, *, shared=False):
        super().__init__()
        if base.model_type != 'llama' or not base.args.tie_word_embeddings:
            raise ValueError('This experiment requires Llama with tied output embeddings')
        self.base = base
        self.shared = shared
        if shared:
            base.model.embed_tokens = SharedFloat32Embedding(base.model.embed_tokens.weight)
        else:
            self.output_weight = base.model.embed_tokens.weight.astype(mx.float32)

    @property
    def layers(self):
        return self.base.layers

    def __call__(self, inputs, cache=None, input_embeddings=None):
        hidden = self.base.model(inputs, cache=cache, input_embeddings=input_embeddings)
        weight = self.base.model.embed_tokens.weight if self.shared else self.output_weight
        return hidden.astype(mx.float32) @ weight.T


def prepare_precision(model, mode):
    if mode not in MODES:
        raise ValueError('Unknown Llama precision experiment')
    if mode == 'float32':
        model.set_dtype(mx.float32)
    elif mode in ('head-float32', 'shared-float32'):
        model = Float32OutputProjection(model, shared=mode == 'shared-float32')
    mx.eval(model.parameters())
    mx.synchronize()
    body = model.base if mode in ('head-float32', 'shared-float32') else model
    expected_body = 'mlx.core.float32' if mode == 'float32' else 'mlx.core.bfloat16'
    body_dtypes = sorted({str(v.dtype) for name, v in tree_flatten(body.parameters())
                          if name != 'model.embed_tokens.weight'})
    if body_dtypes and body_dtypes != [expected_body]:
        raise ValueError('Model body dtype differs from the experiment contract')
    expected_embedding = 'mlx.core.float32' if mode in ('float32', 'shared-float32') else 'mlx.core.bfloat16'
    embedding = body.model.embed_tokens.weight
    if str(embedding.dtype) != expected_embedding:
        raise ValueError('Input embedding storage dtype differs from the experiment contract')
    parameters = {id(v): v for _, v in tree_flatten(model.parameters())}
    metadata = dict(body_parameter_dtypes=body_dtypes,
        embedding_storage_dtype=str(embedding.dtype), embedding_weight_bytes=embedding.nbytes,
        parameter_dtypes=sorted({str(v.dtype) for v in parameters.values()}),
        expected_cache_dtype=expected_body,
        expected_logits_dtype='mlx.core.bfloat16' if mode == 'bf16' else 'mlx.core.float32',
        retained_parameter_bytes=sum(v.nbytes for v in parameters.values()),
        additional_output_weight_bytes=model.output_weight.nbytes if mode == 'head-float32' else 0)
    return model, metadata
