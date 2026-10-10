#!/usr/bin/env python3
"""Create matched legacy and ML Program models from a verified trained fixture."""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--fixture', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = a.parse_args()
    f = json.loads(args.fixture.read_text())
    rows, width = (f['inputBinary'][k] for k in ('rows', 'columns'))
    assert 1 <= rows <= 8192 and 1 <= width <= 1024
    args.output.mkdir(parents=True, exist_ok=False)
    for key, name in [('inputBinary', 'input.f64'), ('expectedBinary', 'expected.f64')]:
        p = args.fixture.parent / f[key]['path']
        assert sha(p) == f[key]['sha256']
        shutil.copy(p, args.output / name)
    layers = f['layers']
    weights = [np.array(x['W'], dtype=np.float32).reshape(x['inDim'], x['outDim']).T.copy() for x in layers]
    biases = [np.array(x['b'], dtype=np.float32) for x in layers]
    outputs = layers[-1]['outDim']
    # The current adapter artifact has Double I/O and Float32-stored weights.
    builder = ct.models.neural_network.NeuralNetworkBuilder(
        [('features', ct.models.datatypes.Array(rows, width))],
        [('result', ct.models.datatypes.Array(rows, outputs))], disable_rank5_shape_mapping=True)
    name = 'features'
    for i, (w,b) in enumerate(zip(weights,biases)):
        final = i == len(weights)-1
        dest = 'result' if final else f'dense_{i}'
        builder.add_inner_product(f'linear_{i}', w, b, w.shape[1], w.shape[0], True, name, dest)
        name = dest
        if not final:
            name = f'relu_{i}'
            builder.add_activation(name, 'RELU', dest, name)
    ct.models.MLModel(builder.spec, skip_model_load=True).save(str(args.output / 'legacy.mlmodel'))
    for io16 in [False, True]:
        @mb.program(input_specs=[mb.TensorSpec(shape=(rows,width), dtype=types.fp16 if io16 else types.fp32)], opset_version=ct.target.iOS16)
        def program(features):
            x = features
            for i,(w,b) in enumerate(zip(weights,biases)):
                dtype = np.float16 if io16 else np.float32
                x = mb.linear(x=x, weight=w.astype(dtype), bias=b.astype(dtype), name=f'linear_{i}')
                if i < len(weights)-1:
                    x = mb.relu(x=x, name=f'relu_{i}')
            return mb.identity(x=x, name='result')
        model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
            outputs=[ct.TensorType(name='result', dtype=np.float16 if io16 else np.float32)],
            skip_model_load=True)
        model.save(str(args.output / ('program16.mlpackage' if io16 else 'program32.mlpackage')))
    manifest = dict(fixture_sha256=sha(args.fixture), rows=rows, width=width, outputs=outputs,
                    coremltools=ct.__version__, numpy=np.__version__,
                    program_compute_precision='float16', legacy_io='double',
                    files={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__':
    main()
