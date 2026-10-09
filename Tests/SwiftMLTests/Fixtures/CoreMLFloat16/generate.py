"""Generate weight-free Float16 ReLU fixtures. Requires coremltools 9.0 and NumPy."""
import hashlib
import json
from pathlib import Path
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np

root = Path(__file__).resolve().parent
files = {}
for name, shape in [('matrix', (3, 2)), ('vector', (2,))]:
    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp16)], opset_version=ct.target.iOS16)
    def program(features):
        return mb.relu(x=features, name='result')
    model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT16,
                       outputs=[ct.TensorType(name='result', dtype=np.float16)], skip_model_load=True)
    path = root / f'{name}.mlmodel'
    spec = model.get_spec()
    spec.description.metadata.userDefined.pop("com.github.apple.coremltools.conversion_date", None)
    path.write_bytes(spec.SerializeToString(deterministic=True))
    files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'manifest.json').write_text(json.dumps(dict(coremltools=ct.__version__, numpy=np.__version__,
                                                  operation='elementwise ReLU', files=files), indent=2) + '\n')
