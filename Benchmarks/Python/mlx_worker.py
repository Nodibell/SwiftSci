#!/usr/bin/env python3
"""Native MLX comparison worker using existing validated fixture contracts."""
import hashlib
import importlib.metadata
from pathlib import Path
import resource
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tools'))
from contracts import read_json, write_json, verified_file, validate_values
from numerical_fixtures import validate_input
from mlx_workloads import MLXWorkload


def main():
    request = read_json(sys.argv[1])
    destination = Path(sys.argv[2])
    version = 'mlx unavailable'
    try:
        version = f"mlx {importlib.metadata.version('mlx')}; mlx-metal {importlib.metadata.version('mlx-metal')}; numpy {np.__version__}"
        verified_file(request['input_path'], request['input_sha256'], request['input_bytes'])
        expected = np.frombuffer(verified_file(request['expected_path'], request['expected_sha256']), dtype='<f8')
        payload = validate_input(read_json(request['input_path']), request['operation'], request['rows'])
        workload = MLXWorkload(payload)
        samples = []
        for index in range(request['warmups'] + request['samples']):
            start = time.perf_counter_ns()
            output = workload.execute()
            duration = time.perf_counter_ns() - start
            actual = np.asarray(output, dtype='<f8')
            error = validate_values(actual, expected, request['atol'], request['rtol'])
            if index >= request['warmups']:
                samples.append(dict(elapsed_ns=duration, timing_resolved=duration >= 1000,
                                    output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                                    maximum_absolute_error=error, validated=True))
            del output, actual
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        response = dict(schema_version=1, case_key=request['case_key'], status='passed', samples=samples,
                        peak_rss_bytes=rss if sys.platform == 'darwin' else rss * 1024, engine_version=version)
    except Exception as error:
        response = dict(schema_version=1, case_key=request['case_key'], status='failed', samples=[],
                        peak_rss_bytes=0, engine_version=version, error=str(error))
    write_json(destination, response)
    return 0 if response['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
