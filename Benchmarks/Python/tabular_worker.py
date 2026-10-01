#!/usr/bin/env python3
"""Fresh-process native tabular worker using the existing evidence protocol."""
import hashlib
from pathlib import Path
import resource
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Tools'))
from contracts import read_json,write_json,verified_file,validate_values
from tabular_workloads import TabularWorkload

engine,request_path,destination=sys.argv[1:]
request=read_json(Path(request_path));destination=Path(destination)
module=__import__(engine)
version=f'{engine} {module.__version__}; numpy {np.__version__}; python {sys.version}'
work=None
try:
    if engine=='polars' and module.thread_pool_size()!=1: raise ValueError('Expected one Polars thread')
    verified_file(request['input_path'],request['input_sha256'],request['input_bytes'])
    expected=np.frombuffer(verified_file(request['expected_path'],request['expected_sha256']),dtype='<f8')
    work=TabularWorkload(engine,request)
    samples=[]
    for index in range(request['warmups']+request['samples']):
        start=time.perf_counter_ns();output=work.execute(destination,index);duration=time.perf_counter_ns()-start
        actual=work.canonical(output)
        error=validate_values(actual,expected,request['atol'],request['rtol'])
        if index>=request['warmups']:
            samples.append(dict(elapsed_ns=duration,timing_resolved=duration>=1000,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),maximum_absolute_error=error,validated=True))
        del output,actual
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(destination,dict(schema_version=1,case_key=request['case_key'],status='passed',samples=samples,peak_rss_bytes=rss if sys.platform=='darwin' else rss*1024,engine_version=version))
except Exception as error:
    write_json(destination,dict(schema_version=1,case_key=request['case_key'],status='failed',samples=[],peak_rss_bytes=0,engine_version=version,error=str(error)))
    raise SystemExit(1)
finally:
    if work is not None: work.close()
