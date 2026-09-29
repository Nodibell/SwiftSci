#!/usr/bin/env python3
"""Freeze original boundary cases and bounded stage/shape recipes with independent answers."""
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Benchmarks/Tools'))
from contracts import digest,write_json
from boundary_cases import cases
from boundary_reference import reference
from boundary_fixtures import validate_input
from boundary_sweep import validate_descriptor, reference as sweep_reference


def sweep_cases():
    for device,dtype in [('cpu','float32'),('cpu','float64'),('gpu','float32')]:
        for rows in (128,1024,8192):
            for columns in (8,64):
                for stage in ('conversion','prepared','pipeline'):
                    yield f'sweep-{device}-{dtype}-{rows}x{columns}-{stage}',dict(operation='dataframe-model-sweep',recipe='dyadic-v1',device=device,dtype=dtype,stage=stage,rows=rows,columns=columns)


def generate():
    directory=ROOT/'Benchmarks/Fixtures/boundary'
    lock_path=directory/'sources.lock.json'
    entries=[]
    for relative in ['generate.py','../../Tools/boundary_cases.py','../../Tools/boundary_reference.py','../../Tools/boundary_sweep.py','../../Tools/boundary_fixtures.py','../../Python/requirements-standardized.txt']:
        raw=(directory/relative).read_bytes();entries.append(dict(path=relative,sha256=digest(raw),bytes=len(raw)))
    write_json(lock_path,dict(schema_version=1,sources=entries));source=lock_path.read_bytes()
    profiles={p:[] for p in ['boundary-conformance','boundary-cpu-conformance','boundary-sweep']}
    for name,payload in list(cases())+list(sweep_cases()):
        sweep=payload['operation']=='dataframe-model-sweep';rows=payload['rows'] if sweep else len(payload['row_ids'])
        (validate_descriptor if sweep else validate_input)(payload,payload['operation'],rows)
        path=directory/'inputs'/(name+'.json');write_json(path,payload);raw=path.read_bytes()
        values=(sweep_reference if sweep else reference)(payload)
        answer=dict(precision_decimal_digits=100)
        if sweep:answer.update(count=len(values),binary64_sha256=digest(struct.pack('<'+'d'*len(values),*values)))
        else:answer['values']=values
        refpath=directory/'references'/(name+'.json');write_json(refpath,dict(operation=payload['operation'],inputIdentity=dict(sha256=digest(raw),bytes=len(raw)),source=dict(sha256=digest(source)),model=dict(observations=rows),independentReference=answer));ref=refpath.read_bytes()
        tol=dict(atol=2e-5 if payload['dtype']=='float32' else 1e-12,rtol=0)
        write_json(ROOT/f'Benchmarks/Specs/datasets/{name}.json',dict(schema_version=1,id=name,kind='numerical-fixture-v1',rows=rows,sha256=digest(raw),size_bytes=len(raw),source='Original deterministic boundary fixtures; see Benchmarks/Fixtures/boundary/README.md',license='MIT, repository LICENSE',generator_version=1,fixture=str(path.relative_to(ROOT)),source_fixture=str(lock_path.relative_to(ROOT)),source_sha256=digest(source),source_size_bytes=len(source),reference_fixture=str(refpath.relative_to(ROOT)),reference_sha256=digest(ref),reference_size_bytes=len(ref),reference_basis='sweep-reference-v1' if sweep else 'boundary-reference-v1',operation=payload['operation'],tolerances=tol))
        entry=dict(id=name,dataset=name,workload=payload['operation']+'-v1')
        profiles['boundary-sweep' if sweep else 'boundary-conformance'].append(entry)
        if not sweep and payload['device']=='cpu':profiles['boundary-cpu-conformance'].append(entry)
    for name,entries in profiles.items():
        write_json(ROOT/f'Benchmarks/Specs/profiles/{name}.json',dict(schema_version=1,id=name,warmups=1,samples=2,batches=1,timeout_seconds=120,cases=entries))


if __name__=='__main__':generate()
