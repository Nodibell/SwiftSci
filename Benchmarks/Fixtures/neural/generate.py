#!/usr/bin/env python3
"""Build original fixed decoder tensors and independent high-precision logits."""
import copy
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'Benchmarks/Tools'))
from contracts import digest, write_json
from neural_fixtures import SHAPES, validate_input
from neural_reference import reference


def cases():
    weights = {}
    for number, (name, shape) in enumerate(SHAPES.items()):
        values = ([.75 + (i % 4)/8 for i in range(8)] if len(shape)==1 else
                  [((i*7 + (i//shape[-1])*3 + number*5) % 17 - 8)/32 for i in range(math.prod(shape))])
        weights[name] = {'shape': shape, 'values': values}
    base = {'operation': 'decoder-fixed-f32', 'device': 'cpu', 'position': 'learned',
            'execution': 'full', 'loading': 'direct', 'tokens': [[0,1,4,2],[6,3,1,5]], 'weights': weights}
    result = []
    for device in ('cpu', 'gpu'):
        for position in ('learned', 'rope'):
            for execution in ('full', 'cached'):
                p = copy.deepcopy(base);p.update(device=device, position=position, execution=execution)
                result.append((f'neural-{device}-{position}-{execution}', p))
        zero = copy.deepcopy(base);zero['device'] = device;zero['tokens'] = [[0,1,2,3]]
        for name, w in zero['weights'].items():
            w['values'] = [1.0 if len(w['shape'])==1 else 0.0]*math.prod(w['shape'])
        zero['weights']['embedding.weight']['values'] = [float(i==j) for i in range(7) for j in range(8)]
        zero['weights']['lmHead.weight']['values'] = [float(i==j) for i in range(7) for j in range(8)]
        result.append((f'neural-{device}-zero-projections', zero))
        p = copy.deepcopy(base);p.update(device=device, loading='public-loader', position='learned' if device=='cpu' else 'rope')
        result.append((f'neural-{device}-public-loader', p))
    for batch in range(2):
        p=copy.deepcopy(base);p.update(device="gpu",position="rope",execution="cached",tokens=[base["tokens"][batch]])
        result.append((f"neural-gpu-rope-cached-single{batch}",p))
    return result


def generate():
    lock_path = ROOT/'Benchmarks/Fixtures/neural/sources.lock.json'
    sources=[]
    for relative in ['generate.py','../../Tools/neural_reference.py','../../Python/requirements-standardized.txt']:
        data=(lock_path.parent/relative).read_bytes()
        sources.append({'path':relative,'sha256':digest(data),'bytes':len(data)})
    write_json(lock_path,{'schema_version':1,'sources':sources})
    source = lock_path.read_bytes()
    profiles = {'neural-conformance': [], 'neural-cpu-conformance': [], 'neural-loader-conformance': []}
    for name, payload in cases():
        rows = sum(map(len, payload['tokens']))
        validate_input(payload, payload['operation'], rows)
        path = ROOT/f'Benchmarks/Fixtures/neural/inputs/{name}.json';write_json(path,payload);raw=path.read_bytes()
        values = reference(payload)
        refpath = ROOT/f'Benchmarks/Fixtures/neural/references/{name}.json'
        write_json(refpath, {'operation':payload['operation'], 'inputIdentity':{'sha256':digest(raw),'bytes':len(raw)},
                            'source':{'sha256':digest(source)}, 'model':{'observations':rows},
                            'independentReference':{'precision_decimal_digits':80,'values':values}})
        ref=refpath.read_bytes()
        write_json(ROOT/f'Benchmarks/Specs/datasets/{name}.json',{
            'schema_version':1,'id':name,'kind':'numerical-fixture-v1','rows':rows,
            'sha256':digest(raw),'size_bytes':len(raw),'source':'Original fixed tensors; see Benchmarks/Fixtures/neural/README.md',
            'license':'MIT, repository LICENSE','generator_version':1,'fixture':str(path.relative_to(ROOT)),
            'source_fixture':str(lock_path.relative_to(ROOT)),'source_sha256':digest(source),'source_size_bytes':len(source),
            'reference_fixture':str(refpath.relative_to(ROOT)),'reference_sha256':digest(ref),'reference_size_bytes':len(ref),
            'reference_basis':'neural-reference-v1','operation':payload['operation'],'tolerances':{'atol':2e-5,'rtol':2e-5}})
        profiles['neural-loader-conformance' if payload['loading']=='public-loader' else 'neural-conformance'].append(
            {'id':name,'dataset':name,'workload':'decoder-fixed-f32-v1'})
    profiles['neural-cpu-conformance'] = [c for c in profiles['neural-conformance'] if c['id'].startswith('neural-cpu-')]
    for name,cases_ in profiles.items():
        write_json(ROOT/f'Benchmarks/Specs/profiles/{name}.json',{'schema_version':1,'id':name,'warmups':1,'samples':2,'batches':1,'timeout_seconds':120,'cases':cases_})


if __name__=='__main__':
    generate()
