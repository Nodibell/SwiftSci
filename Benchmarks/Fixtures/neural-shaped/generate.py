#!/usr/bin/env python3
"""Generate bounded shape/context packs and complete independent scalar answers."""
import copy
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'Benchmarks/Tools'))
from contracts import digest, write_json
from neural_shapes import shapes, validate_input
from neural_shaped_reference import reference


def cases():
    for width, heads, length, batch, vocab, position in [
        (16, 2, 8, 2, 11, 'learned'),
        (32, 4, 32, 1, 17, 'rope'),
        (32, 2, 128, 1, 17, 'rope'),
    ]:
        config=dict(vocab_size=vocab, hidden_dim=width, num_heads=heads,
                    intermediate_size=width*3//2, max_seq_len=length)
        weights={}
        for number,(name,shape) in enumerate(shapes(config).items()):
            values=([.75+(i%4)/8 for i in range(width)] if len(shape)==1 else
                    [((i*7+(i//shape[-1])*3+number*5)%17-8)/128 for i in range(math.prod(shape))])
            weights[name]=dict(shape=shape,values=values)
        payload=dict(operation='decoder-shaped-f32',config=config,device='cpu',position=position,
                     execution='full',loading='direct',chunk_sizes=[],weights=weights,
                     tokens=[[(i*3+b*5)%vocab for i in range(length)] for b in range(batch)])
        yield f'w{width}-h{heads}-s{length}-b{batch}',payload


def generate():
    directory=ROOT/'Benchmarks/Fixtures/neural-shaped'
    lock=directory/'sources.lock.json'
    sources=[]
    for relative in ['generate.py','../../Tools/neural_shapes.py','../../Tools/neural_shaped_reference.py','../../Python/requirements-standardized.txt']:
        raw=(directory/relative).read_bytes()
        sources.append(dict(path=relative,sha256=digest(raw),bytes=len(raw)))
    write_json(lock,dict(schema_version=1,sources=sources));source=lock.read_bytes()
    profiles={'neural-shaped-conformance':[],'neural-shaped-cpu-conformance':[]}
    for shape,base in cases():
        rows=sum(map(len,base['tokens']))
        validate_input(base,base['operation'],rows)
        print('Independent oracle '+shape,flush=True)
        full=reference(base)
        for device in ('cpu','gpu'):
            for execution in ('full','cached'):
                p=copy.deepcopy(base);p.update(device=device,execution=execution)
                if execution=='cached':
                    length=len(p['tokens'][0]);p['chunk_sizes']=[length//2]+[1]*(length-length//2)
                validate_input(p,p['operation'],rows)
                name=f'neural-shaped-{device}-{shape}-{execution}'
                values=full.copy()
                if execution=='cached':
                    chunks=p['chunk_sizes'];values += [len(chunks)]+[sum(chunks[:i+1]) for i in range(len(chunks))]
                path=directory/f'inputs/{name}.json';write_json(path,p);raw=path.read_bytes()
                refpath=directory/f'references/{name}.json'
                write_json(refpath,dict(operation=p['operation'],inputIdentity=dict(sha256=digest(raw),bytes=len(raw)),
                    source=dict(sha256=digest(source)),model=dict(observations=rows),
                    independentReference=dict(precision_decimal_digits=80,values=values)))
                ref=refpath.read_bytes()
                write_json(ROOT/f'Benchmarks/Specs/datasets/{name}.json',dict(
                    schema_version=1,id=name,kind='numerical-fixture-v1',rows=rows,sha256=digest(raw),size_bytes=len(raw),
                    source='Original bounded decoder tensors; see Benchmarks/Fixtures/neural-shaped/README.md',
                    license='MIT, repository LICENSE',generator_version=1,fixture=str(path.relative_to(ROOT)),
                    source_fixture=str(lock.relative_to(ROOT)),source_sha256=digest(source),source_size_bytes=len(source),
                    reference_fixture=str(refpath.relative_to(ROOT)),reference_sha256=digest(ref),reference_size_bytes=len(ref),
                    reference_basis='neural-shaped-reference-v1',operation=p['operation'],tolerances=dict(atol=2e-5,rtol=2e-5)))
                case=dict(id=name,dataset=name,workload='decoder-shaped-f32-v1')
                profiles['neural-shaped-conformance'].append(case)
                if device=='cpu': profiles['neural-shaped-cpu-conformance'].append(case)
    for name,cases_ in profiles.items():
        write_json(ROOT/f'Benchmarks/Specs/profiles/{name}.json',dict(schema_version=1,id=name,
            warmups=1,samples=3,batches=1,timeout_seconds=120,cases=cases_))


if __name__=='__main__': generate()
