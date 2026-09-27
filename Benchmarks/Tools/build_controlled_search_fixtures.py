#!/usr/bin/env python3
"""Reconstruct controlled fixtures with independent exact Fraction calculations."""
from fractions import Fraction as F
from itertools import combinations
from math import factorial, isqrt
from pathlib import Path
import hashlib
import json

import argparse
ROOT=Path(__file__).resolve().parents[1] / "Fixtures" / "controlled-search"

def dump(path,value):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def identity(path):
    raw=path.read_bytes()
    return {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}

def polynomial(model,row):
    return F(model['bias'])+sum(F(w)*F(x) for w,x in zip(model['weights'],row))+F(model['interaction'])*F(row[0])*F(row[1])

def reference(payload):
    if payload['operation']=='vector-cosine':
        query=payload['query']
        q2=sum(F(x)**2 for x in query)
        nq=F(isqrt(q2.numerator),isqrt(q2.denominator))
        assert nq*nq==q2 and nq>0
        matches=[]
        for index,row in enumerate(payload['vectors']):
            r2=sum(F(x)**2 for x in row)
            nr=F(isqrt(r2.numerator),isqrt(r2.denominator))
            assert nr*nr==r2 and nr>0
            score=sum(F(x)*F(y) for x,y in zip(query,row))/(nq*nr)
            matches.append((index,score))
        matches.sort(key=lambda pair:(-pair[1],pair[0]))
        k=payload['top_k']
        assert k==len(matches) or matches[k-1][1]>matches[k][1]
        return [value for pair in matches[:k] for value in pair], 'Exact rational dot products and exact rational norms; descending score, equal-score rows canonicalized by input index.'
    instance=list(map(F,payload['instance']))
    background=payload['background']
    baseline=[sum(F(row[i]) for row in background)/len(background) for i in range(len(instance))]
    m=len(instance)
    coalitions={frozenset(s):polynomial(payload['model'],[instance[i] if i in s else baseline[i] for i in range(m)]) for n in range(m+1) for s in combinations(range(m),n)}
    contributions=[]
    for feature in range(m):
        others=[i for i in range(m) if i!=feature]
        total=F(0)
        for size in range(m):
            for selected in combinations(others,size):
                s=frozenset(selected)
                total+=F(factorial(size)*factorial(m-size-1),factorial(m))*(coalitions[s|{feature}]-coalitions[s])
        contributions.append(total)
    base=coalitions[frozenset()]
    prediction=coalitions[frozenset(range(m))]
    assert sum(contributions)==prediction-base
    return [base]+contributions+[prediction], 'Exhaustive coalition Shapley formula using Fraction arithmetic; missing features use exact background column means.'

def main():
    for path in sorted((ROOT/'sources').glob('*.json')):
        source=json.loads(path.read_text())
        payload=source['input']
        values,method=reference(payload)
        target=ROOT/'inputs'/path.name
        dump(target,payload)
        rows=len(payload['vectors']) if payload['operation']=='vector-cosine' else len(payload['background'])
        result={'operation':payload['operation'],'referenceBasis':'controlled-reference-v1','source':{'path':'sources/'+path.name,**identity(path)},'inputIdentity':identity(target),'model':{'observations':rows},'independentReference':{'values':[float(x) for x in values],'exactValues':[str(x) for x in values],'method':method},'outputOrder':source['outputOrder']}
        dump(ROOT/'references'/path.name,result)
        print(path.stem,len(values),[str(x) for x in values])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=ROOT)
    ROOT=parser.parse_args().base_dir.resolve()
    main()
