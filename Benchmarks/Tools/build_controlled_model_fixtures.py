#!/usr/bin/env python3
"""Generate controlled fixture answers independently with Fraction and Decimal."""
import argparse
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def identity(data):
    return {'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def matrix(rows):
    return [[F(str(x)) for x in row] for row in rows]


def transpose(a):
    return [list(row) for row in zip(*a)]


def mul(a,b):
    bt=transpose(b)
    assert len(a[0]) == len(b)
    return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in bt] for row in a]


def add(a,b,sign=1):
    return [[x+sign*y for x,y in zip(row,other)] for row,other in zip(a,b)]


def inverse(a):
    n=len(a)
    rows=[row[:] + [F(i==j) for j in range(n)] for i,row in enumerate(a)]
    for c in range(n):
        pivot=next(r for r in range(c,n) if rows[r][c])
        rows[c],rows[pivot]=rows[pivot],rows[c]
        factor=rows[c][c]
        rows[c]=[x/factor for x in rows[c]]
        for r in range(n):
            if r != c:
                factor=rows[r][c]
                rows[r]=[x-factor*y for x,y in zip(rows[r],rows[c])]
    return [row[n:] for row in rows]


def flat(rows):
    return [x for row in rows for x in row]


def kalman(payload):
    transition=matrix(payload['transition']); h=matrix(payload['observation_matrix'])
    q=matrix(payload['process_noise']); r=matrix(payload['measurement_noise'])
    state=matrix([[x] for x in payload['initial_mean']]); p=matrix(payload['initial_covariance'])
    values=[]
    for measurement in payload['observations']:
        predicted=mul(transition,state)
        pp=add(mul(mul(transition,p),transpose(transition)),q)
        innovation=add(matrix([[x] for x in measurement]),mul(h,predicted),-1)
        covariance=add(mul(mul(h,pp),transpose(h)),r)
        gain=mul(mul(pp,transpose(h)),inverse(covariance))
        state=add(predicted,mul(gain,innovation))
        # Exact covariance subtraction, independent of the production Joseph-form path.
        p=add(pp,mul(mul(gain,h),pp),-1)
        assert p == transpose(p)
        values += flat(state)+flat(p)
    next_state=mul(transition,state)
    next_p=add(mul(mul(transition,p),transpose(transition)),q)
    return values,flat(next_state)+flat(next_p)


def answers(payload):
    op=payload['operation']
    if op in ('linear-fixed-cpu','logistic-fixed-cpu'):
        x=matrix(payload['features']); weights=[F(v) for v in payload['weights']]; bias=F(payload['bias'])
        logits=[sum((a*b for a,b in zip(row,weights)),bias) for row in x]
        segments=[('parameters',[len(weights)+1],[bias]+weights)]
        model={'observations':len(x),'features':len(weights),'device':'cpu','precision':'float64','parameterOrder':['bias','feature-order weights']}
        if op == 'linear-fixed-cpu':
            segments += [('predictions',[len(x)],logits)]
            method='Exact Fraction affine dot products with supplied weights and bias'
            assert logits == [3,3,-5,10]
        else:
            probabilities=[]
            with localcontext() as ctx:
                ctx.prec=80
                for z in logits:
                    zd=D(z.numerator)/D(z.denominator)
                    p=D(1)/(D(1)+(-zd).exp())
                    probabilities += [D(1)-p,p]
            labels=[int(z>0) for z in logits]
            assert logits == [0,2,-2,-3,0] and labels == [0,1,0,0,0]
            segments += [('probabilities',[len(x),2],probabilities),('labels',[len(x)],labels)]
            method='Exact Fraction logits; Decimal sigmoid and complementary probabilities at 80 digits; strict probability > 0.5 labels'
            model['threshold']=0.5
        return segments,model,method
    if op == 'kmeans-one-cpu':
        x=matrix(payload['features']); n=len(x); p=len(x[0]); nq=len(payload['query'])
        centroid=[sum(row[j] for row in x)/n for j in range(p)]
        inertia=sum((value-centroid[j])**2 for row in x for j,value in enumerate(row))
        assert centroid == [2,1] and inertia == 20
        segments=[('centroids',[1,p],centroid),('trainingLabels',[n],[0]*n),('queryLabels',[nq],[0]*nq),('inertia',[1],[inertia])]
        model={'observations':n,'features':p,'queryObservations':nq,'clusters':1,'device':'cpu','precision':'float64'}
        return segments,model,'Exact Fraction arithmetic mean and sum of squared distances; one cluster fixes every label to zero'
    if op == 'kalman-fixed-cpu':
        states,predicted=kalman(payload)
        n=len(payload['observations']); d=payload['state_size']; width=d+d*d
        if d==1:
            assert states+predicted == list(map(F,['2/3','2/3','3/2','5/8','17/7','13/21','17/7','34/21']))
        segments=[('filteredStates',[n,width],states),('nextPrediction',[width],predicted)]
        model={'observations':n,'stateSize':d,'observationSize':payload['observation_size'],'stateOrder':['mean','row-major covariance'],
               'device':'cpu implementation; no public routing API','precision':'float64','predictUpdatesInternalState':False}
        return segments,model,'Exact Fraction Kalman recurrence with supplied matrices; Ppost = Pprior - K H Pprior; exact Gauss-Jordan innovation inverse'
    raise ValueError(op)


def build(base):
    source_bytes=(base/'source-spec.json').read_bytes()
    source=json.loads(source_bytes)
    source_record={'name':source['name'],'path':'source-spec.json','kind':'synthetic controlled model specification',**identity(source_bytes)}
    fixtures=[]
    for fixture in source['fixtures']:
        name,payload=fixture['name'],fixture['input']
        segments,model,method=answers(payload)
        content=encoded(payload)
        input_path=base/'inputs'/f'{name}.json'; input_path.parent.mkdir(parents=True,exist_ok=True);input_path.write_bytes(content)
        values=[]; order=[]
        for label,shape,part in segments:
            count=1
            for dimension in shape: count*=dimension
            assert count==len(part)
            order.append({'name':label,'shape':shape,'offset':len(values),'count':count,'layout':'row-major'})
            values+=part
        reference={'schema_version':1,'name':name,'operation':payload['operation'],'source':source_record,
                   'inputIdentity':identity(content),'model':model,'outputOrder':order,
                   'independentReference':{'method':method,'values':[float(v) for v in values],'exactValues':[str(v) for v in values]},
                   'tolerances':{'atol':1e-12,'rtol':1e-12}}
        ref_path=base/'references'/f'{name}.json';ref_path.parent.mkdir(parents=True,exist_ok=True);ref_path.write_bytes(encoded(reference))
        shapes={}
        for key,value in payload.items():
            if isinstance(value,list): shapes[key]=[len(value)]+([len(value[0])] if value and isinstance(value[0],list) else [])
            elif key!='operation': shapes[key]=[]
        fixtures.append({'name':name,'operation':payload['operation'],'rows':model['observations'],'input':str(input_path.relative_to(base)),
                         'reference':str(ref_path.relative_to(base)),'inputFields':list(payload),'inputShapes':shapes,
                         'output_values':len(values),'outputOrder':order,'inputIdentity':identity(content),'referenceIdentity':identity(ref_path.read_bytes())})
    (base/'inventory.json').write_bytes(encoded({'reference_basis':'controlled-reference-v1','source':source_record,'fixtures':fixtures}))
    for item in fixtures: print(f'{item["name"]}: {item["output_values"]} independently generated outputs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parents[1] / "Fixtures" / "controlled-models")
    build(parser.parse_args().base_dir.resolve())
