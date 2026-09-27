"""Independent CSV join and exact rational OLS; no NumPy or library worker imports."""
import csv
import io
from fractions import Fraction as F
from decimal import Decimal, localcontext


def solve(features, targets):
    x = [[F(1)] + [F(str(v)) for v in row] for row in features]
    y = [F(str(v)) for v in targets]
    n = len(x[0])
    augmented = [[sum(row[i]*row[j] for row in x) for j in range(n)] +
                 [sum(row[i]*t for row,t in zip(x,y))] for i in range(n)]
    for i in range(n):
        pivot = next((j for j in range(i,n) if augmented[j][i]), None)
        if pivot is None: raise ValueError('Rank-deficient workflow reference')
        augmented[i],augmented[pivot]=augmented[pivot],augmented[i]
        scale=augmented[i][i]; augmented[i]=[v/scale for v in augmented[i]]
        for j in range(n):
            if i != j:
                scale=augmented[j][i]
                augmented[j]=[a-scale*b for a,b in zip(augmented[j],augmented[i])]
    return [row[-1] for row in augmented]


def scientific(p):
    left=list(csv.DictReader(io.StringIO(p['observations_csv'])))
    right=list(csv.DictReader(io.StringIO(p['calibration_csv'])))
    selected=[r for r in left if r['quality'] and F(r['quality'])>=1]
    joined=[]
    for l in selected:
        for r in right:
            if int(l['site'])==int(r['site']): joined.append(dict(l,**{k:v for k,v in r.items() if k!='site'}))
    joined.sort(key=lambda r:(int(r['row_id']),int(r['calibration_id'])))
    x=[[F(r[k]) for k in p['feature_order']] for r in joined]
    y=[F(r['target']) for r in joined]
    beta=solve(x,y)
    pred=[beta[0]+sum(a*b for a,b in zip(row,beta[1:])) for row in x]
    residual=[a-b for a,b in zip(pred,y)]
    values=[len(left),len(selected),len(joined),2]
    values += [int(r['row_id']) for r in joined]+[int(r['calibration_id']) for r in joined]
    values += [v for row in x for v in row]+y+beta+pred+residual+[sum(v*v for v in residual)]
    return list(map(float,values))


def regression(p):
    with localcontext() as ctx:
        ctx.prec=90
        d=lambda x: Decimal(str(x))
        x=[[d(v) for v in row] for row in p['train_features']]
        q=[[d(v) for v in row] for row in p['heldout_features']]
        n=Decimal(len(x))
        mean=[sum(row[j] for row in x)/n for j in range(2)]
        std=[(sum((row[j]-mean[j])**2 for row in x)/n).sqrt() for j in range(2)]
        std=[s if s>=Decimal('1e-12') else Decimal(1) for s in std]
        z=[[(row[j]-mean[j])/std[j] for j in range(2)] for row in x]
        zq=[[(row[j]-mean[j])/std[j] for j in range(2)] for row in q]
        raw=solve(p['train_features'],p['train_targets'])
        b=[Decimal(v.numerator)/Decimal(v.denominator) for v in raw]
        bias=b[0]+sum(b[j+1]*mean[j] for j in range(2))
        weights=[b[j+1]*std[j] for j in range(2)]
        predict=lambda rows:[b[0]+sum(b[j+1]*row[j] for j in range(2)) for row in rows]
        train=predict(x); heldout=predict(q)
        residual=[a-d(b) for a,b in zip(heldout,p['heldout_targets'])]
        values=[len(x),len(q),2]+p['train_ids']+p['heldout_ids']+mean+std
        values += [v for row in z for v in row]+[v for row in zq for v in row]+[bias]+weights+train+heldout+residual
        if p['route']!='fit': values += heldout
        values += [1]  # State unchanged; reload routes also require a different process ID.
        return list(map(float,values))


def reference(p):
    return scientific(p) if p['operation']=='scientific-workflow' else regression(p)
