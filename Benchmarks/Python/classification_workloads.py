"""NumPy full-batch logistic updates with SciPy probability/rank primitives."""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata


def scores(targets, probabilities):
    labels = probabilities > .5
    tn = np.count_nonzero((targets==0)&~labels); fp = np.count_nonzero((targets==0)&labels)
    fn = np.count_nonzero((targets==1)&~labels); tp = np.count_nonzero((targets==1)&labels)
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    p = np.clip(probabilities,1e-15,1-1e-15)
    loss = -np.mean(targets*np.log(p)+(1-targets)*np.log1p(-p))
    npos = np.count_nonzero(targets==1); nneg = len(targets)-npos
    auc = (rankdata(probabilities,method='average')[targets==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    return [tn,fp,fn,tp,(tn+tp)/len(targets),precision,recall,f1,loss,auc,np.mean((targets-probabilities)**2)]


def execute(payload, means, scales, scaled, targets):
    names=('train','validation','test');splits=payload['splits']
    y=targets[splits['train']];x=scaled['train']
    weights=np.zeros(x.shape[1],dtype=np.float64);bias=0.
    rate=payload['training']['learning_rate']
    for _ in range(payload['training']['epochs']):
        errors=expit(x@weights+bias)-y
        weights=weights-rate*(x.T@errors/len(y))
        bias-=rate*errors.mean()
    probabilities={k:expit(scaled[k]@weights+bias) for k in names}
    prevalence=y.mean()
    result=[means,scales,[bias],weights]
    result += [np.column_stack((1-probabilities[k],probabilities[k])).ravel() for k in names]
    result += [(probabilities[k]>.5).astype(np.float64) for k in names]+[[prevalence]]
    for k in ('validation','test'):
        truth=targets[splits[k]]
        result += [scores(truth,probabilities[k]),scores(truth,np.full(len(truth),prevalence))]
    return np.concatenate(result)
