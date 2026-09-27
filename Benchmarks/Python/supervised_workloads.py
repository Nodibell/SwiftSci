"""NumPy/SciPy implementation of the frozen supervised workflow."""
import numpy as np
from scipy import linalg


def prepare(payload):
    return payload, np.asarray(payload['features'],dtype=np.float64), np.asarray(payload['targets'],dtype=np.float64)


def metrics(targets,predictions):
    errors=targets-predictions
    rss=np.dot(errors,errors)
    centered=targets-targets.mean()
    return [np.sqrt(rss/len(targets)),np.mean(np.abs(errors)),1-rss/np.dot(centered,centered)]


def execute(prepared):
    payload,x,y=prepared
    names=('train','validation','test')
    splits=payload['splits']; train=x[splits['train']]
    means=train.mean(axis=0); scales=train.std(axis=0,ddof=0)
    scales=np.where(scales<1e-12,1.,scales)
    z={name:(x[splits[name]]-means)/scales for name in names}
    result=[means,scales]
    if payload['operation']=='supervised-scale':
        return np.concatenate(result+[z[name].ravel() for name in names])
    train_y=y[splits['train']]
    design=np.column_stack((np.ones(len(train)),z['train']))
    beta,_,rank,_=linalg.lstsq(design,train_y,lapack_driver='gelsy')
    if rank!=design.shape[1]: raise ValueError('Rank-deficient supervised training matrix')
    predictions={name:beta[0]+z[name]@beta[1:] for name in names}
    baseline=train_y.mean()
    result += [beta]+[predictions[name] for name in names]+[[baseline]]
    for name in ('validation','test'):
        target=y[splits[name]]
        result += [metrics(target,predictions[name]),metrics(target,np.full(len(target),baseline))]
    return np.concatenate(result)
