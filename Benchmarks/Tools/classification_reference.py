"""High-precision full-batch logistic updates and independent classification scores."""
import mpmath as mp

METRIC_ORDER = ['TN','FP','FN','TP','accuracy','precision','recall','F1','log_loss','ROC_AUC','Brier']


def scores(targets, probabilities):
    labels = [int(p > mp.mpf('.5')) for p in probabilities]
    counts = [sum(int(y == actual and label == predicted) for y,label in zip(targets,labels))
              for actual,predicted in [(0,0),(0,1),(1,0),(1,1)]]
    tn,fp,fn,tp = counts
    precision = mp.mpf(tp)/(tp+fp) if tp+fp else mp.mpf(0)
    recall = mp.mpf(tp)/(tp+fn)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else mp.mpf(0)
    epsilon = mp.mpf(1e-15)
    clipped = [max(epsilon,min(1-epsilon,p)) for p in probabilities]
    loss = -mp.fsum(y*mp.log(p)+(1-y)*mp.log1p(-p) for y,p in zip(targets,clipped))/len(targets)
    positives = [p for y,p in zip(targets,probabilities) if y == 1]
    negatives = [p for y,p in zip(targets,probabilities) if y == 0]
    auc = mp.fsum(1 if a>b else mp.mpf('.5') if a==b else 0 for a in positives for b in negatives)/(len(positives)*len(negatives))
    brier = mp.fsum((y-p)**2 for y,p in zip(targets,probabilities))/len(targets)
    return counts+[mp.mpf(tp+tn)/len(targets),precision,recall,f1,loss,auc,brier]


def reference(payload, means, scales, scaled):
    names = ('train','validation','test')
    targets = {k:[mp.mpf(payload['targets'][i]) for i in payload['splits'][k]] for k in names}
    x,y = scaled['train'],targets['train']
    weights = [mp.mpf(0)]*len(means); bias = mp.mpf(0)
    rate = mp.mpf(payload['training']['learning_rate'])
    for _ in range(payload['training']['epochs']):
        residual = [1/(1+mp.exp(-(bias+mp.fsum(w*v for w,v in zip(weights,row)))))-t for row,t in zip(x,y)]
        gradient = [mp.fsum(r*row[j] for r,row in zip(residual,x))/len(y) for j in range(len(weights))]
        weights = [w-rate*g for w,g in zip(weights,gradient)]
        bias -= rate*mp.fsum(residual)/len(y)
    probabilities = {k:[1/(1+mp.exp(-(bias+mp.fsum(w*v for w,v in zip(weights,row))))) for row in scaled[k]] for k in names}
    prevalence = mp.fsum(y)/len(y)
    result = means+scales+[bias]+weights
    result += [v for k in names for p in probabilities[k] for v in [1-p,p]]
    result += [int(p > mp.mpf('.5')) for k in names for p in probabilities[k]]
    result += [prevalence]
    for k in ('validation','test'):
        result += scores(targets[k],probabilities[k])+scores(targets[k],[prevalence]*len(targets[k]))
    return result
