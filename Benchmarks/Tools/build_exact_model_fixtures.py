#!/usr/bin/env python3
"""Generate small numerical contracts using only exact rational arithmetic."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def identity(content):
    return {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def matrix(values):
    return [[F(v) for v in row] for row in values]


def dot(a, b):
    assert len(a) == len(b)
    return sum((x*y for x,y in zip(a,b)), F(0))


def gram(a, b):
    return [[dot(x,y) for y in b] for x in a]


def flatten(rows):
    return [x for row in rows for x in row]


def emit(base, name, payload, source, model, segments):
    content = encode(payload)
    input_path = base/'inputs'/f'{name}.json'
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(content)
    values = []
    output_order = []
    for label, shape, part in segments:
        count = 1
        for dim in shape:
            count *= dim
        assert len(part) == count
        output_order.append({'name':label, 'shape':shape, 'offset':len(values), 'count':count, 'layout':'row-major'})
        values += [F(x) for x in part]
    reference = {
        'schema_version':1, 'name':name, 'operation':payload['operation'],
        'source':source, 'inputIdentity':identity(content), 'model':model,
        'outputOrder':output_order,
        'exactRationalReference':{
            'method':'Python standard-library Fraction arithmetic; source specification supplies exact eigensystem or class probabilities, verified algebraically by generator',
            'values':[float(v) for v in values],
            'rationalValues':[str(v) for v in values]
        },
        'tolerances':{'atol':1e-12,'rtol':1e-12}
    }
    ref_path = base/'references'/f'{name}.json'
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_bytes(encode(reference))
    return {'name':name, 'operation':payload['operation'], 'input':str(input_path.relative_to(base)),
            'reference':str(ref_path.relative_to(base)), 'rows':model['observations'],
            'output_values':len(values), 'inputIdentity':identity(content), 'referenceIdentity':identity(ref_path.read_bytes())}


def build(base):
    source_path = base/'source-spec.json'
    source_bytes = source_path.read_bytes()
    spec = json.loads(source_bytes)
    source = {'name':spec['name'], 'path':'source-spec.json', **identity(source_bytes),
              'kind':'synthetic rational specification', 'generator':'build_exact_model_fixtures.py'}
    results = []
    pca = spec['pca']
    x, query = matrix(pca['features']), matrix(pca['query'])
    n, p, q = len(x), len(x[0]), len(query)
    mean = [F(v) for v in pca['mean']]
    components = matrix(pca['components'])
    variance = [F(v) for v in pca['explained_variance']]
    assert mean == [sum(row[j] for row in x)/n for j in range(p)]
    centered = [[row[j]-mean[j] for j in range(p)] for row in x]
    covariance = [[sum(row[i]*row[j] for row in centered)/(n-1) for j in range(p)] for i in range(p)]
    assert gram(components,components) == [[1,0],[0,1]]
    for c, value in zip(components,variance):
        assert [dot(row,c) for row in covariance] == [value*v for v in c]
    assert variance == sorted(variance,reverse=True)
    total_variance = sum(covariance[j][j] for j in range(p))
    for k in pca['component_counts']:
        basis = components[:k]
        scores = [[dot(row,c) for c in basis] for row in centered]
        query_scores = [[dot([row[j]-mean[j] for j in range(p)],c) for c in basis] for row in query]
        projectors = [a*b for c in basis for a in c for b in c]
        payload = {'operation':'pca-cpu','features':pca['features'],'query':pca['query'],'n_components':k}
        segments = [
            ('mean',[p],mean), ('explainedVariance',[k],variance[:k]),
            ('explainedVarianceRatio',[k],[v/total_variance for v in variance[:k]]),
            ('componentProjectors',[k,p,p],projectors),
            ('trainingScoreGram',[n,n],flatten(gram(scores,scores))),
            ('queryScoreGram',[q,q],flatten(gram(query_scores,query_scores))),
            ('trainingQueryScoreGram',[n,q],flatten(gram(scores,query_scores)))
        ]
        results.append(emit(base,f'pca-oblique-k{k}',payload,source,
                            {'observations':n,'features':p,'queryObservations':q,'components':k,
                             'device':'cpu','solver':'full','precision':'float64','varianceNormalization':'n-1'},segments))
    nb = spec['naive_bayes']
    features = matrix(nb['features'])
    labels = [F(v) for v in nb['targets']]
    classes = [F(v) for v in nb['classes']]
    alpha = F(nb['alpha'])
    assert classes == sorted(set(labels))
    priors = [F(v) for v in nb['class_priors']]
    likelihoods = matrix(nb['feature_probabilities'])
    for c,label in enumerate(classes):
        subset = [row for row,target in zip(features,labels) if target == label]
        assert priors[c] == F(len(subset),len(features))
        counts = [sum(row[j] for row in subset)+alpha for j in range(len(features[0]))]
        assert likelihoods[c] == [v/sum(counts) for v in counts]
    probabilities = []
    predictions = []
    for row in nb['query']:
        assert all(type(v) is int and v >= 0 for v in row)
        joint = []
        for prior, likelihood in zip(priors,likelihoods):
            value = prior
            for count,probability in zip(row,likelihood):
                value *= probability**count
            joint.append(value)
        probabilities.append([v/sum(joint) for v in joint])
        predictions.append(max(range(len(classes)),key=joint.__getitem__))
    assert probabilities == [[F(1,3),F(2,3)],[F(3,23),F(20,23)],[F(9,13),F(4,13)],[F(27,67),F(40,67)],[F(81,89),F(8,89)]]
    payload = {'operation':'multinomial-nb-cpu',**{key:nb[key] for key in ['features','targets','query','alpha']}}
    results.append(emit(base,'multinomial-nb-noncontiguous',payload,source,
                        {'observations':len(features),'features':len(features[0]),'queryObservations':len(nb['query']),
                         'classes':len(classes),'device':'cpu','precision':'float64','predictionMeaning':'sorted-class indices'},
                        [('classes',[len(classes)],classes),('probabilities',[len(nb['query']),len(classes)],flatten(probabilities)),
                         ('predictionIndices',[len(nb['query'])],predictions)]))
    (base/'inventory.json').write_bytes(encode({'source':source,'reference_basis':'exact-rational-v1','fixtures':results}))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir',type=Path,default=Path(__file__).resolve().parents[1] / 'Fixtures' / 'exact-models')
    build(parser.parse_args().base_dir.resolve())
