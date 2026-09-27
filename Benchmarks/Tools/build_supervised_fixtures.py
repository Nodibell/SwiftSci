#!/usr/bin/env python3
"""Rebuild frozen splits and independent references from pinned raw sources."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import mpmath as mp
from supervised_fixtures import validate_input, SPLITS

DEFAULT_BASE = Path(__file__).resolve().parents[1]/'Fixtures'/'supervised'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def partition(features):
    # Group by feature content, excluding row identity and target, so duplicate predictors cannot leak across splits.
    result = {name: [] for name in SPLITS}
    for i, row in enumerate(features):
        key = json.dumps([float(v).hex() for v in row], separators=(',', ':')).encode('ascii')
        bucket = int(sha(b'SwiftSci-supervised-split-v1\0'+key), 16) % 10
        result['train' if bucket < 6 else 'validation' if bucket < 8 else 'test'].append(i)
    return result


def read_source(path, kind):
    if kind == 'wine-red':
        reader = csv.reader(io.StringIO(path.read_text()), delimiter=';')
        names = next(reader)
        if names[-1] != 'quality' or len(names) != 12: raise ValueError('Unexpected wine schema')
        rows = list(reader)
        if len(rows) != 1599 or any(len(r) != 12 for r in rows): raise ValueError('Unexpected wine rows')
        return [str(i) for i in range(len(rows))], names[:-1], [[float(x) for x in r[:-1]] for r in rows], [float(r[-1]) for r in rows]
    if kind != 'wdbc': raise ValueError('Unknown raw source kind')
    rows = list(csv.reader(io.StringIO(path.read_text())))
    if len(rows) != 569 or any(len(r) != 32 or r[1] not in ('M','B') for r in rows): raise ValueError('Unexpected WDBC schema')
    measures = ['radius','texture','perimeter','area','smoothness','compactness','concavity','concave_points','symmetry','fractal_dimension']
    names = [f'{stat}_{name}' for stat in ['mean','standard_error','worst'] for name in measures]
    return [r[0] for r in rows], names, [[float(x) for x in r[2:]] for r in rows], [float(r[1]=='M') for r in rows]


def scaled_reference(payload):
    x = [[mp.mpf(v) for v in row] for row in payload['features']]
    train = [x[i] for i in payload['splits']['train']]
    means = [mp.fsum(row[j] for row in train)/len(train) for j in range(len(x[0]))]
    scales = [mp.sqrt(mp.fsum((row[j]-means[j])**2 for row in train)/len(train)) for j in range(len(means))]
    scales = [v if v >= mp.mpf(1e-12) else mp.mpf(1) for v in scales]
    z = {name: [[(x[i][j]-means[j])/scales[j] for j in range(len(means))] for i in payload['splits'][name]] for name in SPLITS}
    return means, scales, z


def metrics(actual, predictions):
    errors = [a-b for a,b in zip(actual,predictions)]
    rss = mp.fsum(e*e for e in errors)
    mean = mp.fsum(actual)/len(actual)
    tss = mp.fsum((v-mean)**2 for v in actual)
    if tss == 0: raise ValueError('Constant held-out target is outside R2 contract')
    return [mp.sqrt(rss/len(actual)),mp.fsum(abs(e) for e in errors)/len(actual),1-rss/tss]


def reference(payload):
    means, scales, z = scaled_reference(payload)
    output = means+scales
    if payload['operation'] == 'supervised-scale':
        return output+[v for name in SPLITS for row in z[name] for v in row]
    targets = {name: [mp.mpf(payload['targets'][i]) for i in payload['splits'][name]] for name in SPLITS}
    design = mp.matrix([[1]+row for row in z['train']])
    beta,_ = mp.qr_solve(design, mp.matrix(targets['train']))
    predictions = {name: [beta[0]+mp.fsum(b*v for b,v in zip(list(beta)[1:],row)) for row in z[name]] for name in SPLITS}
    baseline = mp.fsum(targets['train'])/len(targets['train'])
    output += list(beta)+[v for name in SPLITS for v in predictions[name]]+[baseline]
    for name in ['validation','test']:
        output += metrics(targets[name], predictions[name])+metrics(targets[name], [baseline]*len(targets[name]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE)
    args = parser.parse_args(); base = args.base_dir
    if mp.__version__ != '1.4.1': raise ValueError('mpmath 1.4.1 required')
    mp.mp.dps = 80
    lock_path = base/'sources.lock.json'; lock_bytes = lock_path.read_bytes(); lock = json.loads(lock_bytes)
    inventory=[]
    for source in lock['sources']:
        raw=base/source['path']; content=raw.read_bytes()
        if sha(content)!=source['sha256'] or len(content)!=source['bytes']: raise ValueError('Raw source mismatch')
        ids,names,x,y=read_source(raw,source['kind']); splits=partition(x)
        split_artifact={'schema_version':1,'source_sha256':source['sha256'],'algorithm':'feature-content-sha256-v1; buckets 0..5 train, 6..7 validation, 8..9 test; source order retained','indices':splits,'row_ids':{k:[ids[i] for i in v] for k,v in splits.items()}}
        write(base/'splits'/f"{source['kind']}.json",split_artifact)
        for operation in (['supervised-scale','supervised-ols-cpu'] if source['kind']=='wine-red' else ['supervised-scale']):
            name=source['kind']+'-'+operation
            payload=dict(operation=operation,row_ids=ids,feature_names=names,features=x,targets=y,splits=splits)
            validate_input(payload,operation,len(x)); input_path=base/'inputs'/f'{name}.json';write(input_path,payload)
            values=reference(payload)
            ref={'operation':operation,'source':{'sha256':sha(lock_bytes)},'raw_source_sha256':source['sha256'],'inputIdentity':{'sha256':sha(input_path.read_bytes()),'bytes':input_path.stat().st_size},'model':{'observations':len(x)},'method':'mpmath 1.4.1 at 80 decimal digits; exact binary64 raw inputs; population scaling from training rows; high precision QR for OLS','independentReference':{'values':[float(v) for v in values]},'split_counts':{k:len(v) for k,v in splits.items()}}
            write(base/'references'/f'{name}.json',ref)
            inventory.append({'id':name,'operation':operation,'rows':len(x),'features':len(names),'split_counts':ref['split_counts'],'output_count':len(values)})
    write(base/'inventory.json',inventory)


if __name__ == '__main__': main()
