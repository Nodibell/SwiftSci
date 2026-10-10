#!/usr/bin/env python3
"""Bounded multiclass stress comparison using Covertype's original data split."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import warnings

for key in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info
import sklearn
from run import run_case, validate_comparability
from trained import save
from runner import source_identity, verify_uninstrumented, metal_build_record

ROOT = Path(__file__).resolve().parents[2]
BATCHES = [1, 32, 256, 1024, 8192]
WIDTHS = [128, 256, 512]
MODES = ['coreml-cpu-matrix', 'coreml-cpu-matrix-adapter',
         'coreml-gpu-matrix-adapter', 'coreml-neural-matrix-adapter', 'mlx-cpu', 'mlx-gpu']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parameters(layers):
    weights = [np.array(x['W']).reshape(x['inDim'], x['outDim']) for x in layers]
    biases = [np.array(x['b']) for x in layers]
    return weights, biases


def predict(x, parameters):
    weights, biases = parameters
    value = x
    for i, (w, b) in enumerate(zip(weights, biases)):
        value = value @ w + b
        if i+1 < len(weights):
            value = np.maximum(value, 0)
    return value


def quality(y, logits):
    probabilities = softmax(logits, axis=1)
    matrix = confusion_matrix(y, logits.argmax(axis=1), labels=np.arange(7))
    return quality_counts(matrix, float(log_loss(y, probabilities, labels=np.arange(7))))


def quality_counts(matrix, loss):
    tp = np.diag(matrix)
    denom = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2*tp, denom, out=np.zeros(7), where=denom != 0)
    recall = np.divide(tp, matrix.sum(axis=1), out=np.zeros(7), where=matrix.sum(axis=1) != 0)
    return {'accuracy':float(tp.sum()/matrix.sum()), 'macro_f1':float(f1.mean()),
            'log_loss':loss, 'per_class_recall':recall.tolist(), 'confusion':matrix.tolist()}


def write_matrix(path, value):
    array = np.asarray(value, dtype='<f8', order='C')
    array.tofile(path)
    return {'path':path.name, 'rows':array.shape[0], 'columns':array.shape[1], 'sha256':digest(path)}


def prepare(output):
    target = output/'prepared'; target.mkdir(exist_ok=False)
    path = output/'data/covtype.data.gz'
    with gzip.open(path, 'rt') as stream:
        data = np.loadtxt(stream, delimiter=',', dtype=np.float64)
    if data.shape != (581012,55) or not np.isfinite(data).all():
        raise RuntimeError('Covertype shape or finite-value contract failed')
    x, y = data[:,:54], data[:,54].astype(np.int64)-1
    if not np.array_equal(y+1,data[:,54]) or set(y.tolist())!=set(range(7)):
        raise RuntimeError('Invalid Covertype labels')
    # Original UCI split: 11,340 training, 3,780 validation, 565,892 test rows.
    scaler = StandardScaler().fit(x[:11340])
    train = scaler.transform(x[:11340]); validation = scaler.transform(x[11340:15120])
    selected = np.random.default_rng(20261008).choice(np.arange(15120,len(x)),8192,replace=False)
    save(target/'selected-row-indices.json',selected.tolist())
    save(target/'selected-labels.json',y[selected].tolist())
    shared = {}
    for rows in BATCHES:
        shared[rows] = write_matrix(target/f'input-{rows}.f64',scaler.transform(x[selected[:rows]]))
    options = dict(activation='relu', solver='adam', alpha=0.0001, batch_size=256,
        learning_rate_init=0.001, max_iter=100, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=12, random_state=20261008, tol=0.0001)
    for width in WIDTHS:
        print('Training width',width,flush=True)
        model=MLPClassifier(hidden_layer_sizes=(width,width),**options)
        with warnings.catch_warnings(record=True) as caught:
            started=time.perf_counter();model.fit(train,y[:11340]);seconds=time.perf_counter()-started
        layers=[]
        for w,b in zip(model.coefs_,model.intercepts_):
            layers.append({'W':w.astype(np.float32).astype(np.float64).ravel().tolist(),
                'b':b.astype(np.float32).astype(np.float64).tolist(),'inDim':w.shape[0],'outDim':w.shape[1]})
        model.coefs_,model.intercepts_=parameters(layers)
        params=parameters(layers)
        np.testing.assert_allclose(softmax(predict(validation,params),axis=1),model.predict_proba(validation),rtol=1e-12,atol=1e-12)
        validation_score=quality(y[11340:15120],predict(validation,params))
        confusion=np.zeros((7,7),dtype=np.int64);loss=0.0
        for start in range(15120,len(x),8192):
            end=min(start+8192,len(x));logits=predict(scaler.transform(x[start:end]),params)
            q=quality(y[start:end],logits);confusion+=np.array(q['confusion']);loss+=q['log_loss']*(end-start)
        test=quality_counts(confusion,loss/(len(x)-15120))
        timing={}
        for rows in BATCHES:
            raw=x[selected[:rows]];z=scaler.transform(raw);expected=predict(z,params)
            exp=write_matrix(target/f'expected-{width}-{rows}.f64',expected)
            save(target/f'fixture-{width}-{rows}.json',{'layers':layers,'inputBinary':shared[rows],'expectedBinary':exp})
            measurements=[];pipelines=[]
            for i in range(20):
                start=time.perf_counter_ns();actual=predict(z,params).copy();elapsed=time.perf_counter_ns()-start
                np.testing.assert_array_equal(actual,expected)
                start=time.perf_counter_ns();pipeline=predict(scaler.transform(raw),params).copy();pe=time.perf_counter_ns()-start
                np.testing.assert_allclose(pipeline,expected,rtol=1e-12,atol=1e-12)
                if i>=4:measurements.append(elapsed);pipelines.append(pe)
            timing[str(rows)]={'inference_ns':measurements,'scale_and_inference_ns':pipelines}
        metadata={'width':width,'options':options,'training_seconds':seconds,'iterations':int(model.n_iter_),
            'warnings':[str(w.message) for w in caught],'validation':validation_score,'test':test,
            'selected_8192_quality':quality(y[selected],predict(scaler.transform(x[selected]),params)),
            'numpy_timings':timing,'layers':layers}
        save(target/f'model-{width}.json',metadata)
        print('Trained',width,'iterations',model.n_iter_,'test',test['accuracy'],flush=True)
    save(target/'provenance.json',{'source':'https://archive.ics.uci.edu/dataset/31/covertype',
        'license':'CC BY 4.0','doi':'10.24432/C50K5N','source_sha256':digest(path),
        'source_zip_sha256':digest(output/'data/covertype.zip'),
        'split':{'train':[0,11340],'validation':[11340,15120],'test':[15120,581012]},
        'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
        'sklearn':sklearn.__version__,'numpy':np.__version__,'threadpools':threadpool_info(),
        'controller_sha256':digest(Path(__file__)),
        'files':{f.name:digest(f) for f in target.iterdir() if f.is_file()}})


def run(output,worker):
    target=output/'sweep';target.mkdir(exist_ok=False)
    source=source_identity(ROOT);build=json.loads(Path(str(worker)+'.build.json').read_text())
    if source['tree_sha256']!=build['source']['tree_sha256'] or digest(worker)!=build['binary_sha256']:
        raise RuntimeError('Worker/source fingerprint differs')
    verify_uninstrumented(worker)
    if metal_build_record(worker)!=build['metal']:raise RuntimeError('Metal fingerprint differs')
    prepared=output/'prepared';provenance=json.loads((prepared/'provenance.json').read_text())
    for name,expected in provenance['files'].items():
        if digest(prepared/name)!=expected:raise RuntimeError('Prepared artifact changed: '+name)
    cases=[(width,rows,mode) for width in WIDTHS for rows in BATCHES for mode in MODES]
    random.Random(20261008).shuffle(cases);records=[]
    y=np.array(json.loads((prepared/'selected-labels.json').read_text()))
    for i,(width,rows,mode) in enumerate(cases):
        print(i+1,len(cases),width,rows,mode,flush=True)
        case=(f'covertype{width}',rows,rows if 'matrix' in mode else 1,mode)
        env=dict(os.environ,SWIFTSCI_COREML_FIXTURE=str(prepared/f'fixture-{width}-{rows}.json'))
        record=run_case(worker,target,case,16,env)
        if record['outputColumns']!=7:raise RuntimeError('Expected seven class logits')
        logits=np.asarray(record['taskOutputs']).reshape(rows,7)
        if not np.isfinite(logits).all():raise RuntimeError('Nonfinite class logits')
        if len({s['output_sha256'] for s in [record['firstPrediction'],*record['warmPredictions']]})!=1:
            raise RuntimeError('Prediction changed during repeated inference')
        expected=np.fromfile(prepared/f'expected-{width}-{rows}.f64',dtype='<f8').reshape(rows,7)
        record['task_quality']=quality(y[:rows],logits)
        record['reference_label_disagreements']=int(np.count_nonzero(logits.argmax(1)!=expected.argmax(1)))
        records.append(record)
    validate_comparability(records)
    for record in records:
        if record['mode']=='coreml-cpu-matrix-adapter':
            direct=next(r for r in records if r['mode']=='coreml-cpu-matrix' and r['workload']==record['workload'] and r['rows']==record['rows'])
            if direct['firstPrediction']['output_sha256']!=record['firstPrediction']['output_sha256']:
                raise RuntimeError('Public/direct output mismatch')
    if source_identity(ROOT)!=source:raise RuntimeError('Sources changed during run')
    save(target/'results.json',records)
    save(target/'provenance.json',{'source':source,'build':build,'controller_sha256':digest(Path(__file__)),
        'prepared_sha256':digest(prepared/'provenance.json'),'cases':cases,'samples':16,
        'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        'official_mlperf':False,'application_acceptance_threshold':'not established'})
    print('Completed',len(records),'cases',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.action=='run' and args.worker is None:parser.error('run requires --worker')
    with threadpool_limits(limits=1):
        if args.action=='prepare':prepare(args.output.resolve())
        else:run(args.output.resolve(),args.worker.resolve(strict=True))

if __name__=='__main__':main()
