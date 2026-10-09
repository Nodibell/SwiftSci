#!/usr/bin/env python3
"""Frozen real-data MLP comparisons; exploratory evidence, not MLPerf certification."""
import argparse
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
import scipy
from scipy.special import expit
import sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info
from run import run_case, validate_comparability, reference_accuracy_met

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Benchmarks/Tools'))
from runner import source_identity, verify_uninstrumented, metal_build_record


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward(x, layers):
    value = x
    for index, layer in enumerate(layers):
        weights = np.asarray(layer['W']).reshape(layer['inDim'], layer['outDim'])
        value = value @ weights + layer['b']
        if index < len(layers) - 1:
            value = np.maximum(value, 0)
    return value[:, 0]


def scores(kind, y, output, target_mean=0.0, target_scale=1.0):
    if kind == 'wdbc':
        probability = expit(output)
        return {'accuracy': float(accuracy_score(y, probability > 0.5)),
                'roc_auc': float(roc_auc_score(y, probability)),
                'log_loss': float(log_loss(y, probability, labels=[0, 1]))}
    prediction = output * target_scale + target_mean
    return {'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae': float(mean_absolute_error(y, prediction)),
            'r2': float(r2_score(y, prediction))}


def timed(fn, samples):
    values = []
    for i in range(samples + 4):
        start = time.perf_counter_ns()
        result = fn()
        elapsed = time.perf_counter_ns() - start
        if i >= 4:
            values.append(elapsed)
    assert np.isfinite(result).all()
    return {'median_ms': float(np.median(values) / 1e6),
            'p95_ms': float(np.percentile(values, 95) / 1e6), 'elapsed_ns': values}


def prepare(kind, destination, samples):
    ident = 'wdbc-supervised-logistic-cpu-epochs32' if kind == 'wdbc' else 'wine-red-supervised-ols-cpu'
    manifest = json.loads((ROOT / f'Benchmarks/Specs/datasets/{ident}.json').read_text())
    path = ROOT / manifest['fixture']
    if sha(path) != manifest['sha256']:
        raise RuntimeError('Dataset fixture checksum mismatch')
    d = json.loads(path.read_text())
    x, y = np.asarray(d['features'], dtype=np.float64), np.asarray(d['targets'], dtype=np.float64)
    splits = {key: np.asarray(value, dtype=int) for key, value in d['splits'].items()}
    joined = np.concatenate(list(splits.values()))
    if len(set(joined.tolist())) != len(x) or set(joined.tolist()) != set(range(len(x))):
        raise RuntimeError('Dataset partitions must be disjoint and complete')
    train = splits['train']
    scaler = StandardScaler().fit(x[train])
    target_mean = 0.0 if kind == 'wdbc' else float(y[train].mean())
    target_scale = 1.0 if kind == 'wdbc' else float(y[train].std())
    common = dict(hidden_layer_sizes=(64, 64), activation='relu', solver='lbfgs',
                  alpha=0.1, max_iter=1000, max_fun=30000, random_state=20261008, tol=1e-7)
    model = MLPClassifier(**common) if kind == 'wdbc' else MLPRegressor(**common)
    with warnings.catch_warnings(record=True) as caught:
        start = time.perf_counter()
        model.fit(scaler.transform(x[train]), (y[train] - target_mean) / target_scale)
        fit_seconds = time.perf_counter() - start
    # Quantize weights once to the format used by Core ML's exporter. All paths use these values.
    layers = []
    for w, b in zip(model.coefs_, model.intercepts_):
        layers.append({'W': w.astype(np.float32).astype(np.float64).ravel().tolist(),
                       'b': b.astype(np.float32).astype(np.float64).tolist(),
                       'inDim': w.shape[0], 'outDim': w.shape[1]})
    unrounded_scores = {}
    for split in ['validation', 'test']:
        z = scaler.transform(x[splits[split]])
        if kind == 'wdbc':
            prob = model.predict_proba(z)[:, 1]
            unrounded_scores[split] = {'accuracy': float(accuracy_score(y[splits[split]], prob > 0.5)),
                                      'roc_auc': float(roc_auc_score(y[splits[split]], prob)),
                                      'log_loss': float(log_loss(y[splits[split]], prob))}
        else:
            unrounded_scores[split] = scores(kind, y[splits[split]], model.predict(z), target_mean, target_scale)
    model.coefs_ = [np.asarray(v['W']).reshape(v['inDim'], v['outDim']) for v in layers]
    model.intercepts_ = [np.asarray(v['b']) for v in layers]
    test_x = x[splits['test']]
    z = scaler.transform(test_x)
    expected = forward(z, layers)
    reference = model.predict_proba(z)[:, 1] if kind == 'wdbc' else model.predict(z)
    np.testing.assert_allclose(expit(expected) if kind == 'wdbc' else expected, reference, atol=1e-12, rtol=1e-12)
    quality = {}
    for split in ['validation', 'test']:
        output = forward(scaler.transform(x[splits[split]]), layers)
        quality[split] = scores(kind, y[splits[split]], output, target_mean, target_scale)
    baseline = np.log(y[train].mean() / (1-y[train].mean())) if kind == 'wdbc' else 0.0
    quality['constant_test'] = scores(kind, y[splits['test']], np.full(len(z), baseline), target_mean, target_scale)
    metadata = {'dataset': kind, 'source_manifest': manifest, 'split_ids': d['splits'],
                'test_row_ids': [d['row_ids'][i] for i in splits['test']],
                'training_options': common, 'fit_seconds': fit_seconds, 'iterations': int(model.n_iter_),
                'training_warnings': [str(w.message) for w in caught],
                'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
                'target_mean': target_mean, 'target_scale': target_scale,
                'unrounded_quality': unrounded_scores, 'quality': quality,
                'heldout_targets': y[splits['test']].tolist(), 'layers': layers,
                'python_timing': {}, 'fixture_hashes': {}}
    for rows in [1, 32, len(z)]:
        zi = np.ascontiguousarray(z[:rows])
        expected = forward(zi, layers)
        f = destination / f'{kind}-{rows}.json'
        save(f, {'columns': zi.T.tolist(), 'expected': expected.tolist(), 'layers': layers})
        metadata['fixture_hashes'][str(rows)] = sha(f)
        infer = lambda: model.predict_proba(zi) if kind == 'wdbc' else model.predict(zi)
        pipeline = lambda: model.predict_proba(scaler.transform(test_x[:rows])) if kind == 'wdbc' else model.predict(scaler.transform(test_x[:rows]))
        metadata['python_timing'][str(rows)] = {'inference': timed(infer, samples), 'scale_and_inference': timed(pipeline, samples)}
    save(destination / f'{kind}-training.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.samples <= 512:
        parser.error('samples must be in 1..512')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / 'fixtures'; fixtures.mkdir()
    worker = args.worker.resolve(strict=True)
    build = json.loads(Path(str(worker) + '.build.json').read_text())
    source = source_identity(ROOT)
    if build['binary_sha256'] != sha(worker) or build['source']['tree_sha256'] != source['tree_sha256']:
        raise RuntimeError('Worker or source does not match recorded build')
    verify_uninstrumented(worker)
    if build['metal'] != metal_build_record(worker):
        raise RuntimeError('Metal settings mismatch')
    with threadpool_limits(limits=1):
        datasets = {kind: prepare(kind, fixtures, args.samples) for kind in ['wdbc', 'wine']}
        pools = threadpool_info()
    modes = ['mlx-cpu', 'mlx-gpu', 'coreml-cpu']
    for policy in ['cpu', 'gpu', 'neural', 'all']:
        modes += [f'coreml-{policy}-matrix', f'coreml-{policy}-matrix-adapter']
    cases = []
    for kind, metadata in datasets.items():
        for rows in [1, 32, len(metadata['heldout_targets'])]:
            for mode in modes:
                batch = rows if mode.endswith(('-matrix', '-matrix-adapter')) else min(rows, 32) if mode == 'coreml-cpu' else 1
                cases.append((kind, rows, batch, mode))
    random.Random(20261008).shuffle(cases)
    records = []
    for i, case in enumerate(cases):
        print(f'{i+1}/{len(cases)}: {case}', flush=True)
        kind, rows, batch, mode = case
        fixture = fixtures / f'{kind}-{rows}.json'
        if sha(fixture) != datasets[kind]['fixture_hashes'][str(rows)]:
            raise RuntimeError('Trained fixture changed')
        env = dict(os.environ, SWIFTSCI_COREML_FIXTURE=str(fixture))
        record = run_case(worker, output, case, args.samples, env)
        predictions = np.asarray(record['taskOutputs'], dtype=np.float64)
        if predictions.shape != (rows,) or not np.isfinite(predictions).all():
            raise RuntimeError('Missing or invalid task predictions')
        if len({s['output_sha256'] for s in [record['firstPrediction'], *record['warmPredictions']]}) != 1:
            raise RuntimeError('Task predictions changed during repeated inference')
        # A single recorded prediction is valid for task scoring only after that stability check.
        meta = datasets[kind]
        if rows == len(meta['heldout_targets']):
            record['task_quality'] = scores(kind, np.asarray(meta['heldout_targets']), predictions, meta['target_mean'], meta['target_scale'])
            expected = np.asarray(json.loads(fixture.read_text())['expected'])
            if kind == 'wdbc':
                record['label_disagreements'] = int(np.count_nonzero((predictions > 0) != (expected > 0)))
            record['maximum_reference_difference'] = float(np.max(np.abs(predictions - expected)))
        records.append(record)
    validate_comparability(records)
    for record in records:
        if record['mode'].endswith('-matrix-adapter'):
            direct = next(r for r in records if r['workload']==record['workload'] and r['rows']==record['rows'] and r['mode']==record['mode'].removesuffix('-adapter'))
            if record['firstPrediction']['output_sha256'] != direct['firstPrediction']['output_sha256']:
                raise RuntimeError('Public and direct Core ML predictions differ')
    if source_identity(ROOT) != source:
        raise RuntimeError('Sources changed during benchmark')
    save(output/'results.json', records)
    save(output/'provenance.json', {'build':build,'source':source,'controller_sha256':sha(Path(__file__)),
         'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
         'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__, 'threadpools':pools,
         'samples':args.samples,'case_order':cases,'official_mlperf':False,'application_quality_threshold':'not established'})
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(output/'requirements.txt').open('w'),check=True)
    print('Completed trained-model comparisons:', output, flush=True)

if __name__ == '__main__':
    main()
