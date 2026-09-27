#!/usr/bin/env python3
"""Rebuild numerical inputs and separate references from immutable NIST originals."""
import argparse
from collections import OrderedDict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess

import mpmath as mp

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / 'Fixtures' / 'nist-models'
NUM = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
OLS = {'Norris': (36, 1), 'Pontius': (40, 2), 'Longley': (16, 6), **{f'Wampler{i}': (21, 5) for i in range(1, 6)}}
ANOVA = {'SiRstv': (5, 5), 'AtmWtAg': (2, 24), **{f'SmLs{i:02}': (9, [21, 201, 2001][(i-1)%3]) for i in range(1, 10)}}

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def text(x):
    return mp.nstr(x, 80)

def certified_check(actual, expected):
    value = mp.mpf(expected)
    if not value:
        limit = mp.mpf('1e-60')
    else:
        # Half a unit in the last printed place, plus a tiny arithmetic margin.
        exponent = Decimal(expected).as_tuple().exponent
        limit = mp.mpf('0.5000000001') * mp.power(10, exponent)
    error = abs(actual-value)
    if error > limit:
        raise ValueError(f'Independent value {text(actual)} differs from {expected}; error {text(error)} > {text(limit)}')
    return {'absoluteError': text(error), 'roundingBound': text(limit)}

def load(base_dir, name, columns, rows):
    path = base_dir / 'originals' / f'{name}.dat'
    raw = path.read_bytes()
    lines = raw.decode('ascii').splitlines()
    header = '\n'.join(lines[:60])
    bounds = re.search(r'Data\s+\(lines (\d+) to (\d+)\)', header)
    if not bounds:
        raise ValueError(f'Missing data range: {name}')
    first, last = map(int, bounds.groups())
    if first != 61 or last-first+1 != rows:
        raise ValueError(f'Unexpected declared data range: {name}: {first}:{last}')
    if len(lines) < last or any(line.strip() for line in lines[last:]):
        raise ValueError(f'Unexpected file length: {name}')
    values = []
    for number, line in enumerate(lines[first-1:last], first):
        tokens = line.split()
        if len(tokens) != columns or any(not re.fullmatch(NUM, x) for x in tokens):
            raise ValueError(f'Bad row {name}:{number}')
        values.append(tokens)
    return raw, header, values, first, last

def ols_solve(features, targets):
    a = mp.matrix([[mp.mpf(1)] + [mp.mpf(x) for x in row] for row in features])
    y = mp.matrix([mp.mpf(x) for x in targets])
    beta, _ = mp.qr_solve(a, y)
    predictions = a * beta
    rss = mp.fsum((predictions[i]-y[i])**2 for i in range(len(targets)))
    return list(beta) + [rss], list(predictions)

def anova_solve(groups):
    values = [[mp.mpf(x) for x in g] for g in groups]
    counts = [len(g) for g in values]
    n, k = sum(counts), len(values)
    means = [mp.fsum(g)/len(g) for g in values]
    grand = mp.fsum(mp.fsum(g) for g in values)/n
    between = mp.fsum(count*(mean-grand)**2 for count, mean in zip(counts, means))
    within = mp.fsum(mp.fsum((x-mean)**2 for x in g) for g, mean in zip(values, means))
    df1, df2 = k-1, n-k
    f = (between/df1)/(within/df2)
    p = mp.betainc(mp.mpf(df2)/2, mp.mpf(df1)/2, 0, mp.mpf(df2)/(df2+df1*f), regularized=True)
    return [f, mp.mpf(df1), mp.mpf(df2)], p

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE_DIR, help='Fixture directory containing sources.lock.json and originals.')
    parser.add_argument('--fetch', action='store_true', help='Fetch only missing originals, then verify SHA-256.')
    args = parser.parse_args()
    base_dir = args.base_dir.resolve()
    if mp.__version__ != '1.4.1':
        raise SystemExit(f'mpmath 1.4.1 required, found {mp.__version__}')
    mp.mp.dps = 80
    locks = json.loads((base_dir/'sources.lock.json').read_text())
    for source in locks['sources']:
        name = source['name']
        path = base_dir / source['path']
        if args.fetch and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['curl', '-L', '--fail', '--silent', '--show-error', source['url'], '-o', str(path)], check=True)
        if sha(path.read_bytes()) != source['sha256']:
            raise ValueError(f'Source digest mismatch: {name}')
        if name in OLS:
            n, p = OLS[name]
            raw, header, rows, first, last = load(base_dir, name, 7 if name == 'Longley' else 2, n)
            targets = [row[0] for row in rows]
            if name == 'Longley':
                features = [row[1:] for row in rows]
                names = [f'x{i}' for i in range(1, 7)]
            else:
                features = [[str(Decimal(row[1])**degree) for degree in range(1,p+1)] for row in rows]
                names = ['x' if i == 1 else f'x^{i}' for i in range(1,p+1)]
            parameters = re.findall(r'^\s*B(\d+)\s+('+NUM+r')\s+('+NUM+r')\s*$', header, re.M)
            if [int(x[0]) for x in parameters] != list(range(p+1)):
                raise ValueError(f'Unexpected coefficients: {name}')
            residual = re.search(r'^Residual\s+(\d+)\s+('+NUM+r')\s+('+NUM+r')\s*$', header, re.M)
            certified = [x[1] for x in parameters] + [residual[2]]
            order = ['intercept'] + [f'coefficient:{x}' for x in names] + ['residualSumSquares']
            actual, predictions = ols_solve(features, targets)
            binary_actual, binary_predictions = ols_solve([[float(x) for x in row] for row in features], [float(x) for x in targets])
            fixture = {'operation':'ols-cpu', 'features':[[float(x) for x in row] for row in features], 'targets':[float(x) for x in targets]}
            independent = {'method':'mpmath.qr_solve at 80 decimal digits', 'values':[text(x) for x in actual], 'predictions':[text(x) for x in predictions]}
            binary = {'method':'mpmath.qr_solve on exact binary64 input values', 'values':[text(x) for x in binary_actual], 'predictions':[text(x) for x in binary_predictions]}
            model = {'intercept':True, 'featureOrder':names, 'targetColumn':'y', 'observations':n, 'parameters':p+1}
        else:
            k, replicates = ANOVA[name]
            raw, header, rows, first, last = load(base_dir, name, 2, k*replicates)
            groups = OrderedDict()
            for label, value in rows:
                groups.setdefault(label, []).append(value)
            if len(groups) != k or any(len(g) != replicates for g in groups.values()):
                raise ValueError(f'Unbalanced groups: {name}')
            between = re.search(r'^Between\s+\w+\s+(\d+)\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s*$', header, re.M)
            within = re.search(r'^Within\s+\w+\s+(\d+)\s+('+NUM+r')\s+('+NUM+r')\s*$', header, re.M)
            if not between or not within:
                raise ValueError(f'Missing ANOVA reference: {name}')
            certified = [between[4], between[1], within[1]]
            order = ['fStatistic', 'dfBetween', 'dfWithin']
            actual, pvalue = anova_solve(list(groups.values()))
            binary_actual, binary_pvalue = anova_solve([[float(x) for x in g] for g in groups.values()])
            fixture = {'operation':'nist-anova', 'groups':[[float(x) for x in g] for g in groups.values()]}
            independent = {'method':'centered one-way ANOVA sums and regularized incomplete beta survival at 80 decimal digits', 'values':[text(x) for x in actual], 'pValue':text(pvalue), 'pValueIsNISTCertified':False}
            binary = {'method':'same ANOVA formula on exact binary64 input values', 'values':[text(x) for x in binary_actual], 'pValue':text(binary_pvalue)}
            model = {'design':'one-way balanced', 'treatmentOrder':list(groups), 'groups':k, 'replicatesPerGroup':replicates, 'observations':k*replicates}
        checks = [certified_check(a, c) for a, c in zip(actual, certified)]
        inputs = f'inputs/{name}.json'
        reference_path = f'references/{name}.json'
        write_json(base_dir/inputs, fixture)
        input_bytes = (base_dir/inputs).read_bytes()
        reference = {'inputIdentity': {'sha256': sha(input_bytes), 'bytes': len(input_bytes)}, 'name':name, 'source':source, 'dataLines':{'first':first, 'last':last, 'oneBased':True}, 'model':model, 'difficulty':re.search(r'(Lower|Average|Higher) Level of Difficulty', header)[1].lower(), 'outputOrder':order, 'nistCertifiedValues':certified, 'independentDecimalReference':independent, 'binary64InputDiagnostic':binary, 'certifiedRoundingChecks':checks}
        write_json(base_dir/reference_path, reference)
        print(f'{name}: {model["observations"]} rows verified; {len(certified)} outputs')

if __name__ == '__main__':
    main()
