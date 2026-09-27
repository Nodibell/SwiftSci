"""Public workflow input contracts. Worker payloads contain no expected answers."""
import csv
import io
import math
from contracts import fields, require

OPERATIONS = {'scientific-workflow', 'persisted-regression'}


def csv_rows(text, columns):
    require(isinstance(text, str) and len(text) < 100000, 'Invalid bounded CSV')
    reader = csv.DictReader(io.StringIO(text))
    require(reader.fieldnames == columns, 'CSV schema mismatch')
    rows = list(reader)
    require(4 <= len(rows) <= 128, 'CSV row limit')
    require(all(set(row) == set(columns) and all(v is not None for v in row.values()) for row in rows), 'Ragged CSV')
    for row in rows:
        for key, value in row.items():
            if key == 'quality' and value == '': continue
            try: number = float(value)
            except ValueError: require(False, 'Non-numeric CSV field')
            require(math.isfinite(number) and abs(number) <= 10000, 'Invalid CSV value')
            if key in ('row_id', 'calibration_id', 'site'):
                require(value.isdigit(), 'IDs must be nonnegative integer literals')
    return rows


def validate_input(p, operation, rows):
    require(operation == p.get('operation') and operation in OPERATIONS, 'Workflow operation mismatch')
    if operation == 'scientific-workflow':
        fields(p, ['operation', 'observations_csv', 'calibration_csv', 'feature_order'])
        left = csv_rows(p['observations_csv'], ['row_id','site','signal','target','quality'])
        right = csv_rows(p['calibration_csv'], ['calibration_id','site','offset'])
        require(len(left) == rows, 'Observation count mismatch')
        require(len({r['row_id'] for r in left}) == rows, 'Duplicate observation IDs')
        require(len({r['calibration_id'] for r in right}) == len(right), 'Duplicate calibration IDs')
        require(p['feature_order'] in (['signal','offset'], ['offset','signal']), 'Invalid scientific features')
        count = sum(1 for l in left for r in right if l['quality'] and float(l['quality']) >= 1 and l['site']==r['site'])
        require(count >= 4, 'Insufficient joined observations')
    else:
        fields(p, ['operation','route','feature_names','train_ids','train_features','train_targets',
                   'heldout_ids','heldout_features','heldout_targets'])
        require(p['route'] in ('fit','native','coreml'), 'Unknown persistence route')
        require(p['feature_names'] in (['signal','offset'], ['offset','signal']), 'Invalid feature schema')
        require(type(rows) is int and 4 <= rows <= 128 and len(p['train_features']) == rows, 'Invalid training rows')
        def matrix(x):
            require(isinstance(x,list) and 1 <= len(x) <= 128, 'Invalid feature matrix')
            for row in x: vector(row, 2)
        def vector(x, n):
            require(isinstance(x,list) and len(x)==n and all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=10000 for v in x), 'Invalid numeric vector')
        matrix(p['train_features']); matrix(p['heldout_features'])
        vector(p['train_targets'], rows); vector(p['heldout_targets'],len(p['heldout_features']))
        ids = []
        for key, count in [('train_ids',rows),('heldout_ids',len(p['heldout_features']))]:
            require(isinstance(p[key],list) and len(p[key])==count and all(type(v) is int and 0<=v<=2**31-1 for v in p[key]), 'Invalid partition IDs')
            ids += p[key]
        require(len(ids)==len(set(ids)), 'Partitions overlap or IDs repeat')
        require(not {tuple(x) for x in p['train_features']} & {tuple(x) for x in p['heldout_features']}, 'Feature duplicates cross partitions')
    return p
