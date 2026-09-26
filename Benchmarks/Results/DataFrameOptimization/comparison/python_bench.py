import sys, json, time, resource
import numpy as np
import pandas as pd


if sys.argv[1] == 'startup':
    print('{}')
    raise SystemExit
op, n, repeats = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = np.arange(n, dtype=np.int64)
x = ((ids * 7919) % 100003).astype(np.float64) / 100.0
y = x * 0.7 + ids % 17
values = x.copy()
values[ids % 101 == 0] = np.nan
df = pd.DataFrame({'id': ids, 'group': ids % 100, 'value': values})
if op == 'forecast':
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
if op == 'group_two': df['group2'] = ids % 97
if op in ['filter_int','filter_int32']:
    df['value'] = pd.array((ids * 7919) % 100003, dtype='Int64' if op=='filter_int' else 'Int32')
    df.loc[ids % 101 == 0, 'value'] = pd.NA
if op == 'filter_float32': df['value'] = df['value'].astype(np.float32)
times = []
result = None
for iteration in range(repeats + 2):
    result = None
    start = time.perf_counter()
    if op == 'csv': result = pd.read_csv(sys.argv[4])
    elif op == 'filter': result = df.loc[df['value'] > 500.0]
    elif op in ['filter_int','filter_int32']: result = df.loc[df['value'] > 50000]
    elif op == 'filter_float32': result = df.loc[df['value'] > 500.0]
    elif op == 'group_two': result = df.groupby(['group','group2'], sort=False, as_index=False).sum(numeric_only=True)
    elif op == 'pipeline': result = df.loc[df['value'] > 500.0].sort_values('value', ascending=False, na_position='last', kind='stable').groupby('group',sort=False,as_index=False).sum(numeric_only=True)
    elif op == 'sort': result = df.sort_values('value', ascending=False, na_position='last', kind='stable')
    elif op == 'group': result = df.groupby('group', sort=False, as_index=False).sum(numeric_only=True)
    elif op == 'stats': result = [np.mean(x), np.var(x, ddof=1)]
    elif op == 'correlation': result = [np.corrcoef(x, y)[0, 1]]
    elif op == 'forecast':
        result = SimpleExpSmoothing(x, initialization_method='known', initial_level=float(x[0])).fit(
            smoothing_level=0.3, optimized=False).forecast(24)
    else: raise ValueError(op)
    elapsed = time.perf_counter() - start
    if iteration >= 2: times.append(elapsed)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if isinstance(result, pd.DataFrame):
    data = {name: [None if pd.isna(v) else float(v) for v in result[name]] for name in result.columns}
else: data = {'values': [float(v) for v in result]}
print(json.dumps({'times': times, 'peak_rss_bytes': rss, 'result': data}, allow_nan=False))
