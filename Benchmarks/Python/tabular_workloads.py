"""Native Polars and DuckDB operations for existing tabular contracts."""
from pathlib import Path
import numpy as np

OPERATIONS = frozenset({
    'csv-read', 'parquet-read', 'parquet-write', 'filter', 'sort', 'group-sum',
    'group-sum-mean', 'inner-join', 'flat-matrix', 'target', 'standard-scale',
    'minmax-scale', 'mean', 'variance', 'stddev', 'pearson', 'spearman', 'row-sum',
    'h2o-q1', 'h2o-q2', 'h2o-q3', 'h2o-q4', 'h2o-q5', 'wine-pipeline',
})
H2O = {
    'h2o-q1': (['id1'], [('v1', 'sum')]),
    'h2o-q2': (['id1', 'id2'], [('v1', 'sum')]),
    'h2o-q3': (['id3'], [('v1', 'sum'), ('v3', 'mean')]),
    'h2o-q4': (['id4'], [('v1', 'mean'), ('v2', 'mean'), ('v3', 'mean')]),
    'h2o-q5': (['id6'], [('v1', 'sum'), ('v2', 'sum'), ('v3', 'sum')]),
}
WINE = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol']


class TabularWorkload:
    def __init__(self, engine, request):
        self.engine, self.request = engine, request
        self.op = request['operation']
        if self.op not in OPERATIONS:
            raise ValueError('Unsupported native tabular operation: ' + self.op)
        self.mixed = request['dataset_kind'] in ('mixed-table-v1', 'parquet-table-v1')
        self.names = ['id', 'group', 'x', 'y'] + (['flag'] if self.mixed else [])
        types = ['int', 'str' if self.mixed else 'int', 'float', 'float'] + (['bool'] if self.mixed else [])
        if self.op in H2O:
            self.names = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'v1', 'v2', 'v3']
            types = ['str']*3 + ['int']*5 + ['float']
        elif self.op == 'wine-pipeline':
            self.names, types = ['id', *WINE, 'quality'], ['int'] + ['float']*11 + ['int']
        if engine == 'polars':
            import polars as pl
            self.pl = pl
            self.schema = dict(zip(self.names, [{'int':pl.Int64, 'float':pl.Float64, 'str':pl.String, 'bool':pl.Boolean}[t] for t in types]))
        elif engine == 'duckdb':
            import duckdb
            self.connection = duckdb.connect(':memory:', config={'threads':1})
            assert self.connection.execute("SELECT current_setting('threads')").fetchone()[0] == 1
            self.schema = dict(zip(self.names, [{'int':'BIGINT','float':'DOUBLE','str':'VARCHAR','bool':'BOOLEAN'}[t] for t in types]))
        else:
            raise ValueError('Unknown engine: ' + engine)
        self.frame = None
        if request['dataset_kind'] == 'nist-univariate-v1':
            x = np.loadtxt(request['input_path'], skiprows=request['input_skip_rows'])
            if engine == 'polars': self.frame = self.pl.DataFrame({'x':x})
            else:
                import pyarrow as pa
                self.connection.register('input_values', pa.table({'x':x}))
                self.connection.execute('CREATE TABLE source AS SELECT * FROM input_values')
                self.connection.unregister('input_values')
        elif self.op not in ('csv-read','parquet-read','wine-pipeline'):
            if engine == 'polars': self.frame = self.read_csv()
            else: self.read_csv().create('source')
        if self.op == 'inner-join':
            if engine == 'polars':
                self.right = self.pl.DataFrame({'id':np.arange(request['rows'],dtype=np.int64), 'weight':np.arange(request['rows'])/4.0})
            else:
                self.connection.execute('CREATE TABLE rhs AS SELECT range AS id, range / 4.0 AS weight FROM range(?)', [request['rows']])

    def read_csv(self):
        if self.engine == 'polars':
            return self.pl.read_csv(self.request['input_path'], schema=self.schema)
        return self.connection.read_csv(self.request['input_path'], header=True, columns=self.schema)

    def execute(self, destination, index):
        return self.polars(destination,index) if self.engine == 'polars' else self.duckdb(destination,index)

    def polars(self, destination, index):
        pl, f, op = self.pl, self.frame, self.op
        if op == 'csv-read': return self.read_csv()
        if op == 'parquet-read': return pl.read_parquet(self.request['input_path'])
        if op == 'parquet-write':
            path = Path(str(destination)+f'.sample{index}.parquet')
            f.write_parquet(path, compression='snappy')
            return path
        if op == 'filter': return f.filter(pl.col('x') > 0)
        if op == 'sort': return f.sort('x', maintain_order=True)
        if op == 'group-sum': return f.group_by('group').agg(pl.col('x').sum())
        if op == 'group-sum-mean': return f.group_by('group').agg(pl.col('x').sum().alias('x_sum'), pl.col('y').mean().alias('y_mean'))
        if op == 'inner-join': return f.join(self.right, on='id', how='inner')
        if op in H2O:
            keys, aggs = H2O[op]
            return f.group_by(keys).agg([getattr(pl.col(name),method)().alias(name) for name,method in aggs])
        if op == 'target': return f['x'].to_numpy(writable=True)
        if op == 'flat-matrix': return f.select('x','y').to_numpy(order='c', writable=True).ravel(order='C')
        if op in ('mean','variance','stddev'):
            expr = pl.col('x')
            return f.select(expr.mean() if op == 'mean' else expr.var(ddof=1) if op == 'variance' else expr.std(ddof=1)).item()
        if op in ('pearson','spearman'): return f.select(pl.corr('x','y',method=op)).item()
        if op == 'row-sum':
            total = 0.0
            for row in f.iter_rows(named=True): total += row['x']
            return total
        columns = ['x','y']
        if op == 'wine-pipeline':
            f = self.read_csv().filter(pl.col('quality') >= 6).sort('alcohol',maintain_order=True)
            columns = WINE
        expressions = []
        for name in columns:
            c = pl.col(name)
            if op in ('standard-scale','wine-pipeline'):
                scale = c.std(ddof=0)
                expressions.append(((c-c.mean())/pl.when(scale < 1e-12).then(1).otherwise(scale)).alias(name))
            elif op == 'minmax-scale':
                span = c.max()-c.min()
                expressions.append(((c-c.min())*pl.when(span < 1e-12).then(0).otherwise(1/span)).alias(name))
            else: raise ValueError(op)
        matrix = f.select(expressions).to_numpy(order='c',writable=True).ravel(order='C')
        if op == 'wine-pipeline': return np.concatenate((f['id'].to_numpy(), f['quality'].to_numpy(), matrix))
        return matrix

    def duckdb(self, destination, index):
        c, op = self.connection, self.op
        def table(sql): return c.execute(sql).to_arrow_table()
        if op == 'csv-read': return self.read_csv().to_arrow_table()
        if op == 'parquet-read': return c.read_parquet(self.request['input_path']).to_arrow_table()
        if op == 'parquet-write':
            path = Path(str(destination)+f'.sample{index}.parquet')
            c.table('source').write_parquet(str(path), compression='snappy')
            return path
        if op == 'filter': return table('SELECT * FROM source WHERE x > 0')
        if op == 'sort': return table('SELECT * FROM source ORDER BY x, id')
        if op == 'group-sum': return table('SELECT "group", sum(x) AS x FROM source GROUP BY "group"')
        if op == 'group-sum-mean': return table('SELECT "group", sum(x) AS x_sum, avg(y) AS y_mean FROM source GROUP BY "group"')
        if op == 'inner-join': return table('SELECT source.*, rhs.weight FROM source JOIN rhs USING (id)')
        if op in H2O:
            keys, aggs = H2O[op]
            expressions = ', '.join(('avg' if method=='mean' else method)+f'({name}) AS {name}' for name,method in aggs)
            return table(f"SELECT {', '.join(keys)}, {expressions} FROM source GROUP BY {', '.join(keys)}")
        if op in ('mean','variance','stddev'):
            fun = {'mean':'avg','variance':'var_samp','stddev':'stddev_samp'}[op]
            return c.execute(f'SELECT {fun}(x) FROM source').fetchone()[0]
        if op == 'pearson': return c.execute('SELECT corr(x,y) FROM source').fetchone()[0]
        if op == 'spearman':
            return c.execute('SELECT corr(rx,ry) FROM (SELECT rank() OVER (ORDER BY x)+(count(*) OVER (PARTITION BY x)-1)/2.0 AS rx, rank() OVER (ORDER BY y)+(count(*) OVER (PARTITION BY y)-1)/2.0 AS ry FROM source)').fetchone()[0]
        if op == 'row-sum':
            total = 0.0
            for row in c.execute('SELECT * FROM source').fetchall(): total += row[2]
            return total
        if op == 'target': return c.execute('SELECT x FROM source ORDER BY id').fetchnumpy()['x']
        if op == 'flat-matrix':
            result=c.execute('SELECT x,y FROM source ORDER BY id').fetchnumpy()
            return np.column_stack((result['x'],result['y'])).ravel(order='C')
        columns, source, ordering = ['x','y'], 'source', 'id'
        if op == 'wine-pipeline':
            source, columns, ordering = 'wine_input', WINE, 'alcohol, id'
            # Reading and preparation are inside this end-to-end timed workload.
            self.read_csv().create_view('wine_input', replace=True)
        expressions=[]
        for name in columns:
            q='"'+name+'"'
            if op in ('standard-scale','wine-pipeline'):
                scale=f'stddev_pop({q}) OVER ()'
                expressions.append(f'({q}-avg({q}) OVER ())/CASE WHEN {scale}<1e-12 THEN 1 ELSE {scale} END AS {q}')
            elif op == 'minmax-scale':
                span=f'(max({q}) OVER ()-min({q}) OVER ())'
                expressions.append(f'({q}-min({q}) OVER ())*CASE WHEN {span}<1e-12 THEN 0 ELSE 1/{span} END AS {q}')
            else: raise ValueError(op)
        prefix='id, quality, ' if op == 'wine-pipeline' else ''
        where=' WHERE quality >= 6' if op == 'wine-pipeline' else ''
        result=c.execute(f'SELECT {prefix}{", ".join(expressions)} FROM {source}{where} ORDER BY {ordering}').fetchnumpy()
        matrix=np.column_stack([result[name] for name in columns]).ravel(order='C')
        return np.concatenate((result['id'],result['quality'],matrix)) if op == 'wine-pipeline' else matrix

    def canonical(self, output):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if self.op == 'parquet-write': output=pq.read_table(output)
        if self.engine == 'polars' and isinstance(output, self.pl.DataFrame): output=output.to_arrow()
        if not isinstance(output, pa.Table): return np.asarray(output,dtype='<f8').ravel(order='C')
        names=output.column_names
        if self.op in H2O:
            keys,aggs=H2O[self.op];names=keys+[name for name,_ in aggs]
        elif self.op=='group-sum': names=['group','x']
        elif self.op=='group-sum-mean': names=['group','x_sum','y_mean']
        elif self.op=='inner-join': names=[*self.names,'weight']
        arrays=[]
        for name in names:
            col=output[name]
            if col.null_count: raise ValueError('Unexpected null in finite tabular fixture')
            if name=='flag' and col.type != pa.bool_(): raise ValueError('Lost Boolean type')
            if self.mixed and name=='group' or self.op in H2O and name in ('id1','id2','id3'):
                prefix='category' if name=='group' else 'key'
                values=col.to_pylist()
                if not all(isinstance(v,str) and v.startswith(prefix) and v[len(prefix):].isdigit() for v in values): raise ValueError('Malformed category')
                arrays.append(np.array([int(v[len(prefix):]) for v in values],dtype=np.float64))
            else: arrays.append(np.asarray(col.to_numpy(zero_copy_only=False),dtype=np.float64))
        matrix=np.column_stack(arrays)
        if self.op in H2O:
            nkeys=len(H2O[self.op][0]);matrix=matrix[np.lexsort(tuple(matrix[:,i] for i in reversed(range(nkeys))))]
        elif self.op in ('group-sum','group-sum-mean'): matrix=matrix[np.argsort(matrix[:,0],kind='stable')]
        elif self.op=='inner-join': matrix=matrix[np.argsort(matrix[:,0],kind='stable')]
        return np.asarray(matrix,dtype='<f8').ravel(order='C')

    def close(self):
        if self.engine=='duckdb': self.connection.close()
