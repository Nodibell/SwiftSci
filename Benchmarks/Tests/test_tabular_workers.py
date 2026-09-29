import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Benchmarks/Python'))
sys.path.insert(0,str(ROOT/'Benchmarks/Tools'))
from tabular_workloads import TabularWorkload
from contracts import digest


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("polars", "duckdb")),
                     "Optional local trial: install Benchmarks/Python/requirements-trial.txt")
class TabularWorkersTests(unittest.TestCase):
    def test_native_operations_match_complete_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.csv'
            path.write_text('id,group,x,y\n0,0,2,4\n1,1,-1,2\n2,0,2,8\n3,1,5,6\n')
            data=np.array([[0,0,2,4],[1,1,-1,2],[2,0,2,8],[3,1,5,6]],dtype=float)
            matrix=data[:,2:4]
            answers={
                'csv-read':data.ravel(), 'filter':data[[0,2,3]].ravel(),
                'sort':data[[1,0,2,3]].ravel(), 'group-sum':[0,4,1,4],
                'group-sum-mean':[0,4,6,1,4,4],
                'flat-matrix':matrix.ravel(), 'target':matrix[:,0],
                'standard-scale':((matrix-matrix.mean(axis=0))/matrix.std(axis=0)).ravel(),
                'minmax-scale':((matrix-matrix.min(axis=0))/(matrix.max(axis=0)-matrix.min(axis=0))).ravel(),
                'mean':[2], 'variance':[6], 'stddev':[np.sqrt(6)],
                'row-sum':[8], 'pearson':[np.corrcoef(matrix.T)[0,1]],
                'spearman':[np.corrcoef([2.5,1,2.5,4],[2,1,4,3])[0,1]],
                'inner-join':np.column_stack((data,np.arange(4)/4)).ravel(),
                'parquet-write':data.ravel(),
            }
            for engine in ('polars','duckdb'):
                for op,expected in answers.items():
                    with self.subTest(engine=engine,operation=op):
                        w=TabularWorkload(engine,dict(operation=op,dataset_kind='table-v1',input_path=str(path),rows=4))
                        try:np.testing.assert_allclose(w.canonical(w.execute(Path(tmp)/(engine+op),0)),expected,rtol=1e-12,atol=1e-12)
                        finally:w.close()

    def test_constant_scaling_and_mixed_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'mixed.csv';p.write_text('id,group,x,y,flag\n0,category2,3,4,true\n1,category1,3,4,false\n')
            for engine in ('polars','duckdb'):
                for op in ('standard-scale','minmax-scale','csv-read','sort','group-sum'):
                    with self.subTest(engine=engine,operation=op):
                        w=TabularWorkload(engine,dict(operation=op,dataset_kind='mixed-table-v1',input_path=str(p),rows=2))
                        try:
                            actual=w.canonical(w.execute(Path(tmp)/'out',0))
                            expected=[0,0,0,0] if op.endswith('scale') else [1,3,2,3] if op=='group-sum' else [0,2,3,4,1,1,1,3,4,0]
                            np.testing.assert_array_equal(actual,expected)
                        finally:w.close()

    def test_worker_rejects_corrupt_expected_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'id,group,x,y\n0,0,1,2\n1,0,3,4\n';input_path=root/'input.csv';input_path.write_bytes(data)
            for engine in ('polars','duckdb'):
                for corrupt in (False,True):
                    with self.subTest(engine=engine,corrupt=corrupt):
                        expected=np.array([1,99 if corrupt else 3],dtype='<f8').tobytes();ep=root/'expected.f64';ep.write_bytes(expected)
                        request=dict(schema_version=1,case_key='a'*64,operation='target',dataset_kind='table-v1',input_path=str(input_path),input_sha256=digest(data),input_bytes=len(data),expected_path=str(ep),expected_sha256=digest(expected),rows=2,warmups=0,samples=1,atol=0,rtol=0)
                        rp=root/'request.json';rp.write_text(json.dumps(request));dest=root/'response.json'
                        result=subprocess.run([sys.executable,str(ROOT/'Benchmarks/Python/tabular_worker.py'),engine,str(rp),str(dest)],env=dict(os.environ,POLARS_MAX_THREADS='1'),capture_output=True,text=True)
                        response=json.loads(dest.read_text())
                        self.assertEqual(result.returncode,1 if corrupt else 0,result.stderr)
                        self.assertEqual(response['status'],'failed' if corrupt else 'passed')
                        if corrupt:self.assertEqual(response['samples'],[])
                        else:self.assertTrue(response['samples'][0]['validated'])
