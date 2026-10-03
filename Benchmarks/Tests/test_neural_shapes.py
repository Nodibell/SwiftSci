import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Benchmarks/Tools'),str(ROOT/'Benchmarks/Python')]
from contracts import ContractError, load_profile, read_json, validate_values
from datasets import load_manifest
from numerical_fixtures import source_and_input, expected_values, validate_manifest
from neural_shapes import validate_input, output_count
from neural_shaped_reference import reference
from neural_workloads import prepare, execute


class ShapedDecoderTests(unittest.TestCase):
    def inputs(self, profile='neural-shaped-cpu-conformance'):
        for case in load_profile(ROOT,profile)['cases']:
            m=load_manifest(ROOT,case['dataset'])
            source_and_input(ROOT,m)
            p=read_json(ROOT/m['fixture'])
            yield m,p,expected_values(ROOT,m,p)

    def test_numpy_matches_all_pinned_answers(self):
        for m,p,gold in self.inputs('neural-shaped-conformance'):
            with self.subTest(case=m['id']):
                self.assertEqual(len(gold),output_count(p))
                validate_values(execute(prepare(p)),gold,2e-5,2e-5)

    def test_scalar_oracle_reconstructs_each_architecture(self):
        for m,p,gold in self.inputs():
            if p['execution']=='full':
                with self.subTest(case=m['id']): self.assertEqual(reference(p),gold)

    def test_generalized_oracle_preserves_frozen_pack(self):
        from neural_reference import reference as frozen
        for name in ['neural-cpu-learned-full','neural-cpu-rope-cached']:
            p=read_json(ROOT/f'Benchmarks/Fixtures/neural/inputs/{name}.json')
            self.assertEqual(reference(p),frozen(p))

    def test_malformed_shapes_tokens_weights_and_schedules_fail(self):
        m,base,_=next(self.inputs());bad=[]
        for key,value in [('hidden_dim',True),('hidden_dim',33),('hidden_dim',15),('num_heads',0),('num_heads',3),('max_seq_len',129),('vocab_size',0)]:
            p=copy.deepcopy(base);p['config'][key]=value;bad.append(p)
        for key,value in [('tokens',[[True]]),('tokens',[[0]*129]),('chunk_sizes',[4,4]),('loading','public-loader'),('device','auto')]:
            p=copy.deepcopy(base);p[key]=value;bad.append(p)
        for chunks in [[],[0,1],[4,2,1,1],[4,True,1,1,1],[4,1,1]]:
            p=copy.deepcopy(base);p.update(execution='cached',chunk_sizes=chunks);bad.append(p)
        p=copy.deepcopy(base);p['weights']['lmHead.weight']['values'][0]=.1;bad.append(p)
        p=copy.deepcopy(base);p['weights'].pop('embedding.weight');bad.append(p)
        p=copy.deepcopy(base);p['weights']['lmHead.weight']['shape'][0]=True;bad.append(p)
        for p in bad:
            with self.subTest(config=p['config'],chunks=p['chunk_sizes']):
                with self.assertRaises(ContractError): validate_input(p,p['operation'],m['rows'])
        m['tolerances']['atol']=.001
        with self.assertRaises(ContractError): validate_manifest(m)

    def test_cache_reset_causality_and_corruption_controls(self):
        for m,p,gold in self.inputs():
            if p['execution']!='cached': continue
            prepared=prepare(p);first=execute(prepared)
            self.assertEqual(first,execute(prepared))
            for index in [0,3,len(gold)//2,len(gold)-1]:
                wrong=gold.copy();wrong[index]+=1
                with self.assertRaises(ContractError):validate_values(wrong,gold,2e-5,2e-5)
            changed=copy.deepcopy(p)
            for row in changed['tokens']:row[-1]=(row[-1]+1)%p['config']['vocab_size']
            modified=execute(prepare(changed));vocab=p['config']['vocab_size'];length=len(p['tokens'][0])
            for batch in range(len(p['tokens'])):
                start=3+batch*length*vocab;stop=start+(length-1)*vocab
                self.assertEqual(first[start:stop],modified[start:stop])
            self.assertNotEqual(first,modified)

    @unittest.skipUnless(importlib.util.find_spec('mlx'),'Install requirements-mlx.txt')
    def test_mlx_cpu_all_shapes_and_cache_reset(self):
        from mlx_workloads import MLXWorkload
        for m,p,gold in self.inputs():
            with self.subTest(case=m['id']):
                worker=MLXWorkload(p);actual=worker.execute()
                validate_values(actual,gold,2e-5,2e-5)
                validate_values(worker.execute(),actual,0,0)

    @unittest.skipUnless(os.environ.get('SWIFTSCI_MLX_TEST_GPU')=='1','Explicit Metal invocation required')
    def test_mlx_explicit_gpu_all_shapes(self):
        from mlx_workloads import MLXWorkload
        for m,p,gold in self.inputs('neural-shaped-conformance'):
            if p['device']=='gpu':
                with self.subTest(case=m['id']):validate_values(MLXWorkload(p).execute(),gold,2e-5,2e-5)
