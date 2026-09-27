import copy
import importlib.util
import math
import shutil
import tempfile
from pathlib import Path
import sys
import unittest
from test_contracts import ROOT
from contracts import ContractError, load_profile, read_json, validate_values
from datasets import load_manifest
from neural_fixtures import validate_input, output_count
from neural_reference import reference
from numerical_fixtures import source_and_input, expected_values, validate_manifest
sys.path.insert(0,str(ROOT/'Benchmarks/Python'))
from neural_workloads import prepare, execute


class NeuralFixtures(unittest.TestCase):
    def payload(self, name='neural-cpu-learned-full'):
        return read_json(ROOT/f'Benchmarks/Fixtures/neural/inputs/{name}.json')

    def test_profiles_and_pinned_reference_reconstruction(self):
        for profile in ('neural-conformance','neural-cpu-conformance','neural-loader-conformance'):
            for case in load_profile(ROOT,profile)['cases']:
                with self.subTest(case=case['id']):
                    m=load_manifest(ROOT,case['dataset']);source_and_input(ROOT,m)
                    p=read_json(ROOT/m['fixture']);gold=expected_values(ROOT,m,p)
                    self.assertEqual(gold,reference(p))
                    self.assertEqual(len(gold),output_count(p))
                    validate_values(execute(prepare(p)),gold,2e-5,2e-5)

    def test_zero_projection_analytic_answer(self):
        p=self.payload('neural-cpu-zero-projections')
        factor=1/math.sqrt(1/8+2**-10)
        gold=[1,4,7]+[factor if j==token else 0 for token in [0,1,2,3] for j in range(7)]
        validate_values(reference(p),gold,1e-14,1e-14)
        validate_values(execute(prepare(p)),gold,2e-5,2e-5)

    def test_future_tokens_cannot_change_prefix_logits(self):
        for position in ('learned','rope'):
            p=self.payload(f'neural-cpu-{position}-full');q=copy.deepcopy(p)
            for tokens in q['tokens']:tokens[-1]=(tokens[-1]+1)%7
            for run in (reference,lambda x:execute(prepare(x))):
                a,b=run(p),run(q)
                self.assertEqual(a[:3],b[:3])
                for batch in range(2):
                    start=3+batch*4*7
                    self.assertEqual(a[start:start+3*7],b[start:start+3*7])
                self.assertNotEqual(a,b)

    def test_cache_reset_and_full_agreement(self):
        for position in ('learned','rope'):
            p=self.payload(f'neural-cpu-{position}-cached');prepared=prepare(p)
            first,second=execute(prepared),execute(prepared)
            self.assertEqual(first,second)
            self.assertEqual(first[-4:],[3,2,3,4])
            full=copy.deepcopy(p);full['execution']='full'
            validate_values(first[:-4],execute(prepare(full)),2e-5,2e-5)
            self.assertEqual(reference(p)[:-4],reference(full))

    def test_boundary_rejects_inconsistent_or_unbounded_inputs(self):
        base=self.payload();bad=[]
        for key,value in [('device','auto'),('position','alibi'),('execution','random'),('loading','unchecked')]:
            p=copy.deepcopy(base);p[key]=value;bad.append(p)
        for value in (True,1.0,-1,7):
            p=copy.deepcopy(base);p['tokens'][0][0]=value;bad.append(p)
        p=copy.deepcopy(base);p['tokens'][0].pop();bad.append(p)
        p=copy.deepcopy(base);p['weights'].pop('lmHead.weight');bad.append(p)
        p=copy.deepcopy(base);p['weights']['extra']={};bad.append(p)
        for value in (True,.1,float('inf'),3):
            p=copy.deepcopy(base);p['weights']['lmHead.weight']['values'][0]=value;bad.append(p)
        for shape in ([8,7],[7.0,8],[True,8]):
            p=copy.deepcopy(base);p['weights']['lmHead.weight']['shape']=shape;bad.append(p)
        p=copy.deepcopy(base);p['weights']['lmHead.weight']['values'].pop();bad.append(p)
        p=copy.deepcopy(base);p['expected_logits']=[];bad.append(p)
        p=copy.deepcopy(base);p['execution']='cached';p['tokens']=[[0]];bad.append(p)
        for p in bad:
            with self.assertRaises(ContractError):validate_input(p,p['operation'],8)
        with self.assertRaises(ContractError):validate_input(base,base['operation'],7)
        m=load_manifest(ROOT,'neural-cpu-learned-full');m['tolerances']['atol']=.01
        with self.assertRaises(ContractError):validate_manifest(m)

    def test_each_output_region_and_weight_change_is_observable(self):
        p=self.payload('neural-cpu-rope-cached');gold=reference(p)
        for index in (0,3,len(gold)//2,len(gold)-5,len(gold)-1):
            wrong=gold.copy();wrong[index]+=.01
            with self.assertRaises(ContractError):validate_values(wrong,gold,2e-5,2e-5)
        for name in ('embedding.weight','layers.0.attention.query_proj.weight','layers.0.ffn.down.weight','lmHead.weight'):
            changed=copy.deepcopy(p)
            changed['weights'][name]['values']=[v+.125 for v in changed['weights'][name]['values']]
            with self.assertRaises(ContractError):validate_values(reference(changed),gold,2e-5,2e-5)

    def test_generator_reconstructs_inputs(self):
        path=ROOT/'Benchmarks/Fixtures/neural/generate.py'
        spec=importlib.util.spec_from_file_location('neural_generator',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name,payload in module.cases():self.assertEqual(payload,self.payload(name))

    def test_oracle_source_tampering_is_rejected(self):
        manifest=load_manifest(ROOT,'neural-cpu-learned-full')
        lock=read_json(ROOT/manifest['source_fixture'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            for relative in [manifest['source_fixture'],manifest['fixture']]:
                dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
            for entry in lock['sources']:
                relative=(Path(manifest['source_fixture']).parent/entry['path'])
                dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
            source_and_input(root,manifest)
            oracle=root/'Benchmarks/Tools/neural_reference.py'
            oracle.write_text(oracle.read_text()+'\n# changed oracle\n')
            with self.assertRaises(ContractError):source_and_input(root,manifest)
