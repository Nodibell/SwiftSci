import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from test_contracts import ROOT
from contracts import ContractError, load_profile, load_workload, validate_values
from datasets import load_manifest, prepare, reference, resolve_workload
from supervised_fixtures import validate_input, output_count, SPLITS
from build_supervised_fixtures import partition, scaled_reference, reference as high_precision_reference
import mpmath as mp
sys.path.insert(0,str(ROOT/'Benchmarks/Python'))
from supervised_workloads import prepare as prepare_runtime, execute


class SupervisedFixtures(unittest.TestCase):
    def cases(self):
        for c in load_profile(ROOT,'supervised-conformance')['cases']:
            m=load_manifest(ROOT,c['dataset']);w=resolve_workload(m,load_workload(ROOT,c['workload']))
            p=json.loads(prepare(ROOT,m).read_text())
            yield m,w,p,reference(ROOT,m,w)

    def test_all_outputs_match_independent_reference(self):
        cases=list(self.cases());self.assertEqual(len(cases),3)
        for m,w,p,expected in cases:
            self.assertEqual(output_count(p),len(expected))
            for _ in range(2):validate_values(execute(prepare_runtime(p)).tolist(),expected,w['atol'],w['rtol'])

    def test_frozen_splits_cover_rows_without_duplicate_feature_leakage(self):
        for m,_,p,_ in self.cases():
            self.assertEqual(p['splits'],partition(p['features']))
            combined=[i for k in SPLITS for i in p['splits'][k]]
            self.assertEqual(sorted(combined),list(range(m['rows'])))
            groups=[{tuple(p['features'][i]) for i in p['splits'][k]} for k in SPLITS]
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
            self.assertNotIn('quality',p['feature_names']);self.assertNotIn('diagnosis',p['feature_names'])

    def test_regeneration_reproduces_every_derived_byte(self):
        source=ROOT/'Benchmarks/Fixtures/supervised'
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);base=parent/'supervised';shutil.copytree(source/'originals',base/'originals')
            shutil.copyfile(source/'sources.lock.json',base/'sources.lock.json')
            (parent/'uci').mkdir();shutil.copyfile(ROOT/'Benchmarks/Fixtures/uci/winequality-red.csv',parent/'uci/winequality-red.csv')
            subprocess.run([sys.executable,str(ROOT/'Benchmarks/Tools/build_supervised_fixtures.py'),'--base-dir',str(base)],check=True,capture_output=True)
            for folder in ['inputs','references','splits']:
                for p in (source/folder).glob('*.json'):self.assertEqual(p.read_bytes(),(base/folder/p.name).read_bytes())
            self.assertEqual((source/'inventory.json').read_bytes(),(base/'inventory.json').read_bytes())

    def test_held_out_perturbations_cannot_change_training_fit(self):
        for _,w,p,_ in self.cases():
            q=copy.deepcopy(p);width=len(p['feature_names']);ntrain=len(p['splits']['train'])
            for k in ('validation','test'):
                for i in q['splits'][k]:
                    q['features'][i]=[v+1000 for v in q['features'][i]]
                    q['targets'][i]+=10000
            a=execute(prepare_runtime(p));b=execute(prepare_runtime(q))
            prefix=2*width+ntrain*width if p['operation']=='supervised-scale' else 3*width+1+ntrain
            self.assertEqual(a[:prefix].tolist(),b[:prefix].tolist())
            self.assertFalse((a==b).all())
            with mp.workdps(80):
                m1,s1,_=scaled_reference(p);m2,s2,_=scaled_reference(q)
                self.assertEqual(m1,m2);self.assertEqual(s1,s2)

    def test_training_targets_do_not_affect_scaling_or_partitions(self):
        for _,_,p,_ in self.cases():
            if p['operation']!='supervised-scale':continue
            q=copy.deepcopy(p);q['targets']=[-999.]*len(q['targets'])
            self.assertEqual(execute(prepare_runtime(p)).tolist(),execute(prepare_runtime(q)).tolist())
            self.assertEqual(partition(p['features']),partition(q['features']))

    def test_invalid_splits_identity_shape_and_answers_rejected(self):
        for m,_,p,_ in self.cases():
            for change in ['overlap','missing','negative','boolean','identity','ragged','duplicate-feature','leak','extra-split']:
                q=copy.deepcopy(p)
                if change=='overlap':q['splits']['test'][0]=q['splits']['train'][0]
                elif change=='missing':q['splits']['test'].pop()
                elif change=='negative':q['splits']['test'][0]=-1
                elif change=='boolean':q['splits']['test'][0]=True
                elif change=='identity':q['row_ids'][0]=q['row_ids'][1]
                elif change=='ragged':q['features'][0].pop()
                elif change=='duplicate-feature':q['features'][q['splits']['test'][0]]=q['features'][q['splits']['train'][0]].copy()
                elif change=='leak':q['expected']=[1]
                else:q['splits']['other']=[]
                with self.subTest(case=m['id'],change=change),self.assertRaises(ContractError):validate_input(q,p['operation'],m['rows'])

    def test_complete_reference_detects_wrong_last_prediction_and_baseline(self):
        for _,w,p,expected in self.cases():
            wrong=expected.copy();wrong[-1]+=1
            with self.assertRaises(ContractError):validate_values(wrong,expected,w['atol'],w['rtol'])
            if p['operation']=='supervised-ols-cpu':
                index=3*len(p['feature_names'])+len(p['features'])
                wrong=expected.copy();wrong[index]+=1
                with self.assertRaises(ContractError):validate_values(wrong,expected,w['atol'],w['rtol'])
                index+=1
                wrong=expected.copy();wrong[index]+=1
                with self.assertRaises(ContractError):validate_values(wrong,expected,w['atol'],w['rtol'])

    def test_reference_is_stable_at_higher_precision(self):
        for _,_,payload,expected in self.cases():
            with mp.workdps(120):
                higher=[float(v) for v in high_precision_reference(payload)]
            self.assertEqual(higher,expected)
