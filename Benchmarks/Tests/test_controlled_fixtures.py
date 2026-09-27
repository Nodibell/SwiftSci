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
from controlled_fixtures import validate_input, output_count

sys.path.insert(0, str(ROOT/'Benchmarks/Python'))
from controlled_api_workloads import prepare as prepare_model, execute as execute_model, OPERATIONS
from search_workloads import vector_cosine, canonical_cosine, kernel_shap


class ControlledFixtures(unittest.TestCase):
    def cases(self):
        for case in load_profile(ROOT, 'controlled-conformance')['cases']:
            m = load_manifest(ROOT, case['dataset'])
            w = resolve_workload(m, load_workload(ROOT, case['workload']))
            payload = json.loads(prepare(ROOT, m).read_text())
            yield m, w, payload, reference(ROOT, m, w)

    def test_all_ten_cases_resolve_with_complete_independent_answers(self):
        cases = list(self.cases())
        self.assertEqual(len(cases), 10)
        for m, w, p, expected in cases:
            self.assertEqual(len(expected), output_count(p))
            self.assertEqual((w['atol'], w['rtol']), (1e-12, 1e-12))

    def test_both_generators_reproduce_inputs_references_and_inventory(self):
        for family, generator in [('controlled-models','build_controlled_model_fixtures.py'), ('controlled-search','build_controlled_search_fixtures.py')]:
            source = ROOT/'Benchmarks/Fixtures'/family
            with tempfile.TemporaryDirectory() as d:
                base = Path(d)
                if (source/'source-spec.json').exists():
                    shutil.copyfile(source/'source-spec.json', base/'source-spec.json')
                else:
                    shutil.copytree(source/'sources', base/'sources')
                subprocess.run([sys.executable,str(ROOT/'Benchmarks/Tools'/generator),'--base-dir',str(base)],check=True,capture_output=True)
                for folder in ['inputs','references']:
                    for path in (source/folder).glob('*.json'):
                        self.assertEqual(path.read_bytes(), (base/folder/path.name).read_bytes())
                if (source/'inventory.json').exists():
                    self.assertEqual((source/'inventory.json').read_bytes(), (base/'inventory.json').read_bytes())

    def test_runtime_comparators_match_and_reset_state(self):
        for m, w, p, expected in self.cases():
            prepared = prepare_model(p) if p['operation'] in OPERATIONS else p
            for _ in range(3):
                if p['operation'] in OPERATIONS:
                    actual = execute_model(prepared).tolist()
                elif p['operation'] == 'vector-cosine':
                    actual = canonical_cosine(vector_cosine(p), m['rows'], p['top_k'])
                else:
                    actual = kernel_shap(p)
                validate_values(actual, expected, w['atol'], w['rtol'])

    def test_fixed_parameters_have_independent_expected_predictions(self):
        cases = {p['operation']:expected for _,_,p,expected in self.cases()}
        self.assertEqual(cases['linear-fixed-cpu'], [3,2,-1,3,3,-5,10])
        logistic = cases['logistic-fixed-cpu']
        self.assertEqual(logistic[:5], [0,2,-1,.5,.5])
        self.assertEqual(logistic[-5:], [0,1,0,0,0])
        self.assertEqual(cases['kmeans-one-cpu'], [2,1,0,0,0,0,0,0,20])

    def test_kalman_checks_every_state_and_covariance(self):
        for _,_,p,expected in self.cases():
            if p['operation'] != 'kalman-fixed-cpu': continue
            n = p['state_size']; size = n+n*n
            self.assertEqual(len(expected), (len(p['observations'])+1)*size)
            for offset in range(0,len(expected),size):
                cov = expected[offset+n:offset+size]
                self.assertTrue(all(cov[i*n+i]>0 for i in range(n)))
                self.assertTrue(all(abs(cov[i*n+j]-cov[j*n+i])<1e-14 for i in range(n) for j in range(n)))
            wrong=expected.copy(); wrong[-1]+=1
            with self.assertRaises(ContractError): validate_values(wrong,expected,1e-12,1e-12)

    def test_ties_allow_equivalent_order_without_hiding_unsorted_results(self):
        self.assertEqual(canonical_cosine([(1,1.),(0,1.),(2,.6)],3,3),[0,1,1,1,2,.6])
        for matches in [[(0,.6),(1,1.)],[(0,1.),(0,.6)],[(3,1.),(0,.6)]]:
            with self.assertRaises(ValueError): canonical_cosine(matches,3,2)
        for m,_,_,expected in self.cases():
            if m['id']=='controlled-cosine-small-norm': self.assertEqual(expected,[0.,1.])

    def test_explanation_additivity_and_interaction_credit(self):
        outputs={m['id']:expected for m,_,p,expected in self.cases() if p['operation']=='kernel-shap'}
        self.assertEqual(outputs['controlled-kernel-affine'],[1,6,0,7])
        self.assertEqual(outputs['controlled-kernel-interaction'],[2,7,6,15])
        for values in outputs.values(): self.assertEqual(values[0]+sum(values[1:3]),values[3])

    def test_malformed_controlled_inputs_are_rejected(self):
        for m,_,p,_ in self.cases():
            leaked=copy.deepcopy(p); leaked['expected']=[0]
            with self.assertRaises(ContractError): validate_input(leaked,p['operation'],m['rows'])
        for m,_,p,_ in self.cases():
            q=copy.deepcopy(p); op=p['operation']
            if op in ('linear-fixed-cpu','logistic-fixed-cpu'): q['weights'][0]=True
            elif op=='kmeans-one-cpu': q['n_clusters']=2
            elif op=='kalman-fixed-cpu': q['measurement_noise'][0][0]=-1
            elif op=='vector-cosine': q['query']=[0]*len(q['query'])
            else:q['model']['weights']=[1]
            with self.assertRaises(ContractError): validate_input(q,op,m['rows'])
