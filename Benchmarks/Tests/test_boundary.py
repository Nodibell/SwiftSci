from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Tools"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Python"))
import copy
import unittest
from boundary_cases import cases
from boundary_fixtures import validate_input, output_count
from boundary_reference import reference
from boundary_workloads import execute
from contracts import ContractError


class BoundaryTests(unittest.TestCase):
    def test_all_cases_and_repeatability(self):
        for name, p in cases():
            with self.subTest(name=name):
                validate_input(p, p['operation'], len(p['row_ids']))
                expected = reference(p)
                actual = execute(p)
                self.assertEqual(actual, execute(p))
                self.assertEqual(len(actual), output_count(p))
                self.assertEqual(len(actual), len(expected))
                for a, b in zip(actual, expected):
                    self.assertAlmostEqual(a,b,delta=2e-6 if p['dtype']=='float32' else 2e-13)
                self.assertEqual(actual[3:7], [63,45,91,39] if p['ascending'] else [91,39,63,45])
                self.assertEqual(actual[-1], 1)

    def test_dtype_changes_are_observable(self):
        p = dict(cases())['boundary-cpu-float64-narrow']
        q = copy.deepcopy(p); q['dtype']='float32'
        self.assertNotEqual(reference(p)[7:-1], reference(q)[7:-1])

    def test_rejects_bad_boundaries_and_expected_answers(self):
        p = next(cases())[1]
        changes = [('device','auto'),('dtype','int32'),('ascending',1),('bias',float('nan')),
                   ('feature_order',['x0','x0']),('row_ids',[91]*7),('features',[[1]*2]*6),
                   ('weights',[True,1]),('filter_threshold',1025)]
        for key,value in changes:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key), self.assertRaises(ContractError): validate_input(q,q['operation'],7)
        q=copy.deepcopy(p);q['device']='gpu';q['dtype']='float64'
        with self.assertRaises(ContractError):validate_input(q,q['operation'],7)
        q=copy.deepcopy(p);q['expected_predictions']=[]
        with self.assertRaises(ContractError):validate_input(q,q['operation'],7)

    def test_target_alignment_and_feature_order_affect_output(self):
        p=next(cases())[1]
        for field in ('targets', 'feature_order', 'weights'):
            q=copy.deepcopy(p);q[field].reverse()
            self.assertNotEqual(reference(p),reference(q))


class BoundarySweepTests(unittest.TestCase):
    def test_each_stage_actual_output_matches_independent_slice(self):
        import boundary_sweep as sweep
        for dtype,device in [('float32','cpu'),('float32','gpu'),('float64','cpu')]:
            for width in (8,64):
                for stage in ('conversion','prepared','pipeline'):
                    p={'operation':sweep.OPERATION,'recipe':'dyadic-v1','device':device,'dtype':dtype,
                       'rows':128,'columns':width,'stage':stage}
                    with self.subTest(dtype=dtype,device=device,width=width,stage=stage):
                        expected=sweep.reference(p); state=sweep.prepare(p);actual=sweep.execute(state)
                        self.assertEqual(actual,sweep.execute(state))
                        self.assertEqual(len(actual),sweep.output_count(p))
                        self.assertEqual(actual,expected)

    def test_descriptor_rejects_unsupported_sizes_and_embedded_answers(self):
        import boundary_sweep as sweep
        p={'operation':sweep.OPERATION,'recipe':'dyadic-v1','device':'cpu','dtype':'float64',
           'rows':128,'columns':8,'stage':'conversion'}
        for key,value in [('rows',129),('columns',7),('stage','all'),('device','gpu'),('recipe','unknown')]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):sweep.validate_descriptor(q,q['operation'],q['rows'])
        q=copy.deepcopy(p);q['expected_values']=[]
        with self.assertRaises(ContractError):sweep.validate_descriptor(q,q['operation'],q['rows'])

if __name__=='__main__':unittest.main()
