from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Benchmarks/Tools'))
sys.path.insert(0,str(ROOT/'Benchmarks/Python'))
from build_workflow_fixtures import cases
from workflow_fixtures import validate_input
from workflow_reference import reference, solve
from public_workflows import execute
from contracts import ContractError, validate_values
from datasets import load_manifest
from numerical_fixtures import source_and_input, expected_values


class PublicWorkflowTests(unittest.TestCase):
    def test_all_fixtures_match_independent_reference_and_replay(self):
        for name,p,rows in cases():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                validate_input(p,p['operation'],rows)
                expected=reference(p)
                for repeat in range(2):
                    directory=Path(temp)/str(repeat)
                    actual=execute(p,directory)
                    validate_values(actual,expected,1e-8,1e-10)
                    if p.get('route') in ('native','coreml'):
                        response=json.loads((directory/'reload-response.json').read_text())
                        self.assertNotEqual(response['pid'],os.getpid())
                        self.assertEqual(response['predictions'],actual[-4:-1])

    def test_pinned_provenance_and_reference_regeneration(self):
        for name,p,rows in cases():
            m=load_manifest(ROOT,name)
            self.assertEqual(json.loads(source_and_input(ROOT,m)),p)
            self.assertEqual(expected_values(ROOT,m,p),reference(p))

    def test_join_multiplicity_null_filter_and_order(self):
        c={name:p for name,p,_ in cases()}
        v=reference(c['workflow-scientific'])
        self.assertEqual(v[:4],[9,7,8,2])
        self.assertEqual(v[4:12],[13,13,25,39,48,62,62,91])
        self.assertEqual(v[12:20],[2,3,9,9,7,2,3,7])
        self.assertEqual(v,reference(c['workflow-scientific-row-order']))
        self.assertNotEqual(v,reference(c['workflow-scientific-feature-order']))

    def test_heldout_changes_cannot_change_fitted_state(self):
        c={name:p for name,p,_ in cases()}
        for route in ('fit','native','coreml'):
            a=c['workflow-regression-'+route];b=c['workflow-regression-'+route+'-heldout-shift']
            # Prefix includes IDs, fitted means/std, and transformed training rows.
            prefix=3+6+3+4+12
            av,bv=reference(a),reference(b)
            self.assertEqual(av[:prefix],bv[:prefix])
            self.assertEqual(av[prefix+6:prefix+6+9],bv[prefix+6:prefix+6+9])
            self.assertNotEqual(av,bv)

    def test_leaking_heldout_rows_changes_reference(self):
        p=next(p for n,p,_ in cases() if n=='workflow-regression-fit')
        q=copy.deepcopy(p)
        q['train_features']+=q['heldout_features'];q['train_targets']+=q['heldout_targets'];q['train_ids']+=q['heldout_ids']
        self.assertNotEqual(reference(p),reference(q))
        with self.assertRaises(ContractError):validate_input(q,q['operation'],9)

    def test_rank_deficiency_has_no_invented_answer(self):
        with self.assertRaisesRegex(ValueError,'Rank-deficient'):solve([[1,2],[2,4],[3,6]],[3,5,7])

    def test_rejects_malformed_workflow_inputs(self):
        p=next(p for n,p,_ in cases() if n=='workflow-regression-fit')
        for key,value in [('route','unknown'),('feature_names',['signal','signal']),('train_targets',[1]),
                          ('train_ids',[11]*6),('heldout_ids',[11,88,99]),('train_features',[[True,1]]*6),
                          ('heldout_features',[[float('nan'),1]]),('expected_predictions',[1])]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):validate_input(q,q['operation'],6)
        p=next(p for n,p,_ in cases() if n=='workflow-scientific')
        for key,value in [('feature_order',['signal','signal']),('observations_csv',p['observations_csv'].replace('91,1','13,1')),
                          ('calibration_csv',p['calibration_csv'].replace('0.5','nan'))]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):validate_input(q,q['operation'],9)

    def test_each_output_element_is_checked(self):
        for _,p,_ in cases():
            expected=reference(p)
            for index in range(len(expected)):
                wrong=expected.copy();wrong[index]+=1
                with self.assertRaises(ContractError):validate_values(wrong,expected,1e-8,1e-10)
            with self.assertRaises(ContractError):validate_values(expected[:-1],expected,1e-8,1e-10)

    def test_fresh_process_rejects_corrupt_model_and_wrong_schema(self):
        p=next(p for n,p,_ in cases() if n=='workflow-regression-coreml')
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'sample';execute(p,directory)
            source=directory/'reload-request.json';request=json.loads(source.read_text())
            output=directory/'invalid-response.json'
            for mutation in ('checksum','schema','truncated'):
                altered=copy.deepcopy(request)
                if mutation=='checksum':altered['sha256']='0'*64
                elif mutation=='schema':altered['feature_names'].reverse()
                else:
                    broken=directory/'broken.npz';broken.write_bytes(b'invalid')
                    altered['model']=str(broken);altered['sha256']=hashlib.sha256(broken.read_bytes()).hexdigest()
                source.write_text(json.dumps(altered))
                result=subprocess.run([sys.executable,str(ROOT/'Benchmarks/Python/public_workflows.py'),str(source),str(output)],capture_output=True,timeout=60)
                self.assertNotEqual(result.returncode,0,mutation)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
