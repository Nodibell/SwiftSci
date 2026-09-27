import copy
import json
import sys
import unittest
import mpmath as mp
import numpy as np
from test_contracts import ROOT
from contracts import ContractError, validate_values
from supervised_fixtures import validate_input
from classification_reference import scores
from build_supervised_fixtures import scaled_reference
sys.path.insert(0,str(ROOT/'Benchmarks/Python'))
from classification_workloads import scores as runtime_scores
from supervised_workloads import prepare, execute


class ClassificationFixtures(unittest.TestCase):
    def payload(self,epochs):
        return json.loads((ROOT/f'Benchmarks/Fixtures/supervised/inputs/wdbc-supervised-logistic-cpu-epochs{epochs}.json').read_text())

    def test_zero_updates_check_initialization_threshold_and_auc_ties(self):
        p=self.payload(0);a=execute(prepare(p));width=len(p['feature_names']);n=len(p['features'])
        self.assertEqual(a[2*width:3*width+1].tolist(),[0.]*(width+1))
        self.assertEqual(a[3*width+1:3*width+1+2*n].tolist(),[.5]*(2*n))
        self.assertEqual(a[3*width+1+2*n:3*width+1+3*n].tolist(),[0.]*n)
        for start in range(len(a)-44,len(a),11):self.assertEqual(a[start+9],.5)

    def test_first_update_matches_closed_form_at_zero(self):
        p=self.payload(1);a=execute(prepare(p));width=len(p['feature_names'])
        with mp.workdps(80):
            means,scales,z=scaled_reference(p);y=[mp.mpf(p['targets'][i]) for i in p['splits']['train']]
            rate=mp.mpf('.125')
            expected=[-rate*mp.fsum(mp.mpf('.5')-v for v in y)/len(y)]
            expected += [-rate*mp.fsum((mp.mpf('.5')-v)*row[j] for v,row in zip(y,z['train']))/len(y) for j in range(width)]
        validate_values(a[2*width:3*width+1].tolist(),list(map(float,expected)),1e-12,1e-12)

    def test_metric_values_have_hand_checkable_ties_and_zero_denominators(self):
        targets=[0,1,0,1]
        for probabilities,auc in [([.5,.5,.5,.5],.5),([.1,.8,.8,.9],.875)]:
            with mp.workdps(80):expected=list(map(float,scores(targets,list(map(mp.mpf,probabilities)))))
            validate_values(runtime_scores(np.array(targets),np.array(probabilities)),expected,1e-12,1e-12)
            self.assertEqual(expected[9],auc)
        result=list(map(float,scores(targets,[mp.mpf('.5')]*4)))
        self.assertEqual(result[:8],[2,0,2,0,.5,0,0,0])
        self.assertEqual(result[10],.25)

    def test_learning_controls_and_class_coverage_are_strict(self):
        p=self.payload(1)
        for key,value in [('epochs',True),('epochs',-1),('epochs',129),('learning_rate',True),('learning_rate',0),('learning_rate',.1)]:
            q=copy.deepcopy(p);q['training'][key]=value
            with self.assertRaises(ContractError):validate_input(q,q['operation'],len(q['features']))
        for mutation in ['extra-control','fractional-label','single-class']:
            q=copy.deepcopy(p)
            if mutation=='extra-control':q['training']['regularization']=1
            elif mutation=='fractional-label':q['targets'][0]=.5
            else:
                for i in q['splits']['test']:q['targets'][i]=0
            with self.assertRaises(ContractError):validate_input(q,q['operation'],len(q['features']))

    def test_test_labels_cannot_change_any_model_output(self):
        p=self.payload(32);q=copy.deepcopy(p)
        for name in ('validation','test'):
            for i in q['splits'][name]:q['targets'][i]=1-q['targets'][i]
        a,b=execute(prepare(p)),execute(prepare(q))
        self.assertEqual(a[:-44].tolist(),b[:-44].tolist())
        self.assertNotEqual(a[-44:].tolist(),b[-44:].tolist())

    def test_complete_probabilities_labels_and_final_score_are_checked(self):
        p=self.payload(32);a=execute(prepare(p)).tolist();width=len(p['feature_names']);n=len(p['features'])
        for index in (3*width+1+2*n-1,3*width+1+3*n-1,len(a)-1):
            wrong=a.copy();wrong[index]+=1
            with self.assertRaises(ContractError):validate_values(wrong,a,1e-8,1e-9)
