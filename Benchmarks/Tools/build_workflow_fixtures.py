#!/usr/bin/env python3
"""Regenerate original public-workflow fixtures and independent expected values."""
from pathlib import Path
import copy
from contracts import digest, write_json
from workflow_fixtures import validate_input
from workflow_reference import reference

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('Benchmarks/Fixtures/workflows')


def cases():
    p={'operation':'scientific-workflow','feature_order':['offset','signal'],
       'observations_csv':'row_id,site,signal,target,quality\n91,1,2,7,1\n13,2,4,12,1\n77,1,6,17,0\n25,3,8,18,1\n62,2,3,11,1\n48,1,5,15,1\n39,3,7,20,1\n80,9,100,900,1\n99,2,9,80,\n',
       'calibration_csv':'calibration_id,site,offset\n7,1,0.5\n3,2,2.5\n9,3,-1.5\n2,2,3.5\n8,8,40.5\n'}
    yield 'workflow-scientific',p,9
    q=copy.deepcopy(p);q['calibration_csv']=q['calibration_csv'].replace('2,2,3.5','2,7,3.5')
    yield 'workflow-scientific-unique-keys',q,9
    q=copy.deepcopy(p);q['feature_order'].reverse()
    yield 'workflow-scientific-feature-order',q,9
    q=copy.deepcopy(p)
    for key in ['observations_csv','calibration_csv']:
        lines=q[key].splitlines();q[key]='\n'.join([lines[0]]+list(reversed(lines[1:])))+'\n'
    yield 'workflow-scientific-row-order',q,9
    x=[[1,2],[2,-1],[3,4],[4,0],[6,3],[8,-2]]
    train=[3+2*a-.5*b for a,b in x]
    for route in ['fit','native','coreml']:
        p={'operation':'persisted-regression','route':route,'feature_names':['signal','offset'],
           'train_ids':[11,22,33,44,55,66],'train_features':x,'train_targets':train,
           'heldout_ids':[77,88,99],'heldout_features':[[5,1],[9,-3],[0,5]],'heldout_targets':[12.75,23,1]}
        yield 'workflow-regression-'+route,p,6
        q=copy.deepcopy(p);q['heldout_features']=[[50,10],[90,-30],[0,50]];q['heldout_targets']=[-50,70,123]
        yield 'workflow-regression-'+route+'-heldout-shift',q,6


def main():
    names=['build_workflow_fixtures.py','workflow_fixtures.py','workflow_reference.py']
    sources={'schema_version':1,'sources':[{'path':'../../Tools/'+n,'sha256':digest((ROOT/'Benchmarks/Tools'/n).read_bytes()),'bytes':(ROOT/'Benchmarks/Tools'/n).stat().st_size} for n in names]}
    lock=BASE/'sources.lock.json';write_json(ROOT/lock,sources);lock_bytes=(ROOT/lock).read_bytes()
    profiles={'scientific-workflow':[],'persisted-workflow':[]}
    for name,p,rows in cases():
        validate_input(p,p['operation'],rows)
        inp=BASE/'inputs'/f'{name}.json';out=BASE/'references'/f'{name}.json'
        write_json(ROOT/inp,p);raw=(ROOT/inp).read_bytes()
        write_json(ROOT/out,{'operation':p['operation'],'inputIdentity':{'sha256':digest(raw),'bytes':len(raw)},
            'source':{'sha256':digest(lock_bytes)},'model':{'observations':rows},'independentReference':{'values':reference(p)}})
        ans=(ROOT/out).read_bytes()
        manifest={'schema_version':1,'id':name,'kind':'numerical-fixture-v1','generator_version':1,'rows':rows,
            'operation':p['operation'],'fixture':str(inp),'sha256':digest(raw),'size_bytes':len(raw),
            'source_fixture':str(lock),'source_sha256':digest(lock_bytes),'source_size_bytes':len(lock_bytes),
            'source':'Original bounded scientific and training workflow fixtures; see Benchmarks/Fixtures/workflows/README.md',
            'license':'MIT, repository LICENSE','reference_basis':'workflow-reference-v1','reference_fixture':str(out),
            'reference_sha256':digest(ans),'reference_size_bytes':len(ans),'tolerances':{'atol':1e-8,'rtol':1e-10}}
        write_json(ROOT/'Benchmarks/Specs/datasets'/f'{name}.json',manifest)
        profile='scientific-workflow' if p['operation']=='scientific-workflow' else 'persisted-workflow'
        profiles[profile].append({'id':name,'dataset':name,'workload':p['operation']+'-v1'})
    for name,entries in profiles.items():
        write_json(ROOT/'Benchmarks/Specs/profiles'/f'{name}.json',{'schema_version':1,'id':name,'warmups':0,'samples':2,'batches':1,'timeout_seconds':120,'cases':entries})


if __name__=='__main__': main()
