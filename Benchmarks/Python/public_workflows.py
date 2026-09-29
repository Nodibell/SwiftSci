"""CPU comparison via pandas, NumPy/SciPy and fresh-process NumPy state reload."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy import linalg


def execute(p, directory):
    directory=Path(directory);directory.mkdir()
    if p['operation']=='scientific-workflow':
        left_path=directory/'observations.csv';right_path=directory/'calibration.csv'
        left_path.write_text(p['observations_csv']);right_path.write_text(p['calibration_csv'])
        left=pd.read_csv(left_path,dtype={'row_id':'int64','site':'int64','signal':'float64','target':'float64','quality':'float64'})
        right=pd.read_csv(right_path,dtype={'calibration_id':'int64','site':'int64','offset':'float64'})
        filtered=left[left.quality >= 1]
        joined=filtered.merge(right,on='site',how='inner').sort_values(['row_id','calibration_id'],kind='stable')
        x=joined[p['feature_order']].to_numpy(dtype=np.float64);y=joined.target.to_numpy()
        design=np.column_stack([np.ones(len(x)),x]);beta=linalg.lstsq(design,y,lapack_driver='gelsd')[0]
        pred=design@beta;residual=pred-y
        return np.concatenate([[len(left),len(filtered),len(joined),2],joined.row_id,joined.calibration_id,
            x.ravel(),y,beta,pred,residual,[residual@residual]]).tolist()
    x=np.asarray(p['train_features'],dtype=np.float64);q=np.asarray(p['heldout_features'],dtype=np.float64)
    y=np.asarray(p['train_targets'],dtype=np.float64)
    mean=x.mean(axis=0);std=x.std(axis=0);std=np.where(std<1e-12,1.,std)
    z=(x-mean)/std;zq=(q-mean)/std
    design=np.column_stack([np.ones(len(x)),z]);beta=linalg.lstsq(design,y,lapack_driver='gelsd')[0]
    train=design@beta;pred=np.column_stack([np.ones(len(q)),zq])@beta
    residual=pred-np.asarray(p['heldout_targets'])
    values=np.concatenate([[len(x),len(q),2],p['train_ids'],p['heldout_ids'],mean,std,z.ravel(),zq.ravel(),beta,train,pred,residual]).tolist()
    (directory/'before-reload.json').write_text(json.dumps(values))
    if p['route']!='fit':
        model=directory/'model.npz'
        np.savez(model,mean=mean,std=std,beta=beta,feature_names=np.asarray(p['feature_names']))
        checksum=hashlib.sha256(model.read_bytes()).hexdigest()
        request={'model':str(model),'sha256':checksum,'features':p['heldout_features'],'feature_names':p['feature_names'],'parent_pid':os.getpid()}
        source=directory/'reload-request.json';target=directory/'reload-response.json'
        source.write_text(json.dumps(request))
        with (directory/'reload.log').open('w') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(source),str(target)],stdout=log,stderr=subprocess.STDOUT)
            try: code=child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill();child.wait();raise ValueError('Fresh-process reload timed out')
        if code: raise ValueError('NumPy fresh-process reload failed')
        response=json.loads(target.read_text())
        if response['pid'] != child.pid or response['pid']==os.getpid() or response['sha256']!=checksum:
            raise ValueError('Invalid fresh-process evidence')
        values += response['predictions']
    return values+[1]


def reload(source, destination):
    p=json.loads(Path(source).read_text());model=Path(p['model'])
    if hashlib.sha256(model.read_bytes()).hexdigest()!=p['sha256'] or p['parent_pid']==os.getpid():
        raise ValueError('Invalid model identity or process')
    with np.load(model,allow_pickle=False) as state:
        if set(state.files)!={'mean','std','beta','feature_names'} or state['feature_names'].tolist()!=p['feature_names']:
            raise ValueError('Persisted schema differs')
        q=np.asarray(p['features'],dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=2 or not np.isfinite(q).all():raise ValueError('Invalid query shape')
        pred=np.column_stack([np.ones(len(q)),(q-state['mean'])/state['std']])@state['beta']
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
    Path(destination).write_text(json.dumps({'pid':os.getpid(),'sha256':p['sha256'],'predictions':pred.tolist()}))


if __name__=='__main__':reload(sys.argv[1],sys.argv[2])
