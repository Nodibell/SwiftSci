"""Production APIs only; serial rotated batches with full output comparison."""
import os,sys,json,time,datetime,hashlib,subprocess,statistics,platform,shutil
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
BINS=Path(os.environ['BENCH_BIN_DIR']).resolve()
REPOS={'SwiftSci':Path(__file__).resolve().parents[4], 'Kiraa':Path(os.environ['KIRAA_REPO']).resolve()}
DATA=BASE/'data'
OPS=['csv','filter','filter_int','filter_int32','filter_float32','sort','group','group_two','pipeline','stats','correlation','forecast']
COMMANDS={'SwiftSci':[str(BINS/'SciBench')], 'pandas':[sys.executable,str(BASE/'python_bench.py')], 'Kiraa':[str(BINS/'KiraaBench')]}
NO_KIRAA={'filter_int32','filter_float32','correlation','forecast'}
ENV=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for k in list(ENV):
 if k.startswith('SWIFTPANDAS_'): del ENV[k]
def capture(cmd): return subprocess.check_output(cmd,text=True).strip()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=BASE/'results'/stamp;out.mkdir(parents=True)
reps=int(os.environ.get('BENCH_REPS','5')); batches=int(os.environ.get('BENCH_BATCHES','3'))
sizes=[int(n) for n in os.environ.get('BENCH_SIZES','100000,1000000').split(',')]
ops=os.environ.get('BENCH_OPS',','.join(OPS)).split(',')
meta={'time_utc':stamp,'python':sys.version,'pandas':pd.__version__,'numpy':np.__version__,'platform':platform.platform(),
 'chip':capture(['sysctl','-n','machdep.cpu.brand_string']),'memory_bytes':int(capture(['sysctl','-n','hw.memsize'])),
 'swift':capture(['swift','--version']),'warmups_per_process':2,'repetitions_per_process':reps,'batches':batches,
 'commands':COMMANDS,'thread_limits':{k:ENV[k] for k in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},'repos':{},'sha256':{}}
for label,repo in [('SwiftSci','SwiftSci'),('Kiraa','kiraa-swift-pandas')]:
 path=str(REPOS[label])
 diff=capture(['git','-C',path,'diff','HEAD'])
 if diff: raise RuntimeError(f'{label} has tracked modifications; record/review before benchmarking')
 meta['repos'][label]={'commit':capture(['git','-C',path,'rev-parse','HEAD']),'status':capture(['git','-C',path,'status','--short'])}
 meta['sha256'][label]=sha(COMMANDS[label][0])
for p in [BASE/'run.py',BASE/'python_bench.py',BASE/'Package.swift',BASE/'Package.resolved',*BASE.glob('Sources/*/*.swift')]:
 meta['sha256'][str(p.relative_to(BASE))]=sha(p)
 dest=out/'harness'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
(out/'requirements.txt').write_text(capture([sys.executable,'-m','pip','freeze'])+'\n')
startup={name:[] for name in COMMANDS}
for iteration in range(8):
 names=list(COMMANDS); offset=iteration%3;names=names[offset:]+names[:offset]
 for name in names:
  start=time.perf_counter();subprocess.run(COMMANDS[name]+['startup'],env=ENV,capture_output=True,check=True)
  if iteration: startup[name].append(time.perf_counter()-start)
(out/'startup.json').write_text(json.dumps(startup,indent=2))
rows=[]
def parity(actual,expected,op):
 assert set(actual)==set(expected),(actual.keys(),expected.keys())
 keys=['group','group2'] if op=='group_two' else ['group'] if op in ['group','pipeline'] else []
 def order(d):
  return np.lexsort(tuple(np.asarray(d[k],dtype=float) for k in reversed(keys))) if keys else None
 ai,bi=order(actual),order(expected)
 for col in actual:
  a,b=np.array(actual[col],dtype=float),np.array(expected[col],dtype=float)
  if keys:a,b=a[ai],b[bi]
  np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-8,equal_nan=True,err_msg=col)
 return sum(len(v) for v in actual.values())
for size_index,n in enumerate(sizes):
 csv=DATA/f'data-{n}.csv'
 if not csv.exists():raise FileNotFoundError(csv)
 meta.setdefault('inputs',{})[str(csv)]=sha(csv)
 for op_index,op in enumerate(ops):
  libs=[name for name in COMMANDS if name!='Kiraa' or op not in NO_KIRAA]
  row={'rows':n,'operation':op,'libraries':{name:{'times':[],'peak_rss_bytes':[],'output_hashes':[],'parity':'passed'} for name in libs},'batches':[]}
  for batch in range(batches):
   offset=(size_index+op_index+batch)%len(libs);order=libs[offset:]+libs[:offset]
   answers={}; raw={}
   for name in order:
    proc=subprocess.run(COMMANDS[name]+[op,str(n),str(reps),str(csv)],env=ENV,capture_output=True,text=True)
    if proc.returncode:
     (out/f'{n}-{op}-{batch}-{name}-error.txt').write_text(proc.stdout+proc.stderr)
     raise RuntimeError(f'{name}/{op} process failed; see {out}')
    answer=json.loads(proc.stdout);answers[name]=answer.pop('result');raw[name]=answer
    bucket=row['libraries'][name];bucket['times']+=answer['times'];bucket['peak_rss_bytes'].append(answer['peak_rss_bytes']);bucket['output_hashes'].append(hashlib.sha256(json.dumps(answers[name],sort_keys=True).encode()).hexdigest())
   for name in libs:
    try:row['libraries'][name]['checked_values_per_batch']=parity(answers[name],answers['pandas'],op)
    except (AssertionError,ValueError) as error:
     row['libraries'][name]['parity']='FAILED'
     (out/f'{n}-{op}-{batch}-{name}-parity.txt').write_text(str(error))
   row['batches'].append({'order':order,'results':raw})
   print(f'{n} {op} batch {batch+1}/{batches} complete',flush=True)
  rows.append(row);(out/'measurements.json').write_text(json.dumps(rows,indent=2))
  print('  '+' | '.join(f'{name}: {statistics.median(v["times"])*1000:.3f} ms [{v["parity"]}]' for name,v in row['libraries'].items()),flush=True)
(out/'metadata.json').write_text(json.dumps(meta,indent=2))
(BASE/'latest-result.txt').write_text(str(out)+'\n')
print('RESULTS',out,flush=True)
if any(v['parity']!='passed' for r in rows for v in r['libraries'].values()):sys.exit(2)
