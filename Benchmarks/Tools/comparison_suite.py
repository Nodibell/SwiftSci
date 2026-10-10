#!/usr/bin/env python3
"""Run the complete tabular comparison plus existing CPU qualification."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from contracts import read_json, write_json, load_profile, load_workload
from reporting import summarize
from runner import source_identity
from execution_policy import MODES, PRODUCTION_MODE, policy

CONCURRENCY_POLICY = {
    'classification': 'asymmetric-engine-concurrency',
    'effective_thread_counts_measured': False,
    'python_numerical_libraries': 'BLAS/OpenMP thread environment requested at one',
    'polars': 'thread pool verified at one',
    'duckdb': 'connection threads configured at one',
    'swiftsci': 'BLAS environment requested at one; GCD parallelism is not capped by it. '
                'CSV chunk parsing uses up to eight workers; CSV columns and row gathering may run concurrently.',
    'interpretation': 'Observed materialized-operation wall time under these engine-specific policies. '
                      'Not an equal-thread or single-core efficiency comparison.',
}

RUNS = [
    ('standard', 'swiftsci,pandas,polars,duckdb'),
    ('extended', 'swiftsci,pandas,polars,duckdb'),
    ('public-data', 'swiftsci,pandas,polars,duckdb'),
    ('nist', 'swiftsci,pandas,polars,duckdb'),
    ('migration', 'swiftsci,pandas'),
    ('tabular-migration', 'swiftsci,pandas,polars,duckdb'),
]


def report(root, output, manifest):
    rows, failures = [], []
    for entry in manifest['runs']:
        run_path=output/entry['profile']/'run.json'
        if not run_path.is_file():
            failures.append(dict(profile=entry['profile'], error='Run did not produce evidence'))
            continue
        run=read_json(run_path)
        rows.extend(dict(profile=entry['profile'], **row) for row in summarize(run))
        failures.extend(dict(profile=entry['profile'], case=e['case_id'], engine=e['engine'], batch=e['batch'], error=e.get('failed_result',{}).get('error', e.get('error'))) for e in run['events'] if e['status']!='passed')
    mode = manifest.get('execution', {}).get('mode', 'legacy-capped')
    write_json(output/'concurrency-policy.json', manifest.get('execution', CONCURRENCY_POLICY))
    write_json(output/'measurements.json', rows)
    write_json(output/'failures.json', failures)
    lines=['# Four-engine benchmark comparison', '',
           f"Source commit: `{manifest['source']['commit']}`. Source fingerprint: `{manifest['source']['tree_sha256']}`.", '',
           'All runs use native arm64 execution. Python numerical-library thread limits are requested at one; Polars and DuckDB use one thread. Swift may use GCD parallelism, including up to eight CSV chunk workers and concurrent column parsing or row gathering. These are asymmetric engine policies, not an equal-thread or single-core comparison. Effective thread counts were not measured. See [concurrency-policy.json](concurrency-policy.json). Each performance case uses three fresh processes, two warmups, and five measured samples per process. Times retain materialized output; correctness validation follows timing. Peak RSS includes the interpreter, input preparation, independent reference buffers, and validation allocations, so it is whole-process memory rather than kernel allocation.', '',
           'These are sequential desktop measurements, not a formal performance baseline. Group and join outputs are canonicalized after timing. Stable sorting and numerical exports retain declared row alignment. DuckDB results are fetched into Arrow tables or NumPy arrays inside timing. Query setup outside timing applies only to already-prepared inputs; CSV, Parquet, and the wine pipeline include ingestion.', '',
           '## Coverage', '',
           '| Profile | Engines | Result | Audit |', '|---|---|---|---|']
    if mode == PRODUCTION_MODE:
        lines[4] = ('Execution mode is production-default. Each case uses at least five fresh worker processes per engine, with engine order rotated each round. '
                    'The controller removes thread overrides before process startup. Engines select their native pool sizes. '
                    'Worker records contain requested thread settings and available configured pool sizes, not measured active thread counts. '
                    'Power source and system power settings are captured at each profile boundary. See [concurrency-policy.json](concurrency-policy.json). '
                    'Timers retain materialized output; validation follows timing. RSS is the whole worker lifetime high-water mark, including setup and validation. '
                    'A fresh process does not imply a cold filesystem cache. Startup time is excluded from resident-operation timings.')
    for entry in manifest['runs']:
        lines.append(f"| {entry['profile']} | {entry['engines']} | {entry['status']} | {entry.get('audit','not passed')} |")
    full=load_profile(root,'migration');overlap=load_profile(root,'tabular-migration');included={c['id'] for c in overlap['cases']}
    omitted=[dict(case=c['id'],operation=load_workload(root,c['workload'])['operation'],reason='Not implemented in the native tabular adapters under this matched contract; retained in the Swift/Python migration run') for c in full['cases'] if c['id'] not in included]
    write_json(output/'tabular-exclusions.json', omitted)
    lines += ['', 'Polars and DuckDB cover all cases in standard, extended, public-data, and NIST, plus 19 migration cases. The other 17 migration cases remain measured by Swift and the existing Python worker. Numerical/model CPU qualification also uses the existing Swift/Python adapters; this does not claim Polars or DuckDB model coverage.', '', '## Timings', '', 'Milliseconds, median of process medians. A failed or unresolved result has no usable timing. Compare engines within the same profile and case.', '', '| Profile / case | Swift | Python | Polars | DuckDB |', '|---|---:|---:|---:|---:|']
    grouped={}
    for row in rows:grouped.setdefault((row['profile'],row['case_id']),{})[row['engine']]=row
    for (profile,case),engines in grouped.items():
        cells=[]
        for engine in ['swiftsci','pandas','polars','duckdb']:
            r=engines.get(engine)
            cells.append('not run' if r is None else 'failed' if r['status']!='passed' else 'unresolved' if not r['timing_resolved'] else f"{r['median_ns']/1e6:.6f}")
        lines.append(f"| {profile} / {case} | "+' | '.join(cells)+' |')
    lines += ['', '## Failures', '', f'{len(failures)} failed performance engine-case executions. See [failures.json](failures.json) for exact errors. Failed responses and certificates are retained.', '', 'CPU qualification is recorded separately in [cpu-acceptance/acceptance.json](cpu-acceptance/acceptance.json) and [cpu-acceptance/regression-policy.json](cpu-acceptance/regression-policy.json). A passing regression policy does not make retained numerical failures conformant.', '', 'Raw samples and RSS for each case are in [measurements.json](measurements.json); each profile directory includes its resolved contracts, worker versions, run record, and certificate.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swift-worker',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--mode',choices=MODES,default=PRODUCTION_MODE)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[2];output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),source=source_identity(root),execution=policy(args.mode),runs=[])
    for profile,engines in RUNS:
        print('Running',profile,engines,flush=True)
        entry=dict(profile=profile,engines=engines,status='infrastructure-error')
        with (output/(profile+'.log')).open('w') as log:
            base=[args.python,str(root/'Benchmarks/Tools/bench.py')]
            prepare=subprocess.run(base+['prepare','--profile',profile],cwd=root,stdout=log,stderr=subprocess.STDOUT)
            if prepare.returncode==0:
                completed=subprocess.run(base+['run','--mode',args.mode,'--profile',profile,'--engines',engines,'--swift-worker',args.swift_worker,'--python',args.python,'--output',str(output/profile)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
                if (output/profile/'run.json').is_file():entry['status']=read_json(output/profile/'run.json')['status']
                entry['exit_code']=completed.returncode
                if completed.returncode==0:
                    audit=subprocess.run(base+['audit',str(output/profile)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
                    entry['audit']='passed' if audit.returncode==0 else 'failed'
            else:entry['prepare_exit_code']=prepare.returncode
        manifest['runs'].append(entry);write_json(output/'suite.json',manifest)
        print(profile,entry,flush=True)
    print('Running all CPU conformance profiles',flush=True)
    with (output/'cpu-acceptance.log').open('w') as log:
        acceptance=subprocess.run([args.python,str(root/'Benchmarks/Tools/acceptance.py'),'--tier','cpu','--check-baseline','--swift-worker',args.swift_worker,'--python',args.python,'--output',str(output/'cpu-acceptance')],cwd=root,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(),cpu_policy_exit_code=acceptance.returncode)
    if source_identity(root)!=manifest['source']:raise RuntimeError('Source changed during comparison')
    write_json(output/'suite.json',manifest);report(root,output,manifest)
    print('Report:',output/'README.md',flush=True)
    return 0 if acceptance.returncode==0 and all(e['status']=='passed' and e.get('audit')=='passed' for e in manifest['runs']) else 1

if __name__=='__main__':raise SystemExit(main())
