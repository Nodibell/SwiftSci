#!/usr/bin/env python3
"""Report stage timings and explicitly named memory measures for a verified boundary sweep."""
import argparse
import json
from pathlib import Path
from acceptance import inspect_run
from contracts import read_json, require
from reporting import summarize


def metrics(payload, summary):
    rows, columns = payload['rows'], payload['columns']
    selected = rows * 3 // 4
    duration = summary['median_ns'] if summary['status'] == 'passed' and summary['timing_resolved'] else None
    return dict(case_id=summary['case_id'], engine=summary['engine'], requested_swift_device=payload['device'],
                comparator_device='cpu' if summary['engine']=='pandas' else payload['device'],
                dtype=payload['dtype'], stage=payload['stage'], input_rows=rows, selected_rows=selected, columns=columns,
                source_feature_payload_bytes=rows*columns*8,
                selected_tensor_payload_bytes=selected*columns*(4 if payload['dtype']=='float32' else 8),
                process_lifetime_peak_rss_bytes=summary['peak_rss_bytes'],
                median_ns=duration, selected_elements_per_second=selected*columns*1e9/duration if duration else None,
                status=summary['status'])


def report(root, directory):
    run=read_json(directory/'run.json')
    require(run['plan']['profile']['id']=='boundary-sweep', 'Not a boundary sweep run')
    result=inspect_run(directory,run['plan']['profile'],run['plan']['source'],list(run['plan']['engines']),root)
    require(result['coverage_complete'], 'Sweep has incomplete evidence')
    cases={c['case']['id']:read_json(root/c['dataset']['fixture']) for c in run['plan']['cases']}
    return dict(schema_version=1, source=run['plan']['source'], status=result['status'],
        formal_performance_baseline=False,
        memory_scope='Peak RSS is whole-process lifetime including setup and validation. Payload bytes describe logical feature/tensor data only; they are not total resident memory or operation allocations.',
        timing_scope='Stage-specific completed work and extraction; durations across different stages are not interchangeable. NumPy always executes on CPU.',
        results=[metrics(cases[s['case_id']],s) for s in summarize(run)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(report(Path(__file__).resolve().parents[2],args.run_directory.resolve()),indent=2))
