#!/usr/bin/env python3
"""Opt-in pinned Llama integration evidence; never emits a conformance certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'Benchmarks/Fixtures/pretrained/llama-3.2-1b-instruct-bf16.json'
PROMPTS=ROOT/'Benchmarks/Fixtures/pretrained/prompts.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify_checkpoint(directory, manifest):
    for artifact in manifest['artifacts']:
        relative=Path(artifact['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid checkpoint artifact path')
        path=directory/relative
        if not path.is_file() or path.stat().st_size!=artifact['bytes'] or sha(path)!=artifact['sha256']:
            raise ValueError('Missing or changed checkpoint artifact: '+artifact['path'])

    expected = {artifact['path'] for artifact in manifest['artifacts']}
    observed = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                if p.is_file() and p.relative_to(directory).parts[0] != '.cache'}
    observed.update(p.relative_to(directory).as_posix() for p in directory.glob('model*.safetensors'))
    if observed != expected:
        raise ValueError('Checkpoint input inventory differs from the pinned manifest')


def main():
    from runner import source_identity
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True,help='Pinned requirements-llm.txt environment')
    p.add_argument('--swift-worker',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model,python,worker,output=[x.resolve() for x in (args.model,args.python,args.swift_worker,args.output)]
    # Preserve the venv executable path; resolving its symlink would bypass the environment.
    python=args.python.absolute()
    source=source_identity(ROOT);manifest=json.loads(MANIFEST.read_text())
    verify_checkpoint(model,manifest)
    build=json.loads(Path(str(worker)+'.build.json').read_text())
    if build['source']['tree_sha256']!=source['tree_sha256'] or build['binary_sha256']!=sha(worker):
        raise ValueError('Swift worker source/binary differs; rebuild first')
    output.mkdir(parents=True,exist_ok=False)
    record=dict(schema_version=1,purpose='pretrained-integration-diagnostic',status='running',
                source=source,checkpoint=manifest,manifest_sha256=sha(MANIFEST),prompts_sha256=sha(PROMPTS),
                swift_binary_sha256=sha(worker),generation_comparison='not-executed')
    try:
        subprocess.run([str(python),'-m','pip','freeze'],check=True,stdout=(output/'python-environment.txt').open('w'))
        with (output/'reference.log').open('w') as log:
            ref=subprocess.run([str(python),str(ROOT/'Benchmarks/Python/llama_reference_worker.py'),
                '--model',str(model),'--prompts',str(PROMPTS),'--output',str(output/'reference.json')],stdout=log,stderr=subprocess.STDOUT)
        if ref.returncode:raise RuntimeError('Reference smoke failed; inspect reference.log')
        reference=json.loads((output/'reference.json').read_text())
        request=dict(model_path=str(model),cases=reference['cases'])
        (output/'swift-request.json').write_text(json.dumps(request,indent=2)+'\n')
        with (output/'swift.log').open('w') as log:
            swift=subprocess.run([str(worker),'--llama-inspect',str(output/'swift-request.json'),str(output/'swift.json')],stdout=log,stderr=subprocess.STDOUT)
        inspection=json.loads((output/'swift.json').read_text())
        if swift.returncode not in (0,1) or inspection['status'] not in ('blocked','preflight-passed'):
            raise RuntimeError('Swift inspection failed; inspect swift.json and swift.log')
        verify_checkpoint(model,manifest)
        if sha(MANIFEST)!=record['manifest_sha256'] or sha(PROMPTS)!=record['prompts_sha256']:
            raise RuntimeError('Checkpoint manifest or prompts changed during inspection')
        if source_identity(ROOT)!=source:raise RuntimeError('Source changed during inspection')
        record.update(status=inspection['status'],reference_status=reference['status'],findings=inspection['findings'])
    except Exception as e:
        record.update(status='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        (output/'integration.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['status','reference_status','generation_comparison','findings']},indent=2))
    return 1 if record['status']=='blocked' else 0


if __name__=='__main__':raise SystemExit(main())
