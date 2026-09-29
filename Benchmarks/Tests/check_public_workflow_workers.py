#!/usr/bin/env python3
"""Challenge actual workers only after each unchanged case has passed."""
import argparse
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Tools'))
from contracts import digest, read_json, write_json, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report={'controls':[],'ineligible':[],'status':'running'}
    for directory in args.run:
        run=read_json(directory/'run.json')
        for event in run['events']:
            name=f"{event['case_id']}-{event['engine']}-{event['batch']}"
            if event['status']!='passed':
                report['ineligible'].append(name)
                continue
            original=read_json(directory/(name+'.request.json'))
            engine=run['plan']['engines'][event['engine']]['command']
            for mutation in ('unchanged','wrong-answer','truncated-answer','malformed-input','embedded-answer'):
                target=args.output/(name+'-'+mutation);target.mkdir()
                request=copy.deepcopy(original);request.update(warmups=0,samples=1)
                if mutation in ('wrong-answer','truncated-answer'):
                    raw=Path(request['expected_path']).read_bytes()
                    if mutation=='wrong-answer':raw=struct.pack('<d',struct.unpack('<d',raw[:8])[0]+1)+raw[8:]
                    else:raw=raw[:-8]
                    path=target/'expected.f64';path.write_bytes(raw)
                    request.update(expected_path=str(path.resolve()),expected_sha256=digest(raw))
                elif mutation in ('malformed-input','embedded-answer'):
                    p=read_json(request['input_path'])
                    if mutation=='embedded-answer':p['expected_predictions']=[0]
                    elif p['operation']=='scientific-workflow':p['feature_order']=['signal','signal']
                    else:p['train_features'][0].pop()
                    path=target/'input.json';write_json(path,p);raw=path.read_bytes()
                    request.update(input_path=str(path.resolve()),input_sha256=digest(raw),input_bytes=len(raw))
                source=target/'request.json';response=target/'response.json';write_json(source,request)
                with (target/'worker.log').open('w') as log:
                    completed=subprocess.run(engine+[str(source.resolve()),str(response.resolve())],stdout=log,stderr=subprocess.STDOUT,timeout=120)
                result=read_json(response)
                positive=mutation=='unchanged'
                accepted=(completed.returncode==0 and result['status']=='passed' and len(result['samples'])==1) if positive else (
                    completed.returncode!=0 and result['status']=='failed' and result['samples']==[] and bool(result.get('error')))
                report['controls'].append({'case':name,'mutation':mutation,'passed':accepted,'response_sha256':digest(response.read_bytes())})
                write_json(args.output/'controls.json',report)
    report['status']='passed' if report['controls'] and all(x['passed'] for x in report['controls']) else 'failed'
    write_json(args.output/'controls.json',report)
    print(json.dumps({'status':report['status'],'controls':len(report['controls']),'ineligible_cases':len(report['ineligible'])}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
