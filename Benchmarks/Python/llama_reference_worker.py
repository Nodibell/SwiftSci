"""Opt-in real-checkpoint smoke evidence; not a numerical conformance oracle."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not mx.metal.is_available(): raise RuntimeError('Metal required; no CPU fallback')
    pack=json.loads(args.prompts.read_text());cases=pack['cases']
    report=dict(schema_version=1,status='running',purpose='pretrained-reference-smoke',
                device='gpu',max_tokens=32,temperature=0,template_date=pack['template_date'],
                versions={name:importlib.metadata.version(name) for name in ['mlx','mlx-metal','mlx-lm','transformers','tokenizers']},cases=[])
    try:
        with mx.stream(mx.new_stream(mx.gpu)):
            start=time.perf_counter_ns()
            model,tokenizer=load(str(args.model),tokenizer_config={'trust_remote_code':False})
            model.eval();mx.eval(model.parameters());mx.synchronize()
            report['load_elapsed_ns']=time.perf_counter_ns()-start
            for case in cases:
                rendered=tokenizer.apply_chat_template([dict(role='user',content=case['text'])],tokenize=False,add_generation_prompt=True,date_string=pack['template_date'])
                ids=tokenizer.encode(rendered,add_special_tokens=False)
                row=dict(id=case['id'],text=case['text'],raw_tokens=tokenizer.encode(case['text'],add_special_tokens=False),
                         rendered_prompt=rendered,prompt_tokens=ids,samples=[])
                for sample in range(3):
                    start=time.perf_counter_ns();first=None;tokens=[];pieces=[];last=None
                    cache=make_prompt_cache(model)
                    for response in stream_generate(model,tokenizer,ids,max_tokens=32,sampler=make_sampler(temp=0),prompt_cache=cache):
                        if first is None:
                            mx.synchronize();first=time.perf_counter_ns()-start
                        tokens.append(response.token);pieces.append(response.text);last=response
                    mx.synchronize();elapsed=time.perf_counter_ns()-start
                    if not tokens:raise RuntimeError('No generated tokens for '+case['id'])
                    row['samples'].append(dict(warmup=sample==0,elapsed_ns=elapsed,time_to_first_token_ns=first,
                        tokens=tokens,text=''.join(pieces),finish_reason=last.finish_reason,
                        prompt_tps=last.prompt_tps,generation_tps=last.generation_tps,peak_memory_gb=last.peak_memory,
                        cache_dtypes=sorted({str(t.dtype) for layer in cache for t in layer.state})))
                row['greedy_repeat_equal']=all(x['tokens']==row['samples'][0]['tokens'] for x in row['samples'])
                if not row['greedy_repeat_equal']:raise RuntimeError('Greedy replay differs for '+case['id'])
                report['cases'].append(row)
            report['status']='passed'
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}')
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
