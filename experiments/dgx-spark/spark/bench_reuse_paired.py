"""Fresh-seed, reversed-order confirmation after tile selection in the sweep."""
import json
from pathlib import Path
import statistics
import torch
import triton.testing
import native
from pipelined_mxfp8 import quantized_mm
from reuse_mxfp8 import weight_reuse_mm

root=Path('results/reuse-mxfp8')
rows=[]
for name in ('sweep.jsonl','wide.jsonl'):
    rows.extend(json.loads(l) for l in (root/name).read_text().splitlines())
native.set_pack_backend('triton-tiled')
torch.manual_seed(43)
for m,n,k in [(8192,3072,768),(1024,1024,4096)]:
    shape=[m,n,k]
    cases=[r for r in rows if r['shape']==shape and 'total_ms' in r]
    best=min(cases,key=lambda r:statistics.median(r['total_ms']))
    params=best['config']
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def separate():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    paths={'separate':separate,
           'both_in_loop':lambda:quantized_mm(a,w,bm=32,bn=128,bk=32,warps=4,stages=1),
           'weight_reuse':lambda:weight_reuse_mm(a,w,**params)}
    reference=separate()
    for path in paths.values():
        for actual,expected in zip(path(),reference):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    print(json.dumps({'shape':shape,'selected_config':params,'seed':43,'exact':True}),flush=True)
    for repeat in range(4):
        order=list(paths) if repeat%2==0 else list(paths)[::-1]
        for name in order:
            ms=triton.testing.do_bench(paths[name],warmup=100,rep=300)
            print(json.dumps({'shape':shape,'repeat':repeat,'path':name,'ms':ms}),flush=True)
