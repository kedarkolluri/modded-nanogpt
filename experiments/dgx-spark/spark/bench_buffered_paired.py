"""Reversed-order confirmation with identical tile sizes in both fused paths."""
import json
import torch
import triton.testing
import native
from pipelined_mxfp8 import quantized_mm
native.set_pack_backend('triton-tiled')
torch.manual_seed(42)
for m,n,k in [(8192,3072,768),(1024,1024,4096)]:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def separate():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    paths={'separate':separate}
    for buffered in (False,True):
        paths['buffered' if buffered else 'serial']=lambda buffered=buffered:quantized_mm(a,w,bm=32,bn=128,bk=32,warps=4,stages=1,buffered=buffered)
    ref=separate()
    for path in paths.values():
        for actual,expected in zip(path(),ref):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for repeat in range(4):
        order=list(paths) if repeat%2==0 else list(paths)[::-1]
        for name in order:
            ms=triton.testing.do_bench(paths[name],warmup=100,rep=300)
            print(json.dumps({'shape':[m,n,k],'repeat':repeat,'path':name,'ms':ms}),flush=True)
