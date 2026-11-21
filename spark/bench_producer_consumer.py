"""Uninstrumented timing of explicit overlap, no-overlap control and native path."""
import json
import torch
import triton.testing
import native
from producer_consumer_mxfp8 import activation_mm,weight_reuse_mm,CONFIGS
native.set_pack_backend('triton-tiled')
torch.manual_seed(43)
for m,n,k in [(8192,3072,768),(1024,1024,4096)]:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def native_path():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    reference=native_path()
    print(json.dumps({'shape':[m,n,k],'native_ms':[triton.testing.do_bench(native_path,warmup=100,rep=300) for _ in range(3)]}),flush=True)
    for config,(bn,bk,pw) in enumerate(CONFIGS):
        for serial in (True,False):
            out=weight_reuse_mm(a,w,config=config,serial=serial)
            for actual,expected in zip(out,reference):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            times=[triton.testing.do_bench(lambda:weight_reuse_mm(a,w,config=config,serial=serial),warmup=100,rep=300) for _ in range(3)]
            print(json.dumps({'shape':[m,n,k],'config':config,'bn':bn,'bk':bk,'producer_warps':pw,'consumer_warps':4,'serial':serial,'ms':times,'exact':True}),flush=True)
