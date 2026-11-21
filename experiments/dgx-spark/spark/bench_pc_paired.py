"""Fresh-seed reversed-order comparison; trace instrumentation is disabled."""
import json
import torch
import triton.testing
import native
from producer_consumer_mxfp8 import weight_reuse_mm
native.set_pack_backend('triton-tiled')
torch.manual_seed(44)
for m,n,k in [(8192,3072,768),(1024,1024,4096)]:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def native_path():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    paths={'native':native_path}
    for config in (2,9):
        for serial in (True,False):
            paths[f'config{config}-'+('serial' if serial else 'overlap')]=lambda config=config,serial=serial:weight_reuse_mm(a,w,config=config,serial=serial)
    reference=native_path()
    for path in paths.values():
        for actual,expected in zip(path(),reference):
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    for repeat in range(4):
        order=list(paths) if repeat%2==0 else list(paths)[::-1]
        for name in order:
            ms=triton.testing.do_bench(paths[name],warmup=100,rep=300)
            print(json.dumps({'shape':[m,n,k],'path':name,'repeat':repeat,'ms':ms,'exact':True}),flush=True)
