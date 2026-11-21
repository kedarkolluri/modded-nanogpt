import json
import torch
import triton.testing
import native
from pipelined_mxfp8 import quantized_mm

torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
for m,n,k in [(1,128,96),(129,512,256),(8192,3072,768)]:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def reference():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    ref=reference()
    baseline=triton.testing.do_bench(reference,warmup=100,rep=300)
    print(json.dumps({'shape':[m,n,k],'baseline_ms':baseline}),flush=True)
    for config in [(16,64,128,4,2),(32,64,128,4,2),(32,128,128,4,1),(32,128,128,4,2),(32,128,128,4,3),(32,128,256,4,2),(32,128,128,8,2),(32,128,128,8,1),(16,128,128,4,2),(16,128,128,8,2),(64,128,128,4,2)]:
        try:
            out=quantized_mm(a,w,*config,return_kernel=True)
            for actual,expected in zip(out[:3],ref):
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            ms=triton.testing.do_bench(lambda:quantized_mm(a,w,*config),warmup=100,rep=300)
            ptx=out[3].asm['ptx']
            assert 'mma.sync' in ptx and 'tcgen05' not in ptx
            print(json.dumps({'config':config,'ms':ms,'speedup':baseline/ms,'registers':out[3].n_regs,'spills':out[3].n_spills,'cp_async':ptx.count('cp.async'),'exact':True}),flush=True)
        except Exception as e:
            print(json.dumps({'config':config,'error':str(e)}),flush=True)
