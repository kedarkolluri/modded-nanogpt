"""Fair timing includes fresh weight packing and both saved reconstructed inputs."""
import json
import sys
import torch
import triton.testing
import native
from reuse_mxfp8 import activation_mm,weight_reuse_mm

torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
configs=[(32,64,32,4,1),(32,128,32,4,1),(32,256,32,4,1),
         (16,128,64,4,1),(32,128,64,4,1),(32,256,64,4,1),
         (32,128,128,4,1),(32,256,128,8,1),(64,128,64,4,1),
         (32,128,64,4,2),(32,256,64,8,2)]
wide='--wide' in sys.argv
if wide:
    configs=[(16,256,32,4,1),(16,512,32,8,1),(32,512,32,8,1),(16,1024,32,8,1),(16,512,64,8,1),(32,512,64,8,1)]
shapes=[(8192,3072,768)] if wide else [(1,128,96),(129,512,256),(8192,3072,768),(1024,1024,4096)]
for m,n,k in shapes:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def reference():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    ref=reference()
    times=[triton.testing.do_bench(reference,warmup=100,rep=300) for _ in range(3)]
    print(json.dumps({'shape':[m,n,k],'reference_ms':times}),flush=True)
    wq,sw,wd=native.fp8_pack(w,False,True)
    for config in configs:
        for buffered in (False,True):
            params=dict(zip(('bm','bn','bk','warps','stages'),config),buffered=buffered)
            try:
                y,ad,kernel=activation_mm(a,wq,sw,**params,return_kernel=True)
                for x,z in zip((y,ad,wd),ref):
                    torch.testing.assert_close(x,z,rtol=0,atol=0)
                assert 'kind::mxf8f6f4.block_scale' in kernel.asm['ptx']
                assert 'tcgen05' not in kernel.asm['ptx']
                times=[triton.testing.do_bench(lambda:weight_reuse_mm(a,w,**params),warmup=100,rep=300) for _ in range(3)]
                kernel_ms=triton.testing.do_bench(lambda:activation_mm(a,wq,sw,**params),warmup=100,rep=300)
                print(json.dumps({'shape':[m,n,k],'config':params,'total_ms':times,'kernel_only_ms':kernel_ms,'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,'exact':True}),flush=True)
            except Exception as e:
                print(json.dumps({'shape':[m,n,k],'config':params,'error':str(e)}),flush=True)
