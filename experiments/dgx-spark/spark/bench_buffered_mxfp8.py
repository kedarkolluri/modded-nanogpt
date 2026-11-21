"""Compare serial conversion with loop-carried quantized lookahead buffers."""
import json
import sys
from pathlib import Path
import torch
import triton.testing
import native
from pipelined_mxfp8 import quantized_mm

torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
root=Path('results/buffered-mxfp8')
root.mkdir(parents=True,exist_ok=True)
small='--small-bk' in sys.argv
shapes=[(8192,3072,768),(1024,1024,4096)] if small else [(1,128,96),(129,512,256),(8192,3072,768),(1024,1024,4096)]
configs=[(16,64,32,4),(32,64,32,4),(32,128,32,4),(16,64,64,4),(32,64,64,4),(32,128,64,8)] if small else [(16,64,128,4),(32,64,128,4),(32,128,128,8)]
for m,n,k in shapes:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    def reference():
        aq,sa,ad=native.fp8_pack(a,False,True)
        wq,sw,wd=native.fp8_pack(w,False,True)
        return torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16),ad,wd
    ref=reference()
    print(json.dumps({'shape':[m,n,k],'reference_ms':triton.testing.do_bench(reference,warmup=100,rep=300)}),flush=True)
    for bm,bn,bk,warps in configs:
        for buffered,stages,loop_stages in [(False,1,1),(False,2,1),(True,1,1),(True,1,2),(True,2,2)]:
            config=dict(bm=bm,bn=bn,bk=bk,warps=warps,stages=stages,buffered=buffered,loop_stages=loop_stages)
            name=f'{m}-{n}-{k}-'+ '-'.join(str(v) for v in config.values())
            try:
                out=quantized_mm(a,w,**config,return_kernel=True)
                for x,y in zip(out[:3],ref):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
                kernel=out[3]
                if m==8192:
                    for ext in ('ptx','ttgir','llir','cubin'):
                        value=kernel.asm[ext]
                        path=root/f'{name}.{ext}'
                        path.write_bytes(value) if isinstance(value,bytes) else path.write_text(value)
                times=[triton.testing.do_bench(lambda:quantized_mm(a,w,**config),warmup=100,rep=300) for _ in range(3)]
                print(json.dumps({'shape':[m,n,k],'config':config,'ms':times,'exact':True,'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,'cp_async':kernel.asm['ptx'].count('cp.async')}),flush=True)
            except Exception as e:
                print(json.dumps({'shape':[m,n,k],'config':config,'error':str(e)}),flush=True)
