"""Check explicit worker synchronization, native MMA and reconstructed inputs."""
import json
import sys
import torch
import native
from producer_consumer_mxfp8 import activation_mm
native.set_pack_backend('triton-tiled')
torch.manual_seed(42)
for m,n,k in [(1,128,32),(17,144,96),(32,256,256),(256,512,768)]:
    a=torch.randn(m,k,device='cuda',dtype=torch.float32)
    w=torch.randn(n,k,device='cuda',dtype=torch.float32)*.03
    aq,sa,ad=native.fp8_pack(a,False,True)
    wq,sw,wd=native.fp8_pack(w,False,True)
    ref=torch._scaled_mm(aq,wq.T,sa,sw,out_dtype=torch.bfloat16)
    for config in ((2,) if '--selected' in sys.argv else (0,2,4,6,9)):
        for serial in (True,False):
            y,actual=activation_mm(a,wq,sw,config=config,serial=serial)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual,ad,rtol=0,atol=0)
            rel=float((y.float()-ref.float()).norm()/ref.float().norm())
            max_abs=float((y.float()-ref.float()).abs().max())
            torch.testing.assert_close(y,ref,rtol=0,atol=0)
            print(json.dumps({'shape':[m,n,k],'config':config,'serial':serial,'relative_l2':rel,'max_abs':max_abs,'exact':True}),flush=True)
