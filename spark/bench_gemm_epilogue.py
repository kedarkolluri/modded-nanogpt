"""Production-shape fused-epilogue tile sweep; no optimizer or backward timing."""
import json
import torch
import triton
import triton.testing
import native
from activation import pack_activation
from fused_mlp_epilogue import gemm_pack

torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
a,sa,_ = native.fp8_pack(torch.randn(8192,768,device='cuda',dtype=torch.bfloat16),False,True)
w,sw,_ = native.fp8_pack(torch.randn(3072,768,device='cuda',dtype=torch.bfloat16)*.03,False,True)

def separate():
    z = torch._scaled_mm(a,w.T,sa,sw,out_dtype=torch.bfloat16)
    h,hs,hd = pack_activation(z)
    return z,h,hs,hd

reference = separate()
baseline = triton.testing.do_bench(separate,warmup=100,rep=300)
print(json.dumps({'baseline_gemm_activation_pack_ms':baseline,'shape':[8192,3072,768]}),flush=True)
configs = [(16,64,128,4,2),(16,128,128,4,2),(32,64,128,4,2),
           (32,128,128,4,2),(64,64,128,4,2),(64,128,128,4,2),
           (32,128,256,4,2),(32,128,128,4,3),(32,256,128,8,2),(64,128,128,8,2)]
for config in configs:
    try:
        out = gemm_pack(a,sa,w,sw,*config)
        rel = float((out[0].float()-reference[0].float()).norm()/reference[0].float().norm())
        assert rel < .003,rel
        ms = triton.testing.do_bench(lambda:gemm_pack(a,sa,w,sw,*config),warmup=100,rep=300)
        print(json.dumps({'config':config,'ms':ms,'speedup':baseline/ms,'z_relative_l2':rel}),flush=True)
    except Exception as e:
        print(json.dumps({'config':config,'error':str(e)}),flush=True)
