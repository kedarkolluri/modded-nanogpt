"""Single production-shape fused epilogue kernel for Nsight Compute counters."""
import torch
import native
from fused_mlp_epilogue import gemm_pack

torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
a,sa,_=native.fp8_pack(torch.randn(8192,768,device='cuda',dtype=torch.bfloat16),False,True)
b,sb,_=native.fp8_pack(torch.randn(3072,768,device='cuda',dtype=torch.bfloat16)*.03,False,True)
for _ in range(10):
    gemm_pack(a,sa,b,sb)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
z,h,hs,hd=gemm_pack(a,sa,b,sb)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print({'finite':bool(torch.isfinite(z).all() and torch.isfinite(hd).all())})
