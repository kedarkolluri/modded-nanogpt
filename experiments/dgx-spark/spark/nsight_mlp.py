"""Nsight Compute probe: one warmed-up production-shape MLP forward/backward.

Complements full-training Nsight Systems captures; no optimizer or attention.
"""
import argparse
import torch
import native
from record_model import MLP, GPTConfig
from activation import FusedActivationMLP, FusedBF16ActivationMLP

p = argparse.ArgumentParser()
p.add_argument('--precision', choices=['bf16', 'mxfp8'], required=True)
args = p.parse_args()
torch.manual_seed(42)
native.set_pack_backend('triton-tiled')
model = MLP(GPTConfig(n_embd=768, n_head=6)).cuda()
for param in model.parameters():
    param.data.normal_(0, .02)
if args.precision == 'mxfp8':
    for name in ('c_fc', 'c_proj'):
        setattr(model, name, native.NativeLinear(getattr(model, name), True, 'mxfp8'))
    model = FusedActivationMLP(model)
else:
    model = FusedBF16ActivationMLP(model)
x = torch.randn(32, 256, 768, device='cuda', requires_grad=True)
g = torch.randn_like(x, dtype=torch.bfloat16)

def step():
    model.zero_grad(set_to_none=True)
    x.grad = None
    with torch.cuda.nvtx.range('mlp/forward'), torch.autocast('cuda', dtype=torch.bfloat16):
        y = model(x)
    with torch.cuda.nvtx.range('mlp/backward'):
        y.backward(g)

for _ in range(10):
    step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
step()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print({'precision': args.precision, 'shape': list(x.shape), 'finite': bool(torch.isfinite(x.grad).all())})
