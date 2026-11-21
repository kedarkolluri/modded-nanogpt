"""Numerical reference only: dequantized BF16 GEMMs, NOT native FP4/FP8 speed.

Implements 1x16 NVFP4-like forward quantization and transpose/requantize
with stochastic MXFP8 rounding. Does not yet implement Four Over Six.
"""
import torch
from torch import nn


def nvfp4_reference(x, per_token=False):
    shape = x.shape
    z = x.detach().float().reshape(-1, 16)
    # One FP32 outer scale per input tensor (per expert for a dense weight).
    outer = (z.abs().amax() / (448 * 6)).clamp_min(1e-20)
    if per_token:
        row_max = x.detach().float().abs().amax(-1, keepdim=True).reshape(-1, 1)
        outer = (row_max / (448 * 6)).clamp_min(1e-20).repeat_interleave(x.shape[-1] // 16, dim=0)
    scales = (z.abs().amax(-1, keepdim=True) / (6 * outer)).to(
        torch.float8_e4m3fn).float().clamp_min(2**-9)
    scale = scales * outer
    v = z / scale
    levels = torch.tensor([-6, -4, -3, -2, -1.5, -1, -.5, 0,
                           .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    # Midpoint search avoids allocating a values x levels distance tensor.
    indices = torch.bucketize(v.contiguous(), (levels[1:] + levels[:-1]) / 2)
    return (levels[indices] * scale).reshape(shape).to(x.dtype)


def mxfp8_reference(x, stochastic=True):
    shape = x.shape
    z = x.detach().float().contiguous().reshape(-1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(
        (z.abs().amax(-1, keepdim=True) / 448).clamp_min(2**-126))))
    v = z / scale
    if stochastic:
        magnitude = v.abs()
        spacing = torch.exp2((torch.floor(torch.log2(magnitude.clamp_min(2**-6))) - 3))
        lower = torch.floor(magnitude / spacing) * spacing
        probability = (magnitude - lower) / spacing
        rounded = lower + (torch.rand_like(probability) < probability) * spacing
        q = rounded.clamp_max(448).copysign(v)
    else:
        q = v.to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape).to(x.dtype)


class ReferenceLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, stochastic):
        xq = nvfp4_reference(x, per_token=True).to(torch.bfloat16)
        wq = nvfp4_reference(weight).to(torch.bfloat16)
        ctx.save_for_backward(xq, wq)
        ctx.stochastic = stochastic
        return xq @ wq.T

    @staticmethod
    def backward(ctx, grad):
        xq, wq = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        x = xq.reshape(-1, xq.shape[-1])
        # Requantize the actual forward values after regrouping for transposes.
        wt = mxfp8_reference(wq.T.contiguous(), ctx.stochastic)
        xt = mxfp8_reference(x.T.contiguous(), ctx.stochastic)
        gq = mxfp8_reference(g, ctx.stochastic)
        gt = mxfp8_reference(g.T.contiguous(), ctx.stochastic)
        dx = (gq @ wt.T).reshape_as(xq)
        dw = gt @ xt.T
        return dx, dw, None


class ReferenceLinear(nn.Module):
    def __init__(self, linear, stochastic=True):
        super().__init__()
        self.weight = linear.weight
        self.stochastic = stochastic

    def forward(self, x):
        return ReferenceLinearFn.apply(x, self.weight, self.stochastic)


def replace_mlps(model, stochastic=True):
    # Leave the final ceil(N/6) layers in BF16, mirroring the post's protection.
    count = len(model.transformer.h)
    for block in model.transformer.h[:count - (count + 5)//6]:
        for name in ('c_fc', 'c_proj'):
            setattr(block.mlp, name, ReferenceLinear(getattr(block.mlp, name), stochastic))
