"""ReLU-square fused into MXFP8 projection-input packing.

The saved preactivation supports the exact BF16 activation derivative.
Backward pointwise arithmetic is fused, but gradient MXFP8 packing remains
separate. cuBLAS still performs all three projection GEMMs.
"""
import torch
from torch import nn
import triton
import triton.language as tl
import native
from triton_pack import fp8_pack


def pack_activation(z):
    return fp8_pack(z, False, return_dequant=True, relu_square=True)


@triton.jit
def _relu_square_grad(Z, G, D, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    z = tl.load(Z+i, valid, 0).to(tl.float32)
    g = tl.load(G+i, valid, 0).to(tl.float32)
    positive = tl.maximum(z, 0.)
    factor = (2. * positive).to(tl.bfloat16).to(tl.float32)
    result = (g * factor).to(tl.bfloat16)
    tl.store(D+i, tl.where(z <= 0., 0., result), valid)


def activation_backward(z, grad):
    out = torch.empty_like(z)
    _relu_square_grad[(triton.cdiv(z.numel(), 1024),)](
        z, grad, out, z.numel(), 1024, enable_fp_fusion=False)
    return out


class ReluSquareMXFP8Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, weight, stochastic):
        shape = z.shape
        flat = z.reshape(-1, shape[-1])
        aq, sa, ax = pack_activation(flat)
        bq, sb, bw = native.fp8_pack(weight, False, return_dequant=True)
        out = torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.bfloat16)
        ctx.save_for_backward(z, ax, bw)
        ctx.stochastic = stochastic
        return out.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        z, x, w = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        dh = native.mx_mm(g, w.T, ctx.stochastic).reshape_as(z)
        dw = native.mx_mm(g.T, x.T, ctx.stochastic)
        dz = activation_backward(z, dh)
        return dz, dw, None


class ReluSquareProjection(nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.weight = projection.weight
        self.stochastic = projection.stochastic
        self.forward_format = 'mxfp8'

    def forward(self, z):
        return ReluSquareMXFP8Fn.apply(z, self.weight, self.stochastic)


class FusedActivationMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.c_fc = mlp.c_fc
        self.c_proj = ReluSquareProjection(mlp.c_proj)

    def forward(self, x):
        return self.c_proj(self.c_fc(x))


def replace_activation_mlps(model):
    count = 0
    for block in model.transformer.h:
        if getattr(block.mlp.c_fc, 'forward_format', None) == 'mxfp8':
            if block.mlp.c_proj.forward_format != 'mxfp8':
                raise ValueError('Both MLP projections must use MXFP8')
            block.mlp = FusedActivationMLP(block.mlp)
            count += 1
    return count
