"""Experimental native NVFP4 forward / MXFP8 backward for GB10.

Packing can use the eager reference or fused Triton backend.
The GEMMs use cuBLAS; there is no emulation fallback.
"""
import torch
from torch import nn


def swizzle_scale(scales):
    # cuBLAS 128-row by 4-column tiles, interleaved as [32, 4, 4].
    rows, cols = scales.shape
    r, c = (rows+127)//128, (cols+3)//4
    padded = torch.zeros((r*128, c*4), device=scales.device, dtype=torch.uint8)
    padded[:rows, :cols] = scales.view(torch.uint8)
    tiles = padded.reshape(r, 128, c, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).contiguous().flatten().view(scales.dtype)


def fp4_pack(x, per_token=False):
    x = x.detach().float().contiguous()
    rows, cols = x.shape
    blocks = x.reshape(rows, cols//16, 16)
    maximum = x.abs().amax(-1, keepdim=True) if per_token else x.abs().amax()
    outer = (maximum / 2688).clamp_min(1e-20)
    outer_blocks = outer.reshape(rows, 1, 1) if per_token else outer
    scales = (blocks.abs().amax(-1, keepdim=True) / (6*outer_blocks)).to(
        torch.float8_e4m3fn)
    # Preserve zero/subnormal blocks with the smallest positive FP8 scale.
    scales = scales.float().clamp_min(2**-9).to(torch.float8_e4m3fn)
    v = blocks / (scales.float() * outer_blocks)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    indices = torch.bucketize(v.abs().contiguous(), (levels[1:]+levels[:-1])/2)
    codes = (indices | (v.signbit().long() << 3)).reshape(rows, cols).to(torch.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)
    dequant = (levels[indices].copysign(v) * scales.float() * outer_blocks).reshape_as(x)
    return packed, swizzle_scale(scales.reshape(rows, cols//16)), outer, dequant.to(torch.bfloat16)


def fp8_pack(x, stochastic=True, return_dequant=False):
    x = x.detach().float().contiguous()
    rows, cols = x.shape
    blocks = x.reshape(rows, cols//32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(
        (blocks.abs().amax(-1, keepdim=True)/448).clamp_min(2**-126))))
    v = blocks/scale
    if stochastic:
        mag = v.abs()
        spacing = torch.exp2(torch.floor(torch.log2(mag.clamp_min(2**-6))) - 3)
        lower = torch.floor(mag/spacing)*spacing
        probability = (mag-lower)/spacing
        v = (lower + (torch.rand_like(v) < probability)*spacing).clamp_max(448).copysign(v)
    q = v.to(torch.float8_e4m3fn).reshape(rows, cols)
    scales = scale.reshape(rows, cols//32).to(torch.float8_e8m0fnu)
    result = (q, swizzle_scale(scales))
    if return_dequant:
        dequant = (q.float().reshape(rows, cols//32, 32) * scale).reshape_as(x).bfloat16()
        return *result, dequant
    return result


def apply_outer(out, ao, bo):
    return (out * ao * bo).to(torch.bfloat16)


def mx_mm(a, bt, stochastic):
    aq, sa = fp8_pack(a, stochastic)
    bq, sb = fp8_pack(bt, stochastic)
    return torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.bfloat16)


class NativeLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, stochastic):
        shape = x.shape
        aq, sa, ao, ax = fp4_pack(x.reshape(-1, shape[-1]), per_token=True)
        bq, sb, bo, bw = fp4_pack(weight)
        # Apply per-token and per-tensor outer scales before casting the output.
        out = torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.float32)
        out = apply_outer(out, ao, bo)
        ctx.save_for_backward(ax, bw)
        ctx.shape, ctx.stochastic = shape, stochastic
        return out.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        g = grad.reshape(-1, grad.shape[-1])
        dx = mx_mm(g, w.T, ctx.stochastic)
        dw = mx_mm(g.T, x.T, ctx.stochastic)
        return dx.reshape(ctx.shape), dw, None


class MXFP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, stochastic):
        shape = x.shape
        aq, sa, ax = fp8_pack(x.reshape(-1, shape[-1]), False, return_dequant=True)
        bq, sb, bw = fp8_pack(weight, False, return_dequant=True)
        out = torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.bfloat16)
        ctx.save_for_backward(ax, bw)
        ctx.shape, ctx.stochastic = shape, stochastic
        return out.reshape(*shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad):
        # Same estimator as the FP4 path: requantize actual forward values.
        return NativeLinearFn.backward(ctx, grad)


class NativeLinear(nn.Module):
    def __init__(self, linear, stochastic=True, forward_format='nvfp4'):
        super().__init__()
        self.weight = linear.weight
        self.stochastic = stochastic
        self.forward_format = forward_format

    def forward(self, x):
        fn = MXFP8LinearFn if self.forward_format == 'mxfp8' else NativeLinearFn
        return fn.apply(x, self.weight, self.stochastic)


def replace_mlps(model, stochastic=True, forward_format='nvfp4', tail_precision='bf16'):
    count = len(model.transformer.h)
    first_tail = count - (count+5)//6
    for index, block in enumerate(model.transformer.h):
        if index >= first_tail and tail_precision == 'bf16':
            continue
        fmt = 'mxfp8' if index >= first_tail else forward_format
        for name in ('c_fc', 'c_proj'):
            setattr(block.mlp, name, NativeLinear(getattr(block.mlp, name), stochastic, fmt))


eager_fp4_pack = fp4_pack
eager_fp8_pack = fp8_pack

def set_pack_backend(backend):
    global fp4_pack, fp8_pack
    if backend == 'triton':
        from triton_pack import fp4_pack, fp8_pack
    elif backend == 'triton-tiled':
        from triton_pack_tiled import fp4_pack, fp8_pack
    elif backend == 'eager':
        fp4_pack, fp8_pack = eager_fp4_pack, eager_fp8_pack
    else:
        raise ValueError(backend)


eager_apply_outer = apply_outer

def set_epilogue_backend(backend):
    global apply_outer
    if backend == 'triton':
        from epilogue import apply_outer
    elif backend == 'eager':
        apply_outer = eager_apply_outer
    else:
        raise ValueError(backend)
