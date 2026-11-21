"""Fused GB10 quantization/packing kernels; GEMMs remain native cuBLAS.

Scale layout is cuBLAS SWIZZLE_32_4_4. FP4 forward rounding uses the eager
backend's nearest-with-midpoint-toward-zero rule. Random FP8 rounding uses
Triton's counter-based RNG with seeds drawn from PyTorch's CUDA RNG.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_offset(row, col, NC: tl.constexpr):
    return ((row // 128) * (NC // 4) + col // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4


@triton.jit
def _absmax_part(X, P, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    v = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(P + tl.program_id(0), tl.max(tl.abs(v), 0))


@triton.jit
def _absmax_finish(P, M, N: tl.constexpr, B: tl.constexpr):
    i = tl.arange(0, B)
    v = tl.load(P + i, i < N, 0)
    tl.store(M, tl.max(v, 0))


@triton.jit
def _fp4(X, Q, S, O, D, Maximum, ROWS: tl.constexpr, COLS: tl.constexpr,
         NC: tl.constexpr, B: tl.constexpr, PER_TOKEN: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * COLS + col, (row < ROWS) & (col < COLS), 0).to(tl.float32)
    if PER_TOKEN:
        maximum = tl.max(tl.abs(x), 0)
    else:
        maximum = tl.load(Maximum)
    outer = tl.maximum(maximum * (1. / 2688.), 1.e-20)
    if PER_TOKEN:
        tl.store(O + row, outer, row < ROWS)
    else:
        if row == 0:
            tl.store(O, outer)
    blocks = tl.reshape(x, (B // 16, 16))
    block_max = tl.max(tl.abs(blocks), 1)
    scale = tl.div_rn(block_max, 6. * outer).to(tl.float8e4nv).to(tl.float32)
    scale = tl.maximum(scale, 0.001953125)
    v = tl.div_rn(blocks, scale[:, None] * outer)
    mag = tl.abs(v)
    # FP4 code order: 0, .5, 1, 1.5, 2, 3, 4, 6.
    code = ((mag > .25).to(tl.int32) + (mag > .75).to(tl.int32)
            + (mag > 1.25).to(tl.int32) + (mag > 1.75).to(tl.int32)
            + (mag > 2.5).to(tl.int32) + (mag > 3.5).to(tl.int32)
            + (mag > 5.).to(tl.int32))
    value = tl.where(code <= 4, code.to(tl.float32) * .5,
                     tl.where(code == 5, 3., tl.where(code == 6, 4., 6.)))
    sign = (v.to(tl.int32, bitcast=True) >> 31) & 1
    dequant = tl.where(sign != 0, -value, value) * scale[:, None] * outer
    tl.store(D + row * COLS + col, tl.reshape(dequant, (B,)), (row < ROWS) & (col < COLS))
    signed_code = tl.reshape(code | (sign << 3), (B // 2, 2))
    shifts = tl.arange(0, 2) * 4
    packed = tl.sum(signed_code << shifts[None, :], 1)
    pair = tl.arange(0, B // 2)
    tl.store(Q + row * (COLS // 2) + pair, packed.to(tl.uint8), (row < ROWS) & (pair < COLS // 2))
    block = tl.arange(0, B // 16)
    scale_byte = tl.where((row < ROWS) & (block < COLS // 16), scale, 0.).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(S + _scale_offset(row, block, NC), scale_byte, block < NC)


@triton.jit
def _fp8(X, Q, S, Seed, D, ROWS: tl.constexpr, COLS: tl.constexpr,
         STRIDE_R: tl.constexpr, STRIDE_C: tl.constexpr, NC: tl.constexpr,
         B: tl.constexpr, STOCHASTIC: tl.constexpr, SAVE_DEQUANT: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * STRIDE_R + col * STRIDE_C,
                (row < ROWS) & (col < COLS), 0).to(tl.float32)
    blocks = tl.reshape(x, (B // 32, 32))
    maximum = tl.max(tl.abs(blocks), 1)
    exponent = tl.ceil(tl.log2(tl.maximum(tl.div_rn(maximum, 448.), 1.1754943508222875e-38)))
    scale = tl.exp2(exponent)
    v = blocks / scale[:, None]
    if STOCHASTIC:
        magnitude = tl.abs(v)
        spacing = tl.exp2(tl.floor(tl.log2(tl.maximum(magnitude, .015625))) - 3.)
        lower = tl.floor(magnitude / spacing) * spacing
        probability = (magnitude - lower) / spacing
        seed = tl.load(Seed).to(tl.uint32)
        random = tl.reshape(tl.rand(seed, row * COLS + col), (B // 32, 32))
        rounded = tl.minimum(lower + (random < probability) * spacing, 448.)
        sign = (v.to(tl.int32, bitcast=True) >> 31) & 1
        v = tl.where(sign != 0, -rounded, rounded)
    quantized = v.to(tl.float8e4nv)
    tl.store(Q + row * COLS + col, tl.reshape(quantized, (B,)),
             (row < ROWS) & (col < COLS))
    if SAVE_DEQUANT:
        dequant = quantized.to(tl.float32) * scale[:, None]
        tl.store(D + row * COLS + col, tl.reshape(dequant, (B,)),
                 (row < ROWS) & (col < COLS))
    block = tl.arange(0, B // 32)
    encoded = tl.where((row < ROWS) & (block < COLS // 32), exponent + 127., 0.).to(tl.uint8)
    tl.store(S + _scale_offset(row, block, NC), encoded, block < NC)


def fp4_pack(x, per_token=False):
    x = x.detach().contiguous()
    rows, cols = x.shape
    if cols % 16:
        raise ValueError('FP4 contraction dimension must be divisible by 16')
    outer = torch.empty((rows, 1) if per_token else (), device=x.device, dtype=torch.float32)
    maximum = outer  # Unused by the per-token specialization.
    if not per_token:
        parts = triton.cdiv(x.numel(), 4096)
        partial = torch.empty(parts, device=x.device, dtype=torch.float32)
        maximum = torch.empty((), device=x.device, dtype=torch.float32)
        _absmax_part[(parts,)](x, partial, x.numel(), 4096)
        _absmax_finish[(1,)](partial, maximum, parts, triton.next_power_of_2(parts))
    nc = triton.cdiv(cols // 16, 4) * 4
    padded_rows = triton.cdiv(rows, 128) * 128
    packed = torch.empty((rows, cols//2), device=x.device, dtype=torch.uint8)
    scales = torch.empty(padded_rows * nc, device=x.device, dtype=torch.uint8)
    dequant = torch.empty_like(x, dtype=torch.bfloat16)
    _fp4[(padded_rows,)](x, packed, scales, outer, dequant, maximum,
        rows, cols, nc, max(64, triton.next_power_of_2(cols)), per_token, num_warps=4)
    return packed.view(torch.float4_e2m1fn_x2), scales.view(torch.float8_e4m3fn), outer, dequant


def fp8_pack(x, stochastic=True, return_dequant=False):
    x = x.detach()
    rows, cols = x.shape
    if cols % 32:
        raise ValueError('MXFP8 contraction dimension must be divisible by 32')
    nc = triton.cdiv(cols // 32, 4) * 4
    padded_rows = triton.cdiv(rows, 128) * 128
    packed = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded_rows * nc, device=x.device, dtype=torch.uint8)
    seed = torch.randint(0, 2**31-1, (), device=x.device, dtype=torch.int64) if stochastic else scales
    dequant = torch.empty((rows, cols), device=x.device, dtype=torch.bfloat16) if return_dequant else packed
    _fp8[(padded_rows,)](x, packed, scales, seed, dequant, rows, cols, *x.stride(), nc,
        max(128, triton.next_power_of_2(cols)), stochastic, return_dequant, num_warps=4)
    result = (packed, scales.view(torch.float8_e8m0fnu))
    return (*result, dequant) if return_dequant else result
