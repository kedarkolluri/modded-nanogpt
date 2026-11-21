"""MXFP8 packing with coalesced 2D tiles for transpose inputs.

Contiguous operands retain the row kernel. Transposes use 16x128 tiles to
read neighboring source rows together instead of issuing strided row loads.
"""
import torch
import triton
import triton.language as tl
from triton_pack import fp4_pack, fp8_pack as row_fp8_pack, _scale_offset


@triton.jit
def _fp8_transpose(X, Q, S, Seed, ROWS: tl.constexpr, COLS: tl.constexpr,
                   SR: tl.constexpr, SC: tl.constexpr, NC: tl.constexpr,
                   STOCHASTIC: tl.constexpr, R: tl.constexpr = 16, C: tl.constexpr = 128):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.program_id(1) * C + tl.arange(0, C)
    x = tl.load(X + row[:, None] * SR + col[None, :] * SC,
                (row[:, None] < ROWS) & (col[None, :] < COLS), 0).to(tl.float32)
    blocks = tl.reshape(x, (R, C // 32, 32))
    maximum = tl.max(tl.abs(blocks), 2)
    exponent = tl.ceil(tl.log2(tl.maximum(tl.div_rn(maximum, 448.), 1.1754943508222875e-38)))
    scale = tl.exp2(exponent)
    v = blocks / scale[:, :, None]
    if STOCHASTIC:
        magnitude = tl.abs(v)
        spacing = tl.exp2(tl.floor(tl.log2(tl.maximum(magnitude, .015625))) - 3.)
        lower = tl.floor(magnitude / spacing) * spacing
        probability = (magnitude - lower) / spacing
        seed = tl.load(Seed).to(tl.uint32)
        offsets = row[:, None] * COLS + col[None, :]
        random = tl.reshape(tl.rand(seed, offsets), (R, C // 32, 32))
        rounded = tl.minimum(lower + (random < probability) * spacing, 448.)
        sign = (v.to(tl.int32, bitcast=True) >> 31) & 1
        v = tl.where(sign != 0, -rounded, rounded)
    tl.store(Q + row[:, None] * COLS + col[None, :], tl.reshape(v, (R, C)).to(tl.float8e4nv),
             (row[:, None] < ROWS) & (col[None, :] < COLS))
    block = tl.program_id(1) * (C // 32) + tl.arange(0, C // 32)
    encoded = tl.where((row[:, None] < ROWS) & (block[None, :] < COLS // 32), exponent + 127., 0.).to(tl.uint8)
    tl.store(S + _scale_offset(row[:, None], block[None, :], NC), encoded, block[None, :] < NC)


def fp8_pack(x, stochastic=True):
    if x.stride(1) == 1:
        return row_fp8_pack(x, stochastic)
    x = x.detach()
    rows, cols = x.shape
    if cols % 32:
        raise ValueError('MXFP8 contraction dimension must be divisible by 32')
    nc = triton.cdiv(cols // 32, 4) * 4
    padded_rows = triton.cdiv(rows, 128) * 128
    packed = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded_rows * nc, device=x.device, dtype=torch.uint8)
    seed = torch.randint(0, 2**31-1, (), device=x.device, dtype=torch.int64) if stochastic else scales
    _fp8_transpose[(triton.cdiv(padded_rows, 16), triton.cdiv(cols, 128))](
        x, packed, scales, seed, rows, cols, *x.stride(), nc, stochastic, num_warps=4)
    return packed, scales.view(torch.float8_e8m0fnu)
