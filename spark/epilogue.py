"""Fuse NVFP4 outer-scale multiplication and final BF16 conversion."""
import torch
import triton
import triton.language as tl


@triton.jit
def _outer(X, A, B, Y, N: tl.constexpr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(X + i, valid, 0)
    a = tl.load(A + i // COLS, valid, 0)
    b = tl.load(B)
    value = (x * a) * b
    tl.store(Y + i, value.to(tl.bfloat16), valid)


def apply_outer(x, a, b):
    if not x.is_contiguous():
        raise ValueError('Outer-scale epilogue expects a contiguous GEMM result')
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    _outer[(triton.cdiv(x.numel(), 1024),)](x, a, b, out, x.numel(), x.shape[1], 1024,
                                          enable_fp_fusion=False)
    return out
