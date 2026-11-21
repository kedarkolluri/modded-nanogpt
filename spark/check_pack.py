"""Validate fused packing, padded layouts, strided inputs and stochastic RNG."""
import argparse
import json
import torch
import native
import triton_pack as fused


def decoded_mx(q, swizzled):
    # Independent inverse of cuBLAS's layout, including padded rows/columns.
    rows, cols = q.shape
    nr, nc = (rows+127)//128, ((cols//32)+3)//4
    data = swizzled.view(torch.uint8).reshape(nr*nc, 32, 4, 4).transpose(1, 2)
    regular = data.reshape(nr, nc, 128, 4).permute(0, 2, 1, 3).reshape(nr*128, nc*4)
    scale = regular[:rows, :cols//32].contiguous().view(torch.float8_e8m0fnu).float()
    return (q.float().reshape(rows, cols//32, 32)*scale[:, :, None]).reshape_as(q)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tiled', action='store_true')
    args = p.parse_args()
    backend = fused
    if args.tiled:
        import triton_pack_tiled as backend
    torch.manual_seed(1234)
    cases = 0
    for rows, cols in [(1, 32), (17, 96), (129, 256), (384, 768), (256, 3072)]:
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(rows, cols, device='cuda', dtype=dtype)
            x[0] = 0
            if rows > 1:
                x[1, 0] = 1000
            for per_token in (False, True):
                eq, es, eo, ed = native.eager_fp4_pack(x, per_token)
                fq, fs, fo, fd = backend.fp4_pack(x, per_token)
                assert torch.equal(eq.view(torch.uint8), fq.view(torch.uint8)), (rows, cols, dtype, per_token, 'codes')
                assert torch.equal(es.view(torch.uint8), fs.view(torch.uint8)), (rows, cols, dtype, per_token, 'scales')
                assert torch.equal(eo, fo), (rows, cols, dtype, per_token, 'outer')
                assert torch.equal(ed, fd), (rows, cols, dtype, per_token, 'dequant')
            eq, es = native.eager_fp8_pack(x, False)
            fq, fs = backend.fp8_pack(x, False)
            assert torch.equal(eq.view(torch.uint8), fq.view(torch.uint8)), (rows, cols, dtype, 'fp8 codes')
            assert torch.equal(es.view(torch.uint8), fs.view(torch.uint8)), (rows, cols, dtype, 'fp8 scales')
            cases += 1
    # Read transposed tensors directly; rows are deliberately not a tile multiple.
    x = torch.randn(512, 129, device='cuda').T
    eq, es = native.eager_fp8_pack(x, False)
    fq, fs = backend.fp8_pack(x, False)
    assert torch.equal(eq.view(torch.uint8), fq.view(torch.uint8))
    assert torch.equal(es.view(torch.uint8), fs.view(torch.uint8))
    row = torch.tensor([448., 1.07, -1.07, .0007]*8, device='cuda')
    q, s = backend.fp8_pack(row.repeat(32768, 1), True)
    mean_error = (decoded_mx(q, s).mean(0)-row).abs().max().item()
    assert mean_error < .002, mean_error
    x = torch.randn(129, 256, device='cuda')
    state = torch.cuda.get_rng_state()
    a, _ = backend.fp8_pack(x, True)
    b, _ = backend.fp8_pack(x, True)
    torch.cuda.set_rng_state(state)
    c, _ = backend.fp8_pack(x, True)
    assert torch.equal(a.view(torch.uint8), c.view(torch.uint8)), 'RNG state replay failed'
    assert not torch.equal(a.view(torch.uint8), b.view(torch.uint8)), 'Reused rounding noise'
    print(json.dumps({'passed': True, 'tiled': args.tiled, 'exact_eager_parity_cases': cases,
                      'strided_input': True, 'rng_replay': True, 'sr_max_mean_error': mean_error}))


if __name__ == '__main__':
    main()
