"""Isolate eager packing cost from native GEMM cost at one MLP shape."""
import argparse
import json
import time
import torch
import native


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/30


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pack-backend', choices=['eager', 'triton', 'triton-tiled'], default='eager')
    p.add_argument('--tokens', type=int, default=1024)
    args = p.parse_args()
    native.set_pack_backend(args.pack_backend)
    fp4_pack, fp8_pack = native.fp4_pack, native.fp8_pack
    torch.manual_seed(42)
    x = torch.randn(args.tokens, 768, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(3072, 768, device='cuda', dtype=torch.bfloat16)
    aq, sa, ao, _ = fp4_pack(x, True)
    bq, sb, bo, _ = fp4_pack(w)
    fa, fsa = fp8_pack(x, True)
    fb, fsb = fp8_pack(w, True)
    print(json.dumps({
        'shape': [args.tokens, 3072, 768],
        'pack_backend': args.pack_backend,
        'bf16_gemm_ms': measure(lambda: x @ w.T),
        'fp4_pack_inputs_ms': measure(lambda: (fp4_pack(x, True), fp4_pack(w))),
        'nvfp4_gemm_only_ms': measure(lambda: torch._scaled_mm(aq, bq.T, sa, sb, out_dtype=torch.float32)),
        'mxfp8_sr_pack_inputs_ms': measure(lambda: (fp8_pack(x, True), fp8_pack(w, True))),
        'mxfp8_gemm_only_ms': measure(lambda: torch._scaled_mm(fa, fb.T, fsa, fsb, out_dtype=torch.bfloat16)),
        'note': 'GEMM-only times exclude packing and outer-scale application; no training speedup implied.'
    }, indent=2))


if __name__ == '__main__':
    main()
