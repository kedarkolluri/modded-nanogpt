"""Within-layer row pipeline using unchanged pack and cuBLAS kernels.

CUDA graphs remove Python launch gaps. All side-stream work rejoins the caller.
Packing and saved dequantized operands are included in every replay.
"""
import argparse
import json
import torch
import triton.testing
import native
import triton
from triton_pack import _fp8

p = argparse.ArgumentParser()
p.add_argument('--repeats', type=int, default=4)
p.add_argument('--profile', action='store_true')
args = p.parse_args()
native.set_pack_backend('triton-tiled')
torch.manual_seed(45)
a = torch.randn(8192, 768, device='cuda')
w = torch.randn(3072, 768, device='cuda') * .03
producer = torch.cuda.Stream()


def full():
    aq, sa, ad = native.fp8_pack(a, False, True)
    wq, sw, wd = native.fp8_pack(w, False, True)
    return torch._scaled_mm(aq, wq.T, sa, sw, out_dtype=torch.bfloat16), ad, wd


def pack_into(x, dequant):
    rows, cols = x.shape
    nc = triton.cdiv(cols // 32, 4) * 4
    padded = triton.cdiv(rows, 128) * 128
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(padded * nc, device=x.device, dtype=torch.uint8)
    _fp8[(padded,)](x, q, scales, scales, dequant, rows, cols, *x.stride(), nc,
                    max(128, triton.next_power_of_2(cols)), False, True, False, num_warps=4)
    return q, scales.view(torch.float8_e8m0fnu), dequant


def chunks(size, overlap):
    main = torch.cuda.current_stream()
    wq, sw, wd = native.fp8_pack(w, False, True)
    if overlap:
        producer.wait_stream(main)
    y = torch.empty((a.shape[0], w.shape[0]), device=a.device, dtype=torch.bfloat16)
    saved = torch.empty_like(a, dtype=torch.bfloat16)
    # Outputs stay alive until the final join; no buffer is overwritten early.
    for offset in range(0, a.shape[0], size):
        x = a[offset:offset+size]
        d = saved[offset:offset+size]
        if overlap:
            with torch.cuda.stream(producer):
                aq, sa, ad = pack_into(x, d)
                ready = torch.cuda.Event()
                ready.record()
            main.wait_event(ready)
            aq.record_stream(main)
            sa.record_stream(main)
            ad.record_stream(main)
        else:
            aq, sa, ad = pack_into(x, d)
        torch.ops.aten._scaled_mm.out(aq, wq.T, sa, sw, out_dtype=torch.bfloat16, out=y[offset:offset+size])
    if overlap:
        main.wait_stream(producer)
    return y, saved, wd


reference = full()
paths = {'native': full}
for size in (512, 1024, 2048, 4096):
    for overlap in (False, True):
        paths[f'rows{size}-' + ('overlap' if overlap else 'serial')] = lambda size=size, overlap=overlap: chunks(size, overlap)
# Warm JIT, handles, and stream resources before capture.
for fn in paths.values():
    for _ in range(3):
        actual = fn()
    torch.cuda.synchronize()
    for x, ref in zip(actual, reference):
        torch.testing.assert_close(x, ref, atol=0, rtol=0)

graphs = {}
for name, fn in paths.items():
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    graph.replay()
    torch.cuda.synchronize()
    for x, ref in zip(outputs, reference):
        torch.testing.assert_close(x, ref, atol=0, rtol=0)
    graphs[name] = (graph, outputs)
if args.profile:
    torch.cuda.cudart().cudaProfilerStart()
    for name in ('native', 'rows2048-serial', 'rows2048-overlap'):
        with torch.cuda.nvtx.range(name):
            for _ in range(5):
                graphs[name][0].replay()
            torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    raise SystemExit(0)
for repeat in range(args.repeats):
    order = list(graphs) if repeat % 2 == 0 else list(graphs)[::-1]
    for name in order:
        ms = triton.testing.do_bench(graphs[name][0].replay, warmup=100, rep=300)
        print(json.dumps({'path': name, 'repeat': repeat, 'ms': ms,
                          'shape': list(a.shape) + [w.shape[0]], 'exact': True,
                          'cuda_graph': True}), flush=True)
