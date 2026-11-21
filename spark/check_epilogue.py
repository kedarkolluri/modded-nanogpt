"""Validate exact output parity and native gradients after epilogue fusion."""
import json
import torch
import native
from epilogue import apply_outer
from bench_layer import measure


def main():
    torch.manual_seed(71)
    for rows, cols in [(1, 32), (129, 768), (1024, 3072)]:
        x = torch.randn(rows, cols, device='cuda') * 1e6
        a = torch.rand(rows, 1, device='cuda') * .001
        b = torch.tensor(.003, device='cuda')
        ref = native.eager_apply_outer(x, a, b)
        assert torch.equal(ref, apply_outer(x, a, b))
    native.set_pack_backend('triton-tiled')
    x = torch.randn(128, 256, device='cuda', requires_grad=True)
    w = torch.randn(384, 256, device='cuda', requires_grad=True)
    g = torch.randn(128, 384, device='cuda', dtype=torch.bfloat16)
    results = []
    for backend in ('eager', 'triton'):
        native.set_epilogue_backend(backend)
        x.grad = w.grad = None
        y = native.NativeLinearFn.apply(x, w, False)
        y.backward(g)
        results.append((y.detach().clone(), x.grad.clone(), w.grad.clone()))
    for a, b in zip(*results):
        assert torch.equal(a, b)
    x = torch.randn(8192, 3072, device='cuda')
    a = torch.rand(8192, 1, device='cuda') * .001
    b = torch.tensor(.003, device='cuda')
    eager_ms = measure(lambda: native.eager_apply_outer(x, a, b))
    fused_ms = measure(lambda: apply_outer(x, a, b))
    print(json.dumps({'passed': True, 'exact_forward_and_gradient_parity': True,
                      'eager_ms': eager_ms, 'triton_ms': fused_ms}))


if __name__ == '__main__':
    main()
